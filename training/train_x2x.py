"""
Train CIMPx2x on real multi-frame stacks.

From one random crop of one training stack: source = mean of k_in frames,
target = mean of a DISJOINT k_out frames (k in 1..8, independent draws, both
directions), so every target is a real measurement with noise independent of
the source's. Raising the dose is denoising, lowering it adds fresh noise.

Conditioning: target embedding = the stack's metadata at dwell x k_out (with
probability --p-image a CMMP image embedding of another crop of the target
instead), source embedding = dwell x k_in; --p-drop replaces the whole vector
with the null token (classifier-free guidance). Vertical flips only (they keep
the fast-scan axis).

Validation every --metric-every epochs on fixed pairs from the val stacks:
    hf_up / hf_dn   noise amplitude of the output over the real target's, for
                    dose up / dose down (1.00 = correct noise level)
    lpips           AlexNet LPIPS to the real target (passthrough shown as the
                    do-nothing reference); lpips_bm with brightness matched
    bright          median output/target mean intensity (1.00), and its IQR
    indep           noise reuse: corr(output HF, source HF) minus a real target's
    spread          draw-to-draw noise over the real level (0 = deterministic)

    python training/train_x2x.py --run-name x2x_v1 --epochs 200
    python training/train_x2x.py --run-name x2x_v1 --resume
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cimp_denoise.cmmp import EMB, load_cmmp                                # noqa: E402
from cimp_denoise.flow import sample                                     # noqa: E402
from cimp_denoise.imops import gblur, hf_amp, norm_src                      # noqa: E402
from cimp_denoise.metadata import DWELL                                     # noqa: E402
from training.ema import EMA                                           # noqa: E402
from training.paths import CMMP_WEIGHTS, RUNS, STACKS_H5                # noqa: E402
from training.viz import LPIPS, montage, plot_progress                                 # noqa: E402
from cimp_denoise.x2x import CROP, STYLE_DIM, anchor, build_model, style_vector   # noqa: E402
from training.preview import denoise_panel, pick_rows                     # noqa: E402


def pair_counts(n, rng, max_frames):
    """Independent k_in, k_out in 1..max_frames with k_in + k_out <= n."""
    hi = min(max_frames, n - 1)
    for _ in range(32):
        a, b = int(rng.integers(1, hi + 1)), int(rng.integers(1, hi + 1))
        if a + b <= n:
            return a, b
    return 1, n - 1


def local_split(n, k_in, k_out, rng):
    """Source and target frames from ONE contiguous window of k_in + k_out consecutive frames, the smaller
    set spread evenly through the window and the larger set the rest, so both cover the same time span with
    the same mean time (k = 1 vs 8: the centre frame vs its 8 neighbours). The target is then the same moment
    as the source within the measurement, not another random time of the stack."""
    L = k_in + k_out
    s = int(rng.integers(0, n - L + 1))
    m = min(k_in, k_out)
    small = np.unique(np.floor((np.arange(m) + 0.5) * L / m).astype(int))
    rest = np.setdiff1d(np.arange(L), small)
    a, b = (small, rest) if k_in <= k_out else (rest, small)
    if k_in == k_out and rng.random() < 0.5:       # equal sets interleave half a frame apart: random side
        a, b = b, a
    return s + a, s + b


def split_frames(n, k_in, k_out, rng, pairing):
    """(source indices, target indices): 'random' = disjoint frames from anywhere in the stack (x2x_v1),
    'local' = local_split."""
    if pairing == "local":
        return local_split(n, k_in, k_out, rng)
    order = rng.permutation(n)
    return order[:k_in], order[k_in:k_in + k_out]


class Pairs(torch.utils.data.Dataset):
    def __init__(self, keys, rows, max_frames, seed=0, length=10 ** 7, pairing="random"):
        self.keys, self.rows, self.max_frames, self.pairing = keys, rows, max_frames, pairing
        self.seed, self.length, self.f = seed, length, None
        with h5py.File(STACKS_H5, "r") as f:
            self.shp = np.array([f["stacks"][k]["frames"].shape for k in keys])
        area = np.maximum(self.shp[:, 1] - CROP + 1, 1) * np.maximum(self.shp[:, 2] - CROP + 1, 1)
        self.p = area / area.sum()

    def __len__(self):
        return self.length

    def crop(self, ds, rng, H, W, frames=None):
        """Random 128 px crop of the given frames (sorted indices; all if None)."""
        y, x = int(rng.integers(0, H - CROP + 1)), int(rng.integers(0, W - CROP + 1))
        sel = slice(None) if frames is None else frames
        return ds[sel, y:y + CROP, x:x + CROP].astype(np.float32) / 65535.0

    def __getitem__(self, i):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng((self.seed, i, 0 if info is None else info.id))
        if self.f is None:
            self.f = h5py.File(STACKS_H5, "r")
        j = int(rng.choice(len(self.keys), p=self.p))
        n, H, W = self.shp[j]
        ds = self.f["stacks"][self.keys[j]]["frames"]
        k_in, k_out = pair_counts(n, rng, self.max_frames)
        si, ti = split_frames(n, k_in, k_out, rng, self.pairing)
        # read only the frames this pair uses (gzip chunks are per frame: with
        # 32-frame stacks, reading every frame made loading the bottleneck)
        need = np.sort(np.concatenate([si, ti]))
        fr = self.crop(ds, rng, H, W, need)
        at = {f: q for q, f in enumerate(need)}
        src = fr[[at[f] for f in si]].mean(0)
        tgt = fr[[at[f] for f in ti]].mean(0)
        prompt = self.crop(ds, rng, H, W, np.sort(rng.permutation(n)[:k_out])).mean(0)
        if rng.random() < 0.5:
            src, tgt, prompt = src[::-1], tgt[::-1], prompt[::-1]
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None]
        return t(src), t(tgt), t(prompt), int(self.rows[j]), k_in, k_out


def fixed_val(keys, split, n_stacks, max_frames, seed=7, pairing="random"):
    """One deterministic centre-crop pair per val stack."""
    rng = np.random.default_rng(seed)
    vs = list(np.nonzero(split == "val")[0])
    rng.shuffle(vs)
    out = []
    with h5py.File(STACKS_H5, "r") as f:
        for n in vs:
            ds = f["stacks"][keys[n]]["frames"]
            nf, H, W = ds.shape
            if nf < 3 or H < CROP or W < CROP:
                continue
            y, x = (H - CROP) // 2, (W - CROP) // 2
            fr = ds[:, y:y + CROP, x:x + CROP].astype(np.float32) / 65535.0
            r = np.random.default_rng(int(n))
            if pairing == "local":
                a, b = pair_counts(nf, r, max_frames)
                si, ti = local_split(nf, a, b, r)
            else:                                    # x2x_v1's draw order, kept so its val set is unchanged
                order = r.permutation(nf)
                a, b = pair_counts(nf, r, max_frames)
                si, ti = order[:a], order[a:a + b]
            out.append((fr[si].mean(0), fr[ti].mean(0), int(n), a, b))
            if len(out) >= n_stacks:
                break
    return out


def hp(x):
    return x.float() - gblur(x.float(), 1.5)


def rowcorr(a, b):
    a, b = a.flatten(1), b.flatten(1)
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="x2x")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--steps-per-epoch", type=int, default=400)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-decay-from", type=int, default=0,
                    help="epoch after which the learning rate falls linearly (per step) to 0 at the last epoch; "
                         "0 = constant (x2x_v1)")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--p-image", type=float, default=0.3, help="target given as a CMMP image embedding")
    ap.add_argument("--p-image-src", type=float, default=0.0,
                    help="source given as the CMMP image embedding of the source itself (how the metadata-free "
                         "mode conditions at inference)")
    ap.add_argument("--init", default="", help="start from this checkpoint's EMA weights (and its MU / SD)")
    ap.add_argument("--warmup-steps", type=int, default=0, help="linear learning-rate warmup")
    ap.add_argument("--anchor-sigma", type=float, default=0.0,
                    help="0: flow starts from pure noise (v1). >0: starts from the source in target units + "
                         "sigma * noise (source-anchored)")
    ap.add_argument("--pairing", choices=("random", "local"), default="random",
                    help="random: source/target frames from anywhere in the stack (x2x_v1); local: one "
                         "contiguous window, interleaved, so the target is the same moment as the source")
    ap.add_argument("--p-drop", type=float, default=0.1, help="whole conditioning -> null token (CFG)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-stacks", type=int, default=96)
    ap.add_argument("--metric-every", type=int, default=2)
    ap.add_argument("--preview-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--sample-steps", type=int, default=32)
    ap.add_argument("--denoise-dwell", type=float, default=10.0, help="us, dose of the denoising preview")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vram-frac", type=float, default=0.8,
                    help="cap on PyTorch's memory pool: past physical VRAM Windows spills into shared "
                         "memory and everything runs ~10x slower")
    args = ap.parse_args()
    RUN = RUNS / args.run_name
    (RUN / "previews").mkdir(parents=True, exist_ok=True)
    dev = args.device
    torch.manual_seed(0)
    if dev.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.vram_frac)
    json.dump(vars(args), open(RUN / "args.json", "w"), indent=1)

    with h5py.File(STACKS_H5, "r") as f:
        keys = [k.decode() for k in f["index/key"][:]]
        SPLIT = np.array([s.decode() for s in f["index/split"][:]])
        NF = f["index/n_frames"][:]
        META = f["metadata"][:].astype(np.float32)
        RAW = f["metadata_raw"][:].astype(np.float64)
        s_mean, s_std = f["stats_mean"][:], f["stats_std"][:]
    cmmp = load_cmmp(CMMP_WEIGHTS, dev)
    META_T = torch.from_numpy(META).to(dev)
    DW_SHIFT = 1.0 / float(s_std[DWELL])

    def emb(stack_ix, k):
        m = META_T[stack_ix].clone()
        m[:, DWELL] += torch.log10(k.float()) * DW_SHIFT
        return cmmp.embed_meta(m)

    tr = [n for n in range(len(keys)) if SPLIT[n] == "train" and NF[n] >= 3]
    ds = Pairs([keys[n] for n in tr], np.array(tr), args.max_frames, pairing=args.pairing)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.bs, num_workers=args.workers, pin_memory=True,
                                         persistent_workers=args.workers > 0,
                                         prefetch_factor=4 if args.workers > 0 else None)

    # targets on their source's scale, then one global mean/std
    init = None
    if args.resume and (RUN / "last.pt").exists():
        ck = torch.load(RUN / "last.pt", map_location=dev, weights_only=False)
        MU, SD = ck["MU"], ck["SD"]
    elif args.init:
        init = torch.load(args.init, map_location=dev, weights_only=False)
        MU, SD, ck = float(init["MU"]), float(init["SD"]), None
    else:
        r_ = np.random.default_rng(0)
        rel = []
        for i in r_.integers(0, 10 ** 6, 400):
            s_, t_ = ds[int(i)][:2]
            _, lo_, sc_ = norm_src(s_[None], True)
            rel.append(((t_[None] - lo_) / sc_).flatten()[::5].numpy())
        rel = np.concatenate(rel).astype(np.float64)
        MU, SD = float(rel.mean()), float(rel.std())
        if ds.f is not None:
            ds.f.close()
            ds.f = None
        ck = None
    to_y = lambda tgt, lo, sc: ((tgt - lo) / sc - MU) / SD
    from_y = lambda z, lo, sc: (z * SD + MU) * sc + lo
    print(f"{len(tr)} train stacks, target transform (y - {MU:.4f}) / {SD:.4f}", flush=True)

    model = build_model().to(dev)
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    step, start_ep, best = 0, 1, float("inf")
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        ema.m = {k: v.to(dev) for k, v in ck["ema"].items()}
        ema.n = ck["step"]
        step, start_ep, best = ck["step"], ck["epoch"] + 1, ck["best"]
        print(f"resumed at epoch {ck['epoch']}", flush=True)
        del ck
    elif init is not None:
        model.load_state_dict(init["ema"])
        ema.m = {k: v.detach().clone().to(dev) for k, v in init["ema"].items()}
        ema.n = 10 ** 6                                  # full decay from the start: the EMA is already trained
        print(f"initialised from {args.init}", flush=True)
        del init

    # ---- fixed validation
    V = fixed_val(keys, SPLIT, args.val_stacks, args.max_frames, pairing=args.pairing)
    T_ = lambda a: torch.tensor(np.stack(a), dtype=torch.float32, device=dev)[:, None]
    V_src, V_tgt = T_([v[0] for v in V]), T_([v[1] for v in V])
    V_st = torch.tensor([v[2] for v in V], device=dev)
    V_ki, V_ko = torch.tensor([v[3] for v in V], device=dev), torch.tensor([v[4] for v in V], device=dev)
    V_r = (V_ko.float() / V_ki.float()).cpu().numpy()
    V_style = style_vector(emb(V_st, V_ko), emb(V_st, V_ki))
    V_in, V_lo, V_sc = norm_src(V_src, True)
    lp = LPIPS("alex", dev)
    g0 = torch.Generator(device=dev).manual_seed(123)
    V_x0 = [anchor(V_in, torch.randn(V_src.shape, device=dev, generator=g0), MU, SD, args.anchor_sigma)
            for _ in range(2)]
    V_t = torch.rand(len(V), device=dev, generator=g0)
    with torch.no_grad():
        IND_REAL = rowcorr(hp(V_tgt), hp(V_src)).cpu().numpy()
        LP_PASS = float(lp(V_src, V_tgt))
    print(f"val: {len(V)} pairs (dose up {int((V_r > 1).sum())}, down {int((V_r < 1).sum())}), "
          f"passthrough LPIPS {LP_PASS:.4f}", flush=True)

    def ema_net():
        net = build_model().to(dev)
        net.load_state_dict(ema.m)
        net.eval()
        net.anchor = args.anchor_sigma
        return net

    def draw(net, src, style, x0, bs=32):
        with torch.autocast("cuda", torch.bfloat16, enabled=dev != "cpu"):
            return torch.cat([sample(net, src[a:a + bs], style[a:a + bs], x0=x0[a:a + bs],
                                     steps=args.sample_steps).float() for a in range(0, len(src), bs)])

    @torch.no_grad()
    def metrics(net):
        y = to_y(V_tgt, V_lo, V_sc)
        t = V_t.view(-1, 1, 1, 1)
        vloss = float(F.mse_loss(net((1 - t) * V_x0[0] + t * y, V_in, V_t, V_style).float(), y - V_x0[0]))
        d1 = from_y(draw(net, V_in, V_style, V_x0[0]), V_lo, V_sc)
        n2 = min(32, len(V))
        d2 = from_y(draw(net, V_in[:n2], V_style[:n2], V_x0[1][:n2]), V_lo[:n2], V_sc[:n2])
        ratio = (hf_amp(d1) / (hf_amp(V_tgt) + 1e-9)).cpu().numpy()
        am, asd = d1.mean((1, 2, 3), keepdim=True), d1.std((1, 2, 3), keepdim=True)
        bm, bsd = V_tgt.mean((1, 2, 3), keepdim=True), V_tgt.std((1, 2, 3), keepdim=True)
        br = (d1.mean((1, 2, 3)) / V_tgt.mean((1, 2, 3))).cpu().numpy()
        return dict(
            val_loss=vloss, hf_up=float(ratio[V_r > 1].mean()), hf_dn=float(ratio[V_r < 1].mean()),
            lpips=float(lp(d1, V_tgt)), lpips_bm=float(lp((d1 - am) / asd.clamp_min(1e-9) * bsd + bm, V_tgt)),
            bright=float(np.median(br)), bright_iqr=float(np.subtract(*np.percentile(br, [75, 25]))),
            indep=float(np.median(rowcorr(hp(d1), hp(V_src)).cpu().numpy() - IND_REAL)),
            spread=float(np.median((hf_amp(d1[:n2] - d2) / (1.4142 * hf_amp(V_tgt[:n2]) + 1e-9)).cpu().numpy())))

    # denoising preview (training/preview.py): low-dwell val stacks, one frame in
    DEN = pick_rows(keys, SPLIT, NF, RAW)

    def preview(ep, net):
        denoise_panel(net, MU, SD, cmmp, META, RAW, s_mean, s_std, keys, DEN,
                      RUN / "previews" / f"ep{ep:03d}_denoise.png",
                      f"epoch {ep}: column 3 vs 2 = a new frame at the input's dose; 5 vs 4; one window per row "
                      f"(set by columns 4-7)", dwell_us=args.denoise_dwell, steps=args.sample_steps, dev=dev)

    cols = ["epoch", "step", "train_loss", "val_loss", "hf_up", "hf_dn", "lpips", "lpips_bm", "bright",
            "bright_iqr", "indep", "spread", "minutes"]
    log = open(RUN / "log.csv", "a", buffering=1)
    if start_ep == 1:
        log.write(",".join(cols) + "\n")

    def infinite(dl):
        while True:
            yield from dl
    stream = infinite(loader)
    t0, last = time.time(), None
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        losses = []
        bar = tqdm(range(args.steps_per_epoch), desc=f"epoch {ep}/{args.epochs}", ncols=100)
        for it in bar:
            src, tgt, pr, st, ki, ko = next(stream)
            src, tgt, pr = (a.to(dev, non_blocking=True) for a in (src, tgt, pr))
            st, ki, ko = st.to(dev), ki.to(dev), ko.to(dev)
            B = len(tgt)
            x_in, lo, sc = norm_src(src, True)
            y = to_y(tgt, lo, sc)
            e_t = emb(st, ko)
            use_img = torch.rand(B, device=dev) < args.p_image
            if use_img.any():
                e_t = torch.where(use_img[:, None], cmmp.embed_image(pr.clamp(0, 1)), e_t)
            e_s = emb(st, ki)
            use_img_s = torch.rand(B, device=dev) < args.p_image_src
            if use_img_s.any():
                e_s = torch.where(use_img_s[:, None], cmmp.embed_image(src.clamp(0, 1)), e_s)
            style = style_vector(e_t, e_s)
            drop = torch.rand(B, device=dev) < args.p_drop
            style = torch.where(drop[:, None], model.net.null_style.expand(B, -1), style)
            x0, t = anchor(x_in, torch.randn_like(y), MU, SD, args.anchor_sigma), torch.rand(B, device=dev)
            xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * y
            with torch.autocast("cuda", torch.bfloat16, enabled=dev != "cpu"):
                v = model(xt, x_in, t, style)
            loss = F.mse_loss(v.float(), y - x0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            f_lr = 1.0
            if args.lr_decay_from:
                s0, s1 = args.lr_decay_from * args.steps_per_epoch, args.epochs * args.steps_per_epoch
                f_lr = min(1.0, max(0.0, (s1 - step) / (s1 - s0)))
            if args.warmup_steps:
                f_lr *= min(1.0, (step + 1) / args.warmup_steps)
            for g in opt.param_groups:
                g["lr"] = args.lr * f_lr
            opt.step()
            ema.update(model)
            losses.append(loss.item())
            step += 1
            if it % 20 == 0:
                bar.set_postfix(loss=f"{np.mean(losses[-200:]):.4f}", lr=f"{opt.param_groups[0]['lr']:.1e}")

        torch.cuda.empty_cache()
        net = ema_net()
        fresh = ep % args.metric_every == 0 or ep == start_ep
        mark = ""
        if fresh:
            last = metrics(net)
            if last["lpips"] < best:
                best, mark = last["lpips"], " <-- best"
        m = last
        tqdm.write(f"epoch {ep:3d} | train {np.mean(losses):.4f} val {m['val_loss']:.4f} | hf up {m['hf_up']:.2f} "
                   f"dn {m['hf_dn']:.2f} | lpips {m['lpips']:.4f} (bm {m['lpips_bm']:.4f}, passthrough "
                   f"{LP_PASS:.4f}) | bright x{m['bright']:.2f} (IQR {m['bright_iqr']:.2f}) | reuse "
                   f"{m['indep']:+.3f} spread {m['spread']:.2f}{'' if fresh else ' [stale]'} | "
                   f"{(time.time()-t0)/60:.1f} min{mark}")
        log.write(",".join(str(v) for v in [ep, step, f"{np.mean(losses):.5f}"] +
                  [f"{m[k]:.5f}" if fresh else "nan" for k in cols[3:-1]] + [f"{(time.time()-t0)/60:.2f}"]) + "\n")
        plot_progress(RUN / "log.csv", RUN / "progress.png", [
            ("flow loss", [("train_loss", "train"), ("val_loss", "val (fixed pairs)")], None),
            ("noise level / real target (1 = right)", [("hf_up", "dose up"), ("hf_dn", "dose down")], 1.0),
            (f"LPIPS to real target (passthrough {LP_PASS:.3f})", [("lpips", "LPIPS"), ("lpips_bm", "brightness-matched")], LP_PASS),
            ("brightness / draw spread / noise reuse", [("bright", "brightness ratio (1)"), ("spread", "draw spread"),
                                                        ("indep", "noise reuse (0)")], 1.0)])
        ck = {"model": model.state_dict(), "ema": ema.m, "opt": opt.state_dict(), "step": step, "epoch": ep,
              "best": best, "MU": MU, "SD": SD, "args": vars(args)}
        torch.save(ck, RUN / "last.pt")
        if mark:
            torch.save(ck, RUN / "best.pt")
        if ep % args.save_every == 0:
            torch.save({"ema": ema.m, "epoch": ep, "MU": MU, "SD": SD, "args": vars(args)}, RUN / f"ep{ep:03d}.pt")
        if ep % args.preview_every == 0:
            preview(ep, net)
        del net
        torch.cuda.empty_cache()
    print(f"done: best LPIPS {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
