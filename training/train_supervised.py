"""
train_supervised.py -- the main denoising baseline: a plain supervised (Noise2Noise-style) U-Net on the SAME
stacks, pairing and network as CIMPx2x, with NO CMMP embedding and no sampling.

    input   one real frame (per-source normalised as in x2x: black level 0, p99 1)
    target  the mean of 1..max_frames disjoint frames from the same time window (pairing "local", as x2x_v2)
    loss    MSE to the (noisy) target -> converges to the expected clean image (Noise2Noise)
    network cimp_denoise.x2x.build_model() (identical to x2x), called at fixed t with the null style token:
            a deterministic image-to-image regressor with the same capacity

    python training/train_supervised.py --run-name sup_v1 --epochs 200
Validation every --metric-every epochs: MSE and PSNR to the mean of the remaining frames on fixed val pairs.
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
from training.ema import EMA                                           # noqa: E402
from cimp_denoise.imops import norm_src                                     # noqa: E402
from training.paths import RUNS, STACKS_H5                              # noqa: E402
from cimp_denoise.x2x import CROP, build_model                              # noqa: E402
from training.train_x2x import Pairs                                          # noqa: E402


class SupPairs(Pairs):
    """Pairs with the source fixed to ONE frame (k_in = 1)."""

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, 7))
        if self.f is None:
            self.f = h5py.File(STACKS_H5, "r")
        j = int(rng.choice(len(self.keys), p=self.p))
        n, H, W = self.shp[j]
        ds = self.f["stacks"][self.keys[j]]["frames"]
        k_out = int(rng.integers(1, min(self.max_frames, n - 1) + 1))
        L = k_out + 1
        s = int(rng.integers(0, n - L + 1))
        c = s + L // 2                                                   # source = centre frame of the window
        tgt_ix = [f for f in range(s, s + L) if f != c]
        need = np.sort(np.array([c] + tgt_ix))
        fr = self.crop(ds, rng, H, W, need)
        at = {f: q for q, f in enumerate(need)}
        src, tgt = fr[at[c]], fr[[at[f] for f in tgt_ix]].mean(0)
        if rng.random() < 0.5:
            src, tgt = src[::-1], tgt[::-1]
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None]
        return t(src), t(tgt)


def forward(model, x_in):
    """The x2x network as a plain regressor: fixed t, null style, zero 'noisy state' input."""
    B = len(x_in)
    return model(torch.zeros_like(x_in), x_in, torch.zeros(B, device=x_in.device), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="sup_v1")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--steps-per-epoch", type=int, default=400)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--lr-decay-from", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-n", type=int, default=96)
    args = ap.parse_args()
    RUN = RUNS / args.run_name
    RUN.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(RUN / "args.json", "w"), indent=1)
    dev = "cuda"
    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(0.85)
    with h5py.File(STACKS_H5, "r") as f:
        keys = [k.decode() for k in f["index/key"][:]]
        SPLIT = np.array([s.decode() for s in f["index/split"][:]])
        NF = f["index/n_frames"][:]
    tr = [n for n in range(len(keys)) if SPLIT[n] == "train" and NF[n] >= 3]
    va = [n for n in range(len(keys)) if SPLIT[n] == "val" and NF[n] >= 3]
    ds = SupPairs([keys[n] for n in tr], np.array(tr), args.max_frames)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.bs, num_workers=args.workers, pin_memory=True,
                                         persistent_workers=True, prefetch_factor=4)
    # fixed validation: centre crop, frame 0 in, mean of all other frames as reference
    V_src, V_ref = [], []
    with h5py.File(STACKS_H5, "r") as f:
        for n in va[:args.val_n]:
            d = f["stacks"][keys[n]]["frames"]
            nf, H, W = d.shape
            if H < CROP or W < CROP:
                continue
            y, x = (H - CROP) // 2, (W - CROP) // 2
            fr = d[:, y:y + CROP, x:x + CROP].astype(np.float32) / 65535
            V_src.append(fr[0]); V_ref.append(fr[1:].mean(0))
    V_src = torch.tensor(np.stack(V_src), device=dev)[:, None]
    V_ref = torch.tensor(np.stack(V_ref), device=dev)[:, None]
    # target transform: the target on its source's scale, then one global mean / std (as x2x)
    r_ = np.random.default_rng(0)
    rel = []
    for i in r_.integers(0, 10 ** 6, 400):
        s_, t_ = ds[int(i)]
        _, lo_, sc_ = norm_src(s_[None], True)
        rel.append(((t_[None] - lo_) / sc_).flatten()[::5].numpy())
    rel = np.concatenate(rel)
    MU, SD = float(rel.mean()), float(rel.std())
    ds.f = None
    model = build_model().to(dev)
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    log = open(RUN / "log.csv", "w", buffering=1)
    log.write("epoch,train_mse,val_psnr,val_psnr_input,minutes\n")

    @torch.no_grad()
    def val(net):
        x_in, lo, sc = norm_src(V_src, True)
        with torch.autocast("cuda", torch.bfloat16):
            out = (forward(net, x_in).float() * SD + MU) * sc + lo
        def psnr(a):
            lo_ = torch.quantile(V_ref.flatten(1), 0.005, dim=1).view(-1, 1, 1, 1)
            hi_ = torch.quantile(V_ref.flatten(1), 0.995, dim=1).view(-1, 1, 1, 1)
            r = (hi_ - lo_).clamp_min(1e-9)
            mse = (((a.clamp(lo_, hi_) - lo_) / r - (V_ref.clamp(lo_, hi_) - lo_) / r) ** 2).mean((1, 2, 3))
            return float((10 * torch.log10(1 / mse)).median())
        return psnr(out), psnr(V_src)

    stream = iter(())
    def batches():
        while True:
            yield from loader
    stream = batches()
    t0, step = time.time(), 0
    S0, S1 = args.lr_decay_from * args.steps_per_epoch, args.epochs * args.steps_per_epoch
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _ in tqdm(range(args.steps_per_epoch), desc=f"epoch {ep}", ncols=90, leave=False):
            src, tgt = (a.to(dev, non_blocking=True) for a in next(stream))
            x_in, lo, sc = norm_src(src, True)
            y = ((tgt - lo) / sc - MU) / SD
            with torch.autocast("cuda", torch.bfloat16):
                out = forward(model, x_in)
            loss = F.mse_loss(out.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if args.lr_decay_from and step >= S0:
                for g in opt.param_groups:
                    g["lr"] = args.lr * max(0.0, (S1 - step) / (S1 - S0))
            opt.step(); ema.update(model); step += 1
            losses.append(loss.item())
        net = build_model().to(dev); net.load_state_dict(ema.m); net.eval()
        p, p_in = val(net)
        log.write(f"{ep},{np.mean(losses):.5f},{p:.3f},{p_in:.3f},{(time.time() - t0) / 60:.2f}\n")
        tqdm.write(f"epoch {ep}: train mse {np.mean(losses):.4f} | val PSNR {p:.2f} dB (input {p_in:.2f}) | "
                   f"{(time.time() - t0) / 60:.1f} min")
        ck = {"ema": ema.m, "epoch": ep, "MU": MU, "SD": SD, "args": vars(args), "supervised": True}
        torch.save(ck, RUN / "last.pt")
        if ep % 50 == 0 or ep == args.epochs:
            torch.save(ck, RUN / f"ep{ep:03d}.pt")
        del net
    print("done", flush=True)


if __name__ == "__main__":
    main()
