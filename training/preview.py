"""
The CIMPx2x denoising preview, shared by the trainer and the command line.

Per row, one low-dwell val stack (centre 128 px crop), one frame in:
    1 input frame | 2 another REAL frame | 3 model @ input dose
    4 REAL mean of all frames | 5 model @ the real mean's dose
    6 model @ --dwell (one draw) | 7 mean of 4 draws @ --dwell
3 is compared with 2: a new independent "frame" -- same noise level and
structure, different grain (the row label gives its noise level / column 2's).
5 is compared with 4. All columns of a row share one window, set by columns
4-7 so single-frame noise extremes do not flatten them.

    python training/preview.py --ckpt runs/x2x_v1/best.pt          # -> <run>/previews/manual_denoise.png
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cimp_denoise.flow import sample                                        # noqa: E402
from cimp_denoise.imops import gblur, hf_amp, norm_src                      # noqa: E402
from cimp_denoise.metadata import DWELL                                     # noqa: E402
from training.paths import STACKS_H5                                    # noqa: E402
from training.viz import montage                                        # noqa: E402
from cimp_denoise.x2x import CROP, anchor, style_vector                           # noqa: E402


def pick_rows(keys, split, nf, raw, n_rows=6, n_cand=24):
    """The lowest-dwell val stacks with the most structure: (stack row, frame0,
    frame1, mean of all frames, n frames), centre crops."""
    cand = sorted([n for n in np.nonzero(split == "val")[0] if nf[n] >= 4], key=lambda n: raw[n, DWELL])[:n_cand]
    rows = []
    with h5py.File(STACKS_H5, "r") as f:
        for n in cand:
            d = f["stacks"][keys[n]]["frames"]
            _, H, W = d.shape
            fr = d[:, (H - CROP) // 2:(H + CROP) // 2, (W - CROP) // 2:(W + CROP) // 2].astype(np.float32) / 65535
            m = fr.mean(0)
            mt = torch.tensor(m)[None, None]
            score = float((gblur(mt, 1.5) - gblur(mt, 6.0)).std()) / (np.percentile(m, 98) - np.percentile(m, 2) + 1e-9)
            rows.append((score, int(n), fr[0], fr[1], m, len(fr)))
    return [r[1:] for r in sorted(rows, key=lambda r: -r[0])[:n_rows]]


@torch.no_grad()
def denoise_panel(net, MU, SD, cmmp, META, RAW, s_mean, s_std, keys, rows, path, title, dwell_us=10.0,
                  draws=4, steps=32, dev="cuda"):
    meta_t = torch.tensor(META, device=dev)
    shift = 1.0 / float(s_std[DWELL])

    def emb(n, k):
        m = meta_t[[n]].clone()
        m[:, DWELL] += float(np.log10(k)) * shift
        return cmmp.embed_meta(m)

    tiles, labels = [], []
    for n, f0, f1, fmean, nf in rows:
        x_in, lo, sc = norm_src(torch.tensor(f0, device=dev)[None, None], True)
        m_hi = meta_t[[n]].clone()
        m_hi[:, DWELL] = float((np.log10(dwell_us * 1e-6) - s_mean[DWELL]) / s_std[DWELL])
        e_src = emb(n, 1)
        e_t = torch.cat([emb(n, 1), emb(n, nf)] + [cmmp.embed_meta(m_hi)] * draws)
        R = 2 + draws
        style = style_vector(e_t, e_src.repeat(R, 1))
        x0 = anchor(x_in.repeat(R, 1, 1, 1),
                    torch.randn((R, 1, CROP, CROP), device=dev, generator=torch.Generator(device=dev).manual_seed(31)),
                    MU, SD, getattr(net, "anchor", 0.0))
        with torch.autocast("cuda", torch.bfloat16, enabled=dev != "cpu"):
            z = sample(net, x_in.repeat(R, 1, 1, 1), style, x0=x0, steps=steps).float()
        out = ((z * SD + MU) * sc + lo)[:, 0]
        same, at_mean, hi = out[0], out[1], out[2:]
        ratio = float(hf_amp(same[None, None]) / hf_amp(torch.tensor(f1, device=dev)[None, None]))
        o = out.cpu().numpy()
        tiles.append([f0, f1, o[0], fmean, o[1], o[2], o[2:].mean(0)])
        labels.append(f"{keys[n]} {RAW[n, DWELL]*1e6:.2g}us\n{RAW[n, 3]*1e12:.3g}pA, mean of {nf}\n"
                      f"noise col3/col2 x{ratio:.2f}")
    dw = f"{dwell_us:g} us"
    montage(tiles, labels, ["input (1 frame)", "REAL other frame", "model @ input dose", "REAL mean",
                            "model @ mean dose", f"model @ {dw}", f"mean of {draws} @ {dw}"],
            path, 2, True, title, n_ref=3, ref_shared=True)


def main():
    from cimp_denoise.cmmp import load_cmmp
    from training.paths import CMMP_WEIGHTS
    from cimp_denoise.x2x import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--dwell", type=float, default=10.0)
    ap.add_argument("--device", default="cuda", help="cpu works too (6 small crops), e.g. while the GPU trains")
    a = ap.parse_args()
    dev = a.device
    if dev.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.8)
    net, MU, SD = load(a.ckpt, dev)
    ck_ep = torch.load(a.ckpt, map_location="cpu", weights_only=False).get("epoch")
    with h5py.File(STACKS_H5, "r") as f:
        keys = [k.decode() for k in f["index/key"][:]]
        split = np.array([s.decode() for s in f["index/split"][:]])
        nf, META = f["index/n_frames"][:], f["metadata"][:].astype(np.float32)
        RAW, s_mean, s_std = f["metadata_raw"][:].astype(np.float64), f["stats_mean"][:], f["stats_std"][:]
    out = Path(a.out) if a.out else Path(a.ckpt).parent / "previews" / f"manual_denoise_ep{ck_ep}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    denoise_panel(net, MU, SD, load_cmmp(CMMP_WEIGHTS, dev), META, RAW, s_mean, s_std, keys, pick_rows(keys, split, nf, RAW), out,
                  f"{Path(a.ckpt).name} (epoch {ck_ep}): column 3 vs 2 = a new frame at the input's dose; 5 vs 4; "
                  f"one window per row (set by columns 4-7)", dwell_us=a.dwell, dev=dev)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
