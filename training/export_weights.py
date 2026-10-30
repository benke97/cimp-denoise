"""
export_weights.py -- write the four files the `denoise` package loads (cmmp.pt, x2x.pt, x2x_cmmp.pt, supervised.pt) from the
training checkpoints, keeping only what inference needs (EMA weights, output scaling, sampler-relevant args).

    python training/export_weights.py                         # -> ~/.cache/cimp-denoise (or $CIMP_DENOISE_WEIGHTS)
    python training/export_weights.py --out release/weights --x2x runs/x2x_v2_local/ep500.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cimp_denoise.weights import weights_dir                             # noqa: E402
from training.paths import CMMP_WEIGHTS, RUNS                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(weights_dir()))
    ap.add_argument("--cmmp", default=str(CMMP_WEIGHTS))
    ap.add_argument("--x2x", default=str(RUNS / "x2x_v2_local" / "ep500.pt"))
    ap.add_argument("--x2x-cmmp", default=str(RUNS / "x2x_v3_imgsrc" / "ep100.pt"))
    ap.add_argument("--supervised", default=str(RUNS / "sup_v1" / "ep100.pt"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sd = torch.load(a.cmmp, map_location="cpu", weights_only=False)
    torch.save(sd.get("model_state_dict", sd), out / "cmmp.pt")
    for name, src in (("x2x.pt", a.x2x), ("x2x_cmmp.pt", a.x2x_cmmp), ("supervised.pt", a.supervised)):
        ck = torch.load(src, map_location="cpu", weights_only=False)
        args = ck.get("args") or {}
        torch.save({"ema": ck["ema"], "MU": ck["MU"], "SD": ck["SD"],
                    "args": {"anchor_sigma": args.get("anchor_sigma", 0.0)},
                    "source": f"{Path(src).parent.name}/{Path(src).name}"}, out / name)
    for n in ("cmmp.pt", "x2x.pt", "x2x_cmmp.pt", "supervised.pt"):
        print(f"{out / n}: {(out / n).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
