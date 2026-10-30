"""`denoise` command line.

    denoise frame.emd                      every Raw channel, first frame -> frame_<channel>_denoised.tif
    denoise data/ -o out/                  every supported file in a folder
    denoise frame.tif --push 1             no metadata in TIFF: CMMP mode, source pushed one step noisier
    denoise series.emd --frames all --channels HAADF
    denoise frame.emd --mode cmmp          ignore the file's metadata
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .io import EXTENSIONS, read, write_tif


def _inputs(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(q for q in p.iterdir() if q.suffix.lower() in EXTENSIONS)
        elif p.exists():
            yield p
        else:
            sys.exit(f"denoise: {p} does not exist")


def _preview(path, raw, den):
    from PIL import Image as PILImage
    lo, hi = np.percentile(raw, [0.5, 99.5])
    to8 = lambda a: (np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)
    PILImage.fromarray(np.concatenate([to8(raw), np.full((raw.shape[0], 8), 255, np.uint8), to8(den)], 1)).save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="denoise", description="Denoise STEM images with CMMP-steered flow matching.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("inputs", nargs="+", help=f"files or folders ({', '.join(EXTENSIONS)})")
    ap.add_argument("-o", "--out", help="output folder (default: next to each input)")
    ap.add_argument("--mode", choices=("auto", "metadata", "cmmp"), default="auto",
                    help="auto: the file's metadata if it has any (.emd), else CMMP")
    ap.add_argument("--factor", type=float, default=16.0,
                    help="metadata mode: target dwell = factor x min(recorded dwell, 1 us) (default 16)")
    ap.add_argument("--push", type=float, default=0.0,
                    help="CMMP mode: move the source embedding further towards noisy; more denoising (default 0)")
    ap.add_argument("--channels", help="comma-separated .emd channels, e.g. HAADF,BF-S (default: all Raw channels)")
    ap.add_argument("--frames", choices=("first", "all"), default="first", help="series: first frame or every frame")
    ap.add_argument("--draws", type=int, default=8, help="samples averaged per image (default 8)")
    ap.add_argument("--steps", type=int, default=4, help="sampler steps per draw (default 4)")
    ap.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--preview", action="store_true", help="also write a PNG: input | denoised")
    ap.add_argument("--version", action="version", version=f"cimp-denoise {__version__}")
    a = ap.parse_args(argv)

    files = list(_inputs(a.inputs))
    if not files:
        sys.exit("denoise: no supported files found")
    from .api import Denoiser
    den = Denoiser(a.device, a.draws, a.steps)
    print(f"cimp-denoise {__version__} on {den.dev}: {a.draws} draws x {a.steps} steps", flush=True)
    chans = a.channels.split(",") if a.channels else None
    for p in files:
        try:
            images = read(p, chans, a.frames)
        except Exception as e:                                        # keep going over a folder
            print(f"{p.name}: skipped ({e})", flush=True)
            continue
        out_dir = Path(a.out) if a.out else p.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for im in images:
            meta = None if a.mode == "cmmp" else im.metadata
            if a.mode == "metadata" and meta is None:
                print(f"{p.name} {im.channel}: no metadata, skipped (--mode metadata)", flush=True)
                continue
            tag = "_".join(s for s in (p.stem, im.channel.replace(" ", ""),
                                       f"f{im.frame:04d}" if a.frames == "all" else "") if s)
            t0 = time.time()
            prog = lambda j, n: print(f"\r{tag}: draw {j}/{n}", end="", flush=True)
            out, how = den(im.data, meta, a.factor, a.push, progress=prog, return_info=True)
            write_tif(out_dir / f"{tag}_denoised.tif", im.to_file_units(out))
            if a.preview:
                _preview(out_dir / f"{tag}_denoised.png", im.data, out)
            print(f"\r{tag}: {im.data.shape[1]}x{im.data.shape[0]}, {how}, {time.time() - t0:.1f} s".ljust(
                len(tag) + 20), flush=True)


if __name__ == "__main__":
    main()
