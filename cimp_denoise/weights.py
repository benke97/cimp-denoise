"""Where the four weight files live, and fetching them on first use.

    cmmp.pt        the frozen CMMP encoder (image + metadata)
    x2x.pt         the flow denoiser, metadata mode
    x2x_cmmp.pt    the flow denoiser, CMMP mode (also trained on image-embedding sources)
    supervised.pt  the supervised denoiser (supplies the "clean look" target)

Looked up in $CIMP_DENOISE_WEIGHTS, else ~/.cache/cimp-denoise. Missing files
are downloaded from WEIGHTS_URL (the Hugging Face repository Stemson-AI/cimp-denoise).
$HF_TOKEN, if set, is sent along (only needed for a private mirror)."""
import os
import sys
import urllib.request
from pathlib import Path

FILES = ("cmmp.pt", "x2x.pt", "x2x_cmmp.pt", "supervised.pt")
WEIGHTS_URL = "https://huggingface.co/Stemson-AI/cimp-denoise/resolve/main/"


def weights_dir():
    d = os.environ.get("CIMP_DENOISE_WEIGHTS")
    return Path(d) if d else Path.home() / ".cache" / "cimp-denoise"


def _download(url, dst):
    """Fetch url to dst; $HF_TOKEN, if set, is sent as a bearer token."""
    tmp = dst.with_suffix(".part")
    req = urllib.request.Request(url)
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total, n = int(r.headers.get("Content-Length", 0)), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            n += len(chunk)
            if total:
                sys.stderr.write(f"\r  {dst.name}: {n / 1e6:.0f} / {total / 1e6:.0f} MB")
    sys.stderr.write("\n")
    tmp.replace(dst)


def path(name, directory=None):
    """Local path of a weight file, downloading it if needed."""
    d = Path(directory) if directory else weights_dir()
    p = d / name
    if p.exists():
        return p
    if WEIGHTS_URL is None:
        raise FileNotFoundError(
            f"{p} not found. Put {', '.join(FILES)} in {d} (or point $CIMP_DENOISE_WEIGHTS at their folder).")
    d.mkdir(parents=True, exist_ok=True)
    sys.stderr.write(f"downloading {name} -> {d}\n")
    _download(WEIGHTS_URL + name, p)
    return p
