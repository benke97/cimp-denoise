"""Reading microscope images: Velox .emd, Gatan .dm3/.dm4, TIFF and PNG.

Every image comes out as float32 on a [0, 1]-like scale -- the scale CMMP was
trained on (16-bit frames / 65535) -- together with the affine map back to the
file's own units, so the denoised result can be written in the input's units.

Velox traps handled here:
  * frames are uint16 with values past 32767; they are read as uint16
    (a cast to int16 wraps bright pixels negative);
  * multi-detector files (HAADF-DF4-BF ...) hold several Raw channels, and the
    first one is often not HAADF: channels are chosen by their Velox label,
    and DCFI / FFT derived images are skipped.
Only .emd files carry the acquisition metadata the denoiser can use.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import h5py
import numpy as np

EXTENSIONS = (".emd", ".dm3", ".dm4", ".tif", ".tiff", ".png")


@dataclass
class Image:
    data: np.ndarray                  # (H, W) float32, ~[0, 1]
    scale: float                      # file units = data * scale + offset
    offset: float
    channel: str = ""
    frame: int = 0
    metadata: np.ndarray | None = None   # (7,) raw SI values (see metadata.py), .emd only
    info: dict = field(default_factory=dict)

    def to_file_units(self, a):
        return (np.asarray(a, np.float64) * self.scale + self.offset).astype(np.float32)


# ---------------------------------------------------------------- Velox .emd
def _read_json(f, path):
    try:
        s = bytes(f[path][0]).decode("utf-8", errors="ignore").strip("\x00").strip()
        return json.loads(s)
    except Exception:
        return None


def raw_channels(f):
    """{uid: label} of the Raw image channels of an open .emd (DCFI / FFT entries skipped).
    Labels come from /Presentation/Displays/ImageDisplay (adapted from series_alignment, MIT)."""
    group = f.get("Data/Image")
    if not group:
        return {}
    labels = {u: "" for u in group.keys()}
    displays = f.get("Presentation/Displays/ImageDisplay")
    for d in (displays.keys() if displays else []):
        meta = _read_json(f, f"Presentation/Displays/ImageDisplay/{d}")
        uid = (meta or {}).get("dataPath", "").rstrip("/").rsplit("/", 1)[-1]
        if uid in labels:
            labels[uid] = unquote(meta.get("display", {}).get("label", ""))
    derived = lambda lab: "".join(c for c in lab.strip().lower().split("(")[0] if c.isalpha()) in ("dcfi", "fft")
    return {u: lab for u, lab in labels.items() if not derived(lab)}


def haadf_uid(f):
    """The Raw channel labelled HAADF (or the only Raw channel)."""
    ch = raw_channels(f)
    hd = [u for u, lab in ch.items() if lab.upper() == "HAADF"]
    if hd:
        return hd[0]
    if len(ch) == 1:
        return next(iter(ch))
    raise ValueError(f"no HAADF channel among {list(ch.values())}")


def load_frames(path, uid=None, start=0, stop=None):
    """Frames [start, stop) of a channel (default HAADF) as (N, H, W) uint16.
    Slices the dataset before reading (Velox stores Y, X, frames)."""
    with h5py.File(path, "r") as f:
        ds = f["Data/Image"][uid or haadf_uid(f)]["Data"]
        if ds.ndim != 3:
            return ds[:][None]
        stop = ds.shape[-1] if stop is None else min(stop, ds.shape[-1])
        return np.moveaxis(ds[:, :, start:stop], -1, 0)


def _current(optics):
    """Beam current (A). ScreenCurrent is 0 when the screen was not measured
    during the acquisition; LastMeasuredScreenCurrent is what CMMP was trained on."""
    for k in ("LastMeasuredScreenCurrent", "ScreenCurrent"):
        v = float(optics.get(k, 0) or 0)
        if v > 0:
            return v
    raise ValueError("no beam current in the Velox metadata")


def frame_meta(path, uid=None):
    """Per-frame raw metadata (N, 7) in SI units (columns as in metadata.py) and
    per-frame frame time (N,) in s. Gain, offset and collection angle are the
    HAADF detector's, whichever channel uid names."""
    with h5py.File(path, "r") as f:
        md = f["Data/Image"][uid or haadf_uid(f)]["Metadata"]
        cols = [md[:, k] for k in range(md.shape[1])]
    raws, times = [], []
    for c in cols:
        m = json.loads(bytes(c).decode("utf8", "ignore").rstrip("\x00"))
        det = [d for d in m["Detectors"].values() if d.get("DetectorName") == "HAADF"][0]
        raws.append([float(m["BinaryResult"]["PixelSize"]["width"]), float(m["Scan"]["DwellTime"]),
                     float(m["Optics"]["BeamConvergence"]), _current(m["Optics"]),
                     float(det["Gain"]), float(det["Offset"]), float(det["CollectionAngleRange"]["begin"])])
        times.append(float(m["Scan"]["FrameTime"]))
    return np.array(raws, np.float64), np.array(times)


def _read_emd(path, channels, frames):
    out = []
    with h5py.File(path, "r") as f:
        chans = raw_channels(f)
        try:
            hd = haadf_uid(f)
        except ValueError:
            hd = None
        n = {u: (f["Data/Image"][u]["Data"].shape[-1] if f["Data/Image"][u]["Data"].ndim == 3 else 1)
             for u in chans}
    meta = None
    if hd is not None:
        try:
            meta = frame_meta(str(path), hd)[0]
        except (KeyError, IndexError, ValueError):
            meta = None
    for u, lab in chans.items():
        if channels and lab.upper() not in channels:
            continue
        idx = range(n[u]) if frames == "all" else [0]
        for k in idx:
            fr = load_frames(str(path), u, k, k + 1)[0].astype(np.float32)
            m = meta[min(k, len(meta) - 1)] if meta is not None else None
            out.append(Image(fr / 65535.0, 65535.0, 0.0, lab or u[:8], k, m))
    return out


# ---------------------------------------------------------------- other formats
def _scaled(a):
    """Integer data / dtype max (as CMMP saw 16-bit frames); float data mapped
    to [0, 1] by its 0.01 / 99.99 percentiles unless it already lies in [0, 1]."""
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.integer):
        mx = float(np.iinfo(a.dtype).max)
        return a.astype(np.float32) / mx, mx, 0.0
    a = a.astype(np.float32)
    if a.min() >= 0 and a.max() <= 1:
        return a, 1.0, 0.0
    lo, hi = np.percentile(a, [0.01, 99.99])
    sc = float(max(hi - lo, 1e-12))
    return (a - lo) / sc, sc, float(lo)


def _frames_of(a, frames):
    a = np.asarray(a)
    if a.ndim == 2:
        return [a]
    if a.ndim == 3 and a.shape[-1] in (3, 4) and a.shape[0] > 4:        # RGB(A) -> luminance
        return [a[..., :3].astype(np.float32).mean(-1).astype(a.dtype)]
    if a.ndim == 3:
        return list(a) if frames == "all" else [a[0]]
    raise ValueError(f"unsupported image shape {a.shape}")


def _read_plain(path, frames):
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        import tifffile
        arrs = _frames_of(tifffile.imread(path), frames)
    elif ext == ".png":
        from PIL import Image as PILImage
        im = PILImage.open(path)
        arrs = [np.asarray(im.convert("L") if im.mode not in ("L", "I", "I;16", "F") else im)]
    else:                                                                  # .dm3 / .dm4
        from rsciio.digitalmicrograph import file_reader
        arrs = []
        for s in file_reader(str(path)):
            arrs += _frames_of(s["data"], frames)
    out = []
    for k, a in enumerate(arrs):
        d, sc, off = _scaled(a)
        out.append(Image(d, sc, off, "", k))
    return out


def read(path, channels=None, frames="first"):
    """All images of a file as a list of Image.
    channels: e.g. {"HAADF", "BF-S"} for .emd (None = every Raw channel).
    frames: "first" or "all" (series)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in EXTENSIONS:
        raise ValueError(f"{path.name}: unsupported format (supported: {', '.join(EXTENSIONS)})")
    chans = {c.upper() for c in channels} if channels else None
    ims = _read_emd(path, chans, frames) if ext == ".emd" else _read_plain(path, frames)
    for im in ims:
        if min(im.data.shape) < 128:
            raise ValueError(f"{path.name}: images must be at least 128 px per side (got {im.data.shape})")
    return ims


def write_tif(path, a):
    import tifffile
    tifffile.imwrite(path, np.asarray(a, np.float32))
