"""Denoising a single STEM frame with the flow denoiser, steered by CMMP.

Two ways to say what the result should look like (the "style"):

  metadata  the file's own acquisition metadata (Velox .emd): source =
            min(recorded dwell, 1 us), target = factor x the source (default
            16, i.e. the look of a 16-frame average). The cap keeps the target
            inside the dwell range x2x was trained on: a 32 us frame told
            32 -> 512 us comes out grainy, told 1 -> 16 us it comes out clean.
  cmmp      no metadata at all. The supervised denoiser makes a quick clean
            version of the frame; CMMP embeds that ("how this frame looks
            clean", the target) and the frame itself ("how it looks now", the
            source). x2x moves the frame from one look to the other.
            Both copies CMMP embeds get the same black-level lift first
            (frames saved with their dark floor removed are otherwise read
            as ~10x too fine a pixel size); x2x still denoises the untouched
            frame.
            push > 0 moves the source further along clean -> noisy, which
            removes more noise.

The output is the mean of `draws` x2x samples of `steps` steps each. Each mode has its
own x2x weights: x2x.pt (trained with metadata sources) for metadata, x2x_cmmp.pt
(also trained with CMMP image embeddings as sources) for cmmp.
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import weights
from .cmmp import load_cmmp
from .metadata import DWELL, z_meta
from .supervised import Supervised
from .x2x import CROP, Renderer, load, metadata_style, style_vector


SRC_CAP = 1e-6                           # metadata mode: told source dwell = min(recorded, 1 us)
FLOOR = 0.0223                           # median 1st percentile of CMMP's HAADF training frames


def lift_offset(frame):
    """Offset that raises the frame's dark floor to CMMP's training floor (0 on frames that already have one)."""
    return max(0.0, FLOOR - float(np.percentile(frame, 1)))


def lift(frame, offset=None):
    return np.clip(frame + (lift_offset(frame) if offset is None else offset), 0, 1)


def frame_embedding(cmmp, frame, dev, max_tiles=256):
    """CMMP image embedding of a whole frame: normalised mean over its
    non-overlapping 128 px tiles (clamped to [0, 1], as CMMP was trained). (1, 128)."""
    fr = torch.as_tensor(np.asarray(frame, np.float32), device=dev)
    H, W = fr.shape
    t = torch.stack([fr[y:y + CROP, x:x + CROP] for y in range(0, H - CROP + 1, CROP)
                     for x in range(0, W - CROP + 1, CROP)])
    if len(t) > max_tiles:
        t = t[torch.linspace(0, len(t) - 1, max_tiles).long()]
    e = torch.cat([cmmp.embed_image(t[a:a + 256, None].clamp(0, 1)) for a in range(0, len(t), 256)])
    return F.normalize(e.mean(0, keepdim=True), dim=-1)


class Denoiser:
    def __init__(self, device=None, draws=8, steps=4, weights_dir=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.draws = draws
        w = lambda n: weights.path(n, weights_dir)
        self.cmmp = load_cmmp(w("cmmp.pt"), self.dev)
        self.x2x = Renderer(*load(w("x2x.pt"), self.dev), self.dev, steps=steps)
        self.x2x_cmmp = Renderer(*load(w("x2x_cmmp.pt"), self.dev), self.dev, steps=steps)
        self.sup = Supervised(w("supervised.pt"), self.dev)

    def style(self, frame, metadata=None, factor=16.0, push=0.0):
        """(style vector, description). metadata: (7,) raw SI values or None."""
        if metadata is not None:
            dwell = float(metadata[DWELL])
            src = min(dwell, SRC_CAP)
            st = metadata_style(self.cmmp, z_meta(metadata), src, factor * src, self.dev)
            return st, f"metadata: recorded {dwell * 1e6:g} us, told {src * 1e6:g} -> {factor * src * 1e6:g} us"
        off = lift_offset(frame)
        e_x = frame_embedding(self.cmmp, lift(frame, off), self.dev)
        e_s = frame_embedding(self.cmmp, lift(self.sup(frame), off), self.dev)
        st = style_vector(e_s, F.normalize(e_x + push * (e_x - e_s), dim=-1))
        return st, f"cmmp: target = CMMP(supervised(frame)), source = CMMP(frame), push {push:g}, lift {off:.4f}"

    def __call__(self, frame, metadata=None, factor=16.0, push=0.0, progress=None, return_info=False):
        """frame (H, W) float on the ~[0, 1] scale of io.read (>= 128 px per side) -> denoised (H, W)
        (and the style description with return_info)."""
        frame = np.asarray(frame, np.float32)
        if frame.ndim != 2 or min(frame.shape) < CROP:
            raise ValueError(f"need a 2-D image of at least {CROP} px per side, got {frame.shape}")
        st, how = self.style(frame, metadata, factor, push)
        r = self.x2x if metadata is not None else self.x2x_cmmp
        out = r.mean(frame, st, self.draws, progress=progress)
        return (out, how) if return_info else out
