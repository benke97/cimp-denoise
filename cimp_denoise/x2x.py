"""The flow denoiser (CIMPx2x): render a real acquisition as it would look
after a longer exposure.

Style vector (257 d): [target embedding | source embedding | 1]
    target  the look to produce    (CMMP embedding of metadata or of an image)
    source  the look of the input  (likewise)
    the trailing 1 is a constant mode flag, kept so the network matches the
    first trained version.
Intensity: each tile is normalised (black level 0, p99 1), the output is
predicted on that scale and mapped back, so brightness comes out relative to
the input without any absolute reference.
"""
import numpy as np
import torch

from .cmmp import EMB
from .flow import SourceFlow, sample
from .imops import hann_window, norm_src
from .metadata import with_dwell

CROP = 128
STRIDE = 96
STYLE_DIM = 2 * EMB + 1


def build_model():
    return SourceFlow(style_dim=STYLE_DIM, crop=CROP)


def style_vector(e_target, e_source):
    return torch.cat([e_target, e_source, torch.ones(len(e_target), 1, device=e_target.device)], 1)


def anchor(x_in, eps, MU, SD, sigma):
    """Start of the flow. sigma 0: pure noise, the source only conditions the
    network. sigma > 0: the source itself, in target units, plus sigma * noise."""
    return eps if not sigma else (x_in - MU) / SD + sigma * eps


def load(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = build_model().to(dev)
    net.load_state_dict(ck["ema"])
    net.eval()
    net.anchor = float((ck.get("args") or {}).get("anchor_sigma", 0.0) or 0.0)
    return net, float(ck["MU"]), float(ck["SD"])


def tile_positions(H, W, crop=CROP, stride=STRIDE):
    """Top-left corners of overlapping crop x crop tiles covering an H x W image (H, W >= crop)."""
    ys = list(range(0, H - crop + 1, stride))
    xs = list(range(0, W - crop + 1, stride))
    ys += [H - crop] if ys[-1] != H - crop else []
    xs += [W - crop] if xs[-1] != W - crop else []
    return [(y, x) for y in ys for x in xs]


def metadata_style(cmmp, z, dwell_src_s, dwell_out_s, dev):
    """[meta(dwell_out), meta(dwell_src)] from z-scored metadata z (7,)."""
    m = np.stack([with_dwell(z, dwell_out_s), with_dwell(z, dwell_src_s)])
    e = cmmp.embed_meta(torch.tensor(m, device=dev))
    return style_vector(e[:1], e[1:])


class Renderer:
    """Tiled x2x on whole frames of any size (>= 128 px per side).

    128 px tiles at stride 96, Hann-blended, ONE noise field per draw shared by
    all tiles (no seams). Tiling is not optional: the network was trained on
    128 px crops, and a single full-frame pass drifts in brightness and loses
    detail."""

    def __init__(self, net, MU, SD, dev, steps=32, bs=32):
        self.net, self.MU, self.SD, self.dev = net, MU, SD, dev
        self.steps, self.bs = steps, bs
        self.win = torch.tensor(hann_window(CROP), device=dev)

    @torch.no_grad()
    def __call__(self, frame, style, seed=0, guidance=1.0):
        """One draw. frame (H, W) float -> (H, W) at the style's target look, same units."""
        H, W = frame.shape
        pos = tile_positions(H, W)
        src = torch.as_tensor(frame, device=self.dev, dtype=torch.float32)
        tiles = torch.stack([src[y:y + CROP, x:x + CROP] for y, x in pos])[:, None]
        x_in, lo, sc = norm_src(tiles, True)
        field = torch.randn((H, W), device=self.dev, generator=torch.Generator(device=self.dev).manual_seed(seed))
        x0 = anchor(x_in, torch.stack([field[y:y + CROP, x:x + CROP] for y, x in pos])[:, None],
                    self.MU, self.SD, getattr(self.net, "anchor", 0.0))
        out = []
        with torch.autocast("cuda", torch.bfloat16, enabled=str(self.dev).startswith("cuda")):
            for a in range(0, len(pos), self.bs):
                b = slice(a, a + self.bs)
                out.append(sample(self.net, x_in[b], style.expand(len(pos[b]), -1), x0=x0[b],
                                  steps=self.steps, guidance=guidance).float())
        z = (torch.cat(out) * self.SD + self.MU) * sc + lo
        acc = torch.zeros((H, W), device=self.dev)
        wsum = torch.zeros((H, W), device=self.dev)
        for (y, x), t in zip(pos, z[:, 0]):
            acc[y:y + CROP, x:x + CROP] += t * self.win
            wsum[y:y + CROP, x:x + CROP] += self.win
        return (acc / wsum).cpu().numpy()

    def mean(self, frame, style, draws=16, guidance=1.0, progress=None):
        """Mean of `draws` draws (seeds 0..draws-1)."""
        acc = np.zeros(frame.shape, np.float64)
        for j in range(draws):
            acc += self(frame, style, seed=j, guidance=guidance)
            if progress:
                progress(j + 1, draws)
        return (acc / draws).astype(np.float32)
