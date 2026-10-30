"""Small image operations shared by both models (torch, (B, 1, H, W))."""
import numpy as np
import torch
import torch.nn.functional as F


def blur1(x):
    """3-tap separable blur [0.27, 0.46, 0.27]."""
    k = torch.tensor([0.27, 0.46, 0.27], device=x.device, dtype=x.dtype)
    x = F.conv2d(F.pad(x, (1, 1, 0, 0), mode="replicate"), k.view(1, 1, 1, 3))
    return F.conv2d(F.pad(x, (0, 0, 1, 1), mode="replicate"), k.view(1, 1, 3, 1))


def gblur(x, sigma):
    r = int(3 * sigma + 0.5)
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def norm_src(x, return_scale=False):
    """Any units -> black level (p0.5) 0, p99 of the 1 px-blurred image 1.
    No absolute brightness reaches a model through its source image."""
    b = gblur(x, 1.0).flatten(1)
    q = torch.quantile(b, torch.tensor([0.005, 0.99], device=x.device, dtype=x.dtype), dim=1)
    lo, hi = q[0].view(-1, 1, 1, 1), q[1].view(-1, 1, 1, 1)
    sc = (hi - lo).clamp_min(1e-6)
    out = ((x - lo) / sc).clamp(-1.0, 4.0)
    return (out, lo, sc) if return_scale else out


def hf_amp(x):
    """Noise amplitude: std(x - box5(x))."""
    k = torch.ones(1, 1, 5, 5, device=x.device, dtype=torch.float32) / 25.0
    x = x.float()
    return (x - F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), k)).flatten(1).std(1)


def to8(a, lo=None, hi=None):
    """numpy image -> uint8 on [lo, hi] (default p0.5-p99.5)."""
    if lo is None:
        lo, hi = np.percentile(a, [0.5, 99.5])
    return (np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).round().astype(np.uint8)


def hann_window(n):
    w = np.hanning(n + 2)[1:-1]
    return np.outer(w, w).astype(np.float32) + 1e-3
