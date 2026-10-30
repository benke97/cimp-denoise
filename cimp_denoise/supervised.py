"""The supervised denoiser: the x2x network called once at fixed t with the
null style, trained by plain regression to multi-frame means (Noise2Noise
style). Fast and deterministic, but smoother than x2x; here it supplies the
"clean look" whose CMMP embedding becomes x2x's target."""
import numpy as np
import torch

from .imops import hann_window, norm_src
from .x2x import CROP, build_model, tile_positions


class Supervised:
    """Tiled (128 px, stride 96, Hann) like x2x, per-tile normalisation."""

    def __init__(self, path, dev, bs=64):
        ck = torch.load(path, map_location=dev, weights_only=False)
        self.net = build_model().to(dev)
        self.net.load_state_dict(ck["ema"])
        self.net.eval()
        self.MU, self.SD, self.dev, self.bs = float(ck["MU"]), float(ck["SD"]), dev, bs
        self.win = torch.tensor(hann_window(CROP), device=dev)

    @torch.no_grad()
    def __call__(self, frame):
        """frame (H, W) float -> (H, W) denoised, same units."""
        H, W = frame.shape
        pos = tile_positions(H, W)
        src = torch.as_tensor(np.asarray(frame, np.float32), device=self.dev)
        t = torch.stack([src[y:y + CROP, x:x + CROP] for y, x in pos])[:, None]
        x_in, lo, sc = norm_src(t, True)
        null = self.net.net.null_style
        out = []
        with torch.autocast("cuda", torch.bfloat16, enabled=str(self.dev).startswith("cuda")):
            for a in range(0, len(t), self.bs):
                xb = x_in[a:a + self.bs]
                B = len(xb)
                out.append(self.net(torch.zeros_like(xb), xb, torch.zeros(B, device=self.dev),
                                    null.expand(B, -1)).float())
        out = (torch.cat(out) * self.SD + self.MU) * sc + lo
        acc = torch.zeros(H, W, device=self.dev)
        ws = torch.zeros(H, W, device=self.dev)
        for (y, x), o in zip(pos, out[:, 0]):
            acc[y:y + CROP, x:x + CROP] += o * self.win
            ws[y:y + CROP, x:x + CROP] += self.win
        return (acc / ws).cpu().numpy()
