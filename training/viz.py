"""Image grids and the LPIPS judge."""
import warnings

import lpips as lpips_pkg
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from cimp_denoise.imops import to8


def font(sz):
    for p in ("C:/Windows/Fonts/consola.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def montage(tiles, row_labels, col_labels, path, scale=2, shared=True, title="", n_ref=1, ref_shared=False):
    """tiles [R][C] float arrays, nearest-upscaled. With shared=True every row
    is shown on one window, computed from the columns after the first n_ref;
    those first columns get their own window, or with ref_shared=True the
    row's window too (for a reference on the same intensity scale whose noise
    extremes should not widen the window, e.g. a single input frame)."""
    R, C = len(tiles), len(tiles[0])
    h, w = tiles[0][0].shape
    th, tw = h * scale, w * scale
    pad, top, left, head = 6, 44, 150, 34
    canvas = Image.new("L", (left + C * (tw + pad) + pad, head + top + R * (th + pad) + pad), 255)
    d = ImageDraw.Draw(canvas)
    d.text((pad, 6), title, fill=0, font=font(16))
    for c, lab in enumerate(col_labels):
        for k, line in enumerate(str(lab).split("\n")):
            d.text((left + pad + c * (tw + pad) + 2, head + 2 + 16 * k), line, fill=0, font=font(13))
    for r in range(R):
        y0 = head + top + pad + r * (th + pad)
        if shared:
            lo_s, hi_s = np.percentile(np.concatenate([t.ravel() for t in tiles[r][n_ref:]]), [0.5, 99.5])
        for k, line in enumerate(str(row_labels[r]).split("\n")):
            d.text((pad, y0 + 4 + 16 * k), line, fill=0, font=font(13))
        for c in range(C):
            lo, hi = (lo_s, hi_s) if shared and (c >= n_ref or ref_shared) else np.percentile(tiles[r][c], [0.5, 99.5])
            canvas.paste(Image.fromarray(to8(tiles[r][c], lo, hi)).resize((tw, th), Image.NEAREST),
                         (left + pad + c * (tw + pad), y0))
    canvas.save(path)


class LPIPS(torch.nn.Module):
    """LPIPS on single-channel [0, 1] images; kept out of autocast."""

    def __init__(self, net="alex", device="cuda"):
        super().__init__()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.net = lpips_pkg.LPIPS(net=net, verbose=False).to(device).eval()
        for p in self.net.parameters():
            p.requires_grad = False

    def forward(self, a, b, reduce=True):
        with torch.amp.autocast("cuda", enabled=False):
            f = lambda x: x.float().clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
            d = self.net(f(a), f(b))
            return d.mean() if reduce else d.flatten()


# ---------------------------------------------------------------- training progress
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed order, never cycled


def plot_progress(log_csv, out_png, panels):
    """Redraw the training curves from log.csv.

    panels: [(title, [(column, label), ...], reference_value_or_None), ...];
    each panel one small chart, series in fixed colours, a dashed line at the
    reference (e.g. 1.0 = correct) when given. Written atomically, so an open
    viewer never sees half a file; never raises (a plot must not stop training)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = np.genfromtxt(log_csv, delimiter=",", names=True)
        d = np.atleast_1d(d)
        if len(d) < 2:
            return
        fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.6), facecolor=SURF)
        axes = np.atleast_1d(axes)
        for ax, (title, series, ref) in zip(axes, panels):
            ax.set_facecolor(SURF)
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors=INK2, labelsize=8)
            ax.set_xlabel("epoch", fontsize=8, color=INK2)
            for k, (col, lab) in enumerate(series):
                if col not in d.dtype.names:
                    continue
                m = np.isfinite(d[col])
                ax.plot(d["epoch"][m], d[col][m], color=SERIES_COLORS[k], lw=2, label=lab)
            if ref is not None:
                ax.axhline(ref, color=INK2, lw=1, ls=(0, (4, 3)))
            ax.set_title(title, fontsize=10, color=INK, loc="left")
            if len(series) > 1:
                ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        tmp = str(out_png) + ".tmp.png"
        fig.savefig(tmp, dpi=110, facecolor=SURF)
        plt.close(fig)
        import os
        os.replace(tmp, out_png)
    except Exception as ex:                                           # noqa: BLE001
        print(f"(progress plot skipped: {type(ex).__name__}: {ex})", flush=True)
