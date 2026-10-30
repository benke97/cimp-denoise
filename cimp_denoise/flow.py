"""Conditional rectified-flow U-Net and its sampler.

    x0 ~ N(0, I),  x1 = a real image,  t ~ U(0, 1)
    xt = (1 - t) x0 + t x1,   the network regresses the velocity x1 - x0

The network is conditioned on a style vector (CMMP embeddings) through
adaptive group norm in every residual block, and on a source image given as
a second input channel. t = 0 is noise, t = 1 is data.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels, max_groups=32):
    return nn.GroupNorm(math.gcd(channels, max_groups) or 1, channels)


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None] * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1) if dim % 2 else emb


class ResBlock(nn.Module):
    """Pre-activation residual block; the conditioning (time + style) sets a
    per-channel scale and shift. Zero-init output, so it starts as identity."""

    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.0):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect")
        self.emb = nn.Linear(cond_dim, out_ch * 2)
        self.norm2 = _gn(out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="reflect")
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(cond))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift
        return self.conv2(self.drop(F.silu(h))) + self.skip(x)


class Attention(nn.Module):
    """Multi-head self-attention over positions (low resolutions only)."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = _gn(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.num_heads, C // self.num_heads, H * W).unbind(1)
        q, k, v = (t.transpose(-1, -2).contiguous() for t in (q, k, v))
        h = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(h)


class FlowUNet(nn.Module):
    """U-Net velocity field. style=None uses a learned null token
    (classifier-free guidance)."""

    def __init__(self, in_channels=1, base=64, ch_mult=(1, 2, 2, 4), num_res_blocks=2,
                 style_dim=128, cond_dim=None, attn_resolutions=(16,), dropout=0.0,
                 num_heads=4, image_size=128):
        super().__init__()
        cond_dim = cond_dim or base * 4
        self.base = base
        self.time_mlp = nn.Sequential(nn.Linear(base, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.style_mlp = nn.Sequential(nn.Linear(style_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.null_style = nn.Parameter(torch.zeros(1, style_dim))
        self.stem = nn.Conv2d(in_channels, base, 3, padding=1, padding_mode="reflect")

        chans = [base * m for m in ch_mult]
        self.downs = nn.ModuleList()
        skip_ch, ch, res = [base], base, image_size
        for i, out_ch in enumerate(chans):
            for _ in range(num_res_blocks):
                self.downs.append(nn.ModuleList([
                    ResBlock(ch, out_ch, cond_dim, dropout),
                    Attention(out_ch, num_heads) if res in attn_resolutions else nn.Identity()]))
                ch = out_ch
                skip_ch.append(ch)
            if i < len(chans) - 1:
                self.downs.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.Identity()]))
                skip_ch.append(ch)
                res //= 2
        self.mid1 = ResBlock(ch, ch, cond_dim, dropout)
        self.mid_attn = Attention(ch, num_heads)
        self.mid2 = ResBlock(ch, ch, cond_dim, dropout)
        self.ups = nn.ModuleList()
        for i, out_ch in reversed(list(enumerate(chans))):
            for _ in range(num_res_blocks + 1):
                self.ups.append(nn.ModuleList([
                    ResBlock(ch + skip_ch.pop(), out_ch, cond_dim, dropout),
                    Attention(out_ch, num_heads) if res in attn_resolutions else nn.Identity()]))
                ch = out_ch
            if i > 0:
                self.ups.append(nn.ModuleList([
                    nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                  nn.Conv2d(ch, ch, 3, padding=1, padding_mode="reflect")),
                    nn.Identity()]))
                res *= 2
        self.out_norm = _gn(ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1, padding_mode="reflect")
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        self.size_multiple = 2 ** (len(ch_mult) - 1)

    def forward(self, x, t, style=None):
        if style is None:
            style = self.null_style.expand(x.shape[0], -1)
        cond = self.time_mlp(timestep_embedding(t, self.base)) + self.style_mlp(style)
        H0, W0 = x.shape[-2:]
        ph, pw = (-H0) % self.size_multiple, (-W0) % self.size_multiple
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect" if (ph < H0 and pw < W0) else "replicate")
        h = self.stem(x)
        skips = [h]
        for blk, attn in self.downs:
            h = attn(blk(h, cond)) if isinstance(blk, ResBlock) else blk(h)
            skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, cond)), cond)
        for blk, attn in self.ups:
            h = attn(blk(torch.cat([h, skips.pop()], 1), cond)) if isinstance(blk, ResBlock) else blk(h)
        out = self.out_conv(F.silu(self.out_norm(h)))
        return out[..., :H0, :W0] if (ph or pw) else out


class SourceFlow(nn.Module):
    """FlowUNet with the source image as a second input channel. The flow
    denoiser (x2x) and the supervised denoiser are both this network."""

    def __init__(self, style_dim, base=64, crop=128):
        super().__init__()
        self.net = FlowUNet(in_channels=2, base=base, ch_mult=(1, 2, 2, 4), num_res_blocks=2,
                            style_dim=style_dim, attn_resolutions=(16,), image_size=crop)

    def forward(self, xt, src, t, style=None):
        return self.net(torch.cat([xt, src], 1), t, style)[:, :1]


@torch.no_grad()
def sample(model, src, style, steps=32, guidance=1.0, gen=None, temperature=1.0, x0=None, solver="heun"):
    """Integrate from t = 0 (noise) to t = 1 (image). Heun with a plain Euler
    LAST step: the final Heun corrector evaluates the velocity at t = 1 and
    injects fine grain into sparse low-dose images; with the last step in
    Euler, Heun-32 matches Euler-128 at half the cost."""
    x = (torch.randn(src.shape, device=src.device, generator=gen) if x0 is None else x0.clone()) * temperature

    def vel(x, t):
        v = model(x, src, t, style)
        if guidance != 1.0:
            v_u = model(x, src, t, None)
            v = v_u + guidance * (v - v_u)
        return v

    for i in range(steps):
        t = torch.full((src.shape[0],), i / steps, device=src.device)
        v = vel(x, t)
        if solver == "heun" and i < steps - 1:
            v = 0.5 * (v + vel(x + v / steps, torch.full_like(t, (i + 1) / steps)))
        x = x + v / steps
    return x
