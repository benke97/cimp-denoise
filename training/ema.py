"""Weight EMA used by both trainers."""
import torch


class EMA:
    """Weight EMA with warmup ((1 + n) / (10 + n) up to `decay`)."""

    def __init__(self, model, decay=0.999):
        self.m = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.m[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.m[k].copy_(v)
