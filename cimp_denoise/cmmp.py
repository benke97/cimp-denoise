"""CMMP: contrastive metadata-image pretraining. A frozen encoder pair that maps
an acquisition's metadata and an image of it into one 128-d space. The denoiser
is steered by points in that space: either the embedding of the acquisition's
metadata, or the embedding of an image ("style")."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

EMB = 128


class MetadataEncoder(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=256, output_dim=EMB, dropout_prob=0.2, num_layers=3):
        super().__init__()
        layers, d = [], input_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout_prob)]
            d = hidden_dim
        self.net = nn.Sequential(*layers, nn.Linear(d, output_dim))

    def forward(self, x):
        return self.net(x)


class ImageEncoder(nn.Module):
    """ResNet-18 on single-channel input."""

    def __init__(self, output_dim=EMB):
        super().__init__()
        self.model = models.resnet18(weights=None)
        c = self.model.conv1
        self.model.conv1 = nn.Conv2d(1, c.out_channels, c.kernel_size, c.stride, c.padding, bias=c.bias)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dim)

    def forward(self, x):
        return self.model(x)


class CMMP(nn.Module):
    def __init__(self, meta_input_dim=7, embed_dim=EMB, meta_hidden_dim=256, meta_num_layers=3):
        super().__init__()
        self.visual = ImageEncoder(embed_dim)
        self.meta = MetadataEncoder(meta_input_dim, meta_hidden_dim, embed_dim, num_layers=meta_num_layers)
        self.t_prime = nn.Parameter(torch.tensor(np.log(10.0)))
        self.b = nn.Parameter(torch.tensor(-10.0))

    @torch.no_grad()
    def embed_meta(self, m):
        """(B, 7) z-scored metadata -> (B, 128) unit vectors."""
        return F.normalize(self.meta(m), dim=-1)

    @torch.no_grad()
    def embed_image(self, x):
        """(B, 1, H, W) image in [0, 1] -> (B, 128) unit vectors."""
        return F.normalize(self.visual(x), dim=-1)


def load_cmmp(path, dev):
    """The frozen encoder."""
    m = CMMP().to(dev)
    sd = torch.load(path, map_location=dev, weights_only=False)
    m.load_state_dict(sd.get("model_state_dict", sd))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m
