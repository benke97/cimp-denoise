"""cimp-denoise: STEM denoising by flow matching, steered by CMMP embeddings."""
__version__ = "0.1.0"


def __getattr__(name):                      # import torch lazily (keeps `denoise --help` fast)
    if name in ("Denoiser", "frame_embedding"):
        from . import api
        return getattr(api, name)
    raise AttributeError(name)
