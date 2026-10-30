# cimp-denoise

Denoise a single STEM frame so it looks like a long exposure, with no
retraining for your microscope.

```
pip install git+https://github.com/benke97/cimp-denoise
denoise frame.emd                 # -> frame_HAADF_denoised.tif (+ one file per detector channel)
denoise my_folder/ -o denoised/   # every .emd / .dm3 / .dm4 / .tif / .png in a folder
```

The first run downloads the weights (~275 MB) from the Hugging Face repository
[Stemson-AI/cimp-denoise](https://huggingface.co/Stemson-AI/cimp-denoise) to
`~/.cache/cimp-denoise` (or `$CIMP_DENOISE_WEIGHTS`). A CUDA GPU is strongly
recommended; on a CPU it is very slow.

## How it works

**CMMP** (contrastive metadata–image pretraining) maps an image and the
acquisition that produced it (pixel size, dwell time, beam current, detector
gain and angles) into one shared 128-d space. A point in that space describes
how an image *looks*: its noise, contrast and detail at a given dose.

**The denoiser** is a conditional flow-matching network (x2x) trained on real
multi-frame series: given a frame and two CMMP points, "how the input looks"
and "how the output should look", it renders the same specimen in the new look.
Averaging several samples gives the final image.

Where the two points come from:

| mode | source (input look) | target (output look) |
|---|---|---|
| **metadata** (default for .emd) | CMMP(metadata at min(recorded dwell, 1 us)) | CMMP(metadata at 16x that) |
| **cmmp** (no metadata, or `--mode cmmp`) | CMMP(the frame) | CMMP(a quick supervised denoise of the frame) |

Each mode has its own x2x weights. The CMMP-mode model was additionally
trained with CMMP image embeddings as the source description (30 % of
samples), so the input it sees at use time is in its training distribution.

The CMMP mode needs nothing but the image, which makes it the one to use for
microscopes whose metadata means something different from ours: the target is
"this very frame, clean", and its description comes from the image itself.

## Options

```
denoise INPUT... [-o OUT] [--mode auto|metadata|cmmp] [--factor 16] [--push 0]
                 [--channels HAADF,BF-S] [--frames first|all] [--draws 8] [--steps 4]
                 [--device cuda|cpu] [--preview]
```

- `--factor`: metadata mode, target dwell = factor x min(recorded dwell, 1 us). The 1 us cap keeps
  the target inside the dwell range x2x was trained on; long-dwell frames come out grainy without it.
- `--push`: CMMP mode, moves the source description further towards "noisy",
  so more noise is removed (try 1).
- `--preview`: also writes a PNG with input | denoised.

Output is a 32-bit TIFF in the input's units.

From Python:

```python
from cimp_denoise import Denoiser
from cimp_denoise.io import read

den = Denoiser()                       # weights are fetched on first use
im = read("frame.emd", channels=["HAADF"])[0]
clean = den(im.data, metadata=im.metadata)      # metadata=None -> CMMP mode
```

## Limitations

- Trained on HAADF data from one microscope only. Other detector channels
  (BF, DF) are processed too, but the models were not trained for them.
- Slow: the default 8 draws x 4 steps take ~5 s per 512 x 512 px on a desktop GPU
  (~1.5 min for a 2048 x 2048 frame).
- Images must be at least 128 px per side.

## Repository

| | |
|---|---|
| `cimp_denoise/` | the package: CMMP encoder, flow network and sampler, tiling, I/O, CLI |
| `training/` | how the weights were made: multi-frame stacks, x2x and supervised training, weight export |

Training needs the raw data and `pip install -e .[training]`; copy
`training/paths.example.toml` to `training/paths.toml` and set your locations:

```
python training/build_stacks.py                       # multi-frame stacks from raw Velox series
python training/train_x2x.py --run-name x2x_v2_local --pairing local --epochs 500
python training/train_x2x.py --run-name x2x_v3_imgsrc --init runs/x2x_v2_local/ep400_before_decay.pt \
    --epochs 100 --lr 2e-4 --lr-decay-from 1 --warmup-steps 1000 --pairing local --max-frames 32 \
    --p-image 0.3 --p-image-src 0.3
python training/train_supervised.py --run-name sup_v1 --epochs 120
python training/export_weights.py --out release/weights
```

The full arguments of each trained model are stored in its training
checkpoint under `args`.
