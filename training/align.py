"""Series alignment: series_alignment >= 1.3 with integer shifts.

series_alignment 1.3 tracks the drift locally before refining (long series no
longer split into blocks), keeps the refinement within 0.4 lattice periods on
crystals (no hops to a neighbouring column) and always drops aborted,
half-written frames. Integer shifts and a crop to the region every frame
covers: pixels stay as acquired, noise statistics untouched.
"""
import numpy as np
from series_alignment import __version__ as SA_VERSION
from series_alignment import align_stack

assert tuple(int(v) for v in SA_VERSION.split(".")[:2]) >= (1, 3), \
    f"series_alignment {SA_VERSION}: >= 1.3 required (local drift tracking, lattice-safe search, aborted-frame removal)"


def align_series(stack, device="auto"):
    """(N, H, W) uint16 series -> (aligned (M, h, w) uint16, used (M,) positions
    into `stack`, shifts (M, 2) integer, info dict)."""
    stack = np.asarray(stack)
    if len(stack) < 2:
        return stack.copy(), np.arange(len(stack)), np.zeros((len(stack), 2), int), {}
    r = align_stack(stack, device=device, crop_mode="shared", subpixel=False,
                    drop_unreliable=True, repair_output=True)
    used = np.nonzero(np.asarray(r.used, bool))[0]
    shifts = np.asarray(r.shifts).astype(int)
    d = r.diagnostics
    info = dict(confidence=np.asarray(r.confidence, float)[used],
                max_step=int(np.abs(np.diff(shifts, axis=0)).max()) if len(shifts) > 1 else 0,
                drift_span=(shifts.max(0) - shifts.min(0)).tolist(),
                lattice_period_px=d.get("lattice_period_px"), aligner=f"series_alignment {SA_VERSION}",
                flags=[list(r.flags[i]) for i in used])          # one list per returned frame (v3 stacks:
                                                                  # only the non-empty ones, frame index lost)
    return np.asarray(r.aligned_stack).astype(np.uint16), used, shifts, info
