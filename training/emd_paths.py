"""Training-data helpers for the raw Velox archive: locating recorded paths
and splitting a series into constant-setting segments."""
import os
from pathlib import Path

from training.paths import ARCHIVE_ALIASES, ARCHIVE_ANCHOR, RAW_ROOT


def resolve_path(p):
    """Recorded path -> existing local path, or None. Folder renames and the
    archive's anchor folder come from the [archive] section of paths.toml."""
    if os.path.exists(p):
        return p
    q = p.replace("/", "\\")
    cands = [q] + [q.replace(f"\\{a}\\", f"\\{b}\\") for a, b in ARCHIVE_ALIASES]
    # the raw root may have moved too: re-anchor everything after ...\<anchor>\
    if ARCHIVE_ANCHOR:
        k = f"\\{ARCHIVE_ANCHOR}\\"
        cands += [str(Path(RAW_ROOT) / c.split(k, 1)[1]) for c in list(cands) if k in c]
    return next((c for c in cands if os.path.exists(c)), None)


def segments(raw, min_frames=2, drop_transition=True):
    """Runs of consecutive frames with the same pixel size and dwell, as
    [(start, stop)]. The first frame after a change is part-scanned at the old
    setting -- in Velox its metadata switches one frame before the image does --
    so drop_transition skips it."""
    key = [(round(r[0] * 1e15), round(r[1] * 1e12)) for r in raw]
    out, a = [], 0
    for k in range(1, len(key) + 1):
        if k == len(key) or key[k] != key[a]:
            out.append((a + int(drop_transition and a > 0), k))
            a = k
    return [(a, b) for a, b in out if b - a >= min_frames]
