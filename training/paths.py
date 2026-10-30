"""Local data and weight locations for training, read from training/paths.toml
(or $CIMP_PATHS); copy training/paths.example.toml to start. The denoise
package itself needs none of this."""
import os
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load():
    p = Path(os.environ.get("CIMP_PATHS", Path(__file__).resolve().parent / "paths.toml"))
    with open(p, "rb") as f:
        return tomllib.load(f)


_P = _load()
SOURCE_H5 = Path(_P["data"]["source_h5"])
RAW_ROOT = Path(_P["data"]["raw_root"])
STACKS_H5 = Path(_P["data"]["stacks_h5"])
CMMP_WEIGHTS = Path(_P["weights"]["cmmp"])
RUNS = Path(_P["runs"]["root"])
# optional: how paths recorded in the data files map onto the local raw archive
_A = _P.get("archive", {})
ARCHIVE_ANCHOR = _A.get("anchor")                       # folder name the recorded paths share with raw_root
ARCHIVE_ALIASES = tuple(tuple(x) for x in _A.get("aliases", ()))   # (recorded name, current name) folder renames
