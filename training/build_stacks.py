"""
build_stacks.py -- the multi-frame training stacks both flow models learn from.

Every frame of a stack is the same specimen with an independent noise
realisation, and the mean of k frames is that specimen at k x the dose: paired
(low dose, high dose) examples with no detector model and no noise assumption.

One entry per multi-frame acquisition of the single-frame source index (the
data CMMP was trained on). Three things the previous builds got wrong, fixed:

  CHANNEL   multi-detector files (HAADF-DF4-BF ...) hold several Raw channels and
            the old loader took the first: 41 of 87 such files gave DF4, BF or
            DF2 frames under HAADF metadata (contrast inverted for BF). Here the
            channel is the one whose frame matches the image stored in the source
            index -- i.e. exactly the channel CMMP saw -- and a plain "HAADF"
            channel is preferred over a derived one (Average/Crop) when both match.
  SEGMENT   Velox series can change magnification mid-acquisition. Frames are
            taken only from the constant-settings segment that contains the
            source index's frame (whose metadata the stack inherits), without the
            part-scanned transition frame.
  ALIGNMENT series_alignment >= 1.3 (training/align.py): drift tracked locally
            before refinement, so long series no longer split into blocks and
            crystals cannot lock a lattice spacing off; aborted half-written
            frames (zero rows, typically a series' last frame -- 70 stacks
            reached one) are dropped.

Unchanged on purpose: the split (inherited from the source entry, seed 67, so a
field of view never straddles train/val), the frame budget (k x dwell may not
exceed the dataset's p99 dwell; at most --max-frames), uint16 frames shifted by
whole pixels only (noise statistics untouched), and the QC gates.

Output layout = the old v2 file's, chunked (1, 256, 256) for fast random crops
(the separate _c256 copy is no longer needed).

    python data/build_stacks.py --audit             # what the old build got wrong, no writing
    python data/build_stacks.py --limit 5           # smoke test
    python data/build_stacks.py                     # full build (CPU; --device cuda to speed up)
    python data/build_stacks.py --resume            # continue an interrupted build
"""
import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.align import align_series                                 # noqa: E402
from cimp_denoise.io import frame_meta, load_frames, raw_channels                     # noqa: E402
from training.emd_paths import resolve_path, segments                   # noqa: E402
from cimp_denoise.imops import to8                                         # noqa: E402
from training.paths import SOURCE_H5, STACKS_H5                        # noqa: E402

SPLIT_SEED, TRAIN_RATIO = 67, 0.9


def split_assignment(n):
    """The source index's train/val split (seed 67, shuffle, first 90 % train)."""
    rng = np.random.RandomState(SPLIT_SEED)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = np.empty(n, dtype=object)
    split[idx[:int(n * TRAIN_RATIO)]] = "train"
    split[idx[int(n * TRAIN_RATIO):]] = "val"
    return split


def frame_budget(dwell_s, n_available, ceiling_s, cap):
    """k frames with k x dwell <= the dataset's dwell ceiling, at most cap."""
    # x (1 + 1e-4): the ceiling is a percentile of float32 dwells (9.99999 us), and 10 us / 2 us must
    # give 5 frames, not 4 (the old builds were one frame short at every dwell)
    return int(min(max(1, np.floor(ceiling_s / max(dwell_s, 1e-12) * (1 + 1e-4))), n_available, cap))


def noise_scaling(stack, n_max=64):
    """Var(mean of N) N / Var(single); ~1 for independent, dose-linear frames."""
    s = stack[:min(len(stack), n_max)].astype(np.float64)
    v1 = float(((s[0] - s[1]) ** 2).mean() / 2.0)
    h = len(s) // 2
    va = float(((s[:h].mean(0) - s[h:2 * h].mean(0)) ** 2).mean() / 2.0)
    return va * h / max(v1, 1e-12)


def pick_channel(f, frame_idx, ref):
    """Raw channel whose frame `frame_idx` matches the source-index image.
    Returns (uid, label, corr) or (None, reason, corr)."""
    ref = ref.astype(np.float32).ravel()
    scored = []
    for uid, lab in raw_channels(f).items():
        ds = f["Data/Image"][uid]["Data"]
        if ds.shape[0] * ds.shape[1] != ref.size:
            continue
        k = min(frame_idx, ds.shape[-1] - 1) if ds.ndim == 3 else None
        a = (ds[:, :, k] if ds.ndim == 3 else ds[:]).astype(np.float32).ravel()
        if a.std() == 0:
            continue
        scored.append((uid, lab, float(np.corrcoef(a, ref)[0, 1])))
    if not scored:
        return None, "no Raw channel of the source image's size", 0.0
    best = max(scored, key=lambda s: s[2])
    plain = [s for s in scored if s[1].upper() == "HAADF" and s[2] > 0.9]
    if plain and "HAADF" in best[1].upper():
        return plain[0]                       # a derived HAADF channel (Average/Crop) matched best
    if best[2] < 0.99:
        return None, f"best channel {best[1]} correlates only {best[2]:.3f} with the source image", best[2]
    return best


def series_fingerprint(path):
    """md5 of the HAADF channel's first frame + its shape: the same series copied
    into several folders (e.g. an 'Analysis' copy of a session file) is found
    even though the source index stores a different frame of each copy."""
    import hashlib
    with h5py.File(path, "r") as f:
        ch = raw_channels(f)
        uid = next((u for u, lab in ch.items() if lab.upper() == "HAADF"), next(iter(ch)))
        ds = f["Data/Image"][uid]["Data"]
        a = ds[:, :, 0] if ds.ndim == 3 else ds[:]
        return hashlib.md5(a.tobytes()).hexdigest() + str(ds.shape)


def dedupe(cand, resolved, split):
    """One entry per distinct series. Copies straddling train/val keep the VAL
    copy, so held-out content never also appears in training."""
    groups = {}
    for i in cand:
        try:
            groups.setdefault(series_fingerprint(resolved[i]), []).append(i)
        except Exception:                                           # noqa: BLE001
            groups.setdefault(f"unreadable{i}", []).append(i)
    keep, dropped = [], []
    for v in groups.values():
        rep = min(v, key=lambda i: (split[i] != "val", i))
        keep.append(rep)
        dropped += [(i, rep) for i in v if i != rep]
    return keep, dropped


def qc(shape, aligned, shifts, cfg):
    H, W = shape[-2:]
    span = shifts.max(0) - shifts.min(0)
    if max(span[0] / H, span[1] / W) > cfg["max_span_frac"]:
        return False, f"drift span {tuple(span)} px"
    area = aligned.shape[-2] * aligned.shape[-1] / (H * W)
    if area < cfg["min_area_frac"]:
        return False, f"crop kept only {area:.0%} of the frame"
    ns = noise_scaling(aligned)
    if not cfg["noise_lo"] <= ns <= cfg["noise_hi"]:
        return False, f"noise scaling {ns:.2f}"
    return True, f"ok (span {tuple(int(v) for v in span)}, area {area:.0%}, noise {ns:.2f})"


def contact_sheet(items, path, tile=256, cols=6):
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("L", (cols * (tile + 6) + 6, rows * (tile + 28) + 6), 255)
    d = ImageDraw.Draw(canvas)
    for k, (img, lab) in enumerate(items):
        r, c = divmod(k, cols)
        x, y = 6 + c * (tile + 6), 6 + r * (tile + 28)
        canvas.paste(Image.fromarray(to8(img)).resize((tile, tile), Image.BILINEAR), (x, y + 22))
        for j, line in enumerate(lab.split("\n")[:2]):
            d.text((x, y + 2 + 10 * j), line, fill=0)
    tmp = str(path) + ".tmp.png"
    canvas.save(tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STACKS_H5))
    ap.add_argument("--min-frames", type=int, default=8, help="frames the source series must have")
    ap.add_argument("--max-frames", type=int, default=8, help="frames kept per stack")
    ap.add_argument("--dose-ceiling-pct", type=float, default=99.0,
                    help="k x dwell may not exceed this percentile of the dataset's dwell")
    ap.add_argument("--max-span-frac", type=float, default=0.10)
    ap.add_argument("--min-area-frac", type=float, default=0.50)
    ap.add_argument("--noise-lo", type=float, default=0.7)
    ap.add_argument("--noise-hi", type=float, default=1.8)
    ap.add_argument("--device", default="cpu", help="series_alignment refine step: cpu | cuda | auto")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keys", default="", help="comma list of source keys to build (e.g. 001855,001873)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--audit", action="store_true",
                    help="report channel / segment problems of every candidate; write nothing")
    a = ap.parse_args()
    cfg = dict(max_span_frac=a.max_span_frac, min_area_frac=a.min_area_frac,
               noise_lo=a.noise_lo, noise_hi=a.noise_hi)

    with h5py.File(SOURCE_H5, "r") as f:
        paths = [s.decode() for s in f["filepaths"][:]]
        total, fidx = f["total_frames"][:], f["frame_indices"][:]
        meta, meta_raw = f["metadata"][:], f["metadata_raw"][:]
        s_mean, s_std = f["stats_mean"][:], f["stats_std"][:]
        labels = f.attrs.get("metadata_labels")
    split = split_assignment(len(paths))
    ceiling = float(np.percentile(meta_raw[:, 1], a.dose_ceiling_pct))
    resolved = [resolve_path(p) for p in paths]
    cand = sorted((i for i in range(len(paths)) if total[i] >= a.min_frames and resolved[i]),
                  key=lambda i: -total[i])
    if a.keys:
        want = {int(k) for k in a.keys.split(",")}
        cand = [i for i in cand if i in want]
    cand_set = set(dedupe(cand, resolved, split)[0])
    n_dup = len(cand) - len(cand_set)
    cand = [i for i in cand if i in cand_set]
    if n_dup:
        print(f"{n_dup} entries are copies of another series (same data in another folder) and are skipped",
              flush=True)
    if a.limit:
        cand = cand[:a.limit]
    print(f"{len(cand)} multi-frame acquisitions (>= {a.min_frames} frames, file found); "
          f"dwell ceiling p{a.dose_ceiling_pct:g} = {ceiling*1e6:.2f} us, <= {a.max_frames} frames a stack",
          flush=True)

    out_path = Path(a.out)
    prev_dir = out_path.with_name(out_path.stem + "_previews")
    prev_dir.mkdir(parents=True, exist_ok=True)
    mode = "a" if (a.resume and out_path.exists()) else ("r" if a.audit else "w")
    out = None if a.audit else h5py.File(out_path, mode)
    grp = None if a.audit else out.require_group("stacks")
    done = set(grp.keys()) if grp is not None else set()
    audit = dict(channel_fixed=[], first_frames_cross_mag=[], rejected=[])
    recent, kept, t0 = [], 0, time.time()

    with h5py.File(SOURCE_H5, "r") as src_f:
        for c, i in enumerate(cand):
            key = f"{i:06d}"
            if key in done:
                continue
            name = os.path.basename(resolved[i])
            try:
                with h5py.File(resolved[i], "r") as f:
                    first_uid = next(iter(raw_channels(f)), None)
                    uid, lab, corr = pick_channel(f, int(fidx[i]), src_f["full_res"][key][:])
                if uid is None:
                    raise ValueError(lab)
                if uid != first_uid:
                    audit["channel_fixed"].append((key, lab))
                raw, ftime = frame_meta(resolved[i], uid)
                segs = segments(raw, min_frames=1)
                seg = next(((s, e) for s, e in segs if s <= fidx[i] < e), None)
                if np.ptp(raw[:a.max_frames, 0]) > 0 or np.ptp(raw[:a.max_frames, 1]) > 0:
                    audit["first_frames_cross_mag"].append(key)
                if seg is None:
                    raise ValueError(f"source frame {fidx[i]} is a magnification-transition frame")
                s, e = seg
                if not np.allclose(raw[s, [0, 1]], meta_raw[i][[0, 1]], rtol=1e-4):
                    raise ValueError("segment pixel size/dwell differ from the source metadata")
                nb = frame_budget(float(meta_raw[i][1]), e - s, ceiling, a.max_frames)
                if nb < 2:
                    raise ValueError(f"only {e - s} frame(s) in the segment / budget {nb}")
                if a.audit:
                    continue
                # long stacks drift further: if the drift / crop gates fail, retry with
                # half the frames (down to 8), so a longer cap never loses a stack
                while True:
                    stack = load_frames(resolved[i], uid, s, s + nb)
                    try:
                        al, used, shifts, info = align_series(stack, device=a.device)
                        ok, why = qc(stack.shape, al, shifts, cfg)
                    except ValueError as ex:                   # drift leaves < half the frame
                        ok, why = False, f"drift: {ex}"
                    if ok or nb <= 8:
                        break
                    print(f"    {name}: {why} with {nb} frames, retrying with {max(8, nb // 2)}", flush=True)
                    nb = max(8, nb // 2)                  # never below the old build's 8
                if not ok:
                    raise ValueError(why)
            except Exception as ex:                                   # noqa: BLE001
                audit["rejected"].append((key, f"{type(ex).__name__}: {str(ex)[:90]}"))
                if not a.audit:
                    print(f"[{c+1}/{len(cand)}] {name}: REJECT {str(ex)[:90]}", flush=True)
                continue

            g = grp.create_group(key)
            g.create_dataset("frames", data=al, compression="gzip", compression_opts=1,
                             chunks=(1, min(256, al.shape[1]), min(256, al.shape[2])))
            g.create_dataset("shifts", data=shifts.astype(np.int32))
            g.create_dataset("confidence", data=info["confidence"].astype(np.float32))
            g.create_dataset("emd_frames", data=(s + used).astype(np.int32))
            g.create_dataset("mean", data=al.astype(np.float32).mean(0) / 65535.0,
                             compression="gzip", compression_opts=1)
            g.attrs.update(flags=str(info.get("flags", [])), source_index=i, source_path=resolved[i], split=split[i], channel=lab,
                           channel_corr=corr, segment=(s, e), n_frames=len(al), qc=why,
                           max_step=info["max_step"], frame_time=float(ftime[s]))
            kept += 1
            print(f"[{c+1}/{len(cand)}] {name}: {len(al)} fr {al.shape[1]}x{al.shape[2]} [{lab}, {split[i]}] "
                  f"{why}, max step {info['max_step']} px  ({(time.time()-t0)/60:.1f} min)", flush=True)
            recent = (recent + [(al.astype(np.float32).mean(0), f"{key} {split[i]} {lab}\n"
                                 f"{meta_raw[i][1]*1e6:.2g}us n={len(al)}")])[-12:]
            if kept % 10 == 0:
                contact_sheet(recent, prev_dir / "latest.png")

    if a.audit:
        print(f"\nAUDIT of {len(cand)} candidates:")
        print(f"  non-HAADF channel taken by the old loader: {len(audit['channel_fixed'])} "
              f"-> now {sorted(set(l for _, l in audit['channel_fixed']))}")
        print(f"  first {a.max_frames} frames span a magnification/dwell change: "
              f"{len(audit['first_frames_cross_mag'])}")
        print(f"  would be rejected before alignment: {len(audit['rejected'])}")
        for k, w in audit["rejected"][:30]:
            print(f"     {k}: {w}")
        return

    for k in ("index", "metadata", "metadata_raw", "stats_mean", "stats_std"):
        if k in out:
            del out[k]
    keys = sorted(grp.keys())
    if keys:
        srci = np.array([grp[k].attrs["source_index"] for k in keys], np.int64)
        out.create_dataset("index/key", data=np.array(keys, dtype="S16"))
        out.create_dataset("index/source_index", data=srci)
        out.create_dataset("index/split", data=np.array([grp[k].attrs["split"] for k in keys], dtype="S8"))
        out.create_dataset("index/n_frames", data=np.array([grp[k].attrs["n_frames"] for k in keys], np.int32))
        out.create_dataset("index/shape", data=np.array([grp[k]["frames"].shape[1:] for k in keys], np.int32))
        out.create_dataset("metadata", data=meta[srci].astype(np.float32))
        out.create_dataset("metadata_raw", data=meta_raw[srci].astype(np.float32))
    out.create_dataset("stats_mean", data=s_mean)
    out.create_dataset("stats_std", data=s_std)
    from training.align import SA_VERSION
    out.attrs.update(builder="CIMP-flow data/build_stacks.py", alignment=f"series_alignment {SA_VERSION}",
                     channel="matched to the source-index image", split_seed=SPLIT_SEED,
                     train_ratio=TRAIN_RATIO, max_frames=a.max_frames, dwell_ceiling_s=ceiling,
                     note="uint16 frames shifted by whole pixels; divide by 65535")
    if labels is not None:
        out.attrs["metadata_labels"] = labels
    out.close()
    if recent:
        contact_sheet(recent, prev_dir / "latest.png")
    print(f"\nkept {kept} new ({len(keys)} in file), rejected {len(audit['rejected'])}, "
          f"{len(audit['channel_fixed'])} channel fixes  ({(time.time()-t0)/60:.1f} min)")
    for k, w in audit["rejected"][:40]:
        print(f"   REJECTED {k}: {w}")


if __name__ == "__main__":
    main()
