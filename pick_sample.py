"""
Pick a random, mutually-dissimilar sample of HDR images for a fresh sweep.

Why this exists: HDR-Real (SingleHDR training data) looks like 9,786 images but
holds only 820 unique HDRs -- each ground-truth HDR is stored once per paired
LDR photo, so many files are byte-identical. A plain random draw is therefore
full of duplicates (the first 1K sample had only 575 unique images), and those
duplicates leak across GroupKFold folds.

Method:
  1. pool = HDR-Real/HDR_gt + every HDR-Synth source folder
  2. drop byte-identical copies (md5), keeping one file per hash
  3. drop blank/flat images (same rule as sweep.py preflight)
  4. descriptor = z-scored log-luminance 32x32 thumbnail (as scene_grouping.py)
  5. shuffle with --seed and greedily accept an image only if its Pearson
     correlation with EVERY already-accepted image is below --max-corr
     (random order + pairwise rejection = a random sample with no near-duplicates)
  6. resize to --long-edge (INTER_AREA, aspect kept) and write Radiance .hdr
     files named by fresh integer ids starting at --first-id

Output: images/<id>.hdr plus results/manifest.csv mapping every new id back
to its source file, dataset, md5, and (with --old-images) whether that exact
HDR was in a previous sample.
"""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import glob
import hashlib
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

HOME_DATA = os.path.expanduser("~/Downloads/SingleHDR_training_data")
THUMB = (32, 32)
DEGENERATE_COV = 1e-4   # matches sweep.py


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_hdr(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("unreadable")
    img = img[..., :3].astype(np.float32)
    return np.clip(np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0), 0, None)


def describe(path):
    """(md5, descriptor or None, (h, w), reason)"""
    digest = md5(path)
    try:
        hdr = read_hdr(path)
    except Exception as e:
        return digest, None, None, str(e)
    lum = 0.114 * hdr[..., 0] + 0.587 * hdr[..., 1] + 0.299 * hdr[..., 2]
    if lum.max() <= 0 or lum.std() / (lum.mean() + 1e-12) < DEGENERATE_COV:
        return digest, None, hdr.shape[:2], "blank/flat"
    t = cv2.resize(np.log1p(lum), THUMB, interpolation=cv2.INTER_AREA)
    t = (t - t.mean()) / (t.std() + 1e-8)
    return digest, t.ravel().astype(np.float32), hdr.shape[:2], ""


def build_pool(data_root):
    real = sorted(glob.glob(os.path.join(data_root, "HDR-Real", "HDR_gt", "*.hdr")))
    synth = sorted(p for p in glob.glob(os.path.join(data_root, "HDR-Synth", "**", "*"),
                                        recursive=True)
                   if p.lower().endswith((".hdr", ".exr")))
    return [(p, "HDR-Real") for p in real] + \
           [(p, "HDR-Synth/" + os.path.relpath(p, os.path.join(data_root, "HDR-Synth")).split(os.sep)[0])
            for p in synth]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=HOME_DATA)
    ap.add_argument("--images-out", default="images")
    ap.add_argument("--manifest", default="results/manifest.csv")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-corr", type=float, default=0.85,
                    help="reject a candidate whose thumbnail correlation with any "
                         "accepted image is >= this")
    ap.add_argument("--long-edge", type=int, default=512)
    ap.add_argument("--first-id", type=int, default=10000,
                    help="new ids start here so they never collide with HDR-Real ids")
    ap.add_argument("--old-images", default=None,
                    help="folder of a previous sample, only used to flag overlap")
    args = ap.parse_args()

    pool = build_pool(args.data_root)
    print(f"pool: {len(pool)} files")
    with ProcessPoolExecutor() as ex:
        info = list(ex.map(describe, [p for p, _ in pool], chunksize=32))

    # one file per md5 (first in sorted order), and only usable images
    seen, cand = set(), []
    n_dup = n_bad = 0
    for (path, ds), (digest, desc, shape, why) in zip(pool, info):
        if digest in seen:
            n_dup += 1
            continue
        seen.add(digest)
        if desc is None:
            n_bad += 1
            print(f"  skip {path}: {why}")
            continue
        cand.append(dict(path=path, dataset=ds, md5=digest, desc=desc,
                         orig_h=shape[0], orig_w=shape[1]))
    print(f"dropped {n_dup} byte-identical copies, {n_bad} unusable -> {len(cand)} unique candidates")

    D = np.stack([c["desc"] for c in cand])
    D /= np.linalg.norm(D, axis=1, keepdims=True)   # z-scored, so dot = Pearson r

    rng = np.random.default_rng(args.seed)
    accepted = []
    for k in rng.permutation(len(cand)):
        if accepted and float((D[accepted] @ D[k]).max()) >= args.max_corr:
            continue
        accepted.append(k)
        if len(accepted) == args.n:
            break
    if len(accepted) < args.n:
        raise SystemExit(f"only {len(accepted)} images pass --max-corr {args.max_corr}; "
                         f"raise it or lower --n")
    sub = D[accepted] @ D[accepted].T
    np.fill_diagonal(sub, -1)
    print(f"accepted {len(accepted)} (max pairwise r = {sub.max():.3f})")

    old_md5 = ({md5(p) for p in glob.glob(os.path.join(args.old_images, "*.hdr"))}
               if args.old_images else None)

    img_dir = args.images_out
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for new_id, k in enumerate(sorted(accepted, key=lambda k: cand[k]["path"]), args.first_id):
        c = cand[k]
        hdr = read_hdr(c["path"])
        h, w = hdr.shape[:2]
        scale = args.long_edge / max(h, w)
        if scale < 1:
            hdr = cv2.resize(hdr, (round(w * scale), round(h * scale)),
                             interpolation=cv2.INTER_AREA)
        fname = f"{new_id:05d}.hdr"
        cv2.imwrite(os.path.join(img_dir, fname), hdr)
        rows.append(dict(image=new_id, file=fname, dataset=c["dataset"],
                         source=os.path.relpath(c["path"], args.data_root), md5=c["md5"],
                         orig_w=c["orig_w"], orig_h=c["orig_h"],
                         w=hdr.shape[1], h=hdr.shape[0]))
        if old_md5 is not None:
            rows[-1]["in_old_sample"] = c["md5"] in old_md5

    man = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    man.to_csv(args.manifest, index=False)
    print(f"wrote {len(man)} images -> {img_dir}, manifest -> {args.manifest}")
    print(man.dataset.value_counts().to_string())
    if old_md5 is not None:
        print(f"exact HDRs shared with old sample: {man.in_old_sample.sum()}")


if __name__ == "__main__":
    main()
