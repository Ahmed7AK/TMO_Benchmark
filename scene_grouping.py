"""
Derive scene groups for HDR-Real images that ship with no scene metadata.

HDR-Real (SingleHDR paper) provides flat, sequentially-numbered .hdr files with
no video/scene id. Frames from the same source video land at adjacent numeric
ids, so id-adjacency is a *candidate* filter, not proof: two adjacent ids are
only merged into one scene if their downsized, contrast-normalized content
actually matches (histogram shape + spatial correlation), so a same-brightness
coincidence between two unrelated adjacent images won't merge them.

Method per image:
  1. load_hdr() (same loader tmo_pipeline.py uses, for consistency)
  2. Rec.601 luminance -> log1p (compress HDR range) -> resize to a small
     fixed-size thumbnail (INTER_AREA)
  3. z-score normalize the thumbnail (zero mean, unit std) so exposure/contrast
     differences between brackets of the same scene don't block a match
  4. compare only images within ID_WINDOW of each other (adjacency filter):
       - Pearson correlation of the normalized thumbnails (spatial match)
       - histogram correlation of the normalized values (tonal-shape match)
     both must clear their threshold to merge
  5. union-find over confirmed pairs -> scene id = group's smallest image id

Output: scenes.csv (image, scene) - join this into q_matrix.csv / features on
the "image" column to get GroupKFold-ready scene ids.
"""

import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd

from tmo_pipeline import load_hdr

THUMB_SIZE = (32, 24)          # (w, h)
HIST_BINS = 32
HIST_RANGE = (-3.0, 3.0)       # z-scored range


def image_id(path):
    return int(os.path.splitext(os.path.basename(path))[0])


def thumbnail(path):
    hdr = load_hdr(path)  # BGR float32
    b, g, r = hdr[..., 0], hdr[..., 1], hdr[..., 2]
    lum = 0.114 * b + 0.587 * g + 0.299 * r
    lum = np.log1p(np.clip(lum, 0, None))
    thumb = cv2.resize(lum, THUMB_SIZE, interpolation=cv2.INTER_AREA)
    std = thumb.std()
    if std < 1e-8:
        return np.zeros_like(thumb, dtype=np.float32)
    return ((thumb - thumb.mean()) / std).astype(np.float32)


def hist_of(thumb):
    h = cv2.calcHist([thumb], [0], None, [HIST_BINS], HIST_RANGE)
    cv2.normalize(h, h)
    return h


class UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="images")
    ap.add_argument("--out", default="results/scenes.csv")
    ap.add_argument("--id-window", type=int, default=5,
                     help="only compare images whose ids are within this many steps")
    ap.add_argument("--corr-thresh", type=float, default=0.97,
                     help="min Pearson correlation of normalized thumbnails to merge")
    ap.add_argument("--hist-thresh", type=float, default=0.97,
                     help="min histogram correlation to merge")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images_dir, "*.hdr")), key=image_id)
    ids = [image_id(p) for p in paths]
    print(f"loading {len(paths)} thumbnails...")

    thumbs, hists = {}, {}
    for i, p in zip(ids, paths):
        try:
            t = thumbnail(p)
        except Exception as e:
            print(f"  skip {p}: {e}")
            continue
        thumbs[i] = t
        hists[i] = hist_of(t)

    valid_ids = sorted(thumbs)
    uf = UnionFind(valid_ids)

    n_pairs = n_merged = 0
    for pos, i in enumerate(valid_ids):
        for j in valid_ids[pos + 1:]:
            if j - i > args.id_window:
                break
            n_pairs += 1
            corr = float(np.corrcoef(thumbs[i].ravel(), thumbs[j].ravel())[0, 1])
            hcorr = float(cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_CORREL))
            if corr >= args.corr_thresh and hcorr >= args.hist_thresh:
                uf.union(i, j)
                n_merged += 1

    scene_of = {i: uf.find(i) for i in valid_ids}
    n_scenes = len(set(scene_of.values()))
    print(f"compared {n_pairs} candidate pairs within id-window={args.id_window}")
    print(f"merged {n_merged} pairs -> {len(valid_ids)} images grouped into {n_scenes} scenes")

    sizes = pd.Series(list(scene_of.values())).value_counts()
    multi = (sizes > 1).sum()
    print(f"{multi} scenes have >1 image (max group size: {sizes.max()})")

    out = pd.DataFrame({"image": valid_ids, "scene": [scene_of[i] for i in valid_ids]})
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
