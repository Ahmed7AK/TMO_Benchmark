"""
Extract per-image HDR features for the Stage-3 selector (train_selector.py).

Every feature is computed on the raw linear-radiance HDR (via tmo_pipeline's
own load_hdr, so it matches what the operators actually saw -- no resolution
cap was used for the q_matrix.csv sweep, so none is applied here either).
Features fall into four groups:

  - luminance distribution: log-average (Reinhard key), percentiles, dynamic
    range, key value, log-domain std/skew/kurtosis, Shannon entropy
  - clipping: shadow/highlight pixel fractions relative to the image's own
    median (robust to a single outlier pixel, unlike absolute thresholds)
  - spatial/detail: edge density, local-contrast std, high-frequency energy
    ratio, all on a downsized log-luminance thumbnail (bounds compute cost,
    and detail perception is roughly scale-invariant at this level)
  - color: chroma-vs-luminance ratio, and the Hasler-Susstrunk colorfulness
    metric (both scale-invariant, so they're meaningful on radiance units)

These are the kind of scene properties that plausibly predict *which*
operator will score well (e.g. high dynamic range + heavy clipping favors
detail-preserving operators; low contrast scenes leave less room for any
operator to differentiate).
"""

import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from tmo_pipeline import load_hdr

EPS = 1e-6
THUMB_LONG_EDGE = 256


def image_id(path):
    return int(os.path.splitext(os.path.basename(path))[0])


def rec601_luminance(hdr_bgr):
    b, g, r = hdr_bgr[..., 0], hdr_bgr[..., 1], hdr_bgr[..., 2]
    return 0.114 * b + 0.587 * g + 0.299 * r


def thumbnail(lum):
    h, w = lum.shape
    s = THUMB_LONG_EDGE / float(max(h, w))
    if s >= 1.0:
        return lum
    return cv2.resize(lum, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                       interpolation=cv2.INTER_AREA)


def colorfulness(hdr_bgr):
    # Hasler & Susstrunk 2003, adapted for linear radiance (scale-invariant ratio).
    b, g, r = hdr_bgr[..., 0], hdr_bgr[..., 1], hdr_bgr[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, mean_rg = rg.std(), rg.mean()
    std_yb, mean_yb = yb.std(), yb.mean()
    std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
    mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)
    scale = np.median(rec601_luminance(hdr_bgr)) + EPS
    return float((std_root + 0.3 * mean_root) / scale)


def compute_features(path):
    hdr = load_hdr(path)
    h, w = hdr.shape[:2]
    lum = rec601_luminance(hdr)
    lum_pos = np.clip(lum, 0, None)
    log_lum = np.log(lum_pos + EPS)

    p1, p5, p50, p95, p99 = np.percentile(log_lum, [1, 5, 50, 95, 99])
    log_avg_lum = float(np.mean(log_lum))          # Reinhard's log-average ("key")
    dynamic_range_log10 = float((p99 - p1) / np.log(10))
    full_range_log10 = float((log_lum.max() - log_lum.min()) / np.log(10))
    key_value = float((log_avg_lum - p1) / (p99 - p1 + EPS))

    hist, _ = np.histogram(log_lum, bins=64, range=(float(p1), float(p99)), density=True)
    hist = hist / (hist.sum() + EPS)
    entropy = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))

    median_lum = np.median(lum_pos) + EPS
    shadow_frac = float(np.mean(lum_pos < 0.01 * median_lum))
    highlight_frac = float(np.mean(lum_pos > 10.0 * median_lum))

    thumb = thumbnail(log_lum)
    gx = cv2.Sobel(thumb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(thumb, cv2.CV_32F, 0, 1, ksize=3)
    edge_density = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
    lap = cv2.Laplacian(thumb, cv2.CV_32F, ksize=3)
    local_contrast_std = float(lap.std())
    high_freq_ratio = float(lap.var() / (thumb.var() + EPS))

    chroma = (hdr.max(axis=2) - hdr.min(axis=2)) / (lum_pos + EPS)
    color_variation_mean = float(chroma.mean())
    color_variation_std = float(chroma.std())

    return {
        "width": w, "height": h, "aspect_ratio": w / h, "megapixels": (w * h) / 1e6,
        "log_avg_luminance": log_avg_lum,
        "dynamic_range_log10": dynamic_range_log10,
        "full_range_log10": full_range_log10,
        "key_value": key_value,
        "log_lum_p1": float(p1), "log_lum_p5": float(p5), "log_lum_p50": float(p50),
        "log_lum_p95": float(p95), "log_lum_p99": float(p99),
        "log_lum_std": float(log_lum.std()),
        "log_lum_skew": float(skew(log_lum, axis=None)),
        "log_lum_kurtosis": float(kurtosis(log_lum, axis=None)),
        "entropy": entropy,
        "shadow_frac": shadow_frac,
        "highlight_frac": highlight_frac,
        "edge_density": edge_density,
        "local_contrast_std": local_contrast_std,
        "high_freq_ratio": high_freq_ratio,
        "color_variation_mean": color_variation_mean,
        "color_variation_std": color_variation_std,
        "colorfulness": colorfulness(hdr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="images")
    ap.add_argument("--out", default="results/hdr_features.csv")
    ap.add_argument("--failures-log", default="results/hdr_features_failures.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images_dir, "*.hdr")), key=image_id)
    print(f"extracting features for {len(paths)} images...")

    rows, failures = [], []
    for n, p in enumerate(paths, 1):
        try:
            feats = compute_features(p)
            feats["image"] = image_id(p)
            rows.append(feats)
        except Exception as e:
            failures.append({"image": image_id(p), "path": p, "error": str(e)})
        if n % 100 == 0:
            print(f"  {n}/{len(paths)}")

    cols = ["image"] + [c for c in rows[0].keys() if c != "image"]
    out = pd.DataFrame(rows)[cols]
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows x {len(cols) - 1} features -> {args.out}")

    if failures:
        os.makedirs(os.path.dirname(args.failures_log), exist_ok=True)
        pd.DataFrame(failures).to_csv(args.failures_log, index=False)
        print(f"{len(failures)} failures logged -> {args.failures_log}")


if __name__ == "__main__":
    main()
