"""
Wall-clock cost of choosing a TMO per image: the selector against exhaustive search.

  exhaustive  run every candidate operator, score each output with TMQI, keep the best (the oracle's
              cost; it needs the HDR reference and every operator at run time)
  selector    compute the 151 HDR features, predict with the deployed model (select_tmo.py), run only
              the picked operator

Every operator is run fresh into a temporary folder (the tmo_outputs/ cache is not used), the way the
benchmark runs it: luminance-hdr operators as one `docker run` each, the DL operators as one inference
subprocess in their own venv (so model loading is included), att in-process. TMQI is the same
compute_tmqi() the benchmark scored with. Everything runs one image at a time, single process; LightGBM
predicts with one thread. One warm-up image is run first and discarded.

    python timing_benchmark.py [--n 30] [--seed 1234]          # needs Docker and the DL venvs

Writes results/timing/timing_long.csv (image, stage, operator, seconds) and
results/timing/timing_summary.csv; prints the summary.
"""

import argparse
import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

import select_tmo
import selector_features as sf
import tmo_pipeline as P

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results", "timing")


def time_image(img, bundle, ops, registry, tmp):
    path = os.path.join(ROOT, "images", f"{img}.hdr")
    rows = []

    def rec(stage, op, secs):
        rows.append(dict(image=img, stage=stage, operator=op, seconds=secs))

    t0 = time.perf_counter()
    feats = sf.features(*sf.load_path(path))
    rec("features", "", time.perf_counter() - t0)
    t0 = time.perf_counter()
    pick = select_tmo.choose(bundle, select_tmo.predict_scores(bundle, feats))
    rec("predict", "", time.perf_counter() - t0)

    t0 = time.perf_counter()
    hdr_bgr = P.load_hdr(path)
    rec("load_hdr", "", time.perf_counter() - t0)
    for op in ops:
        kind, spec = registry[op]
        out = os.path.join(tmp, f"{img}__{op}.png")
        t0 = time.perf_counter()
        ok = P.produce_ldr(op, kind, spec, path, hdr_bgr, out)
        rec("tmo", op, time.perf_counter() - t0)
        if not ok:
            continue
        t0 = time.perf_counter()
        P.compute_tmqi(hdr_bgr, out, op)
        rec("tmqi", op, time.perf_counter() - t0)
        os.remove(out)
    rec("pick", pick, np.nan)
    return rows


def summarise(long, ops):
    per_img = []
    for img, d in long.groupby("image"):
        s = d.set_index(["stage", "operator"]).seconds
        pick = d[d.stage == "pick"].operator.iloc[0]
        tmo = s.xs("tmo", level="stage")
        tmqi = s.xs("tmqi", level="stage")
        per_img.append(dict(
            image=img,
            features=s[("features", "")], predict=s[("predict", "")], picked=pick,
            picked_tmo=tmo[pick],
            selector_total=s[("features", "")] + s[("predict", "")] + tmo[pick],
            exhaustive_tmo=tmo[ops].sum(), exhaustive_tmqi=tmqi[ops].sum(),
            exhaustive_total=s[("load_hdr", "")] + tmo[ops].sum() + tmqi[ops].sum()))
    per_img = pd.DataFrame(per_img)
    per_img["speedup"] = per_img.exhaustive_total / per_img.selector_total
    return per_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="images to time")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    bundle = select_tmo.load_model("Q")
    ops = bundle["ops"]
    registry = {n: (k, s) for n, k, s in P.TMO_REGISTRY}
    images = pd.read_csv(os.path.join(ROOT, "results", "q_matrix.csv")).image
    sample = images.sample(args.n + 1, random_state=args.seed).tolist()
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="tmo_timing_")
    try:
        print(f"warm-up on {sample[0]}")
        time_image(sample[0], bundle, ops, registry, tmp)
        rows = []
        for i, img in enumerate(sample[1:], 1):
            t0 = time.perf_counter()
            rows += time_image(img, bundle, ops, registry, tmp)
            print(f"[{i}/{args.n}] {img}: {time.perf_counter() - t0:.1f}s", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(OUT_DIR, "timing_long.csv"), index=False)
    per_img = summarise(long, ops)
    per_img.round(4).to_csv(os.path.join(OUT_DIR, "timing_per_image.csv"), index=False)

    per_op = long[long.stage.isin(["tmo", "tmqi"])].pivot_table(
        index="operator", columns="stage", values="seconds", aggfunc=["mean", "median"])
    per_op.columns = [f"{s}_{a}" for a, s in per_op.columns]
    per_op.round(4).to_csv(os.path.join(OUT_DIR, "timing_per_operator.csv"))
    cols = ["features", "predict", "picked_tmo", "selector_total", "exhaustive_tmo", "exhaustive_tmqi",
            "exhaustive_total", "speedup"]
    summary = per_img[cols].agg(["mean", "median", "min", "max"]).T
    summary.round(4).to_csv(os.path.join(OUT_DIR, "timing_summary.csv"))
    pd.set_option("display.width", 200)
    print(f"\n{args.n} images, {len(ops)} candidate operators, seconds per image\n")
    print(summary.round(3).to_string())
    print("\nper operator\n" + per_op.round(3).to_string())


if __name__ == "__main__":
    main()
