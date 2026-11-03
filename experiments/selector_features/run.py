"""
Run feature-set x model combinations under the CV protocol and append the scores to tables/runs.csv.

    ../../.venv/bin/python run.py --fsets hulusic17 pouli10 --models struct_lgbm lgbm_reg rf_clf \
                                  [--metric Q] [--folds 5] [--seeds 1234 2024 7]

OOF scores are cached in oof/, so re-running a combination is free.
"""
import argparse
import os
import time

import pandas as pd

import common as C
from models import MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsets", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--metric", default="Q")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1234])
    ap.add_argument("--tag", default="", help="label stored with the rows (e.g. screen, confirm)")
    args = ap.parse_args()
    rows = []
    for fset in args.fsets:
        D = C.load(args.metric, fset)
        for name in args.models:
            fn, kw = MODELS[name]
            for seed in args.seeds:
                t0 = time.time()
                S = C.cached_cv(name, fn, D, args.folds, seed, **kw)
                r, _ = C.evaluate(D, S)
                rows.append(dict(metric=args.metric, fset=fset, n_features=len(D.cols), model=name,
                                 folds=args.folds, seed=seed, tag=args.tag, secs=round(time.time() - t0, 1), **r))
                print(f"{args.metric:3s} {fset:40s} {name:18s} s{seed:<5d} Q {r['mean_Q']:.5f} "
                      f"exact {r['exact']:.3f} near {r['near_005']:.3f} ({rows[-1]['secs']}s)", flush=True)
    out = os.path.join(C.TAB, "runs.csv")
    new = pd.DataFrame(rows)
    if os.path.exists(out):
        new = pd.concat([pd.read_csv(out), new]).drop_duplicates(
            ["metric", "fset", "model", "folds", "seed"], keep="last")
    new.round(6).to_csv(out, index=False)


if __name__ == "__main__":
    main()
