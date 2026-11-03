"""
Fast screening of feature groups (3-fold x 2 seeds, structured LightGBM without bagging), paired against a
reference feature set on the same folds. Used to optimise the compiled multi-paper set:

    ../../.venv/bin/python screen.py --ref pouli10 --sets pouli10+drago03 pouli10+fattal02 ...
    ../../.venv/bin/python screen.py --ref ALL --loo           # leave-one-group-out from all groups

Writes tables/screen_<name>.csv: mean dQ vs ref (and its worst seed), exact, near-optimal.
"""
import argparse
import os

import numpy as np
import pandas as pd

import common as C
from features import GROUPS
from models import MODELS

MODEL = "struct_lgbm_nobag"
SEEDS = [1234, 7]
ALL = "+".join(GROUPS)


def score(fset, metric="Q", model=MODEL, folds=3):
    D = C.load(metric, fset)
    fn, kw = MODELS[model]
    out = []
    for s in SEEDS:
        S = C.cached_cv(model, fn, D, folds, s, **kw)
        r, q = C.evaluate(D, S)
        out.append((r, q))
    return D, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--loo", action="store_true", help="also screen ref minus each group")
    ap.add_argument("--add", action="store_true", help="also screen ref plus each group not in it")
    ap.add_argument("--name", default="screen")
    ap.add_argument("--metric", default="Q")
    args = ap.parse_args()
    ref = ALL if args.ref == "ALL" else args.ref
    sets = list(args.sets)
    rg = ref.split("+")
    if args.loo:
        sets += ["+".join(g for g in rg if g != x) for x in rg]
    if args.add:
        sets += [ref + "+" + g for g in GROUPS if g not in rg]
    _, ref_out = score(ref, args.metric)
    rows = [dict(fset=ref, label="REF", dQ=0.0, dQ_worst=0.0,
                 mean_Q=np.mean([r["mean_Q"] for r, _ in ref_out]),
                 exact=np.mean([r["exact"] for r, _ in ref_out]),
                 near=np.mean([r["near_005"] for r, _ in ref_out]))]
    for fs in sets:
        D, out = score(fs, args.metric)
        d = [q.mean() - qr.mean() for (_, q), (_, qr) in zip(out, ref_out)]
        lab = ("-" + (set(rg) - set(fs.split("+"))).pop()) if set(fs.split("+")) < set(rg) else \
              ("+" + "+".join(sorted(set(fs.split("+")) - set(rg))) if set(rg) < set(fs.split("+")) else fs)
        rows.append(dict(fset=fs, label=lab, n_features=len(D.cols), dQ=np.mean(d), dQ_worst=min(d),
                         mean_Q=np.mean([r["mean_Q"] for r, _ in out]), exact=np.mean([r["exact"] for r, _ in out]),
                         near=np.mean([r["near_005"] for r, _ in out])))
        print(f"{lab:45s} n={len(D.cols):4d} dQ {np.mean(d):+.5f} (worst {min(d):+.5f}) "
              f"exact {rows[-1]['exact']:.3f}", flush=True)
    t = pd.DataFrame(rows).sort_values("dQ", ascending=False)
    t.round(6).to_csv(os.path.join(C.TAB, f"{args.name}.csv"), index=False)
    print(t[["label", "dQ", "dQ_worst", "mean_Q", "exact", "near"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
