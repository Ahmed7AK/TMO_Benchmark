"""
Paired comparison of (feature set, model) configurations on the full protocol, using cached OOF scores.
Per-image metric values are averaged over the shared seeds, then compared with a reference configuration
(scene-cluster bootstrap 95% CI, scene-level Wilcoxon), overall and per source.

    ../../.venv/bin/python summarise.py --ref pouli10:struct_lgbm --cfg ALL:struct_lgbm hdr22:rf_clf \
        [--seeds 1234 2024 7 42 99] [--metric Q] [--out confirm_Q]
"""
import argparse
import os

import numpy as np
import pandas as pd

import common as C
from models import MODELS
from screen import ALL


def per_image(metric, cfg, seeds, folds):
    fset, model = cfg.split(":")
    fset = ALL if fset == "ALL" else fset
    D = C.load(metric, fset)
    fn, kw = MODELS[model]
    qs, rs = [], []
    for s in seeds:
        S = C.cached_cv(model, fn, D, folds, s, **kw)
        r, q = C.evaluate(D, S)
        qs.append(q)
        rs.append(r)
    return D, np.mean(qs, 0), pd.DataFrame(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cfg", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=C.SEEDS)
    ap.add_argument("--metric", default="Q")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="confirm")
    args = ap.parse_args()
    D, qref, rref = per_image(args.metric, args.ref, args.seeds, args.folds)
    rows = [dict(cfg=args.ref, n_features=len(D.cols), **rref.mean().to_dict(),
                 mean_Q_sd=rref.mean_Q.std(), delta=0.0)]
    for cfg in args.cfg:
        Dc, q, r = per_image(args.metric, cfg, args.seeds, args.folds)
        row = dict(cfg=cfg, n_features=len(Dc.cols), **r.mean().to_dict(), mean_Q_sd=r.mean_Q.std())
        for lab, m in (("", None), ("real_", D.src == "HDR-Real"), ("synth_", D.src == "HDR-Synth")):
            p = C.paired(D, q, qref, mask=m)
            row.update({f"{lab}{k}": v for k, v in p.items()})
        rows.append(row)
        print(f"{cfg:40s} Q {row['mean_Q']:.5f}+-{row['mean_Q_sd']:.5f} exact {row['exact']:.3f} "
              f"d {row['delta']:+.5f} [{row['ci_lo']:+.5f}, {row['ci_hi']:+.5f}] p={row['p']:.1e}", flush=True)
    t = pd.DataFrame(rows)
    t.round(6).to_csv(os.path.join(C.TAB, f"{args.out}.csv"), index=False)
    print(f"ref {args.ref}: Q {rows[0]['mean_Q']:.5f} exact {rows[0]['exact']:.3f}  ({len(args.seeds)} seeds)")


if __name__ == "__main__":
    main()
