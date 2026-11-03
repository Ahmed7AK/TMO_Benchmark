"""
Deployable per-image TMO selector: the final model of compare_selectors.py, fit once on all benchmark
images and applied to new HDR files.

  train    fit structured LightGBM on all images of results/q_matrix*.csv and save it to
           models/selector_<metric>.joblib. Same features (selector_features.py), candidate set
           (operators that win at least once, plus the best fixed operator), targets, parameters and
           5-way scene-grouped bagging as one fold of compare_selectors.py, with every image as a
           training row.
  predict  for each HDR: compute the 151 features, predict every candidate's metric components, compose
           the metric with its formula and pick the best, staying on the best fixed operator unless
           another operator's predicted score beats it. --tonemap DIR also runs the picked operator
           (tmo_pipeline.produce_ldr) and writes DIR/<stem>__<op>.png.

    python select_tmo.py train   [--metric Q|FSITM_TMQI]
    python select_tmo.py predict [--metric Q] [--tonemap out/] path/to/a.hdr [b.exr ...]

TMQI-II (Q2) is not supported here: its naturalness term needs per-operator HDR-side estimates that
depend on each operator's output geometry (make_qmatrix.py), so it cannot be composed from the HDR alone
without that bookkeeping. Accuracy figures for this model come from compare_selectors.py (out-of-fold);
this script only produces the deployable fit.
"""

import argparse
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import selector_features as sf
from make_qmatrix import naturalness

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
PARTS = {"Q": ["S", "u", "sigma"], "FSITM_TMQI": ["S", "u", "sigma", "FSITM"]}
N_BAG = 5
SEED = 1234
LGB_PARAMS = dict(objective="regression", num_leaves=15, learning_rate=0.03, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, min_child_samples=10, verbose=-1)
TMQI_A, TMQI_ALPHA, TMQI_BETA = 0.8012, 0.3046, 0.7088


def model_path(metric):
    return os.path.join(ROOT, "models", f"selector_{metric}.joblib")


def compose(metric, P):
    """the metric from predicted components P (each n x k), as in compare_selectors.compose"""
    n = np.clip(naturalness(P["u"], P["sigma"]), 0, 1)
    q = TMQI_A * np.clip(P["S"], 1e-6, 1) ** TMQI_ALPHA + (1 - TMQI_A) * n ** TMQI_BETA
    return q if metric == "Q" else 0.5 * (P["FSITM"] + q)


def train(metric, n_jobs=8):
    qfile = "q_matrix.csv" if metric == "Q" else f"q_matrix_{metric}.csv"
    qm = pd.read_csv(os.path.join(RESULTS, qfile))
    feat = pd.read_csv(os.path.join(RESULTS, "features_selector.csv"))
    df = qm.merge(feat, on="image").merge(pd.read_csv(os.path.join(RESULTS, "scenes.csv")), on="image")
    feature_cols = [c for c in feat.columns if c != "image"]
    all_ops = [c for c in qm.columns if c != "image"]
    baseline = df[all_ops].mean().idxmax()
    wins = np.bincount(df[all_ops].to_numpy(float).argmax(axis=1), minlength=len(all_ops))
    ops = [o for o, w in zip(all_ops, wins) if w > 0 or o == baseline]
    X = df[feature_cols].to_numpy(float)
    g = df["scene"].to_numpy()
    ug = np.unique(g)
    perm = dict(zip(ug, np.random.default_rng(SEED).permutation(len(ug))))
    bags = list(GroupKFold(N_BAG).split(g, g, np.array([perm[s] for s in g])))

    def fit_one(c, j):
        T = pd.read_csv(os.path.join(RESULTS, f"q_matrix_{c}.csv")).set_index("image") \
            .loc[df["image"], ops[j]].to_numpy(float)
        models = []
        for itr, iva in bags:
            m = lgb.LGBMRegressor(n_estimators=2000, random_state=SEED, n_jobs=1, **LGB_PARAMS)
            m.fit(X[itr], T[itr], eval_set=[(X[iva], T[iva])], callbacks=[lgb.early_stopping(50, verbose=False)])
            models.append(m.booster_)
        return (c, j), models

    from joblib import Parallel, delayed
    t0 = time.time()
    fitted = dict(Parallel(n_jobs=n_jobs)(delayed(fit_one)(c, j) for c in PARTS[metric] for j in range(len(ops))))
    secs = time.time() - t0
    bundle = dict(metric=metric, ops=ops, baseline=baseline, feature_cols=feature_cols, models=fitted,
                  n_train=len(df), train_seconds=secs)
    os.makedirs(os.path.dirname(model_path(metric)), exist_ok=True)
    joblib.dump(bundle, model_path(metric))
    print(f"[{metric}] {len(df)} images x {len(feature_cols)} features, {len(ops)} candidates "
          f"(baseline {baseline}), {len(fitted) * N_BAG} LightGBM models in {secs:.0f}s -> {model_path(metric)}")
    return bundle


def load_model(metric="Q"):
    return joblib.load(model_path(metric))


def predict_scores(bundle, feats):
    """composed metric per candidate for one image's feature dict -> (k,) array"""
    x = np.array([[feats[c] for c in bundle["feature_cols"]]], float)
    k = len(bundle["ops"])
    P = {c: np.zeros((1, k)) for c in PARTS[bundle["metric"]]}
    for (c, j), models in bundle["models"].items():
        P[c][0, j] = np.mean([m.predict(x, num_threads=1)[0] for m in models])
    return compose(bundle["metric"], P)[0]


def choose(bundle, scores):
    """argmax, staying on the baseline unless another operator's score beats it"""
    b = bundle["ops"].index(bundle["baseline"])
    best = int(np.argmax(scores))
    return bundle["ops"][best if scores[best] > scores[b] else b]


def select(bundle, hdr_path):
    """(picked operator, {operator: predicted score}) for one HDR file"""
    scores = predict_scores(bundle, sf.features(*sf.load_path(hdr_path)))
    return choose(bundle, scores), dict(zip(bundle["ops"], scores))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "predict"])
    ap.add_argument("hdr", nargs="*", help="HDR files (predict)")
    ap.add_argument("--metric", default="Q", choices=list(PARTS))
    ap.add_argument("--tonemap", metavar="DIR", help="also run the picked operator and write its PNG here")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    if args.command == "train":
        train(args.metric, args.jobs)
        return
    assert args.hdr, "predict needs at least one HDR file"
    bundle = load_model(args.metric)
    for path in args.hdr:
        t0 = time.time()
        op, scores = select(bundle, path)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        print(f"{path}: {op}  ({time.time() - t0:.2f}s; predicted {args.metric}: "
              + ", ".join(f"{o} {s:.4f}" for o, s in top) + ")")
        if args.tonemap:
            import tmo_pipeline as P
            os.makedirs(args.tonemap, exist_ok=True)
            kind, spec = {n: (k, s) for n, k, s in P.TMO_REGISTRY}[op]
            out = os.path.join(args.tonemap, f"{os.path.splitext(os.path.basename(path))[0]}__{op}.png")
            P.produce_ldr(op, kind, spec, path, P.load_hdr(path), out)
            print(f"  -> {out}")


if __name__ == "__main__":
    main()
