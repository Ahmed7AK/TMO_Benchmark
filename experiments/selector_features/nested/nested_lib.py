"""
Nested cross-validation of the whole selector-selection pipeline (feature-group choice, then model choice).

Outer: the study's own 5-fold scene-grouped StratifiedGroupKFold (seed 1234, common.folds). Inside each outer
training split, using only its images:
  1. greedy backward elimination of feature groups, exactly as greedy.py (struct_lgbm_nobag, inner 3-fold x
     seeds 1234 / 7); then all 14 groups vs the greedy subset with struct_lgbm on the inner 5-fold CV of step 2
  2. the model among the 7 confirmed finalists, highest inner mean Q (inner 5-fold x seeds 1234 / 7)
Every finalist is also fit on the outer training split (seed 1234) and scored on the outer test fold, so any
selection rule can be scored afterwards. The study's code (common.py, models.py) is imported unchanged.

Fold fits are cached in nested/cache/o<outer fold>/ as .npy and run in 8 worker processes.
"""
import hashlib
import os
import sys
import warnings
from functools import lru_cache

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
sys.path.insert(0, STUDY)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import common as C            # noqa: E402
import models as M            # noqa: E402
from features import GROUPS   # noqa: E402

# one LightGBM thread per fit, so 8 worker processes use 8 cores. The outputs equal the study's 2-thread fits
# (run_fold.py checks every all-groups outer fit against the study's oof/ cache).
_d = M.lgb_fit_predict.__defaults__
M.lgb_fit_predict.__defaults__ = (_d[0], _d[1], 1)

METRIC = "Q"
OUTER_SEED, OUTER_K = 1234, 5
SCREEN_MODEL, SCREEN_K = "struct_lgbm_nobag", 3          # greedy.py / screen.py protocol
INNER_SEEDS = [1234, 7]                                  # screen.py seeds
SELECT_K = 5                                             # inner CV for the feature-set confirmation and the model
FINALISTS = ["struct_lgbm", "struct_lgbm_nobag", "struct_ridge", "struct_et", "lgbm_reg", "lgbm_clf", "rf_clf"]
TWO_THREAD = {"struct_et", "rf_clf", "lgbm_clf"}         # n_jobs=2 is fixed inside models.py: run 4 at a time
COST = {"struct_lgbm": 10, "struct_lgbm_nobag": 2, "lgbm_reg": 1}
ALL = "+".join(GROUPS)
WORKERS = 8
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "folds")
TAB = os.path.join(HERE, "tables")
for d in (CACHE, OUT, TAB):
    os.makedirs(d, exist_ok=True)


@lru_cache(maxsize=64)
def data(fset):
    D = C.load(METRIC, fset)
    ref = data(ALL) if fset != ALL else D
    assert (D.df.image.to_numpy() == ref.df.image.to_numpy()).all() and D.ops == ref.ops
    return D


def fkey(fset):
    """cache name of a feature set: group count, hash of the group list, hash of the feature values"""
    n = len(fset.split("+"))
    return "ALL" if fset == ALL else f"{n}g-{hashlib.md5(fset.encode()).hexdigest()[:8]}", C.content_hash(data(fset))


def outer_folds():
    return C.folds(data(ALL), OUTER_K, OUTER_SEED)


def inner_folds(rows, k, seed):
    """common.folds restricted to `rows`: StratifiedGroupKFold on (winner, scene) of those images only"""
    D = data(ALL)
    f = np.empty(len(rows), int)
    for i, (_, te) in enumerate(StratifiedGroupKFold(k, shuffle=True, random_state=seed)
                                .split(rows, D.y[rows], D.groups[rows])):
        f[te] = i
    return f


def path(o, fset, model, tag, seed, j=None):
    name, h = fkey(fset)
    return os.path.join(CACHE, f"o{o}", f"{name}__{h}__{model}__{tag}__s{seed}" + ("" if j is None else f"__f{j}")
                        + ".npy")


def _work(p, fset, model, tr, te, seed):
    if os.path.exists(p):
        return
    fn, kw = M.MODELS[model]
    S = fn(data(fset), tr, te, seed=seed, **kw)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p[:-4] + f".tmp{os.getpid()}.npy"
    np.save(tmp, S)
    os.replace(tmp, p)


def run_tasks(tasks):
    """tasks: (cache path, fset, model, train rows, test rows, seed). Missing ones are fit in parallel."""
    from joblib import Parallel, delayed
    todo = sorted({t[0]: t for t in tasks if not os.path.exists(t[0])}.values(),
                  key=lambda t: -COST.get(t[2], 0.5) * (1.5 if "__outer__" in t[0] else 1))
    one = [t for t in todo if t[2] not in TWO_THREAD]
    two = [t for t in todo if t[2] in TWO_THREAD]
    if one:
        Parallel(n_jobs=WORKERS, backend="loky")(delayed(_work)(*t) for t in one)
    if two:
        Parallel(n_jobs=WORKERS // 2, backend="loky")(delayed(_work)(*t) for t in two)
    return [np.load(t[0]) for t in tasks]


class InnerCV:
    """inner k-fold CV of (fset, model) on the rows of outer fold o's training split, for several seeds"""

    def __init__(self, o, rows, k, seeds):
        self.o, self.rows, self.k, self.seeds = o, rows, k, seeds
        self.f = {s: inner_folds(rows, k, s) for s in seeds}

    def tasks(self, fset, model):
        r, f = self.rows, self.f
        return [(path(self.o, fset, model, f"in{self.k}", s, j), fset, model, r[f[s] != j], r[f[s] == j], s)
                for s in self.seeds for j in range(self.k)]

    def scores(self, fset, model):
        """{seed: OOF scores over self.rows}; fits are taken from the cache (run_tasks first)"""
        out = {}
        for s in self.seeds:
            S = np.full((len(self.rows), data(ALL).k), np.nan)
            for j in range(self.k):
                S[self.f[s] == j] = np.load(path(self.o, fset, model, f"in{self.k}", s, j))
            assert not np.isnan(S).any()
            out[s] = S
        return out


def evaluate_rows(rows, S):
    """common.evaluate on a subset of images: pick (fallback to catmo), mean Q, exact, near-optimal, top-3"""
    D = data(ALL)
    r = np.arange(len(rows))
    b = S.argmax(1)
    ch = np.where(S[r, b] - S[:, D.base_j] > 0, b, D.base_j)
    Q, y = D.Q[rows], D.y[rows]
    q, best = Q[r, ch], Q.max(1)
    top3 = (np.argsort(-S, 1)[:, :3] == y[:, None]).any(1)
    return dict(mean_Q=float(q.mean()), exact=float((ch == y).mean()), near_005=float(((best - q) <= C.TIE_EPS).mean()),
                top3=float(top3.mean())), q, ch


def paired_rows(rows, qa, qb):
    """common.paired (scene-cluster bootstrap CI, scene-level Wilcoxon) on a subset of images"""
    D = data(ALL)
    a, b, m = np.zeros(D.n), np.zeros(D.n), np.zeros(D.n, bool)
    a[rows], b[rows], m[rows] = qa, qb, True
    return C.paired(D, a, b, mask=m)
