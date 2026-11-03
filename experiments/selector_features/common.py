"""
Shared protocol for the feature / model study (same rules as compare_selectors.py):
  * candidates = operators that win >= 1 image under the metric, plus the best fixed operator (baseline)
  * StratifiedGroupKFold(n, shuffle=True, random_state=seed) on (winner, scene); out-of-fold only
  * pick = argmax of a model's scores, falling back to the baseline unless the top score beats it
  * scores: mean true metric of the pick, gap recovered, exact accuracy, near-optimal (<= 0.005), top-3
Feature sets are '+'-joined group names from features/ (see features.py) plus 'hdr22', the project's
original 22 features (results/hdr_features.csv without the 4 image-size columns).
"""
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results")
FEAT = os.path.join(HERE, "features")
OOF = os.path.join(HERE, "oof")
TAB = os.path.join(HERE, "tables")
for d in (OOF, TAB):
    os.makedirs(d, exist_ok=True)
SIZE_COLS = {"width", "height", "aspect_ratio", "megapixels"}
TIE_EPS = 0.005
SEEDS = [1234, 2024, 7, 42, 99]

# components each metric is composed from (q_matrix_<component>.csv), for the structured models
PARTS = {"Q": ["S", "u", "sigma"], "FSITM_TMQI": ["S", "u", "sigma", "FSITM"], "Q2": ["S2", "u2", "sd2"]}


@dataclass
class Data:
    metric: str
    fset: str
    df: pd.DataFrame
    X: np.ndarray
    cols: list
    ops: list
    Q: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    src: np.ndarray
    base_j: int
    extra: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.y)

    @property
    def k(self):
        return len(self.ops)

    def matrix(self, name):
        if name not in self.extra:
            m = pd.read_csv(os.path.join(RES, f"q_matrix_{name}.csv")).set_index("image")
            self.extra[name] = m.loc[self.df.image, self.ops].to_numpy(float)
        return self.extra[name]


ABS_LEVEL = ["p10_logl_mean", "p10_lab_L_mean", "r02_log_avg", "kk08_mu"]   # absolute radiance level


def feature_frame(fset):
    """'+'-joined groups; 'ALL' = every group in features.py; 'scalefree' drops the absolute-level columns;
    'drop=a,b' drops named columns"""
    from features import GROUPS
    feat, drop = None, []
    toks = []
    for g in fset.split("+"):
        toks += list(GROUPS) if g == "ALL" else [g]
    for g in toks:
        if g == "scalefree":
            drop += ABS_LEVEL
            continue
        if g.startswith("drop="):
            drop += g[5:].split(",")
            continue
        if g == "hdr22":
            f = pd.read_csv(os.path.join(RES, "hdr_features.csv")).drop(columns=list(SIZE_COLS))
        else:
            f = pd.read_csv(os.path.join(FEAT, f"{g}.csv"))
        feat = f if feat is None else feat.merge(f, on="image")
    return feat.drop(columns=[c for c in drop if c in feat.columns])


def load(metric="Q", fset="hdr22"):
    qm = pd.read_csv(os.path.join(RES, "q_matrix.csv" if metric == "Q" else f"q_matrix_{metric}.csv"))
    feat = feature_frame(fset)
    sc = pd.read_csv(os.path.join(RES, "scenes.csv"))
    df = qm.merge(feat, on="image").merge(sc, on="image")
    assert len(df) == len(qm), "features missing for some images"
    cols = [c for c in feat.columns if c != "image"]
    all_ops = [c for c in qm.columns if c != "image"]
    Qall = df[all_ops].to_numpy(float)
    base = all_ops[int(Qall.mean(0).argmax())]
    wins = np.bincount(Qall.argmax(1), minlength=len(all_ops))
    ops = [o for i, o in enumerate(all_ops) if wins[i] > 0 or o == base]
    Q = df[ops].to_numpy(float)
    man = pd.read_csv(os.path.join(RES, "manifest.csv")).set_index("image").dataset
    src = np.where(df.image.map(man).eq("HDR-Real"), "HDR-Real", "HDR-Synth")
    return Data(metric, fset, df, df[cols].to_numpy(float), cols, ops, Q, Q.argmax(1),
                df.scene.to_numpy(), src, ops.index(base))


def folds(D, n_splits=5, seed=1234):
    f = np.empty(D.n, int)
    for i, (_, te) in enumerate(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
                                .split(D.X, D.y, D.groups)):
        f[te] = i
    return f


def run_cv(model, D, n_splits=5, seed=1234, jobs=5, **kw):
    fo = folds(D, n_splits, seed)

    def one(i):
        tr, te = np.where(fo != i)[0], np.where(fo == i)[0]
        return te, model(D, tr, te, seed=seed, **kw)

    from joblib import Parallel, delayed
    outs = Parallel(n_jobs=min(jobs, n_splits), backend="threading")(delayed(one)(i) for i in range(n_splits))
    S = np.full((D.n, D.k), np.nan)
    for te, s in outs:
        S[te] = s
    assert not np.isnan(S).any()
    return S


def content_hash(D):
    """short hash of the feature values, columns and candidate set, so the cache can't serve results
    computed from an older version of a feature group"""
    import hashlib
    h = hashlib.md5(np.ascontiguousarray(D.X).tobytes())
    h.update(("|".join(D.cols) + "#" + "|".join(D.ops)).encode())
    return h.hexdigest()[:10]


def oof_path(D, model, n_splits, seed):
    name = D.fset if len(D.fset) <= 80 else f"{len(D.fset.split('+'))}groups"
    return os.path.join(OOF, f"{D.metric}__{name}__{content_hash(D)}__{model}__{n_splits}f__s{seed}.npy")


def cached_cv(name, fn, D, n_splits, seed, refresh=False, **kw):
    p = oof_path(D, name, n_splits, seed)
    if os.path.exists(p) and not refresh:
        return np.load(p)
    S = run_cv(fn, D, n_splits, seed, **kw)
    np.save(p, S)
    return S


def pick(D, S):
    b = S.argmax(1)
    return np.where(S[np.arange(D.n), b] - S[:, D.base_j] > 0, b, D.base_j)


def evaluate(D, S):
    ch = pick(D, S)
    r = np.arange(D.n)
    q, best, qb = D.Q[r, ch], D.Q.max(1), D.Q[:, D.base_j]
    top3 = (np.argsort(-S, 1)[:, :3] == D.y[:, None]).any(1)
    out = dict(mean_Q=q.mean(), gap_rec=(q.mean() - qb.mean()) / (best.mean() - qb.mean()),
               exact=(ch == D.y).mean(), near_005=((best - q) <= TIE_EPS).mean(), top3=top3.mean())
    for s in ("HDR-Real", "HDR-Synth"):
        m = D.src == s
        t = "real" if s == "HDR-Real" else "synth"
        out[f"{t}_Q"], out[f"{t}_exact"] = q[m].mean(), (ch[m] == D.y[m]).mean()
    return out, q


def paired(D, qa, qb, mask=None, n_boot=10000, seed=0):
    """a - b: mean, scene-cluster bootstrap 95% CI of the image-level mean, scene-level Wilcoxon p"""
    g = D.groups if mask is None else D.groups[mask]
    if mask is not None:
        qa, qb = qa[mask], qb[mask]
    d = pd.Series(qa - qb).groupby(g)
    sums, cnts = d.sum().to_numpy(), d.size().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(sums), (n_boot, len(sums)))
    boot = sums[idx].sum(1) / cnts[idx].sum(1)
    sa, sb = pd.Series(qa).groupby(g).mean(), pd.Series(qb).groupby(g).mean()
    p = float(wilcoxon(sa, sb).pvalue) if ((sa - sb) != 0).any() else 1.0
    return dict(delta=float((qa - qb).mean()), ci_lo=float(np.quantile(boot, .025)),
                ci_hi=float(np.quantile(boot, .975)), p=p)
