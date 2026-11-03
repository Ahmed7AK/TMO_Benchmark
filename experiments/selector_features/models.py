"""
Selector models. Each is  model(D, tr, te, seed, **kw) -> scores[len(te), k]  (higher = better operator),
trained on the training rows only. Three families:
  * structured: predict the metric's per-operator components (training targets from results/q_matrix_*)
    and compose the score with the metric's own formula (struct_*)
  * direct regression: predict each operator's score, pick the argmax (*_reg)
  * classification / ranking: predict the winning operator (*_clf) or a per-image ranking (lgbm_rank)
"""
import os
import sys

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C

sys.path.insert(0, C.ROOT)
from make_qmatrix import naturalness, naturalness2   # noqa: E402

LGB = dict(objective="regression", num_leaves=31, learning_rate=0.03, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, min_child_samples=20, verbose=-1)
STRUCT_LGB = dict(LGB, num_leaves=15, min_child_samples=10)
TMQI_A, TMQI_ALPHA, TMQI_BETA = 0.8012, 0.3046, 0.7088


def _splits(groups, seed, bag):
    """early-stopping splits inside the training rows: one 80/20 group split, or `bag` group folds"""
    if bag <= 1:
        return [next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(groups, groups, groups))]
    ug = np.unique(groups)
    perm = dict(zip(ug, np.random.default_rng(seed).permutation(len(ug))))
    return list(GroupKFold(bag).split(groups, groups, np.array([perm[g] for g in groups])))


def lgb_fit_predict(X, y, Xte, groups, seed, params=LGB, bag=1, n_jobs=2):
    pred = np.zeros(len(Xte))
    sp = _splits(groups, seed, bag)
    for itr, iva in sp:
        m = lgb.LGBMRegressor(n_estimators=2000, random_state=seed, n_jobs=n_jobs, **params)
        m.fit(X[itr], y[itr], eval_set=[(X[iva], y[iva])], callbacks=[lgb.early_stopping(50, verbose=False)])
        pred += m.predict(Xte)
    return pred / len(sp)


def _dense(D, tr, te, scale=True):
    steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
    p = make_pipeline(*steps).fit(D.X[tr])
    return p.transform(D.X[tr]), p.transform(D.X[te])


def compose(D, P, te):
    if D.metric == "Q2":
        n2 = naturalness2(P["u2"], P["sd2"], D.matrix("est_mu2")[te], D.matrix("est_sd2")[te])
        return 0.5 * P["S2"] + 0.5 * n2
    n = np.clip(naturalness(P["u"], P["sigma"]), 0, 1)
    q = TMQI_A * np.clip(P["S"], 1e-6, 1) ** TMQI_ALPHA + (1 - TMQI_A) * n ** TMQI_BETA
    return q if D.metric == "Q" else 0.5 * (P["FSITM"] + q)


# ------------------------------------------------------------------------------ structured --
def struct(D, tr, te, seed, reg="lgbm", bag=5, params=STRUCT_LGB):
    """per-operator components -> metric formula. reg: lgbm (one bagged model per operator and component),
    or a multi-output sklearn regressor per component: et, rf, ridge, mlp"""
    P = {}
    g = D.groups[tr]
    if reg != "lgbm":
        Xtr, Xte = _dense(D, tr, te, scale=reg in ("ridge", "mlp"))
    for c in C.PARTS[D.metric]:
        T = D.matrix(c)
        if reg == "lgbm":
            P[c] = np.column_stack([lgb_fit_predict(D.X[tr], T[tr, j], D.X[te], g, seed, params, bag)
                                    for j in range(D.k)])
            continue
        mu, sd = T[tr].mean(0), T[tr].std(0) + 1e-12
        if reg == "et":
            m = ExtraTreesRegressor(500, max_features=0.5, min_samples_leaf=2, n_jobs=2, random_state=seed)
        elif reg == "rf":
            m = RandomForestRegressor(500, max_features=0.33, min_samples_leaf=2, n_jobs=2, random_state=seed)
        elif reg == "ridge":
            m = Ridge(alpha=10.0)
        elif reg == "mlp":
            m = MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3, max_iter=2000,
                             early_stopping=True, random_state=seed)
        m.fit(Xtr, (T[tr] - mu) / sd)
        P[c] = m.predict(Xte) * sd + mu
    return compose(D, P, te)


# ------------------------------------------------------------------------ direct regression --
def lgbm_reg(D, tr, te, seed, bag=1):
    """one LightGBM regressor per operator on the metric itself (as compare_selectors.py)"""
    g = D.groups[tr]
    return np.column_stack([lgb_fit_predict(D.X[tr], D.Q[tr, j], D.X[te], g, seed, LGB, bag) for j in range(D.k)])


def sk_reg(D, tr, te, seed, kind="et"):
    """multi-output sklearn regressor on the metric itself (all operators at once)"""
    Xtr, Xte = _dense(D, tr, te, scale=kind in ("ridge", "mlp"))
    mu, sd = D.Q[tr].mean(0), D.Q[tr].std(0) + 1e-12
    m = {"et": lambda: ExtraTreesRegressor(500, max_features=0.5, min_samples_leaf=2, n_jobs=2, random_state=seed),
         "rf": lambda: RandomForestRegressor(500, max_features=0.33, min_samples_leaf=2, n_jobs=2, random_state=seed),
         "ridge": lambda: Ridge(alpha=10.0),
         "mlp": lambda: MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=2000, early_stopping=True,
                                     random_state=seed)}[kind]()
    m.fit(Xtr, (D.Q[tr] - mu) / sd)
    return m.predict(Xte) * sd + mu


# ------------------------------------------------------------------- classification / ranking --
def _proba(m, X, k):
    P = np.zeros((len(X), k))
    P[:, m.classes_] = m.predict_proba(X)
    return P


def clf(D, tr, te, seed, kind="rf"):
    """predict the winning operator; scores = class probabilities"""
    if kind == "lgbm":
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                               random_state=seed, n_jobs=2)
        return _proba(m.fit(D.X[tr], D.y[tr]), D.X[te], D.k)
    Xtr, Xte = _dense(D, tr, te, scale=kind in ("logreg", "knn"))
    m = {"rf": lambda: RandomForestClassifier(500, max_features="sqrt", n_jobs=2, random_state=seed),
         "et": lambda: ExtraTreesClassifier(500, max_features="sqrt", n_jobs=2, random_state=seed),
         "logreg": lambda: LogisticRegression(C=1.0, max_iter=5000),
         "knn": lambda: KNeighborsClassifier(15, weights="distance")}[kind]()
    return _proba(m.fit(Xtr, D.y[tr]), Xte, D.k)


def lgbm_rank(D, tr, te, seed):
    """LightGBM lambdarank over (image, operator) pairs; relevance from regret (0 / <=0.001 / <=0.005)"""
    k = D.k

    def long(idx):
        X = np.repeat(D.X[idx], k, axis=0)
        return np.column_stack([X, np.tile(np.arange(k), len(idx))])

    reg = D.Q[tr].max(1, keepdims=True) - D.Q[tr]
    rel = np.select([reg == 0, reg <= 0.001, reg <= 0.005], [3, 2, 1], 0).ravel()
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=600, learning_rate=0.03, num_leaves=15,
                       min_child_samples=10, feature_fraction=0.8, verbose=-1, random_state=seed, n_jobs=2)
    m.fit(long(tr), rel, group=[k] * len(tr), categorical_feature=[D.X.shape[1]])
    return m.predict(long(te)).reshape(len(te), k)


MODELS = {
    "struct_lgbm": (struct, {}),                                   # bagged x5, num_leaves 15
    "struct_lgbm_nobag": (struct, dict(bag=1)),
    "struct_et": (struct, dict(reg="et")),
    "struct_rf": (struct, dict(reg="rf")),
    "struct_ridge": (struct, dict(reg="ridge")),
    "struct_mlp": (struct, dict(reg="mlp")),
    "lgbm_reg": (lgbm_reg, {}),
    "et_reg": (sk_reg, dict(kind="et")),
    "rf_reg": (sk_reg, dict(kind="rf")),
    "ridge_reg": (sk_reg, dict(kind="ridge")),
    "mlp_reg": (sk_reg, dict(kind="mlp")),
    "rf_clf": (clf, dict(kind="rf")),
    "et_clf": (clf, dict(kind="et")),
    "lgbm_clf": (clf, dict(kind="lgbm")),
    "logreg_clf": (clf, dict(kind="logreg")),
    "knn_clf": (clf, dict(kind="knn")),
    "lgbm_rank": (lgbm_rank, {}),
}
