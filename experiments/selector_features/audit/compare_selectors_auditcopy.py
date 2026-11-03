"""
Stage 3 -- per-image TMO selector: structured LightGBM on HDR-content features.

For each candidate operator, LightGBM regressors predict the components of the chosen metric from the
HDR's features (selector_features.py), and the metric is composed with its own formula:
  Q           TMQI:  S, u, sigma -> 0.8012 S^0.3046 + 0.1988 N(u, sigma)^0.7088, where N is TMQI's
              naturalness of the output's mean luminance u and mean 11x11 block std sigma
  FSITM_TMQI  S, u, sigma, FSITM -> (FSITM + Q) / 2
  Q2          TMQI-II: S2, u2, sd2 -> (S2 + N2) / 2, N2 from the output mean / std and the HDR-side
              estimates est_mu2 / est_sd2 (make_qmatrix.py)
The selector picks the operator with the highest composed score, staying on the best fixed operator
unless another operator's score beats it.

Model: per operator and component, LightGBM (num_leaves 15, min_child_samples 10, learning rate 0.03,
feature / bagging fraction 0.8, early stopping 50, up to 2000 trees) averaged over 5 group folds of the
training rows. Protocol: scene-grouped StratifiedGroupKFold(5, shuffle, seed), out-of-fold only;
training targets (q_matrix_<component>.csv, from make_qmatrix.py) are read for training rows only.
experiments/selector_features/REPORT.md has the study behind the features and the model choice.

The number that matters is the TRUE metric value of the picked operator, compared with always-<baseline>
(the best fixed operator under the metric, the floor) and the per-image oracle (the ceiling). Exact
accuracy is reported too, but a wrong pick that is a near-tie costs almost nothing.

    python compare_selectors.py [--metric Q|FSITM_TMQI|Q2] [--features all|<group>+<group>...]
                                [--baseline OP] [--drop-ops OP ...] [--seed 1234] [--tag _x]

--features picks groups of results/features_selector.csv by name (results/features_selector_groups.csv),
e.g. --features pouli10 for the single-paper base set; default all 14 groups.

Writes to results/selectors/<metric><_no_op...><tag>/:
  tmo_wins.csv / tmo_wins.png   per-operator wins, win rate, mean / median score
  selector_comparison.csv       always-baseline, random pick, the selector, oracle
  selector_by_dataset.csv       the same split by HDR-Real vs HDR-Synth
  selector_per_operator.csv     per-operator support / picks / precision / recall / F1
  selector_confusion.csv        true best vs picked operator
  selector_predictions.csv      per-image pick, its true score, and the composed score
  selector_tau_sweep.csv        abstention margin sweep
  feature_importance.csv        LightGBM gain share per feature (all component models)
"""

import argparse
import os

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from make_qmatrix import naturalness, naturalness2

ap = argparse.ArgumentParser()
ap.add_argument("--metric", default="Q", help="per-image score: Q (TMQI), FSITM_TMQI or Q2 (TMQI-II)")
ap.add_argument("--features", default="all", help="'all' or '+'-joined groups, e.g. pouli10")
ap.add_argument("--baseline", default=None, help="fixed operator to beat (default: best mean)")
ap.add_argument("--drop-ops", nargs="*", default=[],
                help="leave these operators out (ablation), e.g. --drop-ops att")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--tag", default="", help="suffix for the output folder")
args = ap.parse_args()

PARTS = {"Q": ["S", "u", "sigma"], "FSITM_TMQI": ["S", "u", "sigma", "FSITM"], "Q2": ["S2", "u2", "sd2"]}
assert args.metric in PARTS, f"--metric must be one of {list(PARTS)}"
RESULTS = "results"
AUDIT_OUT = "experiments/selector_features/audit/prod_out"
OUT_DIR = os.path.join(AUDIT_OUT,
                       args.metric + "".join(f"_no_{o}" for o in sorted(args.drop_ops)) + args.tag)
QMATRIX_CSV = os.path.join(RESULTS, "q_matrix.csv" if args.metric == "Q" else f"q_matrix_{args.metric}.csv")
ID_COL, GROUP_COL = "image", "scene"
TIE_EPS = 0.005
N_SPLITS = 5
N_BAG = 5
SEED = args.seed
TAU_GRID = [0.0, 0.002, 0.005, 0.01, 0.02]
LGB_PARAMS = dict(objective="regression", num_leaves=15, learning_rate=0.03, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, min_child_samples=10, verbose=-1)
TMQI_A, TMQI_ALPHA, TMQI_BETA = 0.8012, 0.3046, 0.7088


def out(name):
    return os.path.join(OUT_DIR, name)


# ---- load & join ----------------------------------------------------------
feat = pd.read_csv(os.path.join(RESULTS, "features_selector.csv"))
col_group = pd.read_csv(os.path.join(RESULTS, "features_selector_groups.csv")).set_index("column").group
if args.features != "all":
    keep = set(args.features.split("+"))
    unknown = keep - set(col_group)
    assert not unknown, f"unknown feature groups: {sorted(unknown)}"
    feat = feat[[ID_COL] + [c for c in feat.columns if c != ID_COL and col_group[c] in keep]]
qm = pd.read_csv(QMATRIX_CSV)
df = qm.merge(feat, on=ID_COL).merge(pd.read_csv(os.path.join(RESULTS, "scenes.csv")), on=ID_COL)
assert len(df) == len(qm), "some q_matrix images are missing features or scenes"

feature_cols = [c for c in feat.columns if c != ID_COL]
all_ops = [c for c in qm.columns if c != ID_COL and c not in args.drop_ops]
BASELINE = args.baseline or df[all_ops].mean().idxmax()
os.makedirs(OUT_DIR, exist_ok=True)
Q_all = df[all_ops].to_numpy(float)
X = df[feature_cols].to_numpy(float)
groups = df[GROUP_COL].to_numpy()
n_img = len(df)

# ---- per-operator wins ----------------------------------------------------
win_all = Q_all.argmax(axis=1)
best_q = Q_all.max(axis=1)
wins = pd.DataFrame({
    "operator": all_ops,
    "wins": np.bincount(win_all, minlength=len(all_ops)),
    "near_best": ((best_q[:, None] - Q_all) <= TIE_EPS).sum(axis=0),
    "mean_Q": Q_all.mean(axis=0),
    "median_Q": np.median(Q_all, axis=0),
})
wins["win_rate"] = wins.wins / n_img
wins = wins.sort_values(["wins", "mean_Q"], ascending=False)
wins[["operator", "wins", "win_rate", "near_best", "mean_Q", "median_Q"]].round(6) \
    .to_csv(out("tmo_wins.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(wins.operator, wins.wins, color="#3b6ea5")
for x, w in enumerate(wins.wins):
    ax.text(x, w, str(w), ha="center", va="bottom", fontsize=8)
ax.set_ylabel(f"images where operator has the highest {args.metric}")
ax.set_title(f"TMO wins per image ({n_img} images, {len(all_ops)} operators)", loc="left")
ax.spines[["top", "right"]].set_visible(False)
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
fig.savefig(out("tmo_wins.png"), dpi=150)
plt.close(fig)

# ---- candidate set: operators that win at least once (+ baseline) --------
ops = [c for c in all_ops if (wins.set_index("operator").wins[c] > 0) or c == BASELINE]
dropped = [c for c in all_ops if c not in ops]
Q = df[ops].to_numpy(float)
y = Q.argmax(axis=1)                     # true winner index into ops
base_j = ops.index(BASELINE)
k = len(ops)
print(f"{n_img} images x {len(feature_cols)} features x {k} candidate operators "
      f"(never-win, dropped: {dropped})")

folds = np.empty(n_img, int)
sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
for f, (_, te) in enumerate(sgkf.split(X, y, groups)):
    folds[te] = f


def operator_matrix(name):
    m = pd.read_csv(os.path.join(RESULTS, f"q_matrix_{name}.csv")).set_index(ID_COL)
    return m.loc[df[ID_COL], ops].to_numpy(float)


targets = {c: operator_matrix(c) for c in PARTS[args.metric]}
hdr_est = {c: operator_matrix(c) for c in ("est_mu2", "est_sd2")} if args.metric == "Q2" else {}


def compose(P, te):
    """the metric from predicted components P (each len(te) x k)"""
    if args.metric == "Q2":
        n2 = naturalness2(P["u2"], P["sd2"], hdr_est["est_mu2"][te], hdr_est["est_sd2"][te])
        return 0.5 * P["S2"] + 0.5 * n2
    n = np.clip(naturalness(P["u"], P["sigma"]), 0, 1)
    q = TMQI_A * np.clip(P["S"], 1e-6, 1) ** TMQI_ALPHA + (1 - TMQI_A) * n ** TMQI_BETA
    return q if args.metric == "Q" else 0.5 * (P["FSITM"] + q)


def fold_scores(tr, te):
    """composed scores for the test rows and summed normalised gain importance"""
    g = groups[tr]
    ug = np.unique(g)
    perm = dict(zip(ug, np.random.default_rng(SEED).permutation(len(ug))))
    bags = list(GroupKFold(N_BAG).split(g, g, np.array([perm[s] for s in g])))
    imp = np.zeros(len(feature_cols))
    P = {c: np.zeros((len(te), k)) for c in targets}
    for j in range(k):
        for c, T in targets.items():
            pred = np.zeros(len(te))
            for itr, iva in bags:
                m = lgb.LGBMRegressor(n_estimators=2000, importance_type="gain", random_state=SEED,
                                      n_jobs=2, **LGB_PARAMS)
                m.fit(X[tr][itr], T[tr, j][itr], eval_set=[(X[tr][iva], T[tr, j][iva])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
                pred += m.predict(X[te])
                imp += m.feature_importances_ / max(m.feature_importances_.sum(), 1e-12)
            P[c][:, j] = pred / len(bags)
    return compose(P, te), imp


outs = Parallel(n_jobs=2, backend="threading")(
    delayed(fold_scores)(np.where(folds != f)[0], np.where(folds == f)[0]) for f in range(N_SPLITS))
S = np.full_like(Q, np.nan)
imp = np.zeros(len(feature_cols))
for f, (s, im) in enumerate(outs):
    S[folds == f] = s
    imp += im
assert not np.isnan(S).any()
imp /= imp.sum()
print("structured LightGBM selector done")

# ---- scoring --------------------------------------------------------------
q_base = Q[:, base_j]
q_oracle = Q.max(axis=1)
gap = q_oracle.mean() - q_base.mean()
rows_idx = np.arange(n_img)


def pick(scores, tau=0.0):
    """argmax of scores, but stay on the baseline unless the top score beats it by > tau"""
    best = scores.argmax(axis=1)
    margin = scores[rows_idx, best] - scores[:, base_j]
    return np.where(margin > tau, best, base_j)


def scene_wilcoxon(a, b):
    s = pd.DataFrame({"g": groups, "a": a, "b": b}).groupby("g").mean()
    d = s.a - s.b
    return float(wilcoxon(s.a, s.b).pvalue) if (d != 0).any() else 1.0


def metrics(name, chosen, scores=None):
    q = Q[rows_idx, chosen]
    regret = q_oracle - q
    r = dict(method=name, mean_Q=q.mean(), gap_recovered=(q.mean() - q_base.mean()) / gap,
             mean_regret=regret.mean(), median_regret=np.median(regret),
             exact_accuracy=(chosen == y).mean(), near_optimal_rate=(regret <= TIE_EPS).mean(),
             macro_f1=f1_score(y, chosen, labels=range(k), average="macro", zero_division=0),
             balanced_accuracy=balanced_accuracy_score(y, chosen), override_rate=(chosen != base_j).mean(),
             n_operators_picked=len(np.unique(chosen)),
             wilcoxon_p_vs_baseline=scene_wilcoxon(q, q_base) if (chosen != base_j).any() else np.nan)
    if scores is not None:
        r["top3_accuracy"] = (np.argsort(-scores, axis=1)[:, :3] == y[:, None]).any(axis=1).mean()
    return r, q


NAME = "structured LightGBM selector"
chosen = pick(S)
res_base, _ = metrics(f"always-{BASELINE}", np.full(n_img, base_j))
res_sel, q_sel = metrics(NAME, chosen, S)
res_oracle, _ = metrics("oracle", y)
res_rand = dict(method="random pick (expected)", mean_Q=Q.mean(axis=1).mean(),
                gap_recovered=(Q.mean(axis=1).mean() - q_base.mean()) / gap,
                mean_regret=(q_oracle - Q.mean(axis=1)).mean(), exact_accuracy=1 / k)
comp = pd.DataFrame([res_base, res_rand, res_sel, res_oracle])
cols = ["method", "mean_Q", "gap_recovered", "mean_regret", "median_regret", "exact_accuracy",
        "near_optimal_rate", "top3_accuracy", "macro_f1", "balanced_accuracy", "override_rate",
        "n_operators_picked", "wilcoxon_p_vs_baseline"]
comp[cols].round(6).to_csv(out("selector_comparison.csv"), index=False)

sweep = []
for tau in TAU_GRID:
    r, _ = metrics(NAME, pick(S, tau))
    sweep.append(dict(tau=tau, mean_Q=r["mean_Q"], gap_recovered=r["gap_recovered"],
                      override_rate=r["override_rate"], exact_accuracy=r["exact_accuracy"]))
pd.DataFrame(sweep).round(6).to_csv(out("selector_tau_sweep.csv"), index=False)

p, r, f1, _ = precision_recall_fscore_support(y, chosen, labels=range(k), zero_division=0)
pd.DataFrame({"operator": ops, "true_wins": np.bincount(y, minlength=k), "picks": np.bincount(chosen, minlength=k),
              "precision": p, "recall": r, "f1": f1}).round(4).to_csv(out("selector_per_operator.csv"), index=False)
pd.crosstab(pd.Categorical(np.array(ops)[y], ops), pd.Categorical(np.array(ops)[chosen], ops),
            rownames=["true_best"], colnames=["picked"], dropna=False).to_csv(out("selector_confusion.csv"))

pd.DataFrame({"image": df[ID_COL], "fold": folds, "true_best": np.array(ops)[y], "Q_oracle": q_oracle,
              "Q_baseline": q_base, "pick": np.array(ops)[chosen], "Q_pick": q_sel,
              "pred_score": S[rows_idx, chosen]}).round(6).to_csv(out("selector_predictions.csv"), index=False)

# Operators differ a lot between HDR-Real and HDR-Synth, so report each source on its own.
src = pd.read_csv(os.path.join(RESULTS, "manifest.csv")).set_index("image").dataset
src = np.where(df[ID_COL].map(src).eq("HDR-Real"), "HDR-Real", "HDR-Synth")
by_ds = []
for s in ("HDR-Real", "HDR-Synth", "all"):
    mk = np.ones(n_img, bool) if s == "all" else (src == s)
    g = q_oracle[mk].mean() - q_base[mk].mean()
    by_ds.append(dict(source=s, n_images=int(mk.sum()), baseline_Q=q_base[mk].mean(), oracle_Q=q_oracle[mk].mean(),
                      selector_Q=q_sel[mk].mean(), gap_recovered=(q_sel[mk].mean() - q_base[mk].mean()) / g,
                      exact_accuracy=(chosen[mk] == y[mk]).mean(), baseline_win_rate=(y[mk] == base_j).mean()))
pd.DataFrame(by_ds).round(6).to_csv(out("selector_by_dataset.csv"), index=False)

pd.DataFrame({"feature": feature_cols, "group": [col_group[c] for c in feature_cols], "gain_share": imp}) \
    .sort_values("gain_share", ascending=False).round(6).to_csv(out("feature_importance.csv"), index=False)

# ---- console summary ------------------------------------------------------
pd.set_option("display.width", 200)
print(f"\n[{args.metric}] floor always-{BASELINE} {q_base.mean():.4f} | oracle {q_oracle.mean():.4f} | gap {gap:.4f}\n")
print(comp[["method", "mean_Q", "gap_recovered", "mean_regret", "exact_accuracy", "near_optimal_rate",
            "top3_accuracy", "override_rate", "wilcoxon_p_vs_baseline"]].round(4).to_string(index=False))
print(f"\nwrote CSVs to {OUT_DIR}/")
