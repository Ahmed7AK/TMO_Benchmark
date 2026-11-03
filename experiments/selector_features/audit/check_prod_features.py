"""Round-2 audit: results/features_selector.csv vs the study's ALL frame; group map; univariate source AUC
and NaN counts of the final features."""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import common as C
from screen import ALL
ROOT = C.ROOT
prod = pd.read_csv(f"{ROOT}/results/features_selector.csv")
grp = pd.read_csv(f"{ROOT}/results/features_selector_groups.csv")
study = C.feature_frame(ALL)
print("shapes prod/study:", prod.shape, study.shape, "| groups map rows:", len(grp))
m = prod.merge(study, on="image", suffixes=("_p", "_s"))
cols = [c for c in study.columns if c != "image"]
print("same columns:", list(prod.columns) == list(study.columns) or sorted(prod.columns) == sorted(study.columns))
d = max(np.nanmax(np.abs(m[c + "_p"].to_numpy(float) - m[c + "_s"].to_numpy(float))) for c in cols)
nanmis = sum(int((m[c + "_p"].isna() != m[c + "_s"].isna()).sum()) for c in cols)
print("max |prod - study| =", d, "| NaN-pattern mismatches:", nanmis, "| images:", len(m))
D = C.load("Q", ALL)
y = (D.src == "HDR-Real").astype(int)
rows = []
for j, c in enumerate(D.cols):
    x = D.X[:, j]; ok = np.isfinite(x)
    a = roc_auc_score(y[ok], x[ok]) if np.unique(x[ok]).size > 1 else 0.5
    rows.append((c, max(a, 1 - a), int((~ok).sum())))
t = pd.DataFrame(rows, columns=["feature", "source_auc", "n_nan"]).sort_values("source_auc", ascending=False)
t.to_csv(os.path.join(HERE, "final_feature_source_auc.csv"), index=False)
print(t.head(10).round(3).to_string(index=False)); print("features with NaN:", t[t.n_nan > 0].round(3).to_string(index=False))
