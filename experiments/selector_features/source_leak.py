"""
How well does each feature group identify the source dataset (HDR-Real vs HDR-Synth)? A group that
separates the sources almost perfectly could let a selector learn "which dataset" instead of image
content. Reports the 5-fold cross-validated ROC AUC of a random forest per group -> tables/source_leak.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

import common as C
from features import GROUPS

rows = []
for g in ["hdr22"] + list(GROUPS):
    D = C.load("Q", g)
    y = (D.src == "HDR-Real").astype(int)
    m = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(300, n_jobs=-1, random_state=0))
    auc = cross_val_score(m, D.X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
    rows.append(dict(group=g, n_features=len(D.cols), source_auc=float(np.mean(auc))))
    print(f"{g:12s} {len(D.cols):4d} AUC {np.mean(auc):.3f}", flush=True)
pd.DataFrame(rows).round(4).to_csv(os.path.join(C.TAB, "source_leak.csv"), index=False)
