"""Audit: per-feature source AUC (univariate) and Spearman correlation with image size (pixels), for every
feature in ../features/*.csv; flags features that encode image size or the source dataset."""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
hf = pd.read_csv(f"{ROOT}/results/hdr_features.csv")[["image", "width", "height"]]
hf["pixels"] = hf.width * hf.height
man = pd.read_csv(f"{ROOT}/results/manifest.csv")[["image", "dataset"]]
q = pd.read_csv(f"{ROOT}/results/q_matrix.csv")[["image"]]
base = q.merge(hf, on="image").merge(man, on="image")
real = (base.dataset == "HDR-Real").astype(int).to_numpy()
print("size by source:", base.groupby(base.dataset == "HDR-Real").pixels.describe()[["mean", "min", "max"]].to_string())
print("source AUC of pixels:", round(max(roc_auc_score(real, base.pixels), 1 - roc_auc_score(real, base.pixels)), 3))
rows = []
for p in sorted(glob.glob(os.path.join(HERE, "..", "features", "*.csv"))):
    g = os.path.basename(p)[:-4]
    if g == "timing":
        continue
    f = base[["image", "pixels"]].merge(pd.read_csv(p), on="image")
    for c in f.columns[2:]:
        x = f[c].to_numpy(float)
        ok = np.isfinite(x)
        a = roc_auc_score(real[ok], x[ok]) if ok.sum() > 10 and np.unique(x[ok]).size > 1 else 0.5
        rho = spearmanr(x[ok], f.pixels[ok]).correlation if np.unique(x[ok]).size > 1 else 0.0
        rows.append(dict(group=g, feature=c, source_auc=max(a, 1 - a), spearman_pixels=rho, n_nan=int((~ok).sum()),
                         n_unique=int(np.unique(x[ok]).size)))
t = pd.DataFrame(rows)
t.round(4).to_csv(os.path.join(HERE, "leak_scan.csv"), index=False)
pd.set_option("display.width", 200)
print(t.sort_values("spearman_pixels", key=abs, ascending=False).head(15).round(3).to_string(index=False))
print(t.sort_values("source_auc", ascending=False).head(15).round(3).to_string(index=False))
print(t[t.n_nan > 0].round(3).to_string(index=False))
