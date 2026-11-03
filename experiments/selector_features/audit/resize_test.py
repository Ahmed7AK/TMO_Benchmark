"""Audit: size sensitivity. Every HDR-Real image is 512x512 and every HDR-Synth image is smaller, so any
feature that changes when the SAME image is resampled to a Synth-like size can encode the source dataset.
Recomputes all groups on 10 HDR-Real images at 512x512 and resized (INTER_AREA) to 430x430 (the mean
Synth pixel count) and reports, per feature, the median signed change in units of the feature's
across-image std (an effect size)."""
import os, sys
import cv2, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
import features as F
man = pd.read_csv(f"{ROOT}/results/manifest.csv")
q = pd.read_csv(f"{ROOT}/results/q_matrix.csv")[["image"]].merge(man, on="image")
imgs = q[q.dataset == "HDR-Real"].image.tolist()[::60][:10]
d = []
for img in imgs:
    rgb, Y = F.load(img)
    small = np.maximum(cv2.resize(rgb.astype(np.float32), (430, 430), interpolation=cv2.INTER_AREA).astype(np.float64), 0)
    a, b = {}, {}
    for g, fn in F.GROUPS.items():
        a.update(fn(rgb, Y)); b.update(fn(small, small @ F.W709))
    d.append({k: b[k] - a[k] for k in a})
D = pd.DataFrame(d)
sd = pd.concat([pd.read_csv(f"{HERE}/../features/{g}.csv").drop(columns="image") for g in F.GROUPS], axis=1).std(numeric_only=True)
eff = (D.median() / sd.reindex(D.columns)).rename("median_change_in_sd")
eff.round(3).sort_values(key=abs, ascending=False).to_csv(os.path.join(HERE, "resize_sensitivity.csv"))
print(eff.abs().describe().round(3).to_string())
print(eff.sort_values(key=abs, ascending=False).head(25).round(2).to_string())
print("features moving > 0.5 sd:", int((eff.abs() > 0.5).sum()), "of", len(eff))
