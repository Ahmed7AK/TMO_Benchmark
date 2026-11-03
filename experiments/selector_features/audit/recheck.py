"""Audit: independent re-implementations of selected features (from the cited sources / luminance-hdr code)
compared with ../features/*.csv on 24 images; also quantifies the effect of the deviations found."""
import os, sys
import cv2, numpy as np, pandas as pd
from scipy.signal import convolve
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
import features as F   # only for load(); feature functions are re-derived below
imgs = pd.read_csv(f"{ROOT}/results/q_matrix.csv").image.tolist()[::41][:24]
feat = {g: pd.read_csv(f"{HERE}/../features/{g}.csv").set_index("image")
        for g in ("hulusic17", "reinhard02", "drago03", "durand02", "mai11", "tmqi", "reinhard05", "kimkautz08")}
P = [0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.09150303166, 0.9099517204, 596.3148142]
def pu21(Y):
    Y = np.clip(Y, 0.005, 10000)
    return np.maximum(P[6] * (((P[0] + P[1] * Y ** P[3]) / (1 + P[2] * Y ** P[3])) ** P[4] - P[5]), 0)
win = np.outer(*(2 * [np.exp(-0.5 * ((np.arange(11) - 5) / 1.5) ** 2)])); win /= win.sum()
rows = []
for img in imgs:
    rgb, Y = F.load(img)
    r = {"image": img}
    # Hulusic eq (1)-(3)
    Lp = (Y - Y.min()) / (Y.max() - Y.min()) * (4250 - 0.03) + 0.03
    a, b = np.percentile(Lp, [1, 99]); r["h17_dr"] = np.log10(b / a)
    l1, l99 = np.percentile(Y, [1, 99]); avg = np.mean(np.log(Y + 1e-5))
    r["h17_ik"] = (avg - np.log(max(l1, 1e-5))) / (np.log(max(l99, 1e-5)) - np.log(max(l1, 1e-5)))
    a2, b2 = np.percentile(Lp, [1, 99]); avg2 = np.mean(np.log(Lp + 1e-5))     # variant: IK on L'
    r["h17_ik_on_Lprime"] = (avg2 - np.log(a2)) / (np.log(b2) - np.log(a2))
    # Reinhard JGT key as in lhdr lischinski06 (LogMeanVal eps 1e-6)
    Lav6 = np.exp(np.mean(np.log(Y + 1e-6)))
    lo, hi = np.log2(l1 + 1e-6), np.log2(l99 + 1e-6)
    r["r02_key_lhdr"] = 0.18 * 4 ** ((2 * np.log2(Lav6 + 1e-6) - lo - hi) / (hi - lo))
    # Drago: lhdr eps 1e-4
    av4 = np.exp(np.mean(np.log(Y + 1e-4))); r["d03_divider_lhdr"] = np.log10(Y.max() / av4 + 1)
    # Mai: lhdr safelog10 floor 1e-5
    p, _ = np.histogram(np.log10(np.maximum(Y, 1e-5)), bins=150, range=(-6, 9)); p = p / p.sum()
    r["m11_sum_p13_lhdr"] = (p ** (1 / 3)).sum()
    # Durand: lhdr intensity (20R+40G+B)/61, range Gaussian exp(-d^2/sr^2) -> cv2 sigmaColor = 2/sqrt(2)
    I = np.log(np.maximum((20 * rgb[..., 0] + 40 * rgb[..., 1] + rgb[..., 2]) / 61, F.pos_floor(Y))).astype(np.float32)
    base = cv2.bilateralFilter(I, d=-1, sigmaColor=2 / np.sqrt(2), sigmaSpace=2.0).astype(float)
    q1, q99 = np.percentile(base, [1, 99]); r["d02_base_range_lhdr"] = q99 - q1
    # TMQI HDR side via scipy 'valid' convolution as in tmqi-revised
    L = (2 ** 32 - 1) * (Y - Y.min()) / (Y.max() - Y.min()); f = 32.0
    for lev in range(5):
        f /= 2
        mu = convolve(L, win, "valid"); s = np.sqrt(np.maximum(convolve(L * L, win, "valid") - mu * mu, 0))
        csf = 100 * 2.6 * (0.0192 + 0.114 * f) * np.exp(-(0.114 * f) ** 1.1); u = 128 / (1.4 * csf)
        r[f"tmqi_vis{lev}"] = norm.cdf(s, u, u / 3).mean()
        L = convolve(L, np.ones((2, 2)) / 4, "valid")[::2, ::2]
    # Reinhard 05 (lhdr: ln(max), ln(min) without floor)
    av5 = np.mean(np.log(Y + 2.3e-5)); r["r05_key_lhdr"] = (np.log(Y.max()) - av5) / (np.log(Y.max()) - np.log(Y.min())) if Y.min() > 0 else 0.0
    r["Ymin_is_zero"] = int(Y.min() == 0)
    # Kim & Kautz
    lk = np.log(Y + 1e-6); r["kk08_k1"] = (np.log(300) - np.log(0.3)) / (lk.max() - lk.min())
    rows.append(r)
R = pd.DataFrame(rows).set_index("image")
def cmp(mine, g, theirs):
    t = feat[g].loc[R.index, theirs].to_numpy(float); m = R[mine].to_numpy(float)
    rel = np.abs(m - t) / (np.abs(t) + 1e-12)
    rho = np.corrcoef(m, t)[0, 1] if np.std(m) > 0 and np.std(t) > 0 else np.nan
    return dict(mine=mine, theirs=f"{g}.{theirs}", max_rel_diff=rel.max(), median_rel_diff=np.median(rel), corr=rho)
out = pd.DataFrame([cmp("h17_dr", "hulusic17", "h17_dr"), cmp("h17_ik", "hulusic17", "h17_ik"),
                    cmp("h17_ik_on_Lprime", "hulusic17", "h17_ik"),
                    cmp("r02_key_lhdr", "reinhard02", "r02_key_est"), cmp("d03_divider_lhdr", "drago03", "d03_divider"),
                    cmp("m11_sum_p13_lhdr", "mai11", "m11_sum_p13"), cmp("d02_base_range_lhdr", "durand02", "d02_base_range"),
                    *[cmp(f"tmqi_vis{l}", "tmqi", f"tmqi_vis{l}") for l in range(5)],
                    cmp("r05_key_lhdr", "reinhard05", "r05_key"), cmp("kk08_k1", "kimkautz08", "kk08_k1")])
pd.set_option("display.width", 200)
print(out.round(5).to_string(index=False)); print("images with Y.min()==0:", int(R.Ymin_is_zero.sum()), "of", len(R))
out.round(6).to_csv(os.path.join(HERE, "recheck.csv"), index=False)
