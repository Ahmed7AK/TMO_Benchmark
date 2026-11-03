"""
Export the benchmark as a self-contained dataset for training TMO classifiers / selectors -> dataset/.

  tmo_selection_dataset.csv   one row per image (992): id and source, the 151 selector features, a
                              10-bin log-luminance histogram of the HDR (pixel counts and fractions),
                              TMQI Q of all 17 operators, the winning operator (label), the runner-up,
                              and the cross-validation fold the reported selector used
  scores_long.csv             every (image, operator) pair: TMQI Q / S / N, FSITM, FSITM_TMQI,
                              TMQI-II Q2 / S2 / N2, and the selector's training targets u / sigma
  feature_dictionary.csv      each feature column: group, published source, what the group measures
  selector_oof_predictions.csv  the reported selector's out-of-fold pick per image (for comparison only;
                              never use it as a feature)
  hdr_features_original22.csv the project's original 21 HDR features used before the final set
  DATA_CARD.md                written by hand, describes all of the above

Histogram: Rec. 709 luminance of the linear HDR, log10, 10 equal-width bins from the image's smallest
positive luminance (zeros are floored to it, as in selector_features.pos_floor) to its maximum. Counts
are pixels (images differ in size, so the fractions are the comparable form). It is a statistic of the
HDR input only; it carries no information about any operator's output.

    python export_dataset.py [--bins 10]
"""

import argparse
import os

import numpy as np
import pandas as pd

import selector_features as sf

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "dataset")

GROUP_INFO = {
    "hulusic17": ("Hulusic et al. 2017, Signal Process. Image Commun. 51",
                  "perceived dynamic range: DR, image key, bright area, PU21 contrast, colourfulness"),
    "pouli10": ("Pouli, Cunningham & Reinhard 2010, Eurographics STAR",
                "HDR image statistics: log-luminance moments, power-spectrum slope, gradients, "
                "band-pass statistics, Lαβ colour"),
    "reinhard02": ("Reinhard et al. 2002, SIGGRAPH; Reinhard 2002, J. Graphics Tools 7(1)",
                   "photographic operator set-up: log-average, white point, automatic key"),
    "drago03": ("Drago et al. 2003, Comput. Graph. Forum 22(3)", "adaptive log mapping set-up: Lwmax/Lwa, log base"),
    "durand02": ("Durand & Dorsey 2002, SIGGRAPH", "bilateral base-layer range, compression factor, detail energy"),
    "fattal02": ("Fattal, Lischinski & Werman 2002, SIGGRAPH",
                 "gradient-domain set-up: average gradient, attenuation at pyramid levels 3-5"),
    "reinhard05": ("Reinhard & Devlin 2005, IEEE TVCG 11(1)", "photoreceptor operator: image key, contrast m"),
    "kimkautz08": ("Kim & Kautz 2008, APGV", "consistent tone reproduction: log-mean, compression k1"),
    "mai11": ("Mai et al. 2011, IEEE TIP 20(6)", "optimal tone curve: log10 histogram statistics"),
    "mantiuk08": ("Mantiuk, Daly & Kerofsky 2008, ACM TOG 27(5)",
                  "display-adaptive TMO: band-contrast distribution per Laplacian level"),
    "tmqi": ("Yeganeh & Wang 2013, IEEE TIP 22(2)", "TMQI's HDR-side local standard deviation at 5 scales"),
    "brisque": ("Mittal, Moorthy & Bovik 2012, IEEE TIP 21(12)",
                "BRISQUE natural-scene statistics on log luminance"),
    "si": ("ITU-T Rec. P.910", "spatial perceptual information on log luminance"),
    "tmqiblock": ("Yeganeh & Wang 2013 (adapted)", "11/22/44-px block statistics of log luminance"),
}


def histogram(img, bins):
    rgb, Y = sf.load(img)
    lo = sf.pos_floor(Y)
    L = np.log10(np.maximum(Y, lo))
    counts, edges = np.histogram(L, bins=bins, range=(L.min(), L.max()))
    row = {"image": img, "hist_log10_min": edges[0], "hist_log10_max": edges[-1], "n_pixels": L.size}
    row.update({f"hist_bin_{i + 1:02d}": int(c) for i, c in enumerate(counts)})
    row.update({f"hist_frac_{i + 1:02d}": c / L.size for i, c in enumerate(counts)})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    qm = pd.read_csv(os.path.join(RES, "q_matrix.csv"))
    ops = [c for c in qm.columns if c != "image"]
    images = qm.image.tolist()
    man = pd.read_csv(os.path.join(RES, "manifest.csv")).set_index("image")
    feat = pd.read_csv(os.path.join(RES, "features_selector.csv"))
    groups = pd.read_csv(os.path.join(RES, "features_selector_groups.csv"))
    pred = pd.read_csv(os.path.join(RES, "selectors", "Q", "selector_predictions.csv"))
    scenes = pd.read_csv(os.path.join(RES, "scenes.csv"))

    from joblib import Parallel, delayed
    hist = pd.DataFrame(Parallel(n_jobs=4)(delayed(histogram)(i, args.bins) for i in images))

    Q = qm[ops].to_numpy(float)
    order = np.argsort(-Q, axis=1)
    rows = np.arange(len(qm))
    lab = pd.DataFrame({
        "image": qm.image,
        "best_tmo": np.array(ops)[order[:, 0]],
        "second_best_tmo": np.array(ops)[order[:, 1]],
        "TMQI_best": Q[rows, order[:, 0]],
        "TMQI_second_best": Q[rows, order[:, 1]],
    })
    lab["TMQI_gap_best_second"] = lab.TMQI_best - lab.TMQI_second_best

    ident = pd.DataFrame({
        "image": images,
        "file": [man.loc[i, "file"] for i in images],
        "source": ["HDR-Real" if man.loc[i, "dataset"] == "HDR-Real" else "HDR-Synth" for i in images],
        "source_subset": [man.loc[i, "dataset"] for i in images],
        "width": [man.loc[i, "w"] for i in images],
        "height": [man.loc[i, "h"] for i in images],
    }).merge(scenes, on="image").merge(pred[["image", "fold"]].rename(columns={"fold": "cv_fold"}), on="image")

    tmqi = qm.rename(columns={o: f"TMQI_{o}" for o in ops})
    wide = ident.merge(feat, on="image").merge(hist, on="image").merge(tmqi, on="image").merge(lab, on="image")
    assert len(wide) == len(images) and not wide[[f"TMQI_{o}" for o in ops]].isna().any().any()
    wide.to_csv(os.path.join(OUT, "tmo_selection_dataset.csv"), index=False)

    # long scores: the sweep's per-pair metrics plus the output statistics behind TMQI's naturalness
    pi = pd.read_csv(os.path.join(RES, "per_image.csv"))
    pi = pi[pi.image.isin(images) & pi.operator.isin(ops) & (pi.status == "success")]
    long = pi[["image", "operator", "Q", "S", "N", "FSITM", "FSITM_TMQI", "Q2", "S2", "N2"]]
    for c in ("u", "sigma"):
        m = pd.read_csv(os.path.join(RES, f"q_matrix_{c}.csv")).melt("image", var_name="operator", value_name=c)
        long = long.merge(m, on=["image", "operator"], how="left")
    long = long.sort_values(["image", "operator"])
    assert len(long) == len(images) * len(ops)
    chk = long.pivot(index="image", columns="operator", values="Q").loc[qm.image, ops].to_numpy(float)
    assert np.allclose(chk, Q), "per_image.csv and q_matrix.csv disagree"
    long.to_csv(os.path.join(OUT, "scores_long.csv"), index=False)

    fd = groups.rename(columns={"column": "feature"})
    fd["source"] = fd.group.map(lambda g: GROUP_INFO[g][0])
    fd["measures"] = fd.group.map(lambda g: GROUP_INFO[g][1])
    hist_cols = [c for c in hist.columns if c != "image"]
    fd = pd.concat([fd, pd.DataFrame({"feature": hist_cols, "group": "histogram",
                                      "source": "this export (export_dataset.py)",
                                      "measures": f"{args.bins}-bin histogram of log10 HDR luminance, "
                                                  "per-image min-max range"})])
    fd.to_csv(os.path.join(OUT, "feature_dictionary.csv"), index=False)

    pred.to_csv(os.path.join(OUT, "selector_oof_predictions.csv"), index=False)
    h22 = pd.read_csv(os.path.join(RES, "hdr_features.csv"))
    h22 = h22[h22.image.isin(images)].drop(columns=["width", "height", "aspect_ratio", "megapixels"])
    h22.to_csv(os.path.join(OUT, "hdr_features_original22.csv"), index=False)

    print(f"{len(wide)} images x {wide.shape[1]} columns -> dataset/tmo_selection_dataset.csv")
    print("label distribution:", lab.best_tmo.value_counts().to_dict())


if __name__ == "__main__":
    main()
