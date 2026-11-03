"""
Pivot a sweep's results/per_image.csv into results/q_matrix.csv (images x
operators of TMQI Q), keeping only images every operator scored successfully.
--metric picks another score column (FSITM, FSITM_TMQI, Q2, S, S2, ...), written
to results/q_matrix_<metric>.csv.

--metric u / sigma / u2 / sd2 are not score columns: they are read from the
tone-mapped PNGs in tmo_outputs/ for the same images, and are the output
statistics the naturalness terms depend on (training targets for the
structured selector in compare_selectors.py):
    u      TMQI: mean luminance, Y = .2126 R + .7152 G + .0722 B on 0..255
    sigma  TMQI: mean std over 11x11 blocks of Y, zero-padded as tmqi-revised does
    u2     TMQI-II: mean of Y2 = .213 R + .715 G + .072 B
    sd2    TMQI-II: std (ddof=1) of Y2
naturalness() / naturalness2() below compose N / N2 from them, and the script
checks that they reproduce the recorded scores.

--metric est_mu2 / est_sd2 are TMQI-II's HDR-side estimates of a natural LDR's
mean and std, which N2 scores u2 / sd2 against. They come from the HDR alone,
but per operator: scoring crops (catmo) or resizes the HDR to the operator's
output size, which is a fixed function of the input size (catmo: each side
down to a multiple of 32). They are prediction-time inputs, not targets.

Image ids are written as ints to match features_selector.csv / scenes.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd

OUTPUT_STATS = ("u", "sigma", "u2", "sd2")
HDR_ESTIMATES = ("est_mu2", "est_sd2")


def hdr_estimates(img, ops, outputs_dir):
    """per operator: TMQI-II's est_mu / est_sigma on the HDR as scored against that operator's
    output (tmo_pipeline._scoring_pair geometry; formula as in tmo_metrics.tmqi2)"""
    import tmo_pipeline as P
    hdr_bgr = P.load_hdr(os.path.join(P.HDR_DIR, f"{img}.hdr"))
    out = []
    for op in ops:
        hdr_rgb, _ = P._scoring_pair(hdr_bgr, os.path.join(outputs_dir, f"{img}__{op}.png"), op)
        h = np.asarray(hdr_rgb, dtype=np.float64)
        L = 0.213 * h[..., 0] + 0.715 * h[..., 1] + 0.072 * h[..., 2]
        L = np.round((2 ** 32 - 1) / (L.max() - L.min())) * (L - L.min())
        Ls = 0.12 / np.exp(np.mean(np.log(L + np.finfo(float).eps))) * L
        est = (Ls / (1 + Ls)) ** (1 / 2.2) * 255
        out.append(dict(est_mu2=est.mean(), est_sd2=est.std(ddof=1)))
    return out


def output_stats(path):
    import cv2
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float64)
    L = rgb @ np.array([0.2126, 0.7152, 0.0722])
    T = np.pad(L, ((0, 11 - L.shape[0] % 11), (0, 11 - L.shape[1] % 11)), mode="constant")
    blocks = T.reshape(T.shape[0] // 11, 11, T.shape[1] // 11, 11).swapaxes(1, 2)
    L2 = rgb @ np.array([0.213, 0.715, 0.072])
    return dict(u=L.mean(), sigma=blocks.std(axis=(-1, -2)).mean(), u2=L2.mean(), sd2=L2.std(ddof=1))


def naturalness(u, sigma):
    """TMQI's N: gauss(u; 115.94, 27.99) * beta(sigma / 64.29; 4.4, 10.1), each normalised to peak 1"""
    from scipy.stats import beta, norm
    c0 = beta.pdf((4.4 - 1) / (4.4 + 10.1 - 2), 4.4, 10.1)
    return (norm.pdf(u, 115.94, 27.99) / norm.pdf(115.94, 115.94, 27.99)) * \
        (beta.pdf(np.clip(sigma / 64.29, 1e-6, 1 - 1e-6), 4.4, 10.1) / c0)


def naturalness2(u2, sd2, est_mu, est_sigma):
    """TMQI-II's N2 (vectorised tmo_metrics.tmqi2): the output mean / std scored against the
    HDR-side estimates est_mu / est_sigma (same shape) with tmo_metrics._bounded_cdf."""
    from scipy.stats import norm
    y1, y2 = norm.ppf([0.0001, 0.9999])

    def bounded_cdf(act, est, b_lo, b_hi):
        s_lo = (est - b_lo) / (y2 - y1)
        hi = 2 * b_hi - est
        s_hi = (hi - b_hi) / (y2 - y1)
        return np.where(act > est, norm.cdf(2 * b_hi - act, hi - s_hi * y2, s_hi),
                        norm.cdf(act, est - s_lo * y2, s_lo))

    em, es = np.asarray(est_mu), np.asarray(est_sigma)
    return bounded_cdf(u2, em, 0.6043 * em - 0.1402, 0.6993 * em + 83.6128) * \
        bounded_cdf(sd2, es, 0.6504 * es - 0.0759, 0.9386 * es + 51.3915)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results", help="folder holding per_image.csv")
    ap.add_argument("--metric", default="Q",
                    help="per_image.csv score column, or u / sigma / u2 / sd2 / est_mu2 / est_sd2")
    ap.add_argument("--outputs-dir", default="tmo_outputs", help="tone-mapped PNGs (output stats only)")
    ap.add_argument("--out", default=None,
                    help="default: <results-dir>/q_matrix.csv, or q_matrix_<metric>.csv")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.results_dir, "per_image.csv"), dtype={"image": str})
    ok = df[df.status == "success"]
    col = {"u": "N", "sigma": "N"}.get(args.metric, "N2" if args.metric in OUTPUT_STATS + HDR_ESTIMATES
                                       else args.metric)
    q = ok.pivot_table(index="image", columns="operator", values=col, aggfunc="first")
    n_all = q.shape[0]
    q = q.dropna()

    if args.metric in OUTPUT_STATS + HDR_ESTIMATES:
        from joblib import Parallel, delayed
        pairs = [(img, op) for img in q.index for op in q.columns]
        st = pd.DataFrame(Parallel(n_jobs=8)(delayed(output_stats)(
            os.path.join(args.outputs_dir, f"{img}__{op}.png")) for img, op in pairs))
        S = {c: st[c].to_numpy().reshape(q.shape) for c in st.columns}
        if col == "N":
            err = np.abs(naturalness(S["u"], S["sigma"]) - q.to_numpy())
            print(f"max |N(u, sigma) - recorded N| = {err.max():.2e}")
        else:
            est = Parallel(n_jobs=8)(delayed(hdr_estimates)(img, list(q.columns), args.outputs_dir)
                                     for img in q.index)
            est = pd.DataFrame([e for row in est for e in row])
            S.update({c: est[c].to_numpy().reshape(q.shape) for c in est.columns})
            err = np.abs(naturalness2(S["u2"], S["sd2"], S["est_mu2"], S["est_sd2"]) - q.to_numpy())
            print(f"max |N2(u2, sd2; est_mu2, est_sd2) - recorded N2| = {err.max():.2e}")
        q = pd.DataFrame(S[args.metric], index=q.index, columns=q.columns)

    q.index = q.index.astype(int)
    q.index.name = "image"

    name = "q_matrix.csv" if args.metric == "Q" else f"q_matrix_{args.metric}.csv"
    out = args.out or os.path.join(args.results_dir, name)
    q.sort_index().reset_index().to_csv(out, index=False)
    print(f"{n_all} images scored; {len(q)} with a full row of {q.shape[1]} operators -> {out}")
    bad = df[df.status != "success"].groupby("operator").size()
    if len(bad):
        print("non-success rows per operator:\n" + bad.to_string())


if __name__ == "__main__":
    main()
