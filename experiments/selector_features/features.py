"""
Research-backed HDR features for the per-image TMO selector.

Every feature is a statistic of the HDR radiance map itself (images/*.hdr, linear RGB). No tone-mapping
operator is run, no quality metric is computed between images, and nothing is read from tmo_outputs/.
Each group follows one published source; deviations from the source are stated in the group's docstring.
Rec. 709 luminance Y = 0.2126 R + 0.7152 G + 0.0722 B throughout (the weights TMQI uses).

Base-set candidates (one paper each)
  hulusic17   Hulusic, Debattista, Valenzise, Dufaux. A model of perceived dynamic range for HDR images.
              Signal Processing: Image Communication 51 (2017) 26-39.                     5 features
  pouli10     Pouli, Cunningham, Reinhard. Image statistics and their applications in computer graphics.
              Eurographics 2010 STAR.                                                     27 features

Further groups for the compiled multi-paper set
  reinhard02  Reinhard et al. 2002 (SIGGRAPH) + Reinhard 2002 (JGT 7(1)) automatic parameter estimation
  drago03     Drago et al. 2003 (CGF 22(3)) adaptive logarithmic mapping
  durand02    Durand & Dorsey 2002 (SIGGRAPH) bilateral base / detail decomposition
  fattal02    Fattal, Lischinski & Werman 2002 (SIGGRAPH) gradient-domain attenuation
  reinhard05  Reinhard & Devlin 2005 (IEEE TVCG 11(1)) photoreceptor operator, image key / contrast
  kimkautz08  Kim & Kautz 2008 (EGSR/APGV) consistent tone reproduction
  mai11       Mai et al. 2011 (IEEE TIP 20(6)) optimal tone curve: log-luminance histogram
  mantiuk08   Mantiuk, Daly & Kerofsky 2008 (SIGGRAPH Asia) display adaptive TMO: contrast distribution
  tmqi        Yeganeh & Wang 2013 (IEEE TIP 22(2)) TMQI: HDR-side structural-fidelity statistics
  brisque     Mittal, Moorthy & Bovik 2012 (IEEE TIP 21(12)) BRISQUE NSS features (used for tone-mapped
              image quality by Kundu et al. 2017, IEEE TIP 26(6))
  si          ITU-T Rec. P.910 spatial perceptual information
  tmqiblock   statistics of 11x11 blocks of log luminance: TMQI's naturalness block size (Yeganeh & Wang
              2013), applied to the HDR (an adaptation; TMQI itself applies it to the LDR)

For the operators, the statistics follow the luminance-hdr implementations the benchmark ran
(github.com/LuminanceHDR/LuminanceHDR, src/TonemappingOperators/<op>/), with their default parameters.
These are the image statistics each operator computes from the HDR to set itself up. The operator
itself is never applied.

    ../../.venv/bin/python features.py [--jobs 8] [--groups ...]  -> features/<group>.csv
"""
import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import gamma as G

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "features")
W709 = np.array([0.2126, 0.7152, 0.0722])


def load(img):
    bgr = cv2.imread(os.path.join(ROOT, "images", f"{img:05d}.hdr"), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    rgb = np.maximum(bgr[..., ::-1].astype(np.float64), 0)
    return rgb, rgb @ W709


def pos_floor(Y):
    pos = Y[Y > 0]
    return max(pos.min() if pos.size else 1e-9, Y.max() * 1e-9, 1e-30)


def ln_lum(Y):
    return np.log(np.maximum(Y, pos_floor(Y)))


def moments(x):
    """mean, std, skewness m3/s^3, kurtosis m4/s^4 (Pouli et al. 2010, eq. 10-12)"""
    m, s = x.mean(), x.std()
    if s == 0:
        return m, 0.0, 0.0, 0.0
    d = x - m
    return m, s, (d ** 3).mean() / s ** 3, (d ** 4).mean() / s ** 4


def to_8bit_log(Y):
    """fixed encoding for metrics defined on 8-bit images: ln Y mapped linearly from its range to [0, 255]"""
    l = ln_lum(Y)
    return 255.0 * (l - l.min()) / max(l.max() - l.min(), 1e-12)


# ---------------------------------------------------------------------------------- base candidates --
def pu21_encode(Y):
    """PU21 'banding_glare' encoding (Mantiuk & Azimi 2021; github.com/gfxdisp/pu21 pu21_encoder.m)"""
    p = [0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.09150303166, 0.9099517204, 596.3148142]
    Y = np.clip(Y, 0.005, 10000.0)
    return np.maximum(p[6] * (((p[0] + p[1] * Y ** p[3]) / (1 + p[2] * Y ** p[3])) ** p[4] - p[5]), 0)


def colourfulness_hs(rgb, Y):
    """Hasler & Suesstrunk 2003 colourfulness on linear radiance, divided by the median luminance so that
    it does not depend on the HDR's arbitrary scale (as in this project's hdr_features.py)"""
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    v = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    return v / (np.median(Y) + 1e-12)


def f_hulusic17(rgb, Y):
    """Hulusic et al. 2017, Sec. 3.2, eq. (1)-(6).
    - L' = display-scaled luminance, eq. (1), Disp_min = 0.03, Disp_max = 4250 cd/m^2 (their display).
    - DR  = log10(max L' / min L') after excluding the 1% darkest and brightest pixels, eq. (2).
    - IK  = image key of Akyuz & Reinhard (2006), eq. (3), read literally: on the unscaled relative
      luminance L, ln avg with delta = 1e-5, min / max robust (1%). (Eq. (4) also writes L where it must
      mean the display-scaled L'; on L' the key differs, r = 0.92.)
    - Area: pixels of L' above the 2,400 cd/m^2 diffuse-white threshold, eq. (4). Deviation: reported as a
      fraction of the image (the paper counts pixels; our images differ in size).
    - Contrast: eq. (5)-(6), mean absolute difference to the 8-neighbourhood at Gaussian-pyramid levels
      j = 2..4, averaged. Deviation: PU21 encoding stands in for the PU encoding of Aydin et al. 2008 [38].
    - Colourfulness: one of the paper's five measures, but neither this paper nor Hulusic et al. 2016
      gives its formula. The formula used is Hasler & Suesstrunk (2003), divided by the median luminance;
      it should be credited to Hasler & Suesstrunk, not to Hulusic et al."""
    Lp = (Y - Y.min()) / max(Y.max() - Y.min(), 1e-30) * (4250 - 0.03) + 0.03
    p1, p99 = np.percentile(Lp, [1, 99])
    d = 1e-5
    l1, l99 = np.percentile(Y, [1, 99])
    lo, hi = np.log(max(l1, d)), np.log(max(l99, d))
    pyr = [pu21_encode(Lp).astype(np.float32)]
    for _ in range(4):
        pyr.append(cv2.pyrDown(pyr[-1]))
    lc = []
    for j in (2, 3, 4):
        X = pyr[j].astype(np.float64)
        c = X[1:-1, 1:-1]
        acc = np.zeros_like(c)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    acc += np.abs(c - X[1 + dy:X.shape[0] - 1 + dy, 1 + dx:X.shape[1] - 1 + dx])
        lc.append((acc / 8).mean())
    return {"h17_dr": np.log10(p99 / p1),
            "h17_ik": (np.mean(np.log(Y + d)) - lo) / max(hi - lo, 1e-12),
            "h17_area": float((Lp > 2400).mean()),
            "h17_contrast": float(np.mean(lc)),
            "h17_colourfulness": colourfulness_hs(rgb, Y)}


M_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
M_LMS = np.array([[0.3897, 0.6890, -0.0787], [-0.2298, 1.1834, 0.0464], [0.0, 0.0, 1.0]])   # Pouli eq. 47
M_LAB = np.diag([1 / np.sqrt(3), 1 / np.sqrt(6), 1 / np.sqrt(2)]) @ np.array([[1, 1, 1], [1, 1, -2], [1, -1, 0]])


def laplacian_bands(x, n):
    bands, g = [], x.astype(np.float32)
    for _ in range(n):
        down = cv2.pyrDown(g)
        bands.append((g - cv2.pyrUp(down, dstsize=(g.shape[1], g.shape[0]))).astype(np.float64))
        g = down
    return bands


def f_pouli10(rgb, Y):
    """Pouli, Cunningham & Reinhard 2010 (EG STAR), statistics of natural / HDR images:
    - Sec. 3: central moments of log luminance (mean, std, skewness, kurtosis), eq. (10)-(12)
    - Sec. 4.1: power spectrum S = |F|^2 / N^2 of log luminance, eq. (16), averaged over orientation; slope
      beta of the 1/f^beta fit and intercept on log-log axes (Hann window; DC and the highest frequencies
      excluded)
    - Sec. 4.2: forward-difference gradients of log luminance, eq. (17)-(20): std / kurtosis of Dx, Dy and
      mean / std / kurtosis of the magnitude
    - Sec. 5.2: band-pass (wavelet-type) coefficients; std and kurtosis at 4 scales, here computed with a
      Laplacian pyramid (Burt & Adelson 1983)
    - Sec. 6.2: decorrelated L-alpha-beta colour space of Ruderman et al. (XYZ -> LMS eq. 47, log10, then
      the rotation of eq. (49), the same as Reinhard et al. 2001); mean and std per channel
    The survey defines these statistic families by equation but prescribes no feature set: which
    moments / summaries are taken is this study's choice. The spectral fit uses a fixed lower frequency
    (2 cycles per 288 px, the smallest image side) so that it does not depend on image size."""
    l = ln_lum(Y)
    f = dict(zip(("p10_logl_mean", "p10_logl_std", "p10_logl_skew", "p10_logl_kurt"), moments(l)))
    h, w = l.shape
    P = np.abs(np.fft.fft2((l - l.mean()) * np.outer(np.hanning(h), np.hanning(w)))) ** 2 / (h * w)
    fr = np.hypot(*np.meshgrid(np.fft.fftfreq(w), np.fft.fftfreq(h)))
    edges = np.logspace(np.log10(2.0 / 288), np.log10(0.4), 21)
    lf, lp = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (fr >= a) & (fr < b)
        if m.any():
            lf.append(np.log10(np.sqrt(a * b)))
            lp.append(np.log10(P[m].mean() + 1e-30))
    slope, icpt = np.polyfit(lf, lp, 1)
    f["p10_spectral_slope"], f["p10_spectral_intercept"] = -slope, icpt
    dx, dy = np.diff(l, axis=1), np.diff(l, axis=0)
    mag = np.hypot(dx[:-1, :], dy[:, :-1])
    for name, v in (("dx", dx), ("dy", dy)):
        _, s, _, k = moments(v.ravel())
        f[f"p10_grad_{name}_std"], f[f"p10_grad_{name}_kurt"] = s, k
    mm, ms, _, mk = moments(mag.ravel())
    f.update(p10_grad_mag_mean=mm, p10_grad_mag_std=ms, p10_grad_mag_kurt=mk)
    for i, b in enumerate(laplacian_bands(l, 4)):
        _, s, _, k = moments(b.ravel())
        f[f"p10_band{i}_std"], f[f"p10_band{i}_kurt"] = s, k
    lms = rgb.reshape(-1, 3) @ M_XYZ.T @ M_LMS.T
    lab = np.log10(np.maximum(lms, max(lms[lms > 0].min(), 1e-30))) @ M_LAB.T
    for i, c in enumerate(("L", "alpha", "beta")):
        f[f"p10_lab_{c}_mean"], f[f"p10_lab_{c}_std"] = lab[:, i].mean(), lab[:, i].std()
    return f


# ------------------------------------------------------------------- operator-internal statistics --
def f_reinhard02(rgb, Y):
    """Reinhard et al. 2002 log-average luminance (eq. 1; delta = 1e-5 as in luminance-hdr reinhard02) and
    the white point the operator uses by default (Lwhite = largest scaled luminance, key 0.18), plus
    Reinhard 2002 (JGT) automatic parameter estimation as implemented verbatim in luminance-hdr
    lischinski06 (log-mean with eps 1e-6; Lmin / Lmax = 1st / 99th percentiles):
    key = 0.18 * 4^((2 log2 Lav - log2 Lmin - log2 Lmax) / (log2 Lmax - log2 Lmin)),
    white = 1.5 * 2^(log2 Lmax - log2 Lmin - 5), and the dynamic range in zones (stops) it is based on."""
    eps = 1e-6
    Lav = np.exp(np.mean(np.log(Y + 1e-5)))
    Lav_l = np.exp(np.mean(np.log(Y + eps)))
    p1, p99 = np.percentile(Y, [1, 99])
    a2, lo, hi = np.log2(Lav_l + eps), np.log2(p1 + eps), np.log2(p99 + eps)
    rng = max(hi - lo, 1e-6)
    return {"r02_log_avg": np.log(Lav), "r02_log2_white_default": np.log2(0.18 * Y.max() / Lav),
            "r02_key_est": 0.18 * 4 ** ((2 * a2 - lo - hi) / rng),
            "r02_log2_white_est": np.log2(1.5) + rng - 5, "r02_zones": rng}


def f_drago03(rgb, Y):
    """Drago et al. 2003: world adaptation = log-average, Lwmax / Lwa sets the log base;
    luminance-hdr drago03: log-average with eps = 1e-4, divider = log10(Lwmax / Lwa + 1)"""
    av = np.exp(np.mean(np.log(Y + 1e-4)))
    r = Y.max() / av
    return {"d03_log10_max_over_av": np.log10(r), "d03_divider": np.log10(r + 1)}


def f_durand02(rgb, Y):
    """Durand & Dorsey 2002 (luminance-hdr durand02): intensity I = (20 R + 40 G + B) / 61, base =
    bilateral(ln I), detail = ln I - base; CLI defaults sigma_s = 2 px, sigma_r = 2 with range kernel
    exp(-d^2 / sigma_r^2) (= OpenCV sigmaColor sqrt(2)), base contrast 5; compression factor =
    5 / (p99 - p1 of the base layer)"""
    inten = (20 * rgb[..., 0] + 40 * rgb[..., 1] + rgb[..., 2]) / 61
    I = ln_lum(inten).astype(np.float32)
    base = cv2.bilateralFilter(I, d=-1, sigmaColor=float(np.sqrt(2)), sigmaSpace=2.0).astype(np.float64)
    b1, b99 = np.percentile(base, [1, 99])
    detail = I - base
    return {"d02_base_range": b99 - b1, "d02_compression": 5.0 / max(b99 - b1, 1e-6),
            "d02_detail_std": float(detail.std()), "d02_detail_absmean": float(np.abs(detail).mean())}


def f_fattal02(rgb, Y):
    """Fattal et al. 2002: H = ln(100 Y / max Y + 1e-4) (luminance-hdr fattal02); Gaussian pyramid down to
    8 px (the FFT solver the CLI uses); central-difference gradient magnitude at level k divided by
    2^(k+1); the average gradient sets the attenuation threshold alpha * avg (alpha = 1, beta = 0.9).
    At full resolution the CLI attenuates only levels k >= 3 (detail_level 3), so the features are taken
    at k = 3, 4, 5, which every benchmark image has (smallest side 288 px): average gradient and the mean
    log gradient relative to it (the log attenuation is (beta - 1) times this)."""
    H = np.log(100.0 * Y / max(Y.max(), 1e-30) + 1e-4).astype(np.float32)
    f, g = {}, H
    for k in range(6):
        if k >= 3:
            gx = (np.roll(g, -1, 1) - np.roll(g, 1, 1))[:, 1:-1] / 2 ** (k + 1)
            gy = (np.roll(g, -1, 0) - np.roll(g, 1, 0))[1:-1, :] / 2 ** (k + 1)
            mag = np.hypot(gx[1:-1, :], gy[:, 1:-1]).astype(np.float64)
            avg = mag.mean()
            f[f"f02_avg_grad{k}"] = avg
            f[f"f02_rel_loggrad{k}"] = float(np.mean(np.log((mag + 1e-6) / (avg + 1e-6))))
        g = cv2.pyrDown(g)
    return f


def f_reinhard05(rgb, Y):
    """Reinhard & Devlin 2005 (luminance-hdr reinhard05): adapted average = mean ln(L + 2.3e-5);
    key k = (ln Lmax - avg) / (ln Lmax - ln Lmin); contrast m = 0.3 + 0.7 k^1.4.
    Deviation: for an image whose minimum is 0 (ln Lmin = -inf in the code, k = 0) Lmin is floored at
    2.3e-5; other images use their true minimum."""
    avg = np.mean(np.log(Y + 2.3e-5))
    hi = np.log(max(Y.max(), 2.3e-5))
    lo = np.log(Y.min()) if Y.min() > 0 else np.log(2.3e-5)
    k = (hi - avg) / max(hi - lo, 1e-6)
    return {"r05_key": k, "r05_contrast_m": 0.3 + 0.7 * k ** 1.4}


def f_kimkautz08(rgb, Y):
    """Kim & Kautz 2008 (luminance-hdr kimkautz08): ln(L + 1e-6), mean mu, and the global compression
    k1 = (ln 300 - ln 0.3) / (max - min) of the log luminance"""
    l = np.log(Y + 1e-6)
    return {"kk08_mu": l.mean(), "kk08_k1": (np.log(300) - np.log(0.3)) / max(l.max() - l.min(), 1e-6)}


def f_mai11(rgb, Y):
    """Mai et al. 2011: histogram p_k of log10 L (floored at 1e-5), bins of 0.1 from -6 to 9 (luminance-hdr
    mai11); the optimal tone-curve slopes are proportional to p_k^(1/3): their normaliser sum p_k^(1/3)
    and the largest slope share"""
    l = np.log10(np.maximum(Y, 1e-5))
    p, _ = np.histogram(l, bins=150, range=(-6, 9))
    p = p / max(p.sum(), 1)
    s = p ** (1 / 3)
    return {"m11_sum_p13": s.sum(), "m11_max_slope_share": s.max() / max(s.sum(), 1e-12)}


def f_mantiuk08(rgb, Y):
    """Mantiuk et al. 2008, Sec. 3-4.1: I = log10 L, Gaussian pyramid I_l, band contrast G_l = I_l - I_(l+1)
    (log ratio); their statistic is the histogram of G_l conditional on the background I_(l+1). luminance-hdr
    uses 4 bands (30 px/deg). Summarised per band: mean |G|, 90th percentile |G|, and mean |G| on dark minus
    bright backgrounds (split at the median background). Deviation: decimated pyramid levels (the
    implementation keeps every band at full resolution)."""
    I = np.log10(np.maximum(Y, pos_floor(Y))).astype(np.float32)
    f = {}
    for l in range(4):
        down = cv2.pyrDown(I)
        bg = cv2.pyrUp(down, dstsize=(I.shape[1], I.shape[0]))
        Gl = np.abs(I - bg).astype(np.float64)
        dark = bg <= np.median(bg)
        f[f"m08_absG_mean{l}"] = Gl.mean()
        f[f"m08_absG_p90_{l}"] = np.percentile(Gl, 90)
        f[f"m08_dark_minus_bright{l}"] = Gl[dark].mean() - (Gl[~dark].mean() if (~dark).any() else 0.0)
        I = down
    return f


# --------------------------------------------------------------------------------- metric / IQA --
def f_tmqi(rgb, Y):
    """Metric-derived: Yeganeh & Wang 2013 TMQI structural fidelity, HDR side only (tmqi-revised implementation the
    benchmark scores with): Y rescaled to [0, 2^32 - 1]; 5 levels (2x2 mean + downsample); local sigma
    in an 11x11 Gaussian window (sigma 1.5); mapped by normcdf(sigma; u, u/3), u = 128 / (1.4 CSF(f)),
    CSF(f) = 100 * 2.6 (0.0192 + 0.114 f) exp(-(0.114 f)^1.1), f = 16, 8, 4, 2, 1. Per level: mean mapped
    sigma ('visible structure') and median log10 sigma / u."""
    L = (2 ** 32 - 1) * (Y - Y.min()) / max(Y.max() - Y.min(), 1e-30)
    f, sf = {}, 32.0
    for lev in range(5):
        sf /= 2
        mu = cv2.GaussianBlur(L, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)[5:-5, 5:-5]
        m2 = cv2.GaussianBlur(L * L, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)[5:-5, 5:-5]
        sig = np.sqrt(np.maximum(m2 - mu * mu, 0))
        csf = 100.0 * 2.6 * (0.0192 + 0.114 * sf) * np.exp(-(0.114 * sf) ** 1.1)
        u = 128 / (1.4 * csf)
        f[f"tmqi_vis{lev}"] = float(norm.cdf(sig, u, u / 3).mean())
        f[f"tmqi_logsig_rel{lev}"] = float(np.median(np.log10(sig / u + 1e-12)))
        h, w = (L.shape[0] // 2) * 2, (L.shape[1] // 2) * 2
        L = L[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    return f


def _ggd(x):
    """GGD shape alpha and variance by moment matching (Mittal et al. 2012 / BRISQUE reference code)"""
    a = np.arange(0.2, 10.001, 0.001)
    r = G(1 / a) * G(3 / a) / G(2 / a) ** 2
    s2 = np.mean(x ** 2)
    rho = s2 / (np.mean(np.abs(x)) ** 2 + 1e-12)
    return a[np.argmin((r - rho) ** 2)], s2


def _aggd(x):
    """AGGD shape, mean, left / right variances (BRISQUE reference code)"""
    a = np.arange(0.2, 10.001, 0.001)
    r = G(2 / a) ** 2 / (G(1 / a) * G(3 / a))
    l, rr = x[x < 0], x[x >= 0]
    ls, rs = np.sqrt(np.mean(l ** 2)) if l.size else 1e-6, np.sqrt(np.mean(rr ** 2)) if rr.size else 1e-6
    gam = ls / rs
    rhat = np.mean(np.abs(x)) ** 2 / (np.mean(x ** 2) + 1e-12)
    rhatn = rhat * (gam ** 3 + 1) * (gam + 1) / (gam ** 2 + 1) ** 2
    alpha = a[np.argmin((r - rhatn) ** 2)]
    const = np.sqrt(G(1 / alpha) / G(3 / alpha))
    mean = (rs - ls) * G(2 / alpha) / G(1 / alpha) * const
    return alpha, mean, ls ** 2, rs ** 2


def f_brisque(rgb, Y):
    """Mittal et al. 2012 BRISQUE: MSCN coefficients (7x7 Gaussian window, sigma 7/6, C = 1); GGD (alpha,
    sigma^2) of MSCN, AGGD (shape, mean, left / right variance) of products with the 4 neighbours (H, V,
    D1, D2); 18 features per scale, 2 scales. Deviation: computed on the HDR's log luminance mapped to
    [0, 255] (BRISQUE is defined on 8-bit luminance)."""
    x = to_8bit_log(Y)
    f = {}
    for sc in range(2):
        mu = cv2.GaussianBlur(x, (7, 7), 7 / 6)
        sd = np.sqrt(np.abs(cv2.GaussianBlur(x * x, (7, 7), 7 / 6) - mu * mu))
        m = (x - mu) / (sd + 1)
        f[f"bq{sc}_alpha"], f[f"bq{sc}_var"] = _ggd(m.ravel())
        for name, sh in (("h", (0, 1)), ("v", (1, 0)), ("d1", (1, 1)), ("d2", (1, -1))):
            prod = (m * np.roll(np.roll(m, -sh[0], 0), -sh[1], 1))[1:-1, 1:-1].ravel()
            a, mn, lv, rv = _aggd(prod)
            f.update({f"bq{sc}_{name}_alpha": a, f"bq{sc}_{name}_mean": mn, f"bq{sc}_{name}_lvar": lv,
                      f"bq{sc}_{name}_rvar": rv})
        x = cv2.resize(x, (x.shape[1] // 2, x.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return f


def f_si(rgb, Y):
    """ITU-T P.910 spatial perceptual information: std of the Sobel magnitude. Deviation: on the HDR's log
    luminance mapped to [0, 255] instead of an 8-bit video luma."""
    x = to_8bit_log(Y)
    gx, gy = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3)
    return {"si_p910": float(np.hypot(gx, gy)[1:-1, 1:-1].std())}


BANDS = [(-np.inf, -3), (-3, -1), (-1, 1), (1, 3), (3, np.inf)]


def f_tmqiblock(rgb, Y):
    """Non-overlapping b x b blocks (b = 11, TMQI's naturalness block; also 22, 44) of log2 luminance
    relative to its median: block-std mean / percentiles, block range, share of high-contrast blocks,
    block mean-std correlation, std of block means, and the 11-px block std per brightness band
    (stops relative to the median). Adaptation of TMQI's LDR block statistic to the HDR."""
    l2 = np.log2(np.maximum(Y, pos_floor(Y)))
    r = l2 - np.median(l2)
    f = {}
    for b in (11, 22, 44):
        h, w = (r.shape[0] // b) * b, (r.shape[1] // b) * b
        v = r[:h, :w].reshape(h // b, b, w // b, b).swapaxes(1, 2).reshape(-1, b * b)
        mu, sd, rg = v.mean(1), v.std(1), v.max(1) - v.min(1)
        f[f"k{b}_sd_mean"] = sd.mean()
        f[f"k{b}_sd_p10"], f[f"k{b}_sd_p50"], f[f"k{b}_sd_p90"] = np.percentile(sd, [10, 50, 90])
        f[f"k{b}_range_mean"] = rg.mean()
        f[f"k{b}_frac_sd_gt05"], f[f"k{b}_frac_sd_gt1"] = (sd > 0.5).mean(), (sd > 1.0).mean()
        f[f"k{b}_mean_sd_corr"] = np.corrcoef(mu, sd)[0, 1] if sd.std() > 0 else 0.0
        f[f"k{b}_blockmean_std"] = mu.std()
        if b == 11:
            for i, (a, c) in enumerate(BANDS):
                m = (mu >= a) & (mu < c)
                f[f"k11_b{i}_sd"] = sd[m].mean() if m.sum() >= 5 else np.nan
                f[f"k11_b{i}_share"] = m.mean()
    return f


GROUPS = {"hulusic17": f_hulusic17, "pouli10": f_pouli10, "reinhard02": f_reinhard02, "drago03": f_drago03,
          "durand02": f_durand02, "fattal02": f_fattal02, "reinhard05": f_reinhard05,
          "kimkautz08": f_kimkautz08, "mai11": f_mai11, "mantiuk08": f_mantiuk08, "tmqi": f_tmqi,
          "brisque": f_brisque, "si": f_si, "tmqiblock": f_tmqiblock}


def process(img, groups):
    rgb, Y = load(img)
    out, tim = {}, {"image": img}
    for g in groups:
        t = time.perf_counter()
        out[g] = {"image": img, **GROUPS[g](rgb, Y)}
        tim[g] = time.perf_counter() - t
    return out, tim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--groups", nargs="*", default=list(GROUPS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    images = pd.read_csv(os.path.join(ROOT, "results", "q_matrix.csv")).image.tolist()[:args.limit]
    from joblib import Parallel, delayed
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs)(delayed(process)(i, args.groups) for i in images)
    sfx = "" if args.limit is None else "_test"
    for g in args.groups:
        pd.DataFrame([r[0][g] for r in res]).to_csv(os.path.join(OUT, f"{g}{sfx}.csv"), index=False)
    tim = pd.DataFrame([r[1] for r in res])
    tim.to_csv(os.path.join(OUT, f"timing{sfx}.csv"), index=False)
    print(f"{len(images)} images in {time.time() - t0:.0f}s")
    print(tim.drop(columns="image").median().round(4).to_string())


if __name__ == "__main__":
    main()
