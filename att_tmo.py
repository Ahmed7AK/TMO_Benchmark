"""
ATT (Adaptive TVI tone mapping) -- Python port of the authors' MATLAB code
(design_tmo_att.m + apply_tmo.m, (C) 2018 Ishtiaq Rasool Khan; non-commercial
research use permitted by the authors).

  I. R. Khan, S. Rahardja, M. M. Khan, M. M. Movania, F. Abed, "A tone-mapping
  technique based on histogram using a sensitivity model of the human visual
  system," IEEE Trans. Industrial Electronics, 65(4), 3469-3479, 2018.

A global operator: a luminance histogram whose bins are equally spaced in
just-noticeable differences (TVI model), refined by dropping pixels within a
JND of each other, becomes a cumulative LUT mapping HDR luminance to [0, 1].

Faithful to the MATLAB except:
  * rand() in the bin-width search is replaced by a fixed-seed generator, so
    the operator is deterministic;
  * the bin-width guard is `n <= 0` instead of `n == 0`: the exact-zero test
    never fires in MATLAB, and a negative n would loop forever.

Colour uses the luminance-ratio path with saturation s = 0.67, the setting
recommended in apply_tmo.m ("gamma = 1/1.5") and used for all results in the
paper. example_code_tmo.m calls it with s = 1.
"""
import numpy as np

SCALE_FACTOR = 179.0      # radiance (W/sr/m^2) -> luminance (cd/m^2)
NBINS = (70, 90)          # accepted range for the number of bin edges
W1 = 0.8                  # weight of the raw vs JND-refined histogram
SATURATION = 0.67


def tvi(log_l):
    """log10 adaptation luminance -> log10 JND threshold (piecewise TVI model)."""
    x = np.asarray(log_l, dtype=np.float64)
    t = np.empty_like(x)
    t[x < -3.94] = -2.86
    m = (x >= -3.94) & (x < -1.44)
    t[m] = (0.405 * x[m] + 1.6) ** 2.18 - 2.86
    m = (x >= -1.44) & (x < -0.0184)
    t[m] = x[m] - 0.395
    m = (x >= -0.0184) & (x < 1.9)
    t[m] = (0.249 * x[m] + 0.65) ** 2.7 - 0.72
    m = x >= 1.9
    t[m] = x[m] - 1.255
    return t - 0.95


def _luminance(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def design_tmo_att(hdr_rgb, seed=0):
    """HDR RGB (linear, > 0) -> LUT (x: HDR luminance, y: display value in [0, 1])."""
    rng = np.random.default_rng(seed)
    hdr = SCALE_FACTOR * np.asarray(hdr_rgb, dtype=np.float64)
    lum = _luminance(hdr).ravel()
    # stretch the luminance extremes to the min/max over all channels
    lum[np.argmin(lum)] = hdr.min()
    lum[np.argmax(lum)] = hdr.max()
    lo, hi = lum.min(), lum.max()

    # bin edges n JNDs apart; adapt n until the edge count is within NBINS
    n = 16.0
    while True:
        bins = [lo]
        while bins[-1] < hi:
            bins.append(bins[-1] + n * 10 ** tvi(np.log10(bins[-1])))
        bins[-1] = hi
        if len(bins) < NBINS[0]:
            n = n - 2 - rng.random()
            if n <= 0:
                n = 0.1
        elif len(bins) > NBINS[1]:
            n = n + 1.5 + rng.random()
        else:
            break
    bins = np.asarray(bins)

    centers = np.log10(0.5 * (bins[1:] + bins[:-1]))
    jnd = 10 ** tvi(centers)
    # MATLAB hist(data, centers): edges halfway between centers, open outer bins
    edges = 0.5 * (centers[1:] + centers[:-1])
    counts = np.bincount(np.searchsorted(edges, np.log10(lum), side="right"),
                         minlength=len(centers)).astype(np.float64)

    # refined counts: repeatedly drop the second pixel of each sorted pair that
    # sits within one JND of its partner
    srt = np.sort(lum)
    rcounts = np.empty(len(centers))
    for i in range(len(centers)):
        a, b = np.searchsorted(srt, [bins[i], bins[i + 1]], side="left")
        pix = srt[a:b]
        while True:
            m = len(pix) // 2
            k = np.nonzero(pix[1:2 * m:2] - pix[0:2 * m:2] < jnd[i])[0]
            if k.size == 0:
                break
            pix = np.delete(pix, 2 * k + 1)
        rcounts[i] = max(len(pix), 1)   # MATLAB max(size(<0x1 empty>)) is 1

    counts = W1 * counts / counts.sum() + (1 - W1) * rcounts / rcounts.sum()
    w = np.concatenate([[0.0], np.cumsum(counts / counts.max())])
    return bins / SCALE_FACTOR, w / w.max()


def apply_tmo(hdr_rgb, x, y, s=SATURATION):
    """Map luminance through the LUT, then rescale RGB by the luminance ratio^s."""
    hdr = np.asarray(hdr_rgb, dtype=np.float64)
    in_lum = _luminance(hdr)
    out_lum = np.interp(in_lum, x, y)
    out = (hdr / in_lum[..., None]) ** s * out_lum[..., None]
    return np.clip(out, 0.0, 1.0)


def tonemap_bgr(hdr_bgr):
    """Pipeline entry point: float32 BGR HDR -> uint8 BGR LDR."""
    rgb = np.ascontiguousarray(hdr_bgr[..., ::-1], dtype=np.float64)
    rgb[rgb <= 0] = 1e-6                     # as in example_code_tmo.m
    x, y = design_tmo_att(rgb)
    ldr = apply_tmo(rgb, x, y)
    return (ldr[..., ::-1] * 255.0 + 0.5).astype(np.uint8)
