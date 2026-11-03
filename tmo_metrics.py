"""
Full-reference tone-mapping quality metrics beyond TMQI, ported from the
authors' official MATLAB code (not from third-party reimplementations):

  FSITM   H. Z. Nafchi, A. Shahkolaei, R. F. Moghaddam, M. Cheriet, "FSITM: A
          feature similarity index for tone-mapped images," IEEE SPL 22(8),
          2015.  Source: FSITM.m, FSITM_TMQI.m, phasecong100.m and
          Lowpassfilter.m (P. Kovesi) from MATLAB File Exchange #59814.
          The paper's recommended channel is green (FSITM_G); FSITM_TMQI is
          the paper's combined index, (FSITM_G + TMQI) / 2.

  TMQI-II K. Ma, H. Yeganeh, K. Zeng, Z. Wang, "High dynamic range image
          compression by optimizing tone mapped image quality index," IEEE
          TIP 24(10), 2015.  Source: TMQI2.m from kedema.org/code/TMQI_II_Opt.rar.

Both take the HDR reference (linear RGB float, same height/width as the LDR)
and the LDR (RGB, 0-255). Alignment of the two is the caller's job.

Deviation: TMQI2.m divides the local HDR std by the local mean, which is 0/0 =
NaN (and makes S, and so Q, NaN) wherever a whole 11x11 window sits at the
image minimum (e.g. clipped black). Those windows have no visible contrast, so
their normalised std is set to 0 instead.
"""
import numpy as np
from scipy import ndimage
from scipy.stats import norm


# ----------------------------------------------------------------------------
# Kovesi phase congruency (feature type only), as in phasecong100.m
# ----------------------------------------------------------------------------
def _freq_range(n):
    """MATLAB: odd n -> (-(n-1)/2:(n-1)/2)/(n-1); even n -> (-n/2:n/2-1)/n."""
    if n % 2:
        return np.arange(-(n - 1) / 2, (n - 1) / 2 + 1) / (n - 1)
    return np.arange(-n / 2, n / 2) / n


def lowpassfilter(rows, cols, cutoff, n):
    x, y = np.meshgrid(_freq_range(cols), _freq_range(rows))
    radius = np.sqrt(x ** 2 + y ** 2)
    return np.fft.ifftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * n)))


def phasecong100(im, nscale=2, norient=2, min_wavelength=7, mult=2, sigma_onf=0.65):
    """Locally weighted mean phase angle: pi/2 white line, 0 step, -pi/2 black line."""
    im = np.asarray(im, dtype=np.float64)
    rows, cols = im.shape
    imagefft = np.fft.fft2(im)

    x, y = np.meshgrid(_freq_range(cols), _freq_range(rows))
    radius = np.fft.ifftshift(np.sqrt(x ** 2 + y ** 2))
    theta = np.fft.ifftshift(np.arctan2(-y, x))
    radius[0, 0] = 1.0
    sintheta, costheta = np.sin(theta), np.cos(theta)

    lp = lowpassfilter(rows, cols, 0.45, 15)
    log_gabor = []
    for s in range(nscale):
        fo = 1.0 / (min_wavelength * mult ** s)
        g = np.exp(-(np.log(radius / fo)) ** 2 / (2 * np.log(sigma_onf) ** 2)) * lp
        g[0, 0] = 0.0
        log_gabor.append(g)

    energy = np.zeros((rows, cols, 3))
    for o in range(norient):
        angl = o * np.pi / norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.minimum(np.abs(np.arctan2(ds, dc)) * norient / 2, np.pi)
        spread = (np.cos(dtheta) + 1) / 2
        sum_e = np.zeros((rows, cols))
        sum_o = np.zeros((rows, cols))
        for g in log_gabor:
            eo = np.fft.ifft2(imagefft * (g * spread))
            sum_e += eo.real
            sum_o += eo.imag
        energy[..., 0] += sum_e
        energy[..., 1] += np.cos(angl) * sum_o
        energy[..., 2] += np.sin(angl) * sum_o

    odd = np.sqrt(energy[..., 1] ** 2 + energy[..., 2] ** 2)
    return np.arctan2(energy[..., 0], odd)


# ----------------------------------------------------------------------------
# FSITM (FSITM.m)
# ----------------------------------------------------------------------------
def fsitm(hdr_rgb, ldr_rgb, ch=1):
    """Feature similarity index for one channel (0 R, 1 G, 2 B); higher is better."""
    rows, cols = ldr_rgb.shape[:2]
    num_pixels = rows * cols
    r = np.floor(num_pixels / 2 ** 18)
    alpha = 1 - 1 / r if r > 1 else 0.0

    hdr_ch = np.asarray(hdr_rgb[..., ch], dtype=np.float64)
    ldr_ch = np.asarray(ldr_rgb[..., ch], dtype=np.float64)

    log_h = hdr_ch.copy()
    log_h[hdr_ch == 0] = hdr_ch[hdr_ch != 0].min()
    log_h = np.log(log_h)
    span = log_h.max() - log_h.min()                       # im2uint8(mat2gray(.))
    log_h = (log_h - log_h.min()) / span if span > 0 else np.zeros_like(log_h)
    log_h = np.floor(np.clip(log_h, 0, 1) * 255 + 0.5)

    if alpha != 0:
        phase_hdr8 = phasecong100(hdr_ch, 2, 2, 8, 8)
        phase_ldr8 = phasecong100(ldr_ch, 2, 2, 8, 8)
    else:
        phase_hdr8 = phase_ldr8 = 0.0
    phase_h = alpha * phase_hdr8 + (1 - alpha) * phasecong100(log_h, 2, 2, 2, 2)
    phase_l = alpha * phase_ldr8 + (1 - alpha) * phasecong100(ldr_ch, 2, 2, 2, 2)

    same = ((phase_l <= 0) & (phase_h <= 0)) | ((phase_l > 0) & (phase_h > 0))
    return float(same.sum() / num_pixels)


# ----------------------------------------------------------------------------
# TMQI-II (TMQI2.m)
# ----------------------------------------------------------------------------
def _gaussian_window(size=11, sigma=1.5):
    """fspecial('gaussian', size, sigma)"""
    ax = np.arange(size) - (size - 1) / 2
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _filter_valid(window, x):
    """filter2(window, x, 'valid'): correlation, keeping only full-overlap positions."""
    h, w = window.shape
    full = ndimage.correlate(x, window, mode="constant", origin=0)
    t, l = (h - 1) // 2, (w - 1) // 2
    return full[t:x.shape[0] - (h - 1 - t), l:x.shape[1] - (w - 1 - l)]


def _bounded_cdf(act, est, bound, upper):
    """Gaussian CDF through (bound, 0.0001) and (est, 0.9999), mirrored at `bound`
    when the actual value is above the estimate."""
    y1, y2 = norm.ppf([0.0001, 0.9999])
    if not upper:
        sigma = (est - bound) / (y2 - y1)
        return norm.cdf(act, est - sigma * y2, sigma)
    hi = 2 * bound - est
    sigma = (hi - bound) / (y2 - y1)
    return norm.cdf(2 * bound - act, hi - sigma * y2, sigma)


def tmqi2(hdr_rgb, ldr_rgb):
    """Returns (Q, S, N); higher is better."""
    a, alpha, beta = 0.5, 1.0, 1.0
    hdr_th, ldr_th = 0.06, 2.6303
    k = 0.12
    slopes = (0.6043, 0.6993, 0.6504, 0.9386)
    intercepts = (-0.1402, 83.6128, -0.0759, 51.3915)
    c1, c2 = 0.01, 10.0
    window = _gaussian_window()

    hdr = np.asarray(hdr_rgb, dtype=np.float64)
    ldr = np.asarray(ldr_rgb, dtype=np.float64)
    hdr_l = 0.213 * hdr[..., 0] + 0.715 * hdr[..., 1] + 0.072 * hdr[..., 2]
    min_l, max_l = hdr_l.min(), hdr_l.max()
    hdr_l = np.round((2 ** 32 - 1) / (max_l - min_l)) * (hdr_l - min_l)   # full contrast stretch
    ldr_l = 0.213 * ldr[..., 0] + 0.715 * ldr[..., 1] + 0.072 * ldr[..., 2]

    # ---- structural fidelity
    mu_h = _filter_valid(window, hdr_l)
    mu_l = _filter_valid(window, ldr_l)
    sigma_h = np.sqrt(np.maximum(0, _filter_valid(window, hdr_l * hdr_l) - mu_h * mu_h))
    sigma_l = np.sqrt(np.maximum(0, _filter_valid(window, ldr_l * ldr_l) - mu_l * mu_l))
    sigma_hl = _filter_valid(window, hdr_l * ldr_l) - mu_h * mu_l

    cov_h = np.divide(sigma_h, mu_h, out=np.zeros_like(sigma_h), where=mu_h > 0)
    sigma_hp = norm.cdf(cov_h, hdr_th, hdr_th / 3)
    sigma_lp = norm.cdf(sigma_l, ldr_th, ldr_th / 3)
    s_map = ((2 * sigma_hp * sigma_lp + c1) / (sigma_hp ** 2 + sigma_lp ** 2 + c1)) \
        * ((sigma_hl + c2) / (sigma_h * sigma_l + c2))
    S = float(s_map.mean())

    # ---- statistical naturalness
    log_mu = np.exp(np.mean(np.log(hdr_l + np.finfo(float).eps)))
    hdr_s = k / log_mu * hdr_l
    est = (hdr_s / (1 + hdr_s)) ** (1 / 2.2) * 255
    est_mu, est_sigma = est.mean(), est.std(ddof=1)
    p_mu = _bounded_cdf(ldr_l.mean(), est_mu,
                        slopes[0] * est_mu + intercepts[0] if ldr_l.mean() <= est_mu
                        else slopes[1] * est_mu + intercepts[1],
                        upper=ldr_l.mean() > est_mu)
    act_sigma = ldr_l.std(ddof=1)
    p_sigma = _bounded_cdf(act_sigma, est_sigma,
                           slopes[2] * est_sigma + intercepts[2] if act_sigma <= est_sigma
                           else slopes[3] * est_sigma + intercepts[3],
                           upper=act_sigma > est_sigma)
    N = float(p_mu * p_sigma)
    Q = a * S ** alpha + (1 - a) * N ** beta
    return float(Q), S, N
