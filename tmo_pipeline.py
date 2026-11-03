#!/usr/bin/env python3
"""
TMO selection pipeline for HDR TMO research.

Pipeline:
  1) Apply N tone-mapping operators (TMOs) to M HDR images  -> M*N LDR PNGs
  2) Score every (HDR, LDR) pair with TMQI (Q, plus S and N components)
  3) Aggregate per-TMO (mean / median / std) and rank to select the top-K

Classical operators run via Docker (luminance-hdr-cli inside Ubuntu container).
OpenCV operators run natively in Python.
TMQI scoring runs natively in Python.

Setup:

Usage:
  1. Put your 10 HDR/EXR files in the folder set by HDR_DIR below.
  2. Install your TMQI implementation (see compute_tmqi()).
  3. Run: python3 tmo_tmqi_pipeline.py

NOTHING here is tuned: every operator uses library defaults so the
comparison is fair. Document any deviation in your methods section.
"""

import os
import re
import csv
import glob
import time
import shutil
import tempfile
import subprocess
import statistics as stats
from collections import defaultdict

import numpy as np
import cv2

import att_tmo
import tmo_metrics


# ----------------------------------------------------------------------------
# TMQI COMPATIBILITY SHIMS
# The tmqi-revised package targets an older scientific-Python stack. Two small
# shims let it run unchanged on Python 3.13 + numpy 2.x:
#   1. scipy moved signal.gaussian -> signal.windows.gaussian.
#   2. TMQI depends on PyContracts (abandoned, broken on numpy 2.x) only for
#      input-type validation via @contract. We replace `contracts` with a no-op
#      stub: validation is skipped, but TMQI's numerical output is unchanged.
# Must run BEFORE `from TMQI...` (see compute_tmqi).
# ----------------------------------------------------------------------------
def _install_tmqi_compat():
    import sys
    import types

    import scipy.signal
    if not hasattr(scipy.signal, "gaussian"):
        scipy.signal.gaussian = scipy.signal.windows.gaussian

    if "contracts" not in sys.modules:
        stub = types.ModuleType("contracts")

        def contract(*args, **kwargs):
            # Support both @contract and @contract(x='spec') usage.
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda func: func

        stub.contract = contract
        sys.modules["contracts"] = stub


_install_tmqi_compat()


# ----------------------------------------------------------------------------
# CONFIG -- edit these to match your folder structure
# ----------------------------------------------------------------------------
HDR_DIR     = "images"          # folder holding your 10 .hdr / .exr files
OUT_DIR     = "tmo_outputs"     # where LDR PNGs get written
RESULTS_CSV = "tmqi_results.csv"
TOP_K       = 10                # how many TMOs to select at the end

DOCKER_IMAGE = "lhdr"           # name you gave when running docker build -t

HDR_GLOBS = ("*.hdr", "*.exr")

# ----------------------------------------------------------------------------
# DEEP-LEARNING TMO CONFIG
# Each DL operator runs the upstream repo's own inference in its OWN venv via
# subprocess (see third_party/<repo>), then we collect its LDR PNG. Toggle each
# operator here. See SETUP.md for cloning / checkpoint / venv instructions.
# ----------------------------------------------------------------------------
_HERE        = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY  = os.path.join(_HERE, "third_party")

ENABLE_CATMO   = True           # CATMO  (github.com/xslidi/CATMO)
ENABLE_UNCLTMO = True           # UnCLTMO (github.com/cao-cong/UnCLTMO)
ENABLE_UNPAIRED_HDR = True      # Unpaired-HDR (github.com/yael-vinker/unpaired_hdr_tmo)

# Optional GLOBAL max resolution cap (long edge, px), applied identically to
# every operator's input AND to the HDR used for TMQI scoring. None = full res.
# Use this if full-res 4K inputs OOM at 18 GB. Off by default.
MAX_LONG_EDGE = None


# ----------------------------------------------------------------------------
# TMO REGISTRY
# Each entry: (name, kind, spec)
#   kind="opencv"  -> spec is a callable(hdr_bgr_float32) -> ldr_bgr_uint8
#   kind="python"  -> same contract as "opencv", for operators ported to Python
#                     here (att_tmo.py); a separate kind so `--exclude-kinds
#                     opencv` does not drop them
#   kind="lhdr"    -> spec is the operator name string for luminance-hdr-cli
#                     legal values: ashikhmin drago durand fattal ferradans
#                                   ferwerda kimkautz lischinski mai
#                                   mantiuk06 mantiuk08 pattanaik
#                                   reinhard02 reinhard05 vanhateren
#                     (ferradans and ferwerda are commented out below, with
#                      reasons; reinhard_ocv is likewise excluded -- see there)
# 19 operators are registered: 2 opencv + 1 python + 13 lhdr + 3 dl (the
# benchmark sweep runs with --exclude-kinds opencv, i.e. 17 operators).
# ----------------------------------------------------------------------------
def _ocv(make_tmo):
    """Wrap an OpenCV tonemapper: takes BGR float32, returns BGR uint8."""
    def run(hdr_bgr):
        tmo = make_tmo()
        ldr = tmo.process(hdr_bgr)               # float32, ~[0,1]
        ldr = np.clip(ldr * 255.0, 0, 255).astype(np.uint8)
        return ldr
    return run

TMO_REGISTRY = [
    # -- OpenCV native (no Docker needed) ------------------------------------
    # ("reinhard_ocv", "opencv", _ocv(lambda: cv2.createTonemapReinhard(1.0))),
    #   excluded: TMQI's structural term returns a negative local score on this
    #   operator's output for 36/1005 images; raising it to a fractional exponent
    #   yields NaN, so Q is undefined there (the LDR itself is well-formed -- this
    #   is a metric failure, not an operator failure). Under the all-operators-
    #   must-score rule those images would be dropped for every operator, so the
    #   operator is excluded instead of the images. cv2's Reinhard is also the
    #   least faithful of the three OpenCV TMOs to its published formulation.
    ("drago_ocv",    "opencv", _ocv(lambda: cv2.createTonemapDrago(1.0, 0.7))),
    ("mantiuk_ocv",  "opencv", _ocv(lambda: cv2.createTonemapMantiuk(2.2, 0.85))),

    # -- luminance-hdr-cli via Docker ----------------------------------------
    ("reinhard02",  "lhdr", "reinhard02"),
    ("reinhard05",  "lhdr", "reinhard05"),
    ("drago",       "lhdr", "drago"),
    ("durand",      "lhdr", "durand"),
    ("fattal",      "lhdr", "fattal"),
    ("mantiuk06",   "lhdr", "mantiuk06"),
    ("mantiuk08",   "lhdr", "mantiuk08"),
    ("pattanaik",   "lhdr", "pattanaik"),
    ("ashikhmin",   "lhdr", "ashikhmin"),
    ("mai",         "lhdr", "mai"),
    ("lischinski",  "lhdr", "lischinski"),
    ("vanhateren",  "lhdr", "vanhateren"),
    ("kimkautz",    "lhdr", "kimkautz"),
    # ("ferwerda", "lhdr", "ferwerda"),  # excluded: CLI default adapt_lum=0.5 crushes
    #   scenes to black; a usable value is a per-image exposure knob (no auto mode in
    #   the CLI), which would break the library-defaults fairness rule.
    # ("ferradans", "lhdr", "ferradans"),  # disabled: ~2 min/image, looks hung but isn't

    # -- ported to Python (no Docker needed) ---------------------------------
    ("att",         "python", att_tmo.tonemap_bgr),   # Khan et al. 2018, see att_tmo.py

    # Pretrained DL operators (kind="dl") are appended below, after their adapter
    # functions are defined -- see "register DL operators".
]


# ----------------------------------------------------------------------------
# HDR LOADING
# ----------------------------------------------------------------------------
def load_hdr(path):
    """Load .hdr or .exr as float32 BGR (linear radiance, full dynamic range)."""
    hdr = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if hdr is None:
        raise IOError(f"Could not read HDR file: {path}")
    return hdr.astype(np.float32)


def cap_hdr(hdr_bgr, hdr_path):
    """
    Optional GLOBAL resolution cap (MAX_LONG_EDGE), applied identically to every
    operator. Returns (eff_hdr_bgr, eff_hdr_path, tmpdir):
      - cap off / image already within cap -> originals unchanged, tmpdir=None.
      - capping needed -> a downscaled array (for opencv + scoring) AND a capped
        temp .hdr file (for file-based operators: lhdr Docker + DL subprocess),
        so all operators see the SAME reduced-resolution linear HDR. Caller must
        shutil.rmtree(tmpdir) when done with the image.
    """
    if not MAX_LONG_EDGE:
        return hdr_bgr, hdr_path, None
    h, w = hdr_bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge <= MAX_LONG_EDGE:
        return hdr_bgr, hdr_path, None
    s = MAX_LONG_EDGE / float(long_edge)
    capped = cv2.resize(hdr_bgr, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                        interpolation=cv2.INTER_AREA)
    tmpdir = tempfile.mkdtemp(prefix="tmo_cap_")
    capped_path = os.path.join(tmpdir, os.path.basename(hdr_path))
    # Radiance .hdr keeps the linear float values for the file-based operators.
    cv2.imwrite(capped_path, capped)
    return capped, capped_path, tmpdir


# ----------------------------------------------------------------------------
# TMO APPLICATION
# ----------------------------------------------------------------------------
def apply_opencv(spec, hdr_bgr, out_png):
    ldr_bgr = spec(hdr_bgr)
    cv2.imwrite(out_png, ldr_bgr)


# luminance-hdr-cli writes these operators' output -- linear display luminance --
# straight to the 8-bit PNG: the CLI's --postgamma defaults to 1 and the operators
# apply no transfer curve themselves (pfstools: "results of this TMO require gamma
# correction"). durand and mantiuk06 (sRGB curve inside luminance-hdr), mantiuk08
# (display model) and mai (perceptual curve) are already display-encoded; vanhateren
# is left as is. See experiments/reinhard_check/FINDINGS.md.
LHDR_POSTGAMMA = {op: 2.2 for op in (
    "reinhard02", "reinhard05", "drago", "fattal",
    "pattanaik", "ashikhmin", "lischinski", "kimkautz")}


def apply_lhdr(op, hdr_path, out_png):
    """Run luminance-hdr-cli inside Docker, mounting source and output dirs."""
    hdr_abs = os.path.abspath(hdr_path)
    out_abs = os.path.abspath(out_png)
    hdr_dir = os.path.dirname(hdr_abs)
    out_dir = os.path.dirname(out_abs)

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{hdr_dir}:/input",
        "-v", f"{out_dir}:/output",
        DOCKER_IMAGE,
        "-l", f"/input/{os.path.basename(hdr_abs)}",
        "--tmo", op,
    ]
    if op in LHDR_POSTGAMMA:
        cmd += ["--postgamma", str(LHDR_POSTGAMMA[op])]
    cmd += ["-o", f"/output/{os.path.basename(out_abs)}"]
    subprocess.run(cmd, check=True, capture_output=True)


# ----------------------------------------------------------------------------
# DEEP-LEARNING TMO ADAPTERS
# Each adapter runs the upstream repo's OWN test script in its OWN venv via
# subprocess (isolating colliding models/options/util packages and pins), then
# copies the produced LDR PNG to out_png. We reuse each repo's HDR reading and
# normalization untouched -- we only wrap it. Output resolution may differ from
# the input; compute_tmqi() resizes the scoring HDR to match (see there).
# ----------------------------------------------------------------------------
CATMO_DIR    = os.path.join(THIRD_PARTY, "CATMO")
UNCLTMO_DIR  = os.path.join(THIRD_PARTY, "UnCLTMO")
UNPAIRED_HDR_DIR = os.path.join(THIRD_PARTY, "unpaired_hdr")
UNCLTMO_MODEL_NAME = ("11_08_lr15D_size268_D_[1,1,1]_pad_0_G_ssr_doubleConvT__"
                      "d1.0_struct_1.0[1,1,1]__trans2_replicate__noframe__min_log_0.1hist_fit_")


def _venv_python(repo_dir):
    """Path to the repo's own virtualenv python; error clearly if missing."""
    py = os.path.join(repo_dir, ".venv", "bin", "python")
    if not os.path.exists(py):
        raise FileNotFoundError(
            f"venv python not found at {py}. See SETUP.md to create the repo venv.")
    return py


def _run_dl_subprocess(name, cmd, cwd):
    """
    Run a DL repo's inference. Forces MPS with CPU fallback. Logs device,
    seconds, and any ops that fell back to CPU. Raises on non-zero exit.
    """
    env = dict(os.environ)
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"   # unsupported ops -> CPU, no crash
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-15:])
        raise RuntimeError(f"{name} inference failed (exit {proc.returncode}):\n{tail}")
    # device + CPU-fallback diagnostics. Both repos select MPS when available
    # (patched), else CPU; reflect that, and surface any ops that fell back.
    device = "cpu" if "Torch not compiled with CUDA" in out else "mps"
    fallbacks = sorted(set(re.findall(r"operator '([^']+)' is not currently supported on the MPS", out)))
    fb = f"; CPU-fallback ops: {', '.join(fallbacks)}" if fallbacks else ""
    print(f"    [{name}] device={device}  {elapsed:.1f}s/img{fb}")


def apply_catmo(hdr_path, out_png):
    """CATMO: CUT/pix2pix attention-UNet. Produces full-RGB LDR (revise_B)."""
    ckpt = os.path.join(CATMO_DIR, "checkpoints", "CATMO", "latest_net_G.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"CATMO checkpoint missing at {ckpt}. See SETUP.md (copy latest_net_G.pth).")
    py = _venv_python(CATMO_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        testA = os.path.join(tmp, "testA")
        os.makedirs(testA)
        shutil.copy(hdr_path, os.path.join(testA, os.path.basename(hdr_path)))
        results = os.path.join(tmp, "results")
        cmd = [py, "test.py",
               "--dataroot", testA, "--name", "CATMO", "--model", "test",
               "--netG", "aunet", "--gpu_ids", "-1",
               "--results_dir", results, "--num_test", "1"]
        _run_dl_subprocess("catmo", cmd, cwd=CATMO_DIR)
        # CATMO writes {results}/CATMO/test_latest/images/{id}_revise_B.png
        img_id = os.path.splitext(os.path.basename(hdr_path))[0]
        produced = os.path.join(results, "CATMO", "test_latest", "images",
                                f"{img_id}_revise_B.png")
        if not os.path.exists(produced):
            raise FileNotFoundError(f"CATMO produced no output for {img_id} (looked for {produced})")
        _save_as_png(produced, out_png)


def apply_uncltmo(hdr_path, out_png):
    """UnCLTMO (IVTMNet): full-resolution image TMO. Produces full-RGB LDR."""
    weights = os.path.join(UNCLTMO_DIR, "activate_trained_model",
                           "model_weights_imageTMO", "net_epoch5_iter62.pth")
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"UnCLTMO weights missing at {weights}. See SETUP.md (download from Google Drive).")
    py = _venv_python(UNCLTMO_DIR)
    run_dir = os.path.join(UNCLTMO_DIR, "activate_trained_model")
    with tempfile.TemporaryDirectory() as tmp:
        indir = os.path.join(tmp, "input")
        outdir = os.path.join(tmp, "out")
        lamdir = os.path.join(tmp, "lambda")
        os.makedirs(indir); os.makedirs(outdir); os.makedirs(lamdir)
        shutil.copy(hdr_path, os.path.join(indir, os.path.basename(hdr_path)))
        cmd = [py, "test_imageTMO.py",
               "--model_name", UNCLTMO_MODEL_NAME,
               "--input_images_path", indir,
               "--output_path", outdir,
               "--model_path", "model_weights_imageTMO",
               "--f_factor_path", "lambda_data/input_images_lambdas_HDRSdataset.npy",
               "--mean_hist_path", "lambda_data/ldr_avg_hist_900_images_20_bins.npy",
               "--lambda_output_path", lamdir, "--bins", "20"]
        _run_dl_subprocess("uncltmo", cmd, cwd=run_dir)
        # UnCLTMO writes one PNG per input into outdir; grab the one for this id.
        img_id = os.path.splitext(os.path.basename(hdr_path))[0]
        cands = [p for p in glob.glob(os.path.join(outdir, "*.png"))
                 if os.path.basename(p).startswith(img_id)]
        if not cands:
            raise FileNotFoundError(f"UnCLTMO produced no output for {img_id} in {outdir}")
        _save_as_png(cands[0], out_png)


def apply_unpaired_hdr(hdr_path, out_png):
    """Unpaired-HDR (Vinker et al.): UNet GAN TMO. Produces full-RGB LDR.

    Runs the repo's own activate_trained_model/run_trained_model.py in its venv.
    The repo applies its own per-image range-normalizing curve (f_factor/lambda,
    computed on the fly) and reads the HDR itself -- we only wrap it.
    """
    weights = os.path.join(UNPAIRED_HDR_DIR, "activate_trained_model",
                           "model_weights", "trained_weights.pth")
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"Unpaired-HDR weights missing at {weights}. See SETUP.md (repo ships them in-tree).")
    py = _venv_python(UNPAIRED_HDR_DIR)
    run_dir = os.path.join(UNPAIRED_HDR_DIR, "activate_trained_model")
    with tempfile.TemporaryDirectory() as tmp:
        indir = os.path.join(tmp, "input")
        outdir = os.path.join(tmp, "out")
        lamdir = os.path.join(tmp, "lambda")
        os.makedirs(indir); os.makedirs(outdir); os.makedirs(lamdir)
        shutil.copy(hdr_path, os.path.join(indir, os.path.basename(hdr_path)))
        cmd = [py, "run_trained_model.py",
               "--input_images_path", indir,
               "--output_path", outdir,
               "--model_path", "model_weights",
               "--f_factor_path", "lambda_data/exr_hist_dict_20_bins.npy",
               "--mean_hist_path", "lambda_data/ldr_avg_hist_900_images_20_bins.npy",
               "--lambda_output_path", lamdir, "--bins", "20"]
        _run_dl_subprocess("unpaired_hdr", cmd, cwd=run_dir)
        # Writes one PNG per input: {id}_fake_clamp_and_stretch.png. Grab this id's.
        img_id = os.path.splitext(os.path.basename(hdr_path))[0]
        cands = [p for p in glob.glob(os.path.join(outdir, "*.png"))
                 if os.path.basename(p).startswith(img_id)]
        if not cands:
            raise FileNotFoundError(f"Unpaired-HDR produced no output for {img_id} in {outdir}")
        _save_as_png(cands[0], out_png)


def _save_as_png(src_png, out_png):
    """Re-encode an LDR PNG through the SAME 8-bit writer used for all operators
    (uniform output encoding for the benchmark)."""
    im = cv2.imread(src_png)            # BGR uint8
    if im is None:
        raise IOError(f"Could not read DL output PNG: {src_png}")
    cv2.imwrite(out_png, im)


# -- register DL operators (after adapter fns exist) -------------------------
# Appended so they flow through the same produce_ldr/compute_tmqi/ranking path
# with zero downstream special-casing.
if ENABLE_CATMO:
    TMO_REGISTRY.append(("catmo",   "dl", apply_catmo))
if ENABLE_UNCLTMO:
    TMO_REGISTRY.append(("uncltmo", "dl", apply_uncltmo))
if ENABLE_UNPAIRED_HDR:
    TMO_REGISTRY.append(("unpaired_hdr", "dl", apply_unpaired_hdr))


def produce_ldr(name, kind, spec, hdr_path, hdr_bgr, out_png):
    """
    Generate one tone-mapped PNG.
    Skips generation if the file already exists (allows reruns).
    Returns True if the output looks valid, False if degenerate.
    """
    if os.path.exists(out_png) and os.path.getsize(out_png) > 0:
        pass  # reuse existing output
    elif kind in ("opencv", "python"):
        apply_opencv(spec, hdr_bgr, out_png)
    elif kind == "lhdr":
        apply_lhdr(spec, hdr_path, out_png)
    elif kind == "dl":
        spec(hdr_path, out_png)   # spec is the adapter fn (hdr_path, out_png)
    else:
        raise ValueError(f"Unknown TMO kind: {kind}")

    # Sanity check: reject all-black or blown-out outputs
    ldr = cv2.imread(out_png)
    if ldr is None:
        print(f"    !! {name}: output unreadable")
        return False
    m = float(ldr.mean())
    if m < 1.0 or m > 254.0:
        print(f"    !! {name}: degenerate output (mean pixel={m:.1f})")
        return False
    return True


# ----------------------------------------------------------------------------
# SCORING
# Every metric scores the same aligned (HDR, LDR) pair, built in ONE place.
# ----------------------------------------------------------------------------
# Operators that CENTER-CROP their input instead of resizing it. CATMO crops to
# multiples of 32 px (third_party/CATMO/data/base_dataset.py adjust2test), so
# its reference must be cropped the same way: squashing the full HDR to the
# output size misaligns the pixels and under-scores CATMO. Unpaired-HDR resizes
# its input (bicubic, utils/data_loader_util.py resize_im), so resizing the
# reference is the right match for it and for any other size change.
CENTER_CROP_OPERATORS = {"catmo"}

METRIC_COLUMNS = ["Q", "S", "N", "FSITM", "FSITM_TMQI", "Q2", "S2", "N2"]


def _scoring_pair(hdr_bgr_float, ldr_png_path, operator=None):
    """(hdr_rgb float32, ldr_rgb float32) at the LDR's resolution, aligned."""
    hdr_rgb = cv2.cvtColor(hdr_bgr_float, cv2.COLOR_BGR2RGB)
    ldr = cv2.imread(ldr_png_path)
    if ldr is None:
        raise IOError(f"Cannot read LDR for scoring: {ldr_png_path}")
    # Cast to float: TMQI's luminance math expects float arrays (uint8 overflows).
    ldr_rgb = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB).astype(np.float32)

    (H, W), (h, w) = hdr_rgb.shape[:2], ldr_rgb.shape[:2]
    if (H, W) != (h, w):
        if operator in CENTER_CROP_OPERATORS and h <= H and w <= W:
            y, x = (H - h) // 2, (W - w) // 2
            hdr_rgb = hdr_rgb[y:y + h, x:x + w]
        else:
            hdr_rgb = cv2.resize(hdr_rgb, (w, h), interpolation=cv2.INTER_AREA)
    return hdr_rgb, ldr_rgb


def compute_tmqi(hdr_bgr_float, ldr_png_path, operator=None):
    """
    Score one (HDR, LDR) pair with TMQI (tmqi-revised, a port of the original
    MATLAB). Returns (Q, S, N): overall quality, structural fidelity,
    statistical naturalness.
    """
    from TMQI.TMQI import TMQI  # noqa: E402  (class lives in the TMQI submodule)

    hdr_rgb, ldr_rgb = _scoring_pair(hdr_bgr_float, ldr_png_path, operator)
    result = TMQI()(hdr_rgb, ldr_rgb)   # returns (Q, S, N, ...)
    return float(result[0]), float(result[1]), float(result[2])


def compute_scores(hdr_bgr_float, ldr_png_path, operator=None):
    """
    All metrics for one pair, keyed by METRIC_COLUMNS:
      Q, S, N      TMQI (Yeganeh & Wang 2013)
      FSITM        FSITM, green channel (Nafchi et al. 2015)
      FSITM_TMQI   (FSITM + TMQI Q) / 2, the combined index from the FSITM paper
      Q2, S2, N2   TMQI-II (Ma et al. 2015)
    """
    from TMQI.TMQI import TMQI  # noqa: E402

    hdr_rgb, ldr_rgb = _scoring_pair(hdr_bgr_float, ldr_png_path, operator)
    t = TMQI()(hdr_rgb, ldr_rgb)
    Q, S, N = float(t[0]), float(t[1]), float(t[2])
    F = tmo_metrics.fsitm(hdr_rgb, ldr_rgb, ch=1)
    Q2, S2, N2 = tmo_metrics.tmqi2(hdr_rgb, ldr_rgb)
    return dict(Q=Q, S=S, N=N, FSITM=F, FSITM_TMQI=(F + Q) / 2, Q2=Q2, S2=S2, N2=N2)


# ----------------------------------------------------------------------------
# DRIVER
# ----------------------------------------------------------------------------
def find_hdr_images():
    paths = []
    for g in HDR_GLOBS:
        paths.extend(sorted(glob.glob(os.path.join(HDR_DIR, g))))
    return paths


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    hdr_paths = find_hdr_images()
    if not hdr_paths:
        raise SystemExit(
            f"No HDR files found in '{HDR_DIR}/' "
            f"(looked for {HDR_GLOBS}). Check your HDR_DIR setting."
        )

    print(f"Found {len(hdr_paths)} images x {len(TMO_REGISTRY)} TMOs "
          f"= {len(hdr_paths) * len(TMO_REGISTRY)} outputs\n")

    rows = []  # (image_id, tmo_name, Q, S, N)

    for hp in hdr_paths:
        img_id = os.path.splitext(os.path.basename(hp))[0]
        print(f"[{img_id}]")

        try:
            hdr_bgr = load_hdr(hp)
        except IOError as e:
            print(f"  !! Could not load HDR: {e}")
            continue

        # Apply the optional global resolution cap once per image, identically
        # for every operator (input file + scoring HDR). No-op when cap is off.
        hdr_bgr, hp_eff, cap_tmp = cap_hdr(hdr_bgr, hp)
        try:
            for name, kind, spec in TMO_REGISTRY:
                out_png = os.path.join(OUT_DIR, f"{img_id}__{name}.png")
                try:
                    ok = produce_ldr(name, kind, spec, hp_eff, hdr_bgr, out_png)
                    if not ok:
                        continue
                    Q, S, N = compute_tmqi(hdr_bgr, out_png, name)
                    rows.append((img_id, name, Q, S, N))
                    print(f"    {name:<14s}  Q={Q:.4f}  S={S:.4f}  N={N:.4f}")
                except subprocess.CalledProcessError as e:
                    print(f"    !! {name}: Docker/CLI call failed — {e}")
                except Exception as e:
                    print(f"    !! {name}: {e}")
        finally:
            if cap_tmp:
                shutil.rmtree(cap_tmp, ignore_errors=True)

    # -- Write per-pair results CSV ------------------------------------------
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "tmo", "TMQI", "S", "N"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {RESULTS_CSV}")

    if not rows:
        print("No results to rank. Check errors above.")
        return

    # -- Aggregate per TMO ---------------------------------------------------
    by_tmo = defaultdict(list)
    for _, name, Q, _, _ in rows:
        by_tmo[name].append(Q)

    agg = []
    for name, qs in by_tmo.items():
        agg.append((
            name,
            stats.mean(qs),
            stats.median(qs),
            stats.pstdev(qs) if len(qs) > 1 else 0.0,
            min(qs),
            len(qs),
        ))
    agg.sort(key=lambda r: r[1], reverse=True)  # rank by mean TMQI

    # -- Print ranking table -------------------------------------------------
    print(f"\n{'Rank':<5} {'TMO':<16} {'Mean':>8} {'Median':>8} "
          f"{'Std':>8} {'Worst':>8} {'N':>4}")
    print("-" * 60)
    for i, (name, mean, med, sd, worst, n) in enumerate(agg, 1):
        mark = "  <-- selected" if i <= TOP_K else ""
        print(f"{i:<5} {name:<16} {mean:>8.4f} {med:>8.4f} "
              f"{sd:>8.4f} {worst:>8.4f} {n:>4}{mark}")

    # -- Save selected TMOs --------------------------------------------------
    selected = [r[0] for r in agg[:TOP_K]]
    with open("selected_tmos.txt", "w") as f:
        f.write("\n".join(selected) + "\n")

    print(f"\nTop {TOP_K} TMOs saved -> selected_tmos.txt")
    print("\nIMPORTANT: Visually inspect the outputs of selected TMOs before")
    print("committing — TMQI can score visually poor images highly in edge cases.")


if __name__ == "__main__":
    main()