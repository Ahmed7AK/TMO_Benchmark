#!/usr/bin/env python3
"""
Robust overnight orchestrator for the TMO benchmark sweep.

Evaluates every operator in tmo_pipeline.TMO_REGISTRY over a folder of HDR
images (~20k tone-maps + 20k TMQI evals) in a way that is crash-proof,
resumable, and reproducible for paper-grade output.

It does NOT reimplement any pipeline logic. It imports and reuses tmo_pipeline's
own registry, HDR loader, resolution cap, LDR producer, and TMQI scorer, so the
fairness path (8-bit PNG -> Rec.601 luminance in TMQI -> compared against the
ORIGINAL linear HDR reference) is byte-for-byte identical to the single-image
pipeline. See README_sweep.md.

Run:        python3 sweep.py
Dry run:    python3 sweep.py --dry-run
Smoke test: python3 sweep.py --limit 5
"""
import os
import csv
import json
import time
import random
import shutil
import argparse
import subprocess
import statistics as stats
from datetime import timedelta
from collections import defaultdict

import numpy as np
import cv2

import tmo_pipeline as P   # reuse registry + load_hdr + cap_hdr + produce_ldr + compute_tmqi


# ----------------------------------------------------------------------------
# Output layout
# ----------------------------------------------------------------------------
# every CSV a sweep writes lives in results/
DIAG_DIR    = "results"
LOGS_DIR    = "results"
RESULTS_DIR = "results"

PREFLIGHT_CSV = os.path.join(DIAG_DIR, "preflight.csv")
FAILURES_CSV  = os.path.join(LOGS_DIR, "failures.csv")
PER_IMAGE_CSV = os.path.join(RESULTS_DIR, "per_image.csv")
SUMMARY_CSV   = os.path.join(RESULTS_DIR, "summary.csv")
RUN_CONFIG    = os.path.join(RESULTS_DIR, "run_config.json")

# metric columns come from tmo_pipeline.METRIC_COLUMNS (TMQI, FSITM, TMQI-II);
# tmo_seconds is blank when the LDR came from the tmo_outputs/ cache
PER_IMAGE_HEADER = (["operator", "image"] + P.METRIC_COLUMNS
                    + ["resolution_used", "tmo_seconds", "score_seconds", "status"])
_NO_SCORES = [""] * len(P.METRIC_COLUMNS)


def set_run_dir(root):
    """Redirect every output (results, logs, diagnostics, LDR PNGs) under `root`,
    so a second sweep never mixes rows with -- or reuses cached PNGs from -- an
    earlier run whose image ids may collide."""
    global DIAG_DIR, LOGS_DIR, RESULTS_DIR
    global PREFLIGHT_CSV, FAILURES_CSV, PER_IMAGE_CSV, SUMMARY_CSV, RUN_CONFIG
    DIAG_DIR = LOGS_DIR = RESULTS_DIR = os.path.join(root, "results")
    PREFLIGHT_CSV = os.path.join(DIAG_DIR, "preflight.csv")
    FAILURES_CSV  = os.path.join(LOGS_DIR, "failures.csv")
    PER_IMAGE_CSV = os.path.join(RESULTS_DIR, "per_image.csv")
    SUMMARY_CSV   = os.path.join(RESULTS_DIR, "summary.csv")
    RUN_CONFIG    = os.path.join(RESULTS_DIR, "run_config.json")
    P.OUT_DIR     = os.path.join(root, "tmo_outputs")


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def seed_everything(seed):
    """Seed host-side stochastic sources and propagate a seed to DL subprocesses.
    Each DL repo runs in its own venv/subprocess; tmo_pipeline._run_dl_subprocess
    forwards os.environ, so setting these here reaches the child interpreters."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["SWEEP_SEED"] = str(seed)


def _git_commit(repo_dir):
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        return None
    try:
        out = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def _pkg_versions():
    vers = {}
    for mod in ("numpy", "cv2", "scipy"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:
            vers[mod] = None
    return vers


def write_run_config(operators, valid_images, seed, max_long_edge):
    cfg = {
        "seed": seed,
        "max_long_edge": max_long_edge,
        "num_images": len(valid_images),
        "operators": [{"name": n, "kind": k} for (n, k, _) in operators],
        # luminance-hdr-cli --postgamma per lhdr operator (the rest: CLI default 1)
        "lhdr_postgamma": P.LHDR_POSTGAMMA,
        "repo_commits": {
            "CATMO": _git_commit(P.CATMO_DIR),
            "UnCLTMO": _git_commit(P.UNCLTMO_DIR),
            "unpaired_hdr": _git_commit(P.UNPAIRED_HDR_DIR),
        },
        "package_versions": _pkg_versions(),
        "input_manifest": [os.path.basename(p) for p in valid_images],
        "fairness_notes": (
            "All operators share tmo_pipeline.produce_ldr + compute_scores (TMQI, "
            "FSITM, TMQI-II). The reference is the ORIGINAL linear HDR (capped "
            "identically for all), center-cropped for CATMO (which crops its input "
            "to multiples of 32 px) and resized for any other size change. "
            "lhdr operators whose luminance-hdr output is linear display luminance "
            "get --postgamma 2.2 (lhdr_postgamma); durand, mantiuk06, mantiuk08 and "
            "mai are already display-encoded and keep the CLI default of 1. "
            "vanhateren also keeps 1, but its pixel encoding is unresolved: it shares "
            "the other lhdr operators' PNG container (no gAMA/sRGB/iCCP chunk, one "
            "PngWriter), which says nothing about the transfer curve of its pixel "
            "values. See experiments/reinhard_check/FINDINGS.md."
        ),
    }
    with open(RUN_CONFIG, "w") as f:
        json.dump(cfg, f, indent=2)


# ----------------------------------------------------------------------------
# Pre-flight input validation
# ----------------------------------------------------------------------------
# A dim-but-valid HDR (absolute max < this) is KEPT but flagged low_range, since
# a .hdr stores radiance in arbitrary units and absolute brightness is not a
# validity signal. Only genuinely blank/flat frames are excluded (see below).
LOW_RANGE_MAX = 1.0
# Degeneracy: a truly blank (all-black) or flat (single-colour) frame has no
# content for a tone-mapper. Coefficient of variation (std/mean of luminance)
# is scale-invariant, so this generalises across differently-scaled HDRs.
DEGENERATE_COV = 1e-4


def preflight(hdr_paths):
    """Validate every HDR before any tone-mapping. An image is VALID when:
      1. it reads back as a float array (the loader-bug guard: catches an HDR
         silently decoded as 8-bit LDR), and
      2. it has actual content -- not all-black (luminance max <= 0) and not a
         single flat colour (luminance CoV < DEGENERATE_COV).
    Absolute brightness is NOT used to exclude: dim-but-structured HDRs are kept
    and flagged low_range (abs max < LOW_RANGE_MAX) so they are never silently
    trusted. Writes results/preflight.csv. Returns the valid paths; invalid
    ones are excluded from the whole sweep."""
    os.makedirs(DIAG_DIR, exist_ok=True)
    valid, rows = [], []
    for p in hdr_paths:
        status, detail, w, h, mx, cov, warn = "valid", "", "", "", "", "", ""
        try:
            hdr = P.load_hdr(p)                       # float32 BGR via cv2 float path
            if not np.issubdtype(hdr.dtype, np.floating):
                raise ValueError(f"not a float array (dtype={hdr.dtype}); "
                                 "likely decoded as 8-bit LDR")
            h, w = hdr.shape[:2]
            mx = float(hdr.max())
            # Rec.601 luminance -- same weighting used downstream for scoring.
            lum = 0.114*hdr[:, :, 0] + 0.587*hdr[:, :, 1] + 0.299*hdr[:, :, 2]
            lmax = float(lum.max())
            cov = float(lum.std() / (lum.mean() + 1e-12))
            if lmax <= 0.0:
                raise ValueError("blank frame (luminance max <= 0)")
            if cov < DEGENERATE_COV:
                raise ValueError(f"flat frame (luminance CoV {cov:.2e} < {DEGENERATE_COV:g})")
            if mx < LOW_RANGE_MAX:
                warn = "low_range"     # kept, but flagged: dim absolute values
            valid.append(p)
            cov = f"{cov:.4f}"
        except Exception as e:
            status, detail = "invalid", str(e)
            if cov != "":
                cov = f"{cov:.4f}"
        rows.append([os.path.basename(p), p, status, detail, w, h, mx, cov, warn])

    with open(PREFLIGHT_CSV, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "path", "status", "detail", "width", "height",
                     "max_value", "lum_cov", "warning"])
        wr.writerows(rows)

    n_invalid = len(hdr_paths) - len(valid)
    n_lowrange = sum(1 for r in rows if r[8] == "low_range")
    print(f"Pre-flight: {len(valid)} valid, {n_invalid} invalid/blank "
          f"(excluded) -> {PREFLIGHT_CSV}")
    for r in rows:
        if r[2] == "invalid":
            print(f"    dropped {r[0]}: {r[3]}")
    if n_lowrange:
        print(f"    note: {n_lowrange} valid image(s) flagged low_range "
              f"(abs max < {LOW_RANGE_MAX}, kept): "
              + ", ".join(r[0] for r in rows if r[8] == "low_range"))
    return valid


# ----------------------------------------------------------------------------
# Resume state
# ----------------------------------------------------------------------------
def load_done_pairs():
    """(operator, image) pairs already recorded in per_image.csv, so a rerun
    resumes with no duplicate rows and no repeated TMQI work."""
    done = set()
    if os.path.exists(PER_IMAGE_CSV):
        with open(PER_IMAGE_CSV, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["operator"], row["image"]))
    return done


def _open_append(path, header):
    """Open a CSV for append, writing the header only if the file is new/empty."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(header)
        f.flush()
    return f, w


# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------
def run_sweep(valid_images, operators, done, dry_run):
    """Main loop. One (operator, image) op at a time; failure of any single op is
    logged and skipped, never fatal. Rows/logs are flushed incrementally."""
    total = len(valid_images) * len(operators)
    # planning counts (used by both dry-run and the live header)
    n_cached = sum(1 for hp in valid_images for (n, _, _) in operators
                   if (n, os.path.splitext(os.path.basename(hp))[0]) in done)
    n_todo = total - n_cached

    print(f"\nPlan: {len(operators)} operators x {len(valid_images)} images "
          f"= {total} pairs")
    print(f"      {n_cached} already done (cached), {n_todo} to run")

    if dry_run:
        print("\n--dry-run: per-operator work that WOULD run "
              "(image already excluded if not in valid set):")
        for (name, kind, _) in operators:
            todo = sum(1 for hp in valid_images
                       if (name, os.path.splitext(os.path.basename(hp))[0]) not in done)
            print(f"    {name:<14s} {kind:<7s} todo={todo:<5d} cached={len(valid_images)-todo}")
        print(f"\nNo work executed (dry run). {n_todo} pairs pending.")
        return

    os.makedirs(P.OUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pf, pw = _open_append(PER_IMAGE_CSV, PER_IMAGE_HEADER)
    ff, fw = _open_append(FAILURES_CSV, ["operator", "image", "path", "error", "when"])

    t_start = time.time()
    processed = 0   # pairs actually attempted this run (for ETA)
    try:
        for img_i, hp in enumerate(valid_images, 1):
            img_id = os.path.splitext(os.path.basename(hp))[0]

            # Load + apply the ONE global resolution cap identically for every
            # operator (classical + DL). cap_hdr writes a capped temp .hdr for
            # file-based ops and returns the capped array for opencv + scoring.
            try:
                hdr_bgr = P.load_hdr(hp)
            except Exception as e:
                for (name, _, _) in operators:
                    if (name, img_id) not in done:
                        fw.writerow([name, img_id, hp, f"HDR load failed: {e}",
                                     time.strftime("%Y-%m-%dT%H:%M:%S")])
                ff.flush()
                continue

            hdr_bgr, hp_eff, cap_tmp = P.cap_hdr(hdr_bgr, hp)
            res_used = f"{hdr_bgr.shape[1]}x{hdr_bgr.shape[0]}"
            try:
                for (name, kind, spec) in operators:
                    if (name, img_id) in done:
                        continue   # resume: already scored in a previous run
                    out_png = os.path.join(P.OUT_DIR, f"{img_id}__{name}.png")
                    processed += 1
                    elapsed = time.time() - t_start
                    eta = (elapsed / processed) * (n_todo - processed) if processed else 0
                    print(f"[{img_i}/{len(valid_images)} {img_id}] {name:<14s} "
                          f"res={res_used:<9s} elapsed={timedelta(seconds=int(elapsed))} "
                          f"ETA={timedelta(seconds=int(eta))}", flush=True)

                    t0 = time.time()
                    cached = os.path.exists(out_png) and os.path.getsize(out_png) > 0
                    tmo_secs = ""
                    try:
                        # produce_ldr has skip-if-exists caching built in, so a
                        # crash between generate and score resumes cheaply.
                        ok = P.produce_ldr(name, kind, spec, hp_eff, hdr_bgr, out_png)
                        tmo_secs = "" if cached else round(time.time() - t0, 3)
                        if not ok:
                            pw.writerow([name, img_id, *_NO_SCORES, res_used, tmo_secs, "", "degenerate"])
                            pf.flush()
                            fw.writerow([name, img_id, hp, "degenerate output (rejected)",
                                         time.strftime("%Y-%m-%dT%H:%M:%S")])
                            ff.flush()
                            done.add((name, img_id))
                            continue
                        t1 = time.time()
                        sc = P.compute_scores(hdr_bgr, out_png, name)
                        pw.writerow([name, img_id, *(f"{sc[m]:.6f}" for m in P.METRIC_COLUMNS),
                                     res_used, tmo_secs, round(time.time() - t1, 3), "success"])
                        pf.flush()
                        done.add((name, img_id))
                    except Exception as e:
                        pw.writerow([name, img_id, *_NO_SCORES, res_used, tmo_secs, "", "failed"])
                        pf.flush()
                        fw.writerow([name, img_id, hp, str(e),
                                     time.strftime("%Y-%m-%dT%H:%M:%S")])
                        ff.flush()
                        # not added to done -> retried on next resume (may be transient)
            finally:
                if cap_tmp:
                    shutil.rmtree(cap_tmp, ignore_errors=True)
    finally:
        pf.close()
        ff.close()

    print(f"\nSweep pass complete in {timedelta(seconds=int(time.time()-t_start))}. "
          f"Rows -> {PER_IMAGE_CSV}, failures -> {FAILURES_CSV}")


# ----------------------------------------------------------------------------
# Summary (two ways: all-successes, and every-operator intersection)
# ----------------------------------------------------------------------------
def _agg(values):
    if not values:
        return ("", "", "")
    mean = stats.mean(values)
    med = stats.median(values)
    sd = stats.pstdev(values) if len(values) > 1 else 0.0
    return (f"{mean:.6f}", f"{med:.6f}", f"{sd:.6f}")


def write_summary(operators):
    """summary.csv with per-operator mean/median/std of every metric column and
    the count each is computed over, reported two ways:
      (a) 'all'          -> over every image the operator succeeded on
      (b) 'intersection' -> over images that EVERY operator succeeded on
    Flags operators whose success count is notably below the others."""
    if not os.path.exists(PER_IMAGE_CSV):
        print("No per_image.csv -> no summary.")
        return

    metrics = P.METRIC_COLUMNS
    # operator -> image -> {metric: value}   (success rows only)
    scores = defaultdict(dict)
    with open(PER_IMAGE_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "success":
                continue
            scores[row["operator"]][row["image"]] = {m: float(row[m]) for m in metrics}

    op_names = [n for (n, _, _) in operators if n in scores]
    if not op_names:
        print("No successful rows -> no summary.")
        return

    # intersection of images every listed operator succeeded on
    img_sets = [set(scores[n].keys()) for n in op_names]
    intersection = set.intersection(*img_sets) if img_sets else set()

    counts = {n: len(scores[n]) for n in op_names}
    max_count = max(counts.values())

    rows = []
    for n in op_names:
        for basis, imgs in (("all", set(scores[n].keys())), ("intersection", intersection)):
            row = [n, basis, len(imgs)]
            for m in metrics:
                row.extend(_agg([scores[n][i][m] for i in imgs]))
            rows.append(row)

    with open(SUMMARY_CSV, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["operator", "basis", "count"]
                    + [f"{m}_{s}" for m in metrics for s in ("mean", "median", "std")])
        wr.writerows(rows)

    print(f"\nSummary -> {SUMMARY_CSV}")
    print(f"  Intersection: {len(intersection)} images succeeded on by ALL "
          f"{len(op_names)} operators")
    # flag laggards (notably fewer successes than the best-covered operator)
    laggards = [(n, c) for n, c in counts.items() if c < 0.9 * max_count]
    if laggards:
        print(f"  !! Low-success operators (< 90% of max {max_count}):")
        for n, c in sorted(laggards, key=lambda x: x[1]):
            print(f"       {n:<14s} succeeded on {c}/{max_count}")
    else:
        print(f"  All operators succeeded on >= 90% of the best count ({max_count}).")

    # concise ranking preview by mean Q over the intersection (apples-to-apples)
    print(f"\n  Rank by mean Q over intersection ({len(intersection)} imgs):")
    col = {m: 3 + 3 * i for i, m in enumerate(metrics)}   # index of each <metric>_mean
    inter_rows = [r for r in rows if r[1] == "intersection" and r[3] != ""]
    for i, r in enumerate(sorted(inter_rows, key=lambda r: float(r[col["Q"]]), reverse=True), 1):
        print(f"    {i:>2}. {r[0]:<14s} TMQI={r[col['Q']]}  FSITM={r[col['FSITM']]}  "
              f"TMQI-II={r[col['Q2']]}")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Robust TMO benchmark sweep orchestrator.")
    ap.add_argument("--images-dir", default=P.HDR_DIR, help="folder of HDR/EXR inputs")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N valid images (smoke test)")
    ap.add_argument("--max-long-edge", type=int, default=P.MAX_LONG_EDGE,
                    help="global resolution cap (long edge px) for ALL operators")
    ap.add_argument("--seed", type=int, default=1234, help="global random seed")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would run (counts, cached, excluded) and exit")
    ap.add_argument("--run-dir", default=None,
                    help="write results/ and tmo_outputs/ under this folder "
                         "instead of the repo root")
    ap.add_argument("--exclude-kinds", nargs="*", default=[],
                    help="operator kinds to skip, e.g. --exclude-kinds opencv")
    args = ap.parse_args()

    seed_everything(args.seed)
    # apply the configurable global cap to the pipeline module (used by cap_hdr)
    P.MAX_LONG_EDGE = args.max_long_edge
    if args.run_dir:
        set_run_dir(args.run_dir)
        print(f"Run dir: {args.run_dir}")

    operators = [op for op in P.TMO_REGISTRY if op[1] not in args.exclude_kinds]
    print(f"Operators ({len(operators)}): {', '.join(n for n, _, _ in operators)}")
    print(f"Resolution cap (long edge): {args.max_long_edge}   Seed: {args.seed}")

    # discover + validate inputs
    hdr_paths = P.find_hdr_images() if args.images_dir == P.HDR_DIR else _glob_dir(args.images_dir)
    if not hdr_paths:
        raise SystemExit(f"No HDR files found in '{args.images_dir}/' (looked for {P.HDR_GLOBS}).")
    valid = preflight(hdr_paths)
    if args.limit:
        valid = valid[:args.limit]
        print(f"--limit {args.limit}: restricting to {len(valid)} images (smoke test)")
    if not valid:
        raise SystemExit("No valid HDR images after pre-flight. Aborting.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_run_config(operators, valid, args.seed, args.max_long_edge)

    done = load_done_pairs()
    run_sweep(valid, operators, done, args.dry_run)
    if not args.dry_run:
        write_summary(operators)


def _glob_dir(d):
    import glob
    paths = []
    for g in P.HDR_GLOBS:
        paths.extend(sorted(glob.glob(os.path.join(d, g))))
    return paths


if __name__ == "__main__":
    main()
