# TMO benchmark sweep (`sweep.py`)

Robust, resumable, reproducible orchestrator for evaluating every operator in
`tmo_pipeline.TMO_REGISTRY` over a folder of HDR images. Built for an overnight
~20k tone-map + ~20k TMQI run that must survive crashes and produce paper-grade
output.

It does **not** reimplement pipeline logic. It imports and reuses
`tmo_pipeline`'s own registry, `load_hdr`, `cap_hdr`, `produce_ldr`, and
`compute_tmqi`, so the fairness path is identical to the single-image pipeline.

## Run

```bash
# Full sweep (all HDRs in images/, all operators)
python3 sweep.py

# Dry run: list what WOULD run (per-operator todo / cached / excluded), no work
python3 sweep.py --dry-run

# Smoke test: first 5 valid images x all operators
python3 sweep.py --limit 5

# Global resolution cap (long edge px) applied identically to ALL operators
python3 sweep.py --max-long-edge 1920

# Different input folder / seed
python3 sweep.py --images-dir images --seed 1234
```

Re-running is always safe: work already recorded in `results/per_image.csv` is
skipped, so you can **Ctrl-C and rerun** with no lost work and no duplicate rows.

## What it does

1. **Pre-flight validation** (`results/preflight.csv`) — every HDR is loaded
   through the pipeline's float path before any tone-mapping. An image is
   **invalid** only if it fails to decode as a float array (loader-bug guard) or
   is genuinely blank/flat (luminance max ≤ 0, or coefficient of variation
   < 1e-4). Absolute brightness is **not** used to exclude — a `.hdr` stores
   radiance in arbitrary units, so dim-but-structured images are kept and flagged
   `low_range` (abs max < 1.0) in the `warning` column. Invalid images are
   excluded from the whole sweep and reported.
2. **Resumable, crash-proof sweep** — every (operator, image) op is wrapped in
   try/except: a failure logs operator + filename + error to `logs/failures.csv`,
   is recorded as `status=failed`, and the sweep **continues**. Outputs are
   cached (skip-if-exists), and rows/logs are flushed after every op, so a hard
   crash leaves valid partial results on disk.
3. **One global resolution cap** (`--max-long-edge`) applied identically to
   classical and DL operators (via `cap_hdr`), so a few 4K images can't OOM the
   DL operators while classical ones pass. The resolution actually used is logged
   per row (`resolution_used`).
4. **Fairness invariants** — identical path for all operators: 8-bit PNG →
   TMQI, FSITM and TMQI-II, scored against the **original linear HDR** reference
   (capped identically, center-cropped for CATMO's 32-px crop, resized for any
   other size change), never a network's normalized input. Every PNG must be
   display-encoded: luminance-hdr-cli writes reinhard02, reinhard05, drago,
   fattal, pattanaik, ashikhmin, lischinski and kimkautz as linear display
   luminance, so they run with `--postgamma 2.2`
   (`tmo_pipeline.LHDR_POSTGAMMA`); durand, mantiuk06, mantiuk08 and mai are
   already display-encoded and keep the CLI default of 1. `vanhateren` also
   keeps 1, but whether its pixels need display gamma is unresolved. An earlier
   check found its encoding "identical" to the other `lhdr` operators, but it
   compared only the PNG container (no gAMA/sRGB/iCCP chunk, one PngWriter),
   not the pixel transfer curve. See `experiments/reinhard_check/FINDINGS.md`.
5. **Reproducibility** — `results/run_config.json` records the seed, operator
   list, luminance-hdr `--postgamma` per operator (`lhdr_postgamma`), third-party
   repo commits, package versions, resolution cap, and the input manifest. Host RNG is seeded and the seed is propagated to DL
   subprocesses via the environment.

## Output files

| File | Contents |
|------|----------|
| `results/preflight.csv` | Per-image validity: status, detail, w/h, max_value, lum_cov, warning |
| `logs/failures.csv` | One row per failed op: operator, image, path, error, timestamp |
| `results/per_image.csv` | One row per (operator, image): operator, image, Q, S, N (TMQI), FSITM, FSITM_TMQI, Q2, S2, N2 (TMQI-II), resolution_used, tmo_seconds (blank if cached), score_seconds, status |
| `results/summary.csv` | Per-operator mean/median/std of every metric + count, computed two ways (see below) |
| `results/run_config.json` | Full reproducibility record |
| `tmo_outputs/` | The tone-mapped LDR PNGs (the skip-if-exists cache) |

### summary.csv — two bases

Each operator appears twice, distinguished by the `basis` column:

- **`all`** — statistics over every image that operator succeeded on.
- **`intersection`** — statistics over the images that **every** operator
  succeeded on (apples-to-apples comparison). The run also prints how many images
  are in the intersection and **flags any operator whose success count is < 90%**
  of the best-covered operator (a low count means that operator failed on images
  the others handled, which would bias its `all` mean).

Use `per_image.csv` (not just the means) for variance and significance tests.

## Notes

- Operators come from `tmo_pipeline.TMO_REGISTRY`; enable/disable there (DL
  operators are gated by the `ENABLE_*` flags). `ferradans` and `ferwerda` are
  intentionally disabled (too slow / no tuning-free default, respectively).
- The DL operators run their upstream inference in their own venvs via
  subprocess; ensure their checkpoints/venvs are set up (see `SETUP.md`) or they
  will be logged as failures and skipped.
