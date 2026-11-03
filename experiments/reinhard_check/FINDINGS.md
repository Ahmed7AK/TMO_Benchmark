# reinhard02 vs. Python global Reinhard: findings (2026-10-15)

## Verdict

**This is a pipeline bug (an output-encoding error), not a real difference between the operators.**

- `luminance-hdr-cli` writes the output of 8 of the 13 lhdr operators to the 8-bit PNG as **linear display
  luminance**, with no transfer curve.
  - The CLI's `--postgamma` defaults to 1, and these operators apply no curve themselves.
  - TMQI, FSITM and TMQI-II read PNG codes as display-encoded values, as does every other operator in
    the benchmark.
  - So these operators are scored on an image that is about two stops too dark.
- With `--postgamma 2.2`, luminance-hdr's reinhard02 is the same as the Python `tm_reinhard_gl`:
  - On 20 images: Q 0.931 vs 0.932.
  - On 100 images: mean difference −0.0008, correlation 0.998.
- Affected, and should get `--postgamma 2.2`: **reinhard02, reinhard05, drago, fattal, pattanaik, ashikhmin,
  lischinski, kimkautz**.
- Not affected (already display-encoded): durand, mantiuk06, mantiuk08, mai.
- Unresolved, left unchanged: vanhateren.
- Expected effect, estimated on a 100-image sample:
  - reinhard02 moves from rank 12 (Q 0.800) to rank 2 (Q 0.926).
  - lischinski moves from rank 11 to rank 4 (Q 0.914).
  - The TMQI oracle rises by about +0.009 (0.9505 → 0.9598 on the sample).
  - The `reinhard_gl` "free" oracle gain disappears: adding it after the fix gives +0.0001.

## Cause

`tmo_pipeline.apply_lhdr` runs `luminance-hdr-cli -l <hdr> --tmo <op> -o <png>` with every option at its
default and scores the PNG unchanged. The HDR is passed in unscaled. Pre-gamma is 1 and post-gamma is 1, as
recorded in the PNG's EXIF block (`PreGamma: 1`, Luminance HDR 2.6.1.1).

Inside luminance-hdr (source: LuminanceHDR/LuminanceHDR master, `src/Core/TMWorker.cpp`):
- The only step after the operator is `postprocessFrame`.
- It applies `applyGamma(frame, postgamma)` **only if postgamma ≠ 1**.
- `applyGamma(g)` raises values to 1/g (`src/Libpfs/manip/gamma.cpp`), so `-G 2.2` is the standard display
  encoding.
- The PNG writer then maps [0,1] linearly to 0–255.

Whether an operator's output needs encoding depends on that operator's own code:

| operator | what the operator writes (luminance-hdr source) | pfstools man page | needs gamma |
|---|---|---|---|
| reinhard02 | Reinhard Ld in [0,1], linear | "results of this TMO require gamma correction"; example `pfsgamma -g 2.2` | **yes** |
| reinhard05 | photoreceptor response, normalised, linear | example `pfsgamma -g 2.2` | **yes** |
| drago | log-mapping Ld, linear (the paper's own gamma step is not ported) | "require gamma correction"; example `pfsgamma -g 1.7` | **yes** |
| pattanaik | display luminance from the adaptation model, linear | "its results should be gamma corrected"; `-g 2.2` | **yes** |
| fattal | exp(solved log-lum), renormalised by percentiles, linear | (no gamma in the examples; operator has only a luminance-contrast "gamma") | **yes** (by source) |
| ashikhmin | display luminance / max, linear | (no Debian page) | **yes** (by source) |
| lischinski | input radiance × 2^exposure-map, linear | (luminance-hdr only) | **yes** (by source) |
| kimkautz | exp(compressed log-lum), percentile-normalised, linear | (no Debian page) | **yes** (by source) |
| durand | **sRGB curve applied inside** `tmo_durand02.cpp` (`decode()` = 1.055·x^(1/2.4)−0.055) | pfstools' own version needs gamma; luminance-hdr's port already includes it | no |
| mantiuk06 | **sRGB curve applied inside** `contrast_domain.cpp` (`denormalizeRGB`, "Transform to sRGB") | same as durand | no |
| mantiuk08 | pixel values from the inverse display model (GGBA "lcd") | "does not require gamma correction" | no |
| mai | LUT output in a log-like perceptual code | "does not require gamma correction" | no |
| vanhateren | input pre-gamma'd by 1/1.8 inside the wrapper, then 1 − cone current / max | (luminance-hdr only) | unclear (see Caveats) |

An earlier check in README_sweep.md found "vanhateren encoding verified identical to the other lhdr
operators". That check looked only at the PNG container (no gAMA/sRGB/iCCP chunk, a single PngWriter). It
did not look at the pixel transfer function. The encodings are identical, but the operators differ in
whether their pixel values are already display-encoded.

## Evidence

All scores use `tmo_pipeline.compute_scores`, the benchmark's own scoring path. Two checks confirm the
setup:
- Re-scoring the stored PNGs reproduces `results/per_image.csv` to within 5e-7.
- Re-running luminance-hdr with the default call reproduces the stored PNGs to within 1 code value.

**1. The stored PNG is exactly 255·Ld, with no gamma** (`fit_encoding.py`). Fitting log(code/255) against
log(Ld) over well-exposed pixels, where Ld is the analytic global Reinhard curve (key 0.18, Lwhite = max):

| image | slope | scale | r |
|---|---|---|---|
| 10000 | 0.989 | 1.003 | 0.9993 |
| 10001 | 0.999 | 0.999 | 0.9999 |
| 10500 | 0.999 | 1.000 | 0.9999 |
| 10250 | 0.970 | 0.935 | 0.9982 |
| 10750 | 1.000 | 0.999 | 0.9999 |

A slope of 1 means the output is linear; gamma-2.2 encoding would give a slope of 0.45.

**2. The gap is in N, not S** (all 991 images, `per_image.csv` vs `probe2_reinhard_gl.csv`):

| | Q | S | N |
|---|---|---|---|
| lhdr reinhard02 (stored) | 0.804 | 0.883 | 0.092 |
| Python reinhard_gl | 0.933 | 0.885 | 0.762 |

**3. The N gap is TMQI's brightness term, not its contrast term** (20-image subset, `subset.txt`). TMQI's N
is pb(mean Y) × pc(mean 11×11 block std), and pb peaks at a mean of 115.94.

| variant (20 images) | Q | S | N | pb | pc | mean Y | 11×11 block std |
|---|---|---|---|---|---|---|---|
| lhdr reinhard02, stored | 0.802 | 0.889 | 0.079 | 0.121 | 0.662 | 56.6 | 13.25 |
| Python curve, colour ratio^1, **no gamma** | 0.802 | 0.889 | 0.079 | 0.121 | 0.662 | 56.5 | 13.24 |
| **lhdr reinhard02 `--postgamma 2.2`** | **0.931** | 0.903 | 0.717 | 0.983 | 0.731 | 114.6 | 13.73 |
| Python curve, colour ratio^1, gamma 2.2 | 0.931 | 0.903 | 0.717 | 0.983 | 0.731 | 114.6 | 13.74 |
| Python reinhard_gl (ratio^0.6, gamma 2.2) | 0.932 | 0.904 | 0.720 | 0.984 | 0.733 | 114.8 | 13.77 |

- Rows 1 and 2 match, and rows 3 and 4 match. The display gamma accounts for the entire 0.13 Q gap.
- The probe's colour exponent (0.6 vs 1) accounts for 0.001.
- Every one of the 20 images gains Q, by +0.097 to +0.170.
- Mean Y moves from 46–80 to 107–130.

**4. A corrected call reproduces the probe on a larger sample** (100 images, `sample100.txt`, which includes
the 20): lhdr reinhard02 with `-G 2.2` minus reinhard_gl has a mean Q difference of −0.0008, a maximum
absolute difference of 0.026, and a correlation of 0.998.

## The other lhdr operators (100-image sample, with `--postgamma 2.2`)

| operator | Q stored → fixed | Δ | S stored → fixed | N stored → fixed |
|---|---|---|---|---|
| reinhard02 | 0.800 → 0.926 | +0.126 | 0.876 → 0.883 | 0.082 → 0.719 |
| lischinski | 0.801 → 0.914 | +0.113 | 0.866 → 0.881 | 0.106 → 0.651 |
| ashikhmin | 0.730 → 0.830 | +0.101 | 0.704 → 0.766 | 0.046 → 0.388 |
| pattanaik | 0.718 → 0.812 | +0.094 | 0.703 → 0.850 | 0.006 → 0.162 |
| fattal | 0.792 → 0.869 | +0.077 | 0.855 → 0.903 | 0.110 → 0.385 |
| reinhard05 | 0.764 → 0.829 | +0.066 | 0.778 → 0.821 | 0.099 → 0.307 |
| drago | 0.838 → 0.891 | +0.053 | 0.881 → 0.866 | 0.247 → 0.544 |
| kimkautz | 0.845 → 0.886 | +0.042 | 0.891 → 0.873 | 0.283 → 0.512 |
| *vanhateren (not recommended)* | 0.733 → 0.776 | +0.043 | 0.690 → 0.749 | 0.094 → 0.161 |

S rises for most of these operators (pattanaik +0.15, fattal +0.05), so the gain is not only TMQI rewarding
brightness. S falls slightly for drago and kimkautz.

**Negative control.** Applying `-G 2.2` to the already-encoded operators double-encodes them. On the
20-image subset, Q drops and the images become too bright:

| operator | Q stored → with `-G 2.2` | mean Y |
|---|---|---|
| durand | 0.882 → 0.815 | 122 → 174 |
| mai | 0.889 → 0.784 | 138 → 187 |
| mantiuk08 | 0.885 → 0.862 | 94 → 152 |
| mantiuk06 | 0.873 → 0.876 | 83 → 148 |

The fix must therefore be per operator, not a blanket `-G 2.2`.

## Expected effect on the benchmark (100-image sample estimate)

Only the 8 operators change; all others keep their benchmark scores.

| metric | oracle stored → fixed | best single operator, fixed |
|---|---|---|
| TMQI Q | 0.9505 → 0.9598 (+0.0093) | catmo 0.9311, reinhard02 0.9259, att 0.9255, lischinski 0.9143 |
| FSITM_TMQI | 0.9247 → 0.9318 (+0.0071) | **reinhard02 0.9176**, att 0.9075, lischinski 0.9059 |
| TMQI-II Q2 | 0.9168 → 0.9242 (+0.0074) | **reinhard02 0.8971**, lischinski 0.8223, catmo 0.7622 |

- **Per-image TMQI winners** (100 images, stored → fixed):
  - att 31 → 27
  - catmo 20 → 11
  - reinhard02 0 → 21
  - mai 11 → 6
  - durand 11 → 4
  - reinhard05 1 → 7
  - lischinski 0 → 5
- On Q2, fattal goes from 0 to 24 wins and reinhard02 from 1 to 23.
- **The probe:**
  - Before the fix, reinhard_gl beats the best of the 17 operators on 31% of these images, and adding it
    raises the oracle by +0.0073.
  - After the fix, adding it raises the oracle by +0.0001.
  - So the selector study's "0.9546 → 0.9605" gain is this encoding bug.
- The sample is a little harder than the full set (stored oracle 0.9505 vs 0.9546 on all 991). Treat +0.009
  as an estimate; the full re-sweep will give the exact value.
- Every downstream result changes, because rankings, winners and oracles all move:
  - `q_matrix*.csv`
  - the selectors in `results/selectors`
  - `experiments/selector_accuracy`, including the structured features read from `tmo_outputs/`

## Recommended pipeline change (not applied)

In `tmo_pipeline.py`:

```python
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
```

Also record `LHDR_POSTGAMMA` in `results/run_config.json`, and correct the vanhateren encoding note in
README_sweep.md.

**Re-sweep procedure (needs the user's go-ahead).** Both `produce_ldr` (skip if the PNG exists) and
`sweep.py` (skip if the pair is already in `per_image.csv`) cache results, so a plain rerun changes nothing.
Steps:

1. Back up, then move aside `tmo_outputs/*__{reinhard02,reinhard05,drago,fattal,pattanaik,ashikhmin,lischinski,kimkautz}.png`
   (7,928 files), and delete those operators' rows from `results/per_image.csv`.
2. Run `python3 sweep.py --exclude-kinds opencv`. Only the 7,928 missing pairs will run; the other 9
   operators stay cached.
3. Run `make_qmatrix.py` for Q, FSITM, FSITM_TMQI and Q2. Then re-run selector training and the
   selector_accuracy study.

## Caveats and judgment calls

- **"Library defaults" fairness rule.**
  - `--postgamma` is the output encoding, not an operator parameter.
  - Leaving it at 1 does not make the run "default". It mixes two encodings: 4 lhdr operators, att and the
    DL operators produce display-encoded images, while 8 produce linear ones.
  - The operators' own docs (pfstools) prescribe gamma correction for exactly the linear-output group.
- **Score gains are not the evidence.** TMQI's N rewards a mean near 116, so almost any brightening raises
  Q. The classification above comes from the source code and man pages. The negative control shows that
  the same change hurts the operators that are already encoded.
- **Choice of 2.2.**
  - 2.2 is a pure power curve; luminance-hdr offers no piecewise sRGB option. It matches the pfstools
    examples for reinhard02, reinhard05 and pattanaik, and the Python probe.
  - drago's man-page example uses 1.7. A per-operator value is defensible, but would add a tuning knob.
- **vanhateren.** Its output (1 − cone current / max, after an internal x^(1/1.8) on the input) is already
  compressive. No documentation says whether it needs display gamma, so it is left unchanged. Adding
  `-G 2.2` would raise it 0.733 → 0.776 but changes the oracle by only +0.0001, so the decision barely
  matters.
- **ferwerda** stays excluded. Even with `-G 2.2`, its default `adapt_lum` gives near-black images
  (median mean code 1.7 on the 20-image subset; default 0.0).
- **The OpenCV selector probe has the same issue.** `drago_ocv` uses `cv2.createTonemapDrago(1.0, 0.7)`,
  i.e. gamma 1.0, so it is also linear output. It is only a selector probe, not benchmarked, and is not
  re-tested here.

## Files (all in this folder)

- `fit_encoding.py`: fits stored PNG codes against analytic Reinhard Ld (evidence 1).
- `run_lhdr.sh`: re-runs luminance-hdr-cli for a variant, a CLI-argument string and a list of operators.
  Uses `subset.txt`, or `$SUBSET`.
- `subset.txt`: the 20 images (seed 0). `sample100.txt`: those 20 plus 80 more (seed 1).
- `score.py` → `scores_subset.csv`: 20 images, stored and `-G 2.2` for all 13 lhdr operators, plus the
  Python decompositions. Includes mean Y, block std and clipping.
- `score_sample.py` → `scores_sample100_G22.csv`: 100 images, 9 operators with `-G 2.2`.
- `effect.py` → `effect_sample100.txt`: rankings, wins and oracles before and after the fix (Q, FSITM_TMQI,
  Q2).
- `outputs/`: re-run PNGs (`default/`, `G2.2/`, `ferwerda_*`, `python/`). Nothing in `tmo_outputs/` or
  `results/` was modified.
