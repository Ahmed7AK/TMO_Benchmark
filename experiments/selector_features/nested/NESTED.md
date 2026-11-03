# Nested cross-validation of the final TMO selector

The question: how optimistic are the published numbers for the final selector (all 14 feature groups +
structured LightGBM bagged ×5; TMQI Q 0.95973 · 66.5% exact · 81.7% near-optimal, 5 seeds)? The feature-set
choice and the model choice were made on the same 992 images they are scored on (REPORT.md §9). Here both
choices are re-run inside every outer training split, using only those images, and the chosen
configuration is then scored on the outer test fold it never saw.

## 1. Result

Outer loop: the study's own 5 folds (scene-grouped StratifiedGroupKFold, seed 1234), 992 images.

| | mean Q | exact | near-opt (≤ 0.005) | top-3 | gap recovered |
|---|---|---|---|---|---|
| always-catmo | 0.93567 | 8.2% | 18.1% | – | 0% |
| **nested, primary rule** | **0.95959** | **65.2%** | **80.4%** | **92.4%** | **88.0%** |
| nested, argmax rules (sensitivity) | 0.95967 | 65.4% | 80.7% | 92.5% | 88.3% |
| fixed all-groups + struct_lgbm, same outer folds | 0.95959 | 65.2% | 80.4% | 92.4% | 88.0% |
| published (fixed config, mean of 5 seeds) | 0.95973 ± 0.00015 | 66.5% ± 1.0 | 81.7% | 92.6% | 88.5% |
| oracle | 0.96286 | 100% | 100% | – | 100% |

- **The pipeline chose the published configuration in all five outer folds.** As a result, the nested
  estimate equals the fixed configuration on the same folds, image for image.
- **No selection optimism shows up.** The 0.00014 Q and 1.2-point gap to the published numbers is
  split-to-split variance (§4).
- By source, the nested estimate is HDR-Real 0.95484 and HDR-Synth 0.96771. The published figures are 0.95506
  and 0.96772.

## 2. Protocol

Inside each outer training split (793–794 images), using only those images:

1. **Feature groups: all 14, or the greedy-pruned subset.**
   - **Greedy elimination.** This is `greedy.py`'s logic exactly: structured LightGBM without bagging,
     inner 3-fold × seeds 1234 / 7. A group is dropped when its removal gives the largest mean gain and the
     gain is positive on both seeds; the search stops otherwise.
   - **Choosing between all groups and the subset.** Both are run with struct_lgbm on the step-2 inner CV.
     The greedy subset is taken only if its paired scene-bootstrap 95% CI of ΔQ lies above 0. Otherwise all
     groups are kept.
   - **Why this rule.** It is how the study decided: the full-data greedy subset was "a tie" and lost. On
     the full data, ΔQ was +0.00003 [−0.00030, +0.00049], and the greedy subset beat all groups on 2 of 5
     seeds. A greedy subset chosen on the screening protocol always wins on that protocol, so the choice
     has to be confirmed on a different one.
2. **Model.** The 7 confirmed finalists (`struct_lgbm`, `struct_lgbm_nobag`, `struct_ridge`, `struct_et`,
   `lgbm_reg`, `lgbm_clf`, `rf_clf`) are run on the chosen feature set with inner 5-fold × seeds 1234 / 7.
   The one with the highest mean Q is taken, which is the study's rule.
3. **Outer fit.** The chosen configuration is fit on the whole outer training split (seed 1234) and scores the
   outer test fold. The pick falls back to catmo unless another operator's score beats it, as in the study.

**Sensitivity rules.**
- `fset_argmax`: step 1 takes the higher inner mean Q, with no CI.
- `joint_argmax`: the best of all 14 inner (feature set × model) cells.

Both rules gave the same choices.

**Checks.**
- **Candidate set and fallback.** The 15 candidate operators and the catmo fallback are identical in every
  outer and inner training split, so neither leaks from the test fold.
- **The base-group choice (§3) is not re-run.** It does not affect the final selector, because greedy
  elimination starts from all groups.

## 3. Choices across the 5 outer folds

| | choice | folds |
|---|---|---|
| feature set, primary rule | all 14 groups | 5 / 5 |
| feature set, argmax rules | all 14 groups / greedy subset without brisque | 3 / 2 |
| model, every rule | struct_lgbm (bagged ×5) | 5 / 5 |
| greedy removals | none / brisque | 3 / 2 |

**Greedy elimination is not stable across subsets.**
- On the full data it dropped pouli10 and then kimkautz08. That result was not reproduced in any outer fold.
- Dropping pouli10 gained between −0.00007 and +0.00005 on the inner mean, and was negative on the worse seed
  in every fold.
- In the two folds where brisque was dropped, the subset's ΔQ against all groups had CIs that included 0:
  +0.00011 [−0.00020, +0.00049] and +0.00014 [−0.00021, +0.00055].

This instability supports the study's choice of all groups.

**The model choice is stable.**
- struct_lgbm ranked first in every inner CV, by 0.00016–0.00045 Q over the runner-up. The runner-up was
  struct_lgbm_nobag in 4 folds and struct_ridge in 1.
- On the pooled outer folds the unbagged model ties on Q (0.95959 both) with lower exact accuracy (63.1% vs
  65.2%). This matches REPORT.md §5: the unbagged model is within noise on the mean.

## 4. Gap from the selection-conditional numbers

| | mean Q | exact | near-opt |
|---|---|---|---|
| published (5 seeds) | 0.95973 | 66.5% | 81.7% |
| nested (primary rule) | 0.95959 | 65.2% | 80.4% |
| **gap** | **−0.00014** | **−1.2 pt** | **−1.3 pt** |
| of which: selection (nested − fixed, same folds) | 0.00000 | 0.0 pt | 0.0 pt |
| of which: split (seed 1234 − 5-seed mean) | −0.00014 | −1.2 pt | −1.3 pt |

- Seed 1234's outer split gives the fixed configuration its second-lowest Q and lowest exact accuracy of the
  five seeds:
  - Q: 0.95956–0.95988;
  - exact: 65.2–67.6%.
- **The gap is about 1 sd of the seed-to-seed spread** (0.00015 Q, 1.0 pt exact).

## 5. Paired comparisons on the same outer folds

Scene-cluster bootstrap 95% CI, scene-level Wilcoxon (`common.paired`); every image is its own scene.

| comparison | ΔQ [95% CI] | p | images better / worse |
|---|---|---|---|
| nested vs always-catmo | **+0.02392 [+0.02216, +0.02568]** | < 1e-15 | 835 / 98 |
| — HDR-Real | +0.02456 [+0.02243, +0.02673] | < 1e-15 | 534 / 57 |
| — HDR-Synth | +0.02283 [+0.01990, +0.02581] | < 1e-15 | 301 / 41 |
| nested vs fixed all-groups + struct_lgbm | **0 (identical picks)** | 1 | 0 / 0 |
| argmax rules vs fixed | +0.00008 [−0.00005, +0.00021] | 0.57 | 23 / 22 |

## 6. Per outer fold

| fold | test images | greedy removed | chosen | inner Q | outer Q | outer exact | catmo Q | argmax rules' outer Q |
|---|---|---|---|---|---|---|---|---|
| 0 | 199 | – | ALL + struct_lgbm | 0.96017 | 0.95702 | 68.8% | 0.93088 | same |
| 1 | 198 | – | ALL + struct_lgbm | 0.95927 | 0.96072 | 64.6% | 0.93785 | same |
| 2 | 199 | brisque | ALL + struct_lgbm | 0.95941 | 0.96024 | 70.4% | 0.93466 | 0.96039 (greedy) |
| 3 | 198 | – | ALL + struct_lgbm | 0.95920 | 0.95958 | 56.6% | 0.93809 | same |
| 4 | 198 | brisque | ALL + struct_lgbm | 0.95900 | 0.96040 | 65.7% | 0.93689 | 0.96064 (greedy) |

**Verification.**
- Every all-groups outer fit reproduces the study's cached OOF scores for that fold (`oof/`, seed 1234):
  - six models are bit-identical;
  - struct_et differs by ≤ 4.4e-16, from the float summation order in sklearn's threaded forest prediction,
    with identical picks.
- The harness's subset evaluator reproduces `common.evaluate` exactly.

## 7. What this does not cover

- **Hyperparameters.** The LightGBM settings (num_leaves 15, min_child_samples 10, bag 5) were fixed in an
  earlier study on these same images. Nested CV here treats them as constants.
- **The finalist shortlist.** The 7 finalists were shortlisted from 17 models on all 992 images.
  - Given struct_lgbm's consistent inner margin, the shortlist is unlikely to matter.
  - It is still not nested.
- **Feature design.** The 14 groups were designed, and audited and fixed (e.g. the Fattal level count), with
  the data in view.
- **One outer split.** Only seed 1234's outer split was nested. Its sampling noise is the seed-to-seed sd
  above.
- **One benchmark.** This is the same 992-image benchmark and the same TMQI proxy as the study.

## 8. Statement for the report

> To check for selection bias, we ran nested cross-validation (5 outer scene-grouped folds, seed 1234). Inside
> every outer training split it re-ran both selection steps on those images only: the feature-group choice
> (greedy backward elimination with inner 3-fold × 2 seeds, then all groups vs the greedy subset on an inner
> 5-fold × 2-seed CV) and the model choice among the seven finalists (inner 5-fold × 2 seeds). It chose the
> published configuration, all 14 groups with bagged structured LightGBM, in all five folds. The nested
> estimate therefore equals the fixed configuration on the same folds: mean TMQI 0.9596, 65.2% exact
> accuracy and 80.4% near-optimal, against 0.9357 for always-catmo (+0.0239 [+0.0222, +0.0257]) and 0.9629
> for the oracle. The published five-seed figures (0.9597, 66.5%, 81.7%) are 0.00014 Q and 1.2 points of
> exact accuracy higher. That difference is split-to-split variance (seed-to-seed sd 0.00015 Q and 1.0
> point), not selection optimism. A more aggressive rule that always took the higher inner score would have
> dropped BRISQUE in two folds and scored 0.9597 / 65.4%, again indistinguishable from the fixed
> configuration (+0.00008 [−0.00005, +0.00021]). The nested CV does not cover choices fixed earlier on the same
> images: the LightGBM hyperparameters, the shortlist of seven finalists from 17 models, and the feature
> definitions. A small residual optimism from these cannot be excluded, but the selection steps made in this
> study did not inflate the reported accuracy.

## 9. Files and reproduction

- `nested_lib.py`: the protocol, the cached parallel fits and the subset evaluation. It imports the study's
  `common.py` and `models.py` unchanged.
- `run_fold.py`: one outer fold, in three stages (greedy → select → outer).
- `summarise_nested.py`: pools the folds.
- `folds/fold<i>.json`: per fold, the greedy steps and trials, the inner grid, the CI, the choices and the
  outer scores.
- `tables/nested_{summary,paired,choices,folds}.csv`.
- `logs/`; `cache/` holds the fold fits, about 2 h of compute.

```
cd TMO/experiments/selector_features/nested
export OMP_NUM_THREADS=1
for i in 0 1 2 3 4; do
  ../../../.venv/bin/python run_fold.py --fold $i --stage greedy    # 7–12 min each, 8 cores
  ../../../.venv/bin/python run_fold.py --fold $i                   # 7–14 min each
done
../../../.venv/bin/python summarise_nested.py
```

- **Thread setting.** LightGBM runs one thread per fit, and 8 worker processes share the cores. The outputs
  equal the study's 2-thread fits.
- **Nothing outside `nested/` was modified.**
