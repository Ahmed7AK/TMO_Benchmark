# Research-backed features and model for the per-image TMO selector

Data: the corrected benchmark (luminance-hdr `--postgamma 2.2` fix), 992 images, 15 candidate operators
under TMQI Q. Protocol: scene-grouped StratifiedGroupKFold (5-fold), out-of-fold only, seeds
1234/2024/7/42/99. Statistics: scene-cluster bootstrap 95% CI and scene-level Wilcoxon on per-image values
averaged over seeds. The pick falls back to the best fixed operator (catmo for Q) unless another
operator's score beats it.

Code is in this folder: `features.py`, `common.py`, `models.py`, `run.py`, `screen.py`, `greedy.py`,
`summarise.py` and `source_leak.py`. Tables are in `tables/`, logs in `logs/`, and the independent audit
in `audit/`. The production versions are `selector_features.py` and `compare_selectors.py` at the
repository root; they reproduce this study's out-of-fold picks exactly.

## 1. Rules

- **Features describe the HDR only.** No tone-mapping operator (candidate or stand-in) is run, no
  quality metric is computed, and nothing is read from `tmo_outputs/` at prediction time.
- **Allowed:** statistics an operator computes from its HDR input before mapping, such as Drago's
  log-average or Durand's base-layer range.
- **Training targets:** the per-operator metric components from `results/q_matrix_*.csv`, used for
  training rows only.
- **Every group follows a published source.** Each group's docstring in `features.py` cites it and states
  every deviation.
- **Sources were read, not recalled.** Paper texts were extracted from the PDFs. For the operators, the
  luminance-hdr source code the benchmark ran was used, with its CLI defaults.
- **Unreachable papers:** the Drago, Fattal and Reinhard & Devlin 2005 PDFs could not be fetched (host
  DNS failure or 403). Those groups rest on the operator code.

## 2. Feature groups (151 features)

| group | source | n | role | notes |
|---|---|---|---|---|
| hulusic17 | Hulusic, Debattista, Valenzise, Dufaux 2017, SPIC 51 | 5 | perceived-DR measures | DR, image key, bright area, PU contrast, colourfulness (formula: Hasler & Süsstrunk 2003; the paper gives none); PU21 replaces PU2008 |
| pouli10 | Pouli, Cunningham, Reinhard 2010, EG STAR | 27 | HDR image statistics | log-luminance moments, power-spectrum slope, gradient and band-pass statistics, Lαβ colour; the paper defines the statistics, the summaries chosen are ours |
| reinhard02 | Reinhard et al. 2002; Reinhard 2002 (JGT) | 5 | operator-internal | log-average, default white point, automatic key / white estimate (verbatim from luminance-hdr lischinski06) |
| drago03 | Drago et al. 2003 | 2 | operator-internal | Lwmax / Lwa, the log-base divider |
| durand02 | Durand & Dorsey 2002 | 4 | operator-internal | bilateral base range, compression factor, detail energy |
| fattal02 | Fattal et al. 2002 | 6 | operator-internal | average gradient and log-attenuation at the attenuated levels k = 3–5 |
| reinhard05 | Reinhard & Devlin 2005 | 2 | operator-internal | image key k, contrast m = 0.3 + 0.7 k^1.4 |
| kimkautz08 | Kim & Kautz 2008 | 2 | operator-internal | log-mean, global compression k1 |
| mai11 | Mai et al. 2011 | 2 | operator-internal | Σ p^(1/3) of the log10 histogram, largest slope share |
| mantiuk08 | Mantiuk, Daly, Kerofsky 2008 | 12 | operator-internal | band-contrast distribution per Laplacian level, conditional on background |
| tmqi | Yeganeh & Wang 2013 | 10 | metric-derived | TMQI's HDR-side local-σ visibility at its 5 levels |
| brisque | Mittal, Moorthy, Bovik 2012 | 36 | NSS / IQA | BRISQUE on log luminance mapped to [0, 255]; NSS used for tone-mapped quality by Kundu et al. 2017 |
| si | ITU-T P.910 | 1 | content | spatial information on log luminance |
| tmqiblock | Yeganeh & Wang 2013 (adapted) | 37 | adapted | 11 / 22 / 44-px block statistics of log luminance; TMQI's naturalness block, applied to the HDR |

The project's original 22 features (`hdr_features.py`) were project-made. Their cited parts are already
in the groups above:
- Reinhard's log-average luminance → reinhard02;
- the p1–p99 dynamic range and the log-domain image key (Akyüz & Reinhard 2006) → hulusic17;
- the log-luminance moments → pouli10;
- Hasler & Süsstrunk colourfulness → hulusic17.

## 3. Choosing the single-paper base set

Every group was screened on its own: fast structured model, 3-fold × 2 seeds (`tables/screen_single.csv`).
The two whole-paper candidates were then run on the full protocol with three models (`tables/runs.csv`,
tag `base`):

| feature set | n | structured LightGBM | direct LightGBM | RF classifier |
|---|---|---|---|---|
| Pouli et al. 2010 | 27 | **0.9547 · 50.0%** | 0.9536 · 43.2% | 0.9531 · 47.5% |
| Hulusic et al. 2017 | 5 | 0.9463 · 32.5% | 0.9463 · 29.8% | 0.9441 · 34.6% |
| original 22 (project-made, reference) | 21 | 0.9575 · 57.8% | 0.9560 · 49.0% | 0.9556 · 53.6% |

Each cell is mean Q · exact accuracy, averaged over 3 seeds.

- **Every other single-source group screened 0.0066–0.030 below Pouli.** Only the project-made 22 and the
  TMQI-block adaptation scored higher, and neither is a published set.
- **Base set = Pouli et al. 2010.** On the full protocol over 5 seeds it scores 0.95469 · 49.6%.
- **Caveat:** Hulusic et al. 2017 is the more faithful one-paper set, since its five measures are fully
  specified. It is too small to tell 15 operators apart.

## 4. Building and optimising the compiled set

This repeats the optimisation used for the earlier ad-hoc content features.

1. **Add-one** (`tables/screen_add.csv`): adding each group to the base. TMQI-block, Hulusic and
   Reinhard 2002 add the most, each about +0.0023.
2. **Leave-one-out** (`tables/screen_loo.csv`): dropping each group from all 14. Every removal except
   TMQI-block changes Q by at most +0.0004, so the groups largely overlap.
3. **Greedy backward elimination** (`tables/greedy_steps.csv`): dropped pouli10, then kimkautz08, then
   stopped (no further removal helped on both seeds).
4. **Full protocol, 5 seeds** (`tables/confirm_fsets.csv`):

| feature set | n | mean Q ± sd | exact | near-opt | top-3 | ΔQ vs base [95% CI] |
|---|---|---|---|---|---|---|
| base: Pouli 2010 | 27 | 0.95469 ± 0.00026 | 49.6% | 64.4% | 84.5% | – |
| original 22 | 21 | 0.95752 ± 0.00022 | 57.8% | 74.2% | 89.0% | +0.0028 [+0.0019, +0.0038] |
| **compiled, all 14 groups** | 151 | **0.95973 ± 0.00015** | **66.5%** | **81.7%** | **92.6%** | **+0.0050 [+0.0041, +0.0060]** |
| greedy subset (12 groups) | 122 | 0.95976 ± 0.00006 | 65.4% | 81.3% | 92.6% | +0.0051 [+0.0041, +0.0061] |
| greedy subset + true source label | 123 | 0.95971 | 65.2% | | | +0.0050 |
| greedy subset, absolute level removed | 121 | 0.95948 | 64.2% | | | +0.0048 |
| compiled, absolute level removed (4 columns) | 147 | 0.95951 ± 0.00013 | 64.9% | | | +0.0048 |

**Against the project's original 22 features** (`tables/confirm_vs_hdr22.csv`; the more informative
comparison, because the base is the winner of a screen):
- compiled 151: ΔQ +0.0022 [+0.0016, +0.0028], p = 8e-14; exact accuracy 66.5% vs 57.8%;
  HDR-Real +0.0025, HDR-Synth +0.0017;
- without the absolute-level columns: +0.0020 [+0.0014, +0.0026].

**Decision: all 14 groups.**
- The greedy subset ties on Q (+0.00003).
- Using all groups avoids the greedy selection step. The 1.1-point exact-accuracy difference is an
  untested tie-break, not a reason.
- "151 features" overstates the distinct information. Four columns encode the same absolute radiance level
  (`p10_logl_mean`, `p10_lab_L_mean`, `r02_log_avg`, `kk08_mu`; source AUC 0.81 each), and several groups
  overlap. Gradient boosting tolerates this.

**The gain is not source identification.**
- Adding the true HDR-Real / Synth label changes nothing.
- One leak was found and fixed during the study. The number of Fattal pyramid levels depends on image size,
  so a missing level marked the smaller HDR-Synth images (source AUC 1.0). The features now use levels
  every image has.
- Remaining source AUCs per group are in `tables/source_leak.csv`. They are highest for pouli10 (0.91),
  the original 22 (0.92), reinhard02 and brisque (0.86).
- Every image's long side is 512 px, so pixel scale is the same across sources.

**Compared with the earlier ad-hoc content set:** 22 + 228 hand-made features reached 0.95998 · 68.0%. The
research-backed compiled set gives up 0.0003 Q and 1.5 points of exact accuracy for features that each have
a published source.

## 5. Choosing the model

**Screening:** 17 models on both the base and the compiled set, 5-fold × 3 seeds (`tables/runs.csv`, tag
`models`):
- structured models, which predict each operator's metric components and compose the score with the
  metric's formula: LightGBM bagged ×5, LightGBM without bagging, ExtraTrees, random forest, Ridge, MLP;
- direct per-operator regression: LightGBM, ExtraTrees, random forest, Ridge, MLP;
- classifiers: random forest, ExtraTrees, LightGBM, logistic regression, kNN;
- a LightGBM lambdarank ranker.

**Confirmation:** the leaders, 5 seeds, paired with structured LightGBM (`tables/confirm_models_all.csv`):

| model (compiled set) | mean Q | exact | ΔQ vs structured LightGBM [95% CI] |
|---|---|---|---|
| **structured LightGBM, bagged ×5** | **0.95973** | **66.5%** | – |
| structured LightGBM, no bagging | 0.95950 | 64.7% | −0.00023 [−0.00050, +0.00005], p = 2e-5 |
| structured Ridge | 0.95880 | 61.8% | −0.00093 [−0.00142, −0.00045] |
| structured ExtraTrees | 0.95785 | 57.2% | −0.00188 [−0.00236, −0.00140] |
| direct LightGBM regression | 0.95729 | 52.8% | −0.00244 [−0.00289, −0.00199] |
| LightGBM classifier | 0.95682 | 56.6% | −0.00291 [−0.00361, −0.00227] |
| LightGBM ranker | 0.95624 | 54.3% | −0.00349 [−0.00418, −0.00282] |
| random-forest classifier (the original selector) | 0.95601 | 53.2% | −0.00372 [−0.00446, −0.00302] |

On the base set, structured LightGBM and structured ExtraTrees tie on Q (Δ +0.0001, CI ±0.0008). LightGBM
has the higher exact accuracy and beats every other model (`tables/confirm_models_base.csv`).

**Decision: structured LightGBM, bagged ×5.**
- It has the highest mean Q on both feature sets.
- **Against every other model family, the mean-difference 95% CIs exclude 0.** All 7 Wilcoxon p-values
  stay ≤ 1.3e-4 after a Holm correction for the 7 comparisons.
- **The unbagged version is within noise on the mean.** Its CI [−0.00050, +0.00005] includes 0, although
  the Wilcoxon test (p = 2e-5) says it loses on more images than it wins. Bagging is kept because it is
  never worse.
- The structure matters more than the learner: even a linear Ridge inside the TMQI formula beats every
  direct or classification model.
- **The hyperparameters were not tuned in this study.** num_leaves 15, min_child_samples 10 and bag-5 were
  chosen in the earlier study on these same images.

## 6. Transfer to the other metrics (3 seeds; `tables/transfer_Q2.csv`, `tables/transfer_FSITM_TMQI.csv`)

| metric | floor | base (Pouli) | **compiled + structured LightGBM** | compiled + direct LightGBM | compiled + RF | oracle |
|---|---|---|---|---|---|---|
| TMQI-II | 0.8951 | 0.8890 · 44.9% | **0.9139 · 61.0%** | 0.9107 · 45.2% | 0.9046 · 54.4% | 0.9281 |
| FSITM_TMQI | 0.9196 | 0.9286 · 55.3% | **0.9310 · 68.3%** | 0.9295 · 54.6% | 0.9297 · 60.4% | 0.9328 |

For TMQI-II, the structured composition uses TMQI-II's HDR-side estimates (est_mu2, est_sd2) inside its N2
formula only, as part of the metric's definition; they are never model features. The Pouli base falls
below always-reinhard02 on TMQI-II.

## 7. What the model uses (production run, Q, seed 1234; `results/selectors/Q/feature_importance.csv`)

**LightGBM gain share by group:**

| group | share |
|---|---|
| tmqiblock | 23% |
| reinhard02 | 16% |
| brisque | 15% |
| pouli10 | 11% |
| hulusic17 | 10% |
| tmqi | 10% |
| fattal02 | 5% |
| mantiuk08 | 3% |
| all others | ≤2% each |

- **Top single features:** Reinhard's estimated key and default white point, TMQI's HDR local-σ at the finest
  scale, and the Akyüz–Reinhard image key.
- **Per-operator recall:**
  - highest: reinhard02 0.82, reinhard05 0.81, att 0.78, ashikhmin 0.75, drago 0.73;
  - lowest: catmo 0.33, durand 0.38, uncltmo 0.42, mantiuk06 0.43.

## 8. Independent audit (sub-agent)

**Round 1, feature definitions against the sources.** It found:
- three constants that contradicted the luminance-hdr code: Drago ε, Mai floor, Reinhard 2005 Lmin;
- the wrong intensity and range-kernel scaling for Durand;
- non-attenuated Fattal levels;
- one Mantiuk level too many;
- a size-dependent spectral cut-off;
- colourfulness credited to the wrong paper;
- several derived quantities that are not operator statistics.

It confirmed no use of `q_matrix` or `tmo_outputs`, targets used on training rows only, and that the
harness reproduces `compare_selectors.py`. I verified each finding against the code, fixed all of them,
recomputed every feature and re-ran every result above. The OOF cache is now keyed by a hash of the
feature values. Details: `audit/`.

**Round 2, results and production code.**

*Reproductions.* All three requested runs reproduced bit-exactly from scratch:
- pouli10 seed 1234;
- ALL seed 7;
- `compare_selectors.py` at seed 2024, matching the study's out-of-fold picks 100%.

*Features.* `results/features_selector.csv` is identical to the study frame. No final feature uses
`q_matrix` values, `tmo_outputs` or an operator output, and the source-label control is not in the set.

*Issues it raised, and what I did:*
1. **`compare_selectors.py` could overwrite the main run.** A `--features` or `--seed` run without a tag
   wrote to `results/selectors/Q`. Fixed: the feature groups and a non-default seed are now part of the
   folder name. The base run moved to `results/selectors/Q_pouli10`.
2. **The Q2 docstring.** It now says `est_mu2`/`est_sd2` come from the per-operator scoring HDR and enter
   only the N2 formula.
3. **Selection-conditional numbers, no held-out set.** Stated in §9.
4. **The unbagged-model claim overstated.** Corrected in §5: within noise on the mean CI.
5. **Mean CI and Wilcoxon can disagree.** Both are reported, and claims use the mean CI.
6. **Multiple comparisons.** Holm correction added.
7. **Base vs original 22.** The comparison against the original 22 was added (§4).
8. **ALL vs greedy subset.** It is called a tie, decided by avoiding selection (§4).
9. **The absolute-level duplicates and the scale-free ablation on ALL.** Added (§4).

*Image size.* The auditor withdrew its round-1 image-size concern: every image's long side is 512 px, and
the source-label control shows no gain.

Details: `audit/`.

## 9. Caveats

- **All numbers are selection-conditional; there is no held-out set.**
  - The base group, the group set, the model and the LightGBM hyperparameters were all chosen on the same
    992 images they are scored on.
  - Screening seeds 1234 and 7 are also confirmation seeds.
  - The 5 seeds re-split the same images, so they measure split variance, not independent replication.
  - A clean estimate would need nested CV over the whole selection pipeline.
  - What limits the optimism:
    - the final set uses all groups, so there is no group selection;
    - the hyperparameters were fixed in an earlier study, not tuned here;
    - the margins between model families (0.0009–0.0037) are many times the seed-to-seed sd (0.0002).
- **The base set's score is optimistic, so its gain is conservative.** It is the winner of 14 screened
  single-source groups, which makes the +0.0050 gain over it a conservative estimate. The comparison with
  the project's original 22 features is the more informative one (§4).
- **Many models were compared:** 17 screened, 8 confirmed. The chosen model's margin over its unbagged
  variant is within noise; its margins over the other families are large and survive a Holm correction.
- **Pouli's summaries are our choice.** The survey defines the statistics but prescribes no feature set.
  The TMQI-block group is an adaptation (TMQI applies its block statistic to the LDR).
- **Operator-internal statistics are not tone-mapping.** They are computed from the HDR the way each
  operator would before mapping, and are presented as such.
- **One benchmark.** 512-px images from HDR-Real and HDR-Synth. The selector maximises TMQI, which is a
  proxy for human preference.

## 10. Reproduce

```
cd TMO
.venv/bin/python selector_features.py
.venv/bin/python compare_selectors.py --metric Q            # also Q2, FSITM_TMQI
cd experiments/selector_features                            # the study
../../.venv/bin/python features.py
../../.venv/bin/python summarise.py --ref pouli10:struct_lgbm --cfg ALL:struct_lgbm --out confirm_fsets
../../.venv/bin/python summarise.py --ref ALL:struct_lgbm --cfg ALL:lgbm_reg ALL:rf_clf --out confirm_models_all
```
