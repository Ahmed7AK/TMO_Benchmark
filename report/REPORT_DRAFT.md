# Choosing a tone-mapping operator per image from HDR content: TMQI gains and computational cost

*Draft, 26 October 2026. Every number below comes from the checked-in results. `report/make_figures.py` and
`timing_benchmark.py` regenerate them, and `report/report_numbers.json` holds them all. References are in
author–year form, and the full entries are in `report/references.bib`.*

---

## 1. Research problem

A tone-mapping operator (TMO) compresses a high-dynamic-range (HDR) radiance map into an 8-bit image that a
standard display can show. Dozens of TMOs exist (Reinhard et al. 2010; Eilertsen et al. 2017). Perceptual
studies agree that **no single operator is best for every image**: the ranking depends on the scene and on
the criterion (Ledda et al. 2005; Čadík et al. 2008; Cerdá-Company et al. 2018).

Our benchmark shows the same. Across 992 HDR images and 17 operators:
- 15 different operators are the best choice on at least one image;
- the most frequent winner (ATT; Khan et al. 2018) is best on only 25% of images;
- the best fixed operator on average (CATMO; Li et al. 2024) is best on only 8%.

Always applying the single best operator therefore leaves quality unused. Choosing the best operator per
image has a cost, though. Done exhaustively, it means running every operator and scoring every output
against the HDR reference with a full-reference metric such as TMQI (Yeganeh & Wang 2013). That multiplies
the cost of tone mapping by the number of candidates, and it needs the metric at run time.

This is an instance of per-instance **algorithm selection** (Rice 1976; Kerschke et al. 2019): predict,
from cheap features of the input, which of several algorithms will do best, without running them all.

## 2. Research question

> **Can a model that sees only the HDR input choose, for each image, a tone-mapping operator whose TMQI is
> close to that of the best operator for that image, and at what fraction of the cost of finding that
> operator by exhaustive search?**

It has three parts:

1. **Quality.** How much of the TMQI gap between the best fixed operator and the per-image oracle does the
   selector recover?
2. **Accuracy vs. quality.** How often does it pick exactly the best operator, and what does a wrong pick
   cost in TMQI?
3. **Cost.** How long does selection take per image, compared with running and scoring every candidate?

## 3. Methodology

### 3.1 Data

- **Images.** 1000 HDR images drawn from the HDR-Real (628) and HDR-Synth (372) sets of Liu et al. (2020).
  HDR-Synth gathers images from several public collections, among them Fairchild (2007), Funt & Shi (2010)
  and Xiao et al. (2002).
- **Deduplication.** HDR-Real stores each HDR once per paired LDR, so a random draw is mostly duplicates.
  Images were therefore deduplicated by md5 and by thumbnail similarity before sampling.
- **Size.** Each image was resized so that its long side is 512 px.
- **Analysis set.** 992 images were tone-mapped successfully by every operator. The other eight produced at
  least one failed or degenerate (all-black or all-white) output.

### 3.2 Operators

17 TMOs, each run with its library defaults and no per-image tuning:

- **13 classical operators**, run through luminance-hdr-cli 2.6.1.1 in Docker:
  - Reinhard et al. 2002 (`reinhard02`); Reinhard & Devlin 2005 (`reinhard05`);
  - Drago et al. 2003; Durand & Dorsey 2002; Fattal et al. 2002;
  - Mantiuk et al. 2006 (`mantiuk06`); Mantiuk et al. 2008 (`mantiuk08`);
  - Pattanaik et al. 2000; Ashikhmin 2002; Mai et al. 2011; Lischinski et al. 2006;
  - van Hateren 2006; Kim & Kautz 2008.

  Eight of them write linear luminance, so they are display-encoded with gamma 2.2.
- **ATT** (Khan et al. 2018), ported to Python from the authors' MATLAB code.
- **Three learned operators**, run with their published weights:
  - CATMO (Li et al. 2024);
  - UnCLTMO (Cao et al. 2024);
  - Unpaired-HDR (Vinker et al. 2021).

### 3.3 Quality measure and labels

- **Metric.** Each output is scored with TMQI (Yeganeh & Wang 2013) against the original linear HDR, at the
  output's resolution. TMQI combines a multi-scale structural-fidelity term S, derived from SSIM (Wang et al.
  2004), with a statistical-naturalness term N.
- **Label.** An image's label is the operator with the highest TMQI.
- **Candidate set.** Two operators, Pattanaik and van Hateren, never win, which leaves 15 candidates.
- **Other metrics.** FSITM (Nafchi et al. 2015) and TMQI-II (Ma et al. 2015) were computed too, as
  robustness checks (§4.5).

### 3.4 Features

The selector sees only the HDR. Nothing is tone-mapped and no metric is computed at prediction time.

**The feature set.** 151 features in 14 groups, each following one published source:
- **HDR image statistics:**
  - Pouli et al. 2010: log-luminance moments, power-spectrum slope, gradient and Laplacian-band
    statistics (Burt & Adelson 1983), and Lαβ colour (Ruderman et al. 1998; Reinhard et al. 2001);
  - the perceived-dynamic-range measures of Hulusic et al. 2017, with the image key of Akyüz & Reinhard
    2006, PU21 encoding (Mantiuk & Azimi 2021) and colourfulness (Hasler & Süsstrunk 2003).
- **The statistics each operator computes from its input before mapping,** following the luminance-hdr code
  with its defaults: Reinhard's log-average and automatic key and white point (Reinhard 2002), Drago's
  Lmax/Lavg, Durand's base-layer range, Fattal's gradient attenuation, and the Reinhard–Devlin, Kim–Kautz,
  Mai and Mantiuk 2008 set-up statistics.
- **Quality-model statistics:**
  - TMQI's HDR-side local standard deviation;
  - BRISQUE natural-scene statistics (Mittal et al. 2012), which Kundu et al. (2017) used for tone-mapped
    images;
  - ITU-T P.910 spatial information;
  - 11×11-block statistics of log luminance.

**Comparison sets.**
- The best single-paper set (Pouli et al. 2010, 27 features).
- The project's earlier 21 hand-made features.

**Dataset release.** The dataset also contains a 10-bin log-luminance histogram for every image. It was not
used by the selector.

### 3.5 Selector model: structured LightGBM

Instead of classifying the winner directly, the model predicts, for every candidate operator, the
**ingredients of TMQI**:
- the structural fidelity S;
- the output's mean luminance;
- the output's mean 11×11-block standard deviation.

These three fix TMQI's naturalness term. TMQI is then composed from the predictions with its own formula,
and the operator with the highest predicted TMQI is chosen. If no operator is predicted to beat CATMO, the
choice stays on CATMO.

**Model details.**
- Each of the 15 × 3 regressors is a LightGBM model (Ke et al. 2017) with 15 leaves.
- Each is bagged five times (Breiman 1996) over scene-grouped subsets with early stopping.

**How the model was chosen.** This structured model was compared with 16 alternatives on 5 seeds, and it
beat every other family:
- direct per-operator regression;
- random-forest, ExtraTrees and LightGBM classifiers (Breiman 2001; Geurts et al. 2006);
- logistic regression and kNN;
- a LambdaMART ranker (Burges 2010).

### 3.6 Evaluation protocol

**Cross-validation.**
- 5-fold scene-grouped stratified cross-validation (scikit-learn; Pedregosa et al. 2011), so an image is
  never in its own training fold.
- Every number is out-of-fold.
- Headline figures are for seed 1234; five-seed means are given where they differ.

**Measures.**
- **Mean TMQI of the chosen operator.** This is the primary measure. It is reported against two
  reference points:
  - the floor, always applying the best fixed operator (CATMO);
  - the ceiling, the per-image oracle.
- **Gap recovered** = (selector − floor) / (oracle − floor).
- **Exact accuracy.**
- **Regret** = oracle TMQI − TMQI of the chosen operator.

**Statistics.**
- Paired differences carry scene-cluster bootstrap 95% CIs (Efron & Tibshirani 1993).
- Wilcoxon signed-rank tests (Wilcoxon 1945) are Holm-corrected for multiple comparisons (Holm 1979).

**Selection bias.** The feature set and the model were chosen on the same images they are scored on. The
whole selection was therefore re-run inside nested cross-validation (Varma & Simon 2006; Cawley & Talbot
2010).

### 3.7 Timing protocol

**The two procedures.**
- **Exhaustive search (the oracle's cost):** run all 15 candidate operators on the image, score each
  output with TMQI, and keep the best.
- **Selector:**
  1. compute the 151 features;
  2. run the 225 fitted LightGBM models (fit once on all 992 images);
  3. run only the chosen operator.

**Setup.**
- 50 randomly drawn images (seed 1234), each timed after one discarded warm-up image.
- One Apple M3 Pro laptop, one process, one image at a time, LightGBM on one thread.
- Every operator runs fresh, exactly as the benchmark runs it:
  - each luminance-hdr operator is one `docker run`;
  - each learned operator is one inference subprocess in its own environment on the Apple GPU (MPS),
    including model loading;
  - ATT runs in-process.

## 4. Results

### 4.1 TMQI improvement

**Table 1.** Mean TMQI over 992 images (TMQI Q, out-of-fold, seed 1234).

| method | mean TMQI | gap recovered | exact accuracy | within 0.005 of best |
|---|---|---|---|---|
| random operator (expected) | 0.8919 | — | 6.7% | — |
| always CATMO (best fixed operator) | 0.9357 | 0% | 8.2% | 18.1% |
| selector, Pouli et al. 2010 features only | 0.9547 | 70% | 50.7% | 64.9% |
| random-forest classifier, earlier 21 features | 0.9554 | 73% | 52.2% | 67.6% |
| **selector (151 features, structured LightGBM)** | **0.9596** | **88%** | **65.2%** | **80.4%** |
| oracle (best operator per image) | 0.9629 | 100% | 100% | 100% |

**Against the best fixed operator.** The selector raises mean TMQI from 0.9357 to 0.9596, a gain of
**+0.0239** (95% CI +0.0222 to +0.0257). It recovers 88% of the gap between always-CATMO and the oracle. It
is better than CATMO on 835 images and worse on 98 (Wilcoxon p < 10⁻¹⁵).

**By source.** The gain holds in both image sets (Table 2).

**Table 2.** Selector and reference points by image source.

| source | images | always CATMO | selector | oracle |
|---|---|---|---|---|
| HDR-Real | 626 | 0.9303 | 0.9548 | 0.9580 |
| HDR-Synth | 366 | 0.9449 | 0.9677 | 0.9711 |

**Against the earlier approaches.**
- It is ahead of the project's original random-forest classifier by 0.004 TMQI.
- It is ahead of the best published single-paper feature set by 0.005 (+0.0041 to +0.0060).

**Figure 1** (`figures/fig1_mean_tmqi.png`) places the selector against every fixed operator.

### 4.2 Why low exact accuracy costs little TMQI

**Exact accuracy is a strict measure.** The selector picks the single best of 15 operators on 65% of
images. Exact accuracy counts every miss as a failure, however small, and **the top operators' scores are
often very close.**
- **The best two are close.**
  - The median difference between the best and the second-best operator on an image is **0.0062** TMQI,
    and the mean is 0.0095.
  - The two are within 0.005 of each other on **44%** of images, and within 0.01 on **67%**.
- **The spread across all operators is far larger.** The median difference between the best and the worst
  operator on an image is 0.168.

**Most misses are near misses** (Figure 3, `figures/fig3_pick_rank.png`):
- When the selector misses (345 images), its pick is the **second-best operator 58%** of the time, and
  among the top three 79% of the time.
- The median TMQI lost on a missed image is **0.0063**.

**Over all images the loss is small** (Figure 2, `figures/fig2_regret_ecdf.png`):
- Mean regret is **0.0033** TMQI, 0.34% of the oracle's score; the median is 0.
- The pick is within 0.005 TMQI of the best on **80%** of images, within 0.01 on **90%**, and within 0.02 on
  96%.
- For comparison, always-CATMO's mean regret is 0.0272, eight times larger, and it is within 0.005 of the
  best on only 18% of images.

**A hypothetical comparison makes the same point.** A selector that always took the *second-best* operator,
and so had 0% exact accuracy, would score 0.9533 on average. That is still 0.018 above always-CATMO.

> Exact accuracy is therefore a pessimistic summary. The selector loses most of its accuracy on images where
> two or more operators are practically tied, so a miss costs little TMQI.

**Remaining risk.** The gap is not always small. 4% of images lose more than 0.02 TMQI, and the worst single
loss is 0.196. In those cases the selector picks a clearly worse operator.

### 4.3 Computation time

**Table 3.** Wall-clock seconds per image, mean over 50 images (median in brackets).

| step | selector | exhaustive search |
|---|---|---|
| HDR features (151) | 0.12 (0.13) | — |
| model inference (225 LightGBM models) | 0.05 (0.05) | — |
| tone mapping | 0.92 (0.35) (1 operator) | 16.6 (15 operators) |
| TMQI scoring | — | 0.92 (15 outputs) |
| **total** | **1.09** | **17.5** |

**Selection is cheap.**
- Features and inference together take 0.17 s per image, less than running one classical operator.
- The selector is **16× faster** than exhaustive search on average (ratio of mean times; median
  per-image ratio 34×).
- It delivers 88% of the TMQI improvement that exhaustive search gives over always-CATMO.

**Exhaustive search is expensive.** Its cost is dominated by running all 15 operators, 16.6 s, and
above all the three learned ones (CATMO 2.7 s, UnCLTMO 4.9 s, Unpaired-HDR 3.9 s per image).
The 15 TMQI evaluations add another 0.92 s.

**The selector's cost depends on what it picks.** It is dominated by the one chosen operator.
- When it picks a classical operator or ATT, the total is 0.48 s on average (range 0.16–1.46 s). ATT, the
  most frequent pick, takes 0.05 s.
- When it picks one of the three learned operators, the total is 3.5 s on average (range 2.7–4.5 s). That
  happened on 10 of the 50 timed images, and on 12% of all 992 images.

**A practical alternative to the oracle.** The selector needs no HDR reference metric at run time, and no
outputs from operators it does not use. That makes per-image selection practical where exhaustive search is
not.

**Training.** Fitting the deployed model on all 992 images takes about 66 s once (8 parallel jobs). This is
negligible next to producing the training labels, which takes about 15–20 s of operator and metric time per
training image.

**Figure 4** (`figures/fig4_timing.png`) shows the breakdown.

*Caveat on timing.* The operator times include process start-up, as the benchmark runs them:
- container start for luminance-hdr;
- Python and model start for the learned operators.

In-process implementations would make every operator faster and narrow the ratio. They would not change the
basic difference: one operator against fifteen, and no metric against fifteen metric evaluations.

### 4.4 Robustness of the estimate

**Nested cross-validation.** It repeated both the feature-set choice and the model choice inside each outer
training fold.
- It chose the same configuration in all five folds.
- It reproduced the reported result exactly: 0.9596 TMQI, 65.2% exact accuracy.

**Split variation.** Across five random splits the selector averages 0.9597 ± 0.00015, with 66.5% exact
accuracy.

The reported gains are therefore not an artefact of choosing the configuration on the test images.

### 4.5 Other quality metrics

The same method, trained for a different metric, also beats the best fixed operator under that metric
(Table 4).

**Table 4.** The selector under the other two metrics.

| metric | best fixed operator | selector | oracle | gap recovered | exact accuracy |
|---|---|---|---|---|---|
| FSITM–TMQI average | 0.9196 (reinhard02) | 0.9309 | 0.9328 | 86% | 67.7% |
| TMQI-II | 0.8951 (reinhard02) | 0.9142 | 0.9281 | 58% | 61.1% |

## 5. Limitations

1. **TMQI is a proxy for human preference.** The gains are measured by the metric the selector was trained
   to maximise. A subjective study would be needed to confirm that the differences are visible. TMQI
   differences of about 0.005 are probably not noticeable.
2. **One benchmark.** The images are 512-px images from two collections, and the operators run with default
   parameters. Transfer to other content, resolutions or operator settings is untested.
3. **The hyperparameters were fixed earlier on these images.** This applies to the LightGBM settings and to
   the shortlist of models. Nested CV did not cover them.
4. **Timing is from one machine.** It also reflects the benchmark's process-per-operator execution.
5. **The tail.** About 4% of images lose more than 0.02 TMQI.

## 6. Conclusion

A selector that looks only at the HDR input recovers 88% of the TMQI available from choosing the best
operator per image:
- mean TMQI is 0.9596, against 0.9357 for the best fixed operator and 0.9629 for the oracle;
- selection costs 1.09 s per image, against 17.5 s for exhaustive search.

Its exact accuracy of 65% understates its quality. Most misses choose the second-best operator, which is
usually within 0.006 TMQI of the best. The mean loss against the oracle is 0.0033 TMQI.

## Data and code availability

**Dataset** (`dataset/`, described in `DATA_CARD.md`). For each of the 992 images it contains:
- the 151 features;
- the 10-bin log-luminance histogram;
- the TMQI of all 17 operators;
- the winning and second-best operator;
- the cross-validation folds used here;
- all metric components for every image × operator pair.

**Code.**
- `compare_selectors.py` (evaluation), `select_tmo.py` (the deployable selector) and `timing_benchmark.py`
  (timing).
- The feature and model study, including the nested cross-validation, is in
  `experiments/selector_features/`.

## References

Akyüz, A. O., & Reinhard, E. (2006). Color appearance in high-dynamic-range imaging. *Journal of Electronic Imaging*, 15(3), 033001. https://doi.org/10.1117/1.2238891

Ashikhmin, M. (2002). A tone mapping algorithm for high contrast images. In *Proc. 13th Eurographics Workshop on Rendering* (pp. 145–156). https://doi.org/10.2312/EGWR/EGWR02/145-156

Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123–140. https://doi.org/10.1007/BF00058655

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

Burges, C. J. C. (2010). *From RankNet to LambdaRank to LambdaMART: An overview* (Tech. Rep. MSR-TR-2010-82). Microsoft Research.

Burt, P. J., & Adelson, E. H. (1983). The Laplacian pyramid as a compact image code. *IEEE Transactions on Communications*, 31(4), 532–540. https://doi.org/10.1109/TCOM.1983.1095851

Čadík, M., Wimmer, M., Neumann, L., & Artusi, A. (2008). Evaluation of HDR tone mapping methods using essential perceptual attributes. *Computers & Graphics*, 32(3), 330–349. https://doi.org/10.1016/j.cag.2008.04.003

Cao, C., Yue, H., Liu, X., & Yang, J. (2024). Unsupervised HDR image and video tone mapping via contrastive learning. *IEEE Transactions on Circuits and Systems for Video Technology*, 34(2), 786–798. https://doi.org/10.1109/TCSVT.2023.3290351

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079–2107.

Cerdá-Company, X., Párraga, C. A., & Otazu, X. (2018). Which tone-mapping operator is the best? A comparative study of perceptual quality. *Journal of the Optical Society of America A*, 35(4), 626–638. https://doi.org/10.1364/JOSAA.35.000626

Drago, F., Myszkowski, K., Annen, T., & Chiba, N. (2003). Adaptive logarithmic mapping for displaying high contrast scenes. *Computer Graphics Forum*, 22(3), 419–426. https://doi.org/10.1111/1467-8659.00689

Durand, F., & Dorsey, J. (2002). Fast bilateral filtering for the display of high-dynamic-range images. *ACM Transactions on Graphics*, 21(3), 257–266. https://doi.org/10.1145/566654.566574

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.

Eilertsen, G., Mantiuk, R. K., & Unger, J. (2017). A comparative review of tone-mapping algorithms for high dynamic range video. *Computer Graphics Forum*, 36(2), 565–592. https://doi.org/10.1111/cgf.13148

Fairchild, M. D. (2007). The HDR photographic survey. In *Proc. 15th Color Imaging Conference* (pp. 233–238).

Fattal, R., Lischinski, D., & Werman, M. (2002). Gradient domain high dynamic range compression. *ACM Transactions on Graphics*, 21(3), 249–256. https://doi.org/10.1145/566654.566573

Funt, B., & Shi, L. (2010). The rehabilitation of MaxRGB. In *Proc. 18th Color and Imaging Conference* (pp. 256–259).

Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, 63(1), 3–42. https://doi.org/10.1007/s10994-006-6226-1

Hasler, D., & Süsstrunk, S. (2003). Measuring colourfulness in natural images. In *Human Vision and Electronic Imaging VIII, Proc. SPIE 5007* (pp. 87–95). https://doi.org/10.1117/12.477378

Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.

Hulusic, V., Debattista, K., Valenzise, G., & Dufaux, F. (2017). A model of perceived dynamic range for HDR images. *Signal Processing: Image Communication*, 51, 26–39. https://doi.org/10.1016/j.image.2016.11.005

ITU-T. (2023). *Recommendation P.910: Subjective video quality assessment methods for multimedia applications*. International Telecommunication Union.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30* (pp. 3146–3154).

Kerschke, P., Hoos, H. H., Neumann, F., & Trautmann, H. (2019). Automated algorithm selection: Survey and perspectives. *Evolutionary Computation*, 27(1), 3–45. https://doi.org/10.1162/evco_a_00242

Khan, I. R., Rahardja, S., Khan, M. M., Movania, M. M., & Abed, F. (2018). A tone-mapping technique based on histogram using a sensitivity model of the human visual system. *IEEE Transactions on Industrial Electronics*, 65(4), 3469–3479. https://doi.org/10.1109/TIE.2017.2760247

Kim, M. H., & Kautz, J. (2008). Consistent tone reproduction. In *Proc. 10th IASTED International Conference on Computer Graphics and Imaging (CGIM 2008)* (pp. 152–159).

Kundu, D., Ghadiyaram, D., Bovik, A. C., & Evans, B. L. (2017). No-reference quality assessment of tone-mapped HDR pictures. *IEEE Transactions on Image Processing*, 26(6), 2957–2971. https://doi.org/10.1109/TIP.2017.2685941

Ledda, P., Chalmers, A., Troscianko, T., & Seetzen, H. (2005). Evaluation of tone mapping operators using a high dynamic range display. *ACM Transactions on Graphics*, 24(3), 640–648. https://doi.org/10.1145/1073204.1073242

Li, D., Wang, M., & Rahardja, S. (2024). Contrastive learning for deep tone mapping operator. *Signal Processing: Image Communication*, 126, 117130. https://doi.org/10.1016/j.image.2024.117130

Lischinski, D., Farbman, Z., Uyttendaele, M., & Szeliski, R. (2006). Interactive local adjustment of tonal values. *ACM Transactions on Graphics*, 25(3), 646–653. https://doi.org/10.1145/1141911.1141936

Liu, Y.-L., Lai, W.-S., Chen, Y.-S., Kao, Y.-L., Yang, M.-H., Chuang, Y.-Y., & Huang, J.-B. (2020). Single-image HDR reconstruction by learning to reverse the camera pipeline. In *Proc. IEEE/CVF CVPR* (pp. 1648–1657). https://doi.org/10.1109/CVPR42600.2020.00172

Ma, K., Yeganeh, H., Zeng, K., & Wang, Z. (2015). High dynamic range image compression by optimizing tone mapped image quality index. *IEEE Transactions on Image Processing*, 24(10), 3086–3097. https://doi.org/10.1109/TIP.2015.2436340

Mai, Z., Mansour, H., Mantiuk, R., Nasiopoulos, P., Ward, R., & Heidrich, W. (2011). Optimizing a tone curve for backward-compatible high dynamic range image and video compression. *IEEE Transactions on Image Processing*, 20(6), 1558–1571. https://doi.org/10.1109/TIP.2010.2095866

Mantiuk, R., Daly, S., & Kerofsky, L. (2008). Display adaptive tone mapping. *ACM Transactions on Graphics*, 27(3), 68:1–68:10. https://doi.org/10.1145/1360612.1360667

Mantiuk, R., Myszkowski, K., & Seidel, H.-P. (2006). A perceptual framework for contrast processing of high dynamic range images. *ACM Transactions on Applied Perception*, 3(3), 286–308. https://doi.org/10.1145/1166087.1166095

Mantiuk, R. K., & Azimi, M. (2021). PU21: A novel perceptually uniform encoding for adapting existing quality metrics for HDR. In *Picture Coding Symposium (PCS)* (pp. 1–5). https://doi.org/10.1109/PCS50896.2021.9477471

Mittal, A., Moorthy, A. K., & Bovik, A. C. (2012). No-reference image quality assessment in the spatial domain. *IEEE Transactions on Image Processing*, 21(12), 4695–4708. https://doi.org/10.1109/TIP.2012.2214050

Nafchi, H. Z., Shahkolaei, A., Moghaddam, R. F., & Cheriet, M. (2015). FSITM: A feature similarity index for tone-mapped images. *IEEE Signal Processing Letters*, 22(8), 1026–1029. https://doi.org/10.1109/LSP.2014.2381458

Pattanaik, S. N., Tumblin, J., Yee, H., & Greenberg, D. P. (2000). Time-dependent visual adaptation for fast realistic image display. In *Proc. SIGGRAPH 2000* (pp. 47–54). https://doi.org/10.1145/344779.344810

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Pouli, T., Cunningham, D. W., & Reinhard, E. (2010). Image statistics and their applications in computer graphics. In *Eurographics 2010 – State of the Art Reports* (pp. 83–112). https://doi.org/10.2312/egst.20101063

Reinhard, E. (2002). Parameter estimation for photographic tone reproduction. *Journal of Graphics Tools*, 7(1), 45–51. https://doi.org/10.1080/10867651.2002.10487554

Reinhard, E., Ashikhmin, M., Gooch, B., & Shirley, P. (2001). Color transfer between images. *IEEE Computer Graphics and Applications*, 21(5), 34–41. https://doi.org/10.1109/38.946629

Reinhard, E., & Devlin, K. (2005). Dynamic range reduction inspired by photoreceptor physiology. *IEEE Transactions on Visualization and Computer Graphics*, 11(1), 13–24. https://doi.org/10.1109/TVCG.2005.9

Reinhard, E., Heidrich, W., Debevec, P., Pattanaik, S., Ward, G., & Myszkowski, K. (2010). *High dynamic range imaging: Acquisition, display, and image-based lighting* (2nd ed.). Morgan Kaufmann.

Reinhard, E., Stark, M., Shirley, P., & Ferwerda, J. (2002). Photographic tone reproduction for digital images. *ACM Transactions on Graphics*, 21(3), 267–276. https://doi.org/10.1145/566654.566575

Rice, J. R. (1976). The algorithm selection problem. *Advances in Computers*, 15, 65–118. https://doi.org/10.1016/S0065-2458(08)60520-3

Ruderman, D. L., Cronin, T. W., & Chiao, C.-C. (1998). Statistics of cone responses to natural images: Implications for visual coding. *Journal of the Optical Society of America A*, 15(8), 2036–2045. https://doi.org/10.1364/JOSAA.15.002036

van Hateren, J. H. (2006). Encoding of high dynamic range video with a model of human cones. *ACM Transactions on Graphics*, 25(4), 1380–1399. https://doi.org/10.1145/1183287.1183293

Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91. https://doi.org/10.1186/1471-2105-7-91

Vinker, Y., Huberman-Spiegelglas, I., & Fattal, R. (2021). Unpaired learning for high dynamic range image tone mapping. In *Proc. IEEE/CVF ICCV* (pp. 14637–14646). https://doi.org/10.1109/ICCV48922.2021.01439

Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13(4), 600–612. https://doi.org/10.1109/TIP.2003.819861

Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80–83. https://doi.org/10.2307/3001968

Xiao, F., DiCarlo, J. M., Catrysse, P. B., & Wandell, B. A. (2002). High dynamic range imaging of natural scenes. In *Proc. 10th Color Imaging Conference* (pp. 337–342).

Yeganeh, H., & Wang, Z. (2013). Objective quality assessment of tone-mapped images. *IEEE Transactions on Image Processing*, 22(2), 657–667. https://doi.org/10.1109/TIP.2012.2221725

**Software.**
- Luminance HDR 2.6.1.1, https://github.com/LuminanceHDR/LuminanceHDR.
- tmqi-revised (Python port of TMQI), https://github.com/dvolgyes/TMQI.
- LightGBM 4.6.0, scikit-learn 1.7.2.
- Operator code: https://github.com/xslidi/CATMO, https://github.com/cao-cong/UnCLTMO and
  https://github.com/yael-vinker/unpaired_hdr_tmo.
