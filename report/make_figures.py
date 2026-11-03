"""
Figures and the numbers quoted in REPORT_DRAFT.md, from the checked-in results.

    python report/make_figures.py      (from the TMO/ folder)

Reads results/q_matrix.csv, results/selectors/Q/selector_predictions.csv (the production out-of-fold run,
seed 1234, identical to the nested-CV estimate), experiments/selector_features/tables/ and
results/timing/. Writes report/figures/*.png, report/report_numbers.json and report/TABLES_AND_FIGURES.md
(Tables 1-6 and the figure captions, numbered as in the final report).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "savefig.dpi": 200})

qm = pd.read_csv(os.path.join(RES, "q_matrix.csv")).set_index("image")
pred = pd.read_csv(os.path.join(RES, "selectors", "Q", "selector_predictions.csv")).set_index("image")
wins = pd.read_csv(os.path.join(RES, "selectors", "Q", "tmo_wins.csv")).set_index("operator")
cand = [o for o in qm.columns if wins.wins[o] > 0 or o == "catmo"]
Q = qm.loc[pred.index, cand]
srt = np.sort(Q.to_numpy(), axis=1)[:, ::-1]
oracle, second = srt[:, 0], srt[:, 1]
q_pick = pred.Q_pick.to_numpy()
q_base = pred.Q_baseline.to_numpy()
reg_sel, reg_base, gap12 = oracle - q_pick, oracle - q_base, oracle - second
rank = np.array([(Q.loc[i] > Q.loc[i, p]).sum() + 1 for i, p in zip(pred.index, pred.pick)])
wrong = rank > 1

N = {}
N["n_images"], N["n_candidates"], N["n_operators"] = len(Q), len(cand), qm.shape[1]
N["mean_Q"] = dict(oracle=oracle.mean(), selector=q_pick.mean(), always_catmo=q_base.mean(),
                   always_second_best=second.mean(), random=Q.mean(axis=1).mean(),
                   per_operator=qm.loc[pred.index].mean().sort_values(ascending=False).round(4).to_dict())
N["gap_recovered"] = (q_pick.mean() - q_base.mean()) / (oracle.mean() - q_base.mean())
N["exact_accuracy"] = float((~wrong).mean())
N["picked_rank_counts"] = {int(k): int(v) for k, v in pd.Series(rank).value_counts().sort_index().items()}
N["wrong_picks"] = dict(n=int(wrong.sum()), share_second_best=float((rank[wrong] == 2).mean()),
                        share_top3=float((rank[wrong] <= 3).mean()), mean_regret=reg_sel[wrong].mean(),
                        median_regret=float(np.median(reg_sel[wrong])))
N["regret_selector"] = dict(mean=reg_sel.mean(), median=float(np.median(reg_sel)),
                            p90=float(np.percentile(reg_sel, 90)), max=reg_sel.max(),
                            within_0005=float((reg_sel <= 0.005).mean()), within_001=float((reg_sel <= 0.01).mean()),
                            within_002=float((reg_sel <= 0.02).mean()), relative_mean=float((reg_sel / oracle).mean()))
N["regret_always_catmo"] = dict(mean=reg_base.mean(), median=float(np.median(reg_base)),
                                within_0005=float((reg_base <= 0.005).mean()))
N["best_minus_second"] = dict(mean=gap12.mean(), median=float(np.median(gap12)),
                              p75=float(np.percentile(gap12, 75)), p90=float(np.percentile(gap12, 90)),
                              within_0005=float((gap12 <= 0.005).mean()), within_001=float((gap12 <= 0.01).mean()))
N["best_minus_worst_median"] = float(np.median(srt[:, 0] - srt[:, -1]))
N["selector_vs_catmo_images"] = dict(better=int((q_pick > q_base).sum()), worse=int((q_pick < q_base).sum()),
                                     equal=int((q_pick == q_base).sum()))
N["by_source"] = pd.read_csv(os.path.join(RES, "selectors", "Q", "selector_by_dataset.csv")).round(5) \
    .to_dict(orient="records")

NAMES = {"reinhard02": "Reinhard 2002", "reinhard05": "Reinhard–Devlin", "drago": "Drago", "durand": "Durand",
         "fattal": "Fattal", "mantiuk06": "Mantiuk 2006", "mantiuk08": "Mantiuk 2008", "pattanaik": "Pattanaik",
         "ashikhmin": "Ashikhmin", "mai": "Mai", "lischinski": "Lischinski", "vanhateren": "van Hateren",
         "kimkautz": "Kim–Kautz", "att": "ATT", "catmo": "CATMO", "uncltmo": "UnCLTMO", "unpaired_hdr": "Unpaired-HDR"}
LEARNED = {"catmo", "uncltmo", "unpaired_hdr"}

# ---- Fig 1: mean TMQI, fixed operators vs selector vs oracle (dot plot) ----
per_op = qm.loc[pred.index].mean().sort_values()
rows = [(NAMES[o], v, GREY) for o, v in per_op.items()]
rows += [("random operator", Q.mean(axis=1).mean(), GREY),
         ("selector (this work)", q_pick.mean(), BLUE), ("ceiling (best per image)", oracle.mean(), INK)]
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(6.4, 5.6))
for y, (name, v, c) in enumerate(rows):
    ax.plot([0.70, v], [y, y], color="#e6e5e1", lw=1, zorder=1)
    ax.scatter(v, y, s=46, color=c, zorder=3, edgecolor="white", linewidth=1.5)
    ax.text(v + 0.003, y, f"{v:.4f}", va="center", fontsize=8, color=INK if c != GREY else MUTED)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.set_xlim(0.70, 0.995)
ax.set_xlabel("mean TMQI over 992 images (axis starts at 0.70)")
ax.grid(axis="y", visible=False)
ax.set_title("Per-image selection vs every fixed operator", loc="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_mean_tmqi.png"))
plt.close(fig)

# ---- Fig 2: ECDF of TMQI lost against the per-image best ----
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for v, c, lab in [(reg_sel, BLUE, f"selector (mean {reg_sel.mean():.4f})"),
                  (reg_base, GREY, f"always CATMO (mean {reg_base.mean():.4f})")]:
    x = np.sort(v)
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=c, lw=2, label=lab)
for thr in (0.005, 0.01):
    ax.axvline(thr, color=MUTED, lw=0.8, ls="--")
    ax.text(thr - 0.0004, 0.5, f"regret {thr:g}", color=MUTED, fontsize=8, rotation=90, ha="right", va="center")
    for v, c in [(reg_sel, BLUE), (reg_base, GREY)]:
        s = (v <= thr).mean()
        ax.scatter(thr, s, s=22, color=c, zorder=4, edgecolor="white", linewidth=1)
        ax.text(thr + 0.0008, s - 0.05, f"{s:.0%}", color=c if c != GREY else MUTED, fontsize=8)
ax.set_xlim(0, 0.08)
ax.set_ylim(0, 1.01)
ax.set_xlabel("TMQI lost against the best operator for that image (regret)")
ax.set_ylabel("share of images")
ax.legend(frameon=False, loc="lower right")
ax.set_title("Regret per image, cumulative (992 images)", loc="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_regret_ecdf.png"))
plt.close(fig)

# ---- Fig 3: true rank of the picked operator on the missed images ----
cnt = pd.Series(np.minimum(rank[wrong], 6)).value_counts().reindex(range(2, 7), fill_value=0)
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.bar(range(2, 7), cnt.values, width=0.7, color=[BLUE] + [GREY] * 4)
for x, v in zip(range(2, 7), cnt.values):
    ax.text(x, v + 3, f"{v} ({v / wrong.sum():.0%})", ha="center", fontsize=8, color=INK)
ax.set_xticks(range(2, 7), ["2nd", "3rd", "4th", "5th", "6th or lower"])
ax.set_xlabel("true TMQI rank of the picked operator (of 15)")
ax.set_ylabel("images")
ax.grid(axis="x", visible=False)
ax.set_title(f"Rank of the selector's pick on the {wrong.sum()} missed images", loc="left", fontsize=11,
             color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_pick_rank.png"))
plt.close(fig)

# ---- Fig 4: timing ----
tdir = os.path.join(RES, "timing")
if os.path.exists(os.path.join(tdir, "timing_per_image.csv")):
    t = pd.read_csv(os.path.join(tdir, "timing_per_image.csv"))
    per = pd.read_csv(os.path.join(tdir, "timing_per_operator.csv")).set_index("operator")
    N["timing"] = dict(n_images=len(t), **{c: dict(mean=t[c].mean(), median=float(t[c].median()))
                                           for c in ["features", "predict", "picked_tmo", "selector_total",
                                                     "exhaustive_tmo", "exhaustive_tmqi", "exhaustive_total",
                                                     "speedup"]},
                       per_operator=per.round(4).to_dict(orient="index"),
                       speedup_of_means=t.exhaustive_total.mean() / t.selector_total.mean())
    sel = [("HDR features", t.features.mean(), BLUE), ("model inference", t.predict.mean(), "#86b6ef"),
           ("tone mapping", t.picked_tmo.mean(), ORANGE)]
    exh = [("tone mapping", t.exhaustive_tmo.mean(), ORANGE), ("TMQI scoring", t.exhaustive_tmqi.mean(), GREY)]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    for y, parts in [(1, sel), (0, exh)]:
        left = 0
        for name, v, c in parts:
            ax.barh(y, v, left=left, color=c, height=0.5, edgecolor="white", linewidth=0.8, label=name)
            left += v
        ax.text(left + 0.2, y, f"{left:.2f} s", va="center", fontsize=9, color=INK)
    ax.text(t.selector_total.mean() + 1.9, 1, f"(features + inference {t.features.mean() + t.predict.mean():.2f} s, "
            f"then 1 operator)", va="center", fontsize=8, color=MUTED)
    ax.set_yticks([1, 0], ["selector", "exhaustive search\n(15 operators)"])
    ax.set_xlabel("wall-clock seconds per image (mean of %d images)" % len(t))
    ax.grid(axis="y", visible=False)
    h, l = ax.get_legend_handles_labels()
    keep = dict(zip(l, h))
    ax.legend(keep.values(), keep.keys(), frameon=False, fontsize=8, ncol=4, loc="upper center",
              bbox_to_anchor=(0.45, -0.3))
    ax.set_title("Cost of choosing a TMO for one image", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_timing.png"), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ---- Tables 1-6 -> report/TABLES_AND_FIGURES.md ----
def pct(x, d=0):
    return f"{100 * x:.{d}f}%".replace("-", "−")


OPS = [  # (code, reference, group, implementation)
    ("reinhard02", "Reinhard et al. 2002", "classical", "luminance-hdr"),
    ("reinhard05", "Reinhard & Devlin 2005", "classical", "luminance-hdr"),
    ("drago", "Drago et al. 2003", "classical", "luminance-hdr"),
    ("durand", "Durand & Dorsey 2002", "classical", "luminance-hdr"),
    ("fattal", "Fattal et al. 2002", "classical", "luminance-hdr"),
    ("mantiuk06", "Mantiuk et al. 2006", "classical", "luminance-hdr"),
    ("mantiuk08", "Mantiuk et al. 2008", "classical", "luminance-hdr"),
    ("pattanaik", "Pattanaik et al. 2000", "classical", "luminance-hdr"),
    ("ashikhmin", "Ashikhmin 2002", "classical", "luminance-hdr"),
    ("mai", "Mai et al. 2011", "classical", "luminance-hdr"),
    ("lischinski", "Lischinski et al. 2006", "classical", "luminance-hdr"),
    ("vanhateren", "van Hateren 2006", "classical", "luminance-hdr"),
    ("kimkautz", "Kim & Kautz 2008", "classical", "luminance-hdr"),
    ("att", "Khan et al. 2018", "ATT", "Python port of the authors' MATLAB code"),
    ("catmo", "Li et al. 2024", "learned", "published weights"),
    ("uncltmo", "Cao et al. 2024", "learned", "published weights"),
    ("unpaired_hdr", "Vinker et al. 2021", "learned", "published weights"),
]
LHDR_POSTGAMMA = {"reinhard02", "reinhard05", "drago", "fattal", "pattanaik", "ashikhmin", "lischinski", "kimkautz"}

FEATURE_GROUPS = [  # (category, group key, source, what it measures)
    ("HDR image statistics", "pouli10", "Pouli et al. 2010",
     "log-luminance moments, power-spectrum slope, gradient and Laplacian-band statistics, Lαβ colour"),
    ("HDR image statistics", "hulusic17", "Hulusic et al. 2017",
     "perceived dynamic range, image key, bright area, PU21 contrast, colourfulness"),
    ("operator set-up statistics", "reinhard02", "Reinhard et al. 2002; Reinhard 2002",
     "log-average luminance, default white point, automatic key and white-point estimates"),
    ("operator set-up statistics", "drago03", "Drago et al. 2003", "max / log-average ratio, log-base divider"),
    ("operator set-up statistics", "durand02", "Durand & Dorsey 2002",
     "bilateral base-layer range, compression factor, detail energy"),
    ("operator set-up statistics", "fattal02", "Fattal et al. 2002",
     "average gradient and attenuation at the attenuated pyramid levels"),
    ("operator set-up statistics", "reinhard05", "Reinhard & Devlin 2005", "image key k, contrast m"),
    ("operator set-up statistics", "kimkautz08", "Kim & Kautz 2008", "log-mean, global compression k1"),
    ("operator set-up statistics", "mai11", "Mai et al. 2011",
     "log-histogram normaliser Σp^(1/3), largest tone-curve slope share"),
    ("operator set-up statistics", "mantiuk08", "Mantiuk et al. 2008",
     "band-contrast distribution per pyramid level, conditional on background"),
    ("quality-model statistics", "tmqi", "Yeganeh & Wang 2013",
     "TMQI's HDR-side local-σ visibility at its 5 scales"),
    ("quality-model statistics", "brisque", "Mittal et al. 2012 (Kundu et al. 2017)",
     "BRISQUE natural-scene statistics of log luminance, 2 scales"),
    ("quality-model statistics", "si", "ITU-T P.910", "spatial information of log luminance"),
    ("quality-model statistics", "tmqiblock", "Yeganeh & Wang 2013 (adapted)",
     "11 / 22 / 44-px block statistics of log luminance"),
]

L = ["# Tables and figures", "",
     "*Generated by `report/make_figures.py` from the checked-in results; do not edit by hand. TMQI numbers are "
     "out-of-fold, seed 1234, over 992 images, unless stated otherwise.*", ""]

# Table 1: operators
L += ["**Table 1.** The 17 tone-mapping operators, run with their default parameters.", "",
      "| operator | reference | type | implementation |", "|---|---|---|---|"]
for code, ref, grp, impl in OPS:
    mark = ("†" if code in LHDR_POSTGAMMA else "") + ("*" if wins.wins[code] == 0 else "")
    L.append(f"| {NAMES[code]}{mark} | {ref} | {grp} | {impl} |")
L += ["", "luminance-hdr = luminance-hdr-cli 2.6.1.1 in Docker. "
      "† writes linear luminance; display-encoded with gamma 2.2. "
      "\\* never the best operator on any image under TMQI, so not one of the 15 candidates.", ""]

# Table 2: feature groups
fg = pd.read_csv(os.path.join(RES, "features_selector_groups.csv")).group.value_counts()
assert set(fg.index) == {g for _, g, _, _ in FEATURE_GROUPS}
L += [f"**Table 2.** The {fg.sum()} selector features in {len(fg)} groups, each following one published "
      "source. All are computed from the HDR input alone.", "",
      "| category | source | features | what it measures |", "|---|---|---|---|"]
for cat, g, src, what in FEATURE_GROUPS:
    L.append(f"| {cat} | {src} | {fg[g]} | {what} |")
L += [f"| **total** | | **{fg.sum()}** | |", "",
      "Operator set-up statistics are the quantities each operator computes from its input before mapping, "
      "following the luminance-hdr code with its defaults; the operator itself is never run.", ""]

# Table 3: main comparison
cmp_q = pd.read_csv(os.path.join(RES, "selectors", "Q", "selector_comparison.csv")).set_index("method")
cmp_p = pd.read_csv(os.path.join(RES, "selectors", "Q_pouli10", "selector_comparison.csv")).set_index("method")
runs = pd.read_csv(os.path.join(ROOT, "experiments", "selector_features", "tables", "runs.csv"))
rf = runs[(runs.metric == "Q") & (runs.fset == "hdr22") & (runs.model == "rf_clf") & (runs.seed == 1234)].iloc[0]
rand_near = float((Q.to_numpy() >= oracle[:, None] - 0.005).mean(axis=1).mean())
sel_row = cmp_q.loc["structured LightGBM selector"]
pou_row = cmp_p.loc["structured LightGBM selector"]
base_row, rnd_row = cmp_q.loc["always-catmo"], cmp_q.loc["random pick (expected)"]
L += ["**Table 3.** Mean TMQI of the chosen operator over 992 images. Gap recovered = (method − always CATMO) / "
      "(ceiling − always CATMO).", "",
      "| method | mean TMQI | gap recovered | exact accuracy | within 0.005 of best |", "|---|---|---|---|---|",
      f"| random operator (expected) | {rnd_row.mean_Q:.4f} | {pct(rnd_row.gap_recovered)} | "
      f"{pct(rnd_row.exact_accuracy, 1)} | {pct(rand_near, 1)} |",
      f"| always CATMO (best fixed operator) | {base_row.mean_Q:.4f} | 0% | {pct(base_row.exact_accuracy, 1)} | "
      f"{pct(base_row.near_optimal_rate, 1)} |",
      f"| selector, Pouli et al. 2010 features only (27) | {pou_row.mean_Q:.4f} | {pct(pou_row.gap_recovered)} | "
      f"{pct(pou_row.exact_accuracy, 1)} | {pct(pou_row.near_optimal_rate, 1)} |",
      f"| earlier random-forest classifier ({int(rf.n_features)} hand-made features) | {rf.mean_Q:.4f} | "
      f"{pct(rf.gap_rec)} | {pct(rf.exact, 1)} | {pct(rf.near_005, 1)} |",
      f"| **final selector ({fg.sum()} features, structured LightGBM)** | **{sel_row.mean_Q:.4f}** | "
      f"**{pct(sel_row.gap_recovered)}** | **{pct(sel_row.exact_accuracy, 1)}** | "
      f"**{pct(sel_row.near_optimal_rate, 1)}** |",
      f"| ceiling (best operator per image) | {oracle.mean():.4f} | 100% | 100% | 100% |", ""]

# Table 4: by source
L += ["**Table 4.** Mean TMQI by image source.", "",
      "| source | images | always CATMO | selector | ceiling | gap recovered | exact accuracy |",
      "|---|---|---|---|---|---|---|"]
for r in N["by_source"]:
    name = "**all**" if r["source"] == "all" else r["source"]
    L.append(f"| {name} | {r['n_images']} | {r['baseline_Q']:.4f} | {r['selector_Q']:.4f} | {r['oracle_Q']:.4f} | "
             f"{pct(r['gap_recovered'])} | {pct(r['exact_accuracy'], 1)} |")
L.append("")

# Table 5: timing
if "timing" in N:
    def mm(c, d=2):
        return f"{t[c].mean():.{d}f} ({t[c].median():.{d}f})"
    lrn = t.picked.isin(LEARNED)
    L += [f"**Table 5.** Wall-clock seconds per image: mean over {len(t)} images (median in brackets).", "",
          "| step | selector | exhaustive search |", "|---|---|---|",
          f"| HDR features ({fg.sum()}) | {mm('features')} | — |",
          f"| model inference ({len(cand)} × 3 × 5 = {len(cand) * 15} LightGBM models) | {mm('predict')} | — |",
          f"| tone mapping | {mm('picked_tmo')}, 1 operator | {mm('exhaustive_tmo', 1)}, {len(cand)} operators |",
          f"| TMQI scoring | — | {mm('exhaustive_tmqi')}, {len(cand)} outputs |",
          f"| **total** | **{mm('selector_total')}** | **{mm('exhaustive_total', 1)}** |",
          f"| selector total, classical or ATT pick ({(~lrn).sum()} images) | "
          f"{t.selector_total[~lrn].mean():.2f} (range {t.selector_total[~lrn].min():.2f}–"
          f"{t.selector_total[~lrn].max():.2f}) | |",
          f"| selector total, learned-operator pick ({lrn.sum()} images) | "
          f"{t.selector_total[lrn].mean():.2f} (range {t.selector_total[lrn].min():.2f}–"
          f"{t.selector_total[lrn].max():.2f}) | |", "",
          f"Speed-up: {N['timing']['speedup_of_means']:.0f}× (ratio of mean times); "
          f"{t.speedup.median():.0f}× on the median image. Times include each operator's process start-up, "
          "as the benchmark runs them. Apple M3 Pro, one image at a time.", ""]

# Table 6: other metrics
L += ["**Table 6.** The same method trained and scored under each metric. The floor is the best fixed operator "
      "under that metric.", "",
      "| metric | best fixed operator | selector | ceiling | gap recovered | exact accuracy |",
      "|---|---|---|---|---|---|"]
for key, label in [("Q", "TMQI (reference)"), ("FSITM_TMQI", "FSITM–TMQI average"), ("Q2", "TMQI-II")]:
    c = pd.read_csv(os.path.join(RES, "selectors", key, "selector_comparison.csv")).set_index("method")
    floor = [m for m in c.index if m.startswith("always-")][0]
    s = c.loc["structured LightGBM selector"]
    L.append(f"| {label} | {c.loc[floor].mean_Q:.4f} ({NAMES[floor[len('always-'):]]}) | {s.mean_Q:.4f} | "
             f"{c.loc['oracle'].mean_Q:.4f} | {pct(s.gap_recovered)} | {pct(s.exact_accuracy, 1)} |")
L.append("")

# Figure captions
L += ["**Figure 1.** Mean TMQI over 992 images of every fixed operator, a random operator, the selector and the "
      "ceiling (best operator per image).", "", "![Figure 1](figures/fig1_mean_tmqi.png)", "",
      "**Figure 2.** Cumulative distribution of regret (TMQI lost against the best operator for each image), "
      "selector vs always CATMO. Dots mark the share of images within 0.005 and 0.01 of the best.", "",
      "![Figure 2](figures/fig2_regret_ecdf.png)", "",
      f"**Figure 3.** True TMQI rank of the operator the selector picked, on the {wrong.sum()} images where it "
      "did not pick the best one.", "", "![Figure 3](figures/fig3_pick_rank.png)", "",
      "**Figure 4.** Wall-clock time per image, by step: selector vs exhaustive search over the 15 candidate "
      "operators.", "", "![Figure 4](figures/fig4_timing.png)", ""]
# ---- Benchmark section: Tables B1-B3, Figures B1-B2 ----
from scipy.stats import kendalltau

qa = qm.loc[pred.index]
rk = qa.rank(axis=1, ascending=False, method="min")
summ = pd.read_csv(os.path.join(RES, "summary.csv"))
summ = summ[summ.basis == "intersection"].set_index("operator")
tper = pd.read_csv(os.path.join(RES, "timing", "timing_per_operator.csv")).set_index("operator")
otype = {code: grp for code, _, grp, _ in OPS}
order = qa.mean().sort_values(ascending=False).index
nimg = len(qa)

L += ["", "---", "", "# Benchmark section", "",
      f"*{qm.shape[1]} operators × {nimg} images, each output scored against the original HDR. Insert before the "
      "selector results; the B labels are placeholders until the final numbering.*", "",
      f"**Table B1.** Every operator under TMQI over {nimg} images, sorted by mean TMQI. S = structural fidelity, "
      "N = statistical naturalness (TMQI's two terms). Rank is among all 17 operators on each image. "
      "Time = mean seconds per image over the 50 timed images, including process start-up.", "",
      "| operator | type | mean TMQI | S | N | best on | top 5 on | bottom half on | mean rank | time (s) |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for o in order:
    tm = f"{tper.tmo_mean[o]:.2f}" if o in tper.index else "not timed"
    L.append(f"| {NAMES[o]} | {otype[o]} | {qa[o].mean():.4f} | {summ.S_mean[o]:.3f} | {summ.N_mean[o]:.3f} | "
             f"{int(wins.wins[o])} ({pct(wins.wins[o] / nimg)}) | {pct((rk[o] <= 5).mean())} | "
             f"{pct((rk[o] >= 9).mean())} | {rk[o].mean():.1f} | {tm} |")
L += ["", "Bottom half = ranked 9th–17th of 17. Pattanaik and van Hateren were not timed because they are not "
      "candidates.", ""]

# Table B2: metric disagreement
mats = {}
for key in ["Q", "FSITM_TMQI", "Q2", "FSITM"]:
    f = "q_matrix.csv" if key == "Q" else f"q_matrix_{key}.csv"
    mats[key] = pd.read_csv(os.path.join(RES, f)).set_index("image").loc[qa.index, qa.columns]
MLAB = {"Q": "TMQI", "FSITM_TMQI": "FSITM–TMQI average", "Q2": "TMQI-II", "FSITM": "FSITM"}
L += ["**Table B2.** The answer depends on the metric. Agreement columns compare each image's best operator "
      "and full operator ranking with TMQI's.", "",
      "| metric | best fixed operator (mean) | most frequent winner | ceiling | same best operator as TMQI | "
      "median Kendall τ with TMQI |", "|---|---|---|---|---|---|"]
mstat = {}
for key, M in mats.items():
    mw = M.idxmax(axis=1).value_counts()
    same = (M.idxmax(axis=1) == mats["Q"].idxmax(axis=1)).mean()
    tau = np.median([kendalltau(mats["Q"].loc[i], M.loc[i])[0] for i in M.index])
    mstat[key] = dict(best_fixed=M.mean().idxmax(), same_winner=float(same), median_tau=float(tau))
    agree = "—" if key == "Q" else pct(same)
    taus = "—" if key == "Q" else f"{tau:.2f}"
    L.append(f"| {MLAB[key]} | {NAMES[M.mean().idxmax()]} ({M.mean().max():.4f}) | {NAMES[mw.index[0]]} "
             f"({pct(mw.iloc[0] / nimg)}) | {M.max(axis=1).mean():.4f} | {agree} | {taus} |")
L.append("")

# Table B3: by source
man = pd.read_csv(os.path.join(RES, "manifest.csv")).set_index("image")
srcs = man.dataset.reindex(qa.index).str.split("/").str[0]
w_img = qa.idxmax(axis=1)
L += ["**Table B3.** The five best operators on average, by image source: mean TMQI and number of images won.",
      "", "| operator | HDR-Real mean | HDR-Real wins | HDR-Synth mean | HDR-Synth wins |", "|---|---|---|---|---|"]
for o in order[:5]:
    cells = []
    for s in ["HDR-Real", "HDR-Synth"]:
        m = srcs == s
        cells += [f"{qa.loc[m, o].mean():.4f}", f"{(w_img[m] == o).sum()} of {m.sum()}"]
    L.append(f"| {NAMES[o]} | " + " | ".join(cells) + " |")
L.append("")

N["benchmark"] = dict(
    rank_top5={o: float((rk[o] <= 5).mean()) for o in ["att", "catmo", "reinhard02"]},
    rank_bottom_half={o: float((rk[o] >= 9).mean()) for o in ["att", "catmo", "reinhard02"]},
    rank_last={o: int((rk[o] == rk.shape[1]).sum()) for o in ["att", "catmo"]},
    rank_worst={o: int(rk[o].max()) for o in ["att", "catmo"]},
    metrics=mstat)

# Figure B1: S vs N per operator
TCOL = {"classical": GREY, "ATT": ORANGE, "learned": BLUE}
LABEL_AT = {"drago": (-6, 5, "right"), "mai": (6, 7, "left"), "uncltmo": (8, -1, "left"),
            "kimkautz": (-6, -9, "right"), "unpaired_hdr": (-7, -1, "right"), "durand": (4, -13, "left"),
            "fattal": (6, -11, "left"), "mantiuk06": (6, 4, "left")}
fig, ax = plt.subplots(figsize=(6.6, 4.6))
for o in qa.columns:
    s, n = summ.S_mean[o], summ.N_mean[o]
    dx, dy, ha = LABEL_AT.get(o, (5, 3, "left"))
    ax.scatter(s, n, s=50, color=TCOL[otype[o]], zorder=3, edgecolor="white", linewidth=1.2)
    ax.annotate(f"{NAMES[o]} ({qa[o].mean():.3f})", (s, n), xytext=(dx, dy), textcoords="offset points",
                fontsize=7.5, ha=ha, va="center", color=INK if otype[o] != "classical" else MUTED)
for t, c in TCOL.items():
    ax.scatter([], [], color=c, s=40, label=t)
ax.set_xlabel("mean structural fidelity S")
ax.set_ylabel("mean statistical naturalness N")
ax.set_xlim(right=ax.get_xlim()[1] + 0.03)
ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.set_title("TMQI's two terms per operator (mean TMQI in brackets)", loc="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "figB1_s_vs_n.png"))
plt.close(fig)

# Figure B2: rank distribution, ATT vs CATMO
fig, ax = plt.subplots(figsize=(6.6, 3.4))
ranks = np.arange(1, rk.shape[1] + 1)
for dx, o, c in [(-0.2, "att", ORANGE), (0.2, "catmo", BLUE)]:
    cnt = rk[o].value_counts().reindex(ranks, fill_value=0)
    ax.bar(ranks + dx, cnt.values / nimg, width=0.4, color=c,
           label=f"{NAMES[o]}: best on {pct((rk[o] == 1).mean())}, top 5 on {pct((rk[o] <= 5).mean())}, "
                 f"bottom half on {pct((rk[o] >= 9).mean())}")
ax.axvline(8.5, ymax=0.78, color=MUTED, lw=0.8, ls="--")
ax.text(8.65, ax.get_ylim()[1] * 0.5, "bottom half →", color=MUTED, fontsize=8)
ax.set_xticks(ranks)
ax.set_xlabel("TMQI rank of the operator on an image (1 = best of 17)")
ax.set_ylabel("share of images")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("ATT wins often but is erratic; CATMO rarely wins but is reliable", loc="left", fontsize=11,
             color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "figB2_rank_att_catmo.png"))
plt.close(fig)

L += ["**Figure B1.** Mean structural fidelity S against mean naturalness N for each operator. The operators "
      "with the highest S rank low on TMQI because their naturalness is poor.", "",
      "![Figure B1](figures/figB1_s_vs_n.png)", "",
      "**Figure B2.** How often ATT and CATMO take each TMQI rank among the 17 operators.", "",
      "![Figure B2](figures/figB2_rank_att_catmo.png)", ""]

open(os.path.join(ROOT, "report", "TABLES_AND_FIGURES.md"), "w").write("\n".join(L))


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return round(float(o), 6) if isinstance(o, (float, np.floating)) else o


json.dump(clean(N), open(os.path.join(ROOT, "report", "report_numbers.json"), "w"), indent=1)
print(json.dumps(clean({k: N[k] for k in ("mean_Q", "exact_accuracy", "wrong_picks", "regret_selector",
                                           "best_minus_second")}), indent=1))
