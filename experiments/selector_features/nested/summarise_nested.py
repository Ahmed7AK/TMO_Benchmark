"""
Pool the outer folds of the nested CV (folds/fold<i>.json + cache/) into the nested estimate.

    ../../../.venv/bin/python summarise_nested.py

Rules scored (each fold's choice is fit on its outer training split and scores its outer test fold):
  primary       all groups unless the greedy subset is significantly better (CI above 0), then the best model
  fset_argmax   the higher inner mean Q of all groups / greedy subset (struct_lgbm), then the best model
  joint_argmax  the best of all inner (feature set, model) cells
References on the same outer folds: the fixed all-groups + struct_lgbm configuration, always-catmo, and the
selection-conditional study numbers (all groups + struct_lgbm, 5 seeds, from the study's oof/ cache).
Writes tables/nested_{summary,paired,choices,folds}.csv.
"""
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

import nested_lib as L

C = L.C
RULES = ["primary", "fset_argmax", "joint_argmax"]


def main():
    D = L.data(L.ALL)
    fo = L.outer_folds()
    folds = [json.load(open(os.path.join(L.OUT, f"fold{o}.json"))) for o in range(L.OUTER_K)]
    assert all("outer" in f for f in folds), "run every fold to the outer stage first"
    fsets = [{"ALL": L.ALL, "greedy": f["greedy_subset"]} for f in folds]

    def assemble(choice):
        """outer-test scores of each fold's chosen (feature set, model), stitched over all 992 images"""
        S = np.full((D.n, D.k), np.nan)
        for o in range(L.OUTER_K):
            f, m = choice(o)
            S[fo == o] = np.load(L.path(o, fsets[o][f], m, "outer", L.OUTER_SEED))
        assert not np.isnan(S).any()
        return S

    rows, q = [], {}

    def add(name, S=None, qv=None, ch=None, **extra):
        if S is not None:
            r, qv = C.evaluate(D, S)
        else:
            best, qb = D.Q.max(1), D.Q[:, D.base_j]
            r = dict(mean_Q=qv.mean(), gap_rec=(qv.mean() - qb.mean()) / (best.mean() - qb.mean()),
                     exact=(ch == D.y).mean(), near_005=((best - qv) <= C.TIE_EPS).mean(), top3=np.nan)
            for s, t in (("HDR-Real", "real"), ("HDR-Synth", "synth")):
                m = D.src == s
                r[f"{t}_Q"], r[f"{t}_exact"] = qv[m].mean(), (ch[m] == D.y[m]).mean()
        q[name] = qv
        rows.append(dict(config=name, **r, **extra))

    for rule in RULES:
        add(f"nested_{rule}", assemble(lambda o: folds[o]["choices"][rule]))
    S_fixed = assemble(lambda o: ("ALL", "struct_lgbm"))
    study = {s: np.load(C.oof_path(D, "struct_lgbm", L.OUTER_K, s)) for s in C.SEEDS}
    assert np.array_equal(S_fixed, study[L.OUTER_SEED]), "fixed-config outer fits differ from the study's oof/"
    add("fixed_ALL_struct_lgbm_s1234", S_fixed)
    add("always_catmo", qv=D.Q[:, D.base_j], ch=np.full(D.n, D.base_j))
    add("oracle", qv=D.Q.max(1), ch=D.y)
    pub = pd.DataFrame([C.evaluate(D, study[s])[0] for s in C.SEEDS])
    rows.append(dict(config="published_ALL_struct_lgbm_5seeds", **pub.mean().to_dict(),
                     mean_Q_sd=pub.mean_Q.std(), exact_sd=pub.exact.std()))
    summ = pd.DataFrame(rows)
    summ.round(6).to_csv(os.path.join(L.TAB, "nested_summary.csv"), index=False)

    # paired comparisons on the same outer folds (scene-cluster bootstrap CI, scene-level Wilcoxon)
    prs = []
    for rule in RULES:
        a = f"nested_{rule}"
        for b in ("always_catmo", "fixed_ALL_struct_lgbm_s1234"):
            for lab, m in (("all", None), ("real", D.src == "HDR-Real"), ("synth", D.src == "HDR-Synth")):
                prs.append(dict(a=a, b=b, subset=lab, **C.paired(D, q[a], q[b], mask=m),
                                images_a_better=int((q[a] > q[b])[m if m is not None else slice(None)].sum()),
                                images_b_better=int((q[a] < q[b])[m if m is not None else slice(None)].sum())))
    prs = pd.DataFrame(prs)
    prs.to_csv(os.path.join(L.TAB, "nested_paired.csv"), index=False)

    # choices across outer folds
    ch = []
    for rule in RULES:
        for what, key in (("fset", lambda c: c[0]), ("model", lambda c: c[1]), ("config", lambda c: "+".join(c))):
            for k, n in Counter(key(f["choices"][rule]) for f in folds).items():
                ch.append(dict(rule=rule, what=what, choice=k, n_folds=n))
    for k, n in Counter(g for f in folds for g in f["greedy_removed"]).items():
        ch.append(dict(rule="greedy", what="group_removed", choice=k, n_folds=n))
    ch = pd.DataFrame(ch)
    ch.to_csv(os.path.join(L.TAB, "nested_choices.csv"), index=False)

    # per outer fold
    pf = []
    for o, f in enumerate(folds):
        te = fo == o
        ig = {(g["fset"], g["model"]): g for g in f["inner_grid"]}
        og = {(g["fset"], g["model"]): g for g in f["outer"]}
        c = tuple(f["choices"]["primary"])
        ci = f["greedy_vs_all_ci"] or {}
        pf.append(dict(fold=o, n_test=int(te.sum()), greedy_removed="+".join(f["greedy_removed"]) or "-",
                       greedy_minus_all_inner=ci.get("delta"), ci_lo=ci.get("ci_lo"), ci_hi=ci.get("ci_hi"),
                       chosen="+".join(c), chosen_inner_Q=ig[c]["mean_Q"], chosen_outer_Q=og[c]["mean_Q"],
                       chosen_outer_exact=og[c]["exact"], fixed_outer_Q=og["ALL", "struct_lgbm"]["mean_Q"],
                       fixed_outer_exact=og["ALL", "struct_lgbm"]["exact"], catmo_Q=D.Q[te, D.base_j].mean(),
                       **{f"{r}_chosen": "+".join(f["choices"][r]) for r in RULES[1:]},
                       **{f"{r}_outer_Q": og[tuple(f["choices"][r])]["mean_Q"] for r in RULES[1:]},
                       inner_rank_of_fixed=1 + sorted(ig, key=lambda k: -ig[k]["mean_Q"]).index(("ALL", "struct_lgbm")),
                       all_outer_picks_match_study=all(g["picks_equal_study"] for g in f["outer"] if g["fset"] == "ALL"),
                       max_abs_diff_vs_study=max(g["max_abs_diff_vs_study"] for g in f["outer"] if g["fset"] == "ALL")))
    pf = pd.DataFrame(pf)
    pf.round(6).to_csv(os.path.join(L.TAB, "nested_folds.csv"), index=False)

    pd.set_option("display.width", 200)
    print(summ[["config", "mean_Q", "exact", "near_005", "top3", "gap_rec", "real_Q", "synth_Q"]].round(5)
          .to_string(index=False))
    print()
    print(prs[prs.subset == "all"].drop(columns="subset").round(6).to_string(index=False))
    print()
    print(ch.to_string(index=False))
    print()
    print(pf.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
