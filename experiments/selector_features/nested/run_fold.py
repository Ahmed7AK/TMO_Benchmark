"""
One outer fold of the nested CV (protocol in nested_lib.py):

    ../../../.venv/bin/python run_fold.py --fold 0 [--stage greedy|select|outer]

The stages run in order (greedy -> select -> outer) and every fit is cached, so a later stage re-uses the
earlier ones. Writes folds/fold<i>.json.
"""
import argparse
import json
import os
import time

import numpy as np

import nested_lib as L

STAGES = ["greedy", "select", "outer"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def label(fs):
    return "ALL" if fs == L.ALL else "greedy"


def greedy(o, rows):
    """greedy.py on the outer training rows: drop the group whose removal raises mean Q most, if it does on
    both seeds; stop otherwise"""
    cv = L.InnerCV(o, rows, L.SCREEN_K, L.INNER_SEEDS)

    def score(fs):
        res = [L.evaluate_rows(rows, S)[0] for S in cv.scores(fs, L.SCREEN_MODEL).values()]
        return [r["mean_Q"] for r in res], float(np.mean([r["exact"] for r in res]))

    cur, steps, trials_log = list(L.GROUPS), [], []
    L.run_tasks(cv.tasks("+".join(cur), L.SCREEN_MODEL))
    cur_q, ex = score("+".join(cur))
    steps.append(dict(step=0, removed="", groups="+".join(cur), n_groups=len(cur), mean_Q=float(np.mean(cur_q)),
                      exact=ex, gain=0.0, gain_worst=0.0))
    log(f"o{o} greedy start: {len(cur)} groups, Q {np.mean(cur_q):.5f}")
    while len(cur) > 2:
        cands = ["+".join(x for x in cur if x != g) for g in cur]
        L.run_tasks([t for fs in cands for t in cv.tasks(fs, L.SCREEN_MODEL)])
        trials = []
        for g, fs in zip(cur, cands):
            q, ex = score(fs)
            d = [a - b for a, b in zip(q, cur_q)]
            trials.append((float(np.mean(d)), float(min(d)), g, fs, q, ex))
            trials_log.append(dict(step=len(steps), removed=g, dQ=trials[-1][0], dQ_worst=trials[-1][1]))
            log(f"  step {len(steps)} -{g:12s} dQ {np.mean(d):+.5f} worst {min(d):+.5f}")
        gain, worst, g, fs, q, ex = max(trials, key=lambda t: t[:3])
        if worst <= 0:
            log(f"o{o} greedy stop: best removal -{g} gains {gain:+.5f} but worst seed {worst:+.5f}")
            break
        cur, cur_q = fs.split("+"), q
        steps.append(dict(step=len(steps), removed=g, groups=fs, n_groups=len(cur), mean_Q=float(np.mean(q)),
                          exact=ex, gain=gain, gain_worst=worst))
        log(f"o{o} greedy step {len(steps) - 1}: drop {g} -> {len(cur)} groups, Q {np.mean(q):.5f}")
    return "+".join(cur), steps, trials_log


def select_tasks(o, rows, G):
    cv = L.InnerCV(o, rows, L.SELECT_K, L.INNER_SEEDS)
    fsets = [L.ALL] + ([G] if G != L.ALL else [])
    return cv, fsets, [t for fs in fsets for m in L.FINALISTS for t in cv.tasks(fs, m)]


def outer_tasks(o, tr, te, G):
    fsets = [L.ALL] + ([G] if G != L.ALL else [])
    return [(L.path(o, fs, m, "outer", L.OUTER_SEED), fs, m, tr, te, L.OUTER_SEED)
            for fs in fsets for m in L.FINALISTS]


def select(o, rows, G):
    """step 1: all groups unless the greedy subset is better with a scene-bootstrap 95% CI above 0 (the study
    called a non-significant difference a tie and kept all groups); argmax variant kept for sensitivity.
    step 2: the finalist with the highest inner mean Q on the chosen feature set."""
    cv, fsets, tasks = select_tasks(o, rows, G)
    L.run_tasks(tasks)
    grid, qimg = [], {}
    for fs in fsets:
        for m in L.FINALISTS:
            res = [L.evaluate_rows(rows, S) for S in cv.scores(fs, m).values()]
            qimg[fs, m] = np.mean([q for _, q, _ in res], 0)
            grid.append(dict(fset=label(fs), model=m, **{k: float(np.mean([r[k] for r, _, _ in res]))
                                                          for k in ("mean_Q", "exact", "near_005", "top3")},
                             mean_Q_by_seed=[r["mean_Q"] for r, _, _ in res]))
            log(f"o{o} inner {label(fs):6s} {m:18s} Q {grid[-1]['mean_Q']:.5f} exact {grid[-1]['exact']:.3f}")
    mq = {(g["fset"], g["model"]): g["mean_Q"] for g in grid}
    ci = None
    if G != L.ALL:
        ci = L.paired_rows(rows, qimg[G, "struct_lgbm"], qimg[L.ALL, "struct_lgbm"])
        log(f"o{o} greedy - ALL (struct_lgbm): {ci['delta']:+.5f} [{ci['ci_lo']:+.5f}, {ci['ci_hi']:+.5f}]")
    f_primary = "greedy" if ci and ci["ci_lo"] > 0 else "ALL"
    f_argmax = "greedy" if ci and mq["greedy", "struct_lgbm"] > mq["ALL", "struct_lgbm"] else "ALL"

    def best_model(f):
        return max(L.FINALISTS, key=lambda m: mq[f, m])

    choices = dict(primary=[f_primary, best_model(f_primary)], fset_argmax=[f_argmax, best_model(f_argmax)],
                   joint_argmax=list(max(mq, key=mq.get)))
    for k, (f, m) in choices.items():
        log(f"o{o} choice {k:12s}: {f} + {m}")
    return grid, ci, choices


def outer(o, tr, te, G):
    """every finalist on each candidate feature set, fit on the outer training rows, scored on the test fold"""
    tasks = outer_tasks(o, tr, te, G)
    out = []
    for (_, fs, m, *_), S in zip(tasks, L.run_tasks(tasks)):
        r, _, ch = L.evaluate_rows(te, S)
        study = L.C.oof_path(L.data(fs), m, L.OUTER_K, L.OUTER_SEED)
        diff = same = None
        if os.path.exists(study):    # the study's OOF fit of this fold (struct_et differs by float summation order)
            ref = np.load(study)[te]
            diff, same = float(np.abs(ref - S).max()), bool((L.evaluate_rows(te, ref)[2] == ch).all())
        out.append(dict(fset=label(fs), model=m, **r, max_abs_diff_vs_study=diff, picks_equal_study=same))
        log(f"o{o} outer {label(fs):6s} {m:18s} Q {r['mean_Q']:.5f} exact {r['exact']:.3f} "
            f"vs study oof: max diff {diff}, same picks {same}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--stage", choices=STAGES, default="outer", help="last stage to run")
    args = ap.parse_args()
    o, last = args.fold, STAGES.index(args.stage)
    fo = L.outer_folds()
    tr, te = np.where(fo != o)[0], np.where(fo == o)[0]
    res = dict(fold=o, n_train=len(tr), n_test=len(te), n_train_scenes=len(np.unique(L.data(L.ALL).groups[tr])))
    p = os.path.join(L.OUT, f"fold{o}.json")
    if os.path.exists(p):
        res.update(json.load(open(p)))
    t0 = time.time()
    G, steps, trials = greedy(o, tr)
    res.update(greedy_subset=G, greedy_removed=[s["removed"] for s in steps[1:]], greedy_steps=steps,
               greedy_trials=trials)
    if last >= 2:   # the outer fits don't depend on the choice: queue them with the inner fits
        L.run_tasks(select_tasks(o, tr, G)[2] + outer_tasks(o, tr, te, G))
    if last >= 1:
        grid, ci, choices = select(o, tr, G)
        res.update(inner_grid=grid, greedy_vs_all_ci=ci, choices=choices)
    if last >= 2:
        res.update(outer=outer(o, tr, te, G))
    res.pop(f"secs_{args.stage}", None)
    res.setdefault("runs", []).append(dict(stage=args.stage, secs=round(time.time() - t0),
                                           finished=time.strftime("%Y-%m-%d %H:%M:%S")))
    json.dump(res, open(p, "w"), indent=1)
    log(f"o{o} done ({args.stage}) in {time.time() - t0:.0f}s -> {p}")


if __name__ == "__main__":
    main()
