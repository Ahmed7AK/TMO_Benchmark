"""Audit: re-run a few cached results from scratch (no cache) with <= 4 cores and compare to tables/runs.csv;
also checks that models never read targets of test rows (targets are passed through D, so we poison the
test rows' targets and confirm the test-row scores do not change)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import common as C, models as M
for fset, model, ref in (("hdr22", "rf_clf", 0.95539), ("pouli10", "rf_clf", 0.952875), ("hdr22", "lgbm_reg", 0.955661)):
    D = C.load("Q", fset)
    fn, kw = M.MODELS[model]
    S = C.run_cv(fn, D, 5, 1234, jobs=2, **kw)
    r, _ = C.evaluate(D, S)
    print(f"{fset:8s} {model:9s} seed 1234: {r['mean_Q']:.6f} (runs.csv {ref})  exact {r['exact']:.4f}", flush=True)
# leakage probe: poison targets of the test fold for the structured model's components and Q
D = C.load("Q", "hdr22")
fo = C.folds(D, 5, 1234); tr, te = np.where(fo != 0)[0], np.where(fo == 0)[0]
fn, kw = M.MODELS["struct_lgbm_nobag"]
a = fn(D, tr, te, 1234, **kw)
for c in C.PARTS["Q"]:
    D.matrix(c)[te] = np.random.default_rng(0).normal(size=D.matrix(c)[te].shape) * 1e3
D.Q[te] = -1.0
b = fn(D, tr, te, 1234, **kw)
print("struct scores unchanged after poisoning test-row targets:", np.allclose(a, b))
