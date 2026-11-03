"""Round-2 audit: recompute study configurations from scratch (no cache) with <= 4 threads and compare
with the cached OOF scores the study's tables were built from.
    python repro2.py <fset> <model> <seed>"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import common as C, models as M
from screen import ALL
fset, model, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
D = C.load("Q", ALL if fset == "ALL" else fset)
fn, kw = M.MODELS[model]
t0 = time.time()
S = C.run_cv(fn, D, 5, seed, jobs=2, **kw)
np.save(os.path.join(HERE, f"repro_{fset}_{model}_s{seed}.npy"), S)
r, q = C.evaluate(D, S)
p = C.oof_path(D, model, 5, seed)
line = f"{fset}:{model} seed {seed} from scratch: Q {r['mean_Q']:.6f} exact {r['exact']:.4f} ({time.time()-t0:.0f}s)"
if os.path.exists(p):
    Sc = np.load(p); rc, _ = C.evaluate(D, Sc)
    line += (f" | cached OOF: Q {rc['mean_Q']:.6f} exact {rc['exact']:.4f}; max|dS| {np.abs(S - Sc).max():.2e}; "
             f"same picks {np.mean(C.pick(D, S) == C.pick(D, Sc)):.4f}")
else:
    line += f" | no cached OOF at {os.path.basename(p)}"
print(line, flush=True)
