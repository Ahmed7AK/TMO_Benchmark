"""
Greedy backward elimination of feature groups from the full compiled set (fast screening protocol of
screen.py: structured LightGBM without bagging, 3-fold x 2 seeds). At each step the group whose removal
raises mean Q the most is dropped, if the gain is positive on both seeds; stops otherwise.
-> tables/greedy_steps.csv
"""
import os

import numpy as np
import pandas as pd

import common as C
from features import GROUPS
from screen import score

cur = list(GROUPS)
_, ref = score("+".join(cur))
cur_q = [q.mean() for _, q in ref]
steps = [dict(step=0, removed="", groups="+".join(cur), n_groups=len(cur), mean_Q=np.mean(cur_q),
              exact=np.mean([r["exact"] for r, _ in ref]), gain=0.0, gain_worst=0.0)]
print(f"start: {len(cur)} groups, Q {np.mean(cur_q):.5f}", flush=True)
while len(cur) > 2:
    trials = []
    for g in cur:
        fs = "+".join(x for x in cur if x != g)
        D, out = score(fs)
        d = [q.mean() - b for (_, q), b in zip(out, cur_q)]
        trials.append((np.mean(d), min(d), g, fs, out, len(D.cols)))
        print(f"  step {len(steps)} -{g:12s} dQ {np.mean(d):+.5f} worst {min(d):+.5f}", flush=True)
    gain, worst, g, fs, out, nf = max(trials)
    if worst <= 0:
        print(f"stop: best removal -{g} gains {gain:+.5f} but worst seed {worst:+.5f}")
        break
    cur = fs.split("+")
    cur_q = [q.mean() for _, q in out]
    steps.append(dict(step=len(steps), removed=g, groups=fs, n_groups=len(cur), n_features=nf,
                      mean_Q=np.mean(cur_q), exact=np.mean([r["exact"] for r, _ in out]),
                      gain=gain, gain_worst=worst))
    print(f"step {len(steps) - 1}: drop {g} -> {len(cur)} groups, {nf} features, Q {np.mean(cur_q):.5f}", flush=True)
    pd.DataFrame(steps).round(6).to_csv(os.path.join(C.TAB, "greedy_steps.csv"), index=False)
pd.DataFrame(steps).round(6).to_csv(os.path.join(C.TAB, "greedy_steps.csv"), index=False)
print("final:", "+".join(cur))
