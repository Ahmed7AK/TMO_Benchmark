"""Round-2 audit: production compare_selectors.py (audit copy, seed 2024) vs the study's cached OOF for seed 2024."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import common as C
from screen import ALL
D = C.load("Q", ALL)
S = np.load(C.oof_path(D, "struct_lgbm", 5, 2024))
r, q = C.evaluate(D, S)
ch = C.pick(D, S)
pr = pd.read_csv(os.path.join(HERE, "prod_out", "Q_s2024", "selector_predictions.csv")).set_index("image")
pp = pr.loc[D.df.image, "pick"].to_numpy()
same = np.mean(np.array(D.ops)[ch] == pp)
comp = pd.read_csv(os.path.join(HERE, "prod_out", "Q_s2024", "selector_comparison.csv"))
print(f"study OOF s2024: Q {r['mean_Q']:.6f} exact {r['exact']:.4f} | production s2024: "
      f"{comp.loc[comp.method.str.startswith('structured'), ['mean_Q', 'exact_accuracy']].values.ravel()} | same picks {same:.4f}")
print("folds identical:", (pr.loc[D.df.image, "fold"].to_numpy() == C.folds(D, 5, 2024)).all())
