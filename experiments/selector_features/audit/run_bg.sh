#!/bin/zsh
cd "/Users/ahmed/code/AIMI Research/TMO"
export OMP_NUM_THREADS=2
PYTHONPATH=. .venv/bin/python experiments/selector_features/audit/compare_selectors_auditcopy.py --seed 2024 --tag _s2024 > experiments/selector_features/audit/prod_s2024.log 2>&1
cd experiments/selector_features/audit && ../../../.venv/bin/python repro2.py ALL struct_lgbm 7 > repro_ALL_s7.log 2>&1
echo DONE >> repro_ALL_s7.log
