VENV       := .venv
PYTHON     := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip

IMAGE_NAME := lhdr
DOCKER_DIR := lhdr-docker
DATA_DIR   := images

IMG        ?= 00033.hdr
TMO        ?= reinhard02
OUT        ?= output.png

# Sweep knobs (override on the CLI, e.g. `make smoke LIMIT=3`)
LIMIT      ?= 5
SEED       ?= 1234
MAX_EDGE   ?=
# the reference sweep leaves out the two OpenCV operators (17 operators)
EXCLUDE    ?= opencv

# Assemble optional sweep flags only when set
SWEEP_FLAGS := --seed $(SEED)
ifneq ($(strip $(EXCLUDE)),)
SWEEP_FLAGS += --exclude-kinds $(EXCLUDE)
endif
ifneq ($(strip $(MAX_EDGE)),)
SWEEP_FLAGS += --max-long-edge $(MAX_EDGE)
endif

.PHONY: venv install run docker-build tonemap clean \
        dry-run smoke sweep sweep-time analysis selector report

venv:
	python3.13 -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) tmo_pipeline.py

docker-build:
	docker build -t $(IMAGE_NAME) $(DOCKER_DIR)

tonemap:
	docker run --rm -v "$(CURDIR)/$(DATA_DIR):/data" $(IMAGE_NAME) -l /data/$(IMG) --tmo $(TMO) -o /data/$(OUT)

# --- benchmark sweep (sweep.py) -----------------------------------------------
# Show the plan (op x image counts, cached, excluded) without running anything.
dry-run:
	$(PYTHON) sweep.py --dry-run $(SWEEP_FLAGS)

# Smoke test: first LIMIT valid images x all operators. Resumable — rerun to
# confirm caching (should report "already done (cached), 0 to run").
smoke:
	$(PYTHON) sweep.py --limit $(LIMIT) $(SWEEP_FLAGS)

# Full overnight sweep over every image in images/. Ctrl-C + rerun resumes.
sweep:
	$(PYTHON) sweep.py $(SWEEP_FLAGS)

# Truthful ETA check: force uncached work (incl. slow DL ops) on LIMIT images,
# then restore the cache. Watch the live ETA line.
sweep-time:
	@mv tmo_outputs tmo_outputs_bak 2>/dev/null || true
	-$(PYTHON) sweep.py --limit $(LIMIT) $(SWEEP_FLAGS)
	@rm -rf tmo_outputs && mv tmo_outputs_bak tmo_outputs 2>/dev/null || true

# --- analysis (after the sweep; see README "Analysis stage") ------------------
QMETRICS := Q2 FSITM FSITM_TMQI S S2 u sigma u2 sd2 est_mu2 est_sd2

# score matrices, HDR features, scene groups, and the out-of-fold selector runs.
# Needs images/ and tmo_outputs/.
analysis:
	$(PYTHON) make_qmatrix.py
	for m in $(QMETRICS); do $(PYTHON) make_qmatrix.py --metric $$m || exit 1; done
	$(PYTHON) hdr_features.py
	$(PYTHON) selector_features.py
	$(PYTHON) scene_grouping.py
	for m in Q Q2 FSITM_TMQI; do $(PYTHON) compare_selectors.py --metric $$m || exit 1; done
	$(PYTHON) compare_selectors.py --metric Q --features pouli10

# fit the deployable selector on every image -> models/selector_Q.joblib
selector:
	$(PYTHON) select_tmo.py train

# shareable dataset + report figures/tables, from the files in results/
report:
	$(PYTHON) export_dataset.py
	$(PYTHON) report/make_figures.py

clean:
	rm -rf $(VENV) __pycache__ tmo_outputs tmqi_results.csv selected_tmos.txt
