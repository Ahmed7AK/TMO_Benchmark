# TMO Benchmark

A reproducible benchmark of **tone-mapping operators** (TMOs) over the
[HDR-Real](#1-dataset-images) dataset, scored with **TMQI**, **FSITM** and
**TMQI-II**, plus a per-image **TMO selector** trained on HDR image features.

Four kinds of operator run through one identical fairness path
(`produce_ldr` → `compute_scores`). The reference sweep runs with
`--exclude-kinds opencv`, i.e. 17 operators:

| Kind | Count | Operators | Needs |
|------|-------|-----------|-------|
| `opencv` | 2 | `drago_ocv`, `mantiuk_ocv` (not in the reference sweep) | nothing extra |
| `python` | 1 | `att` (Khan et al. 2018, ported in [att_tmo.py](att_tmo.py)) | nothing extra |
| `lhdr` | 13 | `reinhard02/05`, `drago`, `durand`, `fattal`, `mantiuk06/08`, `pattanaik`, `ashikhmin`, `mai`, `lischinski`, `vanhateren`, `kimkautz` | Docker |
| `dl` | 3 | `catmo`, `uncltmo`, `unpaired_hdr` | PyTorch + per-repo venv + weights |

### Final deliverables

| What | Where |
|---|---|
| Report draft (problem, question, method, results, references) | [report/REPORT_DRAFT.md](report/REPORT_DRAFT.md), BibTeX in [report/references.bib](report/references.bib) |
| Shareable dataset: features, histograms, TMQI per TMO, labels | [dataset/](dataset/), described in [dataset/DATA_CARD.md](dataset/DATA_CARD.md) |
| Pick a TMO for a new HDR | `python select_tmo.py train` once, then `python select_tmo.py predict --tonemap out/ my.hdr` |
| Selector vs exhaustive-search timing | `python timing_benchmark.py --n 50` → `results/timing/` |
| Feature / model study and nested CV | [experiments/selector_features/](experiments/selector_features/REPORT.md) |

---

## Table of contents

- [Platform support](#platform-support)
- [Quick start (no downloads)](#quick-start-no-downloads)
- [Full setup](#full-setup)
  - [0. Prerequisites](#0-prerequisites)
  - [1. Dataset (`images/`)](#1-dataset-images)
  - [2. Core Python environment](#2-core-python-environment)
  - [3. Docker image for classical operators](#3-docker-image-for-classical-operators)
  - [4. Deep-learning operators](#4-deep-learning-operators)
- [Running the benchmark](#running-the-benchmark)
- [Analysis stage](#analysis-stage)
- [Reproducibility](#reproducibility)
- [Repo layout](#repo-layout)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)

---

## Platform support

| Platform | Classical (`opencv` + `lhdr`) | Deep-learning (`dl`) | Notes |
|----------|------|------|-------|
| **macOS (Apple Silicon)** | ✅ | ✅ MPS | Reference platform. All results in `results/` produced here. |
| **macOS (Intel)** | ✅ | ✅ CPU | DL operators fall back to CPU; slow but correct. |
| **Linux (x86_64, CPU)** | ✅ | ✅ CPU | Fully supported. |
| **Linux (x86_64, CUDA)** | ✅ | ⚠️ runs on CPU | See [CUDA note](#cuda-is-not-used-by-default) — a one-line change enables the GPU. |
| **Windows (WSL2)** | ✅ | ✅ CPU or CUDA | Supported — see **[WINDOWS.md](WINDOWS.md)**. |
| **Windows (native)** | ✅ | ❌ | Classical operators only; the DL adapters assume POSIX venv paths. |

**Windows users:** the [Quick start](#quick-start-no-downloads) runs natively
in PowerShell (commands below). For the full benchmark (Docker and the
deep-learning operators), set up WSL2 with **[WINDOWS.md](WINDOWS.md)** first;
from that point everything below is identical to Linux.

### CUDA is not used by default

CATMO is invoked with `--gpu_ids -1`
([tmo_pipeline.py:311](tmo_pipeline.py#L311)), and its patched device selector
only picks CUDA when `gpu_ids` is non-empty. On an NVIDIA box CATMO therefore
runs on CPU. To use the GPU, change that flag to `"0"`. UnCLTMO and
Unpaired-HDR select CUDA automatically when it is available.

---

## Quick start (no downloads)

Everything in this section runs from a fresh clone with only the core Python
environment: no dataset, no Docker, no model weights. You need Python 3.13 and
git (pip uses git to install `tmqi-revised`). All commands are run from the
repo root.

```bash
git clone https://github.com/Ahmed7AK/TMO_Benchmark.git
cd TMO_Benchmark

python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

**1. Pick an operator for an HDR image.** The selector is trained from the
checked-in `results/`, so it needs no images. Training takes about 2 minutes
and writes `models/selector_Q.joblib` (~270 MB).

```bash
.venv/bin/python select_tmo.py train
.venv/bin/python select_tmo.py predict 00033.hdr
# 00033.hdr: reinhard05  (0.30s; predicted Q: reinhard05 0.9712, att 0.9403, kimkautz 0.9308)
```

Add `--tonemap out/` to also write the chosen operator's PNG. That step runs the
operator, so it needs Docker for the `lhdr` operators or the DL setup for the
learned ones (see [Full setup](#full-setup)).

**2. Reproduce the reported numbers from the checked-in results.**

```bash
.venv/bin/python compare_selectors.py --metric Q   # -> results/selectors/Q/ (~8 min)
.venv/bin/python report/make_figures.py            # -> report/figures/, report/TABLES_AND_FIGURES.md
```

Both overwrite their checked-in outputs, and the files they write are
byte-identical to the ones in the repo.

**On Windows (PowerShell),** steps 1 and 2 are the same scripts with Windows
paths. Install [Python 3.13](https://www.python.org/downloads/) (tick "Add
python.exe to PATH") and [Git for Windows](https://git-scm.com/download/win)
first:

```powershell
git clone https://github.com/Ahmed7AK/TMO_Benchmark.git
cd TMO_Benchmark
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt

.venv\Scripts\python select_tmo.py train
.venv\Scripts\python select_tmo.py predict 00033.hdr
.venv\Scripts\python compare_selectors.py --metric Q
.venv\Scripts\python report\make_figures.py
```

If PowerShell refuses to run scripts, nothing above needs activation: calling
`.venv\Scripts\python` directly, as shown, avoids that.

(`export_dataset.py` also rebuilds `dataset/`, but its histograms are computed
from the HDRs, so it needs `images/`.)

**3. Smoke-test the scoring path on the sample HDR.** Use `--run-dir` so the
smoke run writes to its own folder and never touches the reference `results/`:

```bash
mkdir -p smoke/images && cp 00033.hdr smoke/images/
.venv/bin/python sweep.py --images-dir smoke/images --run-dir smoke --exclude-kinds opencv
```

Without Docker and the DL venvs, only `att` scores; the 13 `lhdr` operators and
the 3 `dl` operators are logged to `smoke/results/failures.csv`. That is
expected and non-fatal: the sweep logs each failure and continues. With the
[full setup](#full-setup) in place, all 17 operators score the image in under a
minute.

---

## Full setup

### 0. Prerequisites

- **Python 3.13** (CPython). 3.12 likely works; **avoid 3.14** — no PyTorch
  wheels. The reference environment is 3.13.11.
- **Docker** — required for the 13 `luminance-hdr` operators. On Linux you also
  need your user in the `docker` group, or prefix commands with `sudo`.
- **~16 GB free disk**: dataset 0.8 GB, core venv 0.5 GB, three PyTorch venvs
  ~2.8 GB, model weights 0.12 GB, and `tmo_outputs/` reaches **~10 GB** for the
  full 1000-image × 17-operator sweep.
- **git** (also used by pip for `tmqi-revised`) and either `curl` or `wget`.
- **Windows only:** WSL2, set up per [WINDOWS.md](WINDOWS.md), before any of the above.

### 1. Dataset (`images/`)

The benchmark uses **1000 deduplicated HDR images** drawn from the HDR-Real
(628) and HDR-Synth (372) sets of Liu et al., *Single-Image HDR Reconstruction
by Learning to Reverse the Camera Pipeline*, CVPR 2020, resized so the long
side is 512 px and renamed `10000.hdr` … `10999.hdr` ([pick_sample.py](pick_sample.py)).
992 of them are scored successfully by every operator and form the analysis set.
The exact file list and provenance (source file, md5) are in
[`results/manifest.csv`](results/manifest.csv) and
[`results/run_config.json`](results/run_config.json) under `input_manifest`.

The images are **not** in this repo (739 MB). There are two ways to get them;
either way you end up with `images/10000.hdr` … `images/10999.hdr` inside the
repo folder.

**a. Download the prepared set.** Open
**[tmo_images_512px.zip on Google Drive](<GOOGLE_DRIVE_LINK>)** (634 MB) in a
browser and click *Download*. Drive cannot virus-scan a file this large, so it
shows a warning page first: click *Download anyway*. The zip holds an
`images/` folder and `images.sha256`. Extract it **into the repo folder**:

```bash
unzip ~/Downloads/tmo_images_512px.zip -d .           # macOS / Linux
```

```powershell
Expand-Archive $HOME\Downloads\tmo_images_512px.zip -DestinationPath .   # Windows PowerShell
```

In WSL2 the Windows Downloads folder is under `/mnt/c`:
`unzip /mnt/c/Users/<you>/Downloads/tmo_images_512px.zip -d .`

If you use Explorer's *Extract All* instead, it adds an extra folder level; move
the inner `images` folder so the path is `TMO_Benchmark\images\10000.hdr`.

**b. Rebuild it from the original SingleHDR training data.** Download Liu et
al.'s `SingleHDR_training_data` (HDR-Real and HDR-Synth) from the
[SingleHDR project page](https://github.com/alex04072000/SingleHDR), then:

```bash
.venv/bin/python pick_sample.py --data-root /path/to/SingleHDR_training_data \
                                --manifest smoke/manifest_check.csv
```

`pick_sample.py` deduplicates by md5, draws the 1000 images with
`--seed 2026`, resizes them to a 512 px long side and writes `images/`. The
`--manifest` flag keeps it from overwriting the checked-in
`results/manifest.csv`.

**Verify** either route against the checksums in the repo. This works on
every platform and should print `1000 checked, 0 bad`:

```bash
.venv/bin/python -c "import hashlib; r=[l.split() for l in open('images.sha256')]; bad=[f for h,f in r if hashlib.sha256(open(f,'rb').read()).hexdigest()!=h]; print(len(r),'checked,',len(bad),'bad')"
```

(Windows: `.venv\Scripts\python` in place of `.venv/bin/python`. On macOS /
Linux, `shasum -a 256 -c images.sha256` does the same.)

> **Using your own HDRs instead?** Everything downstream is folder-driven —
> point `--images-dir` at any directory of `.hdr`/`.exr` files. The shipped
> `q_matrix*.csv`, `scenes.csv`, and `features_selector.csv` are specific to
> this image set and would need regenerating.

### 2. Core Python environment

```bash
python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt      # or: make install
```

[requirements.txt](requirements.txt) pins exact versions, plus `tmqi-revised`
from a pinned git commit. `PyContracts` (an abandoned TMQI dependency) is
deliberately **not** installed — [tmo_pipeline.py](tmo_pipeline.py) stubs it out
at runtime.

### 3. Docker image for classical operators

```bash
make docker-build       # docker build -t lhdr lhdr-docker
```

Builds Ubuntu 24.04 + `luminance-hdr-cli`, pinned to the version the results
were produced with (2.6.1.1). It takes a few minutes and needs no files from
this repo besides the [Dockerfile](lhdr-docker/Dockerfile). Verify:

```bash
docker run --rm lhdr --help
```

The pipeline shells out to this image per image/operator, bind-mounting the
input and output directories ([tmo_pipeline.py:238](tmo_pipeline.py#L238)).

### 4. Deep-learning operators

Each DL repo runs its **own** upstream inference in its **own** virtualenv, via
subprocess — this isolates their mutually incompatible pins (`models/`,
`options/`, `utils/` packages collide, and UnCLTMO needs `scipy==1.14.1` while
the core env uses 1.18).

The three repos under `third_party/` are **vendored as locally patched forks** —
Apple-MPS device selection, removal of hardcoded `.cuda()`, numpy-2.x / scipy /
imageio compatibility, and re-enabling on-the-fly lambda computation. Every edit
is tagged with a `PATCH:` comment. The full change list is in
[SETUP.md §4](SETUP.md#4-files-modified-inside-the-third-party-repos). **Do not
replace these with fresh upstream clones** — unpatched upstream will not run
here.

The patches are also exported as a standalone series in [`patches/`](patches/),
so you can audit exactly what was changed, or reapply them onto a fresh upstream
checkout:

```bash
git clone https://github.com/xslidi/CATMO && cd CATMO
git checkout 5c6b8a17b318f77ea7d0197707ccf82d6fcec2f1
git apply ../patches/CATMO.patch
```

Pinned upstream commits (from `results/run_config.json`):

| Repo | Upstream | Commit |
|------|----------|--------|
| CATMO | github.com/xslidi/CATMO | `5c6b8a17b318f77ea7d0197707ccf82d6fcec2f1` |
| UnCLTMO | github.com/cao-cong/UnCLTMO | `dcfb045123879e3e6ae805a6ee8af730d1e710cf` |
| unpaired_hdr | github.com/yael-vinker/unpaired_hdr_tmo | `555439a5ebc319e59bbe97583e172556697297e4` |

Substitute your Python 3.13 interpreter for `python3.13` below.

#### 4a. CATMO

```bash
cd third_party/CATMO
python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision numpy scipy scikit-image opencv-python \
                      imageio dominate pillow requests
cd ../..
```

`pyexr` is intentionally omitted (only needed for `.exr`; its import was made
lazy). Then fetch the generator weights — see [Model weights](#model-weights).

#### 4b. UnCLTMO

```bash
cd third_party/UnCLTMO
python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision numpy "scipy==1.14.1" opencv-python \
                      imageio matplotlib scikit-image pillow timm torchsummary gdown
.venv/bin/python -c "import imageio; imageio.plugins.freeimage.download()"
cd ../..
```

> `scipy` is pinned to **1.14.1** — newer scipy changed `differential_evolution`
> internals and breaks the lambda solver.

Weights (`net_epoch5_iter62.pth`, 57 MB, + `run_settings.npy`) must land in
`third_party/UnCLTMO/activate_trained_model/model_weights_imageTMO/`. See
[Model weights](#model-weights) below.

#### 4c. Unpaired-HDR

```bash
cd third_party/unpaired_hdr
python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision numpy scipy opencv-python imageio \
                      matplotlib scikit-image pillow
cd ../..
```

Weights (`trained_weights.pth`, 52 MB) go in
`third_party/unpaired_hdr/activate_trained_model/model_weights/` — see
[Model weights](#model-weights).

#### Model weights

**Not in this repo** (127 MB, and they are not ours to redistribute) — fetch
them from their original upstream sources. All three are **inference-only**
pretrained generators; nothing here trains them.

| Operator | File | Size | Source |
|----------|------|------|--------|
| CATMO | `latest_net_G.pth` | 8.1 MB | ships in the upstream CATMO repo |
| UnCLTMO | `net_epoch5_iter62.pth` + `run_settings.npy` | 57 MB | Google Drive (see below) |
| Unpaired-HDR | `trained_weights.pth` | 52 MB | ships in the upstream repo |

Run all of this from the repo root.

**CATMO** — pulled straight from the pinned upstream commit:

```bash
curl -L -o third_party/CATMO/latest_net_G.pth \
  https://raw.githubusercontent.com/xslidi/CATMO/5c6b8a17b318f77ea7d0197707ccf82d6fcec2f1/latest_net_G.pth

# the CUT framework loads it from checkpoints/<name>/, not the repo root
mkdir -p third_party/CATMO/checkpoints/CATMO
cp third_party/CATMO/latest_net_G.pth third_party/CATMO/checkpoints/CATMO/
```

**Unpaired-HDR** — same idea:

```bash
mkdir -p third_party/unpaired_hdr/activate_trained_model/model_weights
curl -L -o third_party/unpaired_hdr/activate_trained_model/model_weights/trained_weights.pth \
  https://raw.githubusercontent.com/yael-vinker/unpaired_hdr_tmo/555439a5ebc319e59bbe97583e172556697297e4/activate_trained_model/model_weights/trained_weights.pth
```

**UnCLTMO** — the only one not on GitHub. Uses UnCLTMO's own venv, which
already has `gdown` installed from [§4b](#4b-uncltmo):

```bash
DEST=third_party/UnCLTMO/activate_trained_model/model_weights_imageTMO
mkdir -p "$DEST"

third_party/UnCLTMO/.venv/bin/gdown --folder \
  "https://drive.google.com/drive/folders/17MpuVAcQWmZI_ar5Hr0x3d6e_gN4in8K" \
  -O /tmp/uncltmo_weights
unzip -o /tmp/uncltmo_weights/model_weights_imageTMO.zip -d /tmp/uncltmo_weights
cp /tmp/uncltmo_weights/model_weights_imageTMO/net_epoch5_iter62.pth "$DEST/"
cp /tmp/uncltmo_weights/model_weights_imageTMO/run_settings.npy      "$DEST/"
```

> **`gdown` fails often** — Google throttles folder downloads and serves an
> HTML interstitial instead of the file. If it does, open
> <https://drive.google.com/drive/folders/17MpuVAcQWmZI_ar5Hr0x3d6e_gN4in8K>
> in a browser, download `model_weights_imageTMO.zip` by hand, and unzip it
> into the path above. A zero-byte or few-KB `.pth` means you got the
> interstitial, not the weights — check the checksum below.

**Verify.** These are the exact files used for the results in `results/`:

```
e9feca6dce184e1ee5ac5cb5015668d669f23abfa49bfbf3bbdc08dc4a1fe9e7  third_party/CATMO/latest_net_G.pth
71bf7e5685269c933c7b8d1e52efa573a8cf0e7bfdab041b996c5d3b45188f9e  third_party/UnCLTMO/activate_trained_model/model_weights_imageTMO/net_epoch5_iter62.pth
39b2feaacaf797aabebc2c8b00240c7ab0218c6f0e0c6765d4d280a0ad46d99d  third_party/unpaired_hdr/activate_trained_model/model_weights/trained_weights.pth
```

Save those three lines as `weights.sha256` and run `shasum -a 256 -c
weights.sha256` (Linux: `sha256sum -c`).

**Skipping them is fine.** The pipeline checks each path and raises a clear
`FileNotFoundError` naming the missing file, so a partial setup degrades to
"that operator is skipped" — the other 16 operators still run and score
normally. Note that `catmo` is the best fixed operator under TMQI, the
baseline the selector is measured against in
[compare_selectors.py](compare_selectors.py), so the stage-3 analysis does need
CATMO's weights.

#### Per-operator smoke tests

[SETUP.md](SETUP.md) has standalone single-image invocations for CATMO (§1c) and
UnCLTMO (§2c). Run those first if an operator misbehaves in the sweep — they
isolate the upstream repo from the orchestration layer.

---

## Running the benchmark

The orchestrator is [sweep.py](sweep.py); see
[README_sweep.md](README_sweep.md) for the full design.

```bash
make dry-run                # plan only: per-operator todo / cached / excluded
make smoke                  # first 5 valid images × all operators
make smoke LIMIT=20         # override the count
make sweep                  # full run over images/
make sweep-time LIMIT=5     # forced-uncached timing check with live ETA
```

The `make` targets pass `--seed 1234 --exclude-kinds opencv`, the reference
settings (17 operators); `make sweep EXCLUDE=` also runs the two OpenCV
operators. Or call it directly:

```bash
.venv/bin/python sweep.py --images-dir images --seed 1234 --exclude-kinds opencv
.venv/bin/python sweep.py --exclude-kinds opencv --max-long-edge 1920   # cap long edge on ALL operators
.venv/bin/python sweep.py --exclude-kinds opencv --run-dir runs/new     # write results/ and tmo_outputs/ elsewhere
```

**The sweep writes into the checked-in `results/`** unless you pass
`--run-dir`. It is resumable and idempotent: work already recorded in
`results/per_image.csv` is skipped, so `Ctrl-C` and rerun loses nothing and
creates no duplicate rows, and with the reference results in place a rerun has
nothing to do. A failing operator is logged to `results/failures.csv` and never
aborts the run.

Expect the full 1000 × 17 sweep to take **several hours to overnight** —
dominated by the DL operators on CPU/MPS. Re-scoring with every output already
cached in `tmo_outputs/` takes about 75 minutes (≈0.26 s per pair for all three
metrics).

Outputs:

| Path | Contents |
|------|----------|
| `tmo_outputs/` | one 8-bit LDR PNG per image × operator (~10 GB) |
| `results/per_image.csv` | `operator, image, Q, S, N, FSITM, FSITM_TMQI, Q2, S2, N2, resolution_used, tmo_seconds, score_seconds, status` |
| `results/summary.csv` | per-operator aggregates |
| `results/selectors/<metric>/tmo_wins.csv` / `.png` | win counts per operator (written by compare_selectors.py) |
| `results/run_config.json` | seed, versions, upstream commits, input manifest |
| `results/preflight.csv` | per-image HDR validity check |
| `results/failures.csv` | every operator/image failure with reason |

---

## Analysis stage

Stage 2–3 turn the sweep into a per-image TMO selector, the shareable dataset
and the report. Checked-in copies of every intermediate are in `results/`, so
you can start at any step without redoing the ones before it. The full order,
with what each step needs:

| Step | Command | Needs | Writes |
|---|---|---|---|
| 1. Score matrices and selector targets | `make_qmatrix.py`, then `make_qmatrix.py --metric M` for each metric below | `results/per_image.csv`; `tmo_outputs/` for `u`, `sigma`, `u2`, `sd2`, `est_*` | `results/q_matrix*.csv` |
| 2. Original 21 HDR features (reference set, exported with the dataset) | `hdr_features.py` | `images/` | `results/hdr_features.csv` |
| 3. The 151 selector features (~0.2 s per image) | `selector_features.py` | `images/` | `results/features_selector*.csv` |
| 4. Scene groups for grouped CV | `scene_grouping.py` | `images/` | `results/scenes.csv` |
| 5. Selector, scene-grouped 5-fold CV, out-of-fold only | `compare_selectors.py --metric Q` (also `Q2`, `FSITM_TMQI`, and `--metric Q --features pouli10`) | `results/` | `results/selectors/<metric>/` |
| 6. Deployable selector | `select_tmo.py train` | `results/` | `models/selector_Q.joblib` |
| 7. Timing: selector vs running every operator | `timing_benchmark.py --n 50` | `images/`, Docker, DL venvs, step 6 | `results/timing/` |
| 8. Shareable dataset | `export_dataset.py` | `images/`, `results/` | `dataset/` |
| 9. Report figures, tables and numbers | `report/make_figures.py` | `results/`, `dataset/`, `experiments/selector_features/tables/` | `report/figures/`, `report/report_numbers.json`, `report/TABLES_AND_FIGURES.md` |

Steps 1–5 are `make analysis`, step 6 is `make selector`, and steps 8–9 are
`make report`. Step 1 written out:

```bash
.venv/bin/python make_qmatrix.py                    # TMQI Q -> results/q_matrix.csv
for m in Q2 FSITM FSITM_TMQI S S2 u sigma u2 sd2 est_mu2 est_sd2; do
  .venv/bin/python make_qmatrix.py --metric $m
done
```

Step 4 exists because HDR-Real ships no scene metadata: groups are inferred from
id adjacency and thumbnail similarity so that GroupKFold never splits a scene.
After deduplication every image is its own scene.

### Metrics

| Column | Metric | Source |
|--------|--------|--------|
| `Q`, `S`, `N` | TMQI (Yeganeh & Wang 2013) | `tmqi-revised` package |
| `FSITM` | FSITM, green channel (Nafchi et al. 2015) | [tmo_metrics.py](tmo_metrics.py), ported from the authors' `FSITM.m` |
| `FSITM_TMQI` | (FSITM + TMQI Q) / 2, the combined index from the FSITM paper | — |
| `Q2`, `S2`, `N2` | TMQI-II (Ma et al. 2015) | [tmo_metrics.py](tmo_metrics.py), ported from the authors' `TMQI2.m` |

### The selector

**Features.** [selector_features.py](selector_features.py) computes 151
statistics of the HDR radiance map in 14 groups. Each group follows one
published source, and its docstring states every deviation. Nothing is
tone-mapped, scored or read from `tmo_outputs/` at prediction time. The groups
are:

- **Image statistics:** Pouli et al. 2010; Hulusic et al. 2017.
- **The HDR statistics each benchmarked operator computes from its input:**
  Reinhard 2002 (incl. its automatic key / white-point estimation), Drago 2003,
  Durand & Dorsey 2002, Fattal 2002, Reinhard & Devlin 2005, Kim & Kautz 2008,
  Mai 2011 and Mantiuk 2008, following the luminance-hdr code the sweep ran.
- **Metric and quality statistics:** TMQI's HDR-side structural statistics,
  BRISQUE, ITU-T P.910 spatial information, and 11×11-block statistics (TMQI's
  naturalness block size, applied to the HDR).

**Model.** For each candidate operator, LightGBM predicts the components of the
metric, and the score is composed with the metric's own formula:

- TMQI: structural fidelity S, plus the output's mean luminance and 11×11 block
  std, which fix TMQI's naturalness N.
- TMQI-II: S2, plus the output's mean and std.
- FSITM_TMQI: additionally, FSITM.

The selector picks the highest composed score. Each component regressor has
15 leaves and is averaged over 5 group folds of the training rows.

**Results** (`results/selectors/`, seed 1234; each cell is score · exact accuracy):

| Metric | Best fixed operator | Selector | Gap to oracle closed | Oracle |
|---|---|---|---|---|
| TMQI Q | catmo 0.9357 · 8.2% | **0.9596 · 65.2%** | 88% | 0.9629 |
| TMQI-II | reinhard02 0.8951 · 29.5% | **0.9142 · 61.1%** | 58% | 0.9281 |
| FSITM_TMQI | reinhard02 0.9196 · 21.5% | **0.9309 · 67.7%** | 86% | 0.9328 |
| TMQI Q, single-paper base set (Pouli et al. 2010, 27 features) | | 0.9547 · 50.7% | 70% | |

**How the configuration was chosen.** The study is in
[experiments/selector_features/REPORT.md](experiments/selector_features/REPORT.md).
It used 5-fold × 5 seeds, and an independent agent audited every feature
definition and the headline results.

- **Base set:** Pouli et al. 2010 was the best feature set taken from a single
  paper, at 0.9547 · 49.6%.
- **Compiled set:** the 151-feature set adds +0.0050 TMQI (95% CI +0.0041 to
  +0.0060), reaching 0.9597 · 66.5%.
- **Model:** structured LightGBM beat every other model family among the 16
  alternatives, e.g. direct LightGBM regression (−0.0024) and the original
  random-forest classifier (−0.0037). Its unbagged variant is within noise.
- **Caveat:** all choices were made on these 992 images (no held-out set), so the
  numbers are selection-conditional.

> **Join-key gotcha:** `results/per_image.csv` stores `image` as a zero-padded
> string (`00033`), while `q_matrix.csv`, `scenes.csv`, and `hdr_features.csv`
> store it as an integer (`33`). Cast `image` to `int` before joining.

---

## Reproducibility

- **Fairness rule.** Every operator, classical or learned, is scored through the
  same `produce_ldr` → `compute_scores` path. The reference is always the
  original linear HDR, capped identically for all operators. DL outputs may
  differ in resolution from the input; the scoring HDR is matched to each
  operator's own output: **center-cropped for CATMO** (which crops its input to
  multiples of 32 px) and resized for everything else (Unpaired-HDR resizes its
  input), so every pair is compared pixel-aligned at that operator's resolution.
- **Library defaults.** No per-image parameter tuning. `ferwerda` and
  `ferradans` are registered but disabled — see the comments at
  [tmo_pipeline.py:161-165](tmo_pipeline.py#L161-L165) for why (a per-image
  exposure knob would break the defaults rule; `ferradans` takes ~2 min/image).
- **Seed.** `--seed 1234` throughout; recorded in `results/run_config.json`.
- **Environment capture.** Each run writes upstream repo commits and
  numpy/cv2/scipy versions into `results/run_config.json`.
- **Expected drift.** The DL operators are non-deterministic across
  devices/backends — MPS, CPU, and CUDA will not produce bit-identical PNGs, so
  Q scores for `catmo`/`uncltmo`/`unpaired_hdr` will differ in the 3rd–4th
  decimal from the checked-in `results/`. Classical operators should reproduce
  exactly.

---

## Repo layout

```
tmo_pipeline.py       operator registry, scoring, DL adapters (the core)
att_tmo.py            ATT operator (Khan et al. 2018), ported from MATLAB
tmo_metrics.py        FSITM and TMQI-II, ported from the authors' MATLAB
sweep.py              resumable orchestrator over a folder of HDRs
pick_sample.py        draws the deduplicated 1000-image sample from HDR-Real / HDR-Synth
hdr_features.py       the original HDR features (superseded by selector_features.py; kept as the reference set)
scene_grouping.py     stage 2: infer scene groups for GroupKFold
make_qmatrix.py       per_image.csv -> images x operators matrix, per metric (+ selector targets)
selector_features.py  stage 2: 151 research-backed HDR-content features for the selector
compare_selectors.py  stage 3: the selector (structured LightGBM), per metric, out-of-fold evaluation
select_tmo.py         the deployable selector: fit once on all images, pick a TMO for any new HDR
timing_benchmark.py   wall-clock cost of the selector vs running every operator + TMQI
export_dataset.py     writes dataset/ (features, histograms, per-TMO scores, labels)
dataset/              the shareable dataset; DATA_CARD.md describes every file
report/               the final report draft (REPORT_DRAFT.md) and its figures / numbers
experiments/          selector_features/: the feature / model study behind the selector,
                      incl. nested/ (nested cross-validation); tables/ feeds report/make_figures.py.
                      reinhard_check/FINDINGS.md: why 8 lhdr operators get --postgamma 2.2
models/               select_tmo.py's fitted model — NOT in git, `select_tmo.py train` rebuilds it
Makefile              venv / install / docker / sweep / analysis / selector / report targets
lhdr-docker/          Dockerfile for luminance-hdr-cli
third_party/          three patched DL repos (CATMO, UnCLTMO, unpaired_hdr)
patches/              the local patches to those repos, as a standalone series
images/               HDR inputs — NOT in git, see "Dataset"
images.sha256         SHA-256 of every file in images/, to verify a download or rebuild
tmo_outputs/          LDR PNGs — NOT in git, regenerated by the sweep
results/              checked-in reference results from the reference run (timing/: timing_benchmark.py)
00033.hdr             single sample HDR for smoke tests
SETUP.md              deep-dive on the DL operator patches
WINDOWS.md            WSL2 setup for Windows users
README_sweep.md       sweep.py design notes
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `venv python not found at third_party/<repo>/.venv/bin/python` | That repo's venv isn't created — see [§4](#4-deep-learning-operators). On Windows, use WSL2 ([WINDOWS.md](WINDOWS.md)). |
| `CATMO checkpoint missing at .../checkpoints/CATMO/latest_net_G.pth` | Run the `mkdir -p checkpoints/CATMO && cp latest_net_G.pth ...` step. |
| `UnCLTMO weights missing` | Weights aren't in the repo — see [Model weights](#model-weights). |
| All 13 `lhdr` operators fail | Docker image not built (`make docker-build`) or the daemon isn't running. |
| `docker: permission denied` (Linux) | Add yourself to the `docker` group, or use `sudo`. |
| UnCLTMO crashes in `differential_evolution` | Wrong scipy — it must be `1.14.1` **in UnCLTMO's own venv**. |
| `np.float` / `np.bool` AttributeError | A numpy-2 patch was lost. Compare against the `PATCH:` markers listed in [SETUP.md §4](SETUP.md#4-files-modified-inside-the-third-party-repos). |
| `imageio` can't read `HDR-FI` | Run the `imageio.plugins.freeimage.download()` step in UnCLTMO's venv. |
| Out of memory on 4K inputs | `sweep.py --max-long-edge 1920` (applied identically to all operators, so fairness is preserved). |
| DL operators crawl | Expected on CPU. Use `--limit` for testing; see the [CUDA note](#cuda-is-not-used-by-default) on NVIDIA hardware. |

---

## Known limitations

- **Windows needs WSL2** for the DL operators ([WINDOWS.md](WINDOWS.md)).
- **CUDA is off by default for CATMO** ([why](#cuda-is-not-used-by-default)).
- **FSITM and TMQI-II are line-by-line ports** of the authors' MATLAB; they
  have not been run side by side with MATLAB. FSITM agrees with an independent
  Python port to within 0.0004.
- **The third-party repos are vendored, patched forks**, not submodules. Their
  upstream commits are pinned in `results/run_config.json` and the patches are
  exported to [`patches/`](patches/), but they are not maintained as a rebasing
  fork — pulling upstream changes is a manual job.
- **Scene groups are inferred**, not ground truth — HDR-Real ships no scene
  metadata. See the docstring in [scene_grouping.py](scene_grouping.py#L1-L26).
- **`ferwerda` and `ferradans` are excluded** from the operator set, for the
  documented reasons above.

---

## Citation

If you use this benchmark, please also cite the underlying dataset (HDR-Real /
SingleHDR, Liu et al., CVPR 2020), the TMQI, FSITM and TMQI-II metrics, the
ATT operator (Khan et al., IEEE TIE 2018), and the three learned operators —
CATMO, UnCLTMO, and Unpaired-HDR.
