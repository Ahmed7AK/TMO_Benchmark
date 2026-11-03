# SETUP — Deep-learning TMOs (CATMO + UnCLTMO)

This documents how to set up the two pretrained deep-learning tone-mapping
operators that plug into the benchmark in [tmo_pipeline.py](tmo_pipeline.py):

- **CATMO** — github.com/xslidi/CATMO (contrastive attention-UNet, CUT framework)
- **UnCLTMO** — github.com/cao-cong/UnCLTMO (IVTMNet, unsupervised contrastive)

Both run the upstream repo's **own** inference, each in its **own virtualenv**,
called via `subprocess`. The pipeline then collects each operator's 8-bit LDR
PNG and scores it with the same TMQI call as every classical operator. The
repos live under `third_party/` and are already patched for Apple MPS (see
"Files modified" at the end). You only need to recreate the venvs and, for
UnCLTMO, fetch the weights if they weren't shipped with the repo.

Target environment: **MacBook (Apple Silicon), macOS, no CUDA.** Inference uses
the **MPS** backend with CPU fallback (`PYTORCH_ENABLE_MPS_FALLBACK=1`, set
automatically by the pipeline).

---

## 0. Prerequisites

- Python 3.13 (the venvs below use `/Users/<you>/miniconda3/bin/python3.13`;
  any CPython 3.13 with PyTorch wheels works). **Avoid 3.14** — no torch wheels yet.
- Docker (only for the classical luminance-hdr operators; not needed for the DL ones).
- The main pipeline venv (`.venv/`) installed from [requirements.txt](requirements.txt).

```bash
make install          # or: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

---

## 1. CATMO

The pretrained generator `latest_net_G.pth` is **not committed here** — fetch it
from the upstream repo first (see "Model weights" in [README.md](README.md)).
The CUT framework then expects it at `checkpoints/CATMO/latest_net_G.pth`.

```bash
cd third_party/CATMO

# 1a. fetch + checkpoint placement
curl -L -o latest_net_G.pth \
  https://raw.githubusercontent.com/xslidi/CATMO/5c6b8a17b318f77ea7d0197707ccf82d6fcec2f1/latest_net_G.pth
mkdir -p checkpoints/CATMO
cp latest_net_G.pth checkpoints/CATMO/latest_net_G.pth

# 1b. venv + deps (pyexr is intentionally omitted — only needed for .exr inputs,
#     and its import was made lazy; we feed .hdr)
/Users/$USER/miniconda3/bin/python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision numpy scipy scikit-image opencv-python imageio dominate pillow requests

# 1c. single-image smoke test (writes results/CATMO/test_latest/images/<id>_revise_B.png)
mkdir -p /tmp/catmo_smoke/testA && cp ../../images/00033.hdr /tmp/catmo_smoke/testA/
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python test.py \
  --dataroot /tmp/catmo_smoke/testA --name CATMO --model test --netG aunet \
  --gpu_ids -1 --results_dir /tmp/catmo_smoke/results --num_test 1
```

`--gpu_ids -1` means "no CUDA device list"; the patched `base_model.py` then
selects MPS (or CPU). The final color output is the `*_revise_B.png` file.

---

## 2. UnCLTMO

Pretrained weights are NOT in the repo — they're on Google Drive. The image-TMO
inference needs `net_epoch5_iter62.pth` + `run_settings.npy` inside
`activate_trained_model/model_weights_imageTMO/`.

```bash
cd third_party/UnCLTMO

# 2a. venv + deps. NOTE: scipy is pinned to 1.14.1 — newer scipy changed
#     differential_evolution internals and breaks the lambda solver.
/Users/$USER/miniconda3/bin/python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision numpy "scipy==1.14.1" opencv-python imageio matplotlib scikit-image pillow timm torchsummary gdown
# FreeImage plugin for imageio (the repo asks for HDR-FI/PNG-FI formats):
.venv/bin/python -c "import imageio; imageio.plugins.freeimage.download()"

# 2b. fetch pretrained weights (image TMO). Try gdown on the GDrive folder:
.venv/bin/gdown --folder "https://drive.google.com/drive/folders/17MpuVAcQWmZI_ar5Hr0x3d6e_gN4in8K" -O /tmp/uncltmo_weights
unzip -o /tmp/uncltmo_weights/model_weights_imageTMO.zip -d /tmp/uncltmo_weights
cp /tmp/uncltmo_weights/model_weights_imageTMO/net_epoch5_iter62.pth activate_trained_model/model_weights_imageTMO/
cp /tmp/uncltmo_weights/model_weights_imageTMO/run_settings.npy      activate_trained_model/model_weights_imageTMO/
```

**If `gdown` fails** (Google quota / interstitial): open the folder
<https://drive.google.com/drive/folders/17MpuVAcQWmZI_ar5Hr0x3d6e_gN4in8K> in a
browser, download `model_weights_imageTMO.zip`, unzip it, and place
`net_epoch5_iter62.pth` (and `run_settings.npy`) into
`third_party/UnCLTMO/activate_trained_model/model_weights_imageTMO/`.

```bash
# 2c. single-image smoke test (writes <out>/<id>_*.png)
cd activate_trained_model
mkdir -p /tmp/uncltmo_smoke/testA /tmp/uncltmo_smoke/lambda && cp ../../../images/00033.hdr /tmp/uncltmo_smoke/testA/
PYTORCH_ENABLE_MPS_FALLBACK=1 ../.venv/bin/python test_imageTMO.py \
  --model_name "11_08_lr15D_size268_D_[1,1,1]_pad_0_G_ssr_doubleConvT__d1.0_struct_1.0[1,1,1]__trans2_replicate__noframe__min_log_0.1hist_fit_" \
  --input_images_path /tmp/uncltmo_smoke/testA --output_path /tmp/uncltmo_smoke/out \
  --model_path model_weights_imageTMO \
  --f_factor_path lambda_data/input_images_lambdas_HDRSdataset.npy \
  --mean_hist_path lambda_data/ldr_avg_hist_900_images_20_bins.npy \
  --lambda_output_path /tmp/uncltmo_smoke/lambda --bins 20
```

The first run computes a per-image "lambda" (brightness factor) via the bundled
mean histogram (a few seconds per image), then runs the network in
full-resolution mode.

---

## 3. Run the benchmark

With both venvs ready, the operators are already registered in
[tmo_pipeline.py](tmo_pipeline.py) (`catmo`, `uncltmo`, kind `dl`). Just run:

```bash
.venv/bin/python tmo_pipeline.py      # or: make run
```

Toggle the DL operators or set a global resolution cap at the top of the file:

```python
ENABLE_CATMO   = True
ENABLE_UNCLTMO = True
MAX_LONG_EDGE  = None   # e.g. 2048 to cap long edge (applied to ALL operators) if 4K OOMs
```

Behavior notes:
- **Idempotent**: an operator is skipped for an image if its output PNG already exists.
- **Graceful**: a missing checkpoint or a failed forward pass logs a warning and
  skips that operator for that image — it never aborts the sweep.
- **Per-op logging**: each DL operator prints `device=… Ns/img` and any ops that
  fell back to CPU on MPS.
- **Resolution**: DL outputs may differ in size from the input; the scoring HDR
  is resized to match each operator's output before TMQI (so every pair is
  compared at that operator's own resolution).

---

## 4. Files modified inside the third-party repos

All edits are local, minimal, and tagged with a `PATCH:` comment. Categories:
MPS device selection, removing hardcoded `.cuda()`, numpy-2.x / scipy / imageio
compatibility, and enabling on-the-fly lambda computation.

### CATMO (`third_party/CATMO/`)
| File | Change |
|------|--------|
| `models/base_model.py` | device selection: CUDA → **MPS** → CPU (was CUDA/CPU only) |
| `models/networks.py` | `init_net`: place net on MPS when no CUDA gpu_ids |
| `models/test_model.py` | cast inputs to float32 before `.to(device)` (MPS has no float64) |
| `util/util.py` | `ReviseImage`: replace hardcoded `.cuda()` with `.to(images.device)` |
| `test.py` | `num_threads = 0` (macOS `spawn` can't pickle the dataset's lambda transforms) |
| `data/base_dataset.py` | make `pyexr` import lazy (only needed for `.exr`) |
| `checkpoints/CATMO/latest_net_G.pth` | **added at setup time** (not committed — fetched from upstream, then copied to where the framework expects it; step 1a) |

### UnCLTMO (`third_party/UnCLTMO/`)
| File | Change |
|------|--------|
| `activate_trained_model/test_imageTMO.py` | device: CUDA → **MPS** → CPU; re-enable `calc_lambda` for arbitrary images; `--bins` `type=int` (was `str`, broke `np.histogram`) |
| `utils/model_save_util.py` | `test_big_size_image2`: hardcoded `.cuda()` → model's device; switch to **full-resolution** mode (`scale=1`, overlap 192) for small inputs |
| `utils/hdr_image_util.py` | `.hdr` read & PNG write: fall back from broken FreeImage `HDR-FI`/`PNG-FI` to OpenCV / default imageio |
| `models/unet_multi_filters/gcn_lib/pos_embed.py` | `np.float` → `float` (removed in numpy 2.0) |
| `activate_trained_model/model_weights_imageTMO/net_epoch5_iter62.pth` | **added at setup time** (not committed — downloaded from Google Drive; step 2b) |

> The video-TMO code path (`test_videoTMO.py`, the 5-D `.cuda()` calls in
> `model_save_util.py`) was left untouched — only image TMO is used here.
