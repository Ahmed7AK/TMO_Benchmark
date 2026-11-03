# Running the TMO Benchmark on Windows

Pick the route by what you want to do:

| You want to | Route | You need |
|---|---|---|
| Reproduce the reported selector numbers and figures, or pick an operator for an HDR image | **Native Windows**: the [README Quick start](README.md#quick-start-no-downloads), PowerShell commands included | Python 3.13 and Git for Windows. No WSL, no Docker, no dataset |
| Run the benchmark itself: tone-map the 1000 images with all 17 operators and score them | **WSL2**, this page | Windows 10 2004+ or 11, Docker Desktop, ~25 GB free |

The quick start reads the checked-in `results/`, so it never runs an operator.
The full benchmark needs WSL2 because the deep-learning adapters call each
repo's interpreter as `<repo>/.venv/bin/python`
([tmo_pipeline.py:268](tmo_pipeline.py#L268)), the Linux venv layout; on
Windows it is `.venv\Scripts\python.exe`. Once WSL2 is set up, every command
in [README.md](README.md) works unchanged — you are running real Linux.

---

## Before you start

- **Windows 10 version 2004+ or Windows 11.** Check with `winver`.
- **~25 GB free on your Windows drive.** The Linux filesystem lives in a virtual
  disk that grows on demand: ~16 GB of project data plus WSL and Docker overhead.
- **Virtualization enabled in BIOS/UEFI.** Usually on by default; `wsl --install`
  will tell you if not.

---

## 1. Install WSL2

In **PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot when prompted. On first launch Ubuntu asks you to create a UNIX username
and password — these are unrelated to your Windows account.

Confirm you are on version 2 (not 1 — version 1 has no proper Docker support):

```powershell
wsl -l -v
#   NAME            STATE           VERSION
#   Ubuntu-24.04    Running         2
```

If it says `1`: `wsl --set-version Ubuntu-24.04 2`.

## 2. Give WSL2 enough memory

The DL operators at full resolution are memory-hungry, and WSL2's default
allocation is half your host RAM. Create `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=12GB
processors=8
swap=8GB
```

Then `wsl --shutdown` in PowerShell and reopen Ubuntu. Scale down if your
machine has less; 8 GB is a workable floor, and you can add
`--max-long-edge 1920` to the sweep to cut peak usage.

## 3. Put the project on the Linux filesystem

**This matters more than anything else here.** Work inside `~` (the Linux
filesystem), **never** `/mnt/c/...`. Cross-OS file access in WSL2 goes through a
9P network protocol and is roughly an order of magnitude slower — with ~17,000
output PNGs and 1000 HDR reads per sweep, a run that takes hours on `~` can take
most of a day on `/mnt/c`.

```bash
cd ~
git clone https://github.com/Ahmed7AK/TMO_Benchmark.git
cd TMO_Benchmark
```

You can still reach these files from Windows Explorer at `\\wsl$\Ubuntu-24.04\home\<you>\`.

**The images.** Download `tmo_images_512px.zip` in your Windows browser (link in
[README §1](README.md#1-dataset-images)), then unzip it from Ubuntu straight
into the project, so `images/` lands on the Linux filesystem:

```bash
sudo apt install -y unzip
unzip /mnt/c/Users/<your Windows user>/Downloads/tmo_images_512px.zip -d ~/TMO_Benchmark
ls ~/TMO_Benchmark/images | wc -l      # 1000
```

## 4. Install Python 3.13

Ubuntu 24.04 ships Python 3.12. The reference environment is 3.13, from the
deadsnakes PPA:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.13 python3.13-venv python3.13-dev

sudo apt install -y build-essential git curl unzip
python3.13 --version
```

> Ubuntu's stock `python3.12` also works if you would rather skip the PPA —
> substitute it everywhere the README says `python3.13`. Do **not** use 3.14:
> no PyTorch wheels.

OpenCV needs a couple of system libraries that a headless WSL image lacks:

```bash
sudo apt install -y libgl1 libglib2.0-0
```

Without these, `import cv2` fails with `libGL.so.1: cannot open shared object file`.

## 5. Docker

Needed for the 13 `luminance-hdr` operators. Two options:

**Option A — Docker Desktop (simplest).** Install
[Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/),
then **Settings → Resources → WSL Integration** and enable your Ubuntu distro.
`docker` becomes available inside WSL with no further setup.

**Option B — Docker Engine inside WSL2** (no Docker Desktop licence needed):

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
sudo service docker start
```

Log out and back in for the group change. Docker's daemon does not autostart in
WSL2 — run `sudo service docker start` each session, or add it to `~/.bashrc`.

Verify either way:

```bash
docker run --rm hello-world
```

## 6. Optional: NVIDIA GPU

WSL2 supports CUDA passthrough. Install a recent **NVIDIA driver on Windows** —
do **not** install a driver inside WSL, that breaks the passthrough. Then:

```bash
nvidia-smi     # should list your GPU from inside WSL
```

PyTorch installed per the README picks up CUDA automatically for UnCLTMO and
Unpaired-HDR. CATMO needs a one-line change — see
[CUDA is not used by default](README.md#cuda-is-not-used-by-default).

## 7. Follow the README

From here everything is identical to Linux:

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Continue at [Full setup](README.md#full-setup) — dataset, `make docker-build`,
the three DL venvs and their weights.

---

## Windows-specific troubleshooting

| Symptom | Fix |
|---------|-----|
| `import cv2` → `libGL.so.1: cannot open shared object file` | `sudo apt install -y libgl1 libglib2.0-0` (step 4) |
| `docker: command not found` inside WSL | Docker Desktop WSL integration not enabled for this distro, or Engine not installed (step 5) |
| `Cannot connect to the Docker daemon` | `sudo service docker start` — it does not autostart in WSL2 |
| Sweep is absurdly slow | The project is on `/mnt/c`. Move it to `~` (step 3) |
| OOM / process killed during DL operators | Raise `memory` in `.wslconfig` (step 2), or run with `--max-long-edge 1920` |
| `nvidia-smi: command not found` | Update the NVIDIA driver **on Windows**; never install one inside WSL |
| Disk keeps filling on C: | The WSL vhdx grows but does not shrink. Reclaim with `wsl --shutdown` then `diskpart` → `compact vdisk` |
| Git shows every file as modified | Line endings: `git config --global core.autocrlf false` and re-clone inside WSL |
| `python3.13: command not found` after the PPA | `sudo apt update` again; if deadsnakes has no build for your release, use `python3.12` |

---

## Appendix: native Windows sweep (classical operators only)

Running the sweep itself natively covers `att` and the 13 `lhdr` operators, but
not `catmo`, `uncltmo` or `unpaired_hdr`. It is not tested; WSL2 is the
supported route. The analysis (quick start) does not need this: it runs
natively from the checked-in results.

1. Install Python 3.13, Git for Windows and Docker Desktop, set up `.venv` as in
   the quick start, and build the Docker image:

   ```powershell
   docker build -t lhdr lhdr-docker
   ```

2. Run the sweep without the learned operators, into its own folder:

   ```powershell
   .venv\Scripts\python sweep.py --exclude-kinds opencv dl --limit 5 --run-dir runs\windows
   ```

The `Makefile` targets assume a POSIX shell, so call `sweep.py` directly. Docker
bind mounts are constructed with `os.path.abspath`
([tmo_pipeline.py:238](tmo_pipeline.py#L238)); Docker Desktop generally accepts
Windows-style paths, but if a mount fails, running under WSL2 is the fix.
