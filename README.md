# TRELLIS.2 Dataset Downloader

Standalone downloader for TRELLIS.2 datasets with resume/retry support.

This repository contains only the dataset download code, not the downloaded data.

## What Can Be Downloaded

Profiles:

```text
train = ObjaverseXL, ABO, HSSD, TexVerse
test  = SketchfabPicked, Toys4k
all   = ObjaverseXL, ABO, HSSD, TexVerse, SketchfabPicked, Toys4k
```

Dataset status:

| Dataset | Status |
| --- | --- |
| `ObjaverseXL` | Direct download. Sources: `sketchfab`, `github`. |
| `ABO` | Direct public download. |
| `HSSD` | Requires Hugging Face access to `hssd/hssd-models`. |
| `TexVerse` | Direct Hugging Face dataset download. |
| `Toys4k` | Requires manual `toys4k_blend_files.zip`, then this repo extracts/registers it. |
| `SketchfabPicked` | Requires a manifest or pre-existing local assets; TRELLIS.2 has not published the official picked manifest. |

For a first full training-data download, use `TRELLIS2_DATASETS="train"`.
For training data plus Toys4k evaluation assets, use `TRELLIS2_DATASETS="train,Toys4k"`.
Use `TRELLIS2_DATASETS="all"` only after preparing the Toys4k archive and a SketchfabPicked manifest/local files.

## 1. Clone

```bash
git clone https://github.com/xianzuwu/trellis2_download.git
cd trellis2_download
```

## 2. Create Conda Environment

```bash
conda create -n trellis2_download python=3.10 -y
conda activate trellis2_download
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional but recommended for faster Hugging Face transfers:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
```

## 3. Configure Download

Create a local env file:

```bash
cp scripts/trellis2_download.env.example scripts/trellis2_download.env
```

Edit it:

```bash
nano scripts/trellis2_download.env
```

Example cluster path:

```bash
mkdir -p /export/data/xianfeng/trellis2_datasets
```

Minimal training-data config:

```bash
TRELLIS2_DATA_ROOT="/export/data/xianfeng/trellis2_datasets"
TRELLIS2_DATASETS="train,Toys4k"
TRELLIS2_OBJAVERSEXL_SOURCES="sketchfab,github"
TRELLIS2_TEXVERSE_RESOLUTION="2k"
TRELLIS2_RETRIES="999"
TRELLIS2_SLEEP_SECONDS="10"
HF_TOKEN="your_huggingface_token_here"
TRELLIS2_HF_LOGIN="1"
TRELLIS2_DOWNLOAD_TOYS4K_ZIP="1"
TRELLIS2_TOYS4K_HF_REPO="seanzzzzz/TRELLIS-500K"
TRELLIS2_TOYS4K_HF_FALLBACK_REPO="lihong-cs/3dgeneration_baseline"
```

Full `all` config:

```bash
TRELLIS2_DATA_ROOT="/export/data/xianfeng/trellis2_datasets"
TRELLIS2_DATASETS="all"
TRELLIS2_OBJAVERSEXL_SOURCES="sketchfab,github"
TRELLIS2_TEXVERSE_RESOLUTION="2k"
TRELLIS2_SKETCHFAB_PICKED_MANIFEST="/path/to/sketchfab_picked_manifest.csv"
HF_TOKEN="your_huggingface_token_here"
TRELLIS2_HF_LOGIN="1"
TRELLIS2_RETRIES="999"
TRELLIS2_SLEEP_SECONDS="10"
```

Do not commit `scripts/trellis2_download.env`. It may contain a Hugging Face token.

## 4. Hugging Face Access

`HSSD` requires access to:

```text
https://huggingface.co/datasets/hssd/hssd-models
```

Use a token with that access. Set it in the shell:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

or put it in the untracked local env file:

```bash
HF_TOKEN="your_huggingface_token_here"
```

Do not put real tokens in GitHub, READMEs, job logs, or shared scripts.

## 5. Prepare Manual Test Assets

For `Toys4k`, this downloader expects the Blend-file archive:

```text
<TRELLIS2_DATA_ROOT>/Toys4k/raw/toys4k_blend_files.zip
```

The zip must contain files under:

```text
toys4k_blend_files/<asset_name>.blend
```

Official source: the Toys4K project page asks users to fill out its download form and provides `toys4k_blend_files.zip` together with other archives such as obj files, point clouds, and sample renders:

```text
https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k
```

The wrapper script can download this file automatically when `Toys4k` is selected:

```bash
TRELLIS2_DOWNLOAD_TOYS4K_ZIP="1"
TRELLIS2_TOYS4K_HF_REPO="seanzzzzz/TRELLIS-500K"
TRELLIS2_TOYS4K_HF_FALLBACK_REPO="lihong-cs/3dgeneration_baseline"
```

Equivalent manual command:

```bash
huggingface-cli download seanzzzzz/TRELLIS-500K toys4k_blend_files.zip \
  --repo-type dataset \
  --local-dir "${TRELLIS2_DATA_ROOT}/Toys4k/raw"
```

Use the official Toys4K source if licensing/provenance matters for your cluster.

For `SketchfabPicked`, provide one of:

```bash
TRELLIS2_SKETCHFAB_PICKED_MANIFEST="/path/to/sketchfab_picked_manifest.csv"
```

or local mesh files under:

```text
<TRELLIS2_DATA_ROOT>/SketchfabPicked/raw/
```

Manifest columns supported by `SketchfabPicked`:

```text
file_identifier
sha256
local_path
source_path
```

As of this repository revision, we have not found an official TRELLIS.2 `SketchfabPicked` manifest in the Microsoft TRELLIS.2 repository or issue thread. Without that manifest, `SketchfabPicked` can only be reproduced from a user-provided picked list or local assets.

## 6. Dry Run

Before starting a large cluster job:

```bash
TRELLIS2_DRY_RUN=1 bash scripts/download_trellis2_datasets.sh scripts/trellis2_download.env
```

This prints the resolved command and exits without downloading.

## 7. Run In tmux

```bash
tmux new -s trellis2_download
conda activate trellis2_download
bash scripts/download_trellis2_datasets.sh scripts/trellis2_download.env
```

Detach:

```text
Ctrl-b d
```

Reattach:

```bash
tmux attach -t trellis2_download
```

## 8. Run With Slurm

Edit the module/conda block in:

```text
scripts/slurm_download_trellis2_datasets.sbatch
```

Then submit:

```bash
sbatch scripts/slurm_download_trellis2_datasets.sbatch scripts/trellis2_download.env
```

## 9. Direct Python Commands

Download training datasets:

```bash
python download_trellis2_datasets.py \
  --root /path/to/trellis2_datasets \
  --datasets train \
  --objaversexl-sources sketchfab,github \
  --texverse-resolution 2k \
  --retries 999 \
  --sleep-seconds 10
```

Attempt all datasets:

```bash
python download_trellis2_datasets.py \
  --root /path/to/trellis2_datasets \
  --datasets all \
  --objaversexl-sources sketchfab,github \
  --texverse-resolution 2k \
  --sketchfab-picked-manifest /path/to/sketchfab_picked_manifest.csv \
  --retries 999 \
  --sleep-seconds 10
```

## Resume Behavior

The downloader:

- reuses existing `metadata.csv`;
- merges partial `raw/new_records/part_*.csv` files into `raw/metadata.csv`;
- scans existing files under `raw/` and recovers them into metadata;
- retries unresolved items.

Restart the same command after interruption. It will skip files already present in metadata and on disk.
