# TRELLIS.2 Dataset Downloader

Standalone, resumable downloader for the datasets listed in the TRELLIS.2 data toolkit.

## Datasets

Profiles:

```text
train = ObjaverseXL, ABO, HSSD, TexVerse
test  = SketchfabPicked, Toys4k
all   = ObjaverseXL, ABO, HSSD, TexVerse, SketchfabPicked, Toys4k
```

Dataset notes:

- `HSSD` requires Hugging Face access to `hssd/hssd-models`.
- `Toys4k` requires manually placing `toys4k_blend_files.zip` at `<DATA_ROOT>/Toys4k/raw/toys4k_blend_files.zip`.
- `SketchfabPicked` is listed as a TRELLIS.2 test set, but the official picked-list manifest is not published in the TRELLIS.2 repo. Use `TRELLIS2_SKETCHFAB_PICKED_MANIFEST=/path/to/manifest.csv`, or place local assets under `<DATA_ROOT>/SketchfabPicked/raw`.
- Do not commit Hugging Face tokens, logs, or downloaded data.

## Install

Use an environment with Python 3.10+.

```bash
pip install -r requirements.txt
```

`ObjaverseXL` requires the `objaverse` package. `HSSD` and `TexVerse` use `huggingface_hub`.

## Configure

```bash
cp scripts/trellis2_download.env.example scripts/trellis2_download.env
$EDITOR scripts/trellis2_download.env
```

For HSSD, either export a Hugging Face token or set `HF_TOKEN` in the untracked env file:

```bash
export HF_TOKEN=...
```

The token must have access to `hssd/hssd-models`.

## Run In tmux

```bash
tmux new -s trellis2_download
bash scripts/download_trellis2_datasets.sh scripts/trellis2_download.env
```

Detach with `Ctrl-b d`, reattach with:

```bash
tmux attach -t trellis2_download
```

## Run With Slurm

Edit the module/conda block in `scripts/slurm_download_trellis2_datasets.sbatch`, then:

```bash
sbatch scripts/slurm_download_trellis2_datasets.sbatch scripts/trellis2_download.env
```

## Dry Run

```bash
TRELLIS2_DRY_RUN=1 bash scripts/download_trellis2_datasets.sh scripts/trellis2_download.env
```

## Resume Behavior

The downloader:

- reuses existing `metadata.csv`;
- merges partial `raw/new_records/part_*.csv` files into `raw/metadata.csv`;
- scans existing files under `raw/` and recovers them into metadata;
- retries unresolved items.

Restart the same command after interruption. It will skip files already present in metadata and on disk.
