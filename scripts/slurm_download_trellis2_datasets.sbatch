#!/usr/bin/env bash
#SBATCH --job-name=trellis2-dl
#SBATCH --output=logs/slurm_trellis2_dl_%j.out
#SBATCH --error=logs/slurm_trellis2_dl_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

# Customize this block for your cluster.
# Examples:
# module load cuda/12.4
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate trellis2

ENV_FILE="${1:-${REPO_ROOT}/scripts/trellis2_download.env}"
bash scripts/download_trellis2_datasets.sh "${ENV_FILE}"
