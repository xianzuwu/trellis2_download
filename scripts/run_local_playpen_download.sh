#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECTS_ROOT="/playpen-jfs/xianfeng/Projects"

export XDG_CACHE_HOME="${PROJECTS_ROOT}/.cache"
export HF_HOME="${PROJECTS_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${PROJECTS_ROOT}/.cache/torch"
export PIP_CACHE_DIR="${PROJECTS_ROOT}/.cache/pip"
export GIT_CONFIG_GLOBAL="${PROJECTS_ROOT}/.gitconfig"
export TMPDIR="${PROJECTS_ROOT}/tmp"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p \
    "${XDG_CACHE_HOME}" \
    "${HF_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${HF_DATASETS_CACHE}" \
    "${TORCH_HOME}" \
    "${PIP_CACHE_DIR}" \
    "${TMPDIR}"

cd "${REPO_ROOT}"

ENV_FILE="${1:-${REPO_ROOT}/scripts/local_playpen_download.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
    printf 'Missing env file: %s\n' "${ENV_FILE}" >&2
    printf 'Create it from scripts/local_playpen_download.env.example or ask Codex to generate it locally.\n' >&2
    exit 1
fi

bash scripts/download_trellis2_datasets.sh "${ENV_FILE}"
