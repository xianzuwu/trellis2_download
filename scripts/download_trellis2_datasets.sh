#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRELLIS2_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export TRELLIS2_REPO_ROOT

ENV_FILE="${1:-${TRELLIS2_DOWNLOAD_ENV:-${SCRIPT_DIR}/trellis2_download.env}}"
if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
fi

: "${TRELLIS2_DATA_ROOT:=${TRELLIS2_REPO_ROOT}/datasets}"
: "${TRELLIS2_DATASETS:=train}"
: "${TRELLIS2_OBJAVERSEXL_SOURCES:=sketchfab,github}"
: "${TRELLIS2_TEXVERSE_RESOLUTION:=2k}"
: "${TRELLIS2_RETRIES:=999}"
: "${TRELLIS2_SLEEP_SECONDS:=10}"
: "${TRELLIS2_HF_LOGIN:=1}"
: "${TRELLIS2_DRY_RUN:=0}"

mkdir -p "${TRELLIS2_DATA_ROOT}" "${TRELLIS2_REPO_ROOT}/logs"

if [[ "${TRELLIS2_HF_LOGIN}" == "1" && -n "${HF_TOKEN:-}" ]]; then
    python - <<'PY'
import os
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
PY
fi

cmd=(
    python "${TRELLIS2_REPO_ROOT}/download_trellis2_datasets.py"
    --root "${TRELLIS2_DATA_ROOT}"
    --datasets "${TRELLIS2_DATASETS}"
    --objaversexl-sources "${TRELLIS2_OBJAVERSEXL_SOURCES}"
    --texverse-resolution "${TRELLIS2_TEXVERSE_RESOLUTION}"
    --retries "${TRELLIS2_RETRIES}"
    --sleep-seconds "${TRELLIS2_SLEEP_SECONDS}"
)

if [[ -n "${TRELLIS2_TEXVERSE_METADATA_FILE:-}" ]]; then
    cmd+=(--texverse-metadata-file "${TRELLIS2_TEXVERSE_METADATA_FILE}")
fi
if [[ -n "${TRELLIS2_TEXVERSE_MANIFEST:-}" ]]; then
    cmd+=(--texverse-manifest "${TRELLIS2_TEXVERSE_MANIFEST}")
fi
if [[ -n "${TRELLIS2_TEXVERSE_REPO_ID:-}" ]]; then
    cmd+=(--texverse-repo-id "${TRELLIS2_TEXVERSE_REPO_ID}")
fi
if [[ -n "${TRELLIS2_SKETCHFAB_PICKED_MANIFEST:-}" ]]; then
    cmd+=(--sketchfab-picked-manifest "${TRELLIS2_SKETCHFAB_PICKED_MANIFEST}")
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${TRELLIS2_REPO_ROOT}/logs/download_${TRELLIS2_DATASETS//,/__}_${timestamp}.log"

printf 'Repository: %s\n' "${TRELLIS2_REPO_ROOT}"
printf 'Data root:  %s\n' "${TRELLIS2_DATA_ROOT}"
printf 'Datasets:   %s\n' "${TRELLIS2_DATASETS}"
printf 'Log file:   %s\n' "${log_file}"

if [[ "${TRELLIS2_DRY_RUN}" == "1" ]]; then
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

"${cmd[@]}" 2>&1 | tee -a "${log_file}"
