#!/usr/bin/env bash
# Clone the giants repo (data-prep dependency) and build the GRPO RL parquet
# from the public GiantsBench-train dataset.
#
#   bash scripts/prepare_rl_data.sh
#   HF_DATASET_REPO=<your-org>/<your-dataset> bash scripts/prepare_rl_data.sh  # once public

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/grpo_training/config.sh"

mkdir -p "${GRPO_DIR}/vendor"
if [ ! -d "${GIANTS_DIR}/.git" ]; then
    git clone https://github.com/joyheyueya/giants.git "${GIANTS_DIR}"
fi

bash "${REPO_ROOT}/grpo_training/data_prep/prepare_gb_rl_data.sh"

if [ -z "${HF_DATASET_REPO:-}" ]; then
    echo "[prepare_rl_data] NOTE: HF_DATASET_REPO is not set -- our own dataset/" >&2
    echo "[prepare_rl_data]       checkpoint HF repo isn't public yet. This step" >&2
    echo "[prepare_rl_data]       only prepared the GiantsBench-derived data above." >&2
fi
