#!/usr/bin/env bash
# GiantsBench (giants) RL/GRPO data prep. Thin wrapper around the giants repo's
# scripts/prepare_rl_data.py: downloads GiantsBench-train and reshapes the SAME
# rows into verl's RL schema (prompt / reward_model.ground_truth / extra_info),
# extracting the <insight> block as ground truth. Drops rows with no <insight>.
# Output -> $DATA_DIR/insight_anticipation_grpo. CPU/login node, ~1 min.
#
#   bash grpo_training/prepare_gb_rl_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config.sh"
setup_cluster_env
activate_env

DATASET="${GB_DATASET:-giants2026/GiantsBench-train}"
OUT_DIR="${GRPO_DATA_DIR:-${DATA_DIR}/insight_anticipation_grpo}"

PREP="${GIANTS_DIR}/scripts/prepare_rl_data.py"
[ -f "${PREP}" ] || { echo "ERROR: ${PREP} not found (is vendor/giants cloned?)." >&2; exit 1; }

python "${PREP}" --dataset "${DATASET}" --output-dir "${OUT_DIR}" --drop-empty-insights

echo "✓ GiantsBench RL data written to ${OUT_DIR}"
ls -lh "${OUT_DIR}"
