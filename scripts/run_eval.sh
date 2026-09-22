#!/usr/bin/env bash
# Run the impact + diversity eval, then pool-level diversity + creative-utility
# (CU) plot, over one model's generations.
#
#   bash scripts/run_eval.sh --dataset preds.jsonl --out-prefix runs/my_model \
#     --input-format ideation-si
#
# Produces:
#   <out-prefix>.evals.jsonl           (per-idea impact/diversity scores)
#   <out-prefix>.pool_diversity.jsonl  (pool-level pairwise distances)
#   <out-prefix>.cu.png / .cu.json     (creative-utility curve)
#
# This wrapper only turns on --with-impact/--with-diversity; run_all_evals.py
# also supports soundness/feasibility/grounding judges (see --help) that are
# outside the scope of this release's documented pipeline.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/grpo_training/config.sh"

DATASET=""
OUT_PREFIX=""
INPUT_FORMAT="ideation-si"
EXTRA_EVAL_ARGS=()
EXTRA_POOL_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --out-prefix) OUT_PREFIX="$2"; shift 2 ;;
        --input-format) INPUT_FORMAT="$2"; shift 2 ;;
        --eval-arg) EXTRA_EVAL_ARGS+=("$2"); shift 2 ;;
        --pool-arg) EXTRA_POOL_ARGS+=("$2"); shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "${DATASET}" ] || [ -z "${OUT_PREFIX}" ]; then
    echo "Usage: run_eval.sh --dataset <file> --out-prefix <path> [--input-format ideation-si|ideation-answer]" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUT_PREFIX}")"

echo "[1/3] impact + diversity eval ..."
python3 "${GRPO_DIR}/eval/run_all_evals.py" \
    --dataset "${DATASET}" \
    --out "${OUT_PREFIX}.evals.jsonl" \
    --input-format "${INPUT_FORMAT}" \
    --with-impact \
    --with-diversity \
    "${EXTRA_EVAL_ARGS[@]}"

echo "[2/3] pool-level diversity ..."
python3 "${REPO_ROOT}/qd_tradeoff/score_pool_diversity.py" \
    --in "${OUT_PREFIX}.evals.jsonl" \
    --out "${OUT_PREFIX}.pool_diversity.jsonl" \
    --input-format "${INPUT_FORMAT}" \
    "${EXTRA_POOL_ARGS[@]}"

echo "[3/3] creative-utility plot ..."
python3 "${REPO_ROOT}/qd_tradeoff/plot_pool_creative_utility.py" \
    "MODEL=${OUT_PREFIX}" \
    --strength impact \
    --json "${OUT_PREFIX}.cu.json" \
    --out "${OUT_PREFIX}.cu.png"

echo "Done. See ${OUT_PREFIX}.evals.jsonl / .pool_diversity.jsonl / .cu.png / .cu.json"
