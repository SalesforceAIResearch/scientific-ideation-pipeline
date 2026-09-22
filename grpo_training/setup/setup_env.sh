#!/usr/bin/env bash
# One-time build of the `verl_grpo` conda env: clones verl (pinned via VERL_REF),
# installs a matched torch/vLLM/flash-attn stack, then `pip install -e` verl.
# Run on a GPU node so flash-attn compiles against the right CUDA.
#
# Run directly:           bash grpo_training/setup/setup_env.sh
# Or under SLURM, e.g.:   srun --gres=gpu:1 --cpus-per-task=16 --mem=128G \
#                           --time=03:00:00 --partition=<yours> --account=<yours> \
#                           bash grpo_training/setup/setup_env.sh
# Re-run is safe; it recreates the env from scratch.
#
# RESOURCES: only 1 GPU is needed (nvcc for the flash-attn compile + a
# torch.cuda.is_available() sanity check). ninja's flash-attn compile is
# RAM-hungry, so MAX_JOBS caps parallel compile jobs (see below) -- lower it
# further if you're on a smaller node than 128G RAM / 16 cores.

set -euo pipefail

# Cap parallel compile jobs. ninja defaults to one job per core and flash-attn
# needs several GB per job -- unbounded it can OOM even on a large node.
export MAX_JOBS="${MAX_JOBS:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config.sh"
setup_cluster_env

echo "========== verl env setup =========="
echo "Start:   $(date)"
echo "Job:     ${SLURM_JOB_ID:-local}"
echo "Node:    $(hostname)"
echo "Env:     ${CONDA_ENV_NAME}"
echo "VERL_REF:${VERL_REF}"
echo "===================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# --- Pick the newest available CUDA toolkit for flash-attn / extensions ---
CUDA_DIR=$(ls -d /usr/local/cuda-12.* 2>/dev/null | sort -V | tail -1)
export CUDA_HOME="${CUDA_DIR}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
echo "CUDA_HOME=${CUDA_HOME}"
nvcc --version || true

# --- Fresh conda env ---
# shellcheck disable=SC1090
source "${CONDA_SH}"
set +u
conda env list | grep -qE "^${CONDA_ENV_NAME}\s" && conda env remove -n "${CONDA_ENV_NAME}" -y
conda create -n "${CONDA_ENV_NAME}" python=3.10 -y
conda activate "${CONDA_ENV_NAME}"
set -u
python -m pip install --upgrade pip setuptools wheel packaging ninja

# --- Clone verl (pinned) into vendor/ ---
mkdir -p "${SCRIPT_DIR}/vendor"
if [ ! -d "${VERL_DIR}/.git" ]; then
    git clone https://github.com/verl-project/verl.git "${VERL_DIR}"
fi
cd "${VERL_DIR}"
git fetch --all --tags
git checkout "${VERL_REF}"
git pull --ff-only 2>/dev/null || true
echo "verl at commit: $(git rev-parse --short HEAD)"

# --- Install the inference/runtime stack ---
# This is the exact combo that already serves OLMo-3 on this cluster (torch 2.10
# cu128 / vllm 0.19.0 / transformers 4.57.6). Install torch first so the rest
# build/resolve against it; then pin vllm + transformers so OLMo-3 loads.
pip install ${TORCH_SPEC} --index-url "${TORCH_INDEX_URL}"
pip install "${VLLM_SPEC}"
pip install "${TRANSFORMERS_SPEC}"

# flash-attn must compile against the active CUDA; no build isolation so it sees torch.
pip install flash-attn --no-build-isolation || \
    echo "WARN: flash-attn build failed; verl can still run with the sdpa/eager attention backend."

# --- Install verl itself (editable, WITHOUT letting it pull its own torch/vllm) ---
# verl's extras can try to repin torch/vllm; install deps loosely but then re-assert
# our pins so OLMo-3 support is preserved.
pip install -e . || pip install -e ".[vllm]"
pip install "${VLLM_SPEC}" "${TRANSFORMERS_SPEC}"

# Reward judge client (OpenAI-compatible calls to the self-hosted gpt-oss server):
pip install "openai>=1.40" || true

# Optional but commonly used by verl trainers:
pip install wandb tensordict codetiming hydra-core liger-kernel || true

echo "===================================="
echo "Sanity check:"
python - <<'PY'
import torch, verl  # noqa
print("torch       :", torch.__version__, "| cuda avail:", torch.cuda.is_available())
print("verl        :", getattr(verl, "__version__", "unknown"))
try:
    import vllm; print("vllm        :", vllm.__version__)
except Exception as e:
    print("vllm        : import failed ->", e)
import transformers
print("transformers:", transformers.__version__)
# OLMo-3 must be a known architecture in this transformers build:
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    has_olmo3 = any("olmo3" in k or "olmo_3" in k for k in CONFIG_MAPPING_NAMES)
    print("OLMo-3 arch :", "yes" if has_olmo3 else "NOT FOUND (bump transformers)")
except Exception as e:
    print("OLMo-3 arch : check failed ->", e)
PY
echo "✓ env '${CONDA_ENV_NAME}' ready"
echo "End: $(date)"
