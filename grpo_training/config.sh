#!/usr/bin/env bash
# Shared configuration for all GRPO/verl scripts in this folder.
# Source this from launcher scripts:  source "$(dirname "$0")/config.sh"
#
# Everything cluster- or environment-specific lives here so the run scripts stay
# clean. Every variable below is overridable by exporting it (or passing it on
# the sbatch/CLI command line) before this file is sourced.

# --- Code paths (self-relative; this file lives in grpo_training/) ---
export GRPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VERL_DIR="${GRPO_DIR}/vendor/verl"          # verl checkout (gitignored, see vendor_patches/README.md)
export GIANTS_DIR="${GRPO_DIR}/vendor/giants"      # giants checkout (gitignored)
export LOG_DIR="${LOG_DIR:-${GRPO_DIR}/logs}"

# --- Data / model storage ---
# Overridable: export STORAGE_DIR (or DATA_DIR/CKPT_DIR directly) before launch to
# relocate a run's data+checkpoints onto your cluster's scratch/shared mount, e.g.
#   STORAGE_DIR=/scratch/$USER/grpo_training bash scripts/train_grpo_v6_si.sh
# Defaults to a local ".runs" directory under grpo_training/ so this works out of
# the box on a single machine with no cluster-specific mount assumptions.
export STORAGE_DIR="${STORAGE_DIR:-${GRPO_DIR}/.runs}"
export DATA_DIR="${DATA_DIR:-${STORAGE_DIR}/data}"          # parquet datasets
export CKPT_DIR="${CKPT_DIR:-${STORAGE_DIR}/checkpoints}"    # training checkpoints
export TMP_DIR="${TMP_DIR:-${STORAGE_DIR}/tmp}"
export HF_HOME_DIR="${HF_HOME_DIR:-${STORAGE_DIR}/hf_cache}"

# --- Conda ---
# Auto-detected from `conda` on PATH if CONDA_SH isn't set explicitly.
if [ -z "${CONDA_SH:-}" ] && command -v conda >/dev/null 2>&1; then
    export CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl_grpo}"   # created by setup/setup_env.sh

# --- verl version to install (a tag, branch, or commit SHA) ---
export VERL_REF="${VERL_REF:-main}"

# --- Inference stack ---
export TORCH_SPEC="${TORCH_SPEC:-torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0}"
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
export VLLM_SPEC="${VLLM_SPEC:-vllm==0.19.0}"
export TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==4.57.6}"

# --- Models (7B policy) ---
export POLICY_BASE_MODEL="${POLICY_BASE_MODEL:-allenai/Olmo-3-1025-7B}"
export POLICY_INSTRUCT_MODEL="${POLICY_INSTRUCT_MODEL:-allenai/Olmo-3-7B-Instruct}"
# SFT starts from the Think-SFT stage-1 checkpoint (NOT the raw base): it already
# carries a chat template and emits <think>...</think> reasoning, matching the
# GiantsBench completion format. GRPO then starts from our SFT checkpoint.
export SFT_INIT_MODEL="${SFT_INIT_MODEL:-allenai/Olmo-3-7B-Think-SFT}"

# --- Reward judge (self-hosted by default; no external API calls from compute
# nodes unless you point SCIJUDGE_MODEL/SCIJUDGE_BASE_URL at a hosted model) ---
# A vLLM OpenAI-compatible server (serve/serve_judge.sbatch) writes its live URL
# to JUDGE_ENDPOINT_FILE; the reward reads it if JUDGE_BASE_URL isn't set explicitly.
export JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"
export JUDGE_ENDPOINT_FILE="${JUDGE_ENDPOINT_FILE:-${TMP_DIR}/judge_endpoint.txt}"
export JUDGE_PORT="${JUDGE_PORT:-8001}"

# --- SciJudge reward model (pairwise citation-impact judge; self-hosted) ---
# OpenMOSS-Team/SciJudge-30B served on its own vLLM endpoint (serve/serve_scijudge.sbatch).
# Separate port + endpoint file so it can coexist with the gpt-oss judge above.
export SCIJUDGE_MODEL="${SCIJUDGE_MODEL:-OpenMOSS-Team/SciJudge-30B}"
export SCIJUDGE_ENDPOINT_FILE="${SCIJUDGE_ENDPOINT_FILE:-${TMP_DIR}/scijudge_endpoint.txt}"
export SCIJUDGE_PORT="${SCIJUDGE_PORT:-8002}"
export SCIJUDGE_CONDA_ENV="${SCIJUDGE_CONDA_ENV:-${CONDA_ENV_NAME}}"
# The judge is swappable: the reward + eval just need a model name (+ base URL).
# Self-hosted vLLM models use SCIJUDGE_BASE_URL / the endpoint file (key "EMPTY").
# A gpt-*/claude-*/gemini- judge auto-routes through common/gateway.py using
# OPENAI_API_KEY / ANTHROPIC_API_KEY from the environment; force the backend with
# SCIJUDGE_BACKEND=gateway|vllm if the model name is ambiguous.
#   export SCIJUDGE_MODEL=gpt-4o        # or claude-3-5-sonnet, etc. (keys via env)
export SCIJUDGE_BASE_URL="${SCIJUDGE_BASE_URL:-}"

# --- SLURM defaults ---
# Unset by default -- set these for your cluster, or pass --partition/--account on
# the sbatch command line (which beats these). If you don't have SLURM, use the
# non-SLURM launcher scripts/train_grpo_v6_si.sh instead.
export SLURM_PARTITION="${SLURM_PARTITION:-}"
export SLURM_ACCOUNT_DEFAULT="${SLURM_ACCOUNT_DEFAULT:-}"

# --- Optional cuBLAS/LD_LIBRARY_PATH workaround (opt-in, off by default) ---
# Some clusters' login profile prepends a system CUDA path (e.g.
# /usr/local/cuda-12.x/lib64) to LD_LIBRARY_PATH, which can split cuBLAS across
# two CUDA versions: libcublas.so.12 resolves to torch's bundled build while
# libcublasLt.so.12 resolves to the system toolkit. The two are a matched pair --
# cublasGemmEx with CUBLAS_GEMM_DEFAULT_TENSOR_OP dispatches into cuBLASLt -- so a
# version mismatch surfaces as:
#     RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE ... cublasGemmEx(...)
# Only certain large tensor-core GEMMs are rejected, so it survives import and
# dies later (e.g. in vLLM's profile_run). NOT a torch.compile/cudagraph issue --
# enforce_eager does not fix it. If you hit this, opt in with FIX_CUBLAS_LD=1.
#
# WHY OPT-IN: stripping CUDA from the linker path is wrong while BUILDING --
# setup/setup_env.sh prepends ${CUDA_HOME}/lib64 on purpose so flash-attn compiles
# against the system toolkit. Runtime wants the matched bundle, build wants the
# toolkit, so this is a per-job switch rather than a global edit.
export FIX_CUBLAS_LD="${FIX_CUBLAS_LD:-0}"

# --- Weights & Biases ---
# verl logs to wandb when trainer.logger includes "wandb". If no API key is
# available we fall back to offline mode so a run never crashes on logging
# (sync later with `wandb sync`). Set WANDB_API_KEY (or `wandb login`) to go online.
export WANDB_PROJECT="${WANDB_PROJECT:-$(basename "${STORAGE_DIR}")}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
make_run_name() { echo "${1}-$(date +%Y%m%d_%H%M%S)"; }
setup_wandb() {
    export WANDB_DIR="${GRPO_DIR}"
    export WANDB_CACHE_DIR="${TMP_DIR}/wandb_cache"
    mkdir -p "${WANDB_CACHE_DIR}"

    if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "${HOME}/.netrc" ]; then
        echo "[wandb] no credentials found -> WANDB_MODE=offline (sync later with 'wandb sync')"
        export WANDB_MODE=offline
    else
        echo "[wandb] online | project=${WANDB_PROJECT} | base_url=${WANDB_BASE_URL}"
    fi
}

# Attach this SLURM job's stdout/stderr files to a W&B run so crashes are
# debuggable from the W&B UI (Files tab), not just from scratch logs. Pass the
# training run's id so the logs land on the SAME run (resume="allow"); if that
# run never got created (e.g. an early crash) this creates it. Best-effort: it
# never fails the job, so it is safe to call from an EXIT trap.
#   upload_slurm_logs_to_wandb <project> <run_name> <run_id>
upload_slurm_logs_to_wandb() {
    local project="$1" run_name="$2" run_id="$3"
    local job_name="${SLURM_JOB_NAME:-job}" job_id="${SLURM_JOB_ID:-}"
    if [ -z "${job_id}" ]; then
        echo "[wandb-logs] no SLURM_JOB_ID; skipping log upload."
        return 0
    fi
    WANDB_LOG_PROJECT="${project}" \
    WANDB_LOG_NAME="${run_name}" \
    WANDB_LOG_ID="${run_id}" \
    WANDB_LOG_OUT="${LOG_DIR}/${job_name}_${job_id}.out" \
    WANDB_LOG_ERR="${LOG_DIR}/${job_name}_${job_id}.err" \
    python - <<'PY' || echo "[wandb-logs] upload failed (non-fatal)."
import os
import wandb

run = wandb.init(
    project=os.environ["WANDB_LOG_PROJECT"],
    name=os.environ.get("WANDB_LOG_NAME") or None,
    id=os.environ.get("WANDB_LOG_ID") or None,
    resume="allow",
)
for path in (os.environ.get("WANDB_LOG_OUT"), os.environ.get("WANDB_LOG_ERR")):
    if path and os.path.exists(path):
        # base_path -> the file is stored at top level (its basename) in Files.
        wandb.save(path, base_path=os.path.dirname(path), policy="now")
        print(f"[wandb-logs] uploaded {path}")
run.finish()
PY
}

apply_cublas_ld_fix() {
    [ "${FIX_CUBLAS_LD:-0}" = "1" ] || return 0
    # Pure-bash split/rebuild, no pipeline: callers run under `set -euo pipefail`, and
    # a grep that matches nothing (already-empty LD_LIBRARY_PATH) exits 1, which
    # pipefail would turn into a killed job.
    local cleaned="" entry
    local IFS=':'
    for entry in ${LD_LIBRARY_PATH:-}; do
        [ -n "${entry}" ] || continue
        case "${entry}" in /usr/local/cuda*) continue ;; esac
        cleaned="${cleaned:+${cleaned}:}${entry}"
    done
    export LD_LIBRARY_PATH="${cleaned}"
    echo "[cublas-ld-fix] FIX_CUBLAS_LD=1 -> stripped /usr/local/cuda* from LD_LIBRARY_PATH"
    echo "[cublas-ld-fix] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<empty>}"
}

# --- Optional workaround for vLLM's SageMaker bootstrap on AWS HyperPod clusters ---
# vLLM unconditionally runs a SageMaker "container standards" bootstrap that tries
# to load a handler from ${SAGEMAKER_MODEL_PATH:-/opt/ml/model}/model.py. On a
# HyperPod node without read access to /opt/ml/model this raises PermissionError
# and kills startup, so we point it at a writable empty dir instead. Harmless on
# non-HyperPod machines (the env vars it touches are simply unset there).
setup_cluster_env() {
    mkdir -p "${TMP_DIR}" "${HF_HOME_DIR}" "${LOG_DIR}" \
             "${STORAGE_DIR}" "${DATA_DIR}" "${CKPT_DIR}"
    export TMPDIR="${TMP_DIR}"
    export HF_HOME="${HF_HOME_DIR}"
    export HF_HUB_ENABLE_HF_TRANSFER=1

    apply_cublas_ld_fix   # no-op unless FIX_CUBLAS_LD=1

    # HyperPod exports ROCR_VISIBLE_DEVICES (AMD/ROCm) alongside CUDA_VISIBLE_DEVICES
    # even on NVIDIA nodes; verl's worker init hard-fails when both are set ("Please
    # don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set."). Clear
    # the spurious AMD var so torchrun workers inherit a clean env.
    unset ROCR_VISIBLE_DEVICES 2>/dev/null || true

    local sm_dir="${TMP_DIR}/sagemaker_model"
    mkdir -p "${sm_dir}"
    export SAGEMAKER_MODEL_PATH="${sm_dir}"
    unset SAGEMAKER_PROGRAM 2>/dev/null || true
    unset SM_MODEL_DIR 2>/dev/null || true
}

activate_env() {
    if [ -z "${CONDA_SH:-}" ] || [ ! -f "${CONDA_SH}" ]; then
        echo "[config] CONDA_SH not found -- set it explicitly, e.g." >&2
        echo "[config]   export CONDA_SH=/path/to/conda/etc/profile.d/conda.sh" >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    set +u
    conda activate "${CONDA_ENV_NAME}"
    set -u
}
