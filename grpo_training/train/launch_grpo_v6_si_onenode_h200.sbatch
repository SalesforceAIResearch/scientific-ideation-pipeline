#!/usr/bin/env bash
#SBATCH --job-name=qwen3_8b_grpo_v6_si_onenode
#SBATCH --output=grpo_training/logs/%x_%j.out
#SBATCH --error=grpo_training/logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
# Set --partition/--account for your cluster, either here or on the sbatch
# command line (which overrides these directives), e.g.:
#   sbatch --partition=<yours> --account=<yours> grpo_training/train/launch_grpo_v6_si_onenode_h200.sbatch
# NOTE: --output/--error above are relative to the directory you run `sbatch`
# from; `mkdir -p grpo_training/logs` first (or submit from the repo root).

# ONE-NODE GRPO v6 on the SI (v1) SCHEMA: policy training + SciJudge co-located,
# distance judge on Bedrock.
#
# WHAT THIS REPLACES. The standard flow needs THREE allocations that must all be
# live simultaneously:
#     serve_judge.sbatch     (gpt-oss-120b, 8 GPUs)  -- distance
#     serve_scijudge.sbatch  (SciJudge-30B,  8 GPUs) -- impact
#     launch_grpo_*.sh       (policy,        8 GPUs) -- training
# This script needs ONE, by making two changes:
#   1. DISTANCE -> AWS Bedrock (openai.gpt-oss-120b-1:0), the same open weights
#      on demand. Frees the entire gpt-oss node. Selected by pointing REWARD_FN at
#      rewards/f9b_impdist_len_v6_bedrock.py, a shim that reuses v6's scoring code
#      verbatim and only swaps the client (transport: common/bedrock.py). v6 itself
#      is untouched, so scores stay comparable across the two flows.
#   2. IMPACT -> SciJudge-30B co-located in THIS allocation as an `srun` job-step
#      on a dedicated slice of GPUs. SciJudge is NOT on Bedrock (no OpenMOSS
#      provider), so it must stay self-hosted -- but it does not need its own node.
#
# WHY A JOB-STEP AND NOT A SECOND SBATCH. serve_scijudge.sbatch takes --exclusive
# precisely BECAUSE co-tenancy once got it SIGKILLed as collateral when another
# user's job on the same node was scancel'd. Its own header names the escape
# hatch: the other GPUs are usable "from WITHIN this allocation (e.g. an srun
# job-step)". A step inside our allocation shares our fate and no one else's, so
# we get co-location without reintroducing that failure mode.
#
# GPU SPLIT (8x H200-141GB). SciJudge-30B is a Qwen3-30B-A3B MoE (~60GB bf16), so
# TP=2 holds it with KV headroom; the policy gets the rest.
#   GPUs 0..$((N_GPUS-1))  -> training (Qwen3-8B: ~16GB weights + optim + rollout)
#   GPUs $N_GPUS..7        -> SciJudge server
#
# BATCH DIVISIBILITY. verl shards the batch across training GPUs, so
# TRAIN_BATCH_SIZE and PPO_MINI_BATCH_SIZE must divide by N_GPUS. The 8-GPU
# baseline used 16/16, which does NOT divide by 6 -- so this script defaults to
# 12/12 for N_GPUS=6. That is a REAL departure from the baseline batch size; if
# you need 16 for comparability, set SCIJUDGE_TP_SIZE=4 N_GPUS=4 TRAIN_BATCH_SIZE=16
# instead (fewer training GPUs, unchanged optimization). The preflight below
# hard-fails on a bad combination rather than letting verl die deep in a worker.
#
# PREREQS
#   * AWS credentials reachable by boto3 (e.g. an instance role on AWS, or
#     AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY / a configured AWS_PROFILE
#     elsewhere). A stale AWS_PROFILE pointing at a missing profile breaks the
#     CLI -- this script unsets it in that case. Verify: extras/check_bedrock.py
#   * data_prep parquet under $GRPO_DATA_DIR, built with
#     --prompt-type ideation_si --num-ideas 3 (--tokenizer Qwen/Qwen3-8B)
#   * BASE_MODEL = an SFT checkpoint with HF weights (see SFT_CKPT below)
#
#   sbatch grpo_training/train/launch_grpo_v6_si_onenode_h200.sbatch
#   # 4-GPU training variant, baseline batch size:
#   SCIJUDGE_TP_SIZE=4 N_GPUS=4 TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 sbatch ...
#
# WHAT DIFFERS FROM launch_grpo_v6_onenode_h200.sbatch -- seven defaults, no logic:
#   REWARD_FN          -> rewards/f9b_impdist_len_v6_si_bedrock.py  (SI shim)
#   F9B_NUM_IDEAS      5    -> 3       (forced by the data; see below)
#   F9B_LEN_TARGET     1670 -> 834     (v1 K=3 3-field card p90)
#   GRPO_DATA_DIR      rl   -> rl_si
#   SFT_CKPT           -> the SI SFT checkpoint (MUST be set; see below)
#   MAX_PROMPT_LENGTH  7168 -> 10240   (SI prompts are longer; see below)
#   EXPERIMENT_NAME / job-name
# Everything else -- the SciJudge co-location, the GPU split, the judge gate, the
# Bedrock preflight, the cuBLAS LD fix, the batch-divisibility checks -- is carried
# over verbatim, and so is the reward math itself: the SI shim re-exports v6's
# compute_score unchanged and only rebinds the parser, the format gate and the
# distance rubric. v0 and SI scores come from the same code path.
#
# WHY MAX_PROMPT_LENGTH=10240. Each SI Document N block is a full SI source card and
# the template carries two label taxonomies, so prompts run longer than v0's. On the
# built rl_si parquet (Qwen3-8B tokenizer): train p50 5035 / p90 6480 / p99 8160 /
# max 10758; test p50 5313 / p90 7408 / p99 9996 / max 16243. At v0's 7168 default
# verl would silently drop 3.8% of train and 12.5% of test rows. 10240 leaves 0.06%
# of train and 0.9% of test over budget, and ROLLOUT_MAX_MODEL_LEN derives to
# 10240+16384 = 26624, inside Qwen3-8B's max_position_embeddings of 40960. Fallback
# if step time is tight: MAX_PROMPT_LENGTH=8192 (drops 4.8% of test).
#
# YOU MUST SET SFT_CKPT. There is no SI SFT checkpoint yet -- the default path below
# has a placeholder global_step_N and the preflight will fail on it. Train one first,
# e.g. (adjust to your own SFT launcher/data prep):
#   sbatch --export=ALL,PROMPT_TYPE=ideation_si,\
#     HF_DATA_REPO=${HF_DATASET_REPO},NUM_IDEAS=3,\
#     EXPERIMENT_NAME=qwen3-8b-sft-ideation-si,\
#     SFT_DATA_DIR=${STORAGE_DIR}/sft_ideation_si_qwen3 \
#     grpo_training/train/run_sft_ideation_qwen3_8b_h200.sbatch
# A v0-format policy under the SI reward emits v0 cards, and EVERY rollout then hits
# the terminal-0 format gate -- the run would look broken from step 0.
#
# This script is ADDITIVE: it does not modify the existing launchers, serve
# scripts, common/clients.py, or rewards/f9b_impdist_len_v6.py. The Bedrock path is
# entirely new files (common/bedrock.py, the v6_bedrock shim, extras/check_bedrock.py)
# plus documented F9B_* env knobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config.sh"
setup_cluster_env
setup_wandb
activate_env
export PYTORCH_ALLOC_CONF=expandable_segments:True

# A stale AWS_PROFILE (e.g. an SSO profile with no ~/.aws/config on this cluster)
# makes credential resolution fail even though the instance role is available.
# Unset it so boto3 falls through to the role; BEDROCK_REGION/AWS_REGION still apply.
if [ -n "${AWS_PROFILE:-}" ] && [ ! -f "${HOME}/.aws/config" ]; then
    echo "[bedrock] unsetting AWS_PROFILE=${AWS_PROFILE} (no ~/.aws/config; using instance role)"
    unset AWS_PROFILE
fi
export AWS_REGION="${AWS_REGION:-us-west-2}"

# --------------------------------------------------------------------------- #
# GPU partition between the co-located server and the trainer
# --------------------------------------------------------------------------- #
TOTAL_GPUS="${SLURM_GPUS_ON_NODE:-8}"
SCIJUDGE_TP_SIZE="${SCIJUDGE_TP_SIZE:-2}"
N_GPUS="${N_GPUS:-$(( TOTAL_GPUS - SCIJUDGE_TP_SIZE ))}"
# Contiguous slices: trainer takes the low GPUs, the judge server the high ones.
TRAIN_GPUS="$(seq -s, 0 $((N_GPUS - 1)))"
JUDGE_GPUS="$(seq -s, ${N_GPUS} $((TOTAL_GPUS - 1)))"

# Batch sizes default to values that divide N_GPUS (see BATCH DIVISIBILITY above).
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-12}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-12}"

# ---- preflight: fail FAST and loudly on an unusable split ----
if [ "$(( N_GPUS + SCIJUDGE_TP_SIZE ))" -gt "${TOTAL_GPUS}" ]; then
    echo "ERROR: N_GPUS(${N_GPUS}) + SCIJUDGE_TP_SIZE(${SCIJUDGE_TP_SIZE}) > GPUs on node (${TOTAL_GPUS})." >&2
    exit 1
fi
if [ "${N_GPUS}" -lt 1 ] || [ "${SCIJUDGE_TP_SIZE}" -lt 1 ]; then
    echo "ERROR: both N_GPUS(${N_GPUS}) and SCIJUDGE_TP_SIZE(${SCIJUDGE_TP_SIZE}) must be >= 1." >&2
    exit 1
fi
for _bs_name in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE; do
    _bs="${!_bs_name}"
    if [ "$(( _bs % N_GPUS ))" -ne 0 ]; then
        echo "ERROR: ${_bs_name}=${_bs} is not divisible by N_GPUS=${N_GPUS}." >&2
        echo "       Pick a multiple of ${N_GPUS} (e.g. $(( (_bs / N_GPUS + 1) * N_GPUS )))," >&2
        echo "       or change the split: SCIJUDGE_TP_SIZE=4 N_GPUS=4 keeps the 16/16 baseline." >&2
        exit 1
    fi
done
# SciJudge-30B needs >=2 H200s for weights+KV at 0.90 util; warn (don't block) on 1.
if [ "${SCIJUDGE_TP_SIZE}" -lt 2 ]; then
    echo "WARN: SCIJUDGE_TP_SIZE=1 for a ~60GB MoE leaves little KV headroom on a 141GB card." >&2
fi

# --------------------------------------------------------------------------- #
# Judge routing: impact -> co-located vLLM; distance -> Bedrock
# --------------------------------------------------------------------------- #
# Per-run endpoint file so a co-scheduled run can't read our URL (and vice versa).
export SCIJUDGE_ENDPOINT_FILE="${SCIJUDGE_ENDPOINT_FILE:-${TMP_DIR}/scijudge_endpoint_${SLURM_JOB_ID:-local}.txt}"
export SCIJUDGE_PORT="${SCIJUDGE_PORT:-8002}"

# DISTANCE via Bedrock -- selected by the REWARD FILE, not by an env switch. The
# v6 reward is unmodified and has no bedrock branch; f9b_impdist_len_v6_bedrock.py
# imports it and rebinds only its distance client + model id. So the routing lives
# in exactly one place, and swapping REWARD_FN back to plain v6 restores the
# three-allocation flow with no other edits.
#
# Bedrock's model id is set INSIDE the shim (F9B_BEDROCK_MODEL). Do NOT set
# F9B_JUDGE_MODEL here: that is v6's self-hosted name, and the shim overrides it.
# Bedrock charges reasoning tokens against max_tokens; 1024 fits reasoning+JSON
# (measured ~225-330 tok). Do not lower this.
export F9B_DISTANCE_MAX_TOKENS="${F9B_DISTANCE_MAX_TOKENS:-1024}"
# Bedrock sustained a 64-way burst with zero throttling, so the per-worker cap can
# stay at the tuned default; a 429 here is retried by the SDK (F9B_BEDROCK_MAX_RETRIES).
export F9B_MAX_CONCURRENT_IDEA="${F9B_MAX_CONCURRENT_IDEA:-10}"
export F9B_MAX_CONCURRENT_IMPACT="${F9B_MAX_CONCURRENT_IMPACT:-16}"

# ---- Reward: F9-B v6 (impact x distance + CARD-ONLY length penalty) ----
# The _bedrock shim -- same math as v6, distance judge on Bedrock.
export REWARD_FN="${REWARD_FN:-${SCRIPT_DIR}/rewards/f9b_impdist_len_v6_si_bedrock.py}"
BEDROCK_MODEL="${F9B_BEDROCK_MODEL:-openai.gpt-oss-120b-1:0}"
# K=3, NOT 5: every v1 row carries num_ideas=3 and the v6 gate hard-fails
# len(ideas) != K, so a 5 here would score every rollout as terminal 0.0.
export F9B_NUM_IDEAS="${F9B_NUM_IDEAS:-3}"
# LENGTH PENALTY. OFF by default in the reward file (LEN_COEF=0); set F9B_LEN_COEF
# to enable. LEN_TARGET 834 = the v1 K=3 SFT-set CARD p90 in Qwen3 tokens over the
# THREE JUDGED fields (mean 760 / p50 758 / p90 834) -- the same construction as
# v0's 1670, so it again holds the SFT-init length rather than compressing below it.
# Note this is a 3-field measurement: `inspiration` is 39% of card tokens but is
# excluded from the penalty by SIIdea.card_text(), which is deliberate -- the gate
# requires it, so pricing it would make the policy pay for a mandatory field.
# The terser gold anchor would be 3 x 381 = 1143, i.e. LARGER, so unlike v0 the gold
# applies no downward pressure here. F9B_LEN_TOKENIZER must match the POLICY.
export F9B_LEN_COEF="${F9B_LEN_COEF:-0.0}"
export F9B_LEN_TARGET="${F9B_LEN_TARGET:-834}"
export F9B_LEN_TOKENIZER="${F9B_LEN_TOKENIZER:-Qwen/Qwen3-8B}"

# --------------------------------------------------------------------------- #
# Policy + data
# --------------------------------------------------------------------------- #
SFT_CKPT="${SFT_CKPT:-${STORAGE_DIR}/sft/checkpoints/qwen3-8b-sft-ideation-si/global_step_N/huggingface}"
BASE_MODEL="${BASE_MODEL:-${SFT_CKPT}}"
GRPO_DATA_DIR="${GRPO_DATA_DIR:-${DATA_DIR}/rl_si}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-8b-grpo-v6-si-onenode}"
RUN_NAME="$(make_run_name "${EXPERIMENT_NAME}")"
OUT_DIR="${CKPT_DIR}/${EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUT_DIR}/rollouts}"
mkdir -p "${OUT_DIR}" "${ROLLOUT_DATA_DIR}"

# Scale knobs: same as the proven H200 impdist run except the batch sizes above.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-10240}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.80}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-32768}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.002}"
NORM_ADV_BY_STD="${NORM_ADV_BY_STD:-True}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
# Every 50 STEPS, not every 10: 89G per GRPO checkpoint (24G model shards + 65G
# optimizer) meant freq=10 wrote 5.7T over a 668-step run and filled the scratch
# mount on our cluster. freq=50 => ~13 checkpoints, ~1.2T.
# Override per-run for a job you expect to be preempted: SAVE_FREQ=10 sbatch ...
SAVE_FREQ="${SAVE_FREQ:-50}"
# ROLLING RETENTION, the knob that decouples save FREQUENCY from FOOTPRINT. verl keeps
# every checkpoint by default; with a cap it rmtree's the oldest actor dirs once the
# count exceeds it (vendor path: utils/checkpoint/fsdp_checkpoint_manager.py:121-125,
# wired at trainer/ppo/ray_trainer.py:851-859). That is what makes a small SAVE_FREQ
# affordable: freq=10 with keep=6 is ~530G, freq=10 unbounded is 11T over 1316 steps.
#
# `null` (the default here) preserves the historical behaviour exactly -- no existing
# run changes. Three things to know before setting it:
#   * It is a REAL shutil.rmtree. Deleted checkpoints are gone, not archived. Merge or
#     push anything you want to keep BEFORE it rolls past (scripts/merge_and_push_ckpts).
#   * It tracks the `actor/` subdir only, so global_step_N/ survives as a 17K stub
#     holding data.pt. Cosmetic; `ls` overcounts checkpoints, `du` does not.
#   * previous_saved_paths is IN-MEMORY, so on a resume the list starts empty and
#     checkpoints written by the previous job are never reaped. Expect the cap to be
#     exceeded across a resume boundary, and prune those by hand.
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-null}"
TEST_FREQ="${TEST_FREQ:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
VAL_SMALL_N="${VAL_SMALL_N:-32}"
VAL_FILES="${VAL_FILES:-${GRPO_DATA_DIR}/val_small.parquet}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-${ROLLOUT_N}}"
export WANDB_RUN_ID_TRAIN="${WANDB_RUN_ID_TRAIN:-${EXPERIMENT_NAME}-${SLURM_JOB_ID:-manual}}"

echo "================= ONE-NODE GRPO v6 ================="
echo "Start   : $(date) | Node: $(hostname) | Job: ${SLURM_JOB_ID:-local}"
echo "Policy  : ${BASE_MODEL}"
echo "Reward  : ${REWARD_FN}"
echo "Impact  : ${SCIJUDGE_MODEL} (co-located, GPUs ${JUDGE_GPUS}, TP=${SCIJUDGE_TP_SIZE})"
echo "Distance: ${BEDROCK_MODEL} via bedrock (region ${AWS_REGION})"
echo "GPUs    : train=[${TRAIN_GPUS}] (${N_GPUS})  judge=[${JUDGE_GPUS}] (${SCIJUDGE_TP_SIZE})"
echo "Batch   : train=${TRAIN_BATCH_SIZE} mini=${PPO_MINI_BATCH_SIZE} rollout_n=${ROLLOUT_N}"
echo "Length  : coef=${F9B_LEN_COEF} target=${F9B_LEN_TARGET} tokenizer=${F9B_LEN_TOKENIZER}"
echo "Ckpt    : save_freq=${SAVE_FREQ} keep=${MAX_ACTOR_CKPT_TO_KEEP} (null = keep all)"
echo "===================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# ---- data + weights must exist before we burn 8 GPUs on a doomed run ----
for f in "${GRPO_DATA_DIR}/train.parquet" "${GRPO_DATA_DIR}/val.parquet"; do
    [ -f "$f" ] || { echo "ERROR: missing $f. Build it with data_prep/prepare_ideation_data.py --stage rl first." >&2; exit 1; }
done
if [ "${VAL_FILES}" = "${GRPO_DATA_DIR}/val_small.parquet" ] && [ ! -f "${VAL_FILES}" ]; then
    echo "[val] building ${VAL_FILES} (${VAL_SMALL_N} prompts) from val.parquet ..."
    python3 "${SCRIPT_DIR}/data_prep/make_val_small.py" \
        --src "${GRPO_DATA_DIR}/val.parquet" --out "${VAL_FILES}" --n "${VAL_SMALL_N}" \
        || { echo "WARN: val_small build failed; falling back to full val.parquet." >&2; VAL_FILES="${GRPO_DATA_DIR}/val.parquet"; }
fi
[ -f "${VAL_FILES}" ] || { echo "ERROR: VAL_FILES missing: ${VAL_FILES}" >&2; exit 1; }
[ -f "${BASE_MODEL}/config.json" ] && ls "${BASE_MODEL}"/*.safetensors >/dev/null 2>&1 || \
    { echo "ERROR: BASE_MODEL has no HF weights: ${BASE_MODEL} (run the SFT job, or set BASE_MODEL)." >&2; exit 1; }

# ---- Bedrock reachability: check BEFORE training, not at step-0 validation ----
# val_before_train fires distance calls immediately, so a credential problem would
# otherwise surface as a wall of judge errors ~20 min in.
echo "[bedrock] preflight ..."
python3 "${SCRIPT_DIR}/extras/check_bedrock.py" --model "${BEDROCK_MODEL}" \
    || { echo "ERROR: Bedrock preflight failed -- fix credentials/region before burning the allocation." >&2; exit 1; }

# --------------------------------------------------------------------------- #
# Step 1: SciJudge server as a co-located job-step
# --------------------------------------------------------------------------- #
# --overlap lets this step share the allocation's resources with the trainer step
# (without it SLURM reserves all CPUs for the first step and the trainer blocks).
# CUDA_VISIBLE_DEVICES gives it only the high GPUs.
#
# WHY THE ASSIGNMENT IS INSIDE bash -c. Job 295 OOMed because SciJudge's TP ranks
# landed on GPUs 0,1 -- the trainer's cards -- and the judge-ready gate passed anyway
# (92GB KV cache allocated, HTTP 200), so the failure only surfaced when FSDP tried to
# allocate and found "GPU 0 ... has 129.19 GiB in use".
#
# Two forms that look right and are NOT (both measured on this cluster, jobs 304/306):
#   CUDA_VISIBLE_DEVICES=6,7 srun ...            -> reaches the step as 6,7 here, but
#     relies on srun not resetting it; it is also the form that was in place when 295
#     placed the judge on 0,1, so it is demonstrably not reliable under --gres=none.
#   srun --export=ALL,CUDA_VISIBLE_DEVICES=6,7   -> arrives as "6". --export takes a
#     COMMA-SEPARATED VARIABLE LIST, so the ",7" is parsed as a second variable named
#     7, silently truncating the GPU list. A TP=2 server would get one GPU.
# Setting it inside the step's own shell is unambiguous: no srun parsing, no
# inheritance question. Verified to arrive as "6,7".
rm -f "${SCIJUDGE_ENDPOINT_FILE}"
echo "[scijudge] launching on GPUs ${JUDGE_GPUS} (TP=${SCIJUDGE_TP_SIZE}) ..."
srun --overlap --ntasks=1 --cpus-per-task=16 --gres=none \
    bash -c "export CUDA_VISIBLE_DEVICES='${JUDGE_GPUS}'; exec \"\$@\"" _ \
    python "${SCRIPT_DIR}/inference/serve_vllm.py" serve "${SCIJUDGE_MODEL}" \
        --host 0.0.0.0 \
        --port "${SCIJUDGE_PORT}" \
        --tensor-parallel-size "${SCIJUDGE_TP_SIZE}" \
        --max-model-len "${SCIJUDGE_MAX_MODEL_LEN:-8192}" \
        --gpu-memory-utilization "${SCIJUDGE_GPU_MEM_UTIL:-0.90}" \
        --dtype bfloat16 \
        --seed "${SCIJUDGE_SEED:-42}" \
        --enable-prefix-caching \
        > "${LOG_DIR}/${SLURM_JOB_NAME:-onenode}_${SLURM_JOB_ID:-manual}.scijudge.log" 2>&1 &
SCIJUDGE_PID=$!

# Publish the endpoint ourselves (the step is a bare vllm serve, so nothing else
# writes it). localhost: the reward runs in THIS allocation, on this node.
NODE_IP="$(hostname -I | awk '{print $1}')"
cat > "${SCIJUDGE_ENDPOINT_FILE}" <<EOF
JOB_ID=${SLURM_JOB_ID:-local}
NODE=$(hostname)
NODE_IP=${NODE_IP}
PORT=${SCIJUDGE_PORT}
URL=http://127.0.0.1:${SCIJUDGE_PORT}
MODEL=${SCIJUDGE_MODEL}
EOF

# Tear the server down with the job, however we exit.
cleanup() {
    echo "[cleanup] stopping SciJudge (pid ${SCIJUDGE_PID}) ..."
    kill "${SCIJUDGE_PID}" 2>/dev/null || true
    rm -f "${SCIJUDGE_ENDPOINT_FILE}"
}
trap cleanup EXIT INT TERM

# ---- gate on SciJudge actually answering (weights load takes ~5-15 min) ----
JUDGE_WAIT_TIMEOUT="${JUDGE_WAIT_TIMEOUT:-3600}"
JUDGE_WAIT_INTERVAL="${JUDGE_WAIT_INTERVAL:-20}"
waited=0
echo "[judge-gate] waiting for co-located SciJudge (timeout ${JUDGE_WAIT_TIMEOUT}s) ..."
while :; do
    code="$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
            "http://127.0.0.1:${SCIJUDGE_PORT}/v1/models" 2>/dev/null || echo 000)"
    if [ "${code}" = "200" ]; then
        echo "[judge-gate] SciJudge READY after ${waited}s"; break
    fi
    # If the step died there is nothing to wait for.
    if ! kill -0 "${SCIJUDGE_PID}" 2>/dev/null; then
        echo "ERROR: SciJudge step exited before becoming ready; see *.scijudge.log" >&2
        exit 1
    fi
    if [ "${waited}" -ge "${JUDGE_WAIT_TIMEOUT}" ]; then
        echo "ERROR: SciJudge not ready after ${JUDGE_WAIT_TIMEOUT}s (last_http=${code})." >&2
        exit 1
    fi
    sleep "${JUDGE_WAIT_INTERVAL}"; waited=$((waited + JUDGE_WAIT_INTERVAL))
done

# ---- assert the judge stayed OFF the training GPUs ----
# The gate above only proves SciJudge ANSWERS -- not that it answered from the right
# cards. Job 295 passed the gate and then OOMed instantly because the judge was on
# GPUs 0,1. Checking occupancy here costs one nvidia-smi and converts a 15-minute
# burn-then-crash into an immediate, legible failure.
echo "[gpu-gate] checking training GPUs [${TRAIN_GPUS}] are free ..."
BUSY=""
for g in $(echo "${TRAIN_GPUS}" | tr ',' ' '); do
    used_mb="$(nvidia-smi --id="${g}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)"
    # A few hundred MB is normal driver/ECC overhead; 2GB+ means a real tenant.
    if [ "${used_mb:-0}" -gt 2048 ]; then
        BUSY="${BUSY} gpu${g}=${used_mb}MiB"
    fi
done
if [ -n "${BUSY}" ]; then
    echo "ERROR: training GPUs are already occupied:${BUSY}" >&2
    echo "       SciJudge should be on [${JUDGE_GPUS}] only. If its workers landed on the" >&2
    echo "       low GPUs, CUDA_VISIBLE_DEVICES did not survive into the srun step --" >&2
    echo "       it must be passed via --export=ALL,CUDA_VISIBLE_DEVICES=... (see above)." >&2
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv >&2 || true
    exit 1
fi
echo "[gpu-gate] OK -- training GPUs clear"

# --------------------------------------------------------------------------- #
# Step 2: GRPO training on the low GPUs
# --------------------------------------------------------------------------- #
echo "----- RESOLVED F9B_* ENV (this run) -----"
env | grep -E "^F9B_" | sort | sed "s/^/  /"
echo "-----------------------------------------"

cd "${VERL_DIR}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"

set +e
PYTHONUNBUFFERED=1 WANDB_RUN_ID="${WANDB_RUN_ID_TRAIN}" python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD}" \
    algorithm.use_kl_in_reward=False \
    data.train_files="${GRPO_DATA_DIR}/train.parquet" \
    data.val_files="${VAL_FILES}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.5 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_ROLLOUT_N}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path="${REWARD_FN}" \
    custom_reward_function.name=compute_score \
    reward.custom_reward_function.path="${REWARD_FN}" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.log_val_generations="${LOG_VAL_GENERATIONS}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${OUT_DIR}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.resume_from_path="${RESUME_FROM_PATH}" \
    trainer.total_epochs="${TOTAL_EPOCHS}"
TRAIN_RC=$?
set -e

echo "Training finished (rc=${TRAIN_RC}): $(date)"
upload_slurm_logs_to_wandb "${WANDB_PROJECT}" "${RUN_NAME}" "${WANDB_RUN_ID_TRAIN}" || true
exit "${TRAIN_RC}"
