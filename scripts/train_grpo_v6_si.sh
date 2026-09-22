#!/usr/bin/env bash
# Non-SLURM GRPO-v6-SI launcher: same reward/training config as
# grpo_training/train/launch_grpo_v6_si_onenode_h200.sbatch, for a single
# multi-GPU machine with no scheduler. Co-locates a self-hosted SciJudge (impact
# judge) server on the high GPUs; the distance judge runs on AWS Bedrock.
#
#   BASE_MODEL=/path/to/sft/checkpoint GPU_IDS=0,1,2,3,4,5,6,7 \
#     bash scripts/train_grpo_v6_si.sh
#
# See the .sbatch file's header comment for the full design rationale (GPU
# split, batch divisibility, why co-location, why Bedrock for distance).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/grpo_training/config.sh"
setup_cluster_env
setup_wandb
activate_env
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ -n "${AWS_PROFILE:-}" ] && [ ! -f "${HOME}/.aws/config" ]; then
    echo "[bedrock] unsetting AWS_PROFILE=${AWS_PROFILE} (no ~/.aws/config)"
    unset AWS_PROFILE
fi
export AWS_REGION="${AWS_REGION:-us-west-2}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a _GPU_ARR <<< "${GPU_IDS}"
TOTAL_GPUS="${#_GPU_ARR[@]}"
SCIJUDGE_TP_SIZE="${SCIJUDGE_TP_SIZE:-2}"
N_GPUS="${N_GPUS:-$(( TOTAL_GPUS - SCIJUDGE_TP_SIZE ))}"
TRAIN_GPUS="$(IFS=,; echo "${_GPU_ARR[*]:0:${N_GPUS}}")"
JUDGE_GPUS="$(IFS=,; echo "${_GPU_ARR[*]:${N_GPUS}}")"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-12}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-12}"
if [ "$(( N_GPUS + SCIJUDGE_TP_SIZE ))" -gt "${TOTAL_GPUS}" ] || [ "${N_GPUS}" -lt 1 ]; then
    echo "ERROR: N_GPUS(${N_GPUS}) + SCIJUDGE_TP_SIZE(${SCIJUDGE_TP_SIZE}) > GPUs (${TOTAL_GPUS})." >&2
    exit 1
fi
for _bs_name in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE; do
    _bs="${!_bs_name}"
    if [ "$(( _bs % N_GPUS ))" -ne 0 ]; then
        echo "ERROR: ${_bs_name}=${_bs} is not divisible by N_GPUS=${N_GPUS}." >&2
        exit 1
    fi
done

export SCIJUDGE_ENDPOINT_FILE="${SCIJUDGE_ENDPOINT_FILE:-${TMP_DIR}/scijudge_endpoint_local.txt}"
export SCIJUDGE_PORT="${SCIJUDGE_PORT:-8002}"
export F9B_DISTANCE_MAX_TOKENS="${F9B_DISTANCE_MAX_TOKENS:-1024}"
export F9B_MAX_CONCURRENT_IDEA="${F9B_MAX_CONCURRENT_IDEA:-10}"
export F9B_MAX_CONCURRENT_IMPACT="${F9B_MAX_CONCURRENT_IMPACT:-16}"
export REWARD_FN="${REWARD_FN:-${GRPO_DIR}/rewards/f9b_impdist_len_v6_si_bedrock.py}"
BEDROCK_MODEL="${F9B_BEDROCK_MODEL:-openai.gpt-oss-120b-1:0}"
export F9B_NUM_IDEAS="${F9B_NUM_IDEAS:-3}"
export F9B_LEN_COEF="${F9B_LEN_COEF:-0.0}"
export F9B_LEN_TARGET="${F9B_LEN_TARGET:-834}"
export F9B_LEN_TOKENIZER="${F9B_LEN_TOKENIZER:-Qwen/Qwen3-8B}"

SFT_CKPT="${SFT_CKPT:-${STORAGE_DIR}/sft/checkpoints/qwen3-8b-sft-ideation-si/global_step_N/huggingface}"
BASE_MODEL="${BASE_MODEL:-${SFT_CKPT}}"
GRPO_DATA_DIR="${GRPO_DATA_DIR:-${DATA_DIR}/rl_si}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-8b-grpo-v6-si-onenode}"
RUN_NAME="$(make_run_name "${EXPERIMENT_NAME}")"
OUT_DIR="${CKPT_DIR}/${EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUT_DIR}/rollouts}"
mkdir -p "${OUT_DIR}" "${ROLLOUT_DATA_DIR}"

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
SAVE_FREQ="${SAVE_FREQ:-50}"
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
export WANDB_RUN_ID_TRAIN="${WANDB_RUN_ID_TRAIN:-${EXPERIMENT_NAME}-local}"

echo "================= GRPO-v6-SI (non-SLURM) ================="
echo "Policy  : ${BASE_MODEL}"
echo "Reward  : ${REWARD_FN}"
echo "Distance: ${BEDROCK_MODEL} via bedrock (region ${AWS_REGION})"
echo "GPUs    : train=[${TRAIN_GPUS}] judge=[${JUDGE_GPUS}]"
echo "============================================================"

for f in "${GRPO_DATA_DIR}/train.parquet" "${GRPO_DATA_DIR}/val.parquet"; do
    [ -f "$f" ] || { echo "ERROR: missing $f. Run scripts/prepare_rl_data.sh first." >&2; exit 1; }
done
if [ "${VAL_FILES}" = "${GRPO_DATA_DIR}/val_small.parquet" ] && [ ! -f "${VAL_FILES}" ]; then
    python3 "${GRPO_DIR}/data_prep/make_val_small.py" \
        --src "${GRPO_DATA_DIR}/val.parquet" --out "${VAL_FILES}" --n "${VAL_SMALL_N}" \
        || VAL_FILES="${GRPO_DATA_DIR}/val.parquet"
fi
[ -f "${BASE_MODEL}/config.json" ] && ls "${BASE_MODEL}"/*.safetensors >/dev/null 2>&1 || \
    { echo "ERROR: BASE_MODEL has no HF weights: ${BASE_MODEL}" >&2; exit 1; }

echo "[bedrock] preflight ..."
python3 "${GRPO_DIR}/extras/check_bedrock.py" --model "${BEDROCK_MODEL}" \
    || { echo "ERROR: Bedrock preflight failed." >&2; exit 1; }

rm -f "${SCIJUDGE_ENDPOINT_FILE}"
echo "[scijudge] launching on GPUs ${JUDGE_GPUS} (TP=${SCIJUDGE_TP_SIZE}) ..."
CUDA_VISIBLE_DEVICES="${JUDGE_GPUS}" python "${GRPO_DIR}/inference/serve_vllm.py" serve "${SCIJUDGE_MODEL}" \
    --host 0.0.0.0 --port "${SCIJUDGE_PORT}" \
    --tensor-parallel-size "${SCIJUDGE_TP_SIZE}" \
    --max-model-len "${SCIJUDGE_MAX_MODEL_LEN:-8192}" \
    --gpu-memory-utilization "${SCIJUDGE_GPU_MEM_UTIL:-0.90}" \
    --dtype bfloat16 --seed "${SCIJUDGE_SEED:-42}" --enable-prefix-caching \
    > "${LOG_DIR}/scijudge_local.log" 2>&1 &
SCIJUDGE_PID=$!

cat > "${SCIJUDGE_ENDPOINT_FILE}" <<EOF
URL=http://127.0.0.1:${SCIJUDGE_PORT}
MODEL=${SCIJUDGE_MODEL}
EOF

cleanup() {
    echo "[cleanup] stopping SciJudge (pid ${SCIJUDGE_PID}) ..."
    kill "${SCIJUDGE_PID}" 2>/dev/null || true
    rm -f "${SCIJUDGE_ENDPOINT_FILE}"
}
trap cleanup EXIT INT TERM

JUDGE_WAIT_TIMEOUT="${JUDGE_WAIT_TIMEOUT:-3600}"
waited=0
echo "[judge-gate] waiting for co-located SciJudge (timeout ${JUDGE_WAIT_TIMEOUT}s) ..."
while :; do
    code="$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
            "http://127.0.0.1:${SCIJUDGE_PORT}/v1/models" 2>/dev/null || echo 000)"
    [ "${code}" = "200" ] && { echo "[judge-gate] SciJudge READY after ${waited}s"; break; }
    kill -0 "${SCIJUDGE_PID}" 2>/dev/null || { echo "ERROR: SciJudge process exited; see scijudge_local.log" >&2; exit 1; }
    [ "${waited}" -ge "${JUDGE_WAIT_TIMEOUT}" ] && { echo "ERROR: SciJudge not ready after ${JUDGE_WAIT_TIMEOUT}s." >&2; exit 1; }
    sleep 20; waited=$((waited + 20))
done

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

echo "Training finished (rc=${TRAIN_RC})."
exit "${TRAIN_RC}"
