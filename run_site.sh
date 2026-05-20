#!/bin/bash
set -euo pipefail
set -x

# ─── GAP-GRPO: Parameterized N-Site Launch Script ─────────────────────────
#
# Usage:
#   SITE_INDEX=0 NUM_SITES=3 bash run_site.sh              # primary site
#   SITE_INDEX=1 NUM_SITES=3 bash run_site.sh              # secondary site
#   SITE_INDEX=2 NUM_SITES=3 EXCHANGE_HOST=10.0.1.1 bash run_site.sh
#
# Required env vars:
#   SITE_INDEX     - Integer index for this site (0 = primary, starts rollouter first)
#   NUM_SITES      - Total number of sites participating
#
# Optional env vars:
#   EXCHANGE_HOST  - TCP exchange server hostname (default: 127.0.0.1)
#   EXCHANGE_PORT  - TCP exchange server port (default: 18080)
#   EXCHANGE_RUN_ID - Run ID shared across all sites (default: gapgrpo_run_001)
#   CUDA_VISIBLE_DEVICES - GPUs to use (default: 0,1,2,3)
#   model_path     - Path to model checkpoint (default: $(pwd)/Qwen3-8B)
#   N_RESP_PER_PROMPT - Number of responses per prompt for this site (default: 4)
#   MINI_BATCH_SIZE - Training mini-batch size (default: auto-computed)
#   EXPERIMENT_NAME - SwanLab experiment name (default: auto-generated)
# ───────────────────────────────────────────────────────────────────────────

SITE_INDEX="${SITE_INDEX:?Must set SITE_INDEX (e.g. 0, 1, 2)}"
NUM_SITES="${NUM_SITES:?Must set NUM_SITES (e.g. 2, 3, 4)}"

# ─── Environment ──────────────────────────────────────────────────────────
export VERL_USE_MODELSCOPE=True
export HYDRA_CONFIG_PATH="$(pwd)/verl/verl/trainer/config"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export RAY_ADDRESS=${RAY_ADDRESS:-127.0.0.1:6379}
export EXCHANGE_HOST=${EXCHANGE_HOST:-127.0.0.1}
export EXCHANGE_PORT=${EXCHANGE_PORT:-18080}
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"HPA4rMyhiXXBFNbyKiW4A"}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export RAY_memory_usage_threshold=0.99

# ─── Training parameters ─────────────────────────────────────────────────
adv_estimator="grpo"
train_files="data/gsm8k/train.parquet"
val_files="data/gsm8k/test.parquet"
model_path="${model_path:-$(pwd)/Qwen3-8B}"
project_name="${PROJECT_NAME:-gap_grpo_qwen3_06b_gsm8k}"
experiment_name="${EXPERIMENT_NAME:-site${SITE_INDEX}_${NUM_SITES}sites}"

# ─── Ray cluster setup ───────────────────────────────────────────────────
RAY_BIN="${RAY_BIN:-}"
if [ -z "$RAY_BIN" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/ray" ]; then
  RAY_BIN="${CONDA_PREFIX}/bin/ray"
fi
if [ -z "$RAY_BIN" ] && command -v ray &>/dev/null; then
  RAY_BIN="$(command -v ray)"
fi
RAY_BIN="${RAY_BIN:-ray}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_site_${SITE_INDEX}}"
mkdir -p "$RAY_TEMP_DIR"

# Auto-detect GPU count from CUDA_VISIBLE_DEVICES
IFS=',' read -ra _GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${NUM_GPUS:-${#_GPU_ARRAY[@]}}"
NUM_CPUS="${NUM_CPUS:-60}"

# Worker port range: offset per site to avoid conflicts on shared machine
RAY_PORT_OFFSET=$(( (RAY_PORT - 6379) * 10000 ))
MIN_WORKER_PORT=${MIN_WORKER_PORT:-$(( 20000 + RAY_PORT_OFFSET ))}
MAX_WORKER_PORT=${MAX_WORKER_PORT:-$(( MIN_WORKER_PORT + 9999 ))}

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$RAY_BIN" start \
  --head --port="$RAY_PORT" --num-gpus="$NUM_GPUS" --num-cpus="$NUM_CPUS" \
  --min-worker-port="$MIN_WORKER_PORT" --max-worker-port="$MAX_WORKER_PORT" \
  --temp-dir "$RAY_TEMP_DIR" --include-dashboard=false
sleep 3

export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"

# ─── Rollout/training config ─────────────────────────────────────────────
rollout_mode="async"
rollout_name="vllm"

train_prompt_bsz=0
gen_prompt_bsz=1
max_model_len=${MAX_MODEL_LEN:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
max_num_batched_tokens=$((max_response_length * 4))
n_resp_per_prompt=${N_RESP_PER_PROMPT:-4}
use_dynamic_bsz=true
total_rollout_steps=${TOTAL_ROLLOUT_STEPS:-$((400*1*160))}
mini_batch_size=${MINI_BATCH_SIZE:-320}
require_batches=${REQUIRE_BATCHES:-1}
test_freq=${TEST_FREQ:-1000}

staleness_threshold=${STALENESS_THRESHOLD:-3}
trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP:-1}
partial_rollout=${PARTIAL_ROLLOUT:-false}

# ─── Exchange config ─────────────────────────────────────────────────────
EXCHANGE_RUN_ID="${EXCHANGE_RUN_ID:-gapgrpo_run_001}"

# Determine mode: primary site (index 0) starts rollouter first,
# all other sites start trainer first to consume from exchange
if [ "$SITE_INDEX" = "0" ]; then
    EXCHANGE_MODE="both"
else
    EXCHANGE_MODE="${EXCHANGE_MODE:-train_first}"
fi

# Trainer/rollout GPU split
TRAINER_GPUS_PER_NODE=${TRAINER_GPUS_PER_NODE:-2}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-2}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}

# ─── Launch ───────────────────────────────────────────────────────────────
echo "========================================"
echo "GAP-GRPO Site ${SITE_INDEX} / ${NUM_SITES} sites"
echo "Exchange: ${EXCHANGE_HOST}:${EXCHANGE_PORT}"
echo "Mode: ${EXCHANGE_MODE}"
echo "Mini-batch size: ${mini_batch_size}"
echo "N responses per prompt: ${n_resp_per_prompt}"
echo "========================================"

PYTHONUNBUFFERED=1 python -m verl.experimental.fully_async_policy.fully_async_exchange_main \
    data.train_files=${train_files} \
    data.val_files=${val_files} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat:-True} \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1=\"${VLLM_USE_V1}\"" \
    data.shuffle=True \
    data.seed=99 \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.model.path=${model_path} \
    algorithm.adv_estimator=${adv_estimator} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARALLEL_SIZE} \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    trainer.val_before_train=False \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.test_freq="${test_freq}" \
    trainer.logger='[console,swanlab]' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${TRAINER_GPUS_PER_NODE} \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=${ROLLOUT_GPUS_PER_NODE} \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    async_training.require_batches=${require_batches} \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.partial_rollout="${partial_rollout}" \
    +exchange.side="${SITE_INDEX}" \
    +exchange.mode="${EXCHANGE_MODE}" \
    +exchange.site_index="${SITE_INDEX}" \
    +exchange.run_id="${EXCHANGE_RUN_ID}" \
    +exchange.backend=tcp \
    +exchange.host="${EXCHANGE_HOST}" \
    +exchange.port="${EXCHANGE_PORT}" \
    +exchange.enable_group_merge=true \
    +exchange.expected_per_hash="${NUM_SITES}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.actor.fsdp_config.model_dtype="bfloat16" \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size}
