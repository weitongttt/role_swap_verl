#!/bin/bash
# run_dual_stream_B.sh
# Side B of DualStream: synchronous hybrid-engine training, cross-pool sample exchange.
#
# Flow:
#   step 1 (bootstrap): skip own rollout, pull A's batch, train, update_weights
#   step 2..N          : rollout -> push_B -> pull_A -> train(batch_A) -> update_weights
#
# Start order: exchange server first, then A, then B.
#   bash run_exchange_server.sh   # terminal 1
#   bash run_dual_stream_A.sh     # terminal 2
#   bash run_dual_stream_B.sh     # terminal 3

set -euo pipefail
set -x

# ── environment ────────────────────────────────────────────────────────────
export VERL_USE_MODELSCOPE=${VERL_USE_MODELSCOPE:-True}
export HYDRA_CONFIG_PATH="/zhangshihao/weitong/verl_swap/verl/verl/trainer/config"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_B:-1}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"hlo16D6KKxblfDAgvGxVQ"}

# ── Ray (Side B owns port 6380) ────────────────────────────────────────────
RAY_BIN="${RAY_BIN:-}"
if [ -z "$RAY_BIN" ] && [ -x "/zhangshihao/weitong/anaconda3/envs/verl/bin/ray" ]; then
  RAY_BIN="/zhangshihao/weitong/anaconda3/envs/verl/bin/ray"
fi
RAY_BIN="${RAY_BIN:-ray}"

RAY_PORT_B="${RAY_PORT_B:-6380}"
RAY_TEMP_DIR_B="${RAY_TEMP_DIR_B:-/tmp/ray_ds_b}"
mkdir -p "$RAY_TEMP_DIR_B"

if ! timeout 2 bash -c "</dev/tcp/127.0.0.1/${RAY_PORT_B}" >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_B:-1}" \
    "$RAY_BIN" start --head \
      --port="${RAY_PORT_B}" \
      --num-gpus=1 \
      --num-cpus=30 \
      --temp-dir "$RAY_TEMP_DIR_B" \
      --include-dashboard=false \
      --min-worker-port=30000 \
      --max-worker-port=39999
  sleep 3
fi
export RAY_ADDRESS="127.0.0.1:${RAY_PORT_B}"

# ── exchange server coordinates ────────────────────────────────────────────
export EXCHANGE_HOST=${EXCHANGE_HOST:-127.0.0.1}
export EXCHANGE_PORT=${EXCHANGE_PORT:-18080}

# wait for Side A to write the shared run_id
EXCHANGE_RUN_ID_FILE="${EXCHANGE_RUN_ID_FILE:-/tmp/verl_ds_run_id}"
EXCHANGE_RUN_ID_READY_FILE="${EXCHANGE_RUN_ID_READY_FILE:-/tmp/verl_ds_run_id.ready}"
echo "[dual_stream_B] waiting for run_id ready file: ${EXCHANGE_RUN_ID_READY_FILE}"
for i in $(seq 1 180); do
  if [ -f "$EXCHANGE_RUN_ID_READY_FILE" ] && [ -s "$EXCHANGE_RUN_ID_FILE" ]; then
    READY_RUN_ID="$(cat "$EXCHANGE_RUN_ID_READY_FILE" 2>/dev/null || true)"
    FILE_RUN_ID="$(cat "$EXCHANGE_RUN_ID_FILE" 2>/dev/null || true)"
    if [ -n "$READY_RUN_ID" ] && [ "$READY_RUN_ID" = "$FILE_RUN_ID" ]; then
      break
    fi
  fi
  sleep 1
done
if [ ! -f "$EXCHANGE_RUN_ID_READY_FILE" ] || [ ! -s "$EXCHANGE_RUN_ID_FILE" ]; then
  echo "[dual_stream_B] ERROR: timed out waiting for run_id ready"
  exit 1
fi
EXCHANGE_RUN_ID="$(cat "$EXCHANGE_RUN_ID_FILE")"
echo "[dual_stream_B] run_id=${EXCHANGE_RUN_ID}"

# ── training hyper-parameters (must match Side A) ─────────────────────────
model_path="Qwen/Qwen2.5-0.5B-Instruct"
train_files="data/gsm8k/train.parquet"
val_files="data/gsm8k/test.parquet"

n_resp_per_prompt=4          # vllm rollout.n
max_prompt_length=1024
max_response_length=2048
gpu_memory_utilization=0.5

ppo_mini_batch_size=128      # lower peak memory to avoid OOM in backward
ppo_micro_batch_size_per_gpu=4
require_batches=1            # collect require_batches * ppo_mini_batch_size samples before training
train_batch_size=$((ppo_mini_batch_size * require_batches))  # = required_samples, auto-computed
required_samples=$((ppo_mini_batch_size * require_batches * n_resp_per_prompt))

total_training_steps=200     # total trainer update steps
total_epochs=1             # set large enough so dataloader never runs out
test_freq=50
save_freq=200

project_name="dual_stream"
experiment_name="ds_plan2_B"
echo "[dual_stream_B] ppo_mini_batch_size=${ppo_mini_batch_size} require_batches=${require_batches} rollout_n=${n_resp_per_prompt} required_samples=${required_samples}"

# ── launch ─────────────────────────────────────────────────────────────────
PYTHONUNBUFFERED=1 python -m verl.experimental.fully_async_policy.dual_stream_trainer \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.shuffle=True \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    +actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.max_model_len=3072 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=True \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.logger='[console,swanlab]' \
    trainer.val_before_train=False \
    +exchange.side=B \
    +exchange.host="${EXCHANGE_HOST}" \
    +exchange.port="${EXCHANGE_PORT}" \
    +exchange.run_id="${EXCHANGE_RUN_ID}" \
    +exchange.require_batches=${require_batches} \
    +exchange.wait_timeout_s=300
