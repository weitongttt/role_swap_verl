#!/bin/bash
# plan2 B: strict step alternation (exchange gate) + fully-async separation
set -euo pipefail
set -x

export VERL_USE_MODELSCOPE=True
export HYDRA_CONFIG_PATH="/zhangshihao/weitong/verl_swap/verl/verl/trainer/config"

# Two-card single node defaults: B uses GPU1 only.
# 强制写死，避免环境变量污染
export CUDA_VISIBLE_DEVICES=1

export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"hlo16D6KKxblfDAgvGxVQ"}

RAY_BIN="${RAY_BIN:-}"
if [ -z "$RAY_BIN" ] && [ -x "/zhangshihao/weitong/anaconda3/envs/verl/bin/ray" ]; then
  RAY_BIN="/zhangshihao/weitong/anaconda3/envs/verl/bin/ray"
fi
RAY_BIN="${RAY_BIN:-ray}"

RAY_PORT_B="${RAY_PORT_B:-6380}"
RAY_TEMP_DIR_B="${RAY_TEMP_DIR_B:-/tmp/ray_plan2_b}"
mkdir -p "$RAY_TEMP_DIR_B"

CPUSET_B="${CPUSET_B:-30-59}"
RAY_NUM_CPUS_B="${RAY_NUM_CPUS_B:-30}"
RAY_NUM_GPUS_B="${RAY_NUM_GPUS_B:-1}"

if ! timeout 2 bash -c "</dev/tcp/127.0.0.1/${RAY_PORT_B}" >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    taskset -c "$CPUSET_B" \
    "$RAY_BIN" start --head \
      --port="${RAY_PORT_B}" \
      --num-gpus="${RAY_NUM_GPUS_B}" \
      --num-cpus="${RAY_NUM_CPUS_B}" \
      --temp-dir "$RAY_TEMP_DIR_B" \
      --include-dashboard=false \
      --min-worker-port=30000 \
      --max-worker-port=39999
  sleep 3
else
  GPU_COUNT="$(
    PYTHONUNBUFFERED=1 python - <<'PY'
import ray
ray.init(address="127.0.0.1:6380", ignore_reinit_error=True, logging_level="ERROR")
print(int(ray.cluster_resources().get("GPU", 0)))
ray.shutdown()
PY
  )"
  if [ "${GPU_COUNT:-0}" -lt 1 ]; then
    echo "[plan2_B] ERROR: existing Ray head on port ${RAY_PORT_B} reports GPU=${GPU_COUNT}." >&2
    echo "[plan2_B] Please run: ray stop --force  (will stop both A/B), then restart exchange -> A -> B." >&2
    exit 1
  fi
fi
export RAY_ADDRESS="127.0.0.1:${RAY_PORT_B}"

export EXCHANGE_HOST=${EXCHANGE_HOST:-127.0.0.1}
export EXCHANGE_PORT=${EXCHANGE_PORT:-18080}

EXCHANGE_RUN_ID_FILE="${EXCHANGE_RUN_ID_FILE:-/tmp/verl_plan2_exchange_run_id}"
echo "[plan2_B] waiting run_id file: ${EXCHANGE_RUN_ID_FILE}"
for i in $(seq 1 180); do
  [ -f "$EXCHANGE_RUN_ID_FILE" ] && [ -s "$EXCHANGE_RUN_ID_FILE" ] && break
  sleep 1
done
if [ ! -f "$EXCHANGE_RUN_ID_FILE" ] || [ ! -s "$EXCHANGE_RUN_ID_FILE" ]; then
  echo "[plan2_B] ERROR: timed out waiting for run_id file"
  exit 1
fi
EXCHANGE_RUN_ID="$(cat "$EXCHANGE_RUN_ID_FILE")"
echo "[plan2_B] run_id=${EXCHANGE_RUN_ID}"

model_path="Qwen/Qwen2.5-0.5B-Instruct"
train_files="data/gsm8k/train.parquet"
val_files="data/gsm8k/test.parquet"

n_resp_per_prompt=4
max_prompt_length=1024
max_response_length=2048
gpu_memory_utilization=0.5

ppo_mini_batch_size=128
ppo_micro_batch_size_per_gpu=4
require_batches=1

train_prompt_bsz=0
gen_prompt_bsz=1

total_training_steps=200
total_rollout_steps=$((total_training_steps * ppo_mini_batch_size * 1))

rollout_mode="async"
rollout_name="vllm"
project_name="plan2_gated"
experiment_name="ds_plan2_B"

test_freq=50
save_freq=200

PYTHONUNBUFFERED=1 python -m verl.experimental.fully_async_policy.dual_stream_plan2_main \
  data.train_files=${train_files} \
  data.val_files=${val_files} \
  data.train_batch_size=${train_prompt_bsz} \
  data.gen_batch_size=${gen_prompt_bsz} \
  "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1=\"${VLLM_USE_V1}\"" \
  data.shuffle=True \
  data.max_response_length=${max_response_length} \
  actor_rollout_ref.model.path=${model_path} \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=${rollout_name} \
  actor_rollout_ref.rollout.mode=${rollout_mode} \
  actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
  actor_rollout_ref.rollout.max_model_len=3072 \
  actor_rollout_ref.rollout.response_length=${max_response_length} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${experiment_name}" \
  trainer.test_freq="${test_freq}" \
  trainer.save_freq="${save_freq}" \
  trainer.logger='[console,swanlab]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${total_training_steps}" \
  rollout.nnodes=1 \
  rollout.n_gpus_per_node=1 \
  rollout.total_rollout_steps="${total_rollout_steps}" \
  async_training.require_batches=${require_batches} \
  +exchange.side=B \
  +exchange.mode=train_first \
  +exchange.backend=tcp \
  +exchange.run_id="${EXCHANGE_RUN_ID}" \
  +exchange.host="${EXCHANGE_HOST}" \
  +exchange.port="${EXCHANGE_PORT}" \
  "+exchange.enable_gate=true"

