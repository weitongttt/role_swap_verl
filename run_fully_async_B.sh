#!/bin/bash
set -euo pipefail

set -x

# Side B: 启动时先从 train 开始，吃到 A 的 rollout 后再启动自己的 rollout
export VERL_USE_MODELSCOPE=True
export HYDRA_CONFIG_PATH="$(pwd)/verl/verl/trainer/config"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export RAY_ADDRESS=${RAY_ADDRESS:-127.0.0.1:6379}
export EXCHANGE_HOST=${EXCHANGE_HOST:-127.0.0.1}
export EXCHANGE_PORT=${EXCHANGE_PORT:-18080}
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"HPA4rMyhiXXBFNbyKiW4A"}
# vLLM 这里实际走的是 v1 AsyncLLMEngine；环境变量不一致会直接报错。
export VLLM_USE_V1=${VLLM_USE_V1:-1}
# Ray 默认会对重复日志做去重，容易看起来“卡住没输出”
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
# 防止 Ray 因 Linux 文件缓存 (buff/cache) 误判 OOM 而杀进程
export RAY_memory_usage_threshold=0.99


adv_estimator="grpo"
train_files="data/gsm8k/train.parquet"
val_files="data/gsm8k/test.parquet"
model_path="${model_path:-$(pwd)/Qwen3-8B}"
project_name="gap_grpo_qwen3_8b_gsm8k"
experiment_name="4xa800_siteB_0518"

# 确保找得到 ray（B 不启动 head，但会依赖 ray.init 连接集群）
RAY_BIN="${RAY_BIN:-}"
if [ -z "$RAY_BIN" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/ray" ]; then
  RAY_BIN="${CONDA_PREFIX}/bin/ray"
fi
if [ -z "$RAY_BIN" ] && command -v ray &>/dev/null; then
  RAY_BIN="$(command -v ray)"
fi
RAY_BIN="${RAY_BIN:-ray}"

# B 独立机器，GPU 从 0 开始
CUDA_VISIBLE_DEVICES_B="${CUDA_VISIBLE_DEVICES_B:-0,1,2,3}"
RAY_TEMP_DIR_B="${RAY_TEMP_DIR_B:-/tmp/ray_b}"
mkdir -p "$RAY_TEMP_DIR_B"

# 独立机器：先清理旧集群，再启动新的
"$RAY_BIN" stop --force 2>/dev/null || true
sleep 1
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_B" "$RAY_BIN" start --head --port=6379 --num-gpus=4 --temp-dir "$RAY_TEMP_DIR_B" --include-dashboard=false
sleep 3

rollout_mode="async"
rollout_name="vllm"

train_prompt_bsz=0
gen_prompt_bsz=1
max_model_len=4096
max_response_length=4096
max_num_batched_tokens=$((max_response_length * 4))
n_resp_per_prompt=4
use_dynamic_bsz=true
total_rollout_steps=$((400*1*160))
mini_batch_size=320 # GAP-GRPO: A+B 汇总 (64+64)
require_batches=1
test_freq=1000

staleness_threshold=3
trigger_parameter_sync_step=1
partial_rollout=false


# A/B 共用同一个 exchange.run_id（两边必须一致）
EXCHANGE_RUN_ID="${EXCHANGE_RUN_ID:-gapgrpo_run_001}"

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
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
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
    trainer.n_gpus_per_node=2 \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=2 \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    async_training.require_batches=${require_batches} \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.partial_rollout="${partial_rollout}" \
    +exchange.side=B \
    +exchange.mode=train_first \
    +exchange.run_id="${EXCHANGE_RUN_ID}" \
    +exchange.backend=tcp \
    +exchange.host="${EXCHANGE_HOST}" \
    +exchange.port="${EXCHANGE_PORT}" \
    +exchange.enable_group_merge=true \
    +exchange.expected_per_hash=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.actor.fsdp_config.model_dtype="bfloat16" \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size}

