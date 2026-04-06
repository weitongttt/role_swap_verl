#!/bin/bash
set -euo pipefail
set -x

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${_REPO_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"

export VERL_USE_MODELSCOPE=True
export HYDRA_CONFIG_PATH="${_REPO_ROOT}/verl/verl/trainer/config"
# B 集群仅使用第 2 张物理 GPU（与 A 硬隔离）
export CUDA_VISIBLE_DEVICES=1
export RAY_PORT_B=26380
# 与 A 侧相同：指向同一台 exchange 服务（可路由 IP）；多 Pod 时不要用各 Pod 的 127.0.0.1
export EXCHANGE_HOST="127.0.0.1"
export EXCHANGE_PORT=28080
# 默认关闭：减少 MQ/TCP 路径上的调试输出（需要排查 gate 时再 export VERL_EXCHANGE_DEBUG=1）
export VERL_EXCHANGE_DEBUG="0"
export VLLM_USE_V1=1
# 设备错配调试：默认开启，确保 optimizer/device mismatch 时能打印出具体 state/grad 的 device
export VERL_OPT_DEVICE_DEBUG="1"
# # 参数同步前让 vLLM 真正执行 sleep（绕过 free_cache_engine=false 短路），减轻 update_weights_from_ipc OOM
# export VERL_FORCE_VLLM_SLEEP=1
export RAY_DEDUP_LOGS=0
export SWANLAB_API_KEY="hlo16D6KKxblfDAgvGxVQ"
# SwanLab
PROJECT_NAME="new_role_swap_2026.4"
EXPERIMENT_NAME="isolated_B_v2.1"

MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILES="data/gsm8k/train.parquet"
VAL_FILES="data/gsm8k/train.parquet"
TRAIN_PROMPT_BSZ="0"
MINI_BATCH_SIZE="160"
GEN_PROMPT_BSZ="${MINI_BATCH_SIZE}"
# 与 run_sync_1gpu_test.sh 对齐 vLLM/KV：512+512；长上下文会显著增大 vLLM 显存
MAX_PROMPT_LENGTH="512"
MAX_RESPONSE_LENGTH="512"
N_RESP_PER_PROMPT="4"
TOTAL_ROLLOUT_STEPS="200"
REQUIRE_BATCHES="1"
ROLLOUT_NAME="vllm"
ROLLOUT_MODE="async"
ADV_ESTIMATOR="grpo"
USE_DYNAMIC_BSZ="true"
MAX_MODEL_LEN="1024"
MAX_NUM_BATCHED_TOKENS="8192"
GPU_MEM_UTIL="0.35"
STALENESS_THRESHOLD="100"
TRIGGER_PARAM_SYNC_STEP="1"
PARTIAL_ROLLOUT="false"
TEST_FREQ="1000"
NUM_RAY_GPUS="1"
# 与 A 侧一致：多 AgentLoopWorker 并行打 vLLM（export AGENT_NUM_WORKERS=8 等可再拉高，OOM 则减小）
AGENT_NUM_WORKERS="6"

EXCHANGE_RUN_ID_FILE="/tmp/verl_exchange_run_id"
for i in $(seq 1 180); do
  if [ -f "$EXCHANGE_RUN_ID_FILE" ]; then
    break
  fi
  sleep 1
done
if [ ! -f "$EXCHANGE_RUN_ID_FILE" ]; then
  echo "[side_B] timeout waiting for exchange run id file: $EXCHANGE_RUN_ID_FILE"
  exit 1
fi
EXCHANGE_RUN_ID="$(cat "$EXCHANGE_RUN_ID_FILE")"
EXCHANGE_RUN_ID="$(echo "$EXCHANGE_RUN_ID" | tr -d '\r\n')"
if [ -f "${EXCHANGE_RUN_ID_FILE}.ppo_mini_batch_size" ]; then
  MINI_BATCH_SIZE="$(tr -d '\r\n' < "${EXCHANGE_RUN_ID_FILE}.ppo_mini_batch_size")"
fi
if [ -f "${EXCHANGE_RUN_ID_FILE}.total_rollout_steps" ]; then
  TOTAL_ROLLOUT_STEPS="$(tr -d '\r\n' < "${EXCHANGE_RUN_ID_FILE}.total_rollout_steps")"
fi

# Let ray.init() start an in-process local cluster; avoid external ray start session conflicts.
unset RAY_ADDRESS

PYTHON_BIN=""
if [ -z "$PYTHON_BIN" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
if [ -z "$PYTHON_BIN" ] && [ -x "/zhangshihao/weitong/anaconda3/envs/verl/bin/python" ]; then
  PYTHON_BIN="/zhangshihao/weitong/anaconda3/envs/verl/bin/python"
fi
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

#   "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FORCE_VLLM_SLEEP=\"${VERL_FORCE_VLLM_SLEEP}\"" \
PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m verl.experimental.fully_async_policy.fully_async_isolated_exchange_main \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
  data.gen_batch_size="${GEN_PROMPT_BSZ}" \
  data.return_raw_chat=True \
  data.shuffle=False \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1=\"${VLLM_USE_V1}\"" \
  +ray_kwargs.ray_init.address=local \
  +ray_kwargs.ray_init.num_gpus="${NUM_RAY_GPUS}" \
  +exchange.side=B \
  +exchange.mode=both \
  +exchange.run_id="${EXCHANGE_RUN_ID}" \
  +exchange.backend=tcp \
  +exchange.host="${EXCHANGE_HOST}" \
  +exchange.port="${EXCHANGE_PORT}" \
  +exchange.enable_gate=true \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  algorithm.adv_estimator="${ADV_ESTIMATOR}" \
  algorithm.rollout_correction.bypass_mode=True \
  actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
  actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}" \
  actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.fully_async_policy.agent_loop.agent_loop.FullyAsyncAgentLoopManager \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
  actor_rollout_ref.rollout.free_cache_engine=false \
  actor_rollout_ref.actor.use_rollout_log_probs=True \
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.model.use_remove_padding=True \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger='[console,swanlab]' \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  rollout.nnodes=1 \
  rollout.n_gpus_per_node=1 \
  rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}" \
  async_training.require_batches="${REQUIRE_BATCHES}" \
  async_training.colocate_actor_rollout=true \
  async_training.staleness_threshold="${STALENESS_THRESHOLD}" \
  async_training.trigger_parameter_sync_step="${TRIGGER_PARAM_SYNC_STEP}" \
  async_training.partial_rollout="${PARTIAL_ROLLOUT}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE}"
