#!/bin/bash
set -euo pipefail
set -x

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 本仓库源码在 ${_REPO_ROOT}/verl/verl/；必须加入 PYTHONPATH，否则 -m verl.experimental... 会落到环境里旧版/无该文件的 verl 包
export PYTHONPATH="${_REPO_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"

export VERL_USE_MODELSCOPE=True
export HYDRA_CONFIG_PATH="${_REPO_ROOT}/verl/verl/trainer/config"
# A 集群仅使用第 1 张物理 GPU（与 B 硬隔离）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export RAY_PORT_A=${RAY_PORT_A:-26379}
# 客户端连的地址：必须与「正在跑 run_exchange_server 的那台机器」对训练 Pod 可路由的 IP 一致（同机可 127.0.0.1）。
# 多 Pod 时禁止使用各 Pod 自带的 127.0.0.1，否则 A、B 各连一个空 exchange，表现为 pull 永远阻塞、另一侧队列疯长。
export EXCHANGE_HOST="${EXCHANGE_HOST:-127.0.0.1}"
export EXCHANGE_PORT=${EXCHANGE_PORT:-28080}
# 默认关闭：减少 MQ/TCP 路径上的调试输出（需要排查 gate 时再 export VERL_EXCHANGE_DEBUG=1）
export VERL_EXCHANGE_DEBUG="${VERL_EXCHANGE_DEBUG:-0}"
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VERL_SYNC_DEBUG=1
# 设备错配调试：默认开启，确保 optimizer/device mismatch 时能打印出具体 state/grad 的 device
export VERL_OPT_DEVICE_DEBUG="${VERL_OPT_DEVICE_DEBUG:-1}"
# 参数同步前让 vLLM 真正执行 sleep（绕过 free_cache_engine=false 短路），减轻 update_weights_from_ipc OOM
# export VERL_FORCE_VLLM_SLEEP="${VERL_FORCE_VLLM_SLEEP:-1}"
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"hlo16D6KKxblfDAgvGxVQ"}
# SwanLab：可通过 PROJECT_NAME / EXPERIMENT_NAME 覆盖
PROJECT_NAME="${PROJECT_NAME:-new_role_swap_2026.4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-isolated_A_v1}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
TRAIN_FILES="${TRAIN_FILES:-data/gsm8k/train.parquet}"
VAL_FILES="${VAL_FILES:-data/gsm8k/train.parquet}"
TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-0}"
GEN_PROMPT_BSZ="${GEN_PROMPT_BSZ:-1}"
# 与 run_sync_1gpu_test.sh 对齐 vLLM/KV 占用：max_prompt_length=512、max_response_length=512；长上下文会显著增大 vLLM 显存
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-4}"
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-64000}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-160}"
REQUIRE_BATCHES="${REQUIRE_BATCHES:-1}"
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
ROLLOUT_MODE="${ROLLOUT_MODE:-async}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
# sync 侧等价上下文约 512+512；默认给 1024 余量
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((MAX_RESPONSE_LENGTH * 4))}"
# 与 run_sync_1gpu_test.sh 的 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 一致
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.35}"
STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-100}"
TRIGGER_PARAM_SYNC_STEP="${TRIGGER_PARAM_SYNC_STEP:-1}"
PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT:-false}"
TEST_FREQ="${TEST_FREQ:-1000}"
# 单集群 1 槽位：trainer 与 vLLM hybrid 共置（见 isolated main 传入 shared_actor_rollout_wg）
NUM_RAY_GPUS="${NUM_RAY_GPUS:-1}"
# FullyAsyncAgentLoopWorker 个数：每个 Ray actor 可向 vLLM 发请求；>1 才有「同时多路」推理（单 actor 内任务默认串行）
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-6}"

EXCHANGE_RUN_ID_FILE="${EXCHANGE_RUN_ID_FILE:-/tmp/verl_exchange_run_id}"
# 每次启动都生成新的 run_id，避免 tcp exchange server 复用旧 run 状态（包括历史的 None 终止哨兵）
# 导致下一轮直接在 sample 队列中提前结束。
date +%s%N > "$EXCHANGE_RUN_ID_FILE"
EXCHANGE_RUN_ID="$(cat "$EXCHANGE_RUN_ID_FILE" | tr -d '\r\n')"
# B 侧晚启动时可能拿不到同一 shell 的 env，写入侧车文件保证 ppo_mini_batch_size / rollout 步数一致
printf "%s\n" "$MINI_BATCH_SIZE" > "${EXCHANGE_RUN_ID_FILE}.ppo_mini_batch_size"
printf "%s\n" "$TOTAL_ROLLOUT_STEPS" > "${EXCHANGE_RUN_ID_FILE}.total_rollout_steps"

# Let ray.init() start an in-process local cluster; avoid external ray start session conflicts.
unset RAY_ADDRESS

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
if [ -z "$PYTHON_BIN" ] && [ -x "/zhangshihao/weitong/anaconda3/envs/verl/bin/python" ]; then
  PYTHON_BIN="/zhangshihao/weitong/anaconda3/envs/verl/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

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
  +exchange.side=A \
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
