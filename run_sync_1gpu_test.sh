#!/bin/bash
set -euo pipefail
set -x

export VERL_USE_MODELSCOPE=True
export PYTHONUNBUFFERED=1

# 只用一张卡
export CUDA_VISIBLE_DEVICES=0

export VLLM_USE_V1=${VLLM_USE_V1:-1}

# swanlab key
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"hlo16D6KKxblfDAgvGxVQ"}

# 用 conda env 的 python，避免系统环境缺依赖（ray/vllm 等）
PYTHON_BIN="${PYTHON_BIN:-/zhangshihao/weitong/anaconda3/envs/verl/bin/python}"

model_path="Qwen/Qwen2.5-0.5B-Instruct"
train_files="data/gsm8k/train.parquet"
val_files="data/gsm8k/test.parquet"

# 按 verl 官方 quickstart 严格对齐 PPO 参数（更可能收敛）
#
# 如果你显存不够报 OOM，把这些 batch 相关参数调小（并告诉我你显存大小/报错）。
train_batch_size=256
max_prompt_length=512
max_response_length=512
ppo_mini_batch_size=64
ppo_micro_batch_size_per_gpu=4
rollout_log_prob_micro_batch_size_per_gpu=8
ref_log_prob_micro_batch_size_per_gpu=4
critic_ppo_micro_batch_size_per_gpu=4
# vLLM 对齐基准（与 isolated A/B 默认一致）：gpu_memory_utilization=0.4，max_prompt/response=512
gpu_memory_utilization=0.4

save_freq=10
test_freq=10
total_epochs=15

# 默认训练步数（你可以通过环境变量 TOTAL_TRAINING_STEPS 覆盖）
TOTAL_TRAINING_STEPS_DEFAULT=200
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-$TOTAL_TRAINING_STEPS_DEFAULT}"

# SwanLab：可通过环境变量 PROJECT_NAME / EXPERIMENT_NAME 覆盖默认值
project_name="${PROJECT_NAME:-new_role_swap_2026.4}"
experiment_name="${EXPERIMENT_NAME:-sync_ppo_1gpu_qwen05b}"

# 可选：如果你想快速 smoke test，设置 TOTAL_TRAINING_STEPS=2 之类的值
extra_args=()
if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
  extra_args+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
else
  extra_args+=(trainer.total_epochs="${total_epochs}")
fi

$PYTHON_BIN -m verl.trainer.main_ppo \
  data.train_files="${train_files}" \
  data.val_files="${val_files}" \
  data.train_batch_size="${train_batch_size}" \
  data.max_prompt_length="${max_prompt_length}" \
  data.max_response_length="${max_response_length}" \
  actor_rollout_ref.model.path="${model_path}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ppo_micro_batch_size_per_gpu}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${rollout_log_prob_micro_batch_size_per_gpu}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_memory_utilization}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ref_log_prob_micro_batch_size_per_gpu}" \
  critic.optim.lr=1e-5 \
  critic.model.path="${model_path}" \
  critic.ppo_micro_batch_size_per_gpu="${critic_ppo_micro_batch_size_per_gpu}" \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${experiment_name}" \
  trainer.logger='[console,swanlab]' \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq="${save_freq}" \
  trainer.test_freq="${test_freq}" \
  "${extra_args[@]}"

