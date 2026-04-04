#!/bin/bash
set -euo pipefail
set -x
export VERL_USE_MODELSCOPE=True
export HYDRA_CONFIG_PATH="/zhangshihao/weitong/verl_swap/verl/verl/trainer/config"
export CUDA_VISIBLE_DEVICES=0,1
export SWANLAB_API_KEY="hlo16D6KKxblfDAgvGxVQ"
export VLLM_USE_V1=1
rollout_mode="async"
rollout_name="vllm" # sglang or vllm
return_raw_chat=True

adv_estimator="grpo"
train_files="data/gsm8k/train.parquet"
val_files="data/gsm8k/train.parquet"
model_path="Qwen/Qwen2.5-0.5B-Instruct"

train_prompt_bsz=0
gen_prompt_bsz=1
max_model_len=8192  # 模型最大上下文长度
max_response_length=4096
max_num_batched_tokens=$((max_response_length * 4))  # max_response_length 的 4-8 倍
n_resp_per_prompt=4
use_dynamic_bsz=true  # 动态batch size
total_rollout_steps=$(((400*1*160)))
mini_batch_size=160 # 最小推理批次
require_batches=1 # 一个step几个最小推理批次(一个step进行一次训练)
test_freq=1000
staleness_threshold=100
trigger_parameter_sync_step=1
partial_rollout=false # 中断生成

# 实验名
project_name="role_swap"
experiment_name="baseline_bs160_4gpus"


PYTHONUNBUFFERED=1 python -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files=${train_files} \
    data.val_files=${val_files} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    data.shuffle=False \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.model.path=${model_path} \
    algorithm.adv_estimator=${adv_estimator} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.test_freq="${test_freq}" \
    trainer.logger='[console,swanlab]' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=1 \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    async_training.require_batches=${require_batches} \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.partial_rollout="${partial_rollout}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size}