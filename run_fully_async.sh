#!/bin/bash
set -x
export VERL_USE_MODELSCOPE=True
export HYDRA_CONFIG_PATH="$(pwd)/verl/verl/trainer/config"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_ADDRESS="127.0.0.1:6381"
export SWANLAB_API_KEY="HPA4rMyhiXXBFNbyKiW4A"
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
# 防止 Ray 因 Linux 文件缓存 (buff/cache) 误判 OOM 而杀进程
export RAY_memory_usage_threshold=0.99
return_raw_chat="True"
rollout_mode="async"
rollout_name="vllm" # sglang or vllm

adv_estimator="grpo"
train_files="data/math/train.parquet"
val_files="data/math/test.parquet"
model_path="${model_path:-$(pwd)/Qwen3-8B}"
project_name="gapgrpo_qwen3_8b_MATH"
experiment_name="baseline_4xa800_g4"

# 独立启动 baseline 的 Ray 集群 (避开 A/B 的 6379 和 6380 端口)
RAY_TEMP_DIR_BASE="/tmp/ray_baseline"
mkdir -p "$RAY_TEMP_DIR_BASE"
# 先清理旧集群，防止残留的低 GPU 数集群导致资源死锁
/zhangshihao/weitong/anaconda3/envs/yc/bin/ray stop --force 2>/dev/null || true
sleep 1
/zhangshihao/weitong/anaconda3/envs/yc/bin/ray start --head --port=6381 --num-gpus=4 --num-cpus=60 --temp-dir "$RAY_TEMP_DIR_BASE" --include-dashboard=false
sleep 3

# 训练参数
train_prompt_bsz=0
gen_prompt_bsz=1
max_model_len=8192  # 模型最大上下文长度
max_response_length=4096
max_num_batched_tokens=$((max_response_length * 4))  # max_response_length 的 4-8 倍
n_resp_per_prompt=4
use_dynamic_bsz=true  # 动态batch size
total_rollout_steps=$(((400*1*160)))
mini_batch_size=160 # 为了做到绝对对齐实验，必须和 A 改成一模一样的 160
require_batches=1
test_freq=1000
staleness_threshold=3
trigger_parameter_sync_step=1 # 对齐 A 的同步频率
partial_rollout=false # 中断生成


PYTHONUNBUFFERED=1 /zhangshihao/weitong/anaconda3/envs/yc/bin/python -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files=${train_files} \
    data.val_files=${val_files} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.actor.fsdp_config.model_dtype="bfloat16" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size}