# CCRN: Cross-Cluster Reward Normalization for Distributed GRPO

## 1. 问题背景

在跨数据中心（弱网）环境下进行 LLM Post-training 时，现有方案存在一个根本性矛盾：

- **Rollout 阶段**可以通过多集群并行加速（每端只需生成一半）
- **Update 阶段**因为要在合并后的 2N 数据上训练，计算量翻倍

设单端 Rollout 时间为 $T_{gen}$，Update 时间为 $T_{up}$：

$$T_{current} = 0.5 \times T_{gen} + T_{comm} + 2 \times T_{up} \approx T_{gen} + T_{up} = T_{baseline}$$

**结论**：没有任何加速效果。

## 2. CCRN 核心思路

**核心观察**：GRPO 的关键不在于每条数据都参与 backprop，而在于 advantage 的归一化统计量（$\mu$, $\sigma$）是否来自足够大的 group。

**设计原则**：全局归一化，本地训练。

### 2.1 数据流

```
集群 A (GPU 0)                           集群 B (GPU 1)
┌─────────────────────┐                ┌─────────────────────┐
│ 1. 生成 G 条回复     │                │ 1. 生成 G 条回复     │
│    {y1..yG}          │                │    {y(G+1)..y(2G)}   │
│                      │                │                      │
│ 2. 计算 reward       │                │ 2. 计算 reward       │
│    [r1..rG]          │                │    [r(G+1)..r(2G)]   │
└───────┬──────────────┘                └───────┬──────────────┘
        │                                       │
        │◄─── 只交换 reward 标量（~32 字节）────►│
        │                                       │
┌───────▼──────────────┐                ┌───────▼──────────────┐
│ 3. 全局 advantage     │                │ 3. 全局 advantage     │
│    μ = mean(r1..r2G)  │                │    μ = mean(r1..r2G)  │
│    σ = std(r1..r2G)   │                │    σ = std(r1..r2G)   │
│    A_i = (r_i-μ)/σ    │                │    A_i = (r_i-μ)/σ    │
│                       │                │                       │
│ 4. 只训练本地样本     │                │ 4. 只训练本地样本     │
│    loss on {y1..yG}   │                │    loss on{y(G+1)..G} │
│    with A_1..A_G      │                │    with A_(G+1)..A_2G │
└───────────────────────┘                └───────────────────────┘
```

### 2.2 时间分析

$$T_{CCRN} = 0.5 \times T_{gen} + T_{comm}^{reward} + T_{up}$$

其中 $T_{comm}^{reward}$ 可忽略（仅传输几个浮点数），因此：

$$T_{CCRN} \approx 0.5 \times T_{gen} + T_{up}$$

**加速比**：

$$\text{Speedup} = \frac{T_{gen} + T_{up}}{0.5 \times T_{gen} + T_{up}}$$

当 $T_{gen} \gg T_{up}$（推理密集型任务，如数学推理）：

- $T_{gen} = 4 \times T_{up}$ 时，加速比 ≈ $\frac{5}{3} = 1.67\times$
- $T_{gen} = 10 \times T_{up}$ 时，加速比 ≈ $\frac{11}{6} = 1.83\times$

## 3. 算法正确性论证

### 3.1 与标准 GRPO 的等价性

标准 GRPO 的梯度：

$$\nabla J = \sum_{i=1}^{2G} \hat{A}_i \cdot \nabla \log \pi(y_i | x)$$

CCRN 中，集群 A 计算的梯度：

$$\nabla J_A = \sum_{i=1}^{G} \hat{A}_i \cdot \nabla \log \pi_A(y_i | x)$$

其中 $\hat{A}_i$ 使用的是**全局** $\mu$, $\sigma$，与标准 GRPO 完全一致。

这等价于**数据并行 SGD 的 mini-batch 分片**：每个 worker 处理全局 batch 的一个子集，gradient 方向是全局 gradient 的无偏估计。

### 3.2 模型不会发散的保证

1. **全局 advantage 归一化** → 两端对"好/坏回复"的判断完全一致
2. **相同 prompt 序列** → 同步 dataloader seed，两端看到相同输入
3. **KL 正则化** → 两端都被约束在参考策略 $\pi_{ref}$ 附近
4. **On-policy 生成** → 每步都用最新策略重新生成，自然纠正漂移

### 3.3 与联邦学习的区别

CCRN **不是**联邦学习。区别在于：

| 特性 | 联邦学习 | CCRN |
|------|---------|------|
| 数据 | 各端私有，不可见 | 同一数据集，同一 prompt 序列 |
| Loss 函数 | 各端独立 | **全局统一**（共享 advantage） |
| 模型同步 | 周期性权重平均 | **每步**都通过全局归一化对齐 |
| 发散风险 | 高（client drift） | 低（每步方向一致） |

## 4. 与现有工作的差异

| 工作 | 方式 | CCRN 的区别 |
|-----|------|------------|
| Async RLHF (ICLR 2025) | 解耦生成/训练，off-policy | CCRN 是 on-policy，跨集群 |
| DistFlow / AReaL | 单集群内多 controller | CCRN 面向弱网跨集群 |
| FastGRPO | 投机解码加速 rollout | 正交，可组合 |
| Laminar | relay worker 分发权重 | 单集群内优化，不跨集群 |
| FedAvg | 周期性权重平均 | CCRN 每步共享 loss 信号 |

**CCRN 的独特贡献**：首次提出跨弱网集群的 GRPO 训练方案，通过 reward 层面的轻量通信替代数据层面的重量通信，实现加速而不牺牲收敛性。

## 5. 实现要点

### 5.1 需要修改的模块

#### (a) 同步 prompt 采样
两端使用相同 seed + 确定性 sampler，确保第 $t$ 步看到相同 prompt。

#### (b) Reward 交换协议
扩展现有 TCP exchange channel：新增 `exchange_rewards` op，只传 reward 标量数组。

```python
# 伪代码
local_rewards = compute_rewards(local_rollouts)       # [G] floats
peer_rewards = tcp_exchange.exchange_rewards(local_rewards)  # [G] floats
global_rewards = concat(local_rewards, peer_rewards)   # [2G] floats

# 全局归一化
mu = mean(global_rewards)
sigma = std(global_rewards)
local_advantages = (local_rewards - mu) / sigma        # 只保留本地 advantage

# 训练只用本地数据 + 全局 advantage
train(local_rollouts, local_advantages)
```

#### (c) 修改 GRPO advantage 计算
修改 `compute_grpo_outcome_advantage`，接受外部传入的 `global_mean` 和 `global_std`。

### 5.2 通信开销分析

| 数据 | 大小 | 频率 |
|------|------|------|
| Reward 标量 | G × 4 bytes = 16~128 bytes | 每个 prompt |
| Prompt 同步 | 0（seed 确定） | 0 |
| 权重同步 | 0（各端本地 update） | 0 |
| **总计** | **< 1 KB/step** | — |

对比当前方案的 rollout 数据交换（每条回复 ~几十 KB），通信量降低 **1000×** 以上。

## 6. 扩展讨论

### 6.1 可组合的优化

CCRN 与以下技术正交，可叠加使用：

- **FastGRPO**（投机解码）：加速每个集群本地的 rollout 速度
- **异构采样温度**：A 用 τ=0.6（exploitation），B 用 τ=1.2（exploration），group 多样性更高
- **LoRA 联邦周期性同步**：在 CCRN 基础上，每 K 步额外做一次 LoRA delta 交换，进一步对齐权重

### 6.2 扩展到 N 个集群

CCRN 自然推广到 N 个集群：每个集群生成 G/N 条回复，交换 reward 后在全局 G 上归一化。通信量随 N 线性增长，但仍是标量级（与 N×G 个浮点数成正比），可忽略。

### 6.3 论文建议标题

> **Cross-Cluster Reward Normalization: Communication-Efficient Distributed GRPO via Global Advantage with Local Policy Updates**


```shell
source /zhangshihao/weitong/anaconda3/etc/profile.d/conda.sh
conda activate yc
cd /zhangshihao/weitong/verl_dev/

# 终端 1: TCP exchange server
python -m verl.experimental.fully_async_policy.tcp_exchange_server_main
# 终端 2: Side A (GPU 0,1)
bash run_ccrn_A.sh 2>&1 | tee ccrna_0402.log
# 终端 3: Side B (GPU 2,3)
bash run_ccrn_B.sh 2>&1 | tee ccrnb_0402.log

```