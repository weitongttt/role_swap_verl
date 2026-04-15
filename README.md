# GAP-GRPO: 基于跨集群样本交换的异步大组距 PPO 训练
(原 Role Swap 演进版本)

本项目的核心创新在于实现 **GAP-GRPO (Group Advantage PPO via cross-cluster exchange)**。
在全异步（Fully-Async）训练架构下，我们将集群拆分为互相独立的 Side A 和 Side B，除了原有的采样/训练角色交替（Role Swap）用以避免硬件尾部等待外，更重要的是**通过 TCP 队列跨集群深度交换 Rollout 轨迹，实现了等效 GRPO Group Size 的翻倍**。

## 🌟 核心创新：跨集群 Group Size 无缝扩展

传统的 GRPO 需要在单个节点上完整前向生成 $N$ 个 Responses（例如 $N=8$）来计算组内 Advantage。在显存受限或算力瓶颈（特别是模型向 1.7B 或更高参数演进）时，单机很难同时负担大量的单 prompt 重组生成。

我们在全异步策略上引入了以下协同机制：

1. **基于 `prompt_hash` 的全局 Uid 匹配**
基于原生的 `verl` 实现，我们将 `fully_async_rollouter.py` 中基于自增 `sample_id` 的 `uid` 改作 `prompt_hash`。在优势估计（Advantage Computation）阶段，当 Cluster A 和 Cluster B 针对同一个 Prompt 各自产生 4 个 Responses 并被汇入队列时，Trainer 能够精准识别它们隶属同一 Prompt 并合并为一组。

2. **强制确定性全局采样 (`data.seed`)**
A 和 B 端的 `run_fully_async_A/B.sh` 使用了相同的 `data.seed=99`，保证独立运行的两个计算集群能按照绝对同步的节奏抽取相同的 Prompts 序列，使得跨集群聚合成为可能。

3. **智能 TCP 交换聚合合并**
在协议层面上，通过配置 `+exchange.enable_group_merge=true` 及 `+exchange.expected_per_hash=2`，TCP 服务端会自动合并两端收发的数据包。

**✅ 最终效果**：
两端各自的 Rollouter 引擎仅需负担极小的单批次生成压力 (`n_resp_per_prompt=4`)，从而最大化生成吞吐；但在最终送往 Trainer 的 Advantage 估计池中，实际作为更强大、探索性更高的 `group_size=8` 参与梯度计算，获得更陡峭的收敛曲线。

---

## 🏗 方法概览（底层通信逻辑）

1. **外部交换通道（A <-> B）**  
`fully_async_exchange_main.py` 在 `exchange.backend=tcp` 设定时，使用 `TcpExchangeClient` 把 rollouter 产生的样本送到对端服务器，trainer 同样从对端拉取共享样本。

2. **独立 Ray 侧 + 算力切分物理隔离**  
`run_fully_async_A.sh` 和 `run_fully_async_B.sh` 会由于硬编码隔离，独立启动并连接至不同网段或端口的 Ray head 节点，从根本上隔离了由于模型加载带来的单集群多机通信灾难。

3. **双模式切换防御启动死锁**  
由于强关联了 `expected_per_hash=2`，双侧的启动顺序使用了 `exchange.mode` 分流设计：
- `side=A, exchange.mode=both`：正常调度
- `side=B, exchange.mode=train_first`：强行挂起 Rollouter，先启动 Trainer 等待对侧队列投喂防止生成越界

---

## 🚀 运行方法（GAP-GRPO 完整演示）

### **1) 启动 TCP 样本交换中枢**

在 终端 1 中执行：
```bash
bash run_exchange_server.sh
```
*(交换中枢参数由 `EXCHANGE_HOST` 与 `EXCHANGE_PORT` 变量控制，默认使用 `127.0.0.1:18080`)*

### **2) 启动 Side A（rollout -> train）**

在 终端 2 中执行：
```bash
bash run_fully_async_A.sh
```
**Side A 关键配置**：
- `+exchange.side=A`
- `+exchange.mode=both`
- `+exchange.host=${EXCHANGE_HOST:-127.0.0.1}`
- `data.seed=99` (以保证抽取数据与 B 同步)

### **3) 启动 Side B（train_first）**

在 终端 3 中执行：
```bash
bash run_fully_async_B.sh
```
**Side B 关键配置**：
- `+exchange.side=B`
- `+exchange.mode=train_first`
- B 默认会自动扫描 A 侧抛出的 `exchange.run_id` 握手文件（`/tmp/verl_exchange_run_id`），保证安全对接同一会话。

### 📌 运行与排障提示：
*   **启动时效**：建议依次相隔较短时间分别拉起 1->2->3。
*   **关于卡死排查**：大模型（如 1.7B）在 vLLM 内部会进行数分钟极漫长的 Cudagraph Graph 铺排和占位，并且 Actor 在首次拉起时也容易陷入 fp32 退化而爆显存，一定要确保脚本中拥有 `model_dtype="bfloat16"` 和 `trainer.val_before_train=False` 的补全设定。

---

## 📊 Baseline 对比参考：纯单核组距

为了验证 **GAP-GRPO** 的收敛收益，我们利用 `run_fully_async.sh` 作为 Baseline。
- **命令行**：`bash run_fully_async.sh`
- **特征**：直接作为双卡体系单体运行，它不开启任何外部架构和通道拆分，同一套环境只拥有普通的 `GRPO group_size=4`。通过同时将这两者的数据监控上传至 `SwanLab` ，可以得到 `Group Size=8` 与 `Group Size=4` 同等时长与参数同步情况下的最严格、最直观的收敛速度对比评测。



```shell
source /zhangshihao/weitong/anaconda3/etc/profile.d/conda.sh
conda activate yc
cd /zhangshihao/weitong/verl_dev/

# 终端 1: TCP exchange server
python -m verl.experimental.fully_async_policy.tcp_exchange_server_main
# 终端 2: Side A (GPU 0,1)
bash run_fully_async_A.sh 2>&1 | tee fully_async_A_0412.log
# 终端 3: Side B (GPU 2,3)
bash run_fully_async_B.sh 2>&1 | tee fully_async_B_0412.log

```
