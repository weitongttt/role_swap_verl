## Role Swap：基于 TCP 交换通道的 Fully-Async A/B 训练

本仓库里的 `role_swap` 指的是一种“拆开成两个独立侧（A/B）+ 外部交换通道”的 fully-async 改造方式：  
两个侧各自运行 `FullyAsyncTrainer/FullyAsyncRollouter`，但样本通过 **TCP 双向交换队列** 在 A/B 之间交叉流动，从而让训练/采样的角色在异步过程中实现交替节奏（并避免单机协同结构里容易出现的资源空转/尾部等待）。

代码入口与交换实现分别在：
- `verl/verl/experimental/fully_async_policy/fully_async_exchange_main.py`：exchange-enabled 版 fully async
- `verl/verl/experimental/fully_async_policy/tcp_exchange.py`：TCP 双向队列交换通道

## 方法概览（role_swap 的关键点）

1. **外部交换通道（A<->B）**  
`fully_async_exchange_main.py` 在 `exchange.backend=tcp` 时，会用 `TcpExchangeClient` 把 rollouter 产生的样本送到对端；trainer 从对端拉取样本。

2. **两个侧独立运行 Ray + 独立 GPU 切片**  
`run_fully_async_A.sh` 与 `run_fully_async_B.sh` 会分别启动/连接不同端口的 Ray head，并通过 `CUDA_VISIBLE_DEVICES` 做硬隔离。

3. **A/B 启动顺序使用 `exchange.mode` 控制，避免死锁**  
`fully_async_exchange_main` 在 `_run_training_loop()` 中对不同 `exchange.mode` 做了不同的启动调度：
- `side=A, exchange.mode=both`：A 先 rollouter 后 trainer
- `side=B, exchange.mode=train_first`：B 先 trainer（等待第一批样本），随后启动 rollouter

4. **用较“紧”的参数同步节奏做同步化**（脚本内已固定）  
`run_fully_async_A.sh`/`run_fully_async_B.sh` 设置了：
- `async_training.trigger_parameter_sync_step=1`
- `async_training.staleness_threshold=100`
- `async_training.partial_rollout=false`

这会让参数同步更频繁，从而更容易观察到 A/B 侧在采样-训练之间的交替节奏。

## 运行方法（role_swap）

### 1) 启动 TCP 交换服务器

在一个终端执行：

```bash
bash run_exchange_server.sh
```

脚本使用的参数（可选覆盖）：
- `EXCHANGE_HOST`（默认 `127.0.0.1`）
- `EXCHANGE_PORT`（默认 `18080`）
- `EXCHANGE_MAX_QUEUE_SIZE`（默认 `20000`）

### 2) 启动 Side A（rollout -> train）

另开一个终端执行：

```bash
bash run_fully_async_A.sh
```

Side A 关键配置要点（来自脚本）：
- `+exchange.side=A`
- `+exchange.mode=both`
- `+exchange.backend=tcp`
- `+exchange.host=${EXCHANGE_HOST:-127.0.0.1}`
- `+exchange.port=${EXCHANGE_PORT:-18080}`
- `staleness_threshold=100`
- `trigger_parameter_sync_step=1`

### 3) 启动 Side B（train_first）

再开一个终端执行：

```bash
bash run_fully_async_B.sh
```

Side B 关键配置要点（来自脚本）：
- `+exchange.side=B`
- `+exchange.mode=train_first`
- 同样使用 `exchange.backend=tcp` 指向同一个交换服务器
- B 会等待 A 写入同一个 `exchange.run_id`（默认文件：`/tmp/verl_exchange_run_id`），避免走不同通道

### 运行提示
- 三个脚本最好在三个终端同时跑起来（server -> A -> B）。
- `run_fully_async_{A,B}.sh` 内部含有硬编码的 `HYDRA_CONFIG_PATH` 和默认模型/数据集路径；如果你不是在当前目录结构下运行，需要相应调整。

## Baseline 对比：run_fully_async.sh（单进程 Fully-Async）

`run_fully_async.sh` 作为 baseline，直接启动：
- `python -m verl.experimental.fully_async_policy.fully_async_main`

它不拆分成 A/B 两个侧，也不使用外部 TCP 交换通道；而是把 rollout/trainer 放在同一个 fully-async 体系内协同。

baseline 的关键参数差异（来自脚本）：
- `async_training.trigger_parameter_sync_step=5`（role_swap 里是 `1`）
- `async_training.staleness_threshold=100`
- `async_training.partial_rollout=false`
- GPU：`CUDA_VISIBLE_DEVICES=0,1`（单次启动）

运行 baseline：

```bash
bash run_fully_async.sh
```

## 一句话对比总结

- `role_swap`：两个侧（A/B）+ 外部 TCP 交换通道，让样本在 A/B 之间交叉流动，并通过 `exchange.mode` 选择启动顺序，形成更紧的参数同步节奏（更利于观察交替/重叠效率）。
- `run_fully_async` baseline：同一 fully-async 框架内协同 rollout/trainer，不拆分 A/B，也不走 TCP 交换通道。

