# GAP-GRPO: Cross-Cluster Sample Exchange for Scalable GRPO Training

> **G**roup **A**dvantage **P**PO via cross-cluster exchange — an extension of the veRL fully-async training framework.

---

## Motivation

Most researchers don't have a single datacenter packed with GPUs. In practice, compute is **scattered**: 2 GPUs in one city, another 2 at a different site, a rented node somewhere else. Traditional distributed training frameworks (DeepSpeed, FSDP, multi-node Ray) demand fast, reliable interconnects — they fall apart over high-latency WAN links between cities.

GAP-GRPO takes a different approach: instead of forcing all GPUs into one tightly-coupled cluster, it lets **geographically separated GPU groups train independently** and coordinates them through a lightweight TCP sample exchange. Each site runs its own Ray cluster, generates rollouts locally, and shares only the resulting samples over the network. The only cross-site traffic is serialized rollout data — no gradient allreduce, no parameter sharding across WAN.

**The result**: you combine the compute power of distributed GPU pools into a single, coherent GRPO training run — and as a bonus, the merged samples from multiple sites **double the effective GRPO group size**, improving advantage estimation and convergence.

---

## What GAP-GRPO Does

The system splits training across **two independent compute clusters** (Side A and Side B). Each cluster runs its own rollouter and trainer. A central TCP exchange server merges their rollout samples by prompt, so each trainer sees responses from **both** clusters for the same prompt.

### Concrete Example

| Setting | Per-cluster generation | Effective GRPO group size |
|---|---|---|
| **Baseline** (single cluster, 4 GPUs) | 4 responses/prompt | 4 |
| **GAP-GRPO** (2 GPUs site A + 2 GPUs site B) | 4 responses/prompt each | **8** (4 from A + 4 from B) |

Each site bears only the generation cost of 4 responses, but the trainer at each site computes advantages over a group of 8 — more diverse exploration and steeper convergence.

---

## Architecture Overview

```
   Site A (e.g. Beijing)                                 Site B (e.g. Shanghai)
┌──────────────────┐                                    ┌──────────────────┐
│  Independent Ray  │                                    │  Independent Ray  │
│     Cluster       │                                    │     Cluster       │
│                   │                                    │                   │
│  ┌─────────────┐  │       push(prompt_hash, sample)    │  ┌─────────────┐  │
│  │  Rollouter  │──┼────────────────┐ ┌────────────────┼──│  Rollouter  │  │
│  │  (vLLM)     │  │                │ │                │  │  (vLLM)     │  │
│  └─────────────┘  │                ▼ ▼                │  └─────────────┘  │
│                   │         ┌──────────────┐          │                   │
│                   │         │  TCP Exchange │          │                   │
│                   │         │    Server     │          │                   │
│                   │         │              │          │                   │
│                   │         │  Groups by    │          │                   │
│                   │         │  prompt_hash  │          │                   │
│                   │         └──────┬───────┘          │                   │
│                   │                │ │                │                   │
│  ┌─────────────┐  │   pull(grouped)│ │pull(grouped)   │  ┌─────────────┐  │
│  │  Trainer    │◀─┼────────────────┘ └────────────────┼─▶│  Trainer    │  │
│  │  (FSDP)    │  │                                    │  │  (FSDP)    │  │
│  └─────────────┘  │                                    │  └─────────────┘  │
└──────────────────┘                                    └──────────────────┘
```

The **only** cross-site traffic is serialized rollout samples over TCP. No gradient synchronization, no parameter sharding, no NCCL across WAN.

### Three Processes, Three Terminals

| Terminal | Script | Purpose |
|---|---|---|
| 1 | `run_exchange_server.sh` | Central TCP server that groups samples by `prompt_hash` |
| 2 | `run_fully_async_A.sh` | Site A — runs rollouter + trainer (`exchange.mode=both`) |
| 3 | `run_fully_async_B.sh` | Site B — runs trainer first, then rollouter (`exchange.mode=train_first`) |

---

## How It Works — Step by Step

### 1. Deterministic Prompt Sampling

Both sites use **identical data seeds** (`data.seed=99`) and the same dataset. This guarantees that at any given training step, Site A and Site B generate responses for **the exact same prompt**. Without this, samples from different prompts would never match and grouping would fail.

### 2. Prompt-Hash-Based Grouping

Each rollout sample is tagged with a `prompt_hash` (a hash of the prompt text). When a rollouter pushes a sample to the TCP exchange server:

1. The server stores the sample in **pending dictionaries** (one per trainer side).
2. Once a `prompt_hash` accumulates samples from **both sites** (i.e., 2 payloads), the server moves the group to a **ready queue**.
3. Each trainer pulls completed groups — always receiving samples from both sites for the same prompt.

This is implemented in [`tcp_exchange.py`](verl/verl/experimental/fully_async_policy/tcp_exchange.py):
- `push_grouped` — rollouter sends `(prompt_hash, pickled_sample)` to the server
- `pull_grouped` — trainer blocks until a complete group is ready, then receives all samples for that prompt

### 3. Physical Cluster Isolation

Each site runs a completely **independent Ray cluster** (separate head node, separate temp directory). This is the key design choice: instead of fighting multi-node Ray over unreliable WAN, each site operates autonomously. The TCP exchange is the **only** cross-site link — and it only carries rollout samples, which are tiny compared to gradients or model weights.

### 4. Startup Ordering

The `exchange.mode` controls the order in which the rollouter and trainer Ray tasks are submitted:

- **Site A** (`mode=both`): submits the rollouter `.fit()` first, then the trainer `.fit()`. Both run concurrently as async Ray tasks.
- **Site B** (`mode=train_first`): submits the trainer `.fit()` first, then the rollouter `.fit()`. Again, both run concurrently.

In practice, both sides start generating and training concurrently. The trainer on each side blocks inside `pull_grouped_sync()` until the TCP exchange server has a complete group (samples from both sides for the same prompt), so there is a natural synchronization point.


### 5. Synchronized Training Cadence

Both sites use:
- `staleness_threshold=3` — limits how stale rollout samples can be relative to the current policy.
- `trigger_parameter_sync_step=1` — syncs new weights to the rollouter after every training step.
- `mini_batch_size=320` — each training step consumes 320 samples (160 from self + 160 from peer).

---

## Key Configuration Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `+exchange.side` | `A` or `B` | Which site this instance represents |
| `+exchange.mode` | `both` / `train_first` | Startup behavior (see above) |
| `+exchange.backend` | `tcp` | Uses TCP exchange server (vs. Ray actor) |
| `+exchange.host` / `port` | `127.0.0.1:18080` | TCP exchange server address |
| `+exchange.run_id` | `gapgrpo_run_001` | Must be **identical** on both sites |
| `+exchange.enable_group_merge` | `true` | Enables prompt-hash grouping |
| `+exchange.expected_per_hash` | `2` | Server waits for 2 samples per hash (A + B) |
| `n_resp_per_prompt` | `4` | Each site generates 4 responses per prompt |
| `ppo_mini_batch_size` | `320` | Training batch = 160 (self) + 160 (peer) |
| `data.seed` | `99` | Deterministic sampling — must match on both sites |

---

## Quick Start

### Prerequisites

- Two GPU sites (or one machine with ≥ 4 GPUs split into two logical clusters for testing)
- A model checkpoint (e.g., `Qwen3-1.7B`) in the project root
- Training data at `data/gsm8k/train.parquet` and `data/gsm8k/test.parquet`
- Python environment with `verl`, `ray`, `vllm`, and dependencies installed

### Single Machine (Local Testing with 4 GPUs, 2+2 Split)

```bash
# Terminal 1: Start the TCP exchange server
bash run_exchange_server.sh

# Terminal 2: Start Site A (uses GPU 0,1)
bash run_fully_async_A.sh

# Terminal 3: Start Site B (uses GPU 2,3)
bash run_fully_async_B.sh
```

### Two Separate Machines (Cross-City Deployment)

```bash
# Machine A (e.g. Beijing) — runs exchange server + Site A:
bash run_exchange_server.sh                          # Terminal 1
bash run_fully_async_A.sh                            # Terminal 2

# Machine B (e.g. Shanghai) — runs Site B, connects to Machine A:
EXCHANGE_HOST=<machine_A_public_ip> \
EXCHANGE_RUN_ID=gapgrpo_run_001 \
bash run_fully_async_B.sh                            # Terminal 3
```

> **Note**: Ensure the exchange server port (default `18080`) is reachable from Machine B. The bandwidth requirement is modest — only serialized rollout samples are transferred, not gradients or model weights.

### Running the Baseline (No Exchange)

To compare GAP-GRPO against a standard single-cluster GRPO setup:

```bash
bash run_fully_async.sh
```

This runs a standard fully-async trainer with `group_size=4` on all GPUs, no cross-site exchange. Use SwanLab to compare convergence curves between baseline and GAP-GRPO.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Site B hangs at startup | Exchange server not running, or wrong `EXCHANGE_HOST` | Start `run_exchange_server.sh` first; verify host/port connectivity |
| "Pending backlog" warnings on server | One site is producing much faster than the other | Ensure both sites use identical `data.seed` and `gen_prompt_bsz` |
| vLLM CUDA graph warmup takes minutes | Normal for large models | Wait 3-5 minutes; ensure `model_dtype=bfloat16` is set |
| Ray OOM kills | Linux buffer cache counted as used memory | Already mitigated with `RAY_memory_usage_threshold=0.99` |
| One site stops training | Deadlock in exchange queue | Check that `EXCHANGE_RUN_ID` matches on both sites |
| Log output appears frozen | Ray deduplicates logs by default | Already mitigated with `RAY_DEDUP_LOGS=0` |

---

## Project Structure

```
.
├── run_exchange_server.sh              # Launches the TCP exchange server
├── run_fully_async_A.sh                # Site A launch script
├── run_fully_async_B.sh                # Site B launch script
├── run_fully_async.sh                  # Baseline (single-cluster, no exchange)
├── data/gsm8k/                         # Training and test data (parquet)
└── verl/verl/experimental/fully_async_policy/
    ├── fully_async_exchange_main.py     # Exchange-enabled entry point
    ├── fully_async_main.py             # Original fully-async entry point (baseline)
    ├── fully_async_rollouter.py        # Async rollout generation (vLLM)
    ├── fully_async_trainer.py          # Async trainer (FSDP)
    ├── tcp_exchange.py                 # TCP server + client for hash-grouped exchange
    ├── tcp_exchange_server_main.py      # Server entry point
    ├── message_queue.py                # In-process / Ray-based message queue
    └── detach_utils.py                 # Utilities for detached actor management
```
