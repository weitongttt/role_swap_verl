# GAP-GRPO: Cross-Cluster Sample Exchange for Scalable GRPO Training

> **G**roup **A**dvantage **P**PO via cross-cluster exchange — an extension of the veRL fully-async training framework.

---

## Motivation

Most researchers don't have a single datacenter packed with GPUs. In practice, compute is **scattered**: 2 GPUs in one city, another 2 at a different site, a rented node somewhere else. Traditional distributed training frameworks (DeepSpeed, FSDP, multi-node Ray) demand fast, reliable interconnects — they fall apart over high-latency WAN links between cities.

GAP-GRPO takes a different approach: instead of forcing all GPUs into one tightly-coupled cluster, it lets **geographically separated GPU groups train independently** and coordinates them through a lightweight TCP sample exchange. Each site runs its own Ray cluster, generates rollouts locally, and shares only the resulting samples over the network. The only cross-site traffic is serialized rollout data — no gradient allreduce, no parameter sharding across WAN.

**The result**: you combine the compute power of distributed GPU pools into a single, coherent GRPO training run — and as a bonus, the merged samples from multiple sites **multiply the effective GRPO group size**, improving advantage estimation and convergence.

---

## What GAP-GRPO Does

The system splits training across **N independent compute clusters** (sites). Each cluster runs its own rollouter and trainer. A central TCP exchange server merges their rollout samples by prompt, so each trainer sees responses from **all** clusters for the same prompt.

### Concrete Example

| Setting | Per-site generation | Effective GRPO group size |
|---|---|---|
| **Baseline** (single cluster, 4 GPUs) | 4 responses/prompt | 4 |
| **GAP-GRPO 2-site** (2 GPUs × 2 sites) | 4 responses/prompt each | **8** (4+4) |
| **GAP-GRPO 3-site** (2 GPUs × 3 sites) | 4 resp/prompt each | **12** (4+4+4) |
| **GAP-GRPO 3-site heterogeneous** | 2+2+4 resp/prompt | **8** (2+2+4) |

Each site bears only its own generation cost, but trainers at every site compute advantages over the full merged group.

---

## Architecture Overview

```
   Site 0 (e.g. Beijing)          Site 1 (e.g. Shanghai)        Site N-1 (e.g. Tokyo)
┌──────────────────┐             ┌──────────────────┐          ┌──────────────────┐
│  Independent Ray  │             │  Independent Ray  │          │  Independent Ray  │
│     Cluster       │             │     Cluster       │          │     Cluster       │
│                   │             │                   │          │                   │
│  ┌─────────────┐  │    push     │  ┌─────────────┐  │   push   │  ┌─────────────┐  │
│  │  Rollouter  │──┼─────┐ ┌────┼──│  Rollouter  │──┼────┐ ┌──┼──│  Rollouter  │  │
│  │  (vLLM)     │  │     │ │    │  │  (vLLM)     │  │    │ │  │  │  (vLLM)     │  │
│  └─────────────┘  │     ▼ ▼    │  └─────────────┘  │    ▼ ▼  │  └─────────────┘  │
│                   │  ┌──────────────┐               │         │                   │
│                   │  │  TCP Exchange │               │         │                   │
│                   │  │    Server     │               │         │                   │
│                   │  │              │               │         │                   │
│                   │  │  Groups by    │               │         │                   │
│                   │  │  prompt_hash  │               │         │                   │
│                   │  │  (waits for   │               │         │                   │
│                   │  │   N samples)  │               │         │                   │
│                   │  └──────┬───────┘               │         │                   │
│                   │         │ │ │                    │         │                   │
│  ┌─────────────┐  │  pull   │ │ │   pull            │  pull   │  ┌─────────────┐  │
│  │  Trainer    │◀─┼─────────┘ │ └───────────────────┼─────┐  │  │  Trainer    │  │
│  │  (FSDP)    │  │           │                      │     └──┼─▶│  (FSDP)    │  │
│  └─────────────┘  │           │    ┌─────────────┐  │         │  └─────────────┘  │
└──────────────────┘           └───▶│  Trainer    │◀─┘         └──────────────────┘
                                    │  (FSDP)    │
                                    └─────────────┘
```

The **only** cross-site traffic is serialized rollout samples over TCP. No gradient synchronization, no parameter sharding, no NCCL across WAN.

### N+1 Processes

| Terminal | Script | Purpose |
|---|---|---|
| 1 | `run_exchange_server.sh` | Central TCP server that groups samples by `prompt_hash` |
| 2..N+1 | `run_site.sh` (with `SITE_INDEX=0..N-1`) | Each site runs rollouter + trainer |

---

## How It Works — Step by Step

### 1. Deterministic Prompt Sampling

All sites use **identical data seeds** (`data.seed=99`) and the same dataset. This guarantees that at any given training step, all sites generate responses for **the exact same prompt**. Without this, samples from different prompts would never match and grouping would fail.

### 2. Prompt-Hash-Based Grouping

Each rollout sample is tagged with a `prompt_hash` (a hash of the prompt text). When a rollouter pushes a sample to the TCP exchange server:

1. The server stores the sample in **pending dictionaries** (one per site).
2. Once a `prompt_hash` accumulates samples from **all N sites** (i.e., N payloads), the server moves the group to each site's **ready queue**.
3. Each trainer pulls completed groups — always receiving samples from all sites for the same prompt.

This is implemented in [`tcp_exchange.py`](verl/verl/experimental/fully_async_policy/tcp_exchange.py):
- `push_grouped` — rollouter sends `(prompt_hash, pickled_sample)` to the server
- `pull_grouped` — trainer blocks until a complete group is ready, then receives all samples for that prompt

### 3. Physical Cluster Isolation

Each site runs a completely **independent Ray cluster** (separate head node, separate temp directory). This is the key design choice: instead of fighting multi-node Ray over unreliable WAN, each site operates autonomously. The TCP exchange is the **only** cross-site link — and it only carries rollout samples, which are tiny compared to gradients or model weights.

### 4. Startup Ordering

The `exchange.site_index` controls the order in which the rollouter and trainer Ray tasks are submitted:

- **Site 0** (primary): submits the rollouter `.fit()` first, then the trainer `.fit()`. Both run concurrently as async Ray tasks.
- **Site 1..N-1** (secondary): submits the trainer `.fit()` first, then the rollouter `.fit()`. Again, both run concurrently.

In practice, all sites start generating and training concurrently. The trainer on each site blocks inside `pull_grouped_sync()` until the TCP exchange server has a complete group (samples from all sites for the same prompt), so there is a natural synchronization point.

### 5. Synchronized Training Cadence

All sites should use consistent:
- `staleness_threshold=3` — limits how stale rollout samples can be relative to the current policy.
- `trigger_parameter_sync_step=1` — syncs new weights to the rollouter after every training step.
- `mini_batch_size` — each training step consumes this many samples (sum of contributions from all sites).

### 6. Heterogeneous Sites

Each site can have its own `n_resp_per_prompt`. For example, a 3-site setup with sites generating 2, 2, and 4 responses per prompt respectively. The `expected_per_hash` (= `NUM_SITES`) stays at 3, but the effective GRPO group size becomes 2+2+4=8.

---

## Key Configuration Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `SITE_INDEX` | `0`, `1`, `2`, ... | Integer index for this site (0 = primary) |
| `NUM_SITES` | `2`, `3`, ... | Total number of participating sites |
| `+exchange.side` | Same as `SITE_INDEX` | Site identifier sent to exchange server |
| `+exchange.site_index` | `0`, `1`, ... | Controls startup ordering |
| `+exchange.mode` | `both` / `train_first` | Auto-set: `both` for index 0, `train_first` for others |
| `+exchange.backend` | `tcp` | TCP exchange server |
| `+exchange.host` / `port` | `127.0.0.1:18080` | TCP exchange server address |
| `+exchange.run_id` | `gapgrpo_run_001` | Must be **identical** across all sites |
| `+exchange.expected_per_hash` | Same as `NUM_SITES` | Server waits for N samples per hash |
| `N_RESP_PER_PROMPT` | `4` | Per-site: number of responses per prompt |
| `MINI_BATCH_SIZE` | `320` | Per-site: training mini-batch size |
| `data.seed` | `99` | Deterministic sampling — must match on all sites |

---

## Quick Start

### Prerequisites

- N GPU sites (or one machine with ≥ 2N GPUs split into N logical clusters for testing)
- A model checkpoint (e.g., `Qwen3-8B`) in the project root
- Training data at `data/gsm8k/train.parquet` and `data/gsm8k/test.parquet`
- Python environment with `verl`, `ray`, `vllm`, and dependencies installed

### 2-Site Setup (Backward Compatible)

```bash
# Terminal 1: Start the TCP exchange server
NUM_SITES=2 bash run_exchange_server.sh

# Terminal 2: Start Site 0 (primary)
SITE_INDEX=0 NUM_SITES=2 bash run_site.sh

# Terminal 3: Start Site 1
SITE_INDEX=1 NUM_SITES=2 bash run_site.sh
```

### 3-Site Setup

```bash
# Terminal 1: Exchange server
NUM_SITES=3 bash run_exchange_server.sh

# Terminal 2: Site 0 (primary, starts rollouter first)
SITE_INDEX=0 NUM_SITES=3 bash run_site.sh

# Terminal 3: Site 1
SITE_INDEX=1 NUM_SITES=3 bash run_site.sh

# Terminal 4: Site 2
SITE_INDEX=2 NUM_SITES=3 bash run_site.sh
```

### Cross-Machine Deployment

```bash
# Machine A (Beijing) — runs exchange server + Site 0:
NUM_SITES=3 bash run_exchange_server.sh                     # Terminal 1
SITE_INDEX=0 NUM_SITES=3 bash run_site.sh                   # Terminal 2

# Machine B (Shanghai) — runs Site 1:
SITE_INDEX=1 NUM_SITES=3 EXCHANGE_HOST=<machine_A_ip> bash run_site.sh

# Machine C (Tokyo) — runs Site 2:
SITE_INDEX=2 NUM_SITES=3 EXCHANGE_HOST=<machine_A_ip> bash run_site.sh
```

### Heterogeneous Sites (Different n_resp_per_prompt)

```bash
# Site 0: generates 2 responses per prompt
SITE_INDEX=0 NUM_SITES=3 N_RESP_PER_PROMPT=2 MINI_BATCH_SIZE=240 bash run_site.sh

# Site 1: generates 2 responses per prompt
SITE_INDEX=1 NUM_SITES=3 N_RESP_PER_PROMPT=2 MINI_BATCH_SIZE=240 bash run_site.sh

# Site 2: generates 4 responses per prompt (more powerful GPUs)
SITE_INDEX=2 NUM_SITES=3 N_RESP_PER_PROMPT=4 MINI_BATCH_SIZE=240 bash run_site.sh
```

> **Note**: Ensure the exchange server port (default `18080`) is reachable from all sites. The bandwidth requirement is modest — only serialized rollout samples are transferred, not gradients or model weights.

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
| A site hangs at startup | Exchange server not running, or wrong `EXCHANGE_HOST` | Start `run_exchange_server.sh` first; verify host/port connectivity |
| "Pending backlog" warnings on server | One site is producing much faster than others | Ensure all sites use identical `data.seed` and `gen_prompt_bsz` |
| vLLM CUDA graph warmup takes minutes | Normal for large models | Wait 3-5 minutes; ensure `model_dtype=bfloat16` is set |
| Ray OOM kills | Linux buffer cache counted as used memory | Already mitigated with `RAY_memory_usage_threshold=0.99` |
| One site stops training | Deadlock in exchange queue | Check that `EXCHANGE_RUN_ID` and `NUM_SITES` match across all sites |
| Log output appears frozen | Ray deduplicates logs by default | Already mitigated with `RAY_DEDUP_LOGS=0` |

---

## Project Structure

```
.
├── run_exchange_server.sh              # Launches the TCP exchange server
├── run_site.sh                         # Parameterized N-site launch script
├── run_fully_async_A.sh                # Legacy 2-site: Site A (still works)
├── run_fully_async_B.sh                # Legacy 2-site: Site B (still works)
├── run_fully_async.sh                  # Baseline (single-cluster, no exchange)
├── data/gsm8k/                         # Training and test data (parquet)
└── verl/verl/experimental/fully_async_policy/
    ├── fully_async_exchange_main.py     # Exchange-enabled entry point (N-site)
    ├── fully_async_main.py             # Original fully-async entry point (baseline)
    ├── fully_async_rollouter.py        # Async rollout generation (vLLM)
    ├── fully_async_trainer.py          # Async trainer (FSDP)
    ├── tcp_exchange.py                 # TCP server + client for N-site hash-grouped exchange
    ├── tcp_exchange_server_main.py      # Server entry point
    ├── message_queue.py                # In-process / Ray-based message queue
    └── detach_utils.py                 # Utilities for detached actor management
```
