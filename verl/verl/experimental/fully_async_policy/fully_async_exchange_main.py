# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Exchange-enabled variant of fully_async_main.

- Keep original fully_async_main untouched.
- Reuse FullyAsyncTrainer/FullyAsyncRollouter as-is.
- Swap the MessageQueueClient with an adapter that routes samples through a
  bidirectional exchange channel (A<->B).
- Support bootstrap mode: side B can start as train-only to avoid deadlock.
"""

import asyncio
import os
import socket
import threading
import time
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Literal

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.message_queue import BidirectionalExchangeClient, BidirectionalExchangeQueue
from verl.experimental.fully_async_policy.tcp_exchange import TcpExchangeClient
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.utils import Role
from verl.utils.fs import copy_to_local

Side = Literal["A", "B"]
Mode = Literal["both", "train_only", "train_first"]


@dataclass(frozen=True)
class ExchangeNames:
    run_id: str

    @property
    def exchange_actor(self) -> str:
        return f"exchange_{self.run_id}"


class ExchangeAsMessageQueueClient:
    """
    Adapter to satisfy the subset of MessageQueueClient API used by
    FullyAsyncTrainer/FullyAsyncRollouter, but backed by BidirectionalExchangeQueue.
    """

    def __init__(self, exchange_client: Any, *, enable_gate: bool = False):
        self.exchange_client = exchange_client
        self.enable_gate = bool(enable_gate)

    async def put_sample(self, sample: Any, prompt_hash: str = "") -> bool:
        # Gate: only rollouter put is expected to call put_sample
        if self.enable_gate and hasattr(self.exchange_client, "gate_wait_put_sync"):
            self.exchange_client.gate_wait_put_sync()
        return await self.exchange_client.send_to_peer(sample)

    async def get_sample(self) -> Any | None:
        # Trainer uses sync path today, but keep async for compatibility.
        if self.enable_gate and hasattr(self.exchange_client, "gate_wait_get_sync"):
            self.exchange_client.gate_wait_get_sync()
        res = await self.exchange_client.recv_from_peer()
        return res[0] if isinstance(res, tuple) else res

    def get_sample_sync(self) -> Any | None:
        if self.enable_gate and hasattr(self.exchange_client, "gate_wait_get_sync"):
            self.exchange_client.gate_wait_get_sync()
        res = self.exchange_client.recv_from_peer_sync()
        return res[0] if isinstance(res, tuple) else res

    async def get_queue_size(self) -> int:
        stats = self.get_statistics_sync()
        # For side A: incoming is b_to_a_size; for side B: incoming is a_to_b_size.
        return int(stats.get("incoming_size", 0))

    async def get_statistics(self) -> dict[str, Any]:
        return self.get_statistics_sync()

    def get_statistics_sync(self) -> dict[str, Any]:
        stats = self.exchange_client.get_statistics_sync()
        if self.exchange_client.side == "A":
            incoming = stats.get("b_to_a_size", 0)
            outgoing = stats.get("a_to_b_size", 0)
            produced = stats.get("a_to_b_produced", 0)
            consumed = stats.get("a_to_b_consumed", 0)
            dropped = stats.get("a_to_b_dropped", 0)
        else:
            incoming = stats.get("a_to_b_size", 0)
            outgoing = stats.get("b_to_a_size", 0)
            produced = stats.get("b_to_a_produced", 0)
            consumed = stats.get("b_to_a_consumed", 0)
            dropped = stats.get("b_to_a_dropped", 0)

        # Compatibility shim:
        # FullyAsyncRollouter expects "queue_size" to reflect the queue it is producing into,
        # and uses it for pause/backpressure decisions.
        compat = {
            "incoming_size": incoming,
            "outgoing_size": outgoing,
            "queue_size": outgoing,
            "total_produced": produced,
            "total_consumed": consumed,
            "dropped_samples": dropped,
        }
        return {**stats, **compat}

    async def clear_queue(self):
        # Exchange queue does not currently support clear; no-op for compatibility.
        return None

    async def shutdown(self):
        return None

    async def put_validate(self, data: Any) -> bool:
        # Validation path isn't needed for exchange flow; keep it as no-op.
        return True

    def get_validate_sync(self) -> Any | None:
        return None

    async def get_memory_usage(self) -> dict:
        return {}


class GroupMergeMQClient:
    """Unified MQ client for hash-grouped TCP exchange (GAP-GRPO).

    Used by BOTH rollouter (put_sample → push_grouped) and
    trainer (get_sample_sync → pull_grouped + DataProto.concat merge).

    The TCP server handles all grouping by prompt_hash.  This client simply
    pushes/pulls and performs the DataProto merge on the trainer side.
    """

    def __init__(self, tcp_client: Any):
        self.tcp_client = tcp_client
        self._merged_groups = 0

    # ------------------------------------------------------------------
    # Rollouter side: push sample to server with prompt_hash
    # ------------------------------------------------------------------

    async def put_sample(self, sample: Any, prompt_hash: str = "") -> bool:
        """Push sample to TCP server for hash-based grouping."""
        return await self.tcp_client.push_grouped_async(prompt_hash, sample)

    # ------------------------------------------------------------------
    # Trainer side: pull a completed group and merge DataProto
    # ------------------------------------------------------------------

    def get_sample_sync(self) -> tuple[Any, int]:
        """Pull a merged group from TCP server. Blocks until a group is ready."""
        import ray.cloudpickle as pkl
        from verl import DataProto
        from verl.experimental.fully_async_policy.detach_utils import RolloutSample

        group_bytes, qlen = self.tcp_client.pull_grouped_sync()
        samples = [pkl.loads(b) for b in group_bytes]

        # Extract source sides from non_tensor_batch for logging
        sides = []
        for s in samples:
            try:
                side = str(s.full_batch.non_tensor_batch.get("source_side", ["?"])[0])
            except Exception:
                side = "?"
            sides.append(side)

        # Merge all DataProto batches into one large group
        merged_batch = DataProto.concat([s.full_batch for s in samples])
        merged = RolloutSample(
            full_batch=merged_batch,
            sample_id=samples[0].sample_id + "_merged",
            epoch=samples[0].epoch,
            rollout_status=samples[0].rollout_status,
            prompt_hash=samples[0].prompt_hash,
        )
        self._merged_groups += 1

        ph = samples[0].prompt_hash or "?"
        print(
            f"[GroupMergeMQClient] MERGED hash={ph[:8]} "
            f"sides={sides} merged_batch_size={len(merged_batch)} "
            f"total_merged={self._merged_groups}",
            flush=True,
        )

        return pkl.dumps(merged), qlen

    # ------------------------------------------------------------------
    # Stats (used by rollouter for backpressure)
    # ------------------------------------------------------------------

    async def get_queue_size(self) -> int:
        stats = self.get_statistics_sync()
        return int(stats.get("queue_size", 0))

    async def get_statistics(self) -> dict[str, Any]:
        return self.get_statistics_sync()

    def get_statistics_sync(self) -> dict[str, Any]:
        stats = self.tcp_client.get_statistics_sync()
        return {
            **stats,
            "queue_size": stats.get("queue_size", 0),
            "group_merge/merged_groups": self._merged_groups,
        }

    # ------------------------------------------------------------------
    # Compatibility stubs (required by FullyAsyncTrainer/Rollouter API)
    # ------------------------------------------------------------------

    async def get_sample(self) -> Any | None:
        result = self.get_sample_sync()
        return result[0] if result else None

    async def clear_queue(self):
        return None

    async def shutdown(self):
        return None

    async def put_validate(self, data: Any) -> bool:
        return True

    def get_validate_sync(self) -> Any | None:
        return None

    async def get_memory_usage(self) -> dict:
        return {}


@ray.remote(num_cpus=1)
class FullyAsyncExchangeTaskRunner:
    """
    TaskRunner compatible with verl.trainer.main_ppo.run_ppo.
    """

    def __init__(self):
        self.running = False
        self.components: dict[str, Any] = {}
        self.shutdown_event = threading.Event()

    def run(self, config):
        print("[EXCHANGE MAIN] Starting exchange-enabled fully async PPO training...", flush=True)
        self._initialize_components(config)
        self._run_training_loop()

    def _initialize_components(self, config) -> None:
        print(f"[EXCHANGE MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}", flush=True)
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        exchange_cfg = getattr(config, "exchange", {})
        side: Side = str(getattr(exchange_cfg, "side", "A")).upper()
        mode: Mode = str(getattr(exchange_cfg, "mode", "both")).lower()  # type: ignore[assignment]
        if side not in ("A", "B"):
            raise ValueError(f"exchange.side must be A/B, got: {side}")
        if mode not in ("both", "train_only", "train_first"):
            raise ValueError(f"exchange.mode must be both/train_only/train_first, got: {mode}")

        run_id = str(getattr(exchange_cfg, "run_id", "default"))
        names = ExchangeNames(run_id=run_id)

        self.components["config"] = config
        self.components["exchange_side"] = side
        self.components["exchange_mode"] = mode
        self.components["exchange_run_id"] = run_id

        print("[EXCHANGE MAIN] Initializing model and tokenizer...", flush=True)
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        print(f"[EXCHANGE MAIN] copy_to_local done: {local_path}", flush=True)
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor

        print("[EXCHANGE MAIN] Creating worker mapping and resource pools...", flush=True)
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        # Create trainer/rollouter (same as fully_async_main, but we will wire exchange client)
        from concurrent.futures import ThreadPoolExecutor

        print("[EXCHANGE MAIN] Creating FullyAsyncRollouter and FullyAsyncTrainer in parallel...", flush=True)
        with ThreadPoolExecutor(max_workers=2) as executor:
            trainer_future = executor.submit(self._create_trainer, config)
            trainer_future.result()
            rollouter_future = executor.submit(self._create_rollouter, config)
            rollouter_future.result()

        # Sync total_train_steps between rollouter and trainer
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        print(f"[EXCHANGE MAIN] total_train_steps {total_train_steps}", flush=True)
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

        # Create / reuse a detached exchange actor so A and B can connect to the same channel.
        backend = str(getattr(exchange_cfg, "backend", "ray")).lower()
        max_queue_size = int(getattr(exchange_cfg, "max_queue_size", 20000))
        enable_gate = bool(getattr(exchange_cfg, "enable_gate", False))
        if backend == "ray":
            namespace = ray.get_runtime_context().namespace
            try:
                exchange_actor = ray.get_actor(names.exchange_actor, namespace=namespace)
                print(f"[EXCHANGE MAIN] detected existing exchange actor: {names.exchange_actor} (ns={namespace})")
            except Exception:
                exchange_actor = BidirectionalExchangeQueue.options(
                    name=names.exchange_actor, lifetime="detached", namespace=namespace
                ).remote(max_queue_size=max_queue_size)
                print(f"[EXCHANGE MAIN] created exchange actor: {names.exchange_actor} (ns={namespace})")

            exchange_client = BidirectionalExchangeClient(exchange_actor, side=side)
            mq_client = ExchangeAsMessageQueueClient(exchange_client, enable_gate=enable_gate)
            self.components["exchange_actor"] = exchange_actor
        elif backend == "tcp":
            host = str(getattr(exchange_cfg, "host", "127.0.0.1"))
            port = int(getattr(exchange_cfg, "port", 18080))
            tcp_client = TcpExchangeClient(host=host, port=port, run_id=run_id, side=side)
            # Single GroupMergeMQClient used by both rollouter (put_sample) and trainer (get_sample_sync)
            mq_client = GroupMergeMQClient(tcp_client)
            self.components["exchange_actor"] = None
            print(
                f"[EXCHANGE MAIN] TCP hash-grouped exchange: {host}:{port} "
                f"run_id={run_id} side={side}",
                flush=True,
            )
        else:
            raise ValueError(f"exchange.backend must be ray/tcp, got: {backend}")

        self.components["message_queue_client"] = mq_client

        # Wire client — both rollouter and trainer use the same mq_client
        ray.get(self.components["rollouter"].set_message_queue_client.remote(mq_client))
        ray.get(self.components["trainer"].set_message_queue_client.remote(mq_client))

        # Load checkpoints
        ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

        # Bootstrap: side B rollouter starts paused (no generation) until first param sync.
        bootstrap_b_pause_rollouter = bool(getattr(exchange_cfg, "bootstrap_b_pause_rollouter", True))
        if side == "B" and bootstrap_b_pause_rollouter:
            try:
                ray.get(self.components["rollouter"].bootstrap_pause.remote())
            except Exception as e:
                print(f"[EXCHANGE MAIN] bootstrap_pause failed/unsupported: {e}", flush=True)

        # Parameter sync setup
        # Optional: wrap rollouter.reset_staleness to emit on_param_update to exchange gate (ray/tcp backends).
        backend = str(getattr(getattr(config, "exchange", {}), "backend", "ray")).lower()
        side: Side = self.components["exchange_side"]
        run_id = self.components["exchange_run_id"]
        enable_gate = bool(getattr(getattr(config, "exchange", {}), "enable_gate", False))

        if enable_gate:
            @ray.remote(num_cpus=1)
            class _RollouterProxy:
                def __init__(
                    self,
                    real,
                    *,
                    backend: str,
                    side: str,
                    run_id: str,
                    exchange_actor: Any | None = None,
                    host: str = "127.0.0.1",
                    port: int = 18080,
                ):
                    self.real = real
                    self.backend = backend
                    self.side = side
                    self.run_id = run_id
                    self.exchange_actor = exchange_actor
                    self.host = host
                    self.port = int(port)

                    self._client = None
                    if backend == "tcp":
                        self._client = TcpExchangeClient(host=host, port=self.port, run_id=run_id, side=side)
                    elif backend == "ray":
                        if exchange_actor is not None:
                            self._client = BidirectionalExchangeClient(exchange_actor, side=side)

                def get_replicas(self):
                    return ray.get(self.real.get_replicas.remote())

                def do_validate(self):
                    return ray.get(self.real.do_validate.remote())

                def save_checkpoint(self, local_global_step_folder: str):
                    return ray.get(self.real.save_checkpoint.remote(local_global_step_folder))

                def reset_staleness(self):
                    ret = ray.get(self.real.reset_staleness.remote())
                    # Signal "this side just finished a parameter update" to flip gate phase.
                    try:
                        if self._client is not None and hasattr(self._client, "on_param_update_sync"):
                            self._client.on_param_update_sync()
                    except Exception as e:
                        print(f"[EXCHANGE MAIN] on_param_update failed: {e}", flush=True)
                    return ret

            if backend == "tcp":
                host = str(getattr(getattr(config, "exchange", {}), "host", "127.0.0.1"))
                port = int(getattr(getattr(config, "exchange", {}), "port", 18080))
                proxy = _RollouterProxy.remote(
                    self.components["rollouter"],
                    backend=backend,
                    side=side,
                    run_id=run_id,
                    host=host,
                    port=port,
                )
                ray.get(self.components["trainer"].set_rollouter.remote(proxy))
            elif backend == "ray":
                proxy = _RollouterProxy.remote(
                    self.components["rollouter"],
                    backend=backend,
                    side=side,
                    run_id=run_id,
                    exchange_actor=self.components.get("exchange_actor"),
                )
                ray.get(self.components["trainer"].set_rollouter.remote(proxy))
            else:
                ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))
        else:
            ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))
        print("[EXCHANGE MAIN] Param sync before fit..", flush=True)
        ray.get(self.components["trainer"]._fit_update_weights.remote())

        if config.trainer.get("val_before_train", True):
            ray.get(self.components["trainer"]._fit_validate.remote(True))

        print(
            f"[EXCHANGE MAIN] initialized. side={side} mode={mode} run_id={run_id} exchange={names.exchange_actor}",
            flush=True,
        )

    def _create_rollouter(self, config) -> None:
        rollouter = FullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=None,
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())
        self.components["rollouter"] = rollouter

    def _create_trainer(self, config) -> None:
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }
        trainer = FullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer

    def _run_training_loop(self):
        self.running = True
        side: Side = self.components["exchange_side"]
        mode: Mode = self.components["exchange_mode"]

        # Scheduling policy (to match the intended bootstrap rhythm):
        # - Side A: rollout first, then train (R -> T -> sync -> R ...)
        # - Side B: train first, then rollout (T -> sync -> R ...)
        #
        # Notes:
        # - We already perform an initial trainer->rollouter param sync in _initialize_components
        #   via trainer._fit_update_weights().
        # - Subsequent periodic synchronization is handled inside FullyAsyncTrainer.
        futures = []

        if mode == "train_only":
            print(f"[EXCHANGE MAIN] Starting Trainer (side={side}, mode={mode}) ...")
            futures.append(self.components["trainer"].fit.remote())
            print("[EXCHANGE MAIN] Rollouter not started (train_only)")
        elif mode == "both":
            if side == "A":
                print("[EXCHANGE MAIN] Starting Rollouter first (side=A, mode=both) ...")
                futures.append(self.components["rollouter"].fit.remote())
                print("[EXCHANGE MAIN] Starting Trainer (side=A, mode=both) ...")
                futures.append(self.components["trainer"].fit.remote())
            else:
                print("[EXCHANGE MAIN] Starting Trainer first (side=B, mode=both) ...")
                futures.append(self.components["trainer"].fit.remote())
                print("[EXCHANGE MAIN] Starting Rollouter (side=B, mode=both) ...")
                futures.append(self.components["rollouter"].fit.remote())
        else:
            # train_first is equivalent to "start both, but in different order":
            # - side A: rollout first
            # - side B: train first (wait for first batch to complete before starting rollouter)
            if side == "A":
                print("[EXCHANGE MAIN] train_first (side=A): Starting Rollouter first ...")
                futures.append(self.components["rollouter"].fit.remote())
                print("[EXCHANGE MAIN] train_first (side=A): Starting Trainer ...")
                futures.append(self.components["trainer"].fit.remote())
            else:
                # B side: Start trainer, wait for it to complete one training step, then start rollouter
                print("[EXCHANGE MAIN] train_first (side=B): Starting Trainer first (will wait for first batch) ...")
                
                # Start trainer in background
                trainer_future = self.components["trainer"].fit.remote()
                
                # With bootstrap_pause, we can start rollouter immediately without risking early generation.
                print("[EXCHANGE MAIN] train_first (side=B): Starting Rollouter ...")
                futures.append(trainer_future)
                futures.append(self.components["rollouter"].fit.remote())

        try:
            while futures:
                done_futures, remaining_futures = ray.wait(futures, num_returns=len(futures), timeout=None)
                for future in done_futures:
                    try:
                        ray.get(future)
                        print("[EXCHANGE MAIN] One component completed successfully")
                    except Exception as e:
                        print(f"[EXCHANGE MAIN] Component failed with error: {e}")
                        for remaining_future in remaining_futures:
                            ray.cancel(remaining_future)
                        raise e
                futures = remaining_futures
        except Exception as e:
            print(f"[EXCHANGE MAIN] Training failed: {e}")
            for future in futures:
                ray.cancel(future)
            raise
        finally:
            # Exchange queue does not support clear; keep for symmetry with fully_async_main.
            asyncio.run(self.components["message_queue_client"].clear_queue())
            print("[EXCHANGE MAIN] Training completed or interrupted")


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")
    assert config.async_training.use_trainer_do_validate is False, "use_trainer_do_validate is not ready to use."

    # Keep rollout config unification same as fully_async_main
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node

    from time import time

    start_time = time()
    run_ppo(config, task_runner_class=FullyAsyncExchangeTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

