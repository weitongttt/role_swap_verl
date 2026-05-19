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
  TCP exchange server supporting N independent sites.
- Support bootstrap mode: non-primary sites can start as train-only to avoid deadlock.
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
from verl.experimental.fully_async_policy.tcp_exchange import TcpExchangeClient
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.utils import Role
from verl.utils.fs import copy_to_local

Mode = Literal["both", "train_only", "train_first"]


@dataclass(frozen=True)
class ExchangeNames:
    run_id: str

    @property
    def exchange_actor(self) -> str:
        return f"exchange_{self.run_id}"


class GroupMergeMQClient:
    """Unified MQ client for hash-grouped TCP exchange (GAP-GRPO).

    Used by BOTH rollouter (put_sample → push_grouped) and
    trainer (get_sample_sync → pull_grouped).

    The TCP server groups samples by prompt_hash.  When the server returns a
    completed group (list of pickled samples from all sites), this client
    buffers them and returns one at a time.  This ensures:
    - The trainer's required_samples count works correctly (1 call = 1 sample).
    - Samples from all sites for the same prompt arrive in the same training
      batch (because they are buffered together and returned consecutively).
    """

    def __init__(self, tcp_client: Any):
        from collections import deque

        self.tcp_client = tcp_client
        self._buffer: deque = deque()
        self._buffer_qlen: int = 0
        self._groups_pulled: int = 0
        self._total_returned: int = 0

    # ------------------------------------------------------------------
    # Rollouter side: push sample to server with prompt_hash
    # ------------------------------------------------------------------

    async def put_sample(self, sample: Any, prompt_hash: str = "") -> bool:
        """Push sample to TCP server for hash-based grouping."""
        return await self.tcp_client.push_grouped_async(prompt_hash, sample)

    # ------------------------------------------------------------------
    # Trainer side: return individual samples from grouped pulls
    # ------------------------------------------------------------------

    def get_sample_sync(self) -> tuple[Any, int]:
        """Return one sample at a time. Pulls a new group from the server
        when the buffer is empty; then returns buffered samples one by one.
        """
        if not self._buffer:
            group_bytes, qlen = self.tcp_client.pull_grouped_sync()
            self._buffer_qlen = qlen
            self._groups_pulled += 1

            # Buffer individual samples (do NOT merge — keeps trainer count correct)
            self._buffer.extend(group_bytes)

            # Log: detailed for first 5 groups, then periodic summary
            if self._groups_pulled <= 5 or self._groups_pulled % 50 == 0:
                try:
                    import ray.cloudpickle as pkl

                    sides = []
                    ph_short = "?"
                    for b in group_bytes:
                        s = pkl.loads(b)
                        try:
                            sides.append(str(s.full_batch.non_tensor_batch.get("source_side", ["?"])[0]))
                        except Exception:
                            sides.append("?")
                        if ph_short == "?":
                            ph_short = (s.prompt_hash or "?")[:8]
                    print(
                        f"[GroupMergeMQClient] GROUP #{self._groups_pulled} "
                        f"hash={ph_short} sides={sides} "
                        f"group_size={len(group_bytes)} "
                        f"total_returned={self._total_returned}",
                        flush=True,
                    )
                except Exception:
                    print(
                        f"[GroupMergeMQClient] GROUP #{self._groups_pulled} "
                        f"group_size={len(group_bytes)}",
                        flush=True,
                    )

        self._total_returned += 1
        return self._buffer.popleft(), self._buffer_qlen

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
            "group_merge/groups_pulled": self._groups_pulled,
            "group_merge/total_returned": self._total_returned,
            "group_merge/buffer_size": len(self._buffer),
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
        # site_id: any non-empty string identifying this site (e.g. "0", "A", "beijing")
        site_id = str(getattr(exchange_cfg, "side", "0"))
        mode: Mode = str(getattr(exchange_cfg, "mode", "both")).lower()  # type: ignore[assignment]
        # site_index: integer index used for bootstrap ordering (0 = primary site)
        site_index = int(getattr(exchange_cfg, "site_index", 0))

        if not site_id:
            raise ValueError(f"exchange.side must be a non-empty string, got: {site_id!r}")
        if mode not in ("both", "train_only", "train_first"):
            raise ValueError(f"exchange.mode must be both/train_only/train_first, got: {mode}")

        run_id = str(getattr(exchange_cfg, "run_id", "default"))
        names = ExchangeNames(run_id=run_id)

        self.components["config"] = config
        self.components["exchange_site_id"] = site_id
        self.components["exchange_site_index"] = site_index
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

        # Create TCP exchange client (only TCP backend supported)
        backend = str(getattr(exchange_cfg, "backend", "tcp")).lower()
        if backend != "tcp":
            raise ValueError(
                f"exchange.backend must be 'tcp', got: {backend!r}. "
                f"The legacy 'ray' backend has been removed."
            )

        host = str(getattr(exchange_cfg, "host", "127.0.0.1"))
        port = int(getattr(exchange_cfg, "port", 18080))
        tcp_client = TcpExchangeClient(host=host, port=port, run_id=run_id, site_id=site_id)
        # Single GroupMergeMQClient used by both rollouter (put_sample) and trainer (get_sample_sync)
        mq_client = GroupMergeMQClient(tcp_client)
        self.components["exchange_actor"] = None
        print(
            f"[EXCHANGE MAIN] TCP hash-grouped exchange: {host}:{port} "
            f"run_id={run_id} site_id={site_id} site_index={site_index}",
            flush=True,
        )

        self.components["message_queue_client"] = mq_client

        # Wire client — both rollouter and trainer use the same mq_client
        ray.get(self.components["rollouter"].set_message_queue_client.remote(mq_client))
        ray.get(self.components["trainer"].set_message_queue_client.remote(mq_client))

        # Load checkpoints
        ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

        # Parameter sync setup
        enable_gate = bool(getattr(exchange_cfg, "enable_gate", False))

        if enable_gate:
            @ray.remote(num_cpus=1)
            class _RollouterProxy:
                def __init__(
                    self,
                    real,
                    *,
                    site_id: str,
                    run_id: str,
                    host: str = "127.0.0.1",
                    port: int = 18080,
                ):
                    self.real = real
                    self.site_id = site_id
                    self.run_id = run_id
                    self.host = host
                    self.port = int(port)
                    self._client = TcpExchangeClient(host=host, port=self.port, run_id=run_id, site_id=site_id)

                def get_replicas(self):
                    return ray.get(self.real.get_replicas.remote())

                def do_validate(self):
                    return ray.get(self.real.do_validate.remote())

                def save_checkpoint(self, local_global_step_folder: str):
                    return ray.get(self.real.save_checkpoint.remote(local_global_step_folder))

                def reset_staleness(self):
                    ret = ray.get(self.real.reset_staleness.remote())
                    # Signal "this site just finished a parameter update" to flip gate phase.
                    try:
                        if self._client is not None and hasattr(self._client, "on_param_update_sync"):
                            self._client.on_param_update_sync()
                    except Exception as e:
                        print(f"[EXCHANGE MAIN] on_param_update failed: {e}", flush=True)
                    return ret

            proxy = _RollouterProxy.remote(
                self.components["rollouter"],
                site_id=site_id,
                run_id=run_id,
                host=host,
                port=port,
            )
            ray.get(self.components["trainer"].set_rollouter.remote(proxy))
        else:
            ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))

        print("[EXCHANGE MAIN] Param sync before fit..", flush=True)
        ray.get(self.components["trainer"]._fit_update_weights.remote())

        if config.trainer.get("val_before_train", True):
            ray.get(self.components["trainer"]._fit_validate.remote(True))

        print(
            f"[EXCHANGE MAIN] initialized. site_id={site_id} site_index={site_index} "
            f"mode={mode} run_id={run_id} exchange={names.exchange_actor}",
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
        site_id: str = self.components["exchange_site_id"]
        site_index: int = self.components["exchange_site_index"]
        mode: Mode = self.components["exchange_mode"]

        # Both rollouter and trainer run as concurrent async Ray tasks.
        # Primary site (site_index=0) submits rollouter first to bootstrap.
        # Non-primary sites submit trainer first.
        # The trainer naturally blocks on pull_grouped_sync() until the
        # exchange server has a complete group, so ordering is cosmetic.
        futures = []

        if mode == "train_only":
            print(f"[EXCHANGE MAIN] Starting Trainer (site={site_id}, mode={mode}) ...")
            futures.append(self.components["trainer"].fit.remote())
            print("[EXCHANGE MAIN] Rollouter not started (train_only)")
        else:
            # both / train_first — use site_index for bootstrap ordering
            if site_index == 0 or mode == "both":
                print(f"[EXCHANGE MAIN] Starting Rollouter first (site={site_id}, index={site_index}, mode={mode}) ...")
                futures.append(self.components["rollouter"].fit.remote())
                print(f"[EXCHANGE MAIN] Starting Trainer (site={site_id}, mode={mode}) ...")
                futures.append(self.components["trainer"].fit.remote())
            else:
                print(f"[EXCHANGE MAIN] Starting Trainer first (site={site_id}, index={site_index}, mode={mode}) ...")
                futures.append(self.components["trainer"].fit.remote())
                print(f"[EXCHANGE MAIN] Starting Rollouter (site={site_id}, mode={mode}) ...")
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
