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

import asyncio
import logging
import math
import os
import time
from datetime import datetime
from pprint import pprint
from typing import Any

import ray
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    ValidateMetrics,
    assemble_batch_from_rollout_samples,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking

logger = logging.getLogger(__name__)


class TrainingStopException(Exception):
    """Exception raised to signal training should stop"""

    pass


@ray.remote(num_cpus=10)
class FullyAsyncTrainer(SeparateRayPPOTrainer):
    """
    A fully asynchronous PPO trainer that obtains samples from a MessageQueue for training.
    Based on an improved implementation of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
    ):
        # ==================== RayPPOTrainer config ====================

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # ==================== SeparateRayPPOTrainer config ====================
        self.global_steps = 0
        self.epoch = 0
        self.max_steps_duration = 0
        self.progress_bar = None
        self.is_last_step = False
        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # ==================== fully async config ====================

        self.message_queue_client = None

        # Statistics
        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.progress_bar = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0
        self.train_role = (
            Role.ActorRollout
            if config.async_training.use_trainer_do_validate
            or config.async_training.get("colocate_actor_rollout", False)
            else Role.Actor
        )

        # required_samples means "required queue items per train step".
        # In exchange bs-mode, one queue item can already represent a full rollout batch.
        self.require_batches = config.async_training.require_batches
        train_need = int(config.actor_rollout_ref.actor.ppo_mini_batch_size) * int(self.require_batches)
        gen_bsz = int(config.data.gen_batch_size)
        exchange_cfg = getattr(config, "exchange", None)
        if hasattr(config, "exchange"):
            self.required_samples = int(math.ceil(train_need / max(1, gen_bsz)))
        else:
            self.required_samples = train_need
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)
        self.exchange_side = str(getattr(exchange_cfg, "side", "A")).upper() if exchange_cfg is not None else "A"
        self._skip_local_first_step = self.exchange_side == "B"

        # use trainer to do validation
        if self.config.async_training.use_trainer_do_validate:
            from verl.trainer.main_ppo import create_rl_dataset
            from verl.utils.dataset.rl_dataset import collate_fn

            val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
            rollout_gpus = config.rollout.nnodes * config.rollout.n_gpus_per_node
            print(f"[FullyAsyncTrainer] split before val_dataset total len: {len(val_dataset)}")
            split_dataset = val_dataset.split(total_gpus)
            rollout_val_dataset0 = split_dataset[rollout_gpus:]
            from torch.utils.data import ConcatDataset

            val_dataset = ConcatDataset(rollout_val_dataset0)
            print(f"[FullyAsyncTrainer] split after val_dataset total len: {len(val_dataset)}")
            self.val_dataset = val_dataset
            # update val_dataloader
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(val_dataset)
            from torchdata.stateful_dataloader import StatefulDataLoader

            print(f"[FullyAsyncTrainer] create val_dataloader with batch_size: {val_batch_size}")
            self.val_dataloader = StatefulDataLoader(
                dataset=val_dataset,
                batch_size=val_batch_size,
                num_workers=self.config.data["dataloader_num_workers"],
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )
        # Reference to rollouter for parameter synchronization
        self.rollouter = None
        self.checkpoint_manager = None

        # when use_trainer_do_validate == Ture, use colocate_checkpoint_manager to sync params
        self.colocate_checkpoint_manager = None

    def _setup_checkpoint_manager(self, rollouter):
        """Setup checkpoint manager after rollouter is initialized"""
        replicas = ray.get(rollouter.get_replicas.remote())
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, trainer=self.actor_wg, replicas=replicas
        )
        print("[FullyAsyncTrainer] Checkpoint manager initialized")

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    def get_global_steps(self) -> int:
        return int(self.global_steps)

    def set_rollouter(self, rollouter):
        """Set rollouter reference for parameter synchronization"""
        self.rollouter = rollouter
        # Setup checkpoint manager after rollouter is set
        self._setup_checkpoint_manager(rollouter)

    def set_total_train_steps(self, total_training_steps):
        self.total_train_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

        self.progress_bar = tqdm(total=self.total_train_steps, initial=0, desc="Training Progress")

    def get_actor_wg(self):
        """Get actor worker group"""
        return self.actor_wg

    async def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """
        Get samples from message queue and compose gen_batch_output
        Uses a loop to continuously collect samples until enough are gathered

        Returns:
            tuple: (epoch, batch_dict, gen_batch_output)
        """
        need = self.required_samples
        print(
            f"[FullyAsyncTrainer] MQ consumer active: waiting until incoming>={need} samples for this train step "
            f"(background monitor prints queue depth periodically; disable with VERL_MQ_MONITOR=0)",
            flush=True,
        )

        consumer_start = time.time()
        queue_samples: list[Any] = []
        queue_len = 0
        used_local_batch_this_step = False

        use_batch = hasattr(self.message_queue_client, "get_samples_batch_sync")
        use_local_batch = hasattr(self.message_queue_client, "get_local_samples_batch_sync")
        mq = self.message_queue_client
        exchange_cfg = getattr(self.config, "exchange", None)
        if exchange_cfg is not None and not use_batch:
            raise RuntimeError(
                "exchange mode requires batched queue pull (get_samples_batch_sync) to keep bs-granularity training"
            )
        stop_monitor = asyncio.Event()
        enable_monitor = os.environ.get("VERL_MQ_MONITOR", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

        async def _mq_monitor_loop() -> None:
            raw = os.environ.get("VERL_MQ_MONITOR_INTERVAL_S", "5").strip()
            try:
                interval = float(raw)
            except ValueError:
                interval = 5.0
            interval = max(0.5, interval)
            first = True
            while not stop_monitor.is_set():
                if not first:
                    await asyncio.sleep(interval)
                first = False
                if stop_monitor.is_set():
                    break
                try:
                    if hasattr(mq, "get_statistics") and asyncio.iscoroutinefunction(mq.get_statistics):
                        stats = await mq.get_statistics()
                    else:
                        stats = await asyncio.to_thread(mq.get_statistics_sync)
                    inc = stats.get("incoming_size", stats.get("queue_size", -1))
                    out = stats.get("outgoing_size", -1)
                    ready = isinstance(inc, int) and inc >= need
                    print(
                        f"[FullyAsyncTrainer] MQ monitor: incoming={inc} outgoing={out} need={need} "
                        f"train_ready={ready}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[FullyAsyncTrainer] MQ monitor: stats failed: {e!r}", flush=True)

        monitor_task = None
        if enable_monitor:
            monitor_task = asyncio.create_task(_mq_monitor_loop())

        def _sync_collect_one_by_one() -> tuple[list[Any], int]:
            samples: list[Any] = []
            qlen = 0
            while len(samples) < need:
                dbg_raw = os.environ.get("VERL_EXCHANGE_DEBUG", "").strip().lower()
                dbg_level = 0
                if dbg_raw in ("true", "yes", "on"):
                    dbg_level = 2
                else:
                    try:
                        dbg_level = int(dbg_raw)
                    except ValueError:
                        dbg_level = 0

                if dbg_level >= 2:
                    print(
                        f"[FullyAsyncTrainer][EXCHANGE_DEBUG] about to get_sample_sync "
                        f"have={len(samples)}/{need}",
                        flush=True,
                    )
                sample, qlen = mq.get_sample_sync()

                if sample is None:
                    print(
                        f"[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. "
                        f"Collected {len(samples)}/{need} samples"
                    )
                    break

                samples.append(sample)

                if len(samples) % 64 == 0:
                    print(
                        f"[FullyAsyncTrainer] Collected {len(samples)}/{need} samples. mq_len: {qlen}",
                        flush=True,
                    )
            return samples, qlen

        try:
            if use_batch:
                queue_samples, queue_len = await asyncio.to_thread(
                    mq.get_samples_batch_sync,
                    need,
                )
                # Combine one local batch with MQ batch on every step, except B's first step.
                if use_local_batch:
                    if self._skip_local_first_step:
                        self._skip_local_first_step = False
                    else:
                        local_samples, _ = await asyncio.to_thread(
                            mq.get_local_samples_batch_sync,
                            need,
                        )
                        queue_samples.extend(local_samples)
                        used_local_batch_this_step = True
            else:
                queue_samples, queue_len = await asyncio.to_thread(_sync_collect_one_by_one)
        finally:
            stop_monitor.set()
            if monitor_task is not None:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

        if use_batch:
            if (len(queue_samples) == 1 and queue_samples[0] is None) or (
                not queue_samples and queue_len == 0
            ):
                print(
                    "[FullyAsyncTrainer] batch pull: empty or termination signal",
                    flush=True,
                )
                return None, None
            if any(x is None for x in queue_samples):
                print(
                    f"[FullyAsyncTrainer] batch pull: got None inside batch, "
                    f"len={len(queue_samples)}/{need}",
                    flush=True,
                )
                return None, None
            if len(queue_samples) < need:
                print(
                    f"[FullyAsyncTrainer] batch pull: incomplete {len(queue_samples)}/{need}",
                    flush=True,
                )
                return None, None

        consumer_end = time.time()

        min_need = self.required_samples * (2 if used_local_batch_this_step else 1)
        if not queue_samples or len(queue_samples) < min_need:
            print("[FullyAsyncTrainer] not enough samples collected after loop")
            return None, None
        total_wait_time = consumer_end - consumer_start

        if not use_batch:
            print(
                f"[FullyAsyncTrainer] Loop collection completed: {len(queue_samples)}/{self.required_samples} samples, "
                f"total wait time: {total_wait_time:.2f} seconds. "
                f"mq_len: {queue_len}",
                flush=True,
            )
        else:
            print(
                f"[FullyAsyncTrainer] Batch pull: {len(queue_samples)} samples in one round-trip, "
                f"total wait time: {total_wait_time:.2f}s, mq_len: {queue_len}",
                flush=True,
            )

        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
        # Assemble batch - now working directly with RolloutSample objects
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, self._balance_batch)
        else:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        return 0, batch

    def _create_actor_rollout_classes(self):
        # create actor
        for role in [self.train_role]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                from functools import partial
                from verl.workers.utils.losses import value_loss
                value_loss_ = partial(value_loss, config=self.orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = self.all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_wg = self.all_wg[str(self.train_role)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified

    def get_actor_rollout_wg(self):
        """供同集群 rollouter 走 vLLM init_hybrid 共置（单 Ray GPU，对齐 main_ppo global_pool）。"""
        if self.actor_rollout_wg is None:
            raise RuntimeError("get_actor_rollout_wg: call init_workers first")
        return self.actor_rollout_wg

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.
        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        self._init_reward_loop()
        await self._init_async_rollout_manager()

    def _init_reward_loop(self):
        if self.config.async_training.use_trainer_do_validate:
            print("[FullyAsyncTrainer] Init reward loop")
            super()._init_reward_loop()

    async def _init_async_rollout_manager(self):
        # use async rollout do validate
        print(f"[FullyAsyncTrainer] use_trainer_do_validate: {self.config.async_training.use_trainer_do_validate}")
        if self.config.async_training.use_trainer_do_validate:
            print("[FullyAsyncTrainer] Init async rollout manager")

            # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
            # agent_reward_loop: streaming reward computation with actor rollout
            # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
            enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

            # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
            # to stream reward computation with actor rollout
            reward_loop_worker_handles = (
                self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
            )

            # create async rollout manager and request scheduler
            assert self.config.actor_rollout_ref.rollout.mode == "async"

            self.async_rollout_mode = True
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_manager = await AgentLoopManager.create(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                reward_loop_worker_handles=reward_loop_worker_handles,
            )
            print("[FullyAsyncTrainer] async_rollout_manager initialized")

            # Modify checkpoint_engine config to use naive backend
            checkpoint_engine_cfg = self.config.actor_rollout_ref.rollout.checkpoint_engine
            original_backend = checkpoint_engine_cfg.backend
            with open_dict(checkpoint_engine_cfg):
                checkpoint_engine_cfg.backend = "naive"
            checkpoint_engine_config = omega_conf_to_dataclass(checkpoint_engine_cfg)

            print(f"[FullyAsyncTrainer] checkpoint_engine_config: {checkpoint_engine_config}")

            self.colocate_checkpoint_manager = CheckpointEngineManager(
                config=checkpoint_engine_config,
                trainer=self.actor_rollout_wg,
                replicas=self.async_rollout_manager.rollout_replicas,
            )

            # sleep all replicas to load checkpoint
            await self.colocate_checkpoint_manager.sleep_replicas()

            # Restore original backend value
            with open_dict(checkpoint_engine_cfg):
                checkpoint_engine_cfg.backend = original_backend

            print("[FullyAsyncTrainer] colocate_checkpoint_manager initialized")

        else:
            print("[FullyAsyncTrainer] Skip async rollout manager (use_trainer_do_validate=False)")

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.rollouter is None:
            raise ValueError("rollouter not set. Call set_rollouter() first.")

        self.max_steps_duration = 0

        self.global_steps += 1

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        while True:
            try:
                await self.fit_step()
            except TrainingStopException:
                print("[FullyAsyncTrainer] Training stopped by queue termination signal")
                break

        self.progress_bar.close()
        if self.current_param_version % self.config.trainer.test_freq != 0 or self.local_trigger_step > 1:
            await self._fit_update_weights()
            await self._fit_validate()
        self._fit_save_checkpoint(force=True)

    async def fit_step(self, batch_dict: dict = None):
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update critic -> 9. Update actor -> 10. Post-step processing

        Args:
            batch_dict: Raw data dictionary
        """
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_start_profile()

        with marked_timer("step", self.timing_raw):
            batch = await self._fit_generate(None)
            batch = self._fit_compute_reward(batch)
            batch = self._fit_compute_log_prob(batch)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_critic(batch)
            batch = self._fit_update_actor(batch)
            self._fit_update_local_step()
            await self._fit_update_weights()
            self._fit_dump_data(batch)

        await self._fit_validate()
        self._fit_save_checkpoint()
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self._fit_torch_memory()
        self._fit_postprocess_step()

    async def _fit_generate(self, batch: DataProto = None) -> DataProto | None:
        metrics = self.metrics
        timing_raw = self.timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            epoch, batch = await self._get_samples_from_queue()
            if batch is None:
                raise TrainingStopException("Training terminated: queue returned None")
            self._collect_metrics_from_samples(batch, metrics)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return batch

    def _compute_old_log_prob(self, batch: DataProto):
        """
        If algorithm.rollout_correction.bypass_mode is False,
        use model engine and first version model params to re-calculate old_log_prob.

        If local_trigger_step == 1, load the training engine's parameters to the CPU
          and save a copy for subsequent MIS use.

        If local_trigger_step == 2, 3, ..., restore the parameters of version 1 to calculate the old_log_prob,
        then restore the parameters of the current version.
        """
        if self.local_trigger_step == 1:
            self.actor_rollout_wg.save_model_to_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
            # MIS 计算结束后确保模型回到 GPU。
            # 否则可能出现 optimizer state 在 CPU、参数在 CUDA 的 device mismatch。
            self.actor_rollout_wg.restore_model_from_cpu(1)
        else:
            self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            self.actor_rollout_wg.restore_model_from_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
            self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
            self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
        return old_log_prob, old_log_prob_mfu

    def _fit_update_local_step(self):
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(
            f"[FullyAsyncTrainer] global_steps: {self.global_steps} "
            f"local_trigger_step: {self.local_trigger_step} "
            f"trigger_parameter_sync_step: {self.trigger_parameter_sync_step} "
            f"{time_str}"
        )
        if self.local_trigger_step < self.trigger_parameter_sync_step:
            self.local_trigger_step += 1
        else:
            self.current_param_version += 1
            self.local_trigger_step = 1

    async def _fit_update_weights(self):
        if self.local_trigger_step != 1:
            return

        def _sync_dbg_enabled() -> bool:
            raw = os.environ.get("VERL_SYNC_DEBUG", "").strip().lower()
            if raw in ("", "0", "false", "no", "off"):
                return False
            return True

        def _sync_dbg(msg: str) -> None:
            if not _sync_dbg_enabled():
                return
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(
                f"[FullyAsyncTrainer][SYNC] {ts} {msg} "
                f"(param_version={self.current_param_version} local_trigger_step={self.local_trigger_step})",
                flush=True,
            )

        def _sync_dbg_every_s() -> float:
            # Heartbeat interval; keep it low-noise but actionable.
            raw = os.environ.get("VERL_SYNC_HEARTBEAT_S", "5").strip()
            try:
                v = float(raw)
                return 5.0 if v <= 0 else v
            except Exception:
                return 5.0

        def _sync_timeout_s(name: str, default_s: float) -> float:
            raw = os.environ.get(f"VERL_SYNC_TIMEOUT_{name}_S", str(default_s)).strip()
            try:
                v = float(raw)
                return default_s if v <= 0 else v
            except Exception:
                return default_s

        # Param sync 时会触发 rollouter/vLLM 权重接收（尤其 vLLM v1 的 IPC 逻辑）。
        # main_ppo 的时序是：generate 后 sleep_replicas()，让 KV/cache 释放，避免 update_weights 阶段显存峰值叠加 OOM。
        # 这里对齐 main_ppo：先 bootstrap_pause（阻止继续生成/清理 in-flight），再 sleep_replicas()，
        # update_weights 后再 wake_up_replicas() + reset_staleness。
        try:
            _sync_dbg("bootstrap_pause START")
            ref = self.rollouter.bootstrap_pause.remote()
            deadline = time.time() + _sync_timeout_s("BOOTSTRAP_PAUSE", 120.0)
            next_hb = time.time() + _sync_dbg_every_s()
            while True:
                ready, _ = ray.wait([ref], timeout=1.0)
                if ready:
                    ray.get(ready[0])
                    _sync_dbg("bootstrap_pause DONE")
                    break
                if time.time() >= next_hb:
                    try:
                        st = ray.get(self.rollouter.get_debug_state.remote())
                    except Exception as e:
                        st = {"error": repr(e)}
                    _sync_dbg(f"bootstrap_pause WAITING rollouter_state={st}")
                    next_hb = time.time() + _sync_dbg_every_s()
                if time.time() > deadline:
                    raise TimeoutError("timeout waiting rollouter.bootstrap_pause")
        except Exception:
            # 某些 rollouter 代理可能没有 bootstrap_pause；忽略即可继续走原逻辑。
            _sync_dbg("bootstrap_pause FAILED/UNSUPPORTED -> ignored")
            pass

        # 关键：释放 rollout replicas 的 KV/cache，避免 update_weights 阶段 OOM。
        try:
            _sync_dbg("sleep_replicas START")
            await self.checkpoint_manager.sleep_replicas()
            _sync_dbg("sleep_replicas DONE")
        except Exception:
            _sync_dbg("sleep_replicas FAILED -> ignored")
            pass

        _sync_dbg("update_weights START")
        with marked_timer("timing_s/param_sync", self.timing_raw):
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
        _sync_dbg("update_weights DONE")
        print(
            f"[FullyAsyncTrainer] _fit_update_weights, "
            f"timing_s/param_sync: {self.timing_raw['timing_s/param_sync']:.4f} seconds "
            f"self.current_param_version: {self.current_param_version}"
        )

        # Reset staleness in rollouter
        _sync_dbg("reset_staleness START")
        timing_ref = self.rollouter.reset_staleness.remote()
        deadline = time.time() + _sync_timeout_s("RESET_STALENESS", 300.0)
        next_hb = time.time() + _sync_dbg_every_s()
        while True:
            ready, _ = ray.wait([timing_ref], timeout=1.0)
            if ready:
                timing_raw = ray.get(ready[0])
                _sync_dbg("reset_staleness DONE")
                break
            if time.time() >= next_hb:
                try:
                    st = ray.get(self.rollouter.get_debug_state.remote())
                except Exception as e:
                    st = {"error": repr(e)}
                _sync_dbg(f"reset_staleness WAITING rollouter_state={st}")
                next_hb = time.time() + _sync_dbg_every_s()
            if time.time() > deadline:
                raise TimeoutError("timeout waiting rollouter.reset_staleness")
        # 恢复 rollout replicas 的 KV/weights 常驻显存，供下一轮 generation 使用。
        try:
            _sync_dbg("wake_up_replicas START")
            await self.checkpoint_manager.wake_up_replicas()
            _sync_dbg("wake_up_replicas DONE")
        except Exception:
            _sync_dbg("wake_up_replicas FAILED -> ignored")
            pass
        self.logger.log(
            data=timing_raw,
            step=self.current_param_version,
        )

        # Log aggregated training metrics
        self.logger.log(
            data=self.metrics_aggregator.get_aggregated_metrics(),
            step=self.current_param_version,
        )
        self.metrics_aggregator.reset()

    async def _validate_process(self):
        """Run trainer-side validation using async rollout manager"""
        if self.config.async_training.use_trainer_do_validate:
            print("[FullyAsyncTrainer] _validate_process")
            from verl.utils.profiler import marked_timer

            # Wake up rollouter replicas and sync weights
            print("[FullyAsyncTrainer] wake up replicas before validation")
            await self.colocate_checkpoint_manager.update_weights(global_steps=self.current_param_version)

            with marked_timer("trainer/validate_time", self.timing_raw):
                train_val_metrics = self._validate(True)

            # Sleep rollouter replicas to free GPU memory for validation
            print("[FullyAsyncTrainer] sleep replicas after validation")
            await self.colocate_checkpoint_manager.sleep_replicas()

            print(f"[FullyAsyncTrainer] validate timing: {self.timing_raw['trainer/validate_time']}")
            return train_val_metrics
        else:
            print("[FullyAsyncTrainer] _validate_process without async_rollout_manager")
            return None

    async def _fit_validate(self, val_before_train=False):
        if self.local_trigger_step != 1:
            return

        # Check if validation is needed
        need_validate = (
            self.config.trainer.test_freq > 0
            and self.current_param_version % self.config.trainer.test_freq == 0
            and self.current_param_version > 0
        )
        # Skip validation if not needed and not validation before training
        if not need_validate and not val_before_train:
            return

        # Trigger rollouter validation and get future
        val_future = self.rollouter.do_validate.remote()

        # Run trainer-side validation
        train_val_metrics = await self._validate_process()

        # Wait for rollouter validation result and log
        val_metrics: ValidateMetrics = ray.get(val_future)
        if train_val_metrics:
            # Merge trainer and rollouter validation results
            with marked_timer("timing_s/merge_val", self.timing_raw):
                new_metrics = self._merge_validation_results(train_val_metrics, val_metrics.metrics)
            if new_metrics:
                self.logger.log(data=new_metrics, step=self.current_param_version)
                pprint(
                    f"[FullyAsyncTrainer] parameter version: {self.current_param_version} "
                    f"Validation metrics: {new_metrics}, timing: {self.timing_raw['timing_s/merge_val']}"
                )
        else:
            if val_metrics.metrics:
                self.logger.log(data=val_metrics.metrics, step=self.current_param_version)
                pprint(
                    f"[FullyAsyncTrainer] parameter version: {self.current_param_version} "
                    f"Validation metrics: {val_metrics.metrics}"
                )
        self.logger.log(data=val_metrics.timing_raw, step=self.current_param_version)

    def _fit_save_checkpoint(self, force=False):
        if self.current_param_version == self.last_ckpt_version:
            return

        timing_raw = self.timing_raw
        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        # Check if the conditions for saving a checkpoint are met.
        # The conditions include a mandatory condition (1) and
        # one of the following optional conditions (2/3/4):
        # 1. The save frequency is set to a positive value.
        # 2. It's the last training step.
        # 3. The current step number is a multiple of the save frequency.
        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
        if self.config.trainer.save_freq > 0 and (
            force and self.current_param_version % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                # sleep replicas to avoid OOM during checkpoint saving
                self._save_checkpoint()
                self.last_ckpt_version = self.current_param_version

    def _fit_postprocess_step(self):
        self.global_steps += 1

        self.metrics_aggregator.add_step_metrics(
            metrics=self.metrics, sample_count=self.required_samples, timestamp=time.time()
        )

        if self.local_trigger_step == 1:
            self.progress_bar.update(1)

    def _save_checkpoint(self):
        # Warning: Currently, to align the training process and metrics of colocate,
        # we use current_param_version instead of global step.
        # This can be logically aligned with the original self.global_steps of colocate
        # and is used for metrics and ckpt. which means that the parameter synchronization
        # from trainer to rollouter will increase by 1 each time.

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )

        print(f"[FullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", "actor"
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "[FullyAsyncTrainer] Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    async def load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"[FullyAsyncTrainer] Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.current_param_version = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
        self.last_ckpt_version = self.current_param_version
        print(
            f"[FullyAsyncTrainer] Setting global step to {self.global_steps}, "
            f"current_param_version to {self.current_param_version}"
        )
        print(f"[FullyAsyncTrainer] Resuming from  {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        if self.colocate_checkpoint_manager:
            await self.colocate_checkpoint_manager.update_weights(self.current_param_version)
            await self.colocate_checkpoint_manager.sleep_replicas()

        return self.current_param_version

    def _collect_metrics_from_samples(self, batch, metrics):
        """
        Collect metrics from samples
        """
        if hasattr(batch, "meta_info") and batch.meta_info:
            trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
            # 某些样本在未触发参数版本打点时会产生 None；统计陈旧轨迹时需要跳过这些值。
            stale_traj_count = sum(
                1
                for v in trajectory_param_versions
                if v is not None and self.current_param_version - int(v) >= 1
            )
            self.stale_trajectory_processed += stale_traj_count
            metrics.update(
                {
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.current_param_version,
                }
            )
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async") or key.startswith("timing_s"):
                    metrics[key] = value
