"""
Side 任务进程（A 或 B），挂在 fully_async_policy 下，用与 fully_async_main 相同的 Hydra 流程。
"""

import socket
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Literal

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.single_controller.ray import ResourcePoolManager
from verl.trainer.main_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.utils import Role
from verl.utils.fs import copy_to_local

Side = Literal["A", "B"]


@dataclass(frozen=True)
class Names:
    run_id: str

    @property
    def mq_actor(self) -> str:
        return f"swap_role_{self.run_id}_mq"

    @property
    def gate_actor(self) -> str:
        return f"swap_role_{self.run_id}_gate"

    @property
    def hook_actor(self) -> str:
        return f"swap_role_{self.run_id}_hook"


def _make_pool_manager(pool_name: str, roles: list[Role], gpus: int) -> ResourcePoolManager:
    spec = {pool_name: [int(gpus)]}
    mapping = {role: pool_name for role in roles}
    return ResourcePoolManager(resource_pool_spec=spec, mapping=mapping)


class GateMessageQueueClient:
    """门控包装：不改 trainer/rollouter 逻辑，只在 put/get 时阻塞等待允许。"""

    def __init__(self, inner: MessageQueueClient, gate: Any, side: Side):
        self.inner = inner
        self.gate = gate
        self.side = side

    async def put_sample(self, sample: Any) -> bool:
        await self.gate.wait_put.remote(self.side)
        return await self.inner.put_sample(sample)

    async def put_validate(self, data: Any) -> bool:
        return await self.inner.put_validate(data)

    def get_validate_sync(self) -> Any | None:
        return self.inner.get_validate_sync()

    async def get_sample(self) -> Any | None:
        await self.gate.wait_get.remote(self.side)
        return await self.inner.get_sample()

    def get_sample_sync(self) -> Any | None:
        ray.get(self.gate.wait_get.remote(self.side))
        return self.inner.get_sample_sync()

    async def get_queue_size(self) -> int:
        return await self.inner.get_queue_size()

    async def get_statistics(self) -> dict:
        return await self.inner.get_statistics()

    def get_statistics_sync(self) -> dict:
        return self.inner.get_statistics_sync()

    async def clear_queue(self):
        await self.inner.clear_queue()

    async def shutdown(self):
        await self.inner.shutdown()

    async def get_memory_usage(self) -> dict:
        return await self.inner.get_memory_usage()


@ray.remote(num_cpus=1)
class RollouterProxy:
    """给 trainer 用的 rollouter 代理：截获 reset_staleness 作为 swap 触发。"""

    def __init__(self, real_rollouter: Any, hook: Any, side: Side):
        self.real = real_rollouter
        self.hook = hook
        self.side = side

    def get_replicas(self):
        return ray.get(self.real.get_replicas.remote())

    def do_validate(self):
        return ray.get(self.real.do_validate.remote())

    def save_checkpoint(self, local_global_step_folder: str):
        return ray.get(self.real.save_checkpoint.remote(local_global_step_folder))

    def reset_staleness(self):
        ret = ray.get(self.real.reset_staleness.remote())
        ray.get(self.hook.on_param_update.remote(self.side))
        return ret


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    side: Side = str(config.swap_role.side).upper()
    assert side in ("A", "B"), f"swap_role.side must be A/B, got {side}"

    print(f"[SIDE_TASK {side}] host={socket.gethostname()} pid={ray.util.get_node_ip_address()}")
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Ensure we connect with the expected namespace.
    if ray.is_initialized():
        ray.shutdown()

    default_runtime_env = get_ppo_ray_runtime_env()
    ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
    # Must match MQ server namespace to access named detached actors.
    ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "namespace": "swap_role"})
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
    print(f"ray init kwargs: {ray_init_kwargs}")
    ray.init(**OmegaConf.to_container(ray_init_kwargs))

    run_id = str(getattr(config.swap_role, "run_id", "default"))
    names = Names(run_id=run_id)
    mq = ray.get_actor(names.mq_actor, namespace="swap_role")
    gate = ray.get_actor(names.gate_actor, namespace="swap_role")
    hook = ray.get_actor(names.hook_actor, namespace="swap_role")

    # Defensive re-init: when MQ server reuses detached actors, the gate phase may not be initialized
    # (e.g. allow_get may remain False for both sides), causing both trainers to block forever.
    # Calling init() is idempotent and enforces the intended initial phase: A_put + B_get.
    try:
        ray.get(hook.init.remote())
        print(f"[SIDE_TASK {side}] ensured gate phase initialized via hook.init()")
    except Exception as e:
        print(f"[SIDE_TASK {side}] hook.init() failed/unsupported, continue: {e}")

    local_path = copy_to_local(
        config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
    )
    from verl.utils import hf_processor, hf_tokenizer

    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    # 资源池四分离：A/B 各自 trainer/rollouter 的 GPU 数从 config 读
    if side == "A":
        tr_gpus = int(config.swap_role.a_trainer_gpus)
        ro_gpus = int(config.swap_role.a_rollout_gpus)
        tr_pool_name = "A_trainer_pool"
        ro_pool_name = "A_rollout_pool"
    else:
        tr_gpus = int(config.swap_role.b_trainer_gpus)
        ro_gpus = int(config.swap_role.b_rollout_gpus)
        tr_pool_name = "B_trainer_pool"
        ro_pool_name = "B_rollout_pool"

    from verl.experimental.separation.utils import create_role_worker_mapping

    role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
    trainer_role_mapping = {r: wc for r, wc in role_worker_mapping.items() if r != Role.Rollout}

    trainer_roles = [Role.Actor, Role.Critic, Role.RefPolicy, Role.RewardModel]
    rpm_tr = _make_pool_manager(tr_pool_name, trainer_roles, tr_gpus)
    rpm_ro = _make_pool_manager(ro_pool_name, [Role.Rollout], ro_gpus)

    trainer = FullyAsyncTrainer.remote(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=trainer_role_mapping,
        resource_pool_manager=rpm_tr,
        ray_worker_group_cls=ray_worker_group_cls,
        processor=processor,
        device_name=config.trainer.device,
    )
    rollouter = FullyAsyncRollouter.remote(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=None,
        resource_pool_manager=rpm_ro,
        ray_worker_group_cls=ray_worker_group_cls,
        processor=processor,
        device_name=config.trainer.device,
    )

    ray.get([trainer.init_workers.remote(), rollouter.init_workers.remote()])
    ray.get(rollouter.set_max_required_samples.remote())

    base_client = MessageQueueClient(mq)
    gated_client = GateMessageQueueClient(base_client, gate=gate, side=side)
    ray.get([trainer.set_message_queue_client.remote(gated_client), rollouter.set_message_queue_client.remote(gated_client)])

    proxy = RollouterProxy.remote(rollouter, hook=hook, side=side)
    ray.get(trainer.set_rollouter.remote(proxy))

    # resume / initial sync
    ray.get([trainer.load_checkpoint.remote(), rollouter.load_checkpoint.remote()])
    ray.get(trainer._fit_update_weights.remote())

    print(f"[SIDE_TASK {side}] start running trainer+rollouter")
    futs = [trainer.fit.remote(), rollouter.fit.remote()]
    done, pending = ray.wait(futs, num_returns=1, timeout=None)
    for f in pending:
        ray.cancel(f)
    ray.get(done[0])


if __name__ == "__main__":
    main()

