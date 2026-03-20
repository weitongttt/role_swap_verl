"""
MQ 服务进程：挂在 fully_async_policy 下，使用与 fully_async_main 相同的 Hydra 配置流程。
"""

import socket
import threading
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Literal

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.trainer.main_ppo import get_ppo_ray_runtime_env

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


@ray.remote(num_cpus=1)
class GateController:
    """全局门控：控制哪个 trainer/rollouter 允许进行 get/put。"""

    def __init__(self):
        self.allow_put: dict[Side, bool] = {"A": False, "B": False}
        self.allow_get: dict[Side, bool] = {"A": False, "B": False}
        self._cond = threading.Condition()

    def get_phase(self) -> dict[str, dict[Side, bool]]:
        with self._cond:
            return {"allow_put": dict(self.allow_put), "allow_get": dict(self.allow_get)}

    def set_phase(self, *, put_side: Side, get_side: Side):
        with self._cond:
            for s in ("A", "B"):
                self.allow_put[s] = s == put_side
                self.allow_get[s] = s == get_side
            self._cond.notify_all()

    def wait_put(self, side: Side):
        with self._cond:
            while not self.allow_put[side]:
                self._cond.wait(timeout=1.0)

    def wait_get(self, side: Side):
        with self._cond:
            while not self.allow_get[side]:
                self._cond.wait(timeout=1.0)


@ray.remote(num_cpus=1)
class SwapHook:
    """
    接收“某侧 trainer 完成一次参数更新”的事件，并切换门控相位。

初始相位：
- A 任务从 rollouter 开始：允许 A_put
- B 任务从 trainer 开始：允许 B_get
    """

    def __init__(self, gate: Any):
        self.gate = gate
        self.active_put: Side = "A"
        self.active_get: Side = "B"
        self._lock = threading.Lock()

    def init(self):
        # Force initial phase: A_put + B_get (synchronously apply to avoid stale gate state)
        ray.get(self.gate.set_phase.remote(put_side=self.active_put, get_side=self.active_get))
        return {"put": self.active_put, "get": self.active_get}

    def on_param_update(self, trained_side: Side):
        with self._lock:
            if trained_side != self.active_get:
                return
            self.active_put = "B" if self.active_put == "A" else "A"
            self.active_get = "B" if self.active_get == "A" else "A"
            ray.get(self.gate.set_phase.remote(put_side=self.active_put, get_side=self.active_get))


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    print(f"[MQ_SERVER] host={socket.gethostname()} pid={ray.util.get_node_ip_address()}")
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    run_id = str(getattr(getattr(config, "swap_role", {}), "run_id", "default"))
    names = Names(run_id=run_id)

    # Ensure we connect with the expected namespace.
    if ray.is_initialized():
        ray.shutdown()

    default_runtime_env = get_ppo_ray_runtime_env()
    ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
    ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "namespace": "swap_role"})
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
    print(f"ray init kwargs: {ray_init_kwargs}")
    ray.init(**OmegaConf.to_container(ray_init_kwargs))

    max_queue_size = int(getattr(config, "swap_role", {}).get("max_queue_size", 20000))

    # Idempotent startup: if detached actors already exist, reuse them.
    try:
        mq = ray.get_actor(names.mq_actor, namespace="swap_role")
        gate = ray.get_actor(names.gate_actor, namespace="swap_role")
        hook = ray.get_actor(names.hook_actor, namespace="swap_role")
        print("[MQ_SERVER] detected existing detached actors; reuse.")
        # Defensive: ensure gate phase is initialized even on reuse.
        # This is idempotent and enforces initial phase (A_put + B_get).
        ray.get(hook.init.remote())
    except Exception:
        mq = MessageQueue.options(name=names.mq_actor, lifetime="detached").remote(
            config, max_queue_size=max_queue_size
        )
        _ = MessageQueueClient(mq)

        gate = GateController.options(name=names.gate_actor, lifetime="detached").remote()
        hook = SwapHook.options(name=names.hook_actor, lifetime="detached").remote(gate)
        ray.get(hook.init.remote())

    print(f"[MQ_SERVER] ready. actors: mq={names.mq_actor} gate={names.gate_actor} hook={names.hook_actor}")

    import time
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()

