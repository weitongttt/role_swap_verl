import asyncio
import os
import socket
import time
from pprint import pprint
from typing import Any

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl.experimental.fully_async_policy.fully_async_exchange_main import ExchangeAsMessageQueueClient
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.message_queue import BidirectionalExchangeClient, BidirectionalExchangeQueue
from verl.experimental.fully_async_policy.tcp_exchange import TcpExchangeClient
from verl.experimental.separation.utils import create_role_worker_mapping
from verl.single_controller.ray.base import ResourcePoolManager
from verl.trainer.ppo.utils import Role
from verl.utils.fs import copy_to_local


def _exchange_debug_enabled() -> bool:
    # 约定：VERL_EXCHANGE_DEBUG=1 时基本不刷屏；只有 >=2 才开启详细 MQ/gate debug。
    raw = os.environ.get("VERL_EXCHANGE_DEBUG", "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return False
    if raw in ("true", "yes", "on"):
        return True  # treat as verbose (>=2)
    try:
        return int(raw) >= 2
    except ValueError:
        return False


class _LoggingExchangeAsMQ(ExchangeAsMessageQueueClient):
    """当 VERL_EXCHANGE_DEBUG>=2 时打印 gate / pull 边界（不修改 fully_async_exchange_main）。"""

    async def put_sample(self, sample: Any) -> bool:
        ec = self.exchange_client
        side = getattr(ec, "side", "?")
        print(f"[EXCHANGE_DEBUG][MQ] put_sample ENTER side={side} gate={self.enable_gate}", flush=True)
        out = await super().put_sample(sample)
        print(f"[EXCHANGE_DEBUG][MQ] put_sample LEAVE side={side} ok={out}", flush=True)
        return out

    def get_sample_sync(self) -> Any | None:
        ec = self.exchange_client
        side = getattr(ec, "side", "?")
        print(f"[EXCHANGE_DEBUG][MQ] get_sample_sync ENTER side={side} gate={self.enable_gate}", flush=True)
        out = super().get_sample_sync()
        if isinstance(out, tuple) and len(out) >= 2:
            item, qlen = out[0], out[1]
            print(
                f"[EXCHANGE_DEBUG][MQ] get_sample_sync LEAVE side={side} qlen={qlen} item_is_none={item is None}",
                flush=True,
            )
        else:
            print(f"[EXCHANGE_DEBUG][MQ] get_sample_sync LEAVE side={side} repr={repr(out)[:200]}", flush=True)
        return out


@ray.remote(num_cpus=1)
class FullyAsyncIsolatedExchangeTaskRunner:
    """
    集群内：trainer/rollouter 共用同一资源池（main_ppo 风格本地参数同步）
    集群间：只通过 exchange queue 交换样本（A<->B）
    """

    def __init__(self):
        self.components: dict[str, Any] = {}

    def run(self, config):
        self._initialize_components(config)
        self._run_training_loop()

    def _create_colocated_resource_pool_manager(self, config, role_worker_mapping):
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {role: global_pool_id for role in role_worker_mapping}
        mapping[Role.Rollout] = global_pool_id
        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    def _apply_single_gpu_rollout_safety(self, config):
        """单卡共置场景的防 OOM 逻辑：不改训练脚本参数，仅在 isolated 共置路径收敛 rollout 并发。"""
        try:
            single_gpu = int(config.trainer.nnodes) == 1 and int(config.trainer.n_gpus_per_node) == 1
            colocated = bool(config.async_training.get("colocate_actor_rollout", False))
            cur = int(config.actor_rollout_ref.rollout.get("max_num_seqs", 1024))
            print(f"[ISOLATED MAIN][SAFETY] single_gpu={single_gpu} colocated={colocated} max_num_seqs={cur}", flush=True)
            if single_gpu and colocated:
                safe = min(cur, 64)
                if safe != cur:
                    with open_dict(config.actor_rollout_ref.rollout):
                        config.actor_rollout_ref.rollout.max_num_seqs = safe
                    print(
                        f"[ISOLATED MAIN][SAFETY] clamp actor_rollout_ref.rollout.max_num_seqs: {cur} -> {safe}",
                        flush=True,
                    )

                # 单卡共置下，训练反向更容易 OOM：收敛 actor token budget，避免 step1 前后反向炸显存。
                actor_tok = int(config.actor_rollout_ref.actor.get("ppo_max_token_len_per_gpu", 32768))
                actor_tok_safe = min(actor_tok, 8192)
                if actor_tok_safe != actor_tok:
                    with open_dict(config.actor_rollout_ref.actor):
                        config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu = actor_tok_safe
                    print(
                        f"[ISOLATED MAIN][SAFETY] clamp actor_rollout_ref.actor.ppo_max_token_len_per_gpu: {actor_tok} -> {actor_tok_safe}",
                        flush=True,
                    )

                # 与 run_sync_1gpu_test.sh / main_ppo 常用值对齐：默认 gpu_memory_utilization=0.4，不再压到 0.35
                cur_gpu_util = float(config.actor_rollout_ref.rollout.get("gpu_memory_utilization", 0.5))
                safe_gpu_util = min(cur_gpu_util, 0.4)
                if safe_gpu_util != cur_gpu_util:
                    with open_dict(config.actor_rollout_ref.rollout):
                        config.actor_rollout_ref.rollout.gpu_memory_utilization = safe_gpu_util
                    print(
                        f"[ISOLATED MAIN][SAFETY] clamp actor_rollout_ref.rollout.gpu_memory_utilization: "
                        f"{cur_gpu_util} -> {safe_gpu_util}",
                        flush=True,
                    )

                # 单卡共置：关闭 cudagraph 捕获，降低 vLLM 初始化与 trainer 权重同卡时的峰值显存。
                if not bool(config.actor_rollout_ref.rollout.get("enforce_eager", False)):
                    with open_dict(config.actor_rollout_ref.rollout):
                        config.actor_rollout_ref.rollout.enforce_eager = True
                    print("[ISOLATED MAIN][SAFETY] set actor_rollout_ref.rollout.enforce_eager=True", flush=True)
        except Exception as e:
            print(f"[ISOLATED MAIN][SAFETY] skip clamp due to: {e}", flush=True)

    def _initialize_components(self, config):
        print(f"[ISOLATED MAIN] hostname={socket.gethostname()} pid={os.getpid()}", flush=True)
        OmegaConf.resolve(config)
        self._apply_single_gpu_rollout_safety(config)
        pprint(OmegaConf.to_container(config, resolve=True))

        exchange_cfg = getattr(config, "exchange", {})
        side = str(getattr(exchange_cfg, "side", "A")).upper()
        mode = str(getattr(exchange_cfg, "mode", "both")).lower()
        run_id = str(getattr(exchange_cfg, "run_id", "default"))
        backend = str(getattr(exchange_cfg, "backend", "tcp")).lower()
        ex_host_preview = str(getattr(exchange_cfg, "host", "127.0.0.1"))
        ex_port_preview = int(getattr(exchange_cfg, "port", 18080))

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        resource_pool_manager = self._create_colocated_resource_pool_manager(config, role_worker_mapping)

        trainer_role_mapping = {
            role: worker_cls for role, worker_cls in role_worker_mapping.items() if role != Role.Rollout
        }
        trainer = FullyAsyncTrainer.remote(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            device_name=config.trainer.device,
        )
        rollouter = FullyAsyncRollouter.remote(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=None,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            device_name=config.trainer.device,
        )
        # 先 trainer 起 actor 与资源池，再把同一 RayWorkerGroup 交给 rollouter，
        # vLLM 走 init_hybrid 与训练进程共 GPU（单集群 1 张卡，对齐 main_ppo global_pool）。
        ray.get(trainer.init_workers.remote())
        actor_rollout_wg = ray.get(trainer.get_actor_rollout_wg.remote())
        ray.get(rollouter.init_workers.remote(actor_rollout_wg))
        ray.get(rollouter.set_max_required_samples.remote())

        total_train_steps = ray.get(rollouter.get_total_train_steps.remote())
        ray.get(trainer.set_total_train_steps.remote(total_train_steps))

        exchange_actor = None
        if backend == "ray":
            exchange_actor_name = f"exchange_{run_id}"
            namespace = ray.get_runtime_context().namespace
            try:
                exchange_actor = ray.get_actor(exchange_actor_name, namespace=namespace)
            except Exception:
                exchange_actor = BidirectionalExchangeQueue.options(
                    name=exchange_actor_name,
                    lifetime="detached",
                    namespace=namespace,
                ).remote(max_queue_size=int(getattr(exchange_cfg, "max_queue_size", 20000)))
            exchange_client = BidirectionalExchangeClient(exchange_actor, side=side)
        elif backend == "tcp":
            exchange_client = TcpExchangeClient(
                host=str(getattr(exchange_cfg, "host", "127.0.0.1")),
                port=int(getattr(exchange_cfg, "port", 18080)),
                run_id=run_id,
                side=side,
            )
            # 如果 run_id 在脚本重跑时复用（很多人会这么做），TCP server 会继承旧队列里的终止哨兵。
            # 这里在 side=A 启动时强制 reset_run，保证每次训练从“干净的 gate/队列状态”开始。
            if str(side).upper() == "A":
                try:
                    exchange_client.reset_run_sync()
                except Exception as e:
                    print(f"[ISOLATED MAIN] tcp exchange reset_run failed/unsupported: {e}", flush=True)
            try:
                st0 = exchange_client.get_statistics_sync()
                print(
                    f"[ISOLATED MAIN] TcpExchange stats @ init (this job's TCP view) "
                    f"a_to_b_size={st0.get('a_to_b_size')} b_to_a_size={st0.get('b_to_a_size')} "
                    f"active_put={st0.get('active_put')} active_get={st0.get('active_get')} "
                    f"allow_put={st0.get('allow_put')} allow_get={st0.get('allow_get')}",
                    flush=True,
                )
            except Exception as e:
                print(f"[ISOLATED MAIN] TcpExchange stats @ init failed: {e}", flush=True)
        else:
            raise ValueError(f"exchange.backend must be ray/tcp, got: {backend}")

        enable_gate = bool(getattr(exchange_cfg, "enable_gate", True))
        print(
            f"[ISOLATED MAIN] exchange summary: backend={backend} host={ex_host_preview} port={ex_port_preview} "
            f"run_id={run_id!r} side={side} enable_gate={enable_gate}. "
            f"调试: export VERL_EXCHANGE_DEBUG=1",
            flush=True,
        )
        if backend == "tcp" and ex_host_preview.lower() in ("127.0.0.1", "localhost", "::1"):
            print(
                "[ISOLATED MAIN] WARNING: exchange.host 为回环。若 A/B 在不同物理机，必须指向运行 exchange 服务端"
                "的可路由 IP（两端相同 host/port/run_id），否则会「一侧队列疯长、另一侧 pull 永远空」而不报错。",
                flush=True,
            )
        mq_cls = _LoggingExchangeAsMQ if _exchange_debug_enabled() else ExchangeAsMessageQueueClient
        mq_client = mq_cls(
            exchange_client,
            enable_gate=enable_gate,
        )
        ray.get(trainer.set_message_queue_client.remote(mq_client))
        ray.get(rollouter.set_message_queue_client.remote(mq_client))

        ray.get(trainer.load_checkpoint.remote())
        ray.get(rollouter.load_checkpoint.remote())

        # 与 fully_async_exchange_main 一致：gate 初相位为 A_put+B_get，必须在参数更新后 on_param_update 才能翻转；
        # 否则 A 侧 trainer 永久卡在 gate_wait_get，B 侧 rollouter 稍后会卡在 gate_wait_put。
        bootstrap_b_pause = bool(getattr(exchange_cfg, "bootstrap_b_pause_rollouter", True))
        if str(side).upper() == "B" and bootstrap_b_pause:
            try:
                ray.get(rollouter.bootstrap_pause.remote())
            except Exception as e:
                print(f"[ISOLATED MAIN] bootstrap_pause failed/unsupported: {e}", flush=True)

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
                    exchange_actor_handle=None,
                    host: str = "127.0.0.1",
                    port: int = 18080,
                ):
                    self.real = real
                    self.backend = backend
                    self.side = side
                    self.run_id = run_id
                    self._printed_on_param_update_once = False
                    self._client = None
                    if backend == "tcp":
                        self._client = TcpExchangeClient(host=host, port=int(port), run_id=run_id, side=side)
                    elif backend == "ray" and exchange_actor_handle is not None:
                        self._client = BidirectionalExchangeClient(exchange_actor_handle, side=side)

                def get_replicas(self):
                    return ray.get(self.real.get_replicas.remote())

                def do_validate(self):
                    return ray.get(self.real.do_validate.remote())

                def save_checkpoint(self, local_global_step_folder: str):
                    return ray.get(self.real.save_checkpoint.remote(local_global_step_folder))

                def bootstrap_pause(self):
                    # 透传：在 param_sync 前暂停 rollouter，避免 vLLM 同时生成导致显存峰值。
                    return ray.get(self.real.bootstrap_pause.remote())

                def reset_staleness(self):
                    try:
                        if self._client is not None and hasattr(self._client, "on_param_update_sync"):
                            # 先 flip gate，再等 rollouter 完成 staleness reset。
                            # 否则如果 reset_staleness 卡住，gate 永远翻不了，另一侧会直接超时。
                            out = self._client.on_param_update_sync()

                            # 只打印一次：用于确认 gate 确实被翻转（低噪声，且与 VERL_EXCHANGE_DEBUG 无关）
                            if not self._printed_on_param_update_once:
                                print(
                                    f"[ISOLATED MAIN][GATE] reset_staleness -> on_param_update_sync "
                                    f"side={self.side} out={out}",
                                    flush=True,
                                )
                                self._printed_on_param_update_once = True
                    except Exception as e:
                        print(f"[ISOLATED MAIN] on_param_update failed: {e}", flush=True)
                    ret = ray.get(self.real.reset_staleness.remote())
                    return ret

            if backend == "tcp":
                proxy = _RollouterProxy.remote(
                    rollouter,
                    backend=backend,
                    side=side,
                    run_id=run_id,
                    host=str(getattr(exchange_cfg, "host", "127.0.0.1")),
                    port=int(getattr(exchange_cfg, "port", 18080)),
                )
                ray.get(trainer.set_rollouter.remote(proxy))
            elif backend == "ray" and exchange_actor is not None:
                proxy = _RollouterProxy.remote(
                    rollouter,
                    backend=backend,
                    side=side,
                    run_id=run_id,
                    exchange_actor_handle=exchange_actor,
                )
                ray.get(trainer.set_rollouter.remote(proxy))
            else:
                ray.get(trainer.set_rollouter.remote(rollouter))
        else:
            ray.get(trainer.set_rollouter.remote(rollouter))
        # 不在此处强制 _fit_update_weights：共置+vLLM V1 下首轮同步会与 FSDP gather 叠峰导致 OOM；
        # 首轮对齐依赖同路径 checkpoint/HF 权重，训练循环内 fit_step 末尾会正常 sync。

        self.components["trainer"] = trainer
        self.components["rollouter"] = rollouter
        self.components["mq_client"] = mq_client
        self.components["side"] = side
        self.components["mode"] = mode
        # For gate-aware bootstrapping on side=A.
        self.components["enable_gate"] = enable_gate
        self.components["exchange_backend"] = backend
        self.components["exchange_client"] = exchange_client

    def _run_training_loop(self):
        side = self.components["side"]
        mode = self.components["mode"]
        enable_gate = bool(self.components.get("enable_gate", False))
        exchange_backend = str(self.components.get("exchange_backend", "tcp")).lower()
        exchange_client = self.components.get("exchange_client")
        futures = []
        future_to_label: dict[Any, str] = {}

        def _add_future(label: str, obj_ref: Any) -> None:
            futures.append(obj_ref)
            future_to_label[obj_ref] = label

        def _wait_for_allow_get_a(
            timeout_s: float = float(os.environ.get("VERL_GATE_WAIT_TIMEOUT_A", "1800")),
            poll_s: float = 1.0,
        ) -> None:
            if exchange_client is None or not hasattr(exchange_client, "get_statistics_sync"):
                print(
                    "[ISOLATED MAIN] gate wait skipped: exchange_client has no get_statistics_sync",
                    flush=True,
                )
                return
            deadline = time.time() + float(timeout_s)
            while True:
                st = exchange_client.get_statistics_sync()
                allow_get = st.get("allow_get", {}) or {}
                allow_get_a = bool(allow_get.get("A", False))
                if allow_get_a:
                    # 仅在 VERL_EXCHANGE_DEBUG>=2 下打印 gate 状态，避免刷屏
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
                            f"[ISOLATED MAIN][GATE] allow_get['A'] is True now. phase="
                            f"a_put={st.get('allow_put', {}).get('A')} b_put={st.get('allow_put', {}).get('B')}",
                            flush=True,
                        )
                    return
                if time.time() >= deadline:
                    raise TimeoutError(
                        f"[ISOLATED MAIN][GATE] timeout waiting allow_get['A']=True. allow_get={allow_get} "
                        f"allow_put={st.get('allow_put')} active_put={st.get('active_put')} active_get={st.get('active_get')}"
                    )
                time.sleep(poll_s)

        if mode == "train_only":
            _add_future("trainer", self.components["trainer"].fit.remote())
        else:
            if side == "A":
                # Side A: initial gate phase is A_put + B_get.
                # If we start trainer immediately under 1-GPU colocate + gate mode,
                # trainer may be queued and never reach _get_samples_from_queue().
                # So we start rollouter first (so A can put samples), and only start trainer
                # after exchange server flips phase to allow GET for side=A.
                _add_future("rollouter", self.components["rollouter"].fit.remote())
                if enable_gate and exchange_backend == "tcp":
                    print(
                        "[ISOLATED MAIN][GATE] side=A: waiting exchange allow_get['A']=True before starting trainer.fit",
                        flush=True,
                    )
                    _wait_for_allow_get_a()
                _add_future("trainer", self.components["trainer"].fit.remote())
            else:
                # Side B: start trainer first, optionally pause rollouter to avoid deadlock.
                _add_future("trainer", self.components["trainer"].fit.remote())
                _add_future("rollouter", self.components["rollouter"].fit.remote())

        try:
            while futures:
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)
                futures = remaining_futures
                for future in done_futures:
                    label = future_to_label.get(future, "unknown")
                    try:
                        ray.get(future)
                    except Exception as e:
                        print(f"[ISOLATED MAIN] future(label={label}) failed: {e!r}", flush=True)
                        raise
        finally:
            asyncio.run(self.components["mq_client"].clear_queue())


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")
    assert config.async_training.use_trainer_do_validate is False, "use_trainer_do_validate is not ready to use."
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    run_ppo(config, task_runner_class=FullyAsyncIsolatedExchangeTaskRunner)


if __name__ == "__main__":
    main()
