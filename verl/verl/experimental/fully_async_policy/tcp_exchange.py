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
Tiny TCP exchange channel for strict multi-Ray isolation.

Design goals:
- A/B run in separate Ray clusters and strict CUDA_VISIBLE_DEVICES slices.
- Exchange lives outside Ray, so both clusters can talk to it.
- Two directed queues per run_id: A->B and B->A.
- Blocking pull semantics (like MQ get_sample_sync) to keep existing trainer loop behavior.

Protocol:
- Each request is a length-prefixed (4 bytes big-endian) pickle payload:
  dict: {"op": str, "run_id": str, "payload": Any | None}
  op in: "push_from_A", "push_from_B", "pull_for_A", "pull_for_B", "stats"

- Each response is also length-prefixed pickle:
  dict: {"ok": bool, "result": Any, "error": str | None}

For pull ops, server blocks until an item is available.
"""

from __future__ import annotations

import asyncio
import os
import pickle
import socket
import struct
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Literal


def _pack(obj: Any) -> bytes:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return struct.pack(">I", len(data)) + data


async def _read_exactly(reader: asyncio.StreamReader, n: int) -> bytes:
    data = await reader.readexactly(n)
    return data


async def _recv(reader: asyncio.StreamReader) -> Any:
    header = await _read_exactly(reader, 4)
    (length,) = struct.unpack(">I", header)
    payload = await _read_exactly(reader, length)
    return pickle.loads(payload)


async def _send(writer: asyncio.StreamWriter, obj: Any) -> None:
    writer.write(_pack(obj))
    await writer.drain()


def _recv_sync(sock: socket.socket) -> Any:
    def recvn(n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("socket closed")
            buf.extend(chunk)
        return bytes(buf)

    header = recvn(4)
    (length,) = struct.unpack(">I", header)
    payload = recvn(length)
    return pickle.loads(payload)


def _send_sync(sock: socket.socket, obj: Any) -> None:
    sock.sendall(_pack(obj))


def _tcp_exchange_debug(msg: str) -> None:
    dbg_raw = os.environ.get("VERL_EXCHANGE_DEBUG", "").strip().lower()
    if dbg_raw in ("", "0", "false", "no", "off"):
        return

    # 调试降噪分级：
    # - VERL_EXCHANGE_DEBUG=1：只打印「参数更新触发的相位翻转」(on_param_update_sync LEAVE)，避免 pull/push 轮询刷屏
    # - VERL_EXCHANGE_DEBUG>=2：打印较多门控/拉取结果，但仍过滤 op=stats 和 ENTER 噪声
    try:
        dbg_level = int(dbg_raw)
    except ValueError:
        dbg_level = 1

    lowered = msg.lower()
    if dbg_level <= 1:
        if "on_param_update_sync leave" not in lowered:
            return
    else:
        # 日志压缩：
        # 1) `op=stats` 会在训练过程中高频被轮询，刷屏极其严重 -> 直接过滤
        # 2) 大量 `... ENTER ...` 也会造成噪声 -> 只保留 LEAVE/结果类信息
        if "op=stats" in lowered:
            return
        if " enter" in lowered:
            return

    print(f"[TcpExchange] {msg}", flush=True)


@dataclass
class _RunState:
    max_queue_size: int
    a_to_b: deque
    b_to_a: deque
    cond: asyncio.Condition
    stats: dict[str, int]
    allow_put: dict[str, bool]
    allow_get: dict[str, bool]
    active_put: str
    active_get: str


class TcpExchangeServer:
    def __init__(self, *, host: str, port: int, max_queue_size: int = 20000):
        self.host = host
        self.port = port
        self.default_max_queue_size = int(max_queue_size)
        self._runs: dict[str, _RunState] = {}
        self._lock = asyncio.Lock()

    async def _get_run(self, run_id: str) -> _RunState:
        async with self._lock:
            st = self._runs.get(run_id)
            if st is not None:
                return st
            lock = asyncio.Lock()
            cond = asyncio.Condition(lock)
            st = _RunState(
                max_queue_size=self.default_max_queue_size,
                a_to_b=deque(maxlen=self.default_max_queue_size),
                b_to_a=deque(maxlen=self.default_max_queue_size),
                cond=cond,
                stats=defaultdict(int),
                allow_put={"A": False, "B": False},
                allow_get={"A": False, "B": False},
                active_put="A",
                active_get="B",
            )
            # Initial phase: A_put + B_get
            st.allow_put["A"] = True
            st.allow_put["B"] = False
            st.allow_get["A"] = False
            st.allow_get["B"] = True
            self._runs[run_id] = st
            return st

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                req = await _recv(reader)
                op = req.get("op")
                # 与客户端 TcpExchangeClient.run_id 对齐，避免文件/环境里的首尾空白导致同一物理集群却落到两套 _runs 队列
                run_id = str(req.get("run_id", "default")).strip()
                payload = req.get("payload", None)
                st = await self._get_run(run_id)

                if op == "stats":
                    async with st.cond:
                        res = {
                            "max_queue_size": st.max_queue_size,
                            "a_to_b_size": len(st.a_to_b),
                            "b_to_a_size": len(st.b_to_a),
                            "allow_put": dict(st.allow_put),
                            "allow_get": dict(st.allow_get),
                            "active_put": st.active_put,
                            "active_get": st.active_get,
                            **dict(st.stats),
                        }
                    await _send(writer, {"ok": True, "result": res, "error": None})
                    continue

                if op == "reset_run":
                    # 可重复启动训练时，run_id 可能保持不变；此时必须重置队列/门控状态
                    # 否则会继承旧 run 留下的 `None` 终止哨兵，导致对端立刻停止收样本。
                    async with st.cond:
                        st.a_to_b.clear()
                        st.b_to_a.clear()
                        st.stats.clear()
                        st.active_put = "A"
                        st.active_get = "B"
                        st.allow_put["A"] = True
                        st.allow_put["B"] = False
                        st.allow_get["A"] = False
                        st.allow_get["B"] = True
                        st.stats["phase_resets"] += 1
                        st.cond.notify_all()
                    await _send(
                        writer,
                        {"ok": True, "result": {"reset_done": True, "active_put": st.active_put, "active_get": st.active_get}},
                    )
                    continue

                if op == "wait_put":
                    side = str(req.get("side", "A")).upper()
                    if side not in ("A", "B"):
                        await _send(writer, {"ok": False, "result": None, "error": f"bad side: {side}"})
                        continue
                    async with st.cond:
                        while not st.allow_put[side]:
                            await st.cond.wait()
                    await _send(writer, {"ok": True, "result": True, "error": None})
                    continue

                if op == "wait_get":
                    side = str(req.get("side", "A")).upper()
                    if side not in ("A", "B"):
                        await _send(writer, {"ok": False, "result": None, "error": f"bad side: {side}"})
                        continue
                    async with st.cond:
                        while not st.allow_get[side]:
                            await st.cond.wait()
                    await _send(writer, {"ok": True, "result": True, "error": None})
                    continue

                if op == "on_param_update":
                    trained_side = str(req.get("side", "A")).upper()
                    if trained_side not in ("A", "B"):
                        await _send(writer, {"ok": False, "result": None, "error": f"bad side: {trained_side}"})
                        continue
                    async with st.cond:
                        # Only flip when the side that just trained is the current active_get
                        if trained_side == st.active_get:
                            st.active_put = "B" if st.active_put == "A" else "A"
                            st.active_get = "B" if st.active_get == "A" else "A"
                            for s in ("A", "B"):
                                st.allow_put[s] = s == st.active_put
                                st.allow_get[s] = s == st.active_get
                            st.stats["phase_flips"] += 1
                            st.cond.notify_all()
                    await _send(writer, {"ok": True, "result": {"put": st.active_put, "get": st.active_get}, "error": None})
                    continue

                if op in ("push_from_A", "push_from_B"):
                    async with st.cond:
                        if op == "push_from_A":
                            q = st.a_to_b
                            produced_k, dropped_k = "a_to_b_produced", "a_to_b_dropped"
                        else:
                            q = st.b_to_a
                            produced_k, dropped_k = "b_to_a_produced", "b_to_a_dropped"
                        dropped = False
                        if len(q) >= st.max_queue_size:
                            # deque(maxlen) auto-drop on append, but we count explicitly for clarity
                            q.popleft()
                            st.stats[dropped_k] += 1
                            dropped = True
                        q.append(payload)
                        st.stats[produced_k] += 1
                        st.cond.notify_all()
                    await _send(writer, {"ok": True, "result": (not dropped), "error": None})
                    continue

                if op in ("pull_for_A", "pull_for_B"):
                    # 禁止在持有 st.cond 时 await _send：大 payload 会长时间占锁，阻塞对侧 push，
                    # 多连接下易出现极端饥饿甚至表现为「pull 永远等不到数据」（与 push 抢同一把锁）。
                    async with st.cond:
                        if op == "pull_for_A":
                            q = st.b_to_a
                            consumed_k = "b_to_a_consumed"
                        else:
                            q = st.a_to_b
                            consumed_k = "a_to_b_consumed"
                        while len(q) == 0:
                            await st.cond.wait()
                        item = q.popleft()
                        st.stats[consumed_k] += 1
                        qlen_after = len(q)
                    await _send(writer, {"ok": True, "result": (item, qlen_after), "error": None})
                    continue

                await _send(writer, {"ok": False, "result": None, "error": f"unknown op: {op}"})
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return
        except Exception as e:
            try:
                await _send(writer, {"ok": False, "result": None, "error": repr(e)})
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def run_forever(self) -> None:
        server = await asyncio.start_server(self.handle, self.host, self.port)
        async with server:
            await server.serve_forever()


class TcpExchangeClient:
    def __init__(self, *, host: str, port: int, run_id: str, side: str):
        if side not in ("A", "B"):
            raise ValueError(f"side must be A/B, got: {side}")
        self.host = host
        self.port = int(port)
        self.run_id = str(run_id).strip()
        self.side = side

    def _op_push(self) -> str:
        return "push_from_A" if self.side == "A" else "push_from_B"

    def _op_pull(self) -> str:
        return "pull_for_A" if self.side == "A" else "pull_for_B"

    def request_sync(self, op: str, payload: Any | None = None) -> Any:
        """
        Sync request helper.

        NOTE:
        - Some ops are intentionally *blocking* (wait_put/wait_get/pull_*). For those, we must
          not keep a short socket timeout for recv(), otherwise the client will raise
          TimeoutError during normal operation.
        - We still keep a short connect timeout so a dead server fails fast.
        """
        blocking_ops = {"wait_put", "wait_get", "pull_for_A", "pull_for_B"}
        _tcp_exchange_debug(
            f"request_sync ENTER side={self.side} run_id={self.run_id} {self.host}:{self.port} op={op} "
            f"blocking={op in blocking_ops}"
        )
        connect_timeout = 5 if op in blocking_ops else 30
        with socket.create_connection((self.host, self.port), timeout=connect_timeout) as sock:
            if op in blocking_ops:
                # Allow indefinite blocking reads for gate/pull.
                sock.settimeout(None)
            _send_sync(sock, {"op": op, "run_id": self.run_id, "payload": payload})
            resp = _recv_sync(sock)
            if not resp.get("ok", False):
                raise RuntimeError(resp.get("error") or "unknown error")
            result = resp.get("result")
            if op in ("pull_for_A", "pull_for_B") and isinstance(result, tuple) and len(result) >= 2:
                _tcp_exchange_debug(
                    f"request_sync LEAVE op={op} side={self.side} qlen={result[1]} item_is_none={result[0] is None}"
                )
            else:
                _tcp_exchange_debug(f"request_sync LEAVE op={op} side={self.side}")
            return result

    async def request_async(self, op: str, payload: Any | None = None) -> Any:
        reader, writer = await asyncio.open_connection(self.host, self.port)
        try:
            await _send(writer, {"op": op, "run_id": self.run_id, "payload": payload})
            resp = await _recv(reader)
            if not resp.get("ok", False):
                raise RuntimeError(resp.get("error") or "unknown error")
            return resp.get("result")
        finally:
            writer.close()
            await writer.wait_closed()

    async def send_to_peer(self, sample: Any) -> bool:
        return bool(await self.request_async(self._op_push(), sample))

    async def recv_from_peer(self) -> tuple[Any | None, int]:
        item, qlen = await self.request_async(self._op_pull(), None)
        return item, int(qlen)

    def recv_from_peer_sync(self) -> tuple[Any | None, int]:
        item, qlen = self.request_sync(self._op_pull(), None)
        return item, int(qlen)

    def send_to_peer_sync(self, sample: Any) -> bool:
        """Synchronous push (avoids asyncio.run in plain sync training loops)."""
        return bool(self.request_sync(self._op_push(), sample))

    def recv_from_peer_sync_with_timeout(self, timeout_s: float) -> tuple[Any | None, int]:
        timeout_s = float(timeout_s)
        with socket.create_connection((self.host, self.port), timeout=5) as sock:
            _send_sync(sock, {"op": self._op_pull(), "run_id": self.run_id, "payload": None})
            # pull is server-blocking; timeout lets caller fail fast on deadlock.
            sock.settimeout(timeout_s)
            resp = _recv_sync(sock)
            if not resp.get("ok", False):
                raise RuntimeError(resp.get("error") or "unknown error")
            item, qlen = resp.get("result")
            return item, int(qlen)

    def get_statistics_sync(self) -> dict[str, Any]:
        return dict(self.request_sync("stats", None))

    def reset_run_sync(self) -> None:
        # 同步 reset：清空队列并回到初始相位 (A_put + B_get)。
        self.request_sync("reset_run", None)

    def gate_wait_put_sync(self) -> None:
        _tcp_exchange_debug(
            f"gate_wait_put_sync ENTER (blocking until phase allows PUT) "
            f"side={self.side} run_id={self.run_id} {self.host}:{self.port}"
        )
        with socket.create_connection((self.host, self.port), timeout=5) as sock:
            sock.settimeout(None)
            _send_sync(sock, {"op": "wait_put", "run_id": self.run_id, "side": self.side})
            resp = _recv_sync(sock)
            if not resp.get("ok", False):
                raise RuntimeError(resp.get("error") or "unknown error")
        _tcp_exchange_debug(f"gate_wait_put_sync LEAVE side={self.side} run_id={self.run_id}")

    def gate_wait_get_sync(self) -> None:
        _tcp_exchange_debug(
            f"gate_wait_get_sync ENTER (blocking until phase allows GET) "
            f"side={self.side} run_id={self.run_id} {self.host}:{self.port}"
        )
        with socket.create_connection((self.host, self.port), timeout=5) as sock:
            sock.settimeout(None)
            _send_sync(sock, {"op": "wait_get", "run_id": self.run_id, "side": self.side})
            resp = _recv_sync(sock)
            if not resp.get("ok", False):
                raise RuntimeError(resp.get("error") or "unknown error")
        _tcp_exchange_debug(f"gate_wait_get_sync LEAVE side={self.side} run_id={self.run_id}")

    def on_param_update_sync(self) -> dict[str, Any]:
        _tcp_exchange_debug(
            f"on_param_update_sync ENTER side={self.side} run_id={self.run_id} {self.host}:{self.port}"
        )
        with socket.create_connection((self.host, self.port), timeout=30) as sock:
            _send_sync(sock, {"op": "on_param_update", "run_id": self.run_id, "side": self.side})
            resp = _recv_sync(sock)
            if not resp.get("ok", False):
                raise RuntimeError(resp.get("error") or "unknown error")
            out = dict(resp.get("result") or {})
            _tcp_exchange_debug(f"on_param_update_sync LEAVE side={self.side} result={out}")
            return out

