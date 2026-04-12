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
Simplified TCP exchange channel for cross-cluster rollout merging (GAP-GRPO).

Design:
- 2 hash-grouped queues: one per trainer side (A and B).
- Rollouter push: (prompt_hash, pickled_sample) → stored in BOTH pending dicts.
- When a hash accumulates 2 samples → group moves to ready queue.
- Trainer pull: pops from its ready queue (blocks if empty).
- No alternating turns → no deadlock.

Protocol (wire format):
- Each message: 4-byte big-endian length prefix + pickle payload.
- Request:  {"op": str, "run_id": str, ...op-specific fields...}
- Response: {"ok": bool, "result": Any, "error": str | None}

Ops:
- push_grouped: add a sample payload keyed by prompt_hash
- pull_grouped: blocking pull of a completed group (list of pickled samples)
- stats:        queue statistics
"""

from __future__ import annotations

import asyncio
import pickle
import socket
import struct
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


# ─── Wire protocol helpers ─────────────────────────────────────────────────


def _pack(obj: Any) -> bytes:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return struct.pack(">I", len(data)) + data


async def _read_exactly(reader: asyncio.StreamReader, n: int) -> bytes:
    return await reader.readexactly(n)


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


# ─── Server state ──────────────────────────────────────────────────────────

EXPECTED_PER_HASH = 2  # Always 2 sides (A + B), hardcoded.


@dataclass
class _RunState:
    # Hash-indexed pending groups: hash → [payload_bytes, ...]
    pending_for_a: dict[str, list] = field(default_factory=dict)
    pending_for_b: dict[str, list] = field(default_factory=dict)
    # Ready groups (each item = list[payload_bytes], len == EXPECTED_PER_HASH)
    ready_for_a: deque = field(default_factory=deque)
    ready_for_b: deque = field(default_factory=deque)
    cond: asyncio.Condition = field(default=None)
    stats: dict[str, int] = field(default_factory=lambda: defaultdict(int))


# ─── Server ────────────────────────────────────────────────────────────────


class TcpExchangeServer:
    def __init__(self, *, host: str, port: int, **_kwargs):
        self.host = host
        self.port = port
        self._runs: dict[str, _RunState] = {}
        self._lock = asyncio.Lock()

    async def _get_run(self, run_id: str) -> _RunState:
        async with self._lock:
            st = self._runs.get(run_id)
            if st is not None:
                return st
            cond = asyncio.Condition(asyncio.Lock())
            st = _RunState(cond=cond)
            self._runs[run_id] = st
            return st

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                req = await _recv(reader)
                op = req.get("op")
                run_id = str(req.get("run_id", "default"))
                st = await self._get_run(run_id)

                # ── push_grouped ──────────────────────────────────────
                if op == "push_grouped":
                    prompt_hash = str(req.get("prompt_hash", ""))
                    payload = req.get("payload")
                    side = str(req.get("side", "?"))

                    async with st.cond:
                        if not prompt_hash:
                            # No hash → bypass grouping, put directly as single-sample group
                            st.ready_for_a.append([payload])
                            st.ready_for_b.append([payload])
                            st.stats["passthrough"] += 1
                            print(
                                f"[TCP_EXCHANGE] PUSH passthrough (no hash) side={side}",
                                flush=True,
                            )
                        else:
                            # Add payload to BOTH pending dicts
                            became_ready = False
                            for pending, ready in [
                                (st.pending_for_a, st.ready_for_a),
                                (st.pending_for_b, st.ready_for_b),
                            ]:
                                if prompt_hash not in pending:
                                    pending[prompt_hash] = []
                                pending[prompt_hash].append(payload)
                                if len(pending[prompt_hash]) >= EXPECTED_PER_HASH:
                                    group = pending.pop(prompt_hash)
                                    ready.append(group)
                                    became_ready = True

                            st.stats["pushes"] += 1

                            if became_ready:
                                st.stats["groups_formed"] = st.stats.get("groups_formed", 0) + 1
                                print(
                                    f"[TCP_EXCHANGE] PUSH READY hash={prompt_hash[:8]} "
                                    f"completed_by={side} "
                                    f"ready_a={len(st.ready_for_a)} "
                                    f"ready_b={len(st.ready_for_b)} "
                                    f"pending_a={len(st.pending_for_a)} "
                                    f"pending_b={len(st.pending_for_b)} "
                                    f"total_groups={st.stats.get('groups_formed', 0)}",
                                    flush=True,
                                )
                            else:
                                # Only log every 20th pending push to reduce noise
                                if st.stats["pushes"] % 20 == 0:
                                    print(
                                        f"[TCP_EXCHANGE] PUSH pending (summary) "
                                        f"total_pushes={st.stats['pushes']} "
                                        f"pending_a={len(st.pending_for_a)} "
                                        f"pending_b={len(st.pending_for_b)} "
                                        f"ready_a={len(st.ready_for_a)} "
                                        f"ready_b={len(st.ready_for_b)} "
                                        f"groups_formed={st.stats.get('groups_formed', 0)}",
                                        flush=True,
                                    )

                            # Warn if pending backlog is large
                            if len(st.pending_for_a) > 50 and st.stats["pushes"] % 50 == 0:
                                print(
                                    f"[TCP_EXCHANGE] WARNING pending_a={len(st.pending_for_a)} "
                                    f"(no matching pushes from other side yet)",
                                    flush=True,
                                )

                        st.cond.notify_all()

                    await _send(writer, {"ok": True, "result": True, "error": None})
                    continue

                # ── pull_grouped ──────────────────────────────────────
                if op == "pull_grouped":
                    side = str(req.get("side", "A")).upper()
                    print(
                        f"[TCP_EXCHANGE] PULL request from side={side} "
                        f"ready_{side.lower()}={len(st.ready_for_a if side == 'A' else st.ready_for_b)}",
                        flush=True,
                    )

                    async with st.cond:
                        ready = st.ready_for_a if side == "A" else st.ready_for_b
                        while len(ready) == 0:
                            await st.cond.wait()
                        group = ready.popleft()
                        st.stats[f"consumed_{side.lower()}"] += 1

                    print(
                        f"[TCP_EXCHANGE] PULL delivered side={side} "
                        f"group_size={len(group)} "
                        f"remaining={len(ready)} "
                        f"total_consumed_{side.lower()}={st.stats[f'consumed_{side.lower()}']}",
                        flush=True,
                    )
                    await _send(writer, {"ok": True, "result": (group, len(ready)), "error": None})
                    continue

                # ── stats ─────────────────────────────────────────────
                if op == "stats":
                    async with st.cond:
                        total_pending = sum(len(v) for v in st.pending_for_a.values())
                        res = {
                            "pending_hashes_a": len(st.pending_for_a),
                            "pending_hashes_b": len(st.pending_for_b),
                            "total_pending_samples": total_pending,
                            "ready_for_a": len(st.ready_for_a),
                            "ready_for_b": len(st.ready_for_b),
                            # queue_size only counts READY groups (consumed by trainer).
                            # pending samples must NOT count: they haven't matched yet,
                            # and pausing the rollouter would prevent matching → deadlock.
                            "queue_size": (len(st.ready_for_a) + len(st.ready_for_b)) * EXPECTED_PER_HASH,
                            **dict(st.stats),
                        }
                    await _send(writer, {"ok": True, "result": res, "error": None})
                    continue

                # Unknown op
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


# ─── Client ────────────────────────────────────────────────────────────────


class TcpExchangeClient:
    def __init__(self, *, host: str, port: int, run_id: str, side: str):
        if side not in ("A", "B"):
            raise ValueError(f"side must be A/B, got: {side}")
        self.host = host
        self.port = int(port)
        self.run_id = str(run_id)
        self.side = side

    # ── push (async, called by rollouter) ─────────────────────────────

    async def push_grouped_async(self, prompt_hash: str, payload: Any) -> bool:
        """Push a sample for hash-based grouping."""
        reader, writer = await asyncio.open_connection(self.host, self.port)
        try:
            await _send(writer, {
                "op": "push_grouped",
                "run_id": self.run_id,
                "prompt_hash": prompt_hash,
                "side": self.side,
                "payload": payload,
            })
            resp = await _recv(reader)
            if not resp.get("ok", False):
                raise RuntimeError(resp.get("error") or "push_grouped failed")
            return bool(resp.get("result"))
        finally:
            writer.close()
            await writer.wait_closed()

    # ── pull (sync, called by trainer) ────────────────────────────────

    def pull_grouped_sync(self) -> tuple[list, int]:
        """Pull a merged group. Blocks until ready. Returns (list[bytes], remaining)."""
        with socket.create_connection((self.host, self.port), timeout=5) as sock:
            sock.settimeout(None)  # blocking wait for group
            _send_sync(sock, {
                "op": "pull_grouped",
                "run_id": self.run_id,
                "side": self.side,
            })
            resp = _recv_sync(sock)
            if not resp.get("ok", False):
                raise RuntimeError(resp.get("error") or "pull_grouped failed")
            return resp.get("result")

    # ── stats ─────────────────────────────────────────────────────────

    def get_statistics_sync(self) -> dict[str, Any]:
        with socket.create_connection((self.host, self.port), timeout=10) as sock:
            _send_sync(sock, {"op": "stats", "run_id": self.run_id})
            resp = _recv_sync(sock)
            if not resp.get("ok", False):
                raise RuntimeError(resp.get("error") or "stats failed")
            return dict(resp.get("result") or {})

    # ── No-op stubs (gate removed, kept for old config compatibility) ─

    def gate_wait_put_sync(self) -> None:
        pass

    def gate_wait_get_sync(self) -> None:
        pass

    def on_param_update_sync(self) -> dict[str, Any]:
        return {}
