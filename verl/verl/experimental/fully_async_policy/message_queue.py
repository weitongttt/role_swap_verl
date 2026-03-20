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
from collections import deque
from typing import Any

import ray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    """
    Simplified Ray-based asynchronous message queue for communication between Rollouter and Trainer
    """

    def __init__(self, config: DictConfig, max_queue_size: int = 1000):
        self.config = config
        if max_queue_size is None:
            raise ValueError(f"max_queue_size cannot be None, got: {max_queue_size}")
        self.max_queue_size = int(max_queue_size)
        self.queue = deque(maxlen=self.max_queue_size)

        self.val_queue = deque()

        # Asyncio for message handling
        self.running = True

        # async safe
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)

        # statistic message
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0

        print(f"[MessageQueue] initialized with max_queue_size={max_queue_size}")

    async def put_sample(self, sample: Any) -> bool:
        """
        Put a batch sample into the queue

        Args:
            sample: Sample data

        Returns:
            bool: Whether the sample was successfully put into the queue
        """
        async with self._lock:
            # If queue is full, remove the oldest sample (rarely happens)
            is_drop = False
            if len(self.queue) >= self.max_queue_size:
                self.queue.popleft()
                self.dropped_samples += 1
                is_drop = True
                logger.warning("Queue full, dropped sample")
            self.queue.append(sample)
            self.total_produced += 1

            # Notify waiting consumers
            self._consumer_condition.notify_all()

            if self.total_produced % 100 == 0:
                print(f"MessageQueue stats: produced={self.total_produced}, queue_size={len(self.queue)}")
            if is_drop:
                return False
            return True

    async def get_sample(self) -> Any | None:
        """
        Get a single sample from the queue, wait until one is available

        Returns:
            Any: Single sample data or None if queue is closed
        """
        async with self._lock:
            while len(self.queue) == 0 and self.running:
                await self._consumer_condition.wait()

            # If queue is closed and empty, return None
            if not self.running and len(self.queue) == 0:
                return None

            # Get one sample
            data = self.queue.popleft()
            self.total_consumed += 1
            return data, len(self.queue)

    async def get_queue_size(self) -> int:
        """Get current queue length"""
        async with self._lock:
            return len(self.queue)

    async def get_statistics(self) -> dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            return {
                "queue_size": len(self.queue),
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed,
                "dropped_samples": self.dropped_samples,
                "max_queue_size": self.max_queue_size,
            }

    async def clear_queue(self):
        """Clear the queue"""
        async with self._lock:
            cleared_count = len(self.queue)
            self.queue.clear()
            logger.info(f"Cleared {cleared_count} samples from queue")

    async def shutdown(self):
        """Shutdown the message queue"""
        async with self._lock:
            self.running = False
            # Notify all waiting coroutines so they can exit
            self._consumer_condition.notify_all()
        logger.info("MessageQueue shutdown")

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        async with self._lock:
            # Estimate memory usage of samples in queue
            import sys

            total_size = 0
            sample_count = len(self.queue)

            if sample_count > 0:
                # Estimate size of a single sample (simplified estimation)
                sample = list(self.queue)[0]
                try:
                    sample_size = sys.getsizeof(sample)
                    # Since we now store RolloutSample directly, estimate based on its components
                    if hasattr(sample, "original_batch_dict") and sample.original_batch_dict:
                        # Estimate batch data size
                        batch_data = sample.original_batch_dict.get("batch", {})
                        sample_size += len(batch_data) * 1000  # Roughly estimate 1KB per batch entry
                    if hasattr(sample, "agent_loop_output"):
                        # Estimate AgentLoopOutput size
                        sample_size += 5000  # Roughly estimate 5KB for AgentLoopOutput
                    total_size = sample_size * sample_count
                except Exception:
                    total_size = sample_count * 15000  # Roughly estimate 15KB per RolloutSample

            return {
                "queue_samples": sample_count,
                "estimated_memory_bytes": total_size,
                "estimated_memory_mb": total_size / (1024 * 1024),
            }

    async def put_validate(self, data):
        async with self._lock:
            self.val_queue.append(data)

    async def get_validate(self):
        async with self._lock:
            if self.val_queue:
                return self.val_queue.popleft()
            else:
                return None


class MessageQueueClient:
    """Asyncio-compatible MessageQueue client for communicating with MessageQueue Actor"""

    def __init__(self, queue_actor: Any):
        self.queue_actor = queue_actor

    async def put_sample(self, sample: Any) -> bool:
        """Put batch into queue (async)"""
        future = self.queue_actor.put_sample.remote(sample)
        return await asyncio.wrap_future(future.future())

    async def put_validate(self, data: Any) -> bool:
        future = self.queue_actor.put_validate.remote(data)
        return await asyncio.wrap_future(future.future())

    def get_validate_sync(self) -> Any | None:
        return ray.get(self.queue_actor.get_validate.remote())

    async def get_sample(self) -> Any | None:
        """Get single sample from queue, wait until one is available (async)"""
        future = self.queue_actor.get_sample.remote()
        return await asyncio.wrap_future(future.future())

    async def get_queue_size(self) -> int:
        """Get queue size (async)"""
        future = self.queue_actor.get_queue_size.remote()
        return await asyncio.wrap_future(future.future())

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics (async)"""
        future = self.queue_actor.get_statistics.remote()
        return await asyncio.wrap_future(future.future())

    async def clear_queue(self):
        """Clear queue (async)"""
        future = self.queue_actor.clear_queue.remote()
        await asyncio.wrap_future(future.future())

    async def shutdown(self):
        """Shutdown queue (async)"""
        future = self.queue_actor.shutdown.remote()
        await asyncio.wrap_future(future.future())

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics (async)"""
        future = self.queue_actor.get_memory_usage.remote()
        return await asyncio.wrap_future(future.future())

    def get_sample_sync(self) -> Any | None:
        """Get single sample from queue (sync - deprecated, use get_sample instead)"""
        return ray.get(self.queue_actor.get_sample.remote())

    def get_statistics_sync(self) -> dict[str, Any]:
        """Get statistics (sync - deprecated, use get_statistics instead)"""
        return ray.get(self.queue_actor.get_statistics.remote())


@ray.remote(num_cpus=2, max_concurrency=20)
class BidirectionalExchangeQueue:
    """
    A simple bidirectional exchange queue built on top of Ray.

    - Channel A -> B: for sending rollout samples generated by side A to side B.
    - Channel B -> A: for sending rollout samples generated by side B to side A.

    This is intentionally implemented as a separate actor instead of modifying
    the existing MessageQueue class to avoid breaking existing users.
    """

    def __init__(self, max_queue_size: int = 1000):
        if max_queue_size is None:
            raise ValueError(f"max_queue_size cannot be None, got: {max_queue_size}")
        self.max_queue_size = int(max_queue_size)

        # Use independent deques for each direction so that A/B traffic is isolated.
        self._a_to_b = deque(maxlen=self.max_queue_size)
        self._b_to_a = deque(maxlen=self.max_queue_size)

        # Asyncio primitives for async-safe operations.
        self._lock = asyncio.Lock()
        self._cond = asyncio.Condition(self._lock)

        # Phase gate: control which side is allowed to put/get at a time.
        # Initial phase matches intended bootstrap rhythm:
        # - allow_put["A"] = True  (A rollouter starts producing)
        # - allow_get["B"] = True  (B trainer starts consuming)
        self._allow_put: dict[str, bool] = {"A": False, "B": False}
        self._allow_get: dict[str, bool] = {"A": False, "B": False}
        self._active_put: str = "A"
        self._active_get: str = "B"
        self._allow_put["A"] = True
        self._allow_get["B"] = True

        # Basic statistics
        self._stats = {
            "a_to_b_produced": 0,
            "a_to_b_consumed": 0,
            "a_to_b_dropped": 0,
            "b_to_a_produced": 0,
            "b_to_a_consumed": 0,
            "b_to_a_dropped": 0,
            "phase_flips": 0,
        }

        print(f"[BidirectionalExchangeQueue] initialized with max_queue_size={max_queue_size}")

    async def wait_put(self, side: str) -> bool:
        side = str(side).upper()
        if side not in ("A", "B"):
            raise ValueError(f"side must be A/B, got: {side}")
        async with self._lock:
            while not self._allow_put[side]:
                await self._cond.wait()
            return True

    async def wait_get(self, side: str) -> bool:
        side = str(side).upper()
        if side not in ("A", "B"):
            raise ValueError(f"side must be A/B, got: {side}")
        async with self._lock:
            while not self._allow_get[side]:
                await self._cond.wait()
            return True

    async def on_param_update(self, trained_side: str) -> dict[str, str]:
        """
        Called when a side finishes a parameter update (i.e., trainer sync -> rollouter updated).
        Only flips phase if the trained_side is the current active_get side.
        """
        trained_side = str(trained_side).upper()
        if trained_side not in ("A", "B"):
            raise ValueError(f"trained_side must be A/B, got: {trained_side}")
        async with self._lock:
            if trained_side == self._active_get:
                self._active_put = "B" if self._active_put == "A" else "A"
                self._active_get = "B" if self._active_get == "A" else "A"
                for s in ("A", "B"):
                    self._allow_put[s] = s == self._active_put
                    self._allow_get[s] = s == self._active_get
                self._stats["phase_flips"] += 1
                self._cond.notify_all()
            return {"put": self._active_put, "get": self._active_get}

    async def _push(self, direction: str, sample: Any) -> bool:
        async with self._lock:
            if direction == "A2B":
                queue = self._a_to_b
                produced_key, dropped_key = "a_to_b_produced", "a_to_b_dropped"
            else:
                queue = self._b_to_a
                produced_key, dropped_key = "b_to_a_produced", "b_to_a_dropped"

            is_drop = False
            if len(queue) >= self.max_queue_size:
                queue.popleft()
                self._stats[dropped_key] += 1
                is_drop = True
                logger.warning("[%s] queue full, dropped sample", direction)

            queue.append(sample)
            self._stats[produced_key] += 1
            self._cond.notify_all()
            return not is_drop

    async def _pull(self, direction: str) -> tuple[Any | None, int]:
        async with self._lock:
            if direction == "A2B":
                queue = self._a_to_b
                consumed_key = "a_to_b_consumed"
            else:
                queue = self._b_to_a
                consumed_key = "b_to_a_consumed"

            while len(queue) == 0:
                await self._cond.wait()

            sample = queue.popleft()
            self._stats[consumed_key] += 1
            return sample, len(queue)

    # Public APIs for side A/B

    async def push_from_A(self, sample: Any) -> bool:
        """Push a sample from side A to side B."""
        return await self._push("A2B", sample)

    async def push_from_B(self, sample: Any) -> bool:
        """Push a sample from side B to side A."""
        return await self._push("B2A", sample)

    async def pull_for_A(self) -> tuple[Any | None, int]:
        """
        Pull a single sample destined for side A (i.e., produced by side B).

        Returns:
            (sample, remaining_queue_length)
        """
        return await self._pull("B2A")

    async def pull_for_B(self) -> tuple[Any | None, int]:
        """
        Pull a single sample destined for side B (i.e., produced by side A).

        Returns:
            (sample, remaining_queue_length)
        """
        return await self._pull("A2B")

    async def get_statistics(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "max_queue_size": self.max_queue_size,
                "a_to_b_size": len(self._a_to_b),
                "b_to_a_size": len(self._b_to_a),
                "allow_put": dict(self._allow_put),
                "allow_get": dict(self._allow_get),
                "active_put": self._active_put,
                "active_get": self._active_get,
                **self._stats,
            }


class BidirectionalExchangeClient:
    """
    Lightweight client wrapper for BidirectionalExchangeQueue.

    Usage:
        - side="A": use send_to_peer() to push A->B, recv_from_peer_sync() to pull B->A.
        - side="B": symmetric.
    """

    def __init__(self, queue_actor: Any, side: str):
        if side not in ("A", "B"):
            raise ValueError(f"side must be 'A' or 'B', got: {side}")
        self.queue_actor = queue_actor
        self.side = side

    def gate_wait_put_sync(self) -> None:
        ray.get(self.queue_actor.wait_put.remote(self.side))

    def gate_wait_get_sync(self) -> None:
        ray.get(self.queue_actor.wait_get.remote(self.side))

    def on_param_update_sync(self) -> dict[str, Any]:
        return dict(ray.get(self.queue_actor.on_param_update.remote(self.side)))

    async def send_to_peer(self, sample: Any) -> bool:
        if self.side == "A":
            fut = self.queue_actor.push_from_A.remote(sample)
        else:
            fut = self.queue_actor.push_from_B.remote(sample)
        return await asyncio.wrap_future(fut.future())

    async def recv_from_peer(self) -> tuple[Any | None, int]:
        if self.side == "A":
            fut = self.queue_actor.pull_for_A.remote()
        else:
            fut = self.queue_actor.pull_for_B.remote()
        return await asyncio.wrap_future(fut.future())

    def recv_from_peer_sync(self) -> tuple[Any | None, int]:
        if self.side == "A":
            return ray.get(self.queue_actor.pull_for_A.remote())
        return ray.get(self.queue_actor.pull_for_B.remote())

    def get_statistics_sync(self) -> dict[str, Any]:
        return ray.get(self.queue_actor.get_statistics.remote())
