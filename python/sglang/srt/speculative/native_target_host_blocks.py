"""Fixed-size exclusive host blocks for migrated Target request prefixes."""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _Block:
    owner: object
    used: int


@dataclass(frozen=True)
class HostBlockReservation:
    slots: np.ndarray
    updates: dict[int, _Block]
    new_blocks: list[int]


class TargetHostBlockPool:
    """Resource ownership is separate from lazily allocated pinned backing.

    All blocks save a prefix, from their first slot up to the committed used
    length. Block zero is unused so logical slot zero remains a sentinel.
    """

    def __init__(self, capacity: int, max_requests: int, block_size: int):
        if min(capacity, max_requests, block_size) <= 0:
            raise ValueError("Target host block geometry must be positive")
        self.capacity = capacity
        self.block_size = block_size
        # Reserve room for one partial block per request. Deferred-free blocks
        # draw from the same budget and are waited on only when capacity is tight.
        self.block_capacity = (capacity + block_size - 1) // block_size + max_requests
        self.slot_limit = (self.block_capacity + 1) * block_size
        self.live_tokens = 0
        self.retired_blocks = 0
        self._free = list(range(self.block_capacity, 0, -1))
        self._blocks: dict[int, _Block] = {}
        self._retired = deque()

    @property
    def live_blocks(self):
        return len(self._blocks)

    def _drain(self, required=0):
        while self._retired:
            fence, blocks = self._retired[0]
            if not fence.query():
                if len(self._free) >= required:
                    break
                # The batches all fence the same ordered Target copy stream.
                fence.synchronize()
            self._retired.popleft()
            self._free.extend(blocks)
            self.retired_blocks -= len(blocks)

    def reserve(self, appends) -> HostBlockReservation:
        """Reserve physical spans in request/position order without committing."""
        plans = []
        block_size = self.block_size
        needed_blocks, n = 0, 0
        owners = set()
        for owner, tail_slot, count in appends:
            if count <= 0 or owner in owners:
                raise RuntimeError("Expected one nonempty Target prefix append per owner")
            owners.add(owner)
            tail, offset = divmod(tail_slot, block_size)
            filled = offset + 1 if tail_slot else block_size
            if tail_slot and self._blocks.get(tail) != _Block(owner, filled):
                raise RuntimeError("Target host append must extend its own committed tail")
            fill = min(block_size - filled, count)
            needed_blocks += (count - fill + block_size - 1) // block_size
            n += count
            plans.append((owner, tail, filled, fill, count - fill))
        if not n or self.live_tokens + n > self.capacity:
            raise RuntimeError("Insufficient logical Target host KV capacity")
        self._drain(required=needed_blocks)
        if len(self._free) < needed_blocks:
            raise RuntimeError("Insufficient Target host block capacity")

        slots = np.empty(n, dtype=np.int64)
        updates = {}
        new_blocks = self._free[-needed_blocks:][::-1] if needed_blocks else []
        available = iter(new_blocks)
        cursor = 0
        for owner, tail, filled, fill, remaining in plans:
            if fill:
                start = tail * block_size + filled
                slots[cursor : cursor + fill] = np.arange(start, start + fill)
                updates[tail] = _Block(owner, filled + fill)
                cursor += fill
            while remaining:
                block = next(available)
                used = min(remaining, block_size)
                start = block * block_size
                slots[cursor : cursor + used] = np.arange(start, start + used)
                updates[block] = _Block(owner, used)
                remaining -= used
                cursor += used
        reservation = HostBlockReservation(slots, updates, new_blocks)
        if needed_blocks:
            del self._free[-needed_blocks:]
        return reservation

    def commit(self, reservation: HostBlockReservation):
        self._blocks.update(reservation.updates)
        self.live_tokens += len(reservation.slots)

    def rollback(self, reservation: HostBlockReservation):
        self._free.extend(reversed(reservation.new_blocks))

    def release(self, slots: np.ndarray, record_fence):
        """Retire entire live blocks. GPU candidate rollback has no CPU slots."""
        if not len(slots):
            return
        physical = np.sort(slots)
        ids = physical // self.block_size
        boundaries = np.flatnonzero(np.diff(ids)) + 1
        blocks = []
        for first, end in zip(np.r_[0, boundaries], np.r_[boundaries, len(slots)]):
            block = int(ids[first])
            old = self._blocks.get(block)
            if (old is None or end - first != old.used
                    or physical[first] != block * self.block_size
                    or physical[end - 1] != block * self.block_size + old.used - 1
                    or np.any(np.diff(physical[first:end]) != 1)):
                raise RuntimeError("Target host KV must release each owned block in full")
            blocks.append(block)
        fence = record_fence()
        for block in blocks:
            del self._blocks[block]
        self.live_tokens -= len(slots)
        if fence is None:
            self._free.extend(blocks)
        else:
            self._retired.append((fence, blocks))
            self.retired_blocks += len(blocks)
            self._drain()

    def finish_releases(self):
        # A caller resetting the backing must wait for every deferred read.
        self._drain(required=self.block_capacity)
