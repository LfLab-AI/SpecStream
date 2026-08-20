from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
import logging
from typing import Iterator

import torch

logger = logging.getLogger(__name__)


@dataclass
class _PackedSlab:
    tensor: torch.Tensor
    abs_start: int
    used_tokens: int
    capacity_tokens: int
    nbytes: int
    pinned: bool
    ready_event: torch.cuda.Event | None = None
    pending_sources: list[torch.Tensor] = field(default_factory=list)


@dataclass(frozen=True)
class PackedLayerHistoryChunk:
    slab_id: int
    abs_start: int
    length: int
    tensor: torch.Tensor
    pinned: bool


@dataclass
class SealTicket:
    block_ids: list[int]
    event: torch.cuda.Event | None
    pending_sources: list[torch.Tensor]
    dependency_event: torch.cuda.Event | None = None
    completed: bool = False

    def is_ready(self) -> bool:
        """Return without synchronizing the CPU scheduler thread."""

        return self.completed or self.event is None or bool(self.event.query())

    def wait_safe_to_free(self) -> None:
        if self.event is not None:
            self.event.synchronize()
        self.pending_sources.clear()
        self.dependency_event = None
        self.completed = True


class CPUHistoryStore:
    """Rank-local, layer-major packed CPU History store.

    Each layer/chunk is laid out as ``[token, 2(K/V), Hkv, D]`` so verification
    can issue one contiguous H2D copy for a logical History chunk.
    """

    def __init__(
        self,
        *,
        max_memory_bytes: int,
        chunk_tokens: int,
        layer_ids: tuple[int, ...],
        kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if max_memory_bytes < 1 or chunk_tokens < 1:
            raise ValueError("CPU History budget and chunk size must be positive")
        if not layer_ids or kv_heads < 1 or head_dim < 1:
            raise ValueError("invalid Target KV geometry")
        self.max_memory_bytes = int(max_memory_bytes)
        self.chunk_tokens = int(chunk_tokens)
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.device = torch.device(device)
        # D2H sealing is deliberately isolated from the Target compute stream.
        # A producer event below orders KV writes before this stream consumes
        # them, while the scheduler can immediately continue the next round.
        self.d2h_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda" and torch.cuda.is_available()
            else None
        )
        self._next_id = count()
        self._slabs: dict[int, _PackedSlab] = {}
        self._request_slabs: dict[str, list[int]] = {}
        self.bytes_reserved = 0
        self.bytes_used = 0

    def _allocate_slab(self, rid: str, abs_start: int) -> tuple[int, _PackedSlab]:
        shape = (
            len(self.layer_ids),
            self.chunk_tokens,
            2,
            self.kv_heads,
            self.head_dim,
        )
        nbytes = _numel(shape) * torch.empty((), dtype=self.dtype).element_size()
        if self.bytes_reserved + nbytes > self.max_memory_bytes:
            raise MemoryError(
                "SpecStream CPU History budget exceeded: "
                f"need {nbytes} bytes, available "
                f"{self.max_memory_bytes - self.bytes_reserved}"
            )
        pinned = bool(torch.cuda.is_available())
        try:
            tensor = torch.empty(
                shape, dtype=self.dtype, device="cpu", pin_memory=pinned
            )
        except RuntimeError:
            pinned = False
            tensor = torch.empty(shape, dtype=self.dtype, device="cpu")
            logger.warning(
                "SpecStream could not allocate pinned CPU History; H2D copies "
                "will use the pageable fallback"
            )
        slab_id = next(self._next_id)
        slab = _PackedSlab(
            tensor=tensor,
            abs_start=abs_start,
            used_tokens=0,
            capacity_tokens=self.chunk_tokens,
            nbytes=nbytes,
            pinned=pinned,
        )
        self._slabs[slab_id] = slab
        self._request_slabs.setdefault(rid, []).append(slab_id)
        self.bytes_reserved += nbytes
        return slab_id, slab

    def seal_slots_async(
        self,
        *,
        rid: str,
        abs_start: int,
        slots: torch.Tensor,
        token_to_kv_pool,
    ) -> SealTicket:
        if slots.ndim != 1:
            raise ValueError("sealed KV slots must be a one-dimensional tensor")
        if slots.numel() == 0:
            return SealTicket([], None, [])
        if slots.device.type != self.device.type:
            slots = slots.to(self.device)

        block_ids: list[int] = []
        slabs_to_fill: list[tuple[_PackedSlab, int, int]] = []
        cursor = 0
        total = int(slots.numel())
        while cursor < total:
            take = min(self.chunk_tokens, total - cursor)
            slab_id, slab = self._allocate_slab(rid, abs_start + cursor)
            block_ids.append(slab_id)
            slabs_to_fill.append((slab, cursor, take))
            cursor += take

        pending_sources: list[torch.Tensor] = []
        dependency_event = None
        final_event = None
        use_async = bool(
            self.d2h_stream is not None
            and slots.is_cuda
            and all(slab.pinned for slab, _, _ in slabs_to_fill)
        )

        def enqueue_copies(*, non_blocking: bool) -> None:
            for slab, source_offset, take in slabs_to_fill:
                slab_slots = slots[source_offset : source_offset + take].long()
                for layer_offset, layer_id in enumerate(self.layer_ids):
                    key = token_to_kv_pool.get_key_buffer(layer_id).index_select(
                        0, slab_slots
                    )
                    value = token_to_kv_pool.get_value_buffer(layer_id).index_select(
                        0, slab_slots
                    )
                    if key.shape != value.shape:
                        raise ValueError(
                            "SpecStream packed History currently requires matching "
                            "K/V shapes"
                        )
                    packed = torch.stack((key, value), dim=1).contiguous()
                    if packed.shape[1:] != (
                        2,
                        self.kv_heads,
                        self.head_dim,
                    ):
                        raise ValueError(
                            "Target KV geometry changed while sealing SpecStream "
                            "History"
                        )
                    slab.tensor[layer_offset, :take].copy_(
                        packed, non_blocking=non_blocking
                    )
                    if non_blocking:
                        # Keep the gather/pack temporaries alive until the D2H
                        # completion event is observed by complete_seal().
                        pending_sources.append(packed)

                slab.used_tokens = take
                self.bytes_used += (
                    take
                    * len(self.layer_ids)
                    * 2
                    * self.kv_heads
                    * self.head_dim
                    * torch.empty((), dtype=self.dtype).element_size()
                )

        if use_async:
            dependency_event = torch.cuda.Event()
            dependency_event.record(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.d2h_stream):
                self.d2h_stream.wait_event(dependency_event)
                enqueue_copies(non_blocking=True)
                final_event = torch.cuda.Event()
                final_event.record(self.d2h_stream)
        else:
            enqueue_copies(non_blocking=False)

        for slab, _, _ in slabs_to_fill:
            slab.ready_event = final_event
        if slabs_to_fill:
            slabs_to_fill[-1][0].pending_sources = pending_sources
        return SealTicket(
            block_ids,
            final_event,
            pending_sources,
            dependency_event=dependency_event,
        )

    def complete_seal(self, ticket: SealTicket, *, wait: bool = False) -> bool:
        """Retire a D2H ticket without blocking unless explicitly requested."""

        if ticket.completed:
            return True
        if ticket.event is not None:
            if wait:
                ticket.event.synchronize()
            elif not ticket.event.query():
                return False
        ticket.pending_sources.clear()
        ticket.dependency_event = None
        for block_id in ticket.block_ids:
            slab = self._slabs.get(block_id)
            if slab is not None:
                slab.ready_event = None
                slab.pending_sources.clear()
        ticket.event = None
        ticket.completed = True
        return True

    def iter_layer_chunks(
        self,
        rid: str,
        layer_id: int,
        *,
        history_end: int | None = None,
    ) -> Iterator[PackedLayerHistoryChunk]:
        try:
            layer_offset = self.layer_ids.index(int(layer_id))
        except ValueError as exc:
            raise IndexError(f"layer {layer_id} is outside SpecStream History") from exc
        expected_start = 0
        for slab_id in self._request_slabs.get(rid, ()):  # ordered prefix
            slab = self._slabs[slab_id]
            if slab.ready_event is not None:
                slab.ready_event.synchronize()
                slab.ready_event = None
                slab.pending_sources.clear()
            if slab.abs_start != expected_start:
                raise AssertionError(
                    f"non-contiguous CPU History for {rid}: expected {expected_start}, "
                    f"found {slab.abs_start}"
                )
            length = slab.used_tokens
            if history_end is not None:
                length = min(length, max(0, history_end - slab.abs_start))
            if length <= 0:
                break
            tensor = slab.tensor[layer_offset, :length]
            if not tensor.is_contiguous():
                raise RuntimeError("packed CPU History chunk is not contiguous")
            yield PackedLayerHistoryChunk(
                slab_id=slab_id,
                abs_start=slab.abs_start,
                length=length,
                tensor=tensor,
                pinned=slab.pinned,
            )
            expected_start = slab.abs_start + slab.used_tokens
            if history_end is not None and expected_start >= history_end:
                break

    def request_block_ids(self, rid: str) -> list[int]:
        return list(self._request_slabs.get(rid, ()))

    def release(self, rid: str) -> None:
        for slab_id in self._request_slabs.pop(rid, []):
            slab = self._slabs.pop(slab_id)
            if slab.ready_event is not None:
                slab.ready_event.synchronize()
            self.bytes_reserved -= slab.nbytes
            used_nbytes = (
                slab.used_tokens
                * len(self.layer_ids)
                * 2
                * self.kv_heads
                * self.head_dim
                * slab.tensor.element_size()
            )
            self.bytes_used -= used_nbytes

    def clear(self) -> None:
        for rid in list(self._request_slabs):
            self.release(rid)


def _numel(shape) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result
