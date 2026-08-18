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

    def wait_safe_to_free(self) -> None:
        if self.event is not None:
            self.event.synchronize()
        self.pending_sources.clear()


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
        pending_sources: list[torch.Tensor] = []
        final_event = None
        cursor = 0
        total = int(slots.numel())
        while cursor < total:
            take = min(self.chunk_tokens, total - cursor)
            slab_id, slab = self._allocate_slab(rid, abs_start + cursor)
            slab_slots = slots[cursor : cursor + take].long()
            use_async = bool(slab.pinned and slab_slots.is_cuda)
            for layer_offset, layer_id in enumerate(self.layer_ids):
                key = token_to_kv_pool.get_key_buffer(layer_id).index_select(
                    0, slab_slots
                )
                value = token_to_kv_pool.get_value_buffer(layer_id).index_select(
                    0, slab_slots
                )
                if key.shape != value.shape:
                    raise ValueError(
                        "SpecStream packed History currently requires matching K/V shapes"
                    )
                packed = torch.stack((key, value), dim=1).contiguous()
                if packed.shape[1:] != (
                    2,
                    self.kv_heads,
                    self.head_dim,
                ):
                    raise ValueError(
                        "Target KV geometry changed while sealing SpecStream History"
                    )
                slab.tensor[layer_offset, :take].copy_(packed, non_blocking=use_async)
                if use_async:
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
            block_ids.append(slab_id)
            cursor += take

        if pending_sources:
            final_event = torch.cuda.Event()
            final_event.record(torch.cuda.current_stream(self.device))
            self._slabs[block_ids[-1]].ready_event = final_event
            self._slabs[block_ids[-1]].pending_sources = pending_sources
        return SealTicket(block_ids, final_event, pending_sources)

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
