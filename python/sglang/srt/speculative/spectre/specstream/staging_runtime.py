from __future__ import annotations

from dataclasses import dataclass
import time

import torch


@dataclass(frozen=True)
class StagingTransfer:
    slot: int
    tensor: torch.Tensor
    nbytes: int
    submitted_ns: int
    source_count: int = 1


class StagingWindowPool:
    """Bounded copy/compute double-buffer state machine."""

    def __init__(self, num_buffers: int, device: torch.device | str) -> None:
        if num_buffers < 1:
            raise ValueError("num_buffers must be positive")
        self.num_buffers = int(num_buffers)
        self.device = torch.device(device)
        self.copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda" and torch.cuda.is_available()
            else None
        )
        self._buffers: list[torch.Tensor | None] = [None] * self.num_buffers
        self._views: list[torch.Tensor | None] = [None] * self.num_buffers
        self._ready_events = [
            torch.cuda.Event() if self.copy_stream is not None else None
            for _ in range(self.num_buffers)
        ]
        self._free_events = [
            torch.cuda.Event() if self.copy_stream is not None else None
            for _ in range(self.num_buffers)
        ]
        self._has_free_event = [False] * self.num_buffers

    def _ensure_shape(
        self,
        slot: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not 0 <= slot < self.num_buffers:
            raise IndexError("invalid SpecStream staging slot")
        current = self._buffers[slot]
        required = 1
        for size in shape:
            required *= int(size)
        if current is None or current.dtype != dtype or current.numel() < required:
            if self.copy_stream is not None:
                torch.cuda.current_stream(self.device).synchronize()
            current = torch.empty(required, dtype=dtype, device=self.device)
            self._buffers[slot] = current
        view = current[:required].view(shape)
        self._views[slot] = view
        return view

    def reserve(self, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        """Allocate every bounded slot before the first verification round."""

        for slot in range(self.num_buffers):
            self._ensure_shape(slot, shape, dtype)

    def submit(self, source: torch.Tensor, slot: int) -> StagingTransfer:
        if source.device.type != "cpu":
            raise ValueError("SpecStream H2D source must be a CPU tensor")
        if not source.is_contiguous():
            raise ValueError("SpecStream H2D source must be contiguous")
        destination = self._ensure_shape(slot, tuple(source.shape), source.dtype)
        submitted_ns = time.perf_counter_ns()
        if self.copy_stream is None:
            destination.copy_(source)
        else:
            with torch.cuda.stream(self.copy_stream):
                if self._has_free_event[slot]:
                    self.copy_stream.wait_event(self._free_events[slot])
                destination.copy_(source, non_blocking=bool(source.is_pinned()))
                self._ready_events[slot].record(self.copy_stream)
        return StagingTransfer(slot, destination, source.nbytes, submitted_ns)

    def submit_many(
        self, sources: list[torch.Tensor] | tuple[torch.Tensor, ...], slot: int
    ) -> StagingTransfer:
        """Enqueue several adjacent CPU chunks into one contiguous GPU window.

        The sources remain separately allocated pinned slabs.  Their copies are
        issued back-to-back on the copy stream, followed by a *single* ready
        event.  This removes per-chunk event, attention-launch and Python-loop
        overhead without introducing an extra CPU concatenation.
        """

        if not sources:
            raise ValueError("SpecStream submit_many requires at least one source")
        first = sources[0]
        if first.device.type != "cpu" or not first.is_contiguous():
            raise ValueError("SpecStream H2D sources must be contiguous CPU tensors")
        trailing_shape = tuple(first.shape[1:])
        total_tokens = 0
        total_bytes = 0
        for source in sources:
            if source.device.type != "cpu" or not source.is_contiguous():
                raise ValueError(
                    "SpecStream H2D sources must be contiguous CPU tensors"
                )
            if source.dtype != first.dtype or tuple(source.shape[1:]) != trailing_shape:
                raise ValueError("grouped SpecStream chunks must have one KV geometry")
            total_tokens += int(source.shape[0])
            total_bytes += int(source.nbytes)

        destination = self._ensure_shape(
            slot, (total_tokens, *trailing_shape), first.dtype
        )
        submitted_ns = time.perf_counter_ns()

        def copy_sources() -> None:
            cursor = 0
            for source in sources:
                length = int(source.shape[0])
                destination[cursor : cursor + length].copy_(
                    source,
                    non_blocking=bool(source.is_pinned()),
                )
                cursor += length

        if self.copy_stream is None:
            copy_sources()
        else:
            with torch.cuda.stream(self.copy_stream):
                if self._has_free_event[slot]:
                    self.copy_stream.wait_event(self._free_events[slot])
                copy_sources()
                self._ready_events[slot].record(self.copy_stream)
        return StagingTransfer(
            slot,
            destination,
            total_bytes,
            submitted_ns,
            source_count=len(sources),
        )

    def wait_ready(self, transfer: StagingTransfer) -> torch.Tensor:
        if self.copy_stream is not None:
            torch.cuda.current_stream(self.device).wait_event(
                self._ready_events[transfer.slot]
            )
        return transfer.tensor

    def mark_consumed(self, transfer: StagingTransfer) -> None:
        if self.copy_stream is not None:
            self._free_events[transfer.slot].record(
                torch.cuda.current_stream(self.device)
            )
            self._has_free_event[transfer.slot] = True

    @property
    def allocated_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self._buffers if tensor is not None)
