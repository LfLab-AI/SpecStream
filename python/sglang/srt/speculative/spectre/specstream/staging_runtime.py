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
    valid_tokens: torch.Tensor | None = None
    valid_lengths: tuple[int, ...] = ()


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
        self._host_buffers: list[torch.Tensor | None] = [None] * self.num_buffers
        self._host_views: list[torch.Tensor | None] = [None] * self.num_buffers
        self._valid_buffers: list[torch.Tensor | None] = [None] * self.num_buffers
        self._valid_host_buffers: list[torch.Tensor | None] = [None] * self.num_buffers
        self._ready_events = [
            torch.cuda.Event() if self.copy_stream is not None else None
            for _ in range(self.num_buffers)
        ]
        self._has_ready_event = [False] * self.num_buffers
        self._free_events = [
            torch.cuda.Event() if self.copy_stream is not None else None
            for _ in range(self.num_buffers)
        ]
        self._has_free_event = [False] * self.num_buffers

    def _wait_host_slot_writable(self, slot: int) -> None:
        """Do not overwrite a pinned pack buffer while its H2D is in flight."""

        if self.copy_stream is not None and self._has_ready_event[slot]:
            # The host pack buffer becomes writable as soon as H2D has consumed
            # it.  Waiting for the later compute/free event unnecessarily
            # serialized CPU packing with attention execution.
            self._ready_events[slot].synchronize()

    def _ensure_host_shape(
        self,
        slot: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not 0 <= slot < self.num_buffers:
            raise IndexError("invalid SpecStream staging slot")
        current = self._host_buffers[slot]
        required = 1
        for size in shape:
            required *= int(size)
        if current is None or current.dtype != dtype or current.numel() < required:
            try:
                current = torch.empty(
                    required,
                    dtype=dtype,
                    device="cpu",
                    pin_memory=bool(torch.cuda.is_available()),
                )
            except RuntimeError:
                current = torch.empty(required, dtype=dtype, device="cpu")
            self._host_buffers[slot] = current
        view = current[:required].view(shape)
        self._host_views[slot] = view
        return view

    def _ensure_valid_shape(
        self, slot: int, count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        host = self._valid_host_buffers[slot]
        device = self._valid_buffers[slot]
        if host is None or host.numel() < count:
            try:
                host = torch.empty(
                    count,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=bool(torch.cuda.is_available()),
                )
            except RuntimeError:
                host = torch.empty(count, dtype=torch.int32, device="cpu")
            self._valid_host_buffers[slot] = host
        if device is None or device.numel() < count:
            device = torch.empty(count, dtype=torch.int32, device=self.device)
            self._valid_buffers[slot] = device
        return host[:count], device[:count]

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

    def reserve_cohort_pack(self, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        """Preallocate reusable pinned pack and valid-length metadata buffers."""

        if len(shape) < 2 or int(shape[0]) < 1:
            raise ValueError("cohort reserve shape must begin with batch size")
        for slot in range(self.num_buffers):
            self._ensure_host_shape(slot, shape, dtype)
            self._ensure_valid_shape(slot, int(shape[0]))

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
                self._has_ready_event[slot] = True
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
                self._has_ready_event[slot] = True
        return StagingTransfer(
            slot,
            destination,
            total_bytes,
            submitted_ns,
            source_count=len(sources),
        )

    def submit_cohort_groups(
        self,
        source_groups: (
            list[list[torch.Tensor] | tuple[torch.Tensor, ...]]
            | tuple[list[torch.Tensor] | tuple[torch.Tensor, ...], ...]
        ),
        slot: int,
    ) -> StagingTransfer:
        """Pack one cohort chunk-group and issue one contiguous H2D copy.

        CPU History slabs are request-private and therefore not adjacent.  A
        reusable pinned pack buffer turns all slabs for ``B`` requests and
        several adjacent chunks into one ``[B,N,2,H,D]`` DMA.  Padding is not
        initialized: ``valid_tokens`` masks it in the cohort kernel.
        """

        if not source_groups or not any(source_groups):
            raise ValueError("SpecStream cohort requires at least one source")
        first = next(group[0] for group in source_groups if group)
        if first.device.type != "cpu" or not first.is_contiguous():
            raise ValueError("SpecStream cohort sources must be contiguous CPU tensors")
        trailing_shape = tuple(first.shape[1:])
        valid_lengths: list[int] = []
        source_count = 0
        for group in source_groups:
            length = 0
            for source in group:
                if source.device.type != "cpu" or not source.is_contiguous():
                    raise ValueError(
                        "SpecStream cohort sources must be contiguous CPU tensors"
                    )
                if (
                    source.dtype != first.dtype
                    or tuple(source.shape[1:]) != trailing_shape
                ):
                    raise ValueError("cohort chunks must have one KV geometry")
                length += int(source.shape[0])
                source_count += 1
            valid_lengths.append(length)

        self._wait_host_slot_writable(slot)
        max_tokens = max(valid_lengths)
        host = self._ensure_host_shape(
            slot,
            (len(source_groups), max_tokens, *trailing_shape),
            first.dtype,
        )
        for item_index, group in enumerate(source_groups):
            cursor = 0
            for source in group:
                length = int(source.shape[0])
                host[item_index, cursor : cursor + length].copy_(source)
                cursor += length

        valid_host, valid_device = self._ensure_valid_shape(slot, len(source_groups))
        for index, length in enumerate(valid_lengths):
            valid_host[index] = length

        destination = self._ensure_shape(slot, tuple(host.shape), host.dtype)
        submitted_ns = time.perf_counter_ns()
        if self.copy_stream is None:
            destination.copy_(host)
            valid_device.copy_(valid_host)
        else:
            with torch.cuda.stream(self.copy_stream):
                if self._has_free_event[slot]:
                    self.copy_stream.wait_event(self._free_events[slot])
                destination.copy_(host, non_blocking=bool(host.is_pinned()))
                valid_device.copy_(
                    valid_host,
                    non_blocking=bool(valid_host.is_pinned()),
                )
                self._ready_events[slot].record(self.copy_stream)
                self._has_ready_event[slot] = True
        return StagingTransfer(
            slot=slot,
            tensor=destination,
            nbytes=destination.nbytes + valid_device.nbytes,
            submitted_ns=submitted_ns,
            source_count=source_count,
            valid_tokens=valid_device,
            valid_lengths=tuple(valid_lengths),
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

    @property
    def allocated_host_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self._host_buffers if tensor is not None)
