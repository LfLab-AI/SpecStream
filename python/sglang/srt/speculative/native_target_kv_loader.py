"""Pinned backing and two-slot GPU streaming for committed Target KV."""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from sglang.srt.speculative.spectre.specstream.staging_runtime import (
    StagingTransfer,
    StagingWindowPool,
)


TARGET_STAGE_BUFFERS = 2
DEFAULT_TARGET_STAGE_TOKENS = 8192
DEFAULT_TARGET_CHUNK_TOKENS = 2048


def native_target_stage_tokens() -> int:
    tokens = int(os.getenv("SPECSTREAM_NATIVE_TARGET_STAGE_TOKENS", DEFAULT_TARGET_STAGE_TOKENS))
    if tokens <= 0:
        raise ValueError("Target GPU staging must fit at least one KV token")
    return tokens


def native_target_stage_bytes(stage_tokens, kv_heads, head_dim, dtype) -> int:
    return TARGET_STAGE_BUFFERS * stage_tokens * 2 * kv_heads * head_dim * dtype.itemsize


@dataclass(frozen=True)
class _HostRun:
    slab: int
    first: int
    length: int


class NativeTargetHostHistory:
    """Allocate pinned slabs on first use. A logical slot has one stable host address."""

    def __init__(self, num_layers, capacity, kv_heads, head_dim, dtype, device, *, slab_tokens=1024):
        if num_layers < 1 or capacity < 1 or slab_tokens < 1:
            raise ValueError("Target host KV capacity and slab sizes must be positive")
        self.num_layers = num_layers
        self.capacity = capacity
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.slab_tokens = slab_tokens
        self.slabs: dict[int, torch.Tensor] = {}

    def runs(self, slots: np.ndarray) -> tuple[_HostRun, ...]:
        """Find contiguous backing intervals without changing attention order."""
        if len(slots) == 0:
            return ()
        if slots.min() <= 0 or slots.max() >= self.capacity:
            raise IndexError("Target host KV slot out of bounds")
        limits = np.flatnonzero((np.diff(slots) != 1) | (slots[1:] % self.slab_tokens == 0)) + 1
        starts = np.r_[0, limits]
        ends = np.r_[limits, len(slots)]
        return tuple(
            _HostRun(*divmod(int(slots[start]), self.slab_tokens), int(end - start))
            for start, end in zip(starts, ends)
        )

    def _backing(self, run: _HostRun, layer_offset: int, *, allocate: bool) -> torch.Tensor:
        slab = self.slabs.get(run.slab)
        if slab is None:
            if not allocate:
                raise RuntimeError("Target tried to load an unallocated host KV slab")
            count = min(self.slab_tokens, self.capacity - run.slab * self.slab_tokens)
            slab = torch.empty(
                (self.num_layers, count, 2, self.kv_heads, self.head_dim),
                dtype=self.dtype,
                device="cpu",
                pin_memory=self.device.type == "cuda",
            )
            self.slabs[run.slab] = slab
        return slab[layer_offset, run.first : run.first + run.length]

    def store(self, layer_offset: int, dst_slots: torch.Tensor, k_cpu: torch.Tensor, v_cpu: torch.Tensor) -> None:
        cursor = 0
        for run in self.runs(dst_slots.numpy()):
            destination = self._backing(run, layer_offset, allocate=True)
            destination[:, 0].copy_(k_cpu[cursor : cursor + run.length])
            destination[:, 1].copy_(v_cpu[cursor : cursor + run.length])
            cursor += run.length

    def layer_sources(self, layer_offset: int, slots: np.ndarray) -> tuple[torch.Tensor, ...]:
        return self.sources(layer_offset, self.runs(slots))

    def sources(self, layer_offset: int, runs: tuple[_HostRun, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(self._backing(run, layer_offset, allocate=False) for run in runs)


@dataclass(frozen=True)
class _TargetWindow:
    lengths: tuple[int, ...]
    runs: tuple[_HostRun, ...]


@dataclass(frozen=True)
class NativeTargetLoadPlan:
    windows: tuple[_TargetWindow, ...]


class NativeTargetKVLoader:
    """Copy each packed cohort once so all its queries can reuse the GPU window."""

    def __init__(self, history: NativeTargetHostHistory, device, *, stage_tokens=None,
                 chunk_tokens=None, track_verify_windows=False, host_blocks=False):
        self.history = history
        self.stage_tokens = native_target_stage_tokens() if stage_tokens is None else stage_tokens
        if self.stage_tokens < 1:
            raise ValueError("Target stage_tokens must be positive")
        chunk_tokens = DEFAULT_TARGET_CHUNK_TOKENS if chunk_tokens is None else chunk_tokens
        if chunk_tokens < 1:
            raise ValueError("Target cohort chunks must fit one KV token")
        self.chunk_tokens = min(chunk_tokens, self.stage_tokens)
        self.host_blocks = host_blocks
        if host_blocks and self.stage_tokens < history.slab_tokens:
            raise ValueError("Target staging window must fit a whole CPU KV block")
        self.pool = StagingWindowPool(
            TARGET_STAGE_BUFFERS, device, track_h2d_windows=track_verify_windows,
        )
        self.pool.reserve((self.stage_tokens, 2, history.kv_heads, history.head_dim), history.dtype)
        self.prefetched: tuple[NativeTargetLoadPlan, int, StagingTransfer] | None = None
        self.h2d_bytes = 0
        self.cross_layer_bytes = 0
        self.dma_count = 0
        self.h2d_submit_ns = 0
        self.h2d_chunks = 0  # Pinned source runs passed to the copy stream.
        self.copy_batches = 0  # Bounded windows, each published with one ready event.
        self.d2h_bytes = 0
        self.verify_timing_round = None

    def prepare(self, cpu_rows: list[np.ndarray]) -> NativeTargetLoadPlan:
        """Describe packed request-order windows with per-request causal metadata."""
        if self.host_blocks:
            return self._prepare_block_windows(cpu_rows)
        windows = []
        active = deque((i, 0) for i, row in enumerate(cpu_rows) if len(row))
        while active:
            remaining = self.stage_tokens
            intervals = {}
            while active and remaining:
                row_index, start = active.popleft()
                row_length = len(cpu_rows[row_index])
                count = min(remaining, self.chunk_tokens, row_length - start)
                end = start + count
                first, _ = intervals.get(row_index, (start, start))
                intervals[row_index] = (first, end)
                if end < row_length:
                    active.append((row_index, end))
                remaining -= count
            runs = []
            lengths = [0] * len(cpu_rows)
            for row_index, (first, end) in sorted(intervals.items()):
                runs.extend(self.history.runs(cpu_rows[row_index][first:end]))
                lengths[row_index] = end - first
            windows.append(_TargetWindow(tuple(lengths), tuple(runs)))
        return NativeTargetLoadPlan(tuple(windows))

    def _prepare_block_windows(self, cpu_rows: list[np.ndarray]) -> NativeTargetLoadPlan:
        # Keep cohort scheduling and per-request order. A window may have
        # spare space when the next block cannot fit without being split.
        blocks = [deque(self.history.runs(row)) for row in cpu_rows]
        active = deque(i for i, parts in enumerate(blocks) if parts)
        windows = []
        while active:
            remaining = self.stage_tokens
            by_request = {}
            lengths = [0] * len(cpu_rows)
            while active and remaining:
                i = active.popleft()
                parts = blocks[i]
                if parts[0].length > remaining:
                    active.appendleft(i)
                    break
                selected = by_request.setdefault(i, [])
                cohort_tokens = 0
                while parts and parts[0].length <= remaining:
                    length = parts[0].length
                    if cohort_tokens and cohort_tokens + length > self.chunk_tokens:
                        break
                    selected.append(parts.popleft())
                    cohort_tokens += length
                    remaining -= length
                lengths[i] += cohort_tokens
                if parts:
                    active.append(i)
            runs = tuple(run for i in sorted(by_request) for run in by_request[i])
            windows.append(_TargetWindow(tuple(lengths), runs))
        return NativeTargetLoadPlan(tuple(windows))

    def fence_host_reuse(self):
        # Discarding a prefetch only cancels its consumer, so both completed
        # and discarded prefetches need the copy-stream fence.
        if self.prefetched is not None:
            self.pool.discard(self.prefetched[2])
            self.prefetched = None
        if self.pool.copy_stream is None:
            return None
        fence = torch.cuda.Event()
        fence.record(self.pool.copy_stream)
        return fence

    @torch.inference_mode()
    def migrate(self, cache_pool, source: np.ndarray, destination: torch.Tensor) -> None:
        """Move every layer of a committed range, then allow its GPU slots to be reused."""
        if len(source) != len(destination):
            raise ValueError("Target KV migration source and destination must match")
        if not len(source):
            return
        if self.prefetched is not None:
            self.pool.discard(self.prefetched[2])
            self.prefetched = None
        device = self.pool.device
        source_gpu = torch.as_tensor(source, dtype=torch.int64, device=device)
        destination = destination.numpy()
        runs = [
            self.history.runs(destination[i : i + self.stage_tokens])
            for i in range(0, len(destination), self.stage_tokens)
        ]

        def copy_all() -> None:
            for layer in range(self.history.num_layers):
                key, value = cache_pool.get_kv_buffer(cache_pool.start_layer + layer)
                for group, indices in zip(runs, source_gpu.split(self.stage_tokens)):
                    # Borrow one of the fixed GPU windows. Its preceding attention
                    # consumers finish before the transfer stream starts this migration.
                    packed = self.pool._ensure_shape(
                        0, (len(indices), 2, self.history.kv_heads, self.history.head_dim), self.history.dtype,
                    )
                    torch.index_select(key, 0, indices, out=packed[:, 0])
                    torch.index_select(value, 0, indices, out=packed[:, 1])
                    cursor = 0
                    for run in group:
                        host = self.history._backing(run, layer, allocate=True)
                        host.copy_(packed[cursor : cursor + run.length], non_blocking=host.is_pinned())
                        cursor += run.length

        copy_stream = self.pool.copy_stream
        if copy_stream is None:
            copy_all()
        else:
            copy_stream.wait_stream(torch.cuda.current_stream(device))
            ready = torch.cuda.Event()
            with torch.cuda.stream(copy_stream):
                try:
                    copy_all()
                finally:
                    ready.record(copy_stream)
                    ready.synchronize()
        self.d2h_bytes += len(source) * self.history.num_layers * 2 * self.history.kv_heads * self.history.head_dim * self.history.dtype.itemsize

    def _submit(self, layer: int, window: _TargetWindow, slot: int, *, cross_layer: bool = False):
        submit_started = time.perf_counter_ns()
        transfer = self.pool.submit_many(
            self.history.sources(layer, window.runs), slot, round_id=self.verify_timing_round,
        )
        self.h2d_submit_ns += time.perf_counter_ns() - submit_started
        if self.verify_timing_round is not None and not transfer.h2d_window_id:
            raise RuntimeError("Target verify KV transfer has no CUDA timing events")
        self.h2d_bytes += transfer.nbytes
        self.dma_count += transfer.dma_count
        self.h2d_chunks += transfer.source_count
        self.copy_batches += 1
        if cross_layer:
            self.cross_layer_bytes += transfer.nbytes
        return transfer

    def _submit_next_layer(self, plan: NativeTargetLoadPlan, layer: int, slot: int) -> None:
        if layer + 1 < self.history.num_layers:
            transfer = self._submit(layer + 1, plan.windows[0], slot, cross_layer=True)
            self.prefetched = (plan, layer + 1, transfer)

    def _take_prefetch(self, layer: int, plan: NativeTargetLoadPlan) -> StagingTransfer | None:
        pending = self.prefetched
        self.prefetched = None
        if pending is not None:
            saved_plan, saved_layer, transfer = pending
            if saved_plan is plan and saved_layer == layer:
                return transfer
            self.pool.discard(transfer)
        return None

    def prime(self, layer: int, plan: NativeTargetLoadPlan) -> None:
        """Start the first H2D without adding a dependency to resident attention."""
        transfer = self._take_prefetch(layer, plan)
        if not plan.windows:
            return
        if transfer is None:
            transfer = self._submit(layer, plan.windows[0], 0)
        self.prefetched = (plan, layer, transfer)

    def windows(self, layer: int, plan: NativeTargetLoadPlan):
        """Yield a ready GPU window and release it after the caller enqueues attention."""
        first = self._take_prefetch(layer, plan)
        if not plan.windows:
            return
        if first is None:
            first = self._submit(layer, plan.windows[0], 0)
        pending = {0: first}
        count = len(plan.windows)
        if count > 1:
            pending[1] = self._submit(layer, plan.windows[1], 1 - first.slot)
        else:
            self._submit_next_layer(plan, layer, 1 - first.slot)
        consumed = 0
        try:
            for index, window in enumerate(plan.windows):
                transfer = pending.pop(index)
                packed = self.pool.wait_ready(transfer)
                try:
                    yield packed, window.lengths
                    consumed += 1
                finally:
                    self.pool.mark_consumed(transfer)
                following = index + TARGET_STAGE_BUFFERS
                if following < count:
                    pending[following] = self._submit(layer, plan.windows[following], transfer.slot)
                elif index == count - TARGET_STAGE_BUFFERS:
                    self._submit_next_layer(plan, layer, transfer.slot)
        finally:
            for transfer in pending.values():
                self.pool.discard(transfer)
            # An interrupted layer cannot hand its prefetched data to the next one.
            if consumed < count and self.prefetched is not None:
                self.pool.discard(self.prefetched[2])
                self.prefetched = None
