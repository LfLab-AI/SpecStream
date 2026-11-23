"""GPU-first Target KV placement for native linear speculative verification."""

import os
import weakref
from copy import copy
from dataclasses import dataclass, field
from itertools import accumulate

import numpy as np
import torch

from sglang.srt.layers.attention.triton_ops.merge_state import merge_state_triton
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.native_draft_spill import DraftAttentionPlan, NativeDraftSpill
from sglang.srt.speculative.native_target_kv_loader import (
    NativeTargetHostHistory,
    NativeTargetKVLoader,
)
from sglang.srt.speculative.spectre.specstream.staging_runtime import StagingWindowPool


@dataclass
class TargetAttentionPlan:
    placement: DraftAttentionPlan
    transfer: object
    query_metadata: dict = field(default_factory=dict)


@dataclass
class _VerifyTransferTiming:
    pool: StagingWindowPool | None
    expected_windows: int
    expected_bytes: int
    windows: int = 0
    nbytes: int = 0
    copy_ms: float = 0.0
    exposed_wait_ms: float = 0.0


class NativeTargetSpill(NativeDraftSpill):
    """Target and Draft share logical IDs, but their physical slots are distinct.

    The CPU partition contains previously materialized Target KV. The new
    queries always write to a GPU page, so bottom-right causal masking on the
    GPU partition remains valid even if older keys were moved to the CPU.
    """

    def __init__(self, pool, logical_slots, rank, tp_threads):
        super().__init__(pool, logical_slots, rank, tp_threads)
        # A partial prefill should keep its earlier chunks on the GPU when
        # older committed rows can be moved instead.
        self.prefer_prompt_gpu = True
        self.requests = weakref.WeakValueDictionary()
        self.request_table = None
        self.loader = None
        self.retired_h2d = [0, 0, 0, 0]
        self.retired_d2h_bytes = 0
        self.gpu_online_merge_calls = 0
        self.gpu_cohort_calls = 0
        self._profile_verify = bool(os.getenv("SPECSTREAM_NATIVE_SEGMENT_TRACE"))
        self._verify_capture = None
        self._verify_timings: dict[int, _VerifyTransferTiming] = {}

    def register_requests(self, batch):
        # Retain lightweight owner references from the identity fast path.
        # No GPU table reads until there is an actual migration deficit.
        self.request_table = batch.req_to_token_pool.req_to_token
        for req in batch.reqs:
            if req.req_pool_idx is not None:
                self.requests[int(req.req_pool_idx)] = req

    def activate(self, allocator_free):
        low_free = allocator_free.cpu().numpy()
        low_free = low_free[low_free <= self.gpu_slots]
        live = np.ones(self.gpu_slots + 1, dtype=bool)
        live[0] = False
        live[low_free] = False
        slots = np.flatnonzero(live)
        self.gpu_map = np.zeros(self.logical_slots + 1, dtype=np.int64)
        self.gpu_map[slots] = slots
        self.cpu_map = np.zeros_like(self.gpu_map)
        self.prompt_priority = np.zeros(self.logical_slots + 1, dtype=np.uint8)
        self.gpu_free = low_free[::-1].tolist()
        capacity = self.logical_slots - self.gpu_slots
        self.cpu_free = list(range(capacity, 0, -1))
        self.history = NativeTargetHostHistory(
            self.pool.layer_num, capacity + 1, self.pool.head_num,
            self.pool.head_dim, self.pool.dtype, self.pool.device,
        )
        self.loader = NativeTargetKVLoader(
            self.history, self.pool.device, track_verify_windows=self._profile_verify,
        )
        if self._verify_capture is not None:
            self.loader.verify_timing_round = self._verify_capture[0]

    def reset(self):
        if self._verify_capture is not None:
            raise RuntimeError("Target KV placement reset during verification timing")
        self.retired_h2d = [self.h2d_bytes, self.h2d_cross_layer_bytes,
                            self.h2d_chunks, self.h2d_copy_batches]
        self.retired_d2h_bytes = self.d2h_bytes
        self.loader = None
        self.requests.clear()
        self.request_table = None
        super().reset()

    def begin_verify_timing(self, round_id: int) -> None:
        """Track only copies submitted by this Target verification forward."""
        if not self._profile_verify or self._verify_capture is not None or round_id in self._verify_timings:
            raise RuntimeError("Invalid Target verification timing scope")
        self._verify_capture = (round_id, self.h2d_copy_batches, self.h2d_bytes)
        if self.loader is not None:
            self.loader.verify_timing_round = round_id

    def end_verify_timing(self, round_id: int) -> None:
        capture = self._verify_capture
        if capture is None or capture[0] != round_id:
            raise RuntimeError("Mismatched Target verification timing scope")
        self._verify_capture = None
        if self.loader is not None:
            self.loader.verify_timing_round = None
        self._verify_timings[round_id] = _VerifyTransferTiming(
            self.loader.pool if self.loader is not None else None,
            self.h2d_copy_batches - capture[1],
            self.h2d_bytes - capture[2],
        )

    def take_verify_timing(self, round_id: int):
        """Read finalized CUDA timestamps without synchronizing the host."""
        timing = self._verify_timings[round_id]
        if timing.pool is not None and timing.expected_windows:
            for sample in timing.pool.drain_h2d_timing_samples():
                record = self._verify_timings[sample.round_id]
                if sample.target_wait_ms is None:
                    raise RuntimeError("Target verification KV timing is missing its consumer wait")
                record.windows += 1
                record.nbytes += sample.nbytes
                record.copy_ms += sample.elapsed_ms
                record.exposed_wait_ms += sample.target_wait_ms
            if timing.pool._h2d_tracking_error:
                raise RuntimeError(timing.pool._h2d_tracking_error)
        if timing.windows < timing.expected_windows:
            return None
        if timing.windows != timing.expected_windows or timing.nbytes != timing.expected_bytes:
            raise RuntimeError("Target verification KV timing disagrees with submitted transfer counts")
        del self._verify_timings[round_id]
        return {
            "target_h2d_dma_ms": timing.copy_ms,
            "target_kv_wait_ms": timing.exposed_wait_ms,
            "target_profiled_h2d_bytes": timing.nbytes,
            "target_h2d_windows": timing.windows,
        }

    @property
    def h2d_bytes(self):
        return self.retired_h2d[0] + (self.loader.h2d_bytes if self.loader else 0)

    @property
    def h2d_cross_layer_bytes(self):
        return self.retired_h2d[1] + (self.loader.cross_layer_bytes if self.loader else 0)

    @property
    def h2d_chunks(self):
        return self.retired_h2d[2] + (self.loader.h2d_chunks if self.loader else 0)

    @property
    def h2d_copy_batches(self):
        return self.retired_h2d[3] + (self.loader.copy_batches if self.loader else 0)

    @property
    def d2h_bytes(self):
        return self.retired_d2h_bytes + (self.loader.d2h_bytes if self.loader else 0)

    def _committed_victims(self, eligible, count):
        if self.request_table is None:
            raise RuntimeError("Target offload cannot find the live request owners")
        owners = sorted((index, int(req.kv_committed_len)) for index, req in self.requests.items()
                        if req.req_pool_idx == index and int(req.kv_committed_len) > 0)
        if not owners:
            raise RuntimeError("Target offload has no committed history to migrate")
        indices = torch.tensor([i for i, _ in owners], dtype=torch.int64, device=self.pool.device)
        table = self.request_table[indices, :max(n for _, n in owners)].cpu().numpy()
        by_priority = [[], [], []]
        for row, (_, length) in zip(table, owners):
            history = row[:length]
            on_cpu = self.cpu_map[history] > 0
            boundary = int(np.count_nonzero(on_cpu))
            if not on_cpu[:boundary].all():
                raise RuntimeError("Target CPU keys must be a fixed prefix of each request")
            candidate = history[boundary:]
            blocked = np.flatnonzero(~eligible[candidate])
            if len(blocked):
                candidate = candidate[:blocked[0]]
            if len(candidate):
                by_priority[int(self.prompt_priority[candidate[0]])].append(candidate)
        victims = []
        for priority in by_priority:
            for row in priority:
                take = min(len(row), count)
                victims.append(row[:take])
                count -= take
                if not count:
                    return np.concatenate(victims)
        raise RuntimeError(f"Target KV pool is short of {count} eligible committed prefix slots")

    def _allocate_writes(self, writes, protected):
        new = writes[self.gpu_map[writes] == 0]
        if np.any(self.cpu_map[writes]):
            raise RuntimeError("Target forward tried to overwrite a committed CPU prefix")
        deficit = max(0, len(new) - len(self.gpu_free))
        if deficit:
            if deficit > len(self.cpu_free):
                raise RuntimeError("Insufficient host Target KV capacity")
            eligible = self.gpu_map > 0
            eligible[writes] = False
            eligible[protected] = False
            victims = self._committed_victims(eligible, deficit)
            source = self.gpu_map[victims]
            reserved = [self.cpu_free.pop() for _ in victims]
            # Ascending nearby destination IDs preserve host contiguity after
            # freed slots are reused. Logical token order stays the same.
            destination = torch.tensor(sorted(reserved), dtype=torch.int64)
            try:
                self.loader.migrate(self.pool, source, destination)
            except BaseException:
                self.cpu_free.extend(reversed(reserved))
                raise
            self.cpu_map[victims] = destination.numpy()
            self.gpu_map[victims] = 0
            self.gpu_free.extend(source.tolist())
            self.spilled_records += deficit
            self.prefill_own_evicted += int(np.count_nonzero(self.prompt_priority[victims] == 2))
        self.gpu_map[new] = [self.gpu_free.pop() for _ in new]

    def prepare(self, metadata, batch):
        mode = batch.forward_mode
        placed = copy(batch)
        if mode.is_target_verify():
            # Native EAGLE verification reports the old logical length; its
            # q contiguous provisional slots are already in the request table.
            verify = batch.spec_info
            if verify is None or verify.topk != 1 or verify.draft_token_num < 2:
                raise ValueError("Target KV offload requires linear speculative verification")
            queries = [verify.draft_token_num] * batch.batch_size
            placed.seq_lens_cpu = batch.seq_lens_cpu + verify.draft_token_num
        elif mode.is_decode():
            # Ordinary Target decode has already incremented seq_lens. Use the
            # extend placement path to avoid treating its writes as Draft scratch.
            queries = [1] * batch.batch_size
            placed.forward_mode = ForwardMode.DRAFT_EXTEND
        elif mode.is_extend():
            queries = batch.extend_seq_lens_cpu
        else:
            raise ValueError(f"Unsupported pressure-aware Target forward mode: {mode}")
        if not queries or len(queries) != batch.batch_size or any(q < 1 for q in queries):
            raise ValueError("Target query lengths must be positive and match the batch")
        if sum(queries) != batch.input_ids.numel():
            raise ValueError("Target attention query lengths disagree with model input")

        placed.extend_seq_lens_cpu = queries
        plan = super().prepare(metadata, placed, step=0, draft_steps=0)
        metadata.max_seq_len_q = max(queries)
        metadata.cu_seqlens_q = torch.tensor(
            [0, *accumulate(queries)], dtype=torch.int32, device=batch.seq_lens.device,
        )
        transfer = self.loader.prepare(plan.cpu_rows) if plan.cpu_rows else None
        return TargetAttentionPlan(plan, transfer)

    @staticmethod
    def _window_queries(plan, lengths, device):
        cached = plan.query_metadata.get(lengths)
        if cached is not None:
            return cached
        rows = [ids for ids, length in zip(plan.placement.cpu_queries, lengths) if length]
        counts_q = [len(row) for row in rows]
        counts_k = [n for n in lengths if n]
        cached = (
            torch.as_tensor(np.concatenate(rows), dtype=torch.int64, device=device),
            torch.tensor([0, *accumulate(counts_q)], dtype=torch.int32, device=device),
            torch.tensor([0, *accumulate(counts_k)], dtype=torch.int32, device=device),
            max(counts_q), max(counts_k),
        )
        plan.query_metadata[lengths] = cached
        return cached

    def forward(self, backend, metadata, plan, q, k, v, layer, save_kv_cache):
        if not save_kv_cache or k is None or v is None or layer.is_cross_attention:
            raise RuntimeError("Target KV offload requires writable causal self-attention")
        placement = plan.placement
        self.pool.set_kv_buffer(layer, placement.write_slots, k, v, layer.k_scale, layer.v_scale)
        key, value = self.pool.get_kv_buffer(layer.layer_id)
        query = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        layer_offset = layer.layer_id - self.pool.start_layer
        if plan.transfer is not None:
            self.loader.prime(layer_offset, plan.transfer)
        resident = backend.flash_attn_with_kvcache(
            q=query, k_cache=key.unsqueeze(1), v_cache=value.unsqueeze(1),
            page_table=placement.page_table, cache_seqlens=placement.lengths,
            cu_seqlens_q=metadata.cu_seqlens_q, max_seqlen_q=metadata.max_seq_len_q,
            softmax_scale=layer.scaling, causal=True, softcap=layer.logit_cap,
            return_softmax_lse=plan.transfer is not None, num_splits=backend.num_splits,
        )
        if plan.transfer is None:
            return resident.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
        result = resident[0].float().contiguous()
        lse = resident[1].transpose(0, 1).float().contiguous()

        def merge_window(packed, lengths):
            indices, cu_q, cu_k, max_q, max_k = self._window_queries(plan, tuple(lengths), query.device)
            host_out, host_lse, *_ = backend.flash_attn_varlen_func(
                q=query.index_select(0, indices), k=packed[:, 0], v=packed[:, 1],
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max_q, max_seqlen_k=max_k,
                softmax_scale=layer.scaling, causal=False, softcap=layer.logit_cap,
                return_softmax_lse=True,
            )
            updated, updated_lse = merge_state_triton(
                result.index_select(0, indices), lse.index_select(0, indices),
                host_out.contiguous(), host_lse.transpose(0, 1).float().contiguous(),
            )
            result.index_copy_(0, indices, updated)
            lse.index_copy_(0, indices, updated_lse)
            self.gpu_online_merge_calls += 1
            self.gpu_cohort_calls += sum(length > 0 for length in lengths) > 1

        windows = self.loader.windows(layer_offset, plan.transfer)
        try:
            for window in windows:
                merge_window(*window)
        finally:
            windows.close()
        return result.to(query.dtype).reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
