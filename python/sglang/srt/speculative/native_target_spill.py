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
from sglang.srt.speculative.native_target_host_blocks import TargetHostBlockPool
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
    dma_count: int
    submit_ns: int
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

    def __init__(self, pool, logical_slots, rank, tp_threads, *, max_requests=0):
        super().__init__(pool, logical_slots, rank, tp_threads)
        # A partial prefill should keep its earlier chunks on the GPU when
        # older committed rows can be moved instead.
        self.prefer_prompt_gpu = True
        self.min_gpu_tail = int(os.getenv("SPECSTREAM_NATIVE_TARGET_MIN_GPU_TAIL", "0"))
        if not 0 <= self.min_gpu_tail <= self.gpu_slots:
            raise ValueError("Target GPU tail must fit the Target KV pool")
        self.requests = weakref.WeakValueDictionary()
        self._request_prefix_tokens = {}
        self._request_prefix_rids = {}
        self._pending_prefix_updates = None
        self.request_table = None
        self.loader = None
        self.use_host_blocks = os.getenv("SPECSTREAM_NATIVE_TARGET_HOST_BLOCKS") == "1"
        self.host_block_pool = None
        self.max_requests = max_requests
        if self.use_host_blocks and self.max_requests < 1:
            raise ValueError("Target host blocks require the request pool capacity")
        self.retired_h2d = [0, 0, 0, 0, 0, 0]
        self.retired_d2h_bytes = 0
        self.gpu_online_merge_calls = 0
        self.gpu_cohort_calls = 0
        self._profile_verify = bool(os.getenv("SPECSTREAM_NATIVE_SEGMENT_TRACE")) or (
            os.getenv("SPECSTREAM_NATIVE_DYNAMIC_Q") == "1"
            and os.getenv("SPECSTREAM_NATIVE_Q_CONTROLLER", "paper") == "paper"
        )
        self._verify_capture = None
        self._verify_timings: dict[int, _VerifyTransferTiming] = {}

    def register_requests(self, batch):
        # Retain lightweight owner references from the identity fast path.
        # No GPU table reads until there is an actual migration deficit.
        self.request_table = batch.req_to_token_pool.req_to_token
        for req in batch.reqs:
            if req.req_pool_idx is not None:
                index = int(req.req_pool_idx)
                if self._request_prefix_rids.get(index) != req.rid or int(req.kv_committed_len) == 0:
                    self._request_prefix_tokens[index] = 0
                    self._request_prefix_rids[index] = req.rid
                self.requests[index] = req

    def batch_history_tokens(self, requests):
        """Read committed CPU-prefix counts without downloading a GPU page table."""
        return sum(
            min(self._request_prefix_tokens.get(int(req.req_pool_idx), 0), int(req.kv_committed_len))
            for req in requests
            if req.req_pool_idx is not None
            and self._request_prefix_rids.get(int(req.req_pool_idx)) == req.rid
        )

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
        capacity = self.cpu_capacity
        if self.use_host_blocks:
            self.host_block_pool = TargetHostBlockPool(capacity, self.max_requests, block_size=1024)
            slot_limit = self.host_block_pool.slot_limit
        else:
            self.cpu_free = list(range(capacity, 0, -1))
            slot_limit = capacity + 1
        self.history = NativeTargetHostHistory(
            self.pool.layer_num, slot_limit, self.pool.head_num,
            self.pool.head_dim, self.pool.dtype, self.pool.device,
        )
        self.loader = NativeTargetKVLoader(
            self.history, self.pool.device, track_verify_windows=self._profile_verify,
            host_blocks=self.use_host_blocks,
        )
        if self._verify_capture is not None:
            self.loader.verify_timing_round = self._verify_capture[0]

    def reset(self):
        if self._verify_capture is not None:
            raise RuntimeError("Target KV placement reset during verification timing")
        if self.host_block_pool is not None:
            if self.host_block_pool.live_blocks:
                # Administrative reset can bypass the ordinary finish path.
                fence = self.loader.fence_host_reuse()
                if fence is not None:
                    fence.synchronize()
            self.host_block_pool.finish_releases()
            self.host_block_pool = None
        self.retired_h2d = [self.h2d_bytes, self.h2d_cross_layer_bytes,
                            self.h2d_chunks, self.h2d_copy_batches,
                            self.h2d_dma_count, self.h2d_submit_ns]
        self.retired_d2h_bytes = self.d2h_bytes
        self.loader = None
        self.requests.clear()
        self._request_prefix_tokens.clear()
        self._request_prefix_rids.clear()
        self._pending_prefix_updates = None
        self.request_table = None
        super().reset()

    @property
    def cpu_resident_tokens(self):
        if self.host_block_pool is not None:
            return self.host_block_pool.live_tokens
        return super().cpu_resident_tokens

    def _free(self, logical):
        if self.host_block_pool is None:
            return super()._free(logical)
        host = self.cpu_map[logical]
        self.host_block_pool.release(host[host > 0], self.loader.fence_host_reuse)
        gpu = self.gpu_map[logical]
        self.gpu_free.extend(gpu[gpu > 0].tolist())
        self.gpu_map[logical] = 0
        self.cpu_map[logical] = 0
        self.prompt_priority[logical] = 0

    def begin_verify_timing(self, round_id: int) -> None:
        """Track only copies submitted by this Target verification forward."""
        if not self._profile_verify or self._verify_capture is not None or round_id in self._verify_timings:
            raise RuntimeError("Invalid Target verification timing scope")
        self._verify_capture = (round_id, self.h2d_copy_batches, self.h2d_bytes,
                                self.h2d_dma_count, self.h2d_submit_ns)
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
            self.h2d_dma_count - capture[3],
            self.h2d_submit_ns - capture[4],
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
            "target_h2d_dma_count": timing.dma_count,
            "target_h2d_submit_ms": timing.submit_ns / 1e6,
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
    def h2d_dma_count(self):
        return self.retired_h2d[4] + (self.loader.dma_count if self.loader else 0)

    @property
    def h2d_submit_ns(self):
        return self.retired_h2d[5] + (self.loader.h2d_submit_ns if self.loader else 0)

    @property
    def d2h_bytes(self):
        return self.retired_d2h_bytes + (self.loader.d2h_bytes if self.loader else 0)

    def _committed_victims(self, eligible, count, *, protected_tail=None):
        if self.request_table is None:
            raise RuntimeError("Target offload cannot find the live request owners")
        owners = sorted((index, int(req.kv_committed_len)) for index, req in self.requests.items()
                        if req.req_pool_idx == index and int(req.kv_committed_len) > 0)
        if not owners:
            raise RuntimeError("Target offload has no committed history to migrate")
        indices = torch.tensor([i for i, _ in owners], dtype=torch.int64, device=self.pool.device)
        table = self.request_table[indices, :max(n for _, n in owners)].cpu().numpy()
        by_priority = [[], [], []]
        for row, (index, length) in zip(table, owners):
            history = row[:length]
            on_cpu = self.cpu_map[history] > 0
            boundary = int(np.count_nonzero(on_cpu))
            if not on_cpu[:boundary].all():
                raise RuntimeError("Target CPU keys must be a fixed prefix of each request")
            # Keep each request's recent committed KV in its GPU pool. The
            # same prefix-only migration invariant still holds for CPU KV.
            tail = self.min_gpu_tail if protected_tail is None else max(self.min_gpu_tail, protected_tail)
            gpu_tail_start = max(boundary, length - tail)
            candidate = history[boundary:gpu_tail_start]
            blocked = np.flatnonzero(~eligible[candidate])
            if len(blocked):
                candidate = candidate[:blocked[0]]
            if len(candidate):
                tail_slot = int(self.cpu_map[history[boundary - 1]]) if boundary else 0
                by_priority[int(self.prompt_priority[candidate[0]])].append((index, boundary, tail_slot, candidate))
        victims = []
        updates = {}
        appends = []
        for priority in by_priority:
            for index, boundary, tail_slot, row in priority:
                take = min(len(row), count)
                victims.append(row[:take])
                updates[index] = boundary + take
                if self.host_block_pool is not None:
                    # The request row can be recycled. A live block must match
                    # both the slot and the request that owns its CPU prefix.
                    owner = (index, self._request_prefix_rids[index])
                    appends.append((owner, tail_slot, take))
                count -= take
                if not count:
                    return np.concatenate(victims), updates, appends
        raise RuntimeError(f"Target KV pool is short of {count} eligible committed prefix slots")

    def _migrate_committed(self, eligible, count, *, protected_tail=None):
        victims, updates, appends = self._committed_victims(
            eligible, count, protected_tail=protected_tail,
        )
        source = self.gpu_map[victims].copy()
        block_pool = self.host_block_pool
        reservation = None
        reserved = []
        if block_pool is None:
            if count > len(self.cpu_free):
                raise RuntimeError("Insufficient Target host KV capacity")
            reserved = [self.cpu_free.pop() for _ in victims]
        else:
            reservation = block_pool.reserve(appends)

        self._pending_prefix_updates = updates
        try:
            if reservation is None:
                destination = torch.tensor(sorted(reserved), dtype=torch.int64)
            else:
                destination = torch.as_tensor(reservation.slots, dtype=torch.int64)
            # migrate completes all D2H copies before returning, including on
            # exception, so the CPU reservation is safe to commit or reuse.
            self.loader.migrate(self.pool, source, destination)
        except BaseException:
            if reservation is None:
                self.cpu_free.extend(reversed(reserved))
            else:
                block_pool.rollback(reservation)
            raise
        finally:
            self._pending_prefix_updates = None
        if reservation is not None:
            block_pool.commit(reservation)
        self.cpu_map[victims] = destination.numpy()
        self.gpu_map[victims] = 0
        self.gpu_free.extend(source.tolist())
        self._request_prefix_tokens.update(updates)
        self.spilled_records += count
        return victims

    def shrink_gpu(self, new_slots, *, protected_logical=(), protected_tail=512):
        """Evacuate high GPU pages before the arena crops the Target backing.

        The caller enters at a round boundary with completed device work and
        waits for the queued compaction before changing the backing tensors.
        """
        new_slots = int(new_slots)
        protected_tail = int(protected_tail)
        old_slots = self.gpu_slots
        if not self.active:
            raise RuntimeError("Target GPU shrink requires an initialized ownership map")
        if not 0 < new_slots <= old_slots or self.pool.size < old_slots or protected_tail < 0:
            raise ValueError("Invalid Target GPU shrink capacity or protected tail")
        self._check_resize_safety()
        if self._verify_capture is not None or self._pending_prefix_updates is not None:
            raise RuntimeError("Target GPU shrink must occur outside verification and migration")
        if self.loader.prefetched is not None:
            raise RuntimeError("Target GPU shrink cannot interrupt prefetched KV")
        if isinstance(protected_logical, torch.Tensor):
            protected_logical = protected_logical.detach().cpu().numpy()
        protected = np.asarray(protected_logical, dtype=np.int64).reshape(-1)
        if np.any(protected < 0) or np.any(protected > self.logical_slots):
            raise ValueError("Protected Target logical slots are out of bounds")

        used = int(np.count_nonzero(self.gpu_map))
        migrate_count = max(0, used - new_slots)
        if migrate_count:
            eligible = self.gpu_map > 0
            eligible[protected] = False
            self._migrate_committed(eligible, migrate_count, protected_tail=protected_tail)

        high_logical = np.flatnonzero(self.gpu_map > new_slots)
        low_free = np.asarray([slot for slot in self.gpu_free if slot <= new_slots], dtype=np.int64)
        if len(high_logical) > len(low_free):
            raise RuntimeError("Target shrink cannot compact its protected GPU tail")
        destinations = low_free[:len(high_logical)]
        source = self.gpu_map[high_logical].copy()
        for offset in range(0, len(source), self.loader.stage_tokens):
            count = min(self.loader.stage_tokens, len(source) - offset)
            picks = torch.as_tensor(source[offset:offset + count], dtype=torch.int64, device=self.pool.device)
            writes = torch.as_tensor(destinations[offset:offset + count], dtype=torch.int64, device=self.pool.device)
            packed = self.loader.pool._ensure_shape(
                0, (count, 2, self.pool.head_num, self.pool.head_dim), self.pool.dtype,
            )
            for layer in range(self.pool.layer_num):
                key, value = self.pool.get_kv_buffer(self.pool.start_layer + layer)
                torch.index_select(key, 0, picks, out=packed[:, 0])
                torch.index_select(value, 0, picks, out=packed[:, 1])
                key.index_copy_(0, writes, packed[:, 0])
                value.index_copy_(0, writes, packed[:, 1])
        self.gpu_map[high_logical] = destinations
        assigned = set(destinations.tolist())
        self.gpu_free = [slot for slot in self.gpu_free if slot <= new_slots and slot not in assigned]
        self.gpu_slots = new_slots
        self._check_resize_safety()
        return dict(
            old_slots=old_slots, new_slots=new_slots,
            migrated_tokens=migrate_count, compacted_tokens=len(high_logical),
        )

    def _allocate_writes(self, writes, protected):
        new = writes[self.gpu_map[writes] == 0]
        if np.any(self.cpu_map[writes]):
            raise RuntimeError("Target forward tried to overwrite a committed CPU prefix")
        deficit = max(0, len(new) - len(self.gpu_free))
        if deficit:
            eligible = self.gpu_map > 0
            eligible[writes] = False
            eligible[protected] = False
            victims = self._migrate_committed(eligible, deficit)
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
