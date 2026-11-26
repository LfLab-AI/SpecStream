"""Physical Draft KV placement for native, single-branch speculative decoding."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from sglang.srt.speculative.spectre.drafter.cpu_history_attention import CpuHistoryAttention
from sglang.srt.speculative.spectre.drafter.split_attention import SplitDraftAttention


def native_draft_gpu_slots(runner):
    value = os.getenv("SPECSTREAM_NATIVE_DRAFT_GPU_SLOTS")
    if not value or not runner.is_draft_worker:
        return runner.max_total_num_tokens
    if runner.server_args.speculative_algorithm != "STANDALONE":
        return runner.max_total_num_tokens
    slots = int(value)
    if not 0 < slots <= runner.max_total_num_tokens:
        raise ValueError("The native Draft GPU pool must fit inside the logical KV capacity")
    return slots


@dataclass
class DraftAttentionPlan:
    write_slots: torch.Tensor
    page_table: torch.Tensor
    lengths: torch.Tensor
    cpu_rows: list[np.ndarray]
    cpu_queries: list[np.ndarray]
    cpu_groups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]] | None = None
    allow_grouped: bool = True


class NativeDraftSpill:
    """Keep a common logical slot id and separate physical Draft ownership.

    Before activation, logical ids in use fit the physical Draft pool. Native
    page tables and CUDA graphs can use these ids directly. The first allocation
    above that range establishes the physical map from the live allocator state.
    """

    def __init__(self, pool, logical_slots, rank, tp_threads):
        self.pool = pool
        self.logical_slots = logical_slots
        self.gpu_slots = pool.size
        self.cpu_capacity = logical_slots
        self.rank = rank
        self.tp_threads = tp_threads
        self.strict_gpu_tail = int(os.getenv("SPECSTREAM_NATIVE_STRICT_GPU_TAIL", "0"))
        self.gpu_map = None
        self.cpu_map = None
        self.cpu_attention_calls = 0
        self.grouped_cpu_attention = os.getenv("SPECSTREAM_GROUPED_CPU_ATTN", "0") == "1"
        self.group_only_in_decode = os.getenv("SPECSTREAM_GROUPED_DECODE_ONLY", "0") == "1"
        self.trace_cpu_work = rank == 0 and os.getenv("SPECSTREAM_CPU_ATTN_TIMING") == "1"
        self.last_cpu_work = None
        self._previous_cpu_attention_ns = 0
        self.spilled_records = 0
        self.prefer_prompt_gpu = os.getenv("SPECSTREAM_KEEP_PREFILL_GPU", "0") == "1"
        self.prompt_priority = None
        self.prefill_own_evicted = 0
        self.prefill_host_queries = 0
        self.prefill_host_query_keys = 0
        self.history = None
        self.split = None
        self.tentative = None

    @property
    def active(self):
        return self.gpu_map is not None

    @property
    def cpu_resident_tokens(self):
        if not self.active:
            return 0
        return self.cpu_capacity - len(self.cpu_free)

    @property
    def cpu_attention_ns(self):
        """Rank-local CPU attention call time, summed across active and retired pools."""
        return self._previous_cpu_attention_ns + (self.split.cpu_attention_ns if self.split else 0)

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
        if self.prefer_prompt_gpu:
            self.prompt_priority = np.zeros(self.logical_slots + 1, dtype=np.uint8)
        self.gpu_free = low_free[::-1].tolist()
        # GPU ownership may change later. Host addresses stay valid for the
        # full logical lifetime, including after GPU capacity is returned.
        capacity = self.cpu_capacity
        self.cpu_free = list(range(capacity, 0, -1))
        make_history = getattr(self, "history_factory", CpuHistoryAttention)
        self.history = make_history(
            self.pool.layer_num, capacity + 1, self.pool.head_num,
            head_dim=self.pool.head_dim, dtype=self.pool.dtype,
            num_threads=self.tp_threads, tp_rank=self.rank,
        )
        self.split = SplitDraftAttention(self.history, self.pool.device)

    def free(self, logical):
        if self.active:
            self._free(logical.cpu().numpy())

    def reset(self):
        release_history = getattr(self, "release_shared_history", None)
        if release_history is not None:
            release_history()
        self._previous_cpu_attention_ns = self.cpu_attention_ns
        self.gpu_map = None
        self.cpu_map = None
        self.prompt_priority = None
        self.history = None
        self.split = None
        self.tentative = None

    def _free(self, logical):
        self.gpu_free.extend(self.gpu_map[logical][self.gpu_map[logical] > 0].tolist())
        self.cpu_free.extend(self.cpu_map[logical][self.cpu_map[logical] > 0].tolist())
        self.gpu_map[logical] = 0
        self.cpu_map[logical] = 0
        if self.prompt_priority is not None:
            self.prompt_priority[logical] = 0

    def end_tentative(self):
        if self.tentative is not None:
            self._free(self.tentative)
            self.tentative = None

    def _check_resize_safety(self):
        if not self.active:
            return
        if self.tentative is not None:
            raise RuntimeError("KV capacity cannot change while tentative Draft writes are live")
        physical = self.gpu_map[self.gpu_map > 0]
        free = np.asarray(self.gpu_free, dtype=np.int64)
        if (
            len(physical) + len(free) != self.gpu_slots
            or len(np.unique(physical)) != len(physical)
            or len(np.unique(free)) != len(free)
            or np.any(physical > self.gpu_slots)
            or np.any(free <= 0)
            or np.any(free > self.gpu_slots)
            or np.intersect1d(physical, free).size
        ):
            raise RuntimeError("KV capacity cannot change while GPU scratch is detached or ownership is inconsistent")

    def grow_gpu(self, new_slots):
        """Publish new free pages after the arena has enlarged this GPU backing."""
        new_slots = int(new_slots)
        old_slots = self.gpu_slots
        if not old_slots <= new_slots <= min(self.logical_slots, self.pool.size):
            raise ValueError("New Draft capacity must fit its rebound GPU backing")
        self._check_resize_safety()
        if self.active:
            self.gpu_free.extend(range(old_slots + 1, new_slots + 1))
        self.gpu_slots = new_slots
        return dict(old_slots=old_slots, new_slots=new_slots, added_slots=new_slots - old_slots)

    def _allocate_writes(self, writes, protected):
        new = writes[self.gpu_map[writes] == 0]
        if np.any(self.cpu_map[writes]):
            raise RuntimeError("Attention tried to rewrite a committed CPU Draft slot")
        deficit = max(0, len(new) - len(self.gpu_free))
        if deficit:
            if deficit > len(self.cpu_free):
                raise RuntimeError("Insufficient host Draft KV slots")
            eligible = self.gpu_map > 0
            eligible[writes] = False
            eligible[protected] = False
            candidates = np.flatnonzero(eligible)
            victims = candidates[:deficit]
            if self.prompt_priority is not None:
                # Lower logical slot IDs still win ties, as in the old allocator.
                # Both prompt tiers remain eligible when other slots run out.
                priority = self.prompt_priority[candidates]
                ordinary = candidates[priority == 0]
                victims = ordinary[:deficit]
                if len(victims) < deficit:
                    waiting = candidates[priority == 1]
                    current = candidates[priority == 2]
                    victims = np.concatenate((ordinary, waiting, current))[:deficit]
            if len(victims) != deficit:
                raise RuntimeError("Draft GPU pool cannot fit protected and new KV")
            self._migrate_gpu_victims(victims)
            if self.prompt_priority is not None:
                self.prefill_own_evicted += int(np.count_nonzero(self.prompt_priority[victims] == 2))
        self.gpu_map[new] = [self.gpu_free.pop() for _ in new]

    def _migrate_gpu_victims(self, victims):
        count = len(victims)
        if not count:
            return
        if count > len(self.cpu_free):
            raise RuntimeError("Insufficient host Draft KV slots")
        src = self.gpu_map[victims]
        if np.any(src <= 0) or np.any(self.cpu_map[victims]):
            raise RuntimeError("Draft migration requires committed GPU-only KV slots")
        dst = torch.tensor([self.cpu_free.pop() for _ in victims], dtype=torch.int64, device="cpu")
        src_gpu = torch.as_tensor(src, device=self.pool.device)
        try:
            for layer in range(self.pool.layer_num):
                k, v = self.pool.get_kv_buffer(self.pool.start_layer + layer)
                self.history.store(layer, dst, k.index_select(0, src_gpu).cpu(),
                                   v.index_select(0, src_gpu).cpu())
        except BaseException:
            self.cpu_free.extend(reversed(dst.tolist()))
            raise
        self.cpu_map[victims] = dst.numpy()
        self.gpu_map[victims] = 0
        self.gpu_free.extend(src.tolist())
        self.spilled_records += count

    def _allocate_for_batch(self, writes, protected, rows, q_lengths, mode):
        if self.prompt_priority is None:
            self._allocate_writes(writes, protected)
            return ()
        # Only original input prefill gets preferential placement. DRAFT_EXTEND
        # processes verified output tokens and should release any prompt marks.
        # With disabled radix sharing the first logical slot belongs to one
        # request and stays in its row until release, even when resident on CPU.
        prefill = mode.name in ("EXTEND", "SPLIT_PREFILL", "MIXED")
        active_prompts = []
        for i, (row, queries) in enumerate(zip(rows, q_lengths)):
            if len(row) == 0:
                continue
            if prefill and (mode.name != "MIXED" or queries > 1):
                self.prompt_priority[row] = 2
                active_prompts.append(i)
            elif self.prompt_priority[row[0]]:
                self.prompt_priority[row] = 0
        try:
            self._allocate_writes(writes, protected)
        finally:
            for i in active_prompts:
                self.prompt_priority[rows[i]] = 1
        return active_prompts

    def prepare(self, metadata, batch, step, draft_steps):
        decode = batch.forward_mode.is_decode_or_idle()
        device = self.pool.device
        if decode and step > 0:
            gpu_rows, host_rows, slots = batch.native_draft_spill_decode
        else:
            q_lengths = [1] * batch.batch_size if decode else batch.extend_seq_lens_cpu
            base_lengths = batch.seq_lens_cpu.tolist()
            full_lengths = [n + draft_steps for n in base_lengths] if decode else base_lengths
            table = batch.req_to_token_pool.req_to_token[
                batch.req_pool_indices, : max(full_lengths)
            ].cpu().numpy()
            rows = [table[i, :n] for i, n in enumerate(full_lengths)]
            writes = batch.out_cache_loc.cpu().numpy()
            if writes.ndim != 1 or np.any(writes <= 0) or len(np.unique(writes)) != len(writes):
                raise RuntimeError("Draft KV writes require unique, nonzero logical slots")
            if any(np.any(row <= 0) for row in rows):
                raise RuntimeError("Missing logical slots in the Draft page table")
            write_counts = [draft_steps] * batch.batch_size if decode else q_lengths
            expected_writes = np.concatenate(
                [row[len(row) - n:] for row, n in zip(rows, write_counts)]
            )
            if not np.array_equal(writes, expected_writes):
                raise RuntimeError("New Draft KV must be the causal suffix of each request")
            protected = np.concatenate(
                [row[max(0, n - (self.strict_gpu_tail or 128)):n] for row, n in zip(rows, base_lengths)]
            )
            active_prompts = self._allocate_for_batch(writes, protected, rows, q_lengths, batch.forward_mode)
            gpu_rows = []
            host_rows = []
            for row in rows:
                gpu = self.gpu_map[row]
                host = self.cpu_map[row]
                if np.any((gpu > 0) == (host > 0)):
                    raise RuntimeError("Each live Draft slot must have one physical owner")
                gpu_rows.append(gpu[gpu > 0])
                host_rows.append(host[host > 0])
            for i in active_prompts:
                if len(host_rows[i]):
                    self.prefill_host_queries += int(q_lengths[i])
                    self.prefill_host_query_keys += int(q_lengths[i]) * len(host_rows[i])
            slots = self.gpu_map[writes]
            if decode:
                self.tentative = writes
                slots = slots.reshape(batch.batch_size, draft_steps)
                batch.native_draft_spill_decode = (gpu_rows, host_rows, slots)

        if decode:
            q_lengths = [1] * batch.batch_size
            gpu_rows = [row[:len(row) - draft_steps + step + 1] for row in gpu_rows]
            slots = slots[:, step]
        counts = [len(row) for row in gpu_rows]
        dense = np.zeros((len(counts), max(counts)), dtype=np.int32)
        for i, row in enumerate(gpu_rows):
            dense[i, :len(row)] = row
        query_offsets = np.cumsum([0, *q_lengths], dtype=np.int64)
        cpu_rows, cpu_queries = [], []
        for i, row in enumerate(host_rows):
            if len(row) and query_offsets[i] < query_offsets[i + 1]:
                cpu_rows.append(row)
                cpu_queries.append(np.arange(query_offsets[i], query_offsets[i + 1]))
        if self.trace_cpu_work:
            self.last_cpu_work = [(len(row), len(ids)) for row, ids in zip(cpu_rows, cpu_queries)]
        return DraftAttentionPlan(
            write_slots=torch.as_tensor(np.ascontiguousarray(slots), device=device),
            page_table=torch.as_tensor(dense, device=device),
            lengths=torch.tensor(counts, dtype=torch.int32, device=device),
            cpu_rows=cpu_rows,
            cpu_queries=cpu_queries,
            allow_grouped=(not self.group_only_in_decode or decode or batch.forward_mode.is_draft_extend()),
        )

    def forward(self, backend, metadata, plan, q, k, v, layer, save_kv_cache):
        if not save_kv_cache or k is None or v is None or layer.is_cross_attention:
            raise RuntimeError("Native Draft spill requires writable causal self-attention")
        self.pool.set_kv_buffer(layer, plan.write_slots, k, v, layer.k_scale, layer.v_scale)
        key, value = self.pool.get_kv_buffer(layer.layer_id)
        query = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

        def gpu_partial():
            result = backend.flash_attn_with_kvcache(
                q=query,
                k_cache=key.unsqueeze(1),
                v_cache=value.unsqueeze(1),
                page_table=plan.page_table,
                cache_seqlens=plan.lengths,
                cu_seqlens_q=metadata.cu_seqlens_q,
                max_seqlen_q=metadata.max_seq_len_q,
                softmax_scale=layer.scaling,
                causal=True,
                softcap=layer.logit_cap,
                return_softmax_lse=bool(plan.cpu_rows),
                num_splits=backend.num_splits,
            )
            if plan.cpu_rows:
                return result[0], result[1].transpose(0, 1).contiguous()
            return result

        if not plan.cpu_rows:
            return gpu_partial().reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

        small_group = sum(len(ids) for ids in plan.cpu_queries) <= 128
        if small_group and plan.cpu_groups is None:
            # Reuse small decode/correction tables across all layers. Bound the
            # size of any one CPU state calculation during chunked prefill.
            plan.cpu_groups = list(self._cpu_groups(plan, query.device))
        groups = plan.cpu_groups if small_group else self._cpu_groups(plan, query.device)
        result = None
        lse = None
        for index, slots, lengths, query_offsets in groups:
            def gpu_subset():
                nonlocal result, lse
                if result is None:
                    result, lse = gpu_partial()
                return result.index_select(0, index), lse.index_select(0, index)

            merged = self.split.forward(
                layer.layer_id - self.pool.start_layer, query.index_select(0, index),
                slots, lengths, layer.scaling, gpu_subset, query_offsets=query_offsets,
            )
            result.index_copy_(0, index, merged)
            self.cpu_attention_calls += 1
        return result.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _cpu_groups(self, plan, device):
        if self.grouped_cpu_attention and plan.allow_grouped and any(len(ids) > 1 for ids in plan.cpu_queries):
            # Current writes stay on GPU. Every new query of this request can
            # therefore read every slot in the same committed CPU history.
            # Keep the original indices for scattering results back to FA3.
            ends = np.cumsum([len(ids) for ids in plan.cpu_queries], dtype=np.int64)
            starts = np.concatenate(([0], ends[:-1]))
            indices = np.concatenate(plan.cpu_queries)
            for begin in range(0, len(indices), 128):
                end = min(begin + 128, len(indices))
                first = int(ends.searchsorted(begin, side="right"))
                stop = int(ends.searchsorted(end, side="left")) + 1
                group = plan.cpu_rows[first:stop]
                counts = np.minimum(end, ends[first:stop]) - np.maximum(begin, starts[first:stop])
                offsets = None
                if (counts > 1).any():
                    offsets = torch.from_numpy(np.concatenate(([0], counts.cumsum())))
                yield self._pack_cpu_group(indices[begin:end], group, device, offsets)
            return

        rows = [row for row, ids in zip(plan.cpu_rows, plan.cpu_queries) for _ in ids]
        indices = np.concatenate(plan.cpu_queries)
        for begin in range(0, len(indices), 128):
            yield self._pack_cpu_group(indices[begin:begin + 128], rows[begin:begin + 128], device, None)

    @staticmethod
    def _pack_cpu_group(indices, group, device, offsets):
        index = torch.as_tensor(indices, dtype=torch.int64, device=device)
        lengths = torch.tensor([len(row) for row in group], dtype=torch.int64, device="cpu")
        slots = torch.zeros((len(group), int(lengths.max())), dtype=torch.int64, device="cpu")
        for i, row in enumerate(group):
            slots[i, :len(row)].copy_(torch.from_numpy(row))
        return index, slots, lengths, offsets
