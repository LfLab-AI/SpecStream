from __future__ import annotations

import logging
import math
import time

import torch

from sglang.srt.speculative.spectre.specstream.acceptance_tracker import (
    AcceptanceTracker,
)
from sglang.srt.speculative.spectre.specstream.cohort_scheduler import (
    StreamWorkItem,
    build_cohort_plans,
)
from sglang.srt.speculative.spectre.specstream.config import SpecStreamConfig
from sglang.srt.speculative.spectre.specstream.controller import (
    IOAwareController,
    SpecStreamDecision,
)
from sglang.srt.speculative.spectre.specstream.cost_model import SpecStreamBatchState
from sglang.srt.speculative.spectre.specstream.cpu_history_store import CPUHistoryStore
from sglang.srt.speculative.spectre.specstream.diagnostics import (
    SpecStreamDiagnostics,
)
from sglang.srt.speculative.spectre.specstream.online_softmax import (
    OnlineSoftmaxState,
    finalize_online_softmax_state,
    init_online_softmax_state,
    update_online_softmax_state,
)
from sglang.srt.speculative.spectre.specstream.profiler import SpecStreamProfiler
from sglang.srt.speculative.spectre.specstream.round_meta import (
    SpecStreamRequestMeta,
    SpecStreamRoundMeta,
)
from sglang.srt.speculative.spectre.specstream.staging_runtime import (
    StagingWindowPool,
)
from sglang.srt.speculative.spectre.specstream.state import TargetTieredKVState
from sglang.srt.speculative.spectre.specstream.triton_stream_attn import (
    split_packed_history_cohort_state,
    stack_packed_history_cohort_states,
    update_gpu_tail_state,
    update_packed_history_cohort_batched,
    update_packed_history_state,
)

logger = logging.getLogger(__name__)


def _group_adjacent_chunks(chunks, group_size: int):
    if group_size < 1:
        raise ValueError("SpecStream chunk group size must be positive")
    return [
        chunks[index : index + group_size]
        for index in range(0, len(chunks), group_size)
    ]


def _empty_host(shape, *, dtype, pin_memory: bool):
    """Allocate pinned host memory when available, with a safe pageable fallback."""
    try:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin_memory)
    except RuntimeError:
        logger.warning(
            "Pinned CPU allocation failed; using pageable SpecStream History"
        )
        return torch.empty(shape, dtype=dtype, device="cpu")


class SpecStreamVerifier:
    """One-layer exact History streaming path used by TARGET_VERIFY."""

    def __init__(
        self,
        *,
        config: SpecStreamConfig,
        history_store: CPUHistoryStore,
        staging: StagingWindowPool,
        profiler: SpecStreamProfiler,
        diagnostics: SpecStreamDiagnostics,
    ) -> None:
        self.config = config
        self.history_store = history_store
        self.staging = staging
        self.profiler = profiler
        self.diagnostics = diagnostics

    def forward(self, *, q, k_new, v_new, forward_batch, meta, layer):
        if layer.is_cross_attention:
            raise RuntimeError("SpecStream does not support cross attention")
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            raise RuntimeError("SpecStream v1 does not support sliding-window layers")
        if self.staging.device.type != q.device.type:
            raise RuntimeError("SpecStream staging and query devices differ")

        query = q.contiguous().view(-1, int(layer.tp_q_head_num), int(layer.head_dim))
        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        if key_cache.ndim != 3 or value_cache.ndim != 3:
            raise RuntimeError("SpecStream v1 requires page_size=1 MHA KV buffers")
        num_kv_heads = int(key_cache.shape[1])
        value_head_dim = int(value_cache.shape[-1])
        states: dict[str, OnlineSoftmaxState] = {}
        queries: dict[str, torch.Tensor] = {}
        for item in meta.items:
            item_query = query[item.query_begin : item.query_end]
            if item_query.shape[0] != meta.q_len:
                raise AssertionError("SpecStream query packing is not batch-uniform")
            queries[item.rid] = item_query
            states[item.rid] = init_online_softmax_state(
                item_query, num_kv_heads, value_head_dim
            )

        stream_items = [
            item for item in meta.items if item.stream_enabled and item.history_len > 0
        ]
        if meta.full_restore_baseline:
            for item in stream_items:
                states[item.rid] = self._restore_history(
                    item, queries[item.rid], states[item.rid], meta, layer
                )
        elif meta.cohort_enabled and len(stream_items) > 1:
            self._stream_history_cohorts(stream_items, queries, states, meta, layer)
        else:
            for item in stream_items:
                states[item.rid] = self._stream_history_single(
                    item, queries[item.rid], states[item.rid], meta, layer
                )

        output = torch.empty(
            (query.shape[0], query.shape[1], value_head_dim),
            dtype=query.dtype,
            device=query.device,
        )
        for item in meta.items:
            started = time.perf_counter()
            slots = forward_batch.req_to_token_pool.req_to_token[
                item.req_pool_idx, item.history_len : item.logical_len
            ].long()
            if slots.numel() == 0:
                raise AssertionError("SpecStream Tail/Frontier cannot be empty")
            if self.config.reference_attention or float(layer.logit_cap or 0.0) > 0:
                tail_key = key_cache.index_select(0, slots)
                tail_value = value_cache.index_select(0, slots)
                query_positions = torch.arange(
                    item.committed_len,
                    item.logical_len,
                    dtype=torch.int64,
                    device=query.device,
                )
                key_positions = torch.arange(
                    item.history_len,
                    item.logical_len,
                    dtype=torch.int64,
                    device=query.device,
                )
                states[item.rid] = update_online_softmax_state(
                    states[item.rid],
                    queries[item.rid],
                    tail_key,
                    tail_value,
                    query_positions,
                    key_positions,
                    scale=layer.scaling,
                    causal=True,
                    softcap=float(layer.logit_cap or 0.0),
                )
            else:
                states[item.rid], _ = update_gpu_tail_state(
                    states[item.rid],
                    queries[item.rid],
                    key_cache,
                    value_cache,
                    query_position_start=item.committed_len,
                    key_position_start=item.history_len,
                    token_indices=slots,
                    scale=layer.scaling,
                    require_fused_cuda=True,
                )
            item_output, _ = finalize_online_softmax_state(states[item.rid])
            output[item.query_begin : item.query_end] = item_output.to(query.dtype)
            self.profiler.record_attention(
                meta.round_id,
                (time.perf_counter() - started) * 1000,
                tail=True,
            )

        return output.view(-1, int(layer.tp_q_head_num) * value_head_dim)

    def _update_history_state(
        self,
        state: OnlineSoftmaxState,
        query: torch.Tensor,
        packed: torch.Tensor,
        *,
        meta: SpecStreamRoundMeta,
        item: SpecStreamRequestMeta,
        layer,
    ) -> OnlineSoftmaxState:
        if self.config.reference_attention or float(layer.logit_cap or 0.0) > 0:
            return update_online_softmax_state(
                state,
                query,
                packed[:, 0],
                packed[:, 1],
                tuple(range(query.shape[0])),
                tuple(range(packed.shape[0])),
                scale=layer.scaling,
                causal=False,
                softcap=float(layer.logit_cap or 0.0),
            )

        shadow = None
        if self.config.shadow_attention:
            shadow = update_online_softmax_state(
                OnlineSoftmaxState(
                    state.max_score.clone(),
                    state.normalizer.clone(),
                    state.weighted_value.clone(),
                ),
                query,
                packed[:, 0],
                packed[:, 1],
                tuple(range(query.shape[0])),
                tuple(range(packed.shape[0])),
                scale=layer.scaling,
                causal=False,
            )
        updated, _ = update_packed_history_state(
            state,
            query,
            packed,
            scale=layer.scaling,
            require_fused_cuda=True,
        )
        if shadow is not None:
            candidate_output, _ = finalize_online_softmax_state(updated)
            reference_output, _ = finalize_online_softmax_state(shadow)
            self.diagnostics.record_shadow(
                rid=item.rid,
                round_id=meta.round_id,
                layer_id=layer.layer_id,
                candidate=candidate_output,
                reference=reference_output,
            )
        return updated

    def _stream_history_single(self, item, query, state, meta, layer):
        chunks = list(
            self.history_store.iter_layer_chunks(
                item.rid, layer.layer_id, history_end=item.history_len
            )
        )
        if sum(chunk.length for chunk in chunks) != item.history_len:
            raise AssertionError("CPU History length does not match tiered state")
        if not chunks:
            return state

        # B2 deliberately remains the one-chunk Torch reference.  B3 groups
        # adjacent sealed slabs into a larger bounded window, so one ready/free
        # event and one tiled attention launch cover several CPU chunks.
        group_size = (
            1 if self.config.reference_attention else self.config.chunks_per_transfer
        )
        chunk_groups = _group_adjacent_chunks(chunks, group_size)
        transfers = {}
        for index in range(min(self.staging.num_buffers, len(chunk_groups))):
            transfers[index] = self.staging.submit_many(
                [chunk.tensor for chunk in chunk_groups[index]],
                index,
            )

        for index, chunk_group in enumerate(chunk_groups):
            transfer = transfers.pop(index)
            packed = self.staging.wait_ready(transfer)
            started = time.perf_counter()
            state = self._update_history_state(
                state, query, packed, meta=meta, item=item, layer=layer
            )
            self.profiler.record_attention(
                meta.round_id, (time.perf_counter() - started) * 1000
            )
            self.profiler.record_h2d(
                meta.round_id,
                transfer.nbytes,
                (time.perf_counter_ns() - transfer.submitted_ns) / 1e6,
            )
            self.staging.mark_consumed(transfer)
            next_index = index + self.staging.num_buffers
            if next_index < len(chunk_groups):
                transfers[next_index] = self.staging.submit_many(
                    [chunk.tensor for chunk in chunk_groups[next_index]],
                    transfer.slot,
                )
        return state

    def _restore_history(self, item, query, state, meta, layer):
        chunks = list(
            self.history_store.iter_layer_chunks(
                item.rid, layer.layer_id, history_end=item.history_len
            )
        )
        if not chunks:
            return state
        first = chunks[0].tensor
        host = _empty_host(
            (item.history_len, *first.shape[1:]),
            dtype=first.dtype,
            pin_memory=torch.cuda.is_available(),
        )
        cursor = 0
        for chunk in chunks:
            host[cursor : cursor + chunk.length].copy_(chunk.tensor)
            cursor += chunk.length
        started = time.perf_counter()
        device_history = host.to(query.device, non_blocking=bool(host.is_pinned()))
        state = self._update_history_state(
            state,
            query,
            device_history,
            meta=meta,
            item=item,
            layer=layer,
        )
        elapsed = (time.perf_counter() - started) * 1000
        self.profiler.record_h2d(meta.round_id, host.nbytes, elapsed)
        self.profiler.record_attention(meta.round_id, elapsed)
        return state

    def _stream_history_cohorts(self, items, queries, states, meta, layer) -> None:
        chunks_by_rid = {
            item.rid: list(
                self.history_store.iter_layer_chunks(
                    item.rid, layer.layer_id, history_end=item.history_len
                )
            )
            for item in items
        }
        for item in items:
            if (
                sum(chunk.length for chunk in chunks_by_rid[item.rid])
                != item.history_len
            ):
                raise AssertionError("CPU History length does not match tiered state")

        group_size = (
            1 if self.config.reference_attention else self.config.chunks_per_transfer
        )
        groups_by_rid = {
            rid: _group_adjacent_chunks(chunks, group_size)
            for rid, chunks in chunks_by_rid.items()
        }
        deadline_ns = time.time_ns() + int(self.config.max_cohort_delay_us * 1000)
        request_by_rid = {item.rid: item for item in items}
        work_items = []
        for item in items:
            groups = groups_by_rid[item.rid]
            if not groups:
                continue
            first_group = tuple(groups[0])
            work_items.append(
                StreamWorkItem(
                    rid=item.rid,
                    round_id=meta.round_id,
                    layer_id=layer.layer_id,
                    q_len=meta.q_len,
                    chunk_idx=0,
                    token_begin=first_group[0].abs_start,
                    token_end=first_group[-1].abs_start + first_group[-1].length,
                    bytes=sum(chunk.tensor.nbytes for chunk in first_group),
                    deadline_ns=deadline_ns,
                    cpu_descriptor=first_group,
                )
            )
        plans = build_cohort_plans(
            work_items,
            max_cohort_size=self.config.max_cohort_size,
            num_staging_slots=self.staging.num_buffers,
            max_cohort_delay_us=self.config.max_cohort_delay_us,
        )

        transfer_index = 0
        for plan in plans:
            cohort_queries = torch.stack([queries[work.rid] for work in plan.items])
            max_plan_groups = max(len(groups_by_rid[work.rid]) for work in plan.items)
            batched_state = None
            if not self.config.reference_attention:
                batched_state = stack_packed_history_cohort_states(
                    [states[work.rid] for work in plan.items]
                )

            for group_index in range(max_plan_groups):
                slot = transfer_index % self.staging.num_buffers
                transfer_index += 1
                source_groups = []
                for work in plan.items:
                    request_groups = groups_by_rid[work.rid]
                    chunk_group = (
                        request_groups[group_index]
                        if group_index < len(request_groups)
                        else ()
                    )
                    source_groups.append([chunk.tensor for chunk in chunk_group])
                transfer = self.staging.submit_cohort_groups(source_groups, slot)
                packed = self.staging.wait_ready(transfer)
                started = time.perf_counter()
                if self.config.reference_attention:
                    for index, work in enumerate(plan.items):
                        valid = transfer.valid_lengths[index]
                        if valid == 0:
                            continue
                        states[work.rid] = self._update_history_state(
                            states[work.rid],
                            queries[work.rid],
                            packed[index, :valid],
                            meta=meta,
                            item=request_by_rid[work.rid],
                            layer=layer,
                        )
                else:
                    if transfer.valid_tokens is None:
                        raise AssertionError("cohort transfer is missing valid lengths")
                    if batched_state is None:
                        raise AssertionError("cohort state was not initialized")
                    batched_state, _ = update_packed_history_cohort_batched(
                        batched_state,
                        cohort_queries,
                        packed,
                        transfer.valid_tokens,
                        scale=layer.scaling,
                        require_fused_cuda=True,
                    )
                elapsed = (time.perf_counter() - started) * 1000
                self.profiler.record_h2d(
                    meta.round_id,
                    transfer.nbytes,
                    (time.perf_counter_ns() - transfer.submitted_ns) / 1e6,
                    cohort_size=len(plan.items),
                )
                self.profiler.record_attention(meta.round_id, elapsed)
                self.staging.mark_consumed(transfer)

            if batched_state is not None:
                for work, new_state in zip(
                    plan.items, split_packed_history_cohort_state(batched_state)
                ):
                    states[work.rid] = new_state


class SpecStreamTargetRuntime:
    """Target-rank owner of tiered state, storage, control and profiling."""

    def __init__(
        self,
        *,
        config: SpecStreamConfig,
        model_runner,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        self.config = config
        self.model_runner = model_runner
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.states: dict[str, TargetTieredKVState] = {}
        self._round_id = 0

        kv_pool = self.token_to_kv_pool
        if not all(
            hasattr(kv_pool, attr) for attr in ("head_num", "head_dim", "v_head_dim")
        ):
            raise RuntimeError("SpecStream v1 supports MHA/GQA Target KV pools only")
        if int(kv_pool.head_dim) != int(kv_pool.v_head_dim):
            raise RuntimeError(
                "SpecStream packed v1 requires equal K/V head dimensions"
            )
        layer_ids = tuple(
            range(
                int(kv_pool.start_layer),
                int(kv_pool.start_layer) + int(kv_pool.layer_num),
            )
        )
        self.history_store = CPUHistoryStore(
            max_memory_bytes=config.cpu_memory_bytes,
            chunk_tokens=config.chunk_tokens,
            layer_ids=layer_ids,
            kv_heads=int(kv_pool.head_num),
            head_dim=int(kv_pool.head_dim),
            dtype=kv_pool.dtype,
            device=model_runner.device,
        )
        self.staging = StagingWindowPool(config.num_buffers, model_runner.device)
        if config.enabled and not config.full_restore_baseline:
            reserve_chunks = (
                1 if config.reference_attention else config.chunks_per_transfer
            )
            reserve_shape = (
                (
                    config.max_cohort_size,
                    config.chunk_tokens * reserve_chunks,
                    2,
                    int(kv_pool.head_num),
                    int(kv_pool.head_dim),
                )
                if config.cohort_enabled
                else (
                    config.chunk_tokens * reserve_chunks,
                    2,
                    int(kv_pool.head_num),
                    int(kv_pool.head_dim),
                )
            )
            self.staging.reserve(reserve_shape, kv_pool.dtype)
            if config.cohort_enabled:
                self.staging.reserve_cohort_pack(reserve_shape, kv_pool.dtype)
        if config.enabled and not config.reference_attention:
            logger.info(
                "SpecStream B3 tiled path: chunk_tokens=%d, "
                "chunks_per_transfer=%d, buffers=%d, reserved_staging_bytes=%d, "
                "reserved_host_pack_bytes=%d",
                config.chunk_tokens,
                config.chunks_per_transfer,
                config.num_buffers,
                self.staging.allocated_bytes,
                self.staging.allocated_host_bytes,
            )
        self.profiler = SpecStreamProfiler(config.profile_path, tp_rank, tp_size)
        self.diagnostics = SpecStreamDiagnostics(
            config.profile_path, config.shadow_attention
        )
        self.acceptance_tracker = AcceptanceTracker()
        self.controller = (
            IOAwareController(config.q_candidates, config.q_switch_threshold)
            if config.dynamic_q
            else None
        )
        self.verifier = SpecStreamVerifier(
            config=config,
            history_store=self.history_store,
            staging=self.staging,
            profiler=self.profiler,
            diagnostics=self.diagnostics,
        )

    def batch_requires_streaming(self, batch) -> bool:
        return any(
            self.states.get(req.rid) is not None and self.states[req.rid].stream_enabled
            for req in batch.reqs
        )

    def build_round_meta(self, batch, spec_info, mode: str) -> SpecStreamRoundMeta:
        self._round_id += 1
        q_len = int(spec_info.draft_token_num)
        items = []
        for index, req in enumerate(batch.reqs):
            committed_len = int(batch.seq_lens_cpu[index].item())
            state = self.states.setdefault(
                req.rid,
                TargetTieredKVState(
                    rid=req.rid,
                    committed_len=committed_len,
                    logical_len=committed_len,
                ),
            )
            state.begin_round(committed_len, committed_len + q_len)
            if self.config.strict_invariants:
                state.check(self.config.chunk_tokens)
            query_begin = index * q_len
            items.append(
                SpecStreamRequestMeta(
                    rid=req.rid,
                    req_pool_idx=int(req.req_pool_idx),
                    query_begin=query_begin,
                    query_end=query_begin + q_len,
                    committed_len=committed_len,
                    history_len=state.history_len,
                    logical_len=committed_len + q_len,
                    stream_enabled=state.stream_enabled,
                )
            )
            self._publish_req_state(req, state, mode, q_len)
        enabled = bool(
            self.config.enabled and any(item.stream_enabled for item in items)
        )
        meta = SpecStreamRoundMeta(
            round_id=self._round_id,
            q_len=q_len,
            mode=mode,
            items=tuple(items),
            enabled=enabled,
            cohort_enabled=bool(enabled and self.config.cohort_enabled),
            full_restore_baseline=bool(enabled and self.config.full_restore_baseline),
            fallback=bool(
                getattr(batch, "spectre_draft_timeout", False)
                or getattr(batch, "spectre_policy_fallback", False)
            ),
            fallback_reason=str(getattr(batch, "spectre_fallback_reason", "")),
            missing_draft_count=len(
                getattr(batch, "spectre_missing_draft_rids", ()) or ()
            ),
        )
        self.profiler.begin_round(meta, chunk_tokens=self.config.chunk_tokens)
        batch.specstream_meta = meta
        spec_info.specstream_meta = meta
        return meta

    def record_target_forward(self, meta, elapsed_ms: float) -> None:
        if meta is not None:
            self.profiler.record_target_forward(meta.round_id, elapsed_ms)

    def record_logit_margin(self, meta, logits) -> None:
        if meta is not None:
            self.diagnostics.record_logit_margin(round_id=meta.round_id, logits=logits)

    def after_verify(self, batch, result, meta) -> None:
        accepted = list(result.accept_length_per_req_cpu)
        self.acceptance_tracker.update(meta.q_len, accepted)
        for index, req in enumerate(batch.reqs):
            state = self.states[req.rid]
            committed_len = int(batch.seq_lens_cpu[index].item())
            state.finish_round(committed_len)
            self._maybe_seal(req, state)
            if self.config.strict_invariants:
                state.check(self.config.chunk_tokens)
            self._publish_req_state(req, state, meta.mode, meta.q_len)
        self.profiler.finish_round(
            meta.round_id,
            accepted_tokens=sum(int(value) + 1 for value in accepted),
            staging_bytes=self.staging.allocated_bytes,
            cpu_bytes=self.history_store.bytes_used,
        )

    def after_normal_decode(self, batch, result) -> None:
        for index, req in enumerate(batch.reqs):
            committed_len = int(batch.seq_lens_cpu[index].item())
            state = self.states.setdefault(
                req.rid,
                TargetTieredKVState(
                    rid=req.rid,
                    committed_len=committed_len,
                    logical_len=committed_len,
                ),
            )
            state.committed_len = committed_len
            state.logical_len = committed_len
            self._maybe_seal(req, state)
            self._publish_req_state(req, state, "ordinary", 1)

    def after_extend(self, batch) -> None:
        """Publish prefill state without sealing during prefix-cache bookkeeping.

        A retracted sticky request may be re-prefilled.  In that case its CPU
        History remains authoritative, so discard the newly materialized GPU
        copy of the already sealed prefix after the forward has completed.
        """
        for index, req in enumerate(batch.reqs):
            committed_len = int(batch.seq_lens_cpu[index].item())
            state = self.states.get(req.rid)
            if state is None:
                state = TargetTieredKVState(
                    rid=req.rid,
                    committed_len=committed_len,
                    logical_len=committed_len,
                )
                self.states[req.rid] = state
            else:
                if committed_len < state.history_len:
                    raise AssertionError("re-prefill is shorter than sealed History")
                if state.history_len:
                    slots = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, : state.history_len
                    ].clone()
                    live_slots = slots[slots != 0]
                    if live_slots.numel():
                        self.token_to_kv_pool_allocator.free(live_slots)
                    self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, : state.history_len
                    ] = 0
                state.committed_len = committed_len
                state.logical_len = committed_len
            self._publish_req_state(req, state, "parallel", 1)

    def _maybe_seal(self, req, state: TargetTieredKVState) -> None:
        if not self.config.enabled or state.seal_inflight:
            return
        eligible_end = max(0, state.committed_len - self.config.active_tail_tokens)
        seal_end = (eligible_end // self.config.chunk_tokens) * self.config.chunk_tokens
        if seal_end < self.config.min_history_tokens or seal_end <= state.history_len:
            return
        slots = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, state.history_len : seal_end
        ].clone()
        if slots.numel() == 0:
            return
        state.start_seal()
        try:
            ticket = self.history_store.seal_slots_async(
                rid=req.rid,
                abs_start=state.history_len,
                slots=slots,
                token_to_kv_pool=self.token_to_kv_pool,
            )
            # Lifecycle ordering is deliberate: D2H completion -> allocator
            # free -> page-table tombstone -> history_len publication.
            ticket.wait_safe_to_free()
            self.token_to_kv_pool_allocator.free(slots)
            self.req_to_token_pool.req_to_token[
                req.req_pool_idx, state.history_len : seal_end
            ] = 0
            state.mark_sealed(seal_end, ticket.block_ids)
        except Exception:
            state.seal_inflight = False
            raise

    def collect_batch_state(self, batch) -> SpecStreamBatchState:
        histories = [
            self.states.get(req.rid, TargetTieredKVState(req.rid)).history_len
            for req in batch.reqs
        ]
        contexts = [
            self.states.get(req.rid, TargetTieredKVState(req.rid)).committed_len
            for req in batch.reqs
        ]
        bytes_per_token = (
            len(self.history_store.layer_ids)
            * 2
            * self.history_store.kv_heads
            * self.history_store.head_dim
            * torch.empty((), dtype=self.history_store.dtype).element_size()
        )
        no_draft = sum(not getattr(req, "cur_drafts", None) for req in batch.reqs)
        return SpecStreamBatchState(
            batch_size=len(batch.reqs),
            context_tokens=sum(contexts),
            history_tokens=sum(histories),
            history_bytes=sum(histories) * bytes_per_token,
            num_chunks=sum(
                math.ceil(value / self.config.chunk_tokens) for value in histories
            ),
            no_draft_ratio=no_draft / max(len(batch.reqs), 1),
            rejected=bool(getattr(batch, "specstream_rejected", False)),
            high_overhead=bool(getattr(batch, "is_high_overhead", False)),
        )

    def choose_decision(self, batch) -> SpecStreamDecision | None:
        if self.controller is None:
            return None
        return self.controller.choose(
            self.collect_batch_state(batch),
            self.profiler.snapshot(),
            self.acceptance_tracker.snapshot(self.config.q_candidates),
            allow_ordinary=True,
        )

    def record_network_wait(self, elapsed_ms: float) -> None:
        self.profiler.record_network_wait(elapsed_ms)

    def record_draft_result(
        self,
        *,
        elapsed_ms: float,
        timeout_ms: float,
        missing_count: int,
        total_count: int,
    ) -> None:
        if self.controller is not None:
            self.controller.record_draft_result(
                elapsed_ms=elapsed_ms,
                timeout_ms=timeout_ms,
                missing_count=missing_count,
                total_count=total_count,
            )

    def release_request(self, rid: str) -> None:
        self.states.pop(rid, None)
        self.history_store.release(rid)

    def clear(self) -> None:
        self.states.clear()
        self.history_store.clear()

    @staticmethod
    def _publish_req_state(req, state, mode: str, q: int) -> None:
        req.specstream_history_len = state.history_len
        req.specstream_committed_len = state.committed_len
        req.specstream_logical_len = state.logical_len
        req.specstream_stream_enabled = state.stream_enabled
        req.specstream_round_id = state.round_id
        req.specstream_mode = mode
        req.specstream_q = q
