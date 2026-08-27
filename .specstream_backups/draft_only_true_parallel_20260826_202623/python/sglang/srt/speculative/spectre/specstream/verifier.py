from __future__ import annotations

from dataclasses import dataclass
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
from sglang.srt.speculative.spectre.specstream.coexec_runtime import (
    TargetGrantRuntime,
)
from sglang.srt.speculative.spectre.specstream.controller import (
    IOAwareController,
    SpecStreamDecision,
)
from sglang.srt.speculative.spectre.specstream.cost_model import SpecStreamBatchState
from sglang.srt.speculative.spectre.specstream.cpu_history_store import (
    CPUHistoryStore,
    SealTicket,
)
from sglang.srt.speculative.spectre.specstream.diagnostics import (
    SpecStreamDiagnostics,
)
from sglang.srt.speculative.spectre.specstream.gpu_grant_controller import (
    GpuGrantController,
    GrantState,
)
from sglang.srt.speculative.spectre.specstream.sm_controller import SMController
from sglang.srt.speculative.spectre.specstream.tpc_partition import (
    ComplementaryTPCPartition,
    build_complementary_tpc_partition,
)
from sglang.srt.speculative.spectre.specstream.mps_env import read_mps_environment
from sglang.srt.speculative.spectre.specstream.multi_gpu_tp_policy import (
    MultiGPUTPPolicy,
)
from sglang.srt.speculative.spectre.specstream.online_softmax import (
    OnlineSoftmaxState,
    finalize_online_softmax_state,
    init_online_softmax_state,
    update_online_softmax_state,
)
from sglang.srt.speculative.spectre.specstream.profiler import SpecStreamProfiler
from sglang.srt.speculative.spectre.specstream.resource_profile import (
    ResourceProfile,
    context_bucket,
)
from sglang.srt.speculative.spectre.specstream.round_meta import (
    SpecStreamRequestMeta,
    SpecStreamRoundMeta,
)
from sglang.srt.speculative.spectre.specstream.slack_profiler import SlackProfiler
from sglang.srt.speculative.spectre.specstream.staging_runtime import (
    StagingTransfer,
    StagingWindowPool,
)
from sglang.srt.speculative.spectre.specstream.state import TargetTieredKVState
from sglang.srt.speculative.spectre.specstream.tp_straggler_monitor import (
    TPRankSample,
    TPStragglerMonitor,
    TPStragglerSnapshot,
)
from sglang.srt.speculative.spectre.specstream.triton_stream_attn import (
    split_packed_history_cohort_state,
    stack_packed_history_cohort_states,
    update_gpu_tail_state,
    update_packed_history_cohort_batched,
    update_packed_history_state,
)

logger = logging.getLogger(__name__)

# Host-packing a large cohort saves DMA descriptors but copies the complete KV
# payload once more on the scheduler thread.  Beyond this size the CPU copy is
# normally more expensive than enqueueing direct pinned-slab H2D operations.
_MAX_SYNCHRONOUS_COHORT_PACK_BYTES = 32 * 1024 * 1024


@dataclass
class _PendingSeal:
    rid: str
    req_pool_idx: int
    history_start: int
    seal_end: int
    slots: torch.Tensor
    ticket: SealTicket


@dataclass
class _BatchedLayerPrefetch:
    """First bounded-window transfers queued for a later transformer layer."""

    task_keys: tuple[tuple[object, ...], ...]
    transfers: dict[int, StagingTransfer]


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
        # Single-request B3 retains both bounded staging slots across
        # transformer layers.  Multi-request paths below use a separate global
        # work queue because these slots are shared by every request/cohort.
        self._single_layer_prefetch: dict[
            tuple[int, str, int], dict[int, StagingTransfer]
        ] = {}
        # Multi-request prefetch is stored per *whole ordered work queue*, not
        # per request.  The staging slots are global, so request-private
        # prefetch queues would overwrite one another at concurrency > 1.
        self._batched_layer_prefetch: dict[
            tuple[int, int, str, tuple[str, ...]], _BatchedLayerPrefetch
        ] = {}
        self._logged_batched_prefetch_modes: set[str] = set()
        self._logged_direct_cohort_staging = False

    def forward(self, *, q, k_new, v_new, forward_batch, meta, layer):
        if self._single_layer_prefetch:
            stale_keys = [
                key for key in self._single_layer_prefetch if key[0] != meta.round_id
            ]
            for key in stale_keys:
                del self._single_layer_prefetch[key]
        if self._batched_layer_prefetch:
            stale_keys = [
                key for key in self._batched_layer_prefetch if key[0] != meta.round_id
            ]
            for key in stale_keys:
                del self._batched_layer_prefetch[key]
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
            self._stream_history_cohorts(
                stream_items,
                queries,
                states,
                meta,
                layer,
                prefetch_next_layer=(
                    self.config.layer_prefetch and not self.config.reference_attention
                ),
            )
        elif (
            self.config.layer_prefetch
            and not self.config.reference_attention
            and len(stream_items) > 1
        ):
            self._stream_history_independent_batch(
                stream_items, queries, states, meta, layer
            )
        else:
            allow_layer_prefetch = self.config.layer_prefetch and len(stream_items) == 1
            for item in stream_items:
                states[item.rid] = self._stream_history_single(
                    item,
                    queries[item.rid],
                    states[item.rid],
                    meta,
                    layer,
                    prefetch_next_layer=allow_layer_prefetch,
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

    def _stream_history_single(
        self, item, query, state, meta, layer, *, prefetch_next_layer: bool = False
    ):
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
        next_prefetch = None

        def prefetch_next(available_slots=None):
            nonlocal next_prefetch
            if next_prefetch is None:
                next_layer_id = self._next_history_layer_id(int(layer.layer_id))
                if next_layer_id is None:
                    next_prefetch = ()
                else:
                    next_chunks = list(
                        self.history_store.iter_layer_chunks(
                            item.rid,
                            next_layer_id,
                            history_end=item.history_len,
                        )
                    )
                    next_prefetch = (
                        next_layer_id,
                        _group_adjacent_chunks(
                            next_chunks, self.config.chunks_per_transfer
                        ),
                    )
            self._prefetch_next_single_layer(
                item,
                meta,
                layer,
                available_slots=available_slots,
                prepared=next_prefetch,
            )

        prefetch_key = (meta.round_id, item.rid, int(layer.layer_id))
        transfers = self._single_layer_prefetch.pop(prefetch_key, {})
        for index in range(min(self.staging.num_buffers, len(chunk_groups))):
            if index not in transfers:
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
            elif prefetch_next_layer and not self.config.reference_attention:
                # The ring slot is no longer needed by layer L.  Reuse it for
                # L+1 immediately, while the other slot can still be running
                # the final History-attention group.  Waiting until the whole
                # History loop finishes leaves only Tail/OProj/MLP as overlap.
                prefetch_next((transfer.slot,))
        if prefetch_next_layer and not self.config.reference_attention:
            # Covers short/empty rings and fills any bounded slot which was not
            # stolen above.  Existing early transfers are merged, never redone.
            prefetch_next()
        return state

    def _prefetch_next_single_layer(
        self, item, meta, layer, *, available_slots=None, prepared=None
    ) -> None:
        if prepared is None:
            next_layer_id = self._next_history_layer_id(int(layer.layer_id))
            if next_layer_id is None:
                return
            chunks = list(
                self.history_store.iter_layer_chunks(
                    item.rid, next_layer_id, history_end=item.history_len
                )
            )
            chunk_groups = _group_adjacent_chunks(
                chunks, self.config.chunks_per_transfer
            )
        elif not prepared:
            return
        else:
            next_layer_id, chunk_groups = prepared
        if not chunk_groups:
            return
        key = (meta.round_id, item.rid, next_layer_id)
        transfers = self._single_layer_prefetch.setdefault(key, {})
        slots = (
            tuple(range(self.staging.num_buffers))
            if available_slots is None
            else tuple(int(slot) for slot in available_slots)
        )
        used_slots = {transfer.slot for transfer in transfers.values()}
        prefetch_limit = min(self.staging.num_buffers, len(chunk_groups))
        for slot in slots:
            if slot in used_slots:
                continue
            index = next(
                (
                    candidate
                    for candidate in range(prefetch_limit)
                    if candidate not in transfers
                ),
                None,
            )
            if index is None:
                break
            transfers[index] = self.staging.submit_many(
                [chunk.tensor for chunk in chunk_groups[index]], slot
            )
            used_slots.add(slot)

    def _next_history_layer_id(self, layer_id: int) -> int | None:
        try:
            layer_offset = self.history_store.layer_ids.index(int(layer_id))
        except ValueError:
            return None
        next_offset = layer_offset + 1
        if next_offset >= len(self.history_store.layer_ids):
            return None
        return int(self.history_store.layer_ids[next_offset])

    def _batched_prefetch_key(self, meta, layer_id: int, mode: str, items):
        return (
            int(meta.round_id),
            int(layer_id),
            mode,
            tuple(item.rid for item in items),
        )

    def _take_batched_prefetch(
        self, *, meta, layer_id: int, mode: str, items, task_keys
    ) -> dict[int, StagingTransfer]:
        key = self._batched_prefetch_key(meta, layer_id, mode, items)
        cached = self._batched_layer_prefetch.pop(key, None)
        if cached is None:
            return {}
        if cached.task_keys != tuple(task_keys):
            # Request order, history geometry, or cohort membership changed.
            # Copies already submitted on the copy stream are harmless, but
            # their slot views must never be consumed with different metadata.
            logger.debug("Discarding incompatible SpecStream layer prefetch")
            return {}
        logged_modes = getattr(self, "_logged_batched_prefetch_modes", set())
        if mode not in logged_modes:
            logger.info(
                "SpecStream multi-request layer prefetch active: "
                "mode=%s requests=%d queued_transfers=%d",
                mode,
                len(items),
                len(cached.transfers),
            )
            logged_modes.add(mode)
            self._logged_batched_prefetch_modes = logged_modes
        return cached.transfers

    def _store_batched_prefetch(
        self,
        *,
        meta,
        layer_id: int,
        mode: str,
        items,
        task_keys,
        transfers: dict[int, StagingTransfer],
    ) -> None:
        if not transfers:
            return
        key = self._batched_prefetch_key(meta, layer_id, mode, items)
        task_keys = tuple(task_keys)
        cached = self._batched_layer_prefetch.get(key)
        if cached is not None and cached.task_keys == task_keys:
            merged = dict(cached.transfers)
            for task_index, transfer in transfers.items():
                merged.setdefault(task_index, transfer)
            transfers = merged
        self._batched_layer_prefetch[key] = _BatchedLayerPrefetch(
            task_keys=task_keys, transfers=transfers
        )

    def _build_independent_tasks(self, items, layer_id: int):
        tasks = []
        task_keys = []
        for item in items:
            chunks = list(
                self.history_store.iter_layer_chunks(
                    item.rid, layer_id, history_end=item.history_len
                )
            )
            if sum(chunk.length for chunk in chunks) != item.history_len:
                raise AssertionError("CPU History length does not match tiered state")
            groups = _group_adjacent_chunks(chunks, self.config.chunks_per_transfer)
            for group_index, chunk_group in enumerate(groups):
                tasks.append((item, chunk_group))
                task_keys.append(("request", item.rid, group_index))
        return tasks, tuple(task_keys)

    def _stream_history_independent_batch(
        self, items, queries, states, meta, layer
    ) -> None:
        """Pipeline a multi-request layer through the shared bounded slots.

        This is the non-cohort fallback.  It deliberately flattens all request
        chunks into one queue: the staging pool is global, so independently
        prefetching two slots for every request is unsafe at concurrency > 1.
        """

        tasks, task_keys = self._build_independent_tasks(items, int(layer.layer_id))
        next_prefetch = None

        def prefetch_next(available_slots=None):
            nonlocal next_prefetch
            if next_prefetch is None:
                next_layer_id = self._next_history_layer_id(int(layer.layer_id))
                if next_layer_id is None:
                    next_prefetch = ()
                else:
                    next_tasks, next_task_keys = self._build_independent_tasks(
                        items, next_layer_id
                    )
                    next_prefetch = (
                        next_layer_id,
                        next_tasks,
                        next_task_keys,
                    )
            self._prefetch_next_independent_layer(
                items,
                meta,
                layer,
                available_slots=available_slots,
                prepared=next_prefetch,
            )

        transfers = self._take_batched_prefetch(
            meta=meta,
            layer_id=int(layer.layer_id),
            mode="independent",
            items=items,
            task_keys=task_keys,
        )

        def submit_task(task_index: int, slot: int):
            _, chunk_group = tasks[task_index]
            return self.staging.submit_many(
                [chunk.tensor for chunk in chunk_group], slot
            )

        for task_index in range(min(self.staging.num_buffers, len(tasks))):
            if task_index not in transfers:
                transfers[task_index] = submit_task(task_index, task_index)

        for task_index, (item, _) in enumerate(tasks):
            transfer = transfers.pop(task_index)
            packed = self.staging.wait_ready(transfer)
            started = time.perf_counter()
            states[item.rid] = self._update_history_state(
                states[item.rid],
                queries[item.rid],
                packed,
                meta=meta,
                item=item,
                layer=layer,
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
            next_index = task_index + self.staging.num_buffers
            if next_index < len(tasks):
                transfers[next_index] = submit_task(next_index, transfer.slot)
            else:
                prefetch_next((transfer.slot,))

        prefetch_next()

    def _prefetch_next_independent_layer(
        self, items, meta, layer, *, available_slots=None, prepared=None
    ) -> None:
        if prepared is None:
            next_layer_id = self._next_history_layer_id(int(layer.layer_id))
            if next_layer_id is None:
                return
            tasks, task_keys = self._build_independent_tasks(items, next_layer_id)
        elif not prepared:
            return
        else:
            next_layer_id, tasks, task_keys = prepared
        key = self._batched_prefetch_key(meta, next_layer_id, "independent", items)
        cached = self._batched_layer_prefetch.get(key)
        transfers = (
            dict(cached.transfers)
            if cached is not None and cached.task_keys == tuple(task_keys)
            else {}
        )
        slots = (
            tuple(range(self.staging.num_buffers))
            if available_slots is None
            else tuple(int(slot) for slot in available_slots)
        )
        used_slots = {transfer.slot for transfer in transfers.values()}
        prefetch_limit = min(self.staging.num_buffers, len(tasks))
        new_transfers = {}
        for slot in slots:
            if slot in used_slots:
                continue
            task_index = next(
                (
                    candidate
                    for candidate in range(prefetch_limit)
                    if candidate not in transfers
                ),
                None,
            )
            if task_index is None:
                break
            _, chunk_group = tasks[task_index]
            transfer = self.staging.submit_many(
                [chunk.tensor for chunk in chunk_group], slot
            )
            transfers[task_index] = transfer
            new_transfers[task_index] = transfer
            used_slots.add(slot)
        self._store_batched_prefetch(
            meta=meta,
            layer_id=next_layer_id,
            mode="independent",
            items=items,
            task_keys=task_keys,
            transfers=new_transfers,
        )

    def discard_layer_prefetch(self, rid: str | None = None) -> None:
        if rid is None:
            self._single_layer_prefetch.clear()
            self._batched_layer_prefetch.clear()
            return
        stale_keys = [key for key in self._single_layer_prefetch if key[1] == rid]
        for key in stale_keys:
            del self._single_layer_prefetch[key]
        stale_keys = [key for key in self._batched_layer_prefetch if rid in key[3]]
        for key in stale_keys:
            del self._batched_layer_prefetch[key]

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

    def _build_cohort_layer_work(self, items, meta, layer_id: int):
        chunks_by_rid = {
            item.rid: list(
                self.history_store.iter_layer_chunks(
                    item.rid, layer_id, history_end=item.history_len
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
                    layer_id=layer_id,
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
            # These requests are already members of one scheduled verify
            # batch; no additional cohort wait occurs here.  Using wall clock
            # after Python descriptor construction made a 200-us deadline
            # expire at c=16/32, silently turning every plan into size 1 and
            # also invalidating the prefetched next-layer plan.  Deadline
            # expiry belongs at admission/scheduling, not inside layer replay.
            now_ns=deadline_ns - 1,
        )

        tasks = []
        task_keys = []
        for plan in plans:
            max_plan_groups = max(len(groups_by_rid[work.rid]) for work in plan.items)
            plan_rids = tuple(work.rid for work in plan.items)
            for group_index in range(max_plan_groups):
                tasks.append((plan, group_index))
                task_keys.append(("cohort", plan_rids, group_index))
        return groups_by_rid, request_by_rid, plans, tasks, tuple(task_keys)

    def _submit_cohort_task(
        self, task, groups_by_rid, slot: int, *, direct_async: bool = False
    ):
        plan, group_index = task
        source_groups = []
        for work in plan.items:
            request_groups = groups_by_rid[work.rid]
            chunk_group = (
                request_groups[group_index] if group_index < len(request_groups) else ()
            )
            source_groups.append([chunk.tensor for chunk in chunk_group])
        source_bytes = sum(
            int(source.nbytes) for group in source_groups for source in group
        )
        avoid_sync_pack = (
            direct_async or source_bytes >= _MAX_SYNCHRONOUS_COHORT_PACK_BYTES
        )
        if avoid_sync_pack:
            if not direct_async and not getattr(
                self, "_logged_direct_cohort_staging", False
            ):
                logger.info(
                    "SpecStream cohort uses direct async H2D above %d MiB "
                    "to avoid synchronous CPU packing",
                    _MAX_SYNCHRONOUS_COHORT_PACK_BYTES // (1024 * 1024),
                )
                self._logged_direct_cohort_staging = True
            return self.staging.submit_cohort_groups_direct_async(source_groups, slot)
        return self.staging.submit_cohort_groups(source_groups, slot)

    def _stream_history_cohorts(
        self,
        items,
        queries,
        states,
        meta,
        layer,
        *,
        prefetch_next_layer: bool = False,
    ) -> None:
        (
            groups_by_rid,
            request_by_rid,
            plans,
            tasks,
            task_keys,
        ) = self._build_cohort_layer_work(items, meta, int(layer.layer_id))
        next_prefetch = None

        def prefetch_next(available_slots=None):
            nonlocal next_prefetch
            if next_prefetch is None:
                next_layer_id = self._next_history_layer_id(int(layer.layer_id))
                if next_layer_id is None:
                    next_prefetch = ()
                else:
                    (
                        next_groups_by_rid,
                        _,
                        _,
                        next_tasks,
                        next_task_keys,
                    ) = self._build_cohort_layer_work(items, meta, next_layer_id)
                    next_prefetch = (
                        next_layer_id,
                        next_groups_by_rid,
                        next_tasks,
                        next_task_keys,
                    )
            self._prefetch_next_cohort_layer(
                items,
                meta,
                layer,
                available_slots=available_slots,
                prepared=next_prefetch,
            )

        transfers = self._take_batched_prefetch(
            meta=meta,
            layer_id=int(layer.layer_id),
            mode="cohort",
            items=items,
            task_keys=task_keys,
        )

        for task_index in range(min(self.staging.num_buffers, len(tasks))):
            if task_index not in transfers:
                transfers[task_index] = self._submit_cohort_task(
                    tasks[task_index], groups_by_rid, task_index
                )

        task_index = 0
        for plan in plans:
            cohort_queries = torch.stack([queries[work.rid] for work in plan.items])
            max_plan_groups = max(len(groups_by_rid[work.rid]) for work in plan.items)
            batched_state = None
            if not self.config.reference_attention:
                batched_state = stack_packed_history_cohort_states(
                    [states[work.rid] for work in plan.items]
                )

            for group_index in range(max_plan_groups):
                expected_plan, expected_group = tasks[task_index]
                if expected_plan is not plan or expected_group != group_index:
                    raise AssertionError("SpecStream cohort work queue is inconsistent")
                transfer = transfers.pop(task_index)
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
                next_index = task_index + self.staging.num_buffers
                if next_index < len(tasks):
                    transfers[next_index] = self._submit_cohort_task(
                        tasks[next_index], groups_by_rid, transfer.slot
                    )
                elif prefetch_next_layer:
                    prefetch_next((transfer.slot,))
                task_index += 1

            if batched_state is not None:
                for work, new_state in zip(
                    plan.items, split_packed_history_cohort_state(batched_state)
                ):
                    states[work.rid] = new_state

        if task_index != len(tasks):
            raise AssertionError("SpecStream did not consume the full cohort queue")
        if prefetch_next_layer:
            prefetch_next()

    def _prefetch_next_cohort_layer(
        self, items, meta, layer, *, available_slots=None, prepared=None
    ) -> None:
        if prepared is None:
            next_layer_id = self._next_history_layer_id(int(layer.layer_id))
            if next_layer_id is None:
                return
            groups_by_rid, _, _, tasks, task_keys = self._build_cohort_layer_work(
                items, meta, next_layer_id
            )
        elif not prepared:
            return
        else:
            next_layer_id, groups_by_rid, tasks, task_keys = prepared
        key = self._batched_prefetch_key(meta, next_layer_id, "cohort", items)
        cached = self._batched_layer_prefetch.get(key)
        transfers = (
            dict(cached.transfers)
            if cached is not None and cached.task_keys == tuple(task_keys)
            else {}
        )
        slots = (
            tuple(range(self.staging.num_buffers))
            if available_slots is None
            else tuple(int(slot) for slot in available_slots)
        )
        used_slots = {transfer.slot for transfer in transfers.values()}
        prefetch_limit = min(self.staging.num_buffers, len(tasks))
        new_transfers = {}
        for slot in slots:
            if slot in used_slots:
                continue
            task_index = next(
                (
                    candidate
                    for candidate in range(prefetch_limit)
                    if candidate not in transfers
                ),
                None,
            )
            if task_index is None:
                break
            transfer = self._submit_cohort_task(
                tasks[task_index],
                groups_by_rid,
                slot,
                direct_async=True,
            )
            transfers[task_index] = transfer
            new_transfers[task_index] = transfer
            used_slots.add(slot)
        self._store_batched_prefetch(
            meta=meta,
            layer_id=next_layer_id,
            mode="cohort",
            items=items,
            task_keys=task_keys,
            transfers=new_transfers,
        )


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
        self._pending_seals: dict[str, _PendingSeal] = {}
        self._round_id = 0
        self._tp_sync_counter = 0
        self.target_smctrl = None
        self.target_total_tpcs = 0
        self._last_target_tpc_range: tuple[int, int] | None = None
        self._active_target_partition: ComplementaryTPCPartition | None = None

        if config.smctrl_enabled and config.smctrl_complementary_partition:
            self.target_smctrl = SMController(
                config.smctrl_library or None,
                # Do not derive the CUDA index from model_runner.device:
                # ModelRunner stores server_args.device here, which is the
                # string "cuda", not torch.device("cuda:N").
                #
                # SMController already defaults to
                # torch.cuda.current_device(), which is exactly what we want
                # after CUDA_VISIBLE_DEVICES remapping.
                mask_scope="global",
            )
            self.target_total_tpcs = int(self.target_smctrl.total_tpcs)
            if self.target_total_tpcs > 64:
                raise RuntimeError(
                    "SpecStream complementary process-global TPC partition "
                    "currently supports at most 64 TPCs"
                )
            self._set_target_tpc_range(0, self.target_total_tpcs)
            logger.info(
                "SpecStream complementary Target TPC controller ready: "
                "full=[0,%d)",
                self.target_total_tpcs,
            )

        self.mps_environment = read_mps_environment()
        if config.profile_only:
            logger.info(
                "SpecStream control-only mode: native SPECTRE Target KV remains "
                "GPU-resident; CPU History sealing and H2D streaming are disabled"
            )
        if config.coexec_require_mps and not self.mps_environment.configured:
            raise RuntimeError(
                "SpecStream co-execution requires CUDA MPS, but the Target "
                "process has no visible MPS quota, partition or pipe directory"
            )
        if config.coexec_enabled and not config.smctrl_enabled:
            logger.warning(
                "--specstream-coexec-enabled no longer changes q or grants GPU "
                "execution; enable --specstream-smctrl-enabled with a calibrated "
                "profile for innovation point 2"
            )
        if config.smctrl_enabled:
            logger.info(
                "SpecStream Target-priority SM control: mps=%s "
                "active_threads=%s priority=%s partition=%s profile=%s",
                self.mps_environment.configured,
                self.mps_environment.active_thread_percentage,
                self.mps_environment.client_priority,
                self.mps_environment.sm_partition or "none",
                config.coexec_resource_profile_path,
            )
        if config.colocated_tp_rank >= self.tp_size:
            raise ValueError("SpecStream colocated TP rank is outside the TP group")

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
                "chunks_per_transfer=%d, buffers=%d, layer_prefetch=%s, "
                "async_d2h_seal=%s, reserved_staging_bytes=%d, "
                "reserved_host_pack_bytes=%d",
                config.chunk_tokens,
                config.chunks_per_transfer,
                config.num_buffers,
                config.layer_prefetch,
                self.history_store.d2h_stream is not None,
                self.staging.allocated_bytes,
                self.staging.allocated_host_bytes,
            )
        calibrated_h2d_gbps = (
            self.staging.calibrate_h2d_gbps()
            if config.enabled and not config.full_restore_baseline
            else 0.0
        )
        if calibrated_h2d_gbps > 0:
            logger.info(
                "SpecStream pinned H2D startup calibration: %.3f GB/s",
                calibrated_h2d_gbps,
            )
        self.profiler = SpecStreamProfiler(
            config.profile_path,
            tp_rank,
            tp_size,
            mps_environment=self.mps_environment,
            calibrated_h2d_gbps=calibrated_h2d_gbps,
        )
        self.diagnostics = SpecStreamDiagnostics(
            config.profile_path, config.shadow_attention
        )
        self.acceptance_tracker = AcceptanceTracker()
        self.slack_profiler = SlackProfiler()
        self.grant_runtime = None
        if config.smctrl_enabled:
            calibration = config.smctrl_calibration_tpcs > 0
            resource_profile = (
                None
                if calibration
                else ResourceProfile.load(config.coexec_resource_profile_path)
            )
            self.grant_runtime = TargetGrantRuntime(
                GpuGrantController(
                    resource_profile,
                    target_slowdown_budget=(config.coexec_target_slowdown_budget),
                    guard_us=config.coexec_guard_us,
                    calibration_tpcs=config.smctrl_calibration_tpcs,
                    calibration_allow_overlap=(config.smctrl_calibration_allow_overlap),
                )
            )
            if calibration:
                logger.warning(
                    "SpecStream calibration mode: fixed Draft TPCs=%d overlap=%s; "
                    "do not report this as the online controller",
                    config.smctrl_calibration_tpcs,
                    config.smctrl_calibration_allow_overlap,
                )
        controller_candidates = (
            config.q_candidates if config.dynamic_q else (config.default_q,)
        )
        self.controller = (
            IOAwareController(
                controller_candidates,
                config.q_switch_threshold,
                multi_gpu_policy=(
                    MultiGPUTPPolicy(
                        rank_skew_budget_ms=config.tp_straggler_budget_ms,
                        target_slowdown_budget=config.target_slowdown_budget,
                    )
                    if config.tp_straggler_control
                    else None
                ),
            )
            if (config.dynamic_q or config.tp_straggler_control)
            else None
        )
        self.tp_monitor = (
            TPStragglerMonitor(
                tp_size=self.tp_size,
                colocated_rank=config.colocated_tp_rank,
            )
            if config.tp_straggler_control
            else None
        )
        self.verifier = SpecStreamVerifier(
            config=config,
            history_store=self.history_store,
            staging=self.staging,
            profiler=self.profiler,
            diagnostics=self.diagnostics,
        )

    def _set_target_tpc_range(self, low: int, high: int) -> None:
        # Set the Target process-wide TPC mask and cache redundant switches.
        if self.target_smctrl is None:
            return
        requested = (int(low), int(high))
        if requested == self._last_target_tpc_range:
            return
        stream = torch.cuda.current_stream()
        self.target_smctrl.set_stream_mask(stream, *requested)
        self._last_target_tpc_range = requested

    def _active_slack_fill_draft_ranges(self, batch) -> tuple[tuple[int, int], ...]:
        # Read the same outstanding SLACK_FILL grants that the Drafter obeys.
        if self.grant_runtime is None:
            return ()
        ranges = []
        for req in batch.reqs:
            if str(getattr(req, "rid", "")).startswith("HEALTH_CHECK"):
                continue
            state = self.grant_runtime.state_for(
                str(req.rid), int(getattr(req, "spec_cnt", 0) or 0)
            )
            if (
                state is None
                or state.outstanding_epoch is None
                or state.last_decision is None
                or state.last_decision.state is not GrantState.SLACK_FILL
            ):
                continue
            ranges.append(
                (
                    int(state.last_decision.tpc_low),
                    int(state.last_decision.tpc_high),
                )
            )
        return tuple(ranges)

    def begin_target_partition(self, batch, meta=None) -> bool:
        # During parallel SLACK_FILL: Draft=[0,k), Target=[k,N).
        if self.target_smctrl is None:
            return False
        if getattr(batch, "specstream_mode", "parallel") != "parallel":
            if meta is not None:
                self.profiler.record_target_partition(
                    meta.round_id, 0, self.target_total_tpcs
                )
            return False

        partition = build_complementary_tpc_partition(
            self.target_total_tpcs,
            self._active_slack_fill_draft_ranges(batch),
        )
        if partition is None:
            if meta is not None:
                self.profiler.record_target_partition(
                    meta.round_id, 0, self.target_total_tpcs
                )
            return False

        self._set_target_tpc_range(partition.target_low, partition.target_high)
        self._active_target_partition = partition
        if meta is not None:
            self.profiler.record_target_partition(
                meta.round_id, partition.target_low, partition.target_high
            )
        logger.debug(
            "[SpecStream][TPC] complementary overlap Draft=[%d,%d) "
            "Target=[%d,%d)",
            partition.draft_low,
            partition.draft_high,
            partition.target_low,
            partition.target_high,
        )
        return True

    def end_target_partition(self) -> None:
        # Restore Target=[0,N) only after the asynchronous Target forward ends.
        if self.target_smctrl is None or self._active_target_partition is None:
            return
        self._set_target_tpc_range(0, self.target_total_tpcs)
        self._active_target_partition = None

    def batch_requires_streaming(self, batch) -> bool:
        self._poll_pending_seals()
        return any(
            self.states.get(req.rid) is not None and self.states[req.rid].stream_enabled
            for req in batch.reqs
        )

    def build_round_meta(self, batch, spec_info, mode: str) -> SpecStreamRoundMeta:
        self._poll_pending_seals()
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
            coexec_mode=str(
                getattr(
                    getattr(batch, "specstream_decision", None),
                    "coexec_mode",
                    "COEXEC" if mode == "parallel" else "SERIALIZE",
                )
            ),
        )
        self.profiler.begin_round(meta, chunk_tokens=self.config.chunk_tokens)
        batch.specstream_meta = meta
        spec_info.specstream_meta = meta
        return meta

    def record_target_forward(
        self, meta, elapsed_ms: float, enqueue_ms: float = 0.0
    ) -> None:
        self.slack_profiler.record_target_phase("target_forward", elapsed_ms)
        if self.grant_runtime is not None:
            self.grant_runtime.record_target_forward(elapsed_ms)
        if meta is not None:
            self.profiler.record_target_forward(
                meta.round_id, elapsed_ms, enqueue_ms=enqueue_ms
            )
            if self.tp_monitor is not None:
                self.tp_monitor.record_local(
                    TPRankSample(
                        rank=self.tp_rank,
                        round_id=meta.round_id,
                        target_forward_ms=float(elapsed_ms),
                    )
                )

    def record_logit_margin(self, meta, logits) -> None:
        if meta is not None:
            self.diagnostics.record_logit_margin(round_id=meta.round_id, logits=logits)

    def after_verify(self, batch, result, meta) -> None:
        self._poll_pending_seals()
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
        self._poll_pending_seals()
        self.profiler.finish_round(
            meta.round_id,
            accepted_tokens=sum(int(value) + 1 for value in accepted),
            staging_bytes=self.staging.allocated_bytes,
            cpu_bytes=self.history_store.bytes_used,
        )

    def after_normal_decode(self, batch, result) -> None:
        self._poll_pending_seals()
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
        self._poll_pending_seals()

    def after_extend(self, batch) -> None:
        """Publish prefill state and seal only requests whose prefill is complete.

        A retracted sticky request may be re-prefilled.  In that case its CPU
        History remains authoritative, so discard the newly materialized GPU
        copy of the already sealed prefix after the forward has completed.
        Intermediate chunked-prefill requests must remain fully GPU-resident
        because their next extend attention still consumes the full prefix.
        """
        self._poll_pending_seals()
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
            if getattr(req, "is_chunked", 0) <= 0:
                # Start offload as soon as the final prefill chunk has produced
                # its KV.  Waiting for the first verify round lets many long
                # prompts fill the token pool before any slot is recycled.
                self._maybe_seal(req, state)
            self._publish_req_state(req, state, "parallel", 1)
        self._poll_pending_seals()

    def _maybe_seal(self, req, state: TargetTieredKVState) -> None:
        if (
            not self.config.enabled
            or state.seal_inflight
            or req.rid in self._pending_seals
        ):
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
            # Do not wait here.  Until the completion event is observed, this
            # prefix remains GPU-resident and history_len is intentionally not
            # published, so ordinary Target attention can keep using it.
            self._pending_seals[req.rid] = _PendingSeal(
                rid=req.rid,
                req_pool_idx=int(req.req_pool_idx),
                history_start=state.history_len,
                seal_end=seal_end,
                slots=slots,
                ticket=ticket,
            )
            logger.debug(
                "[SpecStream][D2H] queued rid=%s range=[%d,%d) pending=%d",
                req.rid,
                state.history_len,
                seal_end,
                len(self._pending_seals),
            )
        except Exception:
            state.seal_inflight = False
            raise

    def _poll_pending_seals(
        self, *, wait: bool = False, only_rid: str | None = None
    ) -> int:
        """Publish completed seals and free GPU slots in lifecycle order.

        Normal decode/verify calls use the nonblocking form.  ``wait=True`` is
        reserved for terminal request/cache cleanup, where SGLang is about to
        release the same KV slots through its generic allocator path.
        """

        completed = 0
        pending_items = list(self._pending_seals.items())
        for rid, pending in pending_items:
            if only_rid is not None and rid != only_rid:
                continue
            if not self.history_store.complete_seal(pending.ticket, wait=wait):
                continue
            state = self.states.get(rid)
            if state is None:
                raise AssertionError("completed seal has no Target tiered state")

            # Lifecycle ordering: D2H completion -> allocator free -> page-table
            # tombstone -> history_len publication.  All page-table operations
            # are issued on the current Target stream before the next forward.
            self.token_to_kv_pool_allocator.free(pending.slots)
            self.req_to_token_pool.req_to_token[
                pending.req_pool_idx, pending.history_start : pending.seal_end
            ] = 0
            state.mark_sealed(pending.seal_end, pending.ticket.block_ids)
            del self._pending_seals[rid]
            completed += 1
            logger.debug(
                "[SpecStream][D2H] completed rid=%s history_len=%d pending=%d",
                rid,
                state.history_len,
                len(self._pending_seals),
            )
        return completed

    def prepare_request_release(self, rid: str) -> None:
        """Drain a terminal request before SGLang's generic KV release path."""

        if rid in self._pending_seals:
            self._poll_pending_seals(wait=True, only_rid=rid)

    def collect_batch_state(self, batch) -> SpecStreamBatchState:
        self._poll_pending_seals()
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
        q_candidates = self.controller.q_candidates
        return self.controller.choose(
            self.collect_batch_state(batch),
            self.profiler.snapshot(),
            self.acceptance_tracker.snapshot(q_candidates),
            allow_ordinary=True,
            draft_load=self.controller.draft_load_tracker.snapshot(),
            tp_snapshot=(
                self.tp_monitor.snapshot() if self.tp_monitor is not None else None
            ),
        )

    def record_decision(self, decision: SpecStreamDecision | None) -> None:
        if decision is None or self.controller is None:
            return
        self.profiler.record_decision(
            decision,
            self.controller.draft_load_tracker.snapshot(),
            (
                self.tp_monitor.snapshot()
                if self.tp_monitor is not None
                else TPStragglerSnapshot()
            ),
        )

    def should_sync_tp_profile(self) -> bool:
        if self.tp_monitor is None or self.tp_size < 2:
            return False
        self._tp_sync_counter += 1
        if self.tp_monitor.snapshot().samples == 0:
            return True
        return self._tp_sync_counter % self.config.tp_monitor_interval == 0

    def local_tp_rank_sample(self) -> TPRankSample | None:
        return self.tp_monitor.local_sample() if self.tp_monitor is not None else None

    def record_tp_rank_samples(self, samples) -> None:
        if self.tp_monitor is None:
            return
        self.tp_monitor.observe(samples)
        self.profiler.record_tp_snapshot(self.tp_monitor.snapshot())

    def record_network_wait(self, elapsed_ms: float) -> None:
        self.profiler.record_network_wait(elapsed_ms)

    def prepare_initial_grants(self, batch, desired_q: int):
        if self.grant_runtime is None:
            return []
        reqs = [
            req for req in batch.reqs if not str(req.rid).startswith("HEALTH_CHECK")
        ]
        return self._register_grant_reqs(reqs, desired_q)

    def prepare_retry_grants(self, reqs, desired_q: int):
        if self.grant_runtime is None:
            return []
        reqs = [req for req in reqs if not str(req.rid).startswith("HEALTH_CHECK")]
        return self._register_grant_reqs(reqs, desired_q)

    def _register_grant_reqs(self, reqs, desired_q: int):
        if not reqs:
            return []
        batch_size = len(reqs)
        max_context = max(
            len(getattr(req, "origin_input_ids", ()) or ())
            + len(getattr(req, "output_ids", ()) or ())
            for req in reqs
        )
        ctx_bucket = context_bucket(max_context)
        target_shape = f"verify_bs{batch_size}_q{int(desired_q)}_ctx{ctx_bucket}"
        predicted_slack_us = self.slack_profiler.snapshot(
            "target_forward"
        ).predicted_slack_us
        keys = []
        for req in reqs:
            key = (str(req.rid), int(req.spec_cnt))
            keys.append(key)
            self.grant_runtime.register_round(
                request_id=key[0],
                spec_cnt=key[1],
                desired_q=int(desired_q),
                target_shape=target_shape,
                draft_bs=batch_size,
                draft_ctx_bucket=ctx_bucket,
                predicted_slack_us=predicted_slack_us,
            )
        messages = self.grant_runtime.initial_grants(keys)
        for message in messages:
            self.profiler.record_grant(message, target_phase="target_forward")
        if not messages:
            for key in keys:
                state = self.grant_runtime.state_for(*key)
                if state is not None and state.last_decision is not None:
                    self.profiler.record_grant_decision(
                        state.last_decision, target_phase="target_forward"
                    )
                    break
        return messages

    def overlap_grants(self, keys, *, now_us: int | None = None):
        if self.grant_runtime is None:
            return []
        messages = self.grant_runtime.overlap_grants(
            list(keys), now_us=now_us
        )
        for message in messages:
            self.profiler.record_grant(message, target_phase="target_forward")
        return messages

    def waiting_grants(self, keys, *, deadline_us: int):
        if self.grant_runtime is None:
            return []
        messages = self.grant_runtime.waiting_grants(
            list(keys), deadline_us=int(deadline_us)
        )
        for message in messages:
            self.profiler.record_grant(message, target_phase="target_wait")
        if not messages:
            for key in keys:
                state = self.grant_runtime.state_for(*key)
                if state is not None and state.last_decision is not None:
                    self.profiler.record_grant_decision(
                        state.last_decision, target_phase="target_wait"
                    )
                    break
        return messages

    def acknowledge_grant(self, message) -> bool:
        if self.grant_runtime is None:
            return False
        accepted = self.grant_runtime.acknowledge(message)
        if accepted:
            state = self.grant_runtime.state_for(
                str(message.request_id), int(message.spec_cnt or 0)
            )
            self.profiler.record_grant_ack(
                message,
                wait_ms=(state.last_grant_wait_ms if state is not None else 0.0),
            )
            if float(getattr(message, "draft_step_ms", 0.0) or 0.0) > 0:
                self.slack_profiler.record_draft_step(message.draft_step_ms)
        return accepted

    def pause_grants(self, keys):
        if self.grant_runtime is None:
            return []
        return self.grant_runtime.pause_messages(list(keys))

    def record_draft_result(
        self,
        *,
        elapsed_ms: float,
        rtt_ms: float | None = None,
        timeout_ms: float,
        missing_count: int,
        total_count: int,
    ) -> None:
        if self.controller is not None:
            self.controller.record_draft_result(
                elapsed_ms=elapsed_ms,
                rtt_ms=rtt_ms,
                timeout_ms=timeout_ms,
                missing_count=missing_count,
                total_count=total_count,
            )
            self.profiler.record_draft_load(
                self.controller.draft_load_tracker.snapshot()
            )

    def record_draft_reject(self) -> None:
        if self.controller is None:
            return
        self.controller.record_draft_reject()
        self.profiler.record_draft_load(self.controller.draft_load_tracker.snapshot())

    def release_request(self, rid: str) -> None:
        # The scheduler normally calls prepare_request_release() before its
        # generic KV release.  Keep this drain as a defensive fallback for
        # direct runtime users and tests.
        self.prepare_request_release(rid)
        self.verifier.discard_layer_prefetch(rid)
        self.states.pop(rid, None)
        self.history_store.release(rid)
        if self.grant_runtime is not None:
            self.grant_runtime.release_request(rid)

    def clear(self) -> None:
        self._poll_pending_seals(wait=True)
        self._pending_seals.clear()
        self.verifier.discard_layer_prefetch()
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
        req.specstream_seal_inflight = state.seal_inflight
