from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
import threading
import time
from functools import wraps
from types import SimpleNamespace

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
from sglang.srt.speculative.spectre.specstream.cost_model import (
    SpecStreamBatchState,
    execution_shape_key,
)
from sglang.srt.speculative.spectre.specstream.gpu_history_budget import (
    resolve_gpu_history_budget,
)
from sglang.srt.speculative.spectre.specstream.cpu_history_store import (
    CPUHistoryStore,
    SealTicket,
)
from sglang.srt.speculative.spectre.specstream.diagnostics import (
    SpecStreamDiagnostics,
)
from sglang.srt.speculative.spectre.specstream.gpu_grant_controller import (
    GpuGrantController,
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
from sglang.srt.speculative.spectre.specstream.sm_controller import SMController
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
    init_batched_online_softmax_state,
    finalize_batched_online_softmax_state,
    split_packed_history_cohort_state,
    stack_packed_history_cohort_states,
    update_gpu_tail_state,
    update_gpu_paged_state_batched,
    update_packed_history_cohort_batched,
    update_packed_history_state,
)

logger = logging.getLogger(__name__)


def _grant_locked(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        # Lightweight runtime doubles used by CPU lifecycle tests predate the
        # control pump; production initializes the lock before publishing it.
        lock = getattr(self, "_grant_lock", None)
        if lock is None:
            self._grant_lock = lock = threading.RLock()
        with lock:
            return method(self, *args, **kwargs)

    return locked


def _require_pcie_slack_physical_ceiling(staging: StagingWindowPool) -> None:
    if staging.h2d_bandwidth_ceiling_gbps > 0.0:
        return
    raise RuntimeError(
        "SpecStream PCIe-slack startup refused: no reliable physical PCIe "
        "bandwidth ceiling; "
        f"source={staging.h2d_bandwidth_ceiling_source}. Install a working "
        "pynvml/nvidia-ml-py package and verify that the visible CUDA device "
        "maps to an NVML GPU."
    )


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


def _item_gpu_history_len(item) -> int:
    """Return the contiguous GPU-resident History prefix for old/new metadata."""

    history_len = max(int(item.history_len), 0)
    return max(0, min(int(getattr(item, "gpu_history_len", 0)), history_len))


def _item_cpu_history_len(item) -> int:
    return max(int(item.history_len) - _item_gpu_history_len(item), 0)


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
        self._layer_work_cache = {}
        self._paged_metadata_cache = {}

    def forward(self, *, q, k_new, v_new, forward_batch, meta, layer):
        if self._single_layer_prefetch:
            stale_keys = [
                key
                for key in self._single_layer_prefetch
                if key[0] != meta.round_id or key[2] < int(layer.layer_id)
            ]
            for key in stale_keys:
                for transfer in self._single_layer_prefetch.pop(key).values():
                    self._discard_transfer(transfer)
        if self._batched_layer_prefetch:
            stale_keys = [
                key
                for key in self._batched_layer_prefetch
                if key[0] != meta.round_id or key[1] < int(layer.layer_id)
            ]
            for key in stale_keys:
                for transfer in self._batched_layer_prefetch.pop(
                    key
                ).transfers.values():
                    self._discard_transfer(transfer)
        self._layer_work_cache = {
            key: value
            for key, value in getattr(self, "_layer_work_cache", {}).items()
            if key[0] == meta.round_id and key[1] >= int(layer.layer_id)
        }
        self._paged_metadata_cache = {
            key: value
            for key, value in getattr(self, "_paged_metadata_cache", {}).items()
            if key[0] == meta.round_id
        }
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
        packed_query_order = query.shape[0] == len(meta.items) * meta.q_len
        for index, item in enumerate(meta.items):
            item_query = query[item.query_begin : item.query_end]
            if item_query.shape[0] != meta.q_len:
                raise AssertionError("SpecStream query packing is not batch-uniform")
            queries[item.rid] = item_query
            packed_query_order &= (
                item.query_begin == index * meta.q_len
                and item.query_end == (index + 1) * meta.q_len
            )
        batched_paged = bool(
            not self.config.reference_attention
            and float(layer.logit_cap or 0.0) <= 0
            and query.is_cuda
            and query.dtype in (torch.float16, torch.bfloat16)
            and query.shape[-1] in (64, 128)
            and key_cache.shape == value_cache.shape
            and packed_query_order
            and meta.items
        )
        batch_state = None
        if batched_paged:
            batched_queries = query.view(
                len(meta.items), meta.q_len, query.shape[1], query.shape[2]
            )
            batch_state = init_batched_online_softmax_state(
                batched_queries, num_kv_heads, value_head_dim
            )
            states = dict(
                zip(
                    (item.rid for item in meta.items),
                    split_packed_history_cohort_state(batch_state),
                )
            )
        else:
            states = {
                item.rid: init_online_softmax_state(
                    queries[item.rid], num_kv_heads, value_head_dim
                )
                for item in meta.items
            }

        stream_items = [
            item for item in meta.items if item.stream_enabled and item.history_len > 0
        ]
        # Queue the first CPU History windows before GPU-resident History.
        # Existing L+1 transfers are reused. Metadata and task descriptors are
        # bounded to this round/current+next layer; no payload is recopied.
        if (
            self.config.layer_prefetch
            and not self.config.reference_attention
            and not meta.full_restore_baseline
        ):
            self._prime_history_layer(stream_items, meta, layer)
        if batched_paged:
            started = time.perf_counter()
            metadata, max_count = self._paged_metadata(meta, query.device, history=True)
            if max_count:
                batch_state, _ = update_gpu_paged_state_batched(
                    batch_state,
                    batched_queries,
                    key_cache,
                    value_cache,
                    forward_batch.req_to_token_pool.req_to_token,
                    metadata,
                    max_key_count=max_count,
                    scale=layer.scaling,
                    causal=False,
                )
                self.profiler.record_attention(
                    meta.round_id, (time.perf_counter() - started) * 1000
                )
        for item in () if batched_paged else stream_items:
            if _item_gpu_history_len(item) <= 0:
                continue
            started = time.perf_counter()
            states[item.rid] = self._update_gpu_history_state(
                item,
                queries[item.rid],
                states[item.rid],
                forward_batch=forward_batch,
                key_cache=key_cache,
                value_cache=value_cache,
                layer=layer,
            )
            self.profiler.record_attention(
                meta.round_id, (time.perf_counter() - started) * 1000
            )
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

        if batched_paged:
            started = time.perf_counter()
            # Single/independent CUDA updates mutate the initial batch views
            # directly. Only cohort processing repacks them into new storage.
            if (
                meta.cohort_enabled
                and len(stream_items) > 1
                and not meta.full_restore_baseline
            ):
                batch_state = stack_packed_history_cohort_states(
                    [states[item.rid] for item in meta.items]
                )
            metadata, max_count = self._paged_metadata(
                meta, query.device, history=False
            )
            batch_state, _ = update_gpu_paged_state_batched(
                batch_state,
                batched_queries,
                key_cache,
                value_cache,
                forward_batch.req_to_token_pool.req_to_token,
                metadata,
                max_key_count=max_count,
                scale=layer.scaling,
                causal=True,
            )
            output, _ = finalize_batched_online_softmax_state(
                batch_state, output_dtype=query.dtype
            )
            self.profiler.record_attention(
                meta.round_id, (time.perf_counter() - started) * 1000, tail=True
            )
            return output.view(-1, int(layer.tp_q_head_num) * value_head_dim)

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

    def _paged_metadata(self, meta, device, *, history):
        geometry = tuple(
            (
                item.req_pool_idx,
                item.history_len,
                _item_gpu_history_len(item),
                item.logical_len,
                item.committed_len,
                bool(item.stream_enabled),
            )
            for item in meta.items
        )
        key = (meta.round_id, history, str(device), geometry)
        cached = self._paged_metadata_cache.get(key)
        if cached is None:
            rows = [
                (
                    item.req_pool_idx,
                    0 if history else item.history_len,
                    (
                        _item_gpu_history_len(item)
                        if history and item.stream_enabled
                        else (0 if history else item.logical_len - item.history_len)
                    ),
                    item.committed_len,
                )
                for item in meta.items
            ]
            if not history and any(row[2] <= 0 for row in rows):
                raise AssertionError("SpecStream Tail/Frontier cannot be empty")
            if torch.device(device).type == "cuda":
                # Immutable per-round metadata; retain its pinned source until
                # the descriptor cache is retired, independent of staging slots.
                host = torch.tensor(rows, dtype=torch.int32, pin_memory=True)
                tensor = host.to(device=device, non_blocking=True)
            else:
                host = None
                tensor = torch.tensor(rows, dtype=torch.int32, device=device)
            cached = (tensor, max((row[2] for row in rows), default=0), host)
            self._paged_metadata_cache[key] = cached
        return cached[:2]

    def _discard_transfer(self, transfer):
        # Production staging uses this to retire physical window ownership;
        # simple CPU test doubles have no asynchronous events to retire.
        discard = getattr(self.staging, "discard", None)
        if discard is not None:
            discard(transfer)

    def _prime_history_layer(self, items, meta, layer):
        if not items:
            return
        layer_id = int(layer.layer_id)
        if meta.cohort_enabled and len(items) > 1:
            groups, _, _, tasks, keys = self._build_cohort_layer_work(
                items, meta, layer_id
            )
            self._prefetch_next_cohort_layer(
                items, meta, layer, prepared=(layer_id, groups, tasks, keys)
            )
        elif len(items) > 1:
            tasks, keys = self._build_independent_tasks(items, layer_id)
            self._prefetch_next_independent_layer(
                items, meta, layer, prepared=(layer_id, tasks, keys)
            )
        else:
            item = items[0]
            chunks = list(
                self.history_store.iter_layer_chunks(
                    item.rid,
                    layer_id,
                    history_start=_item_gpu_history_len(item),
                    history_end=item.history_len,
                )
            )
            self._prefetch_next_single_layer(
                item,
                meta,
                layer,
                prepared=(
                    layer_id,
                    _group_adjacent_chunks(chunks, self.config.chunks_per_transfer),
                ),
            )

    def _update_gpu_history_state(
        self,
        item,
        query,
        state,
        *,
        forward_batch,
        key_cache,
        value_cache,
        layer,
    ):
        """Merge the committed, CPU-backed History prefix retained in HBM."""

        slots = forward_batch.req_to_token_pool.req_to_token[
            item.req_pool_idx, : item.gpu_history_len
        ].long()
        if slots.numel() != item.gpu_history_len:
            raise AssertionError("GPU History cache page-table length mismatch")
        if self.config.reference_attention or float(layer.logit_cap or 0.0) > 0:
            cached_key = key_cache.index_select(0, slots)
            cached_value = value_cache.index_select(0, slots)
            return update_online_softmax_state(
                state,
                query,
                cached_key,
                cached_value,
                tuple(range(query.shape[0])),
                tuple(range(item.gpu_history_len)),
                scale=layer.scaling,
                causal=False,
                softcap=float(layer.logit_cap or 0.0),
            )
        state, _ = update_gpu_tail_state(
            state,
            query,
            key_cache,
            value_cache,
            query_position_start=item.committed_len,
            key_position_start=0,
            token_indices=slots,
            scale=layer.scaling,
            require_fused_cuda=True,
        )
        return state

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
                item.rid,
                layer.layer_id,
                history_start=_item_gpu_history_len(item),
                history_end=item.history_len,
            )
        )
        if sum(chunk.length for chunk in chunks) != _item_cpu_history_len(item):
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
                            history_start=_item_gpu_history_len(item),
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
        used_slots = {transfer.slot for transfer in transfers.values()}
        for index in range(min(self.staging.num_buffers, len(chunk_groups))):
            if index not in transfers:
                slot = next(
                    slot
                    for slot in range(self.staging.num_buffers)
                    if slot not in used_slots
                )
                transfers[index] = self.staging.submit_many(
                    [chunk.tensor for chunk in chunk_groups[index]],
                    slot,
                    round_id=meta.round_id,
                )
                used_slots.add(slot)

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
            record_staging = getattr(self.profiler, "record_staging", None)
            if record_staging is not None:
                record_staging(meta.round_id, transfer)
            self.staging.mark_consumed(transfer)
            next_index = index + self.staging.num_buffers
            if next_index < len(chunk_groups):
                transfers[next_index] = self.staging.submit_many(
                    [chunk.tensor for chunk in chunk_groups[next_index]],
                    transfer.slot,
                    round_id=meta.round_id,
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
                    item.rid,
                    next_layer_id,
                    history_start=_item_gpu_history_len(item),
                    history_end=item.history_len,
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
                [chunk.tensor for chunk in chunk_groups[index]],
                slot,
                round_id=meta.round_id,
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
            for transfer in cached.transfers.values():
                self._discard_transfer(transfer)
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
        if cached is not None and cached.task_keys != task_keys:
            for transfer in cached.transfers.values():
                self._discard_transfer(transfer)
            self._batched_layer_prefetch.pop(key)
            cached = None
        if cached is not None and cached.task_keys == task_keys:
            merged = dict(cached.transfers)
            for task_index, transfer in transfers.items():
                if task_index in merged and merged[task_index] is not transfer:
                    self._discard_transfer(transfer)
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
                    item.rid,
                    layer_id,
                    history_start=_item_gpu_history_len(item),
                    history_end=item.history_len,
                )
            )
            if sum(chunk.length for chunk in chunks) != _item_cpu_history_len(item):
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
                [chunk.tensor for chunk in chunk_group],
                slot,
                round_id=meta.round_id,
            )

        used_slots = {transfer.slot for transfer in transfers.values()}
        for task_index in range(min(self.staging.num_buffers, len(tasks))):
            if task_index not in transfers:
                slot = next(
                    slot
                    for slot in range(self.staging.num_buffers)
                    if slot not in used_slots
                )
                transfers[task_index] = submit_task(task_index, slot)
                used_slots.add(slot)

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
            record_staging = getattr(self.profiler, "record_staging", None)
            if record_staging is not None:
                record_staging(meta.round_id, transfer)
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
        if cached is not None and cached.task_keys != tuple(task_keys):
            for transfer in cached.transfers.values():
                self._discard_transfer(transfer)
            self._batched_layer_prefetch.pop(key)
            cached = None
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
                [chunk.tensor for chunk in chunk_group],
                slot,
                round_id=meta.round_id,
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
        stale_keys = [
            key for key in self._single_layer_prefetch if rid is None or key[1] == rid
        ]
        for key in stale_keys:
            for transfer in self._single_layer_prefetch.pop(key).values():
                self._discard_transfer(transfer)
        stale_keys = [
            key for key in self._batched_layer_prefetch if rid is None or rid in key[3]
        ]
        for key in stale_keys:
            for transfer in self._batched_layer_prefetch.pop(key).transfers.values():
                self._discard_transfer(transfer)
        getattr(self, "_layer_work_cache", {}).clear()
        getattr(self, "_paged_metadata_cache", {}).clear()

    def _restore_history(self, item, query, state, meta, layer):
        chunks = list(
            self.history_store.iter_layer_chunks(
                item.rid,
                layer.layer_id,
                history_start=_item_gpu_history_len(item),
                history_end=item.history_len,
            )
        )
        if not chunks:
            return state
        first = chunks[0].tensor
        host = _empty_host(
            (_item_cpu_history_len(item), *first.shape[1:]),
            dtype=first.dtype,
            pin_memory=torch.cuda.is_available(),
        )
        cursor = 0
        for chunk in chunks:
            host[cursor : cursor + chunk.length].copy_(chunk.tensor)
            cursor += chunk.length
        # Keep the K1 full-layer baseline observable with the same copy-stream
        # event machinery as K2/K3. It still waits for the complete layer
        # History before attention; only the unmeasurable Tensor.to path is
        # replaced.
        transfer = self.staging.submit(host, 0, round_id=meta.round_id)
        device_history = self.staging.wait_ready(transfer)
        started = time.perf_counter()
        state = self._update_history_state(
            state,
            query,
            device_history,
            meta=meta,
            item=item,
            layer=layer,
        )
        record_staging = getattr(self.profiler, "record_staging", None)
        if record_staging is not None:
            record_staging(meta.round_id, transfer)
        self.staging.mark_consumed(transfer)
        elapsed = (time.perf_counter() - started) * 1000
        self.profiler.record_h2d(
            meta.round_id,
            transfer.nbytes,
            (time.perf_counter_ns() - transfer.submitted_ns) / 1e6,
        )
        self.profiler.record_attention(meta.round_id, elapsed)
        return state

    def _build_cohort_layer_work(self, items, meta, layer_id: int):
        cache = getattr(self, "_layer_work_cache", None)
        if cache is None:
            self._layer_work_cache = cache = {}
        key = (
            meta.round_id,
            int(layer_id),
            tuple(
                (item.rid, item.history_len, _item_gpu_history_len(item))
                for item in items
            ),
            meta.q_len,
            self.config.chunks_per_transfer,
            self.config.max_cohort_size,
            self.config.reference_attention,
        )
        if key in cache:
            return cache[key]
        chunks_by_rid = {
            item.rid: list(
                self.history_store.iter_layer_chunks(
                    item.rid,
                    layer_id,
                    history_start=_item_gpu_history_len(item),
                    history_end=item.history_len,
                )
            )
            for item in items
        }
        for item in items:
            if sum(
                chunk.length for chunk in chunks_by_rid[item.rid]
            ) != _item_cpu_history_len(item):
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
        result = (groups_by_rid, request_by_rid, plans, tasks, tuple(task_keys))
        cache[key] = result
        return result

    def _submit_cohort_task(
        self,
        task,
        groups_by_rid,
        slot: int,
        *,
        round_id: int,
        direct_async: bool = False,
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
            return self.staging.submit_cohort_groups_direct_async(
                source_groups, slot, round_id=round_id
            )
        return self.staging.submit_cohort_groups(source_groups, slot, round_id=round_id)

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

        used_slots = {transfer.slot for transfer in transfers.values()}
        for task_index in range(min(self.staging.num_buffers, len(tasks))):
            if task_index not in transfers:
                slot = next(
                    slot
                    for slot in range(self.staging.num_buffers)
                    if slot not in used_slots
                )
                transfers[task_index] = self._submit_cohort_task(
                    tasks[task_index],
                    groups_by_rid,
                    slot,
                    round_id=meta.round_id,
                )
                used_slots.add(slot)

        task_index = 0
        reference_history = (
            self.config.reference_attention
            or float(getattr(layer, "logit_cap", 0.0) or 0.0) > 0
        )
        for plan in plans:
            cohort_queries = torch.stack([queries[work.rid] for work in plan.items])
            max_plan_groups = max(len(groups_by_rid[work.rid]) for work in plan.items)
            batched_state = None
            if not reference_history:
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
                if reference_history:
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
                record_staging = getattr(self.profiler, "record_staging", None)
                if record_staging is not None:
                    record_staging(meta.round_id, transfer)
                self.staging.mark_consumed(transfer)
                next_index = task_index + self.staging.num_buffers
                if next_index < len(tasks):
                    transfers[next_index] = self._submit_cohort_task(
                        tasks[next_index],
                        groups_by_rid,
                        transfer.slot,
                        round_id=meta.round_id,
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
        if cached is not None and cached.task_keys != tuple(task_keys):
            for transfer in cached.transfers.values():
                self._discard_transfer(transfer)
            self._batched_layer_prefetch.pop(key)
            cached = None
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
                round_id=meta.round_id,
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
        # Experimental policy: no CPU mirror or GPU release until an actual
        # admission/allocation shortage. Existing cache modes remain unchanged.
        self.gpu_first_pressure = os.environ.get("SPECSTREAM_GPU_FIRST_PRESSURE") == "1"
        self.gpu_first_restore = self.gpu_first_pressure and (
            os.environ.get("SPECSTREAM_GPU_FIRST_RESTORE", "0") == "1"
        )
        self._gpu_first_requests = {}
        if self.gpu_first_pressure and not config.enabled:
            raise ValueError("GPU-first pressure policy requires SpecStream")
        self._gpu_history_req_pool_idx: dict[str, int] = {}
        self._round_id = 0
        self._tp_sync_counter = 0
        self._h2d_grant_window_ids: dict[tuple[str, int], int] = {}
        self._grant_lock = threading.RLock()
        self._round_batch_states = {}
        self.tp_window_mailbox = None
        if config.pcie_slack_coexec and int(os.environ.get("SPECSTREAM_DRAFT_TP_SIZE", "1")) > 1:
            from .tp_window_mailbox import TPWindowMailbox

            directory = os.environ.get("SPECSTREAM_TP_WINDOW_DIR")
            if not directory:
                raise ValueError("Draft TP overlap requires SPECSTREAM_TP_WINDOW_DIR")
            self.tp_window_mailbox = TPWindowMailbox(directory, self.tp_rank, self.tp_size)

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
        if config.pcie_slack_coexec:
            logger.info(
                "SpecStream PCIe-slack colocated Draft mode: SLACK_FILL is "
                "armed only while the Target compute stream is blocked inside "
                "a current-round CUDA H2D wait gate; unobserved or unbounded "
                "windows stay Target-exclusive; grant_poll_us=%d",
                config.pcie_grant_poll_us,
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
            allocation_group_chunks=config.chunks_per_transfer,
        )
        allocator_size = int(getattr(self.token_to_kv_pool_allocator, "size", 0))
        cache_room = max(0, allocator_size - int(config.gpu_history_min_free_tokens))
        self._gpu_history_cache_capacity = min(
            int(config.gpu_history_cache_tokens), cache_room
        )
        self._update_gpu_history_budget()
        if config.enabled and (
            self._gpu_history_cache_capacity > 0
            or config.gpu_history_cache_tokens == -1
            or self.gpu_first_pressure
        ):
            # common.evict_from_tree_cache calls this immediately before native
            # KV allocation, with the exact number of slots the batch needs.
            self.token_to_kv_pool_allocator._external_kv_cache_evictor = (
                self.evict_gpu_history_for_allocation
            )
        if self.gpu_first_pressure:
            self.token_to_kv_pool_allocator._specstream_pressure_admission = (
                self.evict_gpu_history_for_allocation
            )
            logger.info("[SpecStream][GPU-FIRST] enabled: seal only on allocation/admission shortage")
        if config.enabled:
            logger.info(
                "SpecStream committed GPU History cache: requested=%d "
                "effective=%d min_free=%d target_pool=%d",
                config.gpu_history_cache_tokens,
                self._gpu_history_cache_capacity,
                config.gpu_history_min_free_tokens,
                allocator_size,
            )
        self.staging = StagingWindowPool(
            config.num_buffers,
            model_runner.device,
            # Exact H2D/stall telemetry is required for ordinary K1-K5 too.
            track_h2d_windows=config.enabled,
            serialize_h2d=config.serialize_h2d,
        )
        if config.pcie_slack_coexec:
            _require_pcie_slack_physical_ceiling(self.staging)
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
                "h2d_execution=%s, "
                "async_d2h_seal=%s, reserved_staging_bytes=%d, "
                "reserved_host_pack_bytes=%d",
                config.chunk_tokens,
                config.chunks_per_transfer,
                config.num_buffers,
                config.layer_prefetch,
                "serialized" if config.serialize_h2d else "async",
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
        if config.pcie_slack_coexec:
            logger.info(
                "SpecStream exposed-H2D gate: physical_ceiling=%.3f GB/s " "source=%s",
                self.staging.h2d_bandwidth_ceiling_gbps,
                self.staging.h2d_bandwidth_ceiling_source,
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
            catchup_tpcs = 0
            if calibration:
                # DRAFT_CATCHUP runs while Target is synchronously waiting for
                # the next Draft sequence, outside the overlap critical path.
                # It can therefore use the full validated device range while
                # SLACK_FILL remains bounded by calibration_tpcs.
                catchup_tpcs = SMController(
                    config.smctrl_library,
                    mask_scope=config.smctrl_mask_scope,
                ).total_tpcs
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
                    catchup_tpcs=catchup_tpcs,
                    catchup_token_quantum=config.grant_token_quantum,
                    calibration_allow_overlap=(config.smctrl_calibration_allow_overlap),
                )
            )
            if calibration:
                logger.warning(
                    "SpecStream fixed-TPC mode: overlap TPCs=%d catchup TPCs=%d "
                    "overlap=%s; "
                    "overlap remains gated by online slack timing, a launch "
                    "deadline, and Target slowdown protection",
                    config.smctrl_calibration_tpcs,
                    catchup_tpcs,
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
        if self.controller is not None and self.grant_runtime is not None and self.tp_rank == 0:
            self.profiler.parallel_feedback = self.controller.record_parallel_result
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

    def batch_requires_streaming(self, batch) -> bool:
        self._poll_pending_seals()
        self._restore_gpu_history_when_free(batch)
        return any(
            self.states.get(req.rid) is not None
            and self.states[req.rid].stream_enabled
            and self.states[req.rid].history_len > self.states[req.rid].gpu_history_len
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
                    gpu_history_len=state.gpu_history_len,
                    logical_len=committed_len + q_len,
                    stream_enabled=state.stream_enabled,
                )
            )
            self._publish_req_state(req, state, mode, q_len)
        enabled = bool(
            self.config.enabled
            and any(
                item.stream_enabled and _item_cpu_history_len(item) > 0
                for item in items
            )
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
        batch_state = self.collect_batch_state(batch)
        self._round_batch_states = {meta.round_id: batch_state}
        self.profiler.begin_round(
            meta, chunk_tokens=self.config.chunk_tokens, batch_state=batch_state
        )
        batch.specstream_meta = meta
        spec_info.specstream_meta = meta
        return meta

    def begin_native_decode(self, batch):
        # Build the same diagnostic row for native q=1 before native code
        # advances seq_lens. Native attention itself does not consume this meta.
        return self.build_round_meta(
            batch, SimpleNamespace(draft_token_num=1), "ordinary"
        )

    @_grant_locked
    def record_target_forward(
        self, meta, elapsed_ms: float, enqueue_ms: float = 0.0
    ) -> None:
        self.slack_profiler.record_target_phase("target_forward", elapsed_ms)
        overlap_active = False
        if self.grant_runtime is not None:
            overlap_active = self.grant_runtime.overlap_status()
            self.grant_runtime.record_target_forward(elapsed_ms)
        if meta is not None:
            # Target completion implies that every History copy it consumed is
            # also complete.  Harvest event timings nonblockingly before the
            # profile row is finalized; this never introduces an extra GPU
            # synchronization beyond the existing Target-forward event wait.
            self._harvest_h2d_timing_samples()
            self.profiler.record_target_forward(
                meta.round_id, elapsed_ms, enqueue_ms=enqueue_ms
            )
            if self.tp_monitor is not None:
                self.tp_monitor.record_local(
                    TPRankSample(
                        rank=self.tp_rank,
                        round_id=meta.round_id,
                        target_forward_ms=float(elapsed_ms),
                        shape_key=execution_shape_key(
                            self._round_batch_states[meta.round_id], meta.q_len
                        ),
                        overlap_active=overlap_active,
                    )
                )

    def record_logit_margin(self, meta, logits) -> None:
        if meta is not None:
            self.diagnostics.record_logit_margin(round_id=meta.round_id, logits=logits)

    def after_verify(self, batch, result, meta) -> None:
        self._poll_pending_seals()
        accepted = list(result.accept_length_per_req_cpu)
        sealed_history_floors = [int(item.history_len) for item in meta.items]
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
        post_states = [self.states[req.rid] for req in batch.reqs]
        accepted_with_bonus = [int(value) + 1 for value in accepted]
        self.profiler.finish_round(
            meta.round_id,
            accepted_tokens=sum(accepted_with_bonus),
            staging_bytes=self.staging.allocated_bytes,
            cpu_bytes=self.history_store.bytes_used,
            accepted_per_req=accepted_with_bonus,
            post_states=post_states,
            sealed_history_floors=sealed_history_floors,
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

        meta = getattr(batch, "specstream_meta", None)
        if meta is not None:
            self.profiler.finish_round(
                meta.round_id,
                accepted_tokens=len(batch.reqs),
                staging_bytes=self.staging.allocated_bytes,
                cpu_bytes=self.history_store.bytes_used,
                accepted_per_req=[1] * len(batch.reqs),
                post_states=[self.states[req.rid] for req in batch.reqs],
                sealed_history_floors=[item.history_len for item in meta.items],
            )

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
                    state.gpu_history_len = 0
                    self._gpu_history_req_pool_idx.pop(req.rid, None)
                state.committed_len = committed_len
                state.logical_len = committed_len
            if getattr(req, "is_chunked", 0) <= 0:
                # Start offload as soon as the final prefill chunk has produced
                # its KV.  Waiting for the first verify round lets many long
                # prompts fill the token pool before any slot is recycled.
                self._maybe_seal(req, state)
            self._publish_req_state(req, state, "parallel", 1)
        self._poll_pending_seals()

    def _maybe_seal(self, req, state: TargetTieredKVState, *, pressure_tokens: int = 0) -> None:
        if getattr(self, "gpu_first_pressure", False):
            self._gpu_first_requests[req.rid] = req
            if pressure_tokens <= 0:
                return
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
        if getattr(self, "gpu_first_pressure", False):
            chunk = self.config.chunk_tokens
            requested_end = state.history_len + ((pressure_tokens + chunk - 1) // chunk) * chunk
            minimum_end = ((self.config.min_history_tokens + chunk - 1) // chunk) * chunk
            seal_end = min(seal_end, max(requested_end, minimum_end))
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

    def _restore_gpu_history_when_free(self, batch) -> int:
        """Restore complete committed prefixes using idle Target pool slots.

        Called by every TP rank before forward; never from rank-zero policy
        selection. CPU copies remain valid eviction victims. Reserve only the
        live batch's remaining decode budget, not a full future prefill. A new
        admission can reclaim these replicas without another D2H transfer.
        """
        if not getattr(self, "gpu_first_restore", False):
            return 0
        allocator = self.token_to_kv_pool_allocator
        reserve = 0
        candidates = []
        for req in batch.reqs:
            if req.finished():
                continue
            remaining = max(0, int(req.sampling_params.max_new_tokens) - len(req.output_ids))
            reserve += remaining + max(self.config.q_candidates or (1,))
            state = self.states.get(req.rid)
            if state is not None and not state.seal_inflight:
                missing = state.history_len - state.gpu_history_len
                if missing > 0:
                    candidates.append((missing, req.rid, req, state))
        restored = 0
        # Whole prefixes avoid making every small decode allocation re-copy
        # the same partial prefix. Smaller requests reach native attention first.
        for missing, rid, req, state in sorted(candidates, key=lambda x: (x[0], x[1])):
            if int(allocator.available_size()) < missing + reserve:
                continue
            slots = allocator.alloc(missing)
            if slots is None:
                continue
            begin, end = int(state.gpu_history_len), int(state.history_len)
            device = slots.device
            try:
                self.verifier.discard_layer_prefetch(rid)
                for layer_id in self.history_store.layer_ids:
                    key = self.token_to_kv_pool.get_key_buffer(layer_id)
                    value = self.token_to_kv_pool.get_value_buffer(layer_id)
                    copied = 0
                    for chunk in self.history_store.iter_layer_chunks(
                        rid, layer_id, history_start=begin, history_end=end
                    ):
                        if chunk.abs_start != begin + copied:
                            raise AssertionError("non-contiguous GPU History restore")
                        packed = chunk.tensor.to(device=device, non_blocking=chunk.pinned)
                        dst = slots[copied : copied + chunk.length].long()
                        key.index_copy_(0, dst, packed[:, 0])
                        value.index_copy_(0, dst, packed[:, 1])
                        copied += chunk.length
                    if copied != missing:
                        raise AssertionError("incomplete GPU History restore")
                # One completion boundary per restored request; scratch is only
                # one layer/chunk, not a second whole-history GPU allocation.
                if device.type == "cuda":
                    torch.cuda.current_stream(device).synchronize()
                self.req_to_token_pool.req_to_token[req.req_pool_idx, begin:end] = slots
                state.gpu_history_len = end
                self._gpu_history_req_pool_idx[rid] = int(req.req_pool_idx)
                self._publish_req_state(req, state, getattr(req, "specstream_mode", "ordinary"),
                                        getattr(req, "specstream_q", 1))
            except Exception:
                if device.type == "cuda":
                    torch.cuda.current_stream(device).synchronize()
                allocator.free(slots)
                raise
            restored += missing
            logger.info("[SpecStream][GPU-FIRST] restored rid=%s tokens=%d free_after=%d",
                        rid, missing, int(allocator.available_size()))
        return restored

    def _evict_gpu_history_suffix(self, rid: str, keep_tokens: int) -> int:
        """Drop only CPU-backed committed replicas from the Target KV pool."""

        state = self.states.get(rid)
        if state is None:
            return 0
        old_tokens = int(state.gpu_history_len)
        keep_tokens = max(0, min(int(keep_tokens), old_tokens))
        if keep_tokens == old_tokens:
            return 0
        req_pool_idx = self._gpu_history_req_pool_idx.get(rid)
        if req_pool_idx is None:
            raise AssertionError("GPU History cache entry has no request pool index")
        slots = self.req_to_token_pool.req_to_token[
            req_pool_idx, keep_tokens:old_tokens
        ].clone()
        live_slots = slots[slots != 0]
        if int(live_slots.numel()) != old_tokens - keep_tokens:
            # CPU History is authoritative.  A stale or partially released GPU
            # replica must not abort serving; if a retained prefix would contain
            # a hole, conservatively drop the entire replica instead.
            keep_tokens = 0
            slots = self.req_to_token_pool.req_to_token[
                req_pool_idx, :old_tokens
            ].clone()
            live_slots = slots[slots != 0]
            logger.warning(
                "[SpecStream][GPU-History] stale page-table holes rid=%s; "
                "dropping the cache replica",
                rid,
            )
        if live_slots.numel():
            self.token_to_kv_pool_allocator.free(live_slots)
        self.req_to_token_pool.req_to_token[req_pool_idx, keep_tokens:old_tokens] = 0
        state.gpu_history_len = keep_tokens
        logger.debug(
            "[SpecStream][GPU-History] evicted rid=%s range=[%d,%d)",
            rid,
            keep_tokens,
            old_tokens,
        )
        return old_tokens - keep_tokens

    def _update_gpu_history_budget(self, reclaimable_seal_tokens: int = 0) -> None:
        allocator = self.token_to_kv_pool_allocator
        runner = getattr(self, "model_runner", None)
        args = getattr(runner, "server_args", None)
        model = getattr(runner, "model_config", None)
        context_len = int(
            getattr(args, "context_length", 0) or getattr(model, "context_len", 0) or 0
        )
        prefill_limit = int(getattr(args, "max_prefill_tokens", 0) or 0)
        prefill_requests = int(getattr(args, "prefill_max_requests", 0) or 1)
        allocator_size = int(getattr(allocator, "size", 0))
        available_size = getattr(allocator, "available_size", None)
        requested_tokens = int(
            getattr(
                self.config,
                "gpu_history_cache_tokens",
                getattr(self, "_gpu_history_cache_capacity", 0),
            )
        )
        # Explicit limits do not need allocator free-space introspection.
        # Automatic admission fails closed if the allocator cannot report it.
        available_tokens = int(available_size()) if callable(available_size) else 0
        prefill_reserve = (
            max(context_len, prefill_limit) * max(prefill_requests, 1)
            if callable(available_size)
            else 0
        )
        states = getattr(self, "states", {})
        q_candidates = getattr(self.config, "q_candidates", ())
        budget = resolve_gpu_history_budget(
            requested_tokens=requested_tokens,
            allocator_size=allocator_size,
            available_tokens=available_tokens,
            retained_history_tokens=sum(s.gpu_history_len for s in states.values()),
            reclaimable_seal_tokens=reclaimable_seal_tokens,
            active_requests=len(states),
            active_tail_tokens=int(getattr(self.config, "active_tail_tokens", 512)),
            chunk_tokens=self.config.chunk_tokens,
            max_q=(
                max(q_candidates)
                if q_candidates
                else int(getattr(self.config, "default_q", 1))
            ),
            min_free_tokens=self.config.gpu_history_min_free_tokens,
            prefill_reserve_tokens=prefill_reserve,
            page_size=int(getattr(allocator, "page_size", 1)),
        )
        self._gpu_history_budget = budget
        self._gpu_history_cache_capacity = budget.capacity_tokens

    def _rebalance_gpu_history_cache(
        self, admitting_rid: str, reclaimable_seal_tokens: int = 0
    ) -> int:
        """Give active requests a stable fair share; misses never pollute it."""

        if getattr(self.config, "gpu_history_cache_tokens", 0) == -1:
            self._update_gpu_history_budget(reclaimable_seal_tokens)
        active_rids = [
            rid
            for rid, state in self.states.items()
            if state.history_len > 0 or rid == admitting_rid
        ]
        if admitting_rid not in active_rids:
            active_rids.append(admitting_rid)
        quota = self._gpu_history_cache_capacity // max(len(active_rids), 1)
        for rid in active_rids:
            state = self.states.get(rid)
            if state is not None and state.gpu_history_len > quota:
                self._evict_gpu_history_suffix(rid, quota)
        return quota

    def evict_gpu_history_for_allocation(self, required_tokens: int) -> int:
        """Release CPU-backed replicas only when native allocation needs them."""

        available_size = getattr(
            self.token_to_kv_pool_allocator, "available_size", None
        )
        if available_size is None:
            return 0

        required_tokens = max(0, int(required_tokens))
        allocation_guard = max(0, int(self.config.gpu_history_min_free_tokens))
        target_free = required_tokens + allocation_guard

        # Completed D2H seals may immediately make more committed replicas
        # evictable. Poll first; block only when allocation would otherwise
        # fail and an in-flight seal is the only possible source of slots.
        self._poll_pending_seals()
        if int(available_size()) < target_free and self._pending_seals:
            self._poll_pending_seals(wait=True)

        evicted = 0
        while int(available_size()) < target_free:
            candidates = [
                (state.gpu_history_len, rid)
                for rid, state in self.states.items()
                if state.gpu_history_len > 0
            ]
            if not candidates:
                break
            cached_tokens, rid = max(candidates)
            need = target_free - int(available_size())
            drop = min(cached_tokens, max(need, self.config.chunk_tokens))
            evicted += self._evict_gpu_history_suffix(rid, cached_tokens - drop)
        if getattr(self, "gpu_first_pressure", False) and int(available_size()) < target_free:
            # These requests completed prefill; uncommitted Frontier and a
            # partially prefilling request are never eviction candidates.
            before_free = int(available_size())
            candidates = []
            for rid, req in self._gpu_first_requests.items():
                state = self.states.get(rid)
                if state is None or state.seal_inflight or req.finished():
                    continue
                end = max(0, state.committed_len - self.config.active_tail_tokens)
                end = (end // self.config.chunk_tokens) * self.config.chunk_tokens
                if end >= self.config.min_history_tokens and end > state.history_len:
                    candidates.append((end - state.history_len, rid, req, state))
            for _, rid, req, state in sorted(candidates, key=lambda x: (-x[0], x[1])):
                shortage = target_free - int(available_size())
                if shortage <= 0:
                    break
                self._maybe_seal(req, state, pressure_tokens=shortage)
                # Publication/free must follow D2H completion, on every TP rank.
                self._poll_pending_seals(wait=True, only_rid=rid)
                self._publish_req_state(req, state, getattr(req, "specstream_mode", "ordinary"),
                                        getattr(req, "specstream_q", 1))
            evicted += int(available_size()) - before_free
            if evicted:
                logger.info(
                    "[SpecStream][GPU-FIRST] pressure required=%d free_before=%d "
                    "released=%d free_after=%d pool=%d cpu_bytes=%d",
                    target_free, before_free, evicted, int(available_size()),
                    self.token_to_kv_pool_allocator.size, self.history_store.bytes_used,
                )
        if evicted:
            logger.debug(
                "[SpecStream][GPU-History] allocation-driven eviction "
                "required=%d guard=%d evicted=%d available=%d",
                required_tokens,
                allocation_guard,
                evicted,
                int(available_size()),
            )
        return evicted

    def _publish_completed_seal(
        self, pending: _PendingSeal, state: TargetTieredKVState
    ) -> tuple[int, int]:
        """Publish CPU History, retain a bounded GPU prefix, then free the rest."""

        self._gpu_history_req_pool_idx[pending.rid] = pending.req_pool_idx
        quota = (
            int(state.gpu_history_len)
            if getattr(self, "gpu_first_pressure", False)
            else self._rebalance_gpu_history_cache(
                pending.rid, int(pending.seal_end - pending.history_start)
            )
        )
        retain_begin = int(pending.history_start)
        retain_end = int(state.gpu_history_len)
        # Cache residency is a contiguous prefix. Once a prefix has a CPU-only
        # gap, sequential misses bypass admission and cannot cause LRU thrash.
        if state.gpu_history_len == retain_begin:
            retain_end = min(int(pending.seal_end), int(quota))
        retain_count = max(0, retain_end - retain_begin)
        released = pending.slots[retain_count:]
        if released.numel():
            self.token_to_kv_pool_allocator.free(released)
        release_begin = retain_begin + retain_count
        self.req_to_token_pool.req_to_token[
            pending.req_pool_idx, release_begin : pending.seal_end
        ] = 0
        state.mark_sealed(pending.seal_end, pending.ticket.block_ids)
        state.gpu_history_len = retain_end
        return retain_count, int(released.numel())

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

            # CPU publication is the consistency boundary. Only after it is
            # complete may a bounded committed prefix remain as an evictable
            # GPU replica; rollback can never reach any retained slot.
            retained, released = self._publish_completed_seal(pending, state)
            del self._pending_seals[rid]
            completed += 1
            logger.debug(
                "[SpecStream][D2H] completed rid=%s history_len=%d "
                "gpu_history_len=%d retained=%d released=%d pending=%d",
                rid,
                state.history_len,
                state.gpu_history_len,
                retained,
                released,
                len(self._pending_seals),
            )
        return completed

    def prepare_request_release(self, rid: str) -> None:
        """Drain a terminal request before SGLang's generic KV release path."""

        getattr(self, "_gpu_first_requests", {}).pop(rid, None)
        if rid in self._pending_seals:
            self._poll_pending_seals(wait=True, only_rid=rid)
        state = self.states.get(rid)
        if state is not None and state.gpu_history_len > 0:
            self._evict_gpu_history_suffix(rid, 0)

    def collect_batch_state(self, batch) -> SpecStreamBatchState:
        self._poll_pending_seals()
        histories = []
        for req in batch.reqs:
            state = self.states.get(req.rid, TargetTieredKVState(req.rid))
            histories.append(max(state.history_len - state.gpu_history_len, 0))
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
            gpu_history_tokens=sum(
                self.states.get(req.rid, TargetTieredKVState(req.rid)).gpu_history_len
                for req in batch.reqs
            ),
            phase="decode",
            attention_impl=(
                "reference"
                if getattr(self.config, "reference_attention", False)
                else (
                    "split_kv_cohort"
                    if getattr(self.config, "cohort_enabled", False)
                    else "split_kv"
                )
            ),
            tp_size=getattr(self, "tp_size", 1),
        )

    def choose_decision(self, batch) -> SpecStreamDecision | None:
        if self.controller is None:
            return None
        q_candidates = self.controller.q_candidates
        batch_state = self.collect_batch_state(batch)
        return self.controller.choose(
            batch_state,
            self.profiler.snapshot(),
            self.acceptance_tracker.snapshot(q_candidates),
            allow_ordinary=True,
            force_ordinary=self.config.force_ordinary_mode,
            # Already collected on CPU; no device synchronization or extra TP
            # collective. Only TP0 chooses and the existing broadcast admits it.
            parallel_ready=(self.grant_runtime is None or batch_state.no_draft_ratio == 0.0),
            allow_pipeline_seed=(os.environ.get("SPECSTREAM_PIPELINE_SEED", "0") == "1"),
            draft_load=self.controller.draft_load_tracker.snapshot(),
            tp_snapshot=(
                self.tp_monitor.snapshot() if self.tp_monitor is not None else None
            ),
            tp_snapshots_by_q=(
                {
                    q: self.tp_monitor.snapshot(execution_shape_key(batch_state, q))
                    for q in q_candidates
                }
                if self.tp_monitor is not None
                else None
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
        """Propose a sampling gate on TP0; the scheduler broadcasts it to peers.

        This method reads local samples and must never independently gate a
        collective on each rank. Asynchronous History retirement can leave
        ranks with different local execution shapes or baseline readiness.
        """
        if self.tp_monitor is None or self.tp_size < 2:
            return False
        self._tp_sync_counter += 1
        # Sample new shapes throughout warmup, then retain the configured
        # periodic cadence. Only TP0 evaluates this local proposal.
        sample = self.tp_monitor.local_sample()
        snapshot = (
            self.tp_monitor.snapshot(sample.shape_key)
            if sample is not None
            else self.tp_monitor.snapshot()
        )
        if not snapshot.baseline_ready or snapshot.overlap_active is not False:
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

    def _observe_current_h2d_window(self, round_id: int):
        observation = self.staging.observe_h2d_window(int(round_id))
        self._harvest_h2d_timing_samples()
        self.profiler.record_h2d_window_observation(observation)
        return observation

    def _harvest_h2d_timing_samples(self) -> None:
        for sample in self.staging.drain_h2d_timing_samples():
            self.profiler.record_h2d_event(
                sample.round_id,
                sample.nbytes,
                sample.elapsed_ms,
                target_wait_ms=sample.target_wait_ms,
            )

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

    @_grant_locked
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
        if getattr(
            getattr(self.grant_runtime, "controller", None), "fixed_tpc_mode", False
        ):
            miss = sum(
                max(
                    self.states.get(req.rid, TargetTieredKVState(req.rid)).history_len
                    - self.states.get(
                        req.rid, TargetTieredKVState(req.rid)
                    ).gpu_history_len,
                    0,
                )
                for req in reqs
            )
            hit = sum(
                self.states.get(req.rid, TargetTieredKVState(req.rid)).gpu_history_len
                for req in reqs
            )
            target_shape += f"_miss{(miss + 1023) // 1024}_hit{(hit + 1023) // 1024}_tp{self.tp_size}"
        slack_source = "target_forward"
        target_phase = "target_forward"
        predicted_slack_us = self.slack_profiler.snapshot(
            target_phase
        ).predicted_slack_us
        if self.config.pcie_slack_coexec:
            slack_source = "history_h2d"
            target_phase = "history_h2d"
            # This call occurs before the Target has enqueued the current
            # verify forward, so no current-round H2D event can exist yet.
            # Register the round but never bootstrap it from a previous-round
            # queue-residence EMA.  overlap_grants() arms it after observing a
            # real begin/end event from this round.
            predicted_slack_us = 0.0
        draft_step_ms = self.slack_profiler.snapshot(
            draft_bs=batch_size,
            draft_ctx_bucket=ctx_bucket,
        ).draft_step_ms
        keys = []
        for req in reqs:
            key = (str(req.rid), int(req.spec_cnt))
            for stale_key in [
                stale_key
                for stale_key in self._h2d_grant_window_ids
                if stale_key[0] == key[0] and stale_key[1] != key[1]
            ]:
                self._h2d_grant_window_ids.pop(stale_key, None)
            keys.append(key)
            self.grant_runtime.register_round(
                request_id=key[0],
                spec_cnt=key[1],
                desired_q=int(desired_q),
                target_shape=target_shape,
                draft_bs=batch_size,
                draft_ctx_bucket=ctx_bucket,
                predicted_slack_us=predicted_slack_us,
                draft_step_ms=draft_step_ms,
                slack_source=slack_source,
            )
        # In PCIe mode predicted_slack_us is exactly zero here, so the
        # controller records a Target-only baseline shape but cannot issue a
        # grant.  The first possible SLACK_FILL remains overlap_grants() after
        # a current-round H2D event has been observed.
        messages = self.grant_runtime.initial_grants(keys)
        for message in messages:
            self.profiler.record_grant(
                message,
                target_phase=target_phase,
                predicted_slack_us=predicted_slack_us,
            )
        if not messages:
            for key in keys:
                state = self.grant_runtime.state_for(*key)
                if state is not None and state.last_decision is not None:
                    self.profiler.record_grant_decision(
                        state.last_decision,
                        target_phase=target_phase,
                        predicted_slack_us=predicted_slack_us,
                    )
                    break
        return messages

    @_grant_locked
    def overlap_grants(self, keys):
        if self.grant_runtime is None:
            return []
        keys = list(keys)
        if self.config.pcie_slack_coexec:
            observation = self._observe_current_h2d_window(self._round_id)
            if self.tp_window_mailbox is not None:
                self.tp_window_mailbox.publish(observation)
                observation = self.tp_window_mailbox.intersect(observation)
            if (
                not observation.active
                or observation.remaining_us <= 0.0
                or observation.window_end_us <= 0
            ):
                return []
            armed = False
            for key in keys:
                state = self.grant_runtime.state_for(*key)
                if state is None or state.slack_source != "history_h2d":
                    continue
                previous_window = self._h2d_grant_window_ids.get(key)
                if previous_window == observation.window_id:
                    armed = True
                    continue
                # Never move the deadline of a grant already sent to Drafter.
                # After its ACK, a later physical copy window may safely arm
                # the remaining token horizon with a fresh absolute deadline.
                if state.outstanding_epoch is not None:
                    continue
                state.predicted_slack_us = float(observation.remaining_us)
                # Preserve the observation's absolute lower-bound end.  Using
                # ``now + remaining`` here or in TargetGrantRuntime would add
                # verifier iteration, profiling and ZMQ preparation time to the
                # physical H2D window.
                state.overlap_window_end_us = int(observation.window_end_us)
                self._h2d_grant_window_ids[key] = int(observation.window_id)
                armed = True
            if not armed:
                return []
        messages = self.grant_runtime.overlap_grants(keys)
        for message in messages:
            state = self.grant_runtime.state_for(
                str(message.request_id), int(message.spec_cnt or 0)
            )
            slack_source = state.slack_source if state is not None else "target_forward"
            self.profiler.record_grant(
                message,
                target_phase=slack_source,
                predicted_slack_us=(
                    state.predicted_slack_us if state is not None else 0.0
                ),
            )
        return messages

    @_grant_locked
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

    @_grant_locked
    def acknowledge_grant(self, message) -> bool:
        if self.grant_runtime is None:
            return False
        accepted = self.grant_runtime.acknowledge(message)
        state = self.grant_runtime.state_for(
            str(message.request_id), int(message.spec_cnt or 0)
        )
        # A zero-token SUPERSEDED ACK can arrive after register_round() has
        # retired the old spec_cnt.  It no longer changes live scheduling
        # state, but it is still the authoritative terminal disposition for
        # offline issued/ACK accounting and must not be discarded.
        self.profiler.record_grant_ack(
            message,
            wait_ms=(
                state.last_grant_wait_ms if accepted and state is not None else 0.0
            ),
        )
        if accepted:
            if (
                float(getattr(message, "draft_step_ms", 0.0) or 0.0) > 0
                and str(getattr(message, "grant_state", "")) != "PREFILL_COMPLETE"
                and int(getattr(message, "grant_tokens", 0) or 0) > 0
            ):
                self.slack_profiler.record_draft_step(
                    message.draft_step_ms,
                    draft_bs=(state.draft_bs if state is not None else None),
                    draft_ctx_bucket=(
                        state.draft_ctx_bucket if state is not None else None
                    ),
                )
        return accepted

    @_grant_locked
    def pause_grants(self, keys):
        if self.grant_runtime is None:
            return []
        return self.grant_runtime.pause_messages(list(keys))

    def record_draft_result(
        self,
        *,
        q: int | None = None,
        elapsed_ms: float,
        rtt_ms: float | None = None,
        timeout_ms: float,
        missing_count: int,
        total_count: int,
    ) -> None:
        if q is not None and total_count > 0:
            self.profiler.record_draft_rtt(q, elapsed_ms if rtt_ms is None else rtt_ms)
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

    @_grant_locked
    def release_request(self, rid: str) -> None:
        # The scheduler normally calls prepare_request_release() before its
        # generic KV release.  Keep this drain as a defensive fallback for
        # direct runtime users and tests.
        self.prepare_request_release(rid)
        self.verifier.discard_layer_prefetch(rid)
        self.states.pop(rid, None)
        self._gpu_history_req_pool_idx.pop(rid, None)
        self.history_store.release(rid)
        for key in [key for key in self._h2d_grant_window_ids if key[0] == rid]:
            self._h2d_grant_window_ids.pop(key, None)
        if self.grant_runtime is not None:
            self.grant_runtime.release_request(rid)

    @_grant_locked
    def clear(self) -> None:
        self._poll_pending_seals(wait=True)
        self._pending_seals.clear()
        getattr(self, "_gpu_first_requests", {}).clear()
        self.verifier.discard_layer_prefetch()
        self.states.clear()
        getattr(self, "_gpu_history_req_pool_idx", {}).clear()
        self._h2d_grant_window_ids.clear()
        if self.grant_runtime is not None:
            self.grant_runtime.clear()
        self.history_store.clear()

    @staticmethod
    def _publish_req_state(req, state, mode: str, q: int) -> None:
        req.specstream_history_len = state.history_len
        req.specstream_gpu_history_len = state.gpu_history_len
        req.specstream_committed_len = state.committed_len
        req.specstream_logical_len = state.logical_len
        req.specstream_stream_enabled = state.stream_enabled
        req.specstream_round_id = state.round_id
        req.specstream_mode = mode
        req.specstream_q = q
        req.specstream_seal_inflight = state.seal_inflight
