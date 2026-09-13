from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import threading
import time
import zlib

from sglang.srt.speculative.spectre.specstream.cost_model import (
    EmpiricalCandidateCost,
    SpecStreamBatchState,
    SpecStreamCostProfile,
    workload_shape_key,
)
from sglang.srt.speculative.spectre.specstream.draft_load_tracker import (
    DraftLoadSnapshot,
)
from sglang.srt.speculative.spectre.specstream.mps_env import MPSEnvironment
from sglang.srt.speculative.spectre.specstream.tp_straggler_monitor import (
    TPStragglerSnapshot,
)


@dataclass
class SpecStreamProfileRow:
    timestamp: float
    tp_rank: int
    rid: str
    round_id: int
    mode: str
    q: int
    batch_size: int = 1
    context_tokens: int = 0
    committed_len: int = 0
    history_len: int = 0
    gpu_history_hit_tokens: int = 0
    cpu_history_miss_tokens: int = 0
    committed_tokens_total: int = 0
    history_tokens_total: int = 0
    post_committed_tokens_total: int = 0
    post_history_tokens_total: int = 0
    post_logical_tokens_total: int = 0
    history_advanced_tokens: int = 0
    sealed_request_count: int = 0
    seal_inflight_count: int = 0
    rollback_floor_min: int = 0
    rollback_crossed_history: bool = False
    tail_tokens: int = 0
    chunk_tokens: int = 0
    num_chunks: int = 0
    cohort_size: int = 1
    h2d_bytes: int = 0
    h2d_ops: int = 0
    h2d_source_bytes: int = 0
    h2d_padding_bytes: int = 0
    h2d_dma_ops: int = 0
    h2d_source_slabs: int = 0
    host_slot_wait_ms: float = 0.0
    host_pack_ms: float = 0.0
    metadata_cache_hits: int = 0
    h2d_ms: float = 0.0
    h2d_event_ops: int = 0
    h2d_event_bytes: int = 0
    h2d_event_ms: float = 0.0
    h2d_wait_event_ops: int = 0
    h2d_gbps: float = 0.0
    h2d_timing_source: str = ""
    h2d_window_id: int = 0
    h2d_window_active: bool = False
    h2d_window_remaining_us: float = 0.0
    h2d_window_reason: str = ""
    copy_floor_ms: float = 0.0
    stream_attn_ms: float = 0.0
    stream_attn_ops: int = 0
    tail_attn_ms: float = 0.0
    attention_timing_source: str = "cpu_enqueue"
    stream_attn_event_ms: float = 0.0
    stream_attn_event_ops: int = 0
    tail_attn_ops: int = 0
    tail_attn_event_ms: float = 0.0
    tail_attn_event_ops: int = 0
    target_forward_ms: float = 0.0
    target_enqueue_ms: float = 0.0
    grant_pump_iterations: int = 0
    grant_pump_wall_ms: float = 0.0
    draft_ms: float = 0.0
    draft_rtt_ema_ms: float = 0.0
    draft_rtt_p95_ms: float = 0.0
    draft_rtt_by_q: str = ""
    draft_rtt_samples_by_q: str = ""
    pending_drafts: int = 0
    draft_timeout_rate: float = 0.0
    draft_missing_ratio: float = 0.0
    draft_reject_rate: float = 0.0
    network_wait_ms: float = 0.0
    round_ms: float = 0.0
    round_timing_source: str = ""
    exposed_copy_ms: float = 0.0
    copy_compute_overlap: float = 0.0
    draft_verify_overlap: float = 0.0
    draft_verify_overlap_ms: float = 0.0
    draft_overlap_timing_source: str = "unmeasured"
    staging_wait_ms: float = 0.0
    gpu_kv_bytes: int = 0
    staging_bytes: int = 0
    cpu_history_bytes: int = 0
    accepted_tokens: int = 0
    rejected_requests: int = 0
    rollback_tokens: int = 0
    reject_position: int = -1
    rollback_ratio: float = 0.0
    controller_cost_parallel: float = 0.0
    controller_cost_ordinary: float = 0.0
    controller_candidate_costs: str = ""
    controller_selected_q: int = 0
    controller_selected_mode: str = ""
    controller_planned_mode: str = ""
    slack_fill_issued_tokens: int = 0
    coexec_mode: str = ""
    coexec_reason: str = ""
    target_phase: str = ""
    predicted_slack_us: float = 0.0
    grant_state: str = ""
    grant_epoch: int = 0
    grant_wait_ms: float = 0.0
    draft_step_ms: float = 0.0
    draft_tpc_low: int = -1
    draft_tpc_high: int = -1
    mps_active_thread_percentage: int = 0
    mps_client_priority: int = -1
    mps_sm_partition: str = ""
    tp_colocated_rank: int = -1
    tp_rank_forward_ms: float = 0.0
    tp_collective_wait_ms: float = 0.0
    tp_rank_skew_ms: float = 0.0
    tp_target_slowdown: float = 0.0
    tp_baseline_ready: bool = False
    tp_baseline_samples: int = 0
    tp_overlap_active: str = "unknown"
    tp_shape_key: str = ""
    fallback: bool = False
    fallback_reason: str = ""
    missing_draft_count: int = 0
    shadow_max_abs: float = 0.0
    logit_margin: float = 0.0
    token_mismatch: bool = False


class SpecStreamProfiler:
    def __init__(
        self,
        path: str,
        tp_rank: int,
        tp_size: int,
        mps_environment: MPSEnvironment | None = None,
        calibrated_h2d_gbps: float = 0.0,
    ) -> None:
        base = Path(path)
        if tp_size > 1:
            base = base.with_name(f"{base.stem}.tp{tp_rank}{base.suffix or '.csv'}")
        self.path = base
        self.grant_path = base.with_name(f"{base.stem}.grants.csv")
        self.tp_rank = tp_rank
        self.mps_environment = mps_environment or MPSEnvironment()
        self._lock = threading.Lock()
        self._active: dict[int, SpecStreamProfileRow] = {}
        self._h2d_gbps_ema = (
            max(float(calibrated_h2d_gbps), 0.1) if calibrated_h2d_gbps > 0 else 12.0
        )
        self._h2d_timing_source = (
            "cuda_event_startup_calibration"
            if calibrated_h2d_gbps > 0
            else "conservative_default"
        )
        self._attn_q1_ema = 0.08
        self._target_other_ema = 0.25
        self._target_compute_ratio_ema = 0.0
        self._exposed_copy_ema = 0.0
        self._completed_profile_rounds = 0
        self._network_ema = 0.05
        self._draft_rtt_ema_by_q: dict[int, float] = {}
        self._draft_rtt_samples_by_q: dict[int, int] = {}
        self._pending_network_ms = 0.0
        self._pending_decision = None
        self._draft_load = DraftLoadSnapshot()
        self._tp_snapshot = TPStragglerSnapshot()
        self._pending_grant: dict[str, object] = {}
        self._pending_round_started_ns: int | None = None
        self._round_started_ns: dict[int, int] = {}
        self._round_batch_states: dict[int, SpecStreamBatchState] = {}
        self._empirical: OrderedDict[tuple, EmpiricalCandidateCost] = OrderedDict()
        self._empirical_seen: dict[tuple, int] = {}
        self._draft_overlap_fraction_ema = 0.0
        self.parallel_feedback = None

    def start_round_clock(self, started_ns: int | None = None) -> None:
        """Start before Draft receive/Target submission in the outer scheduler.

        This clock is consumed by begin_round. Passing perf_counter_ns keeps
        the complete round boundary testable without CUDA synchronization.
        """
        self._pending_round_started_ns = (
            time.perf_counter_ns() if started_ns is None else int(started_ns)
        )

    def begin_round(
        self,
        meta,
        *,
        chunk_tokens: int = 0,
        batch_state: SpecStreamBatchState | None = None,
        round_started_ns: int | None = None,
    ) -> None:
        chunk_tokens = int(chunk_tokens)
        decision = self._pending_decision
        tp_forward = self._tp_snapshot.rank_forward_ms
        tp_collective = self._tp_snapshot.rank_collective_wait_ms
        gpu_history_tokens = sum(
            int(getattr(item, "gpu_history_len", 0)) for item in meta.items
        )
        cpu_history_tokens = sum(
            int(item.history_len) - int(getattr(item, "gpu_history_len", 0))
            for item in meta.items
        )
        candidate_costs = [
            {
                "q": int(cost.q),
                "parallel": float(cost.parallel_ms_per_useful_token),
                "ordinary": float(cost.ordinary_ms_per_useful_token),
                "verify_ms": float(cost.verify_ms),
                "draft_rtt_ms": float(cost.draft_rtt_ms),
            }
            for cost in getattr(decision, "candidate_costs", ())
        ]
        self._active[meta.round_id] = SpecStreamProfileRow(
            timestamp=time.time(),
            tp_rank=self.tp_rank,
            rid="|".join(item.rid for item in meta.items),
            round_id=meta.round_id,
            mode=meta.mode,
            q=meta.q_len,
            batch_size=max(len(meta.items), 1),
            context_tokens=meta.context_tokens,
            committed_len=max((item.committed_len for item in meta.items), default=0),
            history_len=max((item.history_len for item in meta.items), default=0),
            gpu_history_hit_tokens=gpu_history_tokens,
            cpu_history_miss_tokens=cpu_history_tokens,
            committed_tokens_total=sum(item.committed_len for item in meta.items),
            history_tokens_total=sum(item.history_len for item in meta.items),
            sealed_request_count=sum(item.history_len > 0 for item in meta.items),
            tail_tokens=max((item.tail_tokens for item in meta.items), default=0),
            chunk_tokens=chunk_tokens,
            num_chunks=(
                sum(
                    (
                        int(item.history_len)
                        - int(getattr(item, "gpu_history_len", 0))
                        + chunk_tokens
                        - 1
                    )
                    // chunk_tokens
                    for item in meta.items
                )
                if chunk_tokens > 0
                else 0
            ),
            network_wait_ms=self._pending_network_ms,
            fallback=bool(getattr(meta, "fallback", False)),
            fallback_reason=str(getattr(meta, "fallback_reason", "")),
            missing_draft_count=int(getattr(meta, "missing_draft_count", 0)),
            controller_selected_q=int(getattr(decision, "q", 0) or 0),
            controller_selected_mode=str(getattr(decision, "mode", "")),
            controller_planned_mode=str(getattr(decision, "planned_mode", "") or getattr(decision, "mode", "")),
            coexec_mode=str(getattr(decision, "coexec_mode", "")),
            coexec_reason=str(getattr(decision, "reason", "")),
            target_phase=str(self._pending_grant.get("target_phase", "")),
            predicted_slack_us=float(
                self._pending_grant.get("predicted_slack_us", 0.0) or 0.0
            ),
            grant_state=str(self._pending_grant.get("grant_state", "")),
            grant_epoch=int(self._pending_grant.get("grant_epoch", 0) or 0),
            grant_wait_ms=float(self._pending_grant.get("grant_wait_ms", 0.0) or 0.0),
            draft_step_ms=float(self._pending_grant.get("draft_step_ms", 0.0) or 0.0),
            draft_tpc_low=int(self._pending_grant.get("draft_tpc_low", -1) or 0),
            draft_tpc_high=int(self._pending_grant.get("draft_tpc_high", -1) or 0),
            draft_rtt_ema_ms=self._draft_load.rtt_ema_ms,
            draft_rtt_p95_ms=self._draft_load.rtt_p95_ms,
            draft_rtt_by_q=json.dumps(
                self._draft_rtt_ema_by_q, sort_keys=True, separators=(",", ":")
            ),
            draft_rtt_samples_by_q=json.dumps(
                self._draft_rtt_samples_by_q,
                sort_keys=True,
                separators=(",", ":"),
            ),
            pending_drafts=self._draft_load.pending_p95,
            draft_timeout_rate=self._draft_load.timeout_rate,
            draft_missing_ratio=self._draft_load.missing_ratio_ema,
            draft_reject_rate=self._draft_load.reject_rate,
            mps_active_thread_percentage=int(
                self.mps_environment.active_thread_percentage or 0
            ),
            mps_client_priority=(
                int(self.mps_environment.client_priority)
                if self.mps_environment.client_priority is not None
                else -1
            ),
            mps_sm_partition=self.mps_environment.sm_partition,
            tp_colocated_rank=(
                self._tp_snapshot.colocated_rank if self._tp_snapshot.samples else -1
            ),
            tp_rank_forward_ms=(
                tp_forward[self.tp_rank] if self.tp_rank < len(tp_forward) else 0.0
            ),
            tp_collective_wait_ms=(
                tp_collective[self.tp_rank]
                if self.tp_rank < len(tp_collective)
                else 0.0
            ),
            tp_rank_skew_ms=self._tp_snapshot.rank_skew_ms,
            tp_target_slowdown=self._tp_snapshot.target_slowdown,
            tp_baseline_ready=self._tp_snapshot.baseline_ready,
            tp_baseline_samples=self._tp_snapshot.baseline_samples,
            tp_overlap_active=str(self._tp_snapshot.overlap_active),
            tp_shape_key=json.dumps(self._tp_snapshot.shape_key, separators=(",", ":")),
            h2d_timing_source=self._h2d_timing_source,
            controller_cost_parallel=float(
                getattr(decision, "parallel_cost", 0.0) or 0.0
            ),
            controller_cost_ordinary=float(
                getattr(decision, "ordinary_cost", 0.0) or 0.0
            ),
            controller_candidate_costs=json.dumps(
                candidate_costs, sort_keys=True, separators=(",", ":")
            ),
        )
        started_ns = (
            self._pending_round_started_ns
            if round_started_ns is None
            else int(round_started_ns)
        )
        self._round_started_ns[meta.round_id] = (
            time.perf_counter_ns() if started_ns is None else started_ns
        )
        self._active[meta.round_id].round_timing_source = (
            "metadata_to_commit_wall" if started_ns is None else "outer_round_wall"
        )
        if batch_state is not None:
            self._round_batch_states[meta.round_id] = batch_state
        self._pending_round_started_ns = None
        self._pending_decision = None
        self._pending_grant = {}
        self._pending_network_ms = 0.0

    def record_grant(
        self, message, *, target_phase: str, predicted_slack_us: float = 0.0
    ) -> None:
        values = {
            "target_phase": str(target_phase),
            "predicted_slack_us": max(float(predicted_slack_us), 0.0),
            "grant_state": str(getattr(message, "grant_state", "") or ""),
            "grant_epoch": int(getattr(message, "grant_epoch", 0) or 0),
            "draft_tpc_low": int(getattr(message, "tpc_low", -1)),
            "draft_tpc_high": int(getattr(message, "tpc_high", -1)),
        }
        if self._active:
            row = self._active[max(self._active)]
            if values["grant_state"] == "SLACK_FILL":
                row.slack_fill_issued_tokens += max(int(getattr(message, "grant_tokens", 0) or 0), 0)
            for key, value in values.items():
                setattr(row, key, value)
        else:
            self._pending_grant.update(values)
        self._append_grant_event(
            "issued",
            message,
            target_phase=target_phase,
            predicted_slack_us=predicted_slack_us,
        )

    def record_grant_decision(
        self, decision, *, target_phase: str, predicted_slack_us: float = 0.0
    ) -> None:
        values = {
            "target_phase": str(target_phase),
            "predicted_slack_us": max(float(predicted_slack_us), 0.0),
            "grant_state": str(getattr(getattr(decision, "state", ""), "value", "")),
            "draft_tpc_low": int(getattr(decision, "tpc_low", -1)),
            "draft_tpc_high": int(getattr(decision, "tpc_high", -1)),
        }
        if self._active:
            row = self._active[max(self._active)]
            for key, value in values.items():
                setattr(row, key, value)
        else:
            self._pending_grant.update(values)

    def record_grant_ack(self, message, *, wait_ms: float = 0.0) -> None:
        values = {
            "grant_epoch": int(getattr(message, "grant_epoch", 0) or 0),
            "grant_wait_ms": max(float(wait_ms), 0.0),
            "draft_step_ms": max(
                float(getattr(message, "draft_step_ms", 0.0) or 0.0), 0.0
            ),
            "draft_tpc_low": int(getattr(message, "tpc_low", -1)),
            "draft_tpc_high": int(getattr(message, "tpc_high", -1)),
        }
        if self._active:
            row = self._active[max(self._active)]
            for key, value in values.items():
                setattr(row, key, value)
        else:
            self._pending_grant.update(values)
        self._append_grant_event("ack", message, wait_ms=wait_ms)

    def _append_grant_event(
        self,
        event: str,
        message,
        *,
        target_phase: str = "",
        predicted_slack_us: float = 0.0,
        wait_ms: float = 0.0,
    ) -> None:
        values = {
            "timestamp_ns": time.time_ns(),
            # Grant deadlines use the monotonic clock.  Preserve wall-clock
            # time for trace correlation, but record the comparable clock as
            # well so offline validation never mixes clock domains.
            "monotonic_ns": time.monotonic_ns(),
            "event": str(event),
            "request_id": str(getattr(message, "request_id", "") or ""),
            "spec_cnt": int(getattr(message, "spec_cnt", 0) or 0),
            "grant_epoch": int(getattr(message, "grant_epoch", 0) or 0),
            "grant_state": str(getattr(message, "grant_state", "") or ""),
            "grant_tokens": int(getattr(message, "grant_tokens", 0) or 0),
            "deadline_us": int(getattr(message, "deadline_us", 0) or 0),
            "tpc_low": int(getattr(message, "tpc_low", -1)),
            "tpc_high": int(getattr(message, "tpc_high", -1)),
            "draft_step_ms": max(
                float(getattr(message, "draft_step_ms", 0.0) or 0.0), 0.0
            ),
            "target_phase": str(target_phase),
            "predicted_slack_us": max(float(predicted_slack_us), 0.0),
            "wait_ms": max(float(wait_ms), 0.0),
        }
        with self._lock:
            self.grant_path.parent.mkdir(parents=True, exist_ok=True)
            exists = self.grant_path.exists() and self.grant_path.stat().st_size > 0
            with self.grant_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values))
                if not exists:
                    writer.writeheader()
                writer.writerow(values)

    def record_h2d(
        self, round_id: int, nbytes: int, elapsed_ms: float, cohort_size: int = 1
    ) -> None:
        row = self._active.get(round_id)
        if row is None:
            return
        row.h2d_bytes += int(nbytes)
        row.h2d_ops += 1
        row.h2d_ms += float(elapsed_ms)
        row.cohort_size = max(row.cohort_size, int(cohort_size))
        # ``elapsed_ms`` is transfer queue-residence time observed by Python,
        # not an isolated cudaMemcpy duration.  Keep it for scheduling-latency
        # diagnostics but never feed it back as PCIe bandwidth.

    def record_staging(self, round_id: int, transfer) -> None:
        """Record source/DMA work independently of copy and GPU wait timing.

        Call once beside record_h2d when a transfer is consumed. A staging
        window may contain many DMA submissions; metadata-cache hits and
        contiguous History regions reduce these independently of h2d_ops.
        """
        row = self._active.get(int(round_id))
        if row is None:
            return
        row.h2d_source_bytes += max(0, int(getattr(transfer, "source_nbytes", 0)))
        row.h2d_padding_bytes += max(0, int(getattr(transfer, "padding_nbytes", 0)))
        row.h2d_dma_ops += max(0, int(getattr(transfer, "dma_count", 0)))
        row.h2d_source_slabs += max(0, int(getattr(transfer, "source_count", 0)))
        row.host_slot_wait_ms += max(0.0, float(getattr(transfer, "host_wait_ms", 0.0)))
        row.host_pack_ms += max(0.0, float(getattr(transfer, "host_pack_ms", 0.0)))
        row.metadata_cache_hits += int(
            bool(getattr(transfer, "metadata_cache_hit", False))
        )

    def record_h2d_event(
        self,
        round_id: int,
        nbytes: int,
        elapsed_ms: float,
        *,
        target_wait_ms: float | None = None,
    ) -> None:
        """Record isolated copy time and its exact Target-stream stall."""

        row = self._active.get(int(round_id))
        elapsed_ms = float(elapsed_ms)
        nbytes = int(nbytes)
        if row is not None:
            row.h2d_event_ops += 1
            row.h2d_event_bytes += max(nbytes, 0)
            row.h2d_event_ms += max(elapsed_ms, 0.0)
            if target_wait_ms is not None:
                row.h2d_wait_event_ops += 1
                row.staging_wait_ms += max(float(target_wait_ms), 0.0)
                row.h2d_timing_source = "cuda_event_runtime+target_wait_event"
            else:
                row.h2d_timing_source = "cuda_event_runtime"
        if elapsed_ms <= 0.0 or nbytes <= 0:
            return
        measured_gbps = nbytes / elapsed_ms / 1e6
        if measured_gbps <= 0.0:
            return
        self._h2d_gbps_ema = 0.8 * self._h2d_gbps_ema + 0.2 * measured_gbps
        self._h2d_timing_source = "cuda_event_runtime"

    def record_h2d_window_observation(self, observation) -> None:
        row = self._active.get(int(getattr(observation, "round_id", -1)))
        if row is None:
            return
        active = bool(getattr(observation, "active", False))
        remaining_us = max(float(getattr(observation, "remaining_us", 0.0) or 0.0), 0.0)
        if active or not row.h2d_window_reason:
            row.h2d_window_id = int(getattr(observation, "window_id", 0) or 0)
            row.h2d_window_reason = str(getattr(observation, "reason", "") or "")
        row.h2d_window_active = row.h2d_window_active or active
        row.h2d_window_remaining_us = max(row.h2d_window_remaining_us, remaining_us)

    def record_attention(
        self, round_id: int, elapsed_ms: float, *, tail: bool = False
    ) -> None:
        row = self._active.get(round_id)
        if row is None:
            return
        if tail:
            row.tail_attn_ms += float(elapsed_ms)
            row.tail_attn_ops += 1
        else:
            row.stream_attn_ms += float(elapsed_ms)
            row.stream_attn_ops += 1
        # perf_counter around asynchronous calls measures Python submission,
        # including any host buffer waits. It is never a GPU kernel cost.

    def record_attention_event(
        self, round_id: int, elapsed_ms: float, *, tail: bool = False
    ) -> None:
        """Record a completed CUDA-event interval; callers must not synchronize."""
        row = self._active.get(round_id)
        if row is None or elapsed_ms < 0.0 or not math.isfinite(elapsed_ms):
            return
        row.attention_timing_source = "cpu_enqueue+cuda_event"
        if tail:
            row.tail_attn_event_ms += float(elapsed_ms)
            row.tail_attn_event_ops += 1
        else:
            row.stream_attn_event_ms += float(elapsed_ms)
            row.stream_attn_event_ops += 1
            per_query_ms = float(elapsed_ms) / max(row.q, 1)
            self._attn_q1_ema = 0.8 * self._attn_q1_ema + 0.2 * per_query_ms

    def record_target_forward(
        self, round_id: int, elapsed_ms: float, enqueue_ms: float = 0.0
    ) -> None:
        row = self._active.get(round_id)
        if row is None:
            return
        row.target_forward_ms = float(elapsed_ms)
        row.target_enqueue_ms = float(enqueue_ms)

    def record_grant_pump(
        self, round_id: int, *, iterations: int, wall_ms: float
    ) -> None:
        with self._lock:
            row = self._active.get(round_id)
            if row is not None:
                row.grant_pump_iterations = max(0, int(iterations))
                row.grant_pump_wall_ms = max(0.0, float(wall_ms))

    def record_round_timing(
        self,
        round_id: int,
        *,
        wall_ms: float,
        overlap_ms: float | None = None,
        overlap_source: str = "",
    ) -> None:
        row = self._active.get(round_id)
        if row is None:
            return
        if wall_ms < 0.0 or not math.isfinite(wall_ms):
            raise ValueError("round walltime must be finite and nonnegative")
        row.round_ms = float(wall_ms)
        row.round_timing_source = "explicit_round_wall"
        if overlap_ms is not None:
            if overlap_ms < 0.0 or not math.isfinite(overlap_ms):
                raise ValueError("overlap time must be finite and nonnegative")
            row.draft_verify_overlap_ms = float(overlap_ms)
            row.draft_overlap_timing_source = str(overlap_source or "measured_interval")

    def record_decision(
        self,
        decision,
        draft_load: DraftLoadSnapshot,
        tp_snapshot: TPStragglerSnapshot,
    ) -> None:
        self._pending_decision = decision
        self._draft_load = draft_load
        self._tp_snapshot = tp_snapshot

    def record_draft_load(self, draft_load: DraftLoadSnapshot) -> None:
        self._draft_load = draft_load
        if not self._active:
            return
        row = self._active[max(self._active)]
        row.draft_rtt_ema_ms = draft_load.rtt_ema_ms
        row.draft_rtt_p95_ms = draft_load.rtt_p95_ms
        row.pending_drafts = draft_load.pending_p95
        row.draft_timeout_rate = draft_load.timeout_rate
        row.draft_missing_ratio = draft_load.missing_ratio_ema
        row.draft_reject_rate = draft_load.reject_rate

    def record_tp_snapshot(self, snapshot: TPStragglerSnapshot) -> None:
        self._tp_snapshot = snapshot

    def record_network_wait(self, elapsed_ms: float) -> None:
        elapsed_ms = float(elapsed_ms)
        self._network_ema = 0.8 * self._network_ema + 0.2 * elapsed_ms
        if self._active:
            self._active[max(self._active)].network_wait_ms += elapsed_ms
        else:
            self._pending_network_ms += elapsed_ms

    def record_draft_rtt(self, q: int, elapsed_ms: float) -> None:
        """Update the observed end-to-end Drafter RTT bucket for one q."""

        q = int(q)
        elapsed_ms = float(elapsed_ms)
        if q < 1 or elapsed_ms < 0.0 or not math.isfinite(elapsed_ms):
            return
        previous = self._draft_rtt_ema_by_q.get(q)
        self._draft_rtt_ema_by_q[q] = (
            elapsed_ms if previous is None else 0.8 * previous + 0.2 * elapsed_ms
        )
        self._draft_rtt_samples_by_q[q] = self._draft_rtt_samples_by_q.get(q, 0) + 1

    def finish_round(
        self,
        round_id: int,
        accepted_tokens: int,
        staging_bytes: int,
        cpu_bytes: int,
        *,
        accepted_per_req: list[int] | tuple[int, ...] | None = None,
        post_states: list[object] | tuple[object, ...] | None = None,
        sealed_history_floors: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        row = self._active.pop(round_id, None)
        if row is None:
            return
        row.accepted_tokens = int(accepted_tokens)
        if accepted_per_req is not None:
            accepted_values = [int(value) for value in accepted_per_req]
            row.rejected_requests = sum(value < row.q for value in accepted_values)
            row.rollback_tokens = sum(
                max(row.q - value, 0) for value in accepted_values
            )
            rejected_positions = [value for value in accepted_values if value < row.q]
            row.reject_position = min(rejected_positions, default=-1)
        else:
            row.rollback_tokens = max(row.q * row.batch_size - row.accepted_tokens, 0)
            row.rejected_requests = int(row.rollback_tokens > 0)
            row.reject_position = (
                min(row.accepted_tokens, row.q) if row.rejected_requests else -1
            )
        if post_states is not None:
            states = list(post_states)
            row.post_committed_tokens_total = sum(
                int(getattr(state, "committed_len", 0)) for state in states
            )
            row.post_history_tokens_total = sum(
                int(getattr(state, "history_len", 0)) for state in states
            )
            row.post_logical_tokens_total = sum(
                int(getattr(state, "logical_len", 0)) for state in states
            )
            row.history_advanced_tokens = max(
                row.post_history_tokens_total - row.history_tokens_total, 0
            )
            row.sealed_request_count = sum(
                int(getattr(state, "history_len", 0)) > 0 for state in states
            )
            row.seal_inflight_count = sum(
                bool(getattr(state, "seal_inflight", False)) for state in states
            )
            floors = (
                [int(value) for value in sealed_history_floors]
                if sealed_history_floors is not None
                else [int(getattr(state, "history_len", 0)) for state in states]
            )
            margins = [
                int(getattr(state, "committed_len", 0)) - floor
                for state, floor in zip(states, floors)
            ]
            row.rollback_floor_min = min(margins, default=0)
            row.rollback_crossed_history = any(margin < 0 for margin in margins)
        row.staging_bytes = int(staging_bytes)
        row.cpu_history_bytes = int(cpu_bytes)
        started_ns = self._round_started_ns.pop(round_id, None)
        if row.round_timing_source != "explicit_round_wall" and started_ns is not None:
            row.round_ms = max(0.0, (time.perf_counter_ns() - started_ns) / 1e6)
        row.h2d_gbps = self._h2d_gbps_ema
        row.copy_floor_ms = row.h2d_bytes / (self._h2d_gbps_ema * 1e6)
        # Complete CUDA-event coverage replaces Python queue residence.  When
        # every transfer also has a Target wait-gate interval, exposed copy is
        # measured directly rather than inferred from enqueue timings.
        complete_event_coverage = bool(
            row.h2d_ops > 0
            and row.h2d_event_ops == row.h2d_ops
            and row.h2d_event_bytes >= row.h2d_bytes
        )
        measured_h2d_ms = row.h2d_event_ms if complete_event_coverage else row.h2d_ms
        complete_wait_coverage = bool(
            complete_event_coverage and row.h2d_wait_event_ops == row.h2d_ops
        )
        if complete_wait_coverage:
            estimated_exposed_copy_ms = min(
                row.target_forward_ms,
                row.h2d_event_ms,
                max(row.staging_wait_ms, 0.0),
            )
        else:
            # Without complete GPU wait events there is no measured overlap.
            # CPU submission intervals and copy queue-residence times cannot
            # be subtracted from the Target CUDA timeline.
            estimated_exposed_copy_ms = 0.0
        row.exposed_copy_ms = estimated_exposed_copy_ms
        row.copy_compute_overlap = (
            max(0.0, 1.0 - estimated_exposed_copy_ms / measured_h2d_ms)
            if measured_h2d_ms > 0 and complete_wait_coverage
            else 0.0
        )
        complete_attn_coverage = bool(
            row.stream_attn_ops + row.tail_attn_ops > 0
            and row.stream_attn_event_ops == row.stream_attn_ops
            and row.tail_attn_event_ops == row.tail_attn_ops
        )
        if complete_attn_coverage and (complete_wait_coverage or row.h2d_ops == 0):
            other = max(
                0.0,
                row.target_forward_ms
                - row.stream_attn_event_ms
                - row.tail_attn_event_ms
                - estimated_exposed_copy_ms,
            )
            self._target_other_ema = 0.8 * self._target_other_ema + 0.2 * other
        compute_ratio = (
            max(0.0, row.target_forward_ms - estimated_exposed_copy_ms)
            / row.target_forward_ms
            if row.target_forward_ms > 0
            and (complete_wait_coverage or row.h2d_ops == 0)
            else 0.0
        )
        if self._completed_profile_rounds == 0:
            self._target_compute_ratio_ema = compute_ratio
            self._exposed_copy_ema = estimated_exposed_copy_ms
        else:
            self._target_compute_ratio_ema = (
                0.8 * self._target_compute_ratio_ema + 0.2 * compute_ratio
            )
            self._exposed_copy_ema = (
                0.8 * self._exposed_copy_ema + 0.2 * estimated_exposed_copy_ms
            )
        self._completed_profile_rounds += 1
        rollback_denominator = max(row.q * row.batch_size, 1)
        row.rollback_ratio = min(
            max(row.rollback_tokens / rollback_denominator, 0.0), 1.0
        )
        if row.draft_overlap_timing_source != "unmeasured":
            denominator = min(row.target_forward_ms, row.draft_rtt_ema_ms)
            row.draft_verify_overlap = (
                min(row.draft_verify_overlap_ms / denominator, 1.0)
                if denominator > 0
                else 0.0
            )
            self._draft_overlap_fraction_ema = (
                0.8 * self._draft_overlap_fraction_ema + 0.2 * row.draft_verify_overlap
            )
        batch_state = self._round_batch_states.pop(round_id, None)
        if batch_state is not None:
            self._record_empirical_cost(batch_state, row)
            if self.parallel_feedback is not None:
                self.parallel_feedback(batch_state, row)
        self._append(row)

    def _record_empirical_cost(
        self, batch_state: SpecStreamBatchState, row: SpecStreamProfileRow
    ) -> None:
        if row.fallback or row.target_forward_ms <= 0.0 or row.accepted_tokens <= 0:
            return
        key = (workload_shape_key(batch_state), row.q, row.mode)
        seen = self._empirical_seen.get(key, 0) + 1
        self._empirical_seen[key] = seen
        # Discard initial compiler/cache warmup for each execution shape.
        if seen <= 2:
            return
        previous = self._empirical.get(key)
        full_round = row.round_timing_source in (
            "outer_round_wall",
            "explicit_round_wall",
        )
        round_ms = row.round_ms if full_round else 0.0
        useful = row.accepted_tokens / max(row.batch_size, 1)
        point = EmpiricalCandidateCost(
            shape_key=key[0],
            q=row.q,
            mode=row.mode,
            verify_ms=(
                row.target_forward_ms
                if previous is None
                else 0.8 * previous.verify_ms + 0.2 * row.target_forward_ms
            ),
            round_ms=(
                round_ms
                if previous is None or previous.round_ms <= 0
                else (
                    0.8 * previous.round_ms + 0.2 * round_ms
                    if full_round
                    else previous.round_ms
                )
            ),
            useful_tokens=(
                useful
                if previous is None
                else 0.8 * previous.useful_tokens + 0.2 * useful
            ),
            samples=1 if previous is None else previous.samples + 1,
        )
        self._empirical[key] = point
        self._empirical.move_to_end(key)
        while len(self._empirical) > 256:
            evicted_key, _ = self._empirical.popitem(last=False)
            self._empirical_seen.pop(evicted_key, None)
        # Bound one-off shapes that never survive the two-round warmup too.
        while len(self._empirical_seen) > 512:
            self._empirical_seen.pop(next(iter(self._empirical_seen)))

    def snapshot(self) -> SpecStreamCostProfile:
        return SpecStreamCostProfile(
            h2d_gbps=max(self._h2d_gbps_ema, 0.1),
            target_other_ms=max(self._target_other_ema, 0.0),
            attention_chunk_ms_q1=max(self._attn_q1_ema, 0.001),
            network_ms=max(self._network_ema, 0.0),
            draft_rtt_ms_by_q=tuple(sorted(self._draft_rtt_ema_by_q.items())),
            draft_rtt_samples_by_q=tuple(sorted(self._draft_rtt_samples_by_q.items())),
            empirical_candidates=tuple(self._empirical.values()),
            draft_overlap_fraction=self._draft_overlap_fraction_ema,
            target_compute_ratio=(
                self._target_compute_ratio_ema
                if self._completed_profile_rounds >= 4
                else 0.0
            ),
            exposed_copy_ms=(
                self._exposed_copy_ema if self._completed_profile_rounds >= 4 else 0.0
            ),
        )

    def _append(self, row: SpecStreamProfileRow) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            values = asdict(row)
            fieldnames = list(values)
            path = self._schema_compatible_path(fieldnames)
            exists = path.exists() and path.stat().st_size > 0
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if not exists:
                    writer.writeheader()
                writer.writerow(values)

    def _schema_compatible_path(self, fieldnames: list[str]) -> Path:
        """Never append a new row layout below an old CSV header.

        SpecStream profiles are commonly reused across code updates.  Appending
        a wider dataclass row to an older header silently shifts every later
        column, producing plausible-looking but invalid H2D/acceptance numbers.
        Keep the requested path when its header matches; otherwise route this
        schema to a deterministic sibling file.
        """

        if not self.path.exists() or self.path.stat().st_size == 0:
            return self.path
        with self.path.open(encoding="utf-8", newline="") as handle:
            current = next(csv.reader(handle), [])
        if current == fieldnames:
            return self.path
        digest = f"{zlib.crc32(','.join(fieldnames).encode()):08x}"
        suffix = self.path.suffix or ".csv"
        candidate = self.path.with_name(f"{self.path.stem}.schema-{digest}{suffix}")
        if candidate.exists() and candidate.stat().st_size > 0:
            with candidate.open(encoding="utf-8", newline="") as handle:
                candidate_header = next(csv.reader(handle), [])
            if candidate_header != fieldnames:
                raise RuntimeError("SpecStream profile schema hash collision")
        return candidate
