from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import threading
import time
import zlib

from sglang.srt.speculative.spectre.specstream.cost_model import (
    SpecStreamCostProfile,
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
    tail_tokens: int = 0
    chunk_tokens: int = 0
    num_chunks: int = 0
    cohort_size: int = 1
    h2d_bytes: int = 0
    h2d_ops: int = 0
    h2d_ms: float = 0.0
    h2d_gbps: float = 0.0
    h2d_timing_source: str = ""
    copy_floor_ms: float = 0.0
    stream_attn_ms: float = 0.0
    stream_attn_ops: int = 0
    tail_attn_ms: float = 0.0
    target_forward_ms: float = 0.0
    target_enqueue_ms: float = 0.0
    draft_ms: float = 0.0
    draft_rtt_ema_ms: float = 0.0
    draft_rtt_p95_ms: float = 0.0
    pending_drafts: int = 0
    draft_timeout_rate: float = 0.0
    draft_missing_ratio: float = 0.0
    draft_reject_rate: float = 0.0
    network_wait_ms: float = 0.0
    round_ms: float = 0.0
    exposed_copy_ms: float = 0.0
    copy_compute_overlap: float = 0.0
    draft_verify_overlap: float = 0.0
    staging_wait_ms: float = 0.0
    gpu_kv_bytes: int = 0
    staging_bytes: int = 0
    cpu_history_bytes: int = 0
    accepted_tokens: int = 0
    reject_position: int = -1
    rollback_ratio: float = 0.0
    controller_cost_parallel: float = 0.0
    controller_cost_ordinary: float = 0.0
    controller_selected_q: int = 0
    controller_selected_mode: str = ""
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
        self._pending_network_ms = 0.0
        self._pending_decision = None
        self._draft_load = DraftLoadSnapshot()
        self._tp_snapshot = TPStragglerSnapshot()
        self._pending_grant: dict[str, object] = {}

    def begin_round(self, meta, *, chunk_tokens: int = 0) -> None:
        chunk_tokens = int(chunk_tokens)
        decision = self._pending_decision
        tp_forward = self._tp_snapshot.rank_forward_ms
        tp_collective = self._tp_snapshot.rank_collective_wait_ms
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
            tail_tokens=max((item.tail_tokens for item in meta.items), default=0),
            chunk_tokens=chunk_tokens,
            num_chunks=(
                sum(
                    (item.history_len + chunk_tokens - 1) // chunk_tokens
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
            h2d_timing_source=self._h2d_timing_source,
        )
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
            for key, value in values.items():
                setattr(row, key, value)
        else:
            self._pending_grant.update(values)

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

    def record_attention(
        self, round_id: int, elapsed_ms: float, *, tail: bool = False
    ) -> None:
        row = self._active.get(round_id)
        if row is None:
            return
        if tail:
            row.tail_attn_ms += float(elapsed_ms)
        else:
            row.stream_attn_ms += float(elapsed_ms)
            row.stream_attn_ops += 1
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
        stream = row.stream_attn_ms + row.tail_attn_ms
        other = max(0.0, elapsed_ms - stream)
        self._target_other_ema = 0.8 * self._target_other_ema + 0.2 * other

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

    def finish_round(
        self, round_id: int, accepted_tokens: int, staging_bytes: int, cpu_bytes: int
    ) -> None:
        row = self._active.pop(round_id, None)
        if row is None:
            return
        row.accepted_tokens = int(accepted_tokens)
        row.staging_bytes = int(staging_bytes)
        row.cpu_history_bytes = int(cpu_bytes)
        row.round_ms = row.target_forward_ms + row.network_wait_ms
        row.h2d_gbps = self._h2d_gbps_ema
        row.copy_floor_ms = row.h2d_bytes / (self._h2d_gbps_ema * 1e6)
        estimated_exposed_copy_ms = min(
            row.target_forward_ms,
            max(0.0, row.h2d_ms - row.stream_attn_ms),
        )
        row.exposed_copy_ms = estimated_exposed_copy_ms
        row.copy_compute_overlap = (
            max(0.0, 1.0 - estimated_exposed_copy_ms / row.h2d_ms)
            if row.h2d_ms > 0
            else 0.0
        )
        compute_ratio = (
            max(0.0, row.target_forward_ms - estimated_exposed_copy_ms)
            / row.target_forward_ms
            if row.target_forward_ms > 0
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
        row.rollback_ratio = 1.0 - min(row.accepted_tokens, row.q) / max(row.q, 1)
        self._append(row)

    def snapshot(self) -> SpecStreamCostProfile:
        return SpecStreamCostProfile(
            h2d_gbps=max(self._h2d_gbps_ema, 0.1),
            target_other_ms=max(self._target_other_ema, 0.0),
            attention_chunk_ms_q1=max(self._attn_q1_ema, 0.001),
            network_ms=max(self._network_ema, 0.0),
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
