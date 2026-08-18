from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import threading
import time

from sglang.srt.speculative.spectre.specstream.cost_model import (
    SpecStreamCostProfile,
)


@dataclass
class SpecStreamProfileRow:
    timestamp: float
    tp_rank: int
    rid: str
    round_id: int
    mode: str
    q: int
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
    stream_attn_ms: float = 0.0
    tail_attn_ms: float = 0.0
    target_forward_ms: float = 0.0
    draft_ms: float = 0.0
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
    fallback: bool = False
    fallback_reason: str = ""
    missing_draft_count: int = 0
    shadow_max_abs: float = 0.0
    logit_margin: float = 0.0
    token_mismatch: bool = False


class SpecStreamProfiler:
    def __init__(self, path: str, tp_rank: int, tp_size: int) -> None:
        base = Path(path)
        if tp_size > 1:
            base = base.with_name(f"{base.stem}.tp{tp_rank}{base.suffix or '.csv'}")
        self.path = base
        self.tp_rank = tp_rank
        self._lock = threading.Lock()
        self._active: dict[int, SpecStreamProfileRow] = {}
        self._h2d_gbps_ema = 12.0
        self._attn_q1_ema = 0.08
        self._target_other_ema = 0.25
        self._network_ema = 0.05
        self._pending_network_ms = 0.0

    def begin_round(self, meta, *, chunk_tokens: int = 0) -> None:
        chunk_tokens = int(chunk_tokens)
        self._active[meta.round_id] = SpecStreamProfileRow(
            timestamp=time.time(),
            tp_rank=self.tp_rank,
            rid="|".join(item.rid for item in meta.items),
            round_id=meta.round_id,
            mode=meta.mode,
            q=meta.q_len,
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
        )
        self._pending_network_ms = 0.0

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
        if elapsed_ms > 0:
            gbps = nbytes / elapsed_ms / 1e6
            self._h2d_gbps_ema = 0.8 * self._h2d_gbps_ema + 0.2 * gbps

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
            per_query_ms = float(elapsed_ms) / max(row.q, 1)
            self._attn_q1_ema = 0.8 * self._attn_q1_ema + 0.2 * per_query_ms

    def record_target_forward(self, round_id: int, elapsed_ms: float) -> None:
        row = self._active.get(round_id)
        if row is None:
            return
        row.target_forward_ms = float(elapsed_ms)
        stream = row.stream_attn_ms + row.tail_attn_ms
        other = max(0.0, elapsed_ms - stream)
        self._target_other_ema = 0.8 * self._target_other_ema + 0.2 * other

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
        row.h2d_gbps = row.h2d_bytes / row.h2d_ms / 1e6 if row.h2d_ms > 0 else 0.0
        row.rollback_ratio = 1.0 - min(row.accepted_tokens, row.q) / max(row.q, 1)
        self._append(row)

    def snapshot(self) -> SpecStreamCostProfile:
        return SpecStreamCostProfile(
            h2d_gbps=max(self._h2d_gbps_ema, 0.1),
            target_other_ms=max(self._target_other_ema, 0.0),
            attention_chunk_ms_q1=max(self._attn_q1_ema, 0.001),
            network_ms=max(self._network_ema, 0.0),
        )

    def _append(self, row: SpecStreamProfileRow) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            exists = self.path.exists() and self.path.stat().st_size > 0
            with self.path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(asdict(row)))
                if not exists:
                    writer.writeheader()
                writer.writerow(asdict(row))
