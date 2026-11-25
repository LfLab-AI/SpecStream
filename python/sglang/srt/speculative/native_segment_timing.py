"""Per-rank timing for sampled Draft/Target decoding rounds."""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import torch


@dataclass
class _Round:
    round_id: int
    timestamp: float
    batch_size: int
    q: int
    draft_cpu_start_ns: int
    state: dict | None = None
    round_start: torch.cuda.Event | None = None
    round_end: torch.cuda.Event | None = None
    target_spill: object = None
    target_start: torch.cuda.Event | None = None
    target_end: torch.cuda.Event | None = None
    target_stage_wall_ms: float = 0.0
    post_verify_events: list = field(default_factory=list)
    proposal_events: list = field(default_factory=list)
    proposal_host_ms: float = 0.0
    post_verify_host_ms: float = 0.0
    lookahead_admitted: bool = False


class NativeSegmentTiming:
    def __init__(self, path, rank, stride, *, feedback=False):
        if stride < 1:
            raise ValueError("Native segment profiling stride must be positive")
        self.path = None
        if path:
            path = Path(path)
            self.path = path.with_name(f"{path.stem}.tp{rank}{path.suffix}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.stride = stride
        self.round_id = 0
        self.active = None
        self.pending = deque()
        self.completed = deque()
        self.feedback = feedback
        self.lookahead_admission = None

    def begin(self, timestamp, batch_size, q, draft_spill, *, state=None, sample=None):
        self.round_id += 1
        if sample is None:
            sample = (self.round_id - 1) % self.stride == 0
        self.active = None
        if sample:
            cpu_ns = draft_spill.cpu_attention_ns if draft_spill is not None else 0
            self.active = _Round(self.round_id, timestamp, batch_size, q, cpu_ns, state=state)
            self.active.round_start = torch.cuda.Event(enable_timing=True)
            self.active.round_start.record()

    @contextmanager
    def target_stage(self, target_spill):
        sample = self.active
        if sample is None:
            yield
            return
        sample.target_spill = target_spill
        if target_spill is not None:
            target_spill.begin_verify_timing(sample.round_id)
        start = time.perf_counter()
        try:
            yield
        finally:
            sample.target_stage_wall_ms = (time.perf_counter() - start) * 1e3
            if target_spill is not None:
                target_spill.end_verify_timing(sample.round_id)

    @contextmanager
    def proposal_draft(self):
        sample = self.active
        if sample is None:
            yield
            return
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        start = time.perf_counter()
        try:
            yield
        finally:
            end.record()
            sample.proposal_host_ms += (time.perf_counter() - start) * 1e3
            sample.proposal_events.append((begin, end))

    @contextmanager
    def target_forward(self):
        sample = self.active
        if sample is None:
            yield
            return
        if sample.target_start is not None:
            raise RuntimeError("Expected one Target model forward per sampled round")
        sample.target_start = torch.cuda.Event(enable_timing=True)
        sample.target_end = torch.cuda.Event(enable_timing=True)
        sample.target_start.record()
        try:
            yield
        finally:
            sample.target_end.record()

    @contextmanager
    def post_verify_draft(self):
        sample = self.active
        if sample is None:
            yield
            return
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        start = time.perf_counter()
        try:
            yield
        finally:
            end.record()
            sample.post_verify_host_ms += (time.perf_counter() - start) * 1e3
            sample.post_verify_events.append((begin, end))

    def finish(self, wall_ms, committed_draft, draft_spill, *, useful_output=None,
               accepted_lengths=None, lookahead_metrics=None, actual_state=None):
        if self.active is not None:
            sample = self.active
            self.active = None
            if sample.target_end is None:
                raise RuntimeError("Sampled native decode round did not record Target forward")
            sample.round_end = torch.cuda.Event(enable_timing=True)
            sample.round_end.record()
            cpu_ns = draft_spill.cpu_attention_ns if draft_spill is not None else 0
            row = {
                "round_id": sample.round_id,
                "timestamp": sample.timestamp,
                "tp_rank": self.rank,
                "batch_size": sample.batch_size,
                "q": sample.q,
                "state": sample.state,
                "actual_state": actual_state,
                "placement_changed": bool(sample.state is not None and actual_state is not None and any(
                    sample.state[key] != actual_state[key] for key in ("target_cpu", "draft_cpu")
                )),
                "committed_output_tokens": (sample.batch_size + committed_draft
                                            if useful_output is None else useful_output),
                "accepted_lengths": accepted_lengths,
                "round_wall_ms": wall_ms,
                "draft_cpu_attention_ms": (cpu_ns - sample.draft_cpu_start_ns) / 1e6,
                "target_stage_wall_ms": sample.target_stage_wall_ms,
                "post_verify_draft_host_ms": sample.post_verify_host_ms,
                "lookahead_admitted": sample.lookahead_admitted,
                "draft_proposal_host_ms": sample.proposal_host_ms,
                "target_overlap_ms": (lookahead_metrics or {}).get("target_overlap_ms", 0.0),
            }
            self.pending.append((sample, row))
        self.flush()

    def flush(self):
        rows = []
        while self.pending:
            sample, row = self.pending[0]
            if not sample.round_end.query():
                break
            timings = (
                sample.target_spill.take_verify_timing(sample.round_id)
                if sample.target_spill is not None else dict(
                    target_h2d_dma_ms=0.0, target_kv_wait_ms=0.0,
                    target_profiled_h2d_bytes=0, target_h2d_windows=0,
                    target_h2d_dma_count=0, target_h2d_submit_ms=0.0,
                )
            )
            if timings is None:
                break
            row.update(
                timings, target_forward_cuda_ms=sample.target_start.elapsed_time(sample.target_end),
                post_verify_draft_ms=sum(begin.elapsed_time(end) for begin, end in sample.post_verify_events),
                draft_proposal_ms=sum(begin.elapsed_time(end) for begin, end in sample.proposal_events),
                round_elapsed_cuda_ms=sample.round_start.elapsed_time(sample.round_end),
                target_h2d_event_ms=timings["target_h2d_dma_ms"],
                target_h2d_bytes=timings["target_profiled_h2d_bytes"],
            )
            self.pending.popleft()
            if self.lookahead_admission is not None:
                self.lookahead_admission.observe_transfer(row)
            if self.feedback:
                self.completed.append(row)
            rows.append(row)
        if rows and self.path is not None:
            with self.path.open("a", encoding="utf-8") as output:
                output.writelines(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)

    def drain_ready(self):
        """Return completed samples without waiting for CUDA work to finish."""
        self.flush()
        rows = list(self.completed)
        self.completed.clear()
        return rows
