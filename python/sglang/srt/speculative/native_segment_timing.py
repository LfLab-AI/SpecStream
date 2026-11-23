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
    target_spill: object = None
    target_start: torch.cuda.Event | None = None
    target_end: torch.cuda.Event | None = None
    target_stage_wall_ms: float = 0.0
    post_verify_events: list = field(default_factory=list)
    post_verify_host_ms: float = 0.0
    lookahead_admitted: bool = False


class NativeSegmentTiming:
    def __init__(self, path, rank, stride):
        if stride < 1:
            raise ValueError("Native segment profiling stride must be positive")
        path = Path(path)
        self.path = path.with_name(f"{path.stem}.tp{rank}{path.suffix}")
        self.rank = rank
        self.stride = stride
        self.round_id = 0
        self.active = None
        self.pending = deque()
        self.lookahead_admission = None

    def begin(self, timestamp, batch_size, q, draft_spill):
        self.round_id += 1
        if (self.round_id - 1) % self.stride == 0:
            self.active = _Round(self.round_id, timestamp, batch_size, q, draft_spill.cpu_attention_ns)

    @contextmanager
    def target_stage(self, target_spill):
        sample = self.active
        if sample is None:
            yield
            return
        sample.target_spill = target_spill
        target_spill.begin_verify_timing(sample.round_id)
        start = time.perf_counter()
        try:
            yield
        finally:
            sample.target_stage_wall_ms = (time.perf_counter() - start) * 1e3
            target_spill.end_verify_timing(sample.round_id)

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

    def finish(self, wall_ms, committed_draft, draft_spill):
        if self.active is not None:
            sample = self.active
            self.active = None
            if sample.target_end is None:
                raise RuntimeError("Sampled native decode round did not record Target forward")
            row = {
                "round_id": sample.round_id,
                "timestamp": sample.timestamp,
                "tp_rank": self.rank,
                "batch_size": sample.batch_size,
                "q": sample.q,
                "committed_output_tokens": sample.batch_size + committed_draft,
                "round_wall_ms": wall_ms,
                "draft_cpu_attention_ms": (draft_spill.cpu_attention_ns - sample.draft_cpu_start_ns) / 1e6,
                "target_stage_wall_ms": sample.target_stage_wall_ms,
                "post_verify_draft_host_ms": sample.post_verify_host_ms,
                "lookahead_admitted": sample.lookahead_admitted,
            }
            self.pending.append((sample, row))
        self.flush()

    def flush(self):
        rows = []
        while self.pending:
            sample, row = self.pending[0]
            if not sample.target_end.query() or any(not end.query() for _, end in sample.post_verify_events):
                break
            timings = sample.target_spill.take_verify_timing(sample.round_id)
            if timings is None:
                break
            row.update(
                timings, target_forward_cuda_ms=sample.target_start.elapsed_time(sample.target_end),
                post_verify_draft_ms=sum(begin.elapsed_time(end) for begin, end in sample.post_verify_events),
            )
            self.pending.popleft()
            if self.lookahead_admission is not None:
                self.lookahead_admission.observe_transfer(row)
            rows.append(row)
        if rows:
            with self.path.open("a", encoding="utf-8") as output:
                output.writelines(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
