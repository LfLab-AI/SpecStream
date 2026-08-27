from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RoundEvents:
    target_stream: Any
    candidate_ready: Any
    target_start: Any
    ahead_start: Any
    ahead_done: Any
    repair_start: Any
    repair_done: Any


@dataclass(frozen=True)
class RoundTiming:
    verify_ms: float
    ahead_ms: float
    actual_overlap_ms: float
    repair_ms: float = 0.0

    @property
    def overlap_ratio(self) -> float:
        if self.ahead_ms <= 0:
            return 0.0
        return min(1.0, self.actual_overlap_ms / self.ahead_ms)


class InProcessStreamRuntime:
    """Own the low-priority Draft stream and round-scoped CUDA events."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.device_module = torch.get_device_module(device)
        priority = 0
        get_priority_range = getattr(
            self.device_module, "get_stream_priority_range", None
        )
        if get_priority_range is not None:
            try:
                least_priority, _ = get_priority_range()
                priority = least_priority
            except (RuntimeError, TypeError):
                priority = 0
        self.draft_stream = self.device_module.Stream(priority=priority)

    def new_round(self) -> RoundEvents:
        event = lambda: self.device_module.Event(enable_timing=True)
        target_stream = self.device_module.current_stream()
        events = RoundEvents(
            target_stream=target_stream,
            candidate_ready=event(),
            target_start=event(),
            ahead_start=event(),
            ahead_done=event(),
            repair_start=event(),
            repair_done=event(),
        )
        events.candidate_ready.record(target_stream)
        events.target_start.record(target_stream)
        return events

    @contextlib.contextmanager
    def draft_context(self, events: RoundEvents):
        with self.device_module.stream(self.draft_stream):
            self.draft_stream.wait_event(events.candidate_ready)
            events.ahead_start.record(self.draft_stream)
            yield
            events.ahead_done.record(self.draft_stream)

    def wait_target_for_candidates(self, candidate_ready: Any) -> None:
        self.device_module.current_stream().wait_event(candidate_ready)

    def record_repair_done(self, events: RoundEvents) -> None:
        events.repair_done.record(self.device_module.current_stream())

    def record_repair_start(self, events: RoundEvents) -> None:
        events.repair_start.record(self.device_module.current_stream())

    def repair_timing(self, events: RoundEvents, timing: RoundTiming) -> RoundTiming:
        events.repair_done.synchronize()
        return RoundTiming(
            verify_ms=timing.verify_ms,
            ahead_ms=timing.ahead_ms,
            actual_overlap_ms=timing.actual_overlap_ms,
            repair_ms=float(events.repair_start.elapsed_time(events.repair_done)),
        )

    def wait_for_reconciliation(self, events: RoundEvents, verify_done: Any) -> None:
        # This is the only CPU wait in the optimistic path.  It is scoped to
        # the two round events and never synchronizes the whole device/stream.
        verify_done.synchronize()
        events.ahead_done.synchronize()

    def timing(self, events: RoundEvents, verify_done: Any) -> RoundTiming:
        verify_ms = float(events.target_start.elapsed_time(verify_done))
        ahead_start_ms = float(events.target_start.elapsed_time(events.ahead_start))
        ahead_end_ms = float(events.target_start.elapsed_time(events.ahead_done))
        ahead_ms = max(0.0, ahead_end_ms - ahead_start_ms)
        overlap_ms = max(
            0.0,
            min(verify_ms, ahead_end_ms) - max(0.0, ahead_start_ms),
        )
        return RoundTiming(
            verify_ms=verify_ms,
            ahead_ms=ahead_ms,
            actual_overlap_ms=overlap_ms,
        )
