from __future__ import annotations

import atexit
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Iterable

from sglang.srt.speculative.specstream_inproc.ahead_state import ReconcileResult
from sglang.srt.speculative.specstream_inproc.stream_runtime import RoundTiming

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoundProfile:
    round_id: int
    batch_size: int
    ahead_depth: int
    promoted: bool
    ahead_tokens_generated: int
    ahead_tokens_reused: int
    ahead_tokens_discarded: int
    ahead_reuse_ratio: float
    local_rollback_tokens: int
    repair_tokens: int
    verify_ms: float
    ahead_ms: float
    repair_ms: float
    actual_overlap_ms: float
    overlap_ratio: float
    target_slowdown: float | None


class InProcessAheadProfiler:
    def __init__(self, *, profile_path: str, flush_interval: int = 128) -> None:
        self.profile_path = profile_path
        self.flush_interval = max(1, int(flush_interval))
        self._lock = Lock()
        self._buffer: list[RoundProfile] = []
        self.rounds = 0
        self.generated = 0
        self.reused = 0
        self.discarded = 0
        self.rollback_tokens = 0
        self.repair_tokens = 0
        atexit.register(self.flush)

    def record(
        self,
        *,
        batch_size: int,
        ahead_depth: int,
        results: Iterable[ReconcileResult],
        timing: RoundTiming,
        target_slowdown: float | None = None,
    ) -> RoundProfile:
        result_list = list(results)
        generated = sum(result.ahead_generated for result in result_list)
        reused = sum(result.ahead_reused for result in result_list)
        discarded = sum(result.ahead_discarded for result in result_list)
        rollback = sum(result.rollback_tokens for result in result_list)
        repair = sum(result.repair_tokens for result in result_list)
        round_id = max((result.round_id for result in result_list), default=0)
        profile = RoundProfile(
            round_id=round_id,
            batch_size=batch_size,
            ahead_depth=ahead_depth,
            promoted=bool(result_list) and all(r.promotable for r in result_list),
            ahead_tokens_generated=generated,
            ahead_tokens_reused=reused,
            ahead_tokens_discarded=discarded,
            ahead_reuse_ratio=reused / generated if generated else 0.0,
            local_rollback_tokens=rollback,
            repair_tokens=repair,
            verify_ms=timing.verify_ms,
            ahead_ms=timing.ahead_ms,
            repair_ms=timing.repair_ms,
            actual_overlap_ms=timing.actual_overlap_ms,
            overlap_ratio=timing.overlap_ratio,
            target_slowdown=target_slowdown,
        )

        with self._lock:
            self.rounds += 1
            self.generated += generated
            self.reused += reused
            self.discarded += discarded
            self.rollback_tokens += rollback
            self.repair_tokens += repair
            if self.profile_path:
                self._buffer.append(profile)
                if len(self._buffer) >= self.flush_interval:
                    self._flush_locked()

        if self.rounds % self.flush_interval == 0:
            logger.info("SpecStream in-process metrics: %s", self.snapshot())
        return profile

    def snapshot(self) -> dict[str, float | int]:
        reuse_ratio = self.reused / self.generated if self.generated else 0.0
        return {
            "rounds": self.rounds,
            "ahead_tokens_generated": self.generated,
            "ahead_tokens_reused": self.reused,
            "ahead_tokens_discarded": self.discarded,
            "ahead_reuse_ratio": reuse_ratio,
            "local_rollback_tokens": self.rollback_tokens,
            "repair_tokens": self.repair_tokens,
        }

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer or not self.profile_path:
            return
        path = Path(self.profile_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            for profile in self._buffer:
                output.write(json.dumps(asdict(profile), ensure_ascii=False) + "\n")
        self._buffer.clear()
