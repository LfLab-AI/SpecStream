from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SlackSnapshot:
    target_phase: str = "unknown"
    target_phase_ms: float = 0.0
    draft_step_ms: float = 0.0
    predicted_slack_us: float = 0.0
    samples: int = 0


class SlackProfiler:
    """Small online estimator; it never creates uncalibrated TPC choices."""

    def __init__(self, alpha: float = 0.2) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._phase_ms: dict[str, float] = {}
        self._draft_step_ms = 0.0
        self._draft_step_ms_by_shape: dict[tuple[int, str], float] = {}
        self._samples = 0
        self._last_phase = "unknown"

    def _update(self, previous: float, value: float) -> float:
        return (
            value
            if previous <= 0
            else (1.0 - self.alpha) * previous + self.alpha * value
        )

    def record_target_phase(self, phase: str, elapsed_ms: float) -> None:
        elapsed_ms = float(elapsed_ms)
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            elapsed_ms = 0.0
        self._last_phase = str(phase or "unknown")
        self._phase_ms[self._last_phase] = self._update(
            self._phase_ms.get(self._last_phase, 0.0), elapsed_ms
        )
        self._samples += 1

    def _update_draft_step(self, previous: float, elapsed_ms: float) -> float:
        # A slower observation must affect the very next launch deadline.  A
        # faster observation is incorporated gradually so the estimator does
        # not become optimistic after a single unusually fast token.
        if previous <= 0:
            return elapsed_ms
        return max(elapsed_ms, self._update(previous, elapsed_ms))

    def record_draft_step(
        self,
        elapsed_ms: float,
        *,
        draft_bs: int | None = None,
        draft_ctx_bucket: str | None = None,
    ) -> None:
        elapsed_ms = float(elapsed_ms)
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            return
        self._draft_step_ms = self._update_draft_step(self._draft_step_ms, elapsed_ms)
        if draft_bs is not None and draft_ctx_bucket is not None:
            key = (int(draft_bs), str(draft_ctx_bucket))
            self._draft_step_ms_by_shape[key] = self._update_draft_step(
                self._draft_step_ms_by_shape.get(key, 0.0), elapsed_ms
            )

    def snapshot(
        self,
        phase: str | None = None,
        *,
        draft_bs: int | None = None,
        draft_ctx_bucket: str | None = None,
    ) -> SlackSnapshot:
        phase = str(phase or self._last_phase)
        phase_ms = self._phase_ms.get(phase, 0.0)
        draft_step_ms = self._draft_step_ms
        if draft_bs is not None and draft_ctx_bucket is not None:
            draft_step_ms = self._draft_step_ms_by_shape.get(
                (int(draft_bs), str(draft_ctx_bucket)), 0.0
            )
        return SlackSnapshot(
            target_phase=phase,
            target_phase_ms=phase_ms,
            draft_step_ms=draft_step_ms,
            predicted_slack_us=max(phase_ms * 1000.0, 0.0),
            samples=self._samples,
        )
