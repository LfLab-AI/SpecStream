from __future__ import annotations

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
        self._samples = 0
        self._last_phase = "unknown"

    def _update(self, previous: float, value: float) -> float:
        return (
            value
            if previous <= 0
            else (1.0 - self.alpha) * previous + self.alpha * value
        )

    def record_target_phase(self, phase: str, elapsed_ms: float) -> None:
        elapsed_ms = max(float(elapsed_ms), 0.0)
        self._last_phase = str(phase or "unknown")
        self._phase_ms[self._last_phase] = self._update(
            self._phase_ms.get(self._last_phase, 0.0), elapsed_ms
        )
        self._samples += 1

    def record_draft_step(self, elapsed_ms: float) -> None:
        self._draft_step_ms = self._update(
            self._draft_step_ms, max(float(elapsed_ms), 0.0)
        )

    def snapshot(self, phase: str | None = None) -> SlackSnapshot:
        phase = str(phase or self._last_phase)
        phase_ms = self._phase_ms.get(phase, 0.0)
        return SlackSnapshot(
            target_phase=phase,
            target_phase_ms=phase_ms,
            draft_step_ms=self._draft_step_ms,
            predicted_slack_us=max(phase_ms * 1000.0, 0.0),
            samples=self._samples,
        )
