from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from sglang.srt.speculative.spectre.specstream.acceptance_tracker import (
    AcceptanceSnapshot,
)
from sglang.srt.speculative.spectre.specstream.cost_model import (
    SpecStreamBatchState,
    SpecStreamCostProfile,
    estimate_candidate_cost,
)


@dataclass(frozen=True)
class SpecStreamDecision:
    q: int
    mode: str
    reason: str
    estimated_cost: float


class IOAwareController:
    """Batch-level joint ``(mode, q)`` controller with load feedback.

    History I/O and acceptance determine the steady-state cost choice.  Remote
    Drafter observations add a safety envelope: a timeout causes a short AR
    backoff, while sustained near-timeout latency limits the largest candidate
    until successful probes show that the Drafter has recovered.
    """

    def __init__(
        self,
        q_candidates: Iterable[int],
        switch_threshold: float = 0.08,
    ) -> None:
        self.q_candidates = tuple(sorted(set(int(q) for q in q_candidates)))
        if not self.q_candidates or any(q < 1 for q in self.q_candidates):
            raise ValueError("q_candidates must contain positive integers")
        if not 0.0 <= switch_threshold < 1.0:
            raise ValueError("switch_threshold must be in [0, 1)")
        self.switch_threshold = switch_threshold
        self._last = SpecStreamDecision(1, "parallel", "cold_start", float("inf"))
        self._draft_pressure_samples: deque[float] = deque(maxlen=32)
        self._draft_timeout_samples: deque[int] = deque(maxlen=32)
        self._draft_missing_ratio_samples: deque[float] = deque(maxlen=32)
        self._draft_pending_samples: deque[int] = deque(maxlen=32)
        self._draft_backoff_rounds = 0

    def record_draft_result(
        self,
        *,
        elapsed_ms: float,
        timeout_ms: float,
        missing_count: int,
        total_count: int,
    ) -> None:
        """Feed observed Drafter queue/network pressure into future choices."""
        elapsed_ms = max(float(elapsed_ms), 0.0)
        timeout_ms = max(float(timeout_ms), 1e-6)
        missing_count = max(int(missing_count), 0)
        total_count = max(int(total_count), 0)
        pressure = elapsed_ms / timeout_ms
        missing_ratio = missing_count / max(total_count, 1)
        timed_out = int(missing_count > 0)
        self._draft_pressure_samples.append(pressure)
        self._draft_timeout_samples.append(timed_out)
        self._draft_missing_ratio_samples.append(missing_ratio)
        self._draft_pending_samples.append(total_count)
        if timed_out:
            # Batch-uniform verification makes one request-level miss a batch
            # fallback.  Four AR rounds drain work without permanently
            # disabling the Drafter; larger missing fractions back off longer.
            penalty = 4 + min(12, int(round(12 * missing_ratio)))
            self._draft_backoff_rounds = max(self._draft_backoff_rounds, penalty)

    @property
    def draft_timeout_rate(self) -> float:
        if not self._draft_timeout_samples:
            return 0.0
        return sum(self._draft_timeout_samples) / len(self._draft_timeout_samples)

    @property
    def draft_pressure_p95(self) -> float:
        if not self._draft_pressure_samples:
            return 0.0
        ordered = sorted(self._draft_pressure_samples)
        index = max(0, int(0.95 * len(ordered) + 0.999999) - 1)
        return ordered[min(index, len(ordered) - 1)]

    @property
    def draft_pending_p95(self) -> int:
        if not self._draft_pending_samples:
            return 0
        ordered = sorted(self._draft_pending_samples)
        index = max(0, int(0.95 * len(ordered) + 0.999999) - 1)
        return ordered[min(index, len(ordered) - 1)]

    def choose(
        self,
        batch_state: SpecStreamBatchState,
        profile: SpecStreamCostProfile,
        acceptance: AcceptanceSnapshot,
        *,
        allow_ordinary: bool = True,
    ) -> SpecStreamDecision:
        if batch_state.rejected or batch_state.high_overhead:
            self._last = SpecStreamDecision(1, "ordinary", "safety_fallback", 0.0)
            return self._last

        if self._draft_backoff_rounds > 0:
            self._draft_backoff_rounds -= 1
            self._last = SpecStreamDecision(
                1, "ordinary", "draft_timeout_backoff", 0.0
            )
            return self._last

        max_q = max(self.q_candidates)
        pressure_p95 = self.draft_pressure_p95
        pending_p95 = self.draft_pending_p95
        timeout_rate = self.draft_timeout_rate
        if timeout_rate >= 0.25 or pressure_p95 >= 0.90:
            max_q = 2
        elif pending_p95 >= 16 and pressure_p95 >= 0.50:
            max_q = 2
        elif timeout_rate > 0.0 or pressure_p95 >= 0.70:
            max_q = max(2, (max_q + 1) // 2)
        elif pending_p95 >= 8 and pressure_p95 >= 0.50:
            max_q = max(2, (max_q + 1) // 2)
        eligible_q = tuple(q for q in self.q_candidates if q <= max_q)
        if not eligible_q:
            eligible_q = (min(self.q_candidates),)
        pressure_limited = max(eligible_q) < max(self.q_candidates)

        candidates: list[SpecStreamDecision] = []
        for q in eligible_q:
            cost = estimate_candidate_cost(q, batch_state, profile, acceptance)
            reason = (
                "draft_pressure_limited"
                if pressure_limited
                else "minimum_estimated_cost"
            )
            candidates.append(
                SpecStreamDecision(
                    q,
                    "parallel",
                    reason,
                    cost.parallel_ms_per_useful_token,
                )
            )
            if allow_ordinary:
                candidates.append(
                    SpecStreamDecision(
                        q,
                        "ordinary",
                        reason,
                        cost.ordinary_ms_per_useful_token,
                    )
                )

        best = min(candidates, key=lambda item: item.estimated_cost)
        previous = next(
            (
                item
                for item in candidates
                if item.q == self._last.q and item.mode == self._last.mode
            ),
            None,
        )
        if previous is not None and previous.estimated_cost > 0:
            relative_gain = (
                previous.estimated_cost - best.estimated_cost
            ) / previous.estimated_cost
            if relative_gain < self.switch_threshold:
                best = SpecStreamDecision(
                    previous.q,
                    previous.mode,
                    "hysteresis_hold",
                    previous.estimated_cost,
                )
        self._last = best
        return best
