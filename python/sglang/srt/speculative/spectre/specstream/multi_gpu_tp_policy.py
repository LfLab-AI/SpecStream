from __future__ import annotations

from dataclasses import dataclass

from sglang.srt.speculative.spectre.specstream.tp_straggler_monitor import (
    TPStragglerSnapshot,
)


COEXEC = "COEXEC"
SERIALIZE = "SERIALIZE"
FALLBACK = "FALLBACK"


@dataclass(frozen=True)
class MultiGPUConstraint:
    """The q/mode limit produced only by the multi-GPU TP policy."""

    max_q: int
    force_ordinary: bool = False
    coexec_mode: str = COEXEC
    reason: str = "minimum_estimated_cost"
    fallback: bool = False


class MultiGPUTPPolicy:
    """Step 3 policy: prevent the colocated TP rank from slowing all ranks.

    This module deliberately knows nothing about MPS percentages or Drafter
    queue pressure.  It only consumes Target TP rank timing observations.
    """

    def __init__(
        self,
        *,
        rank_skew_budget_ms: float = 1.0,
        target_slowdown_budget: float = 0.10,
    ) -> None:
        if rank_skew_budget_ms < 0:
            raise ValueError("rank_skew_budget_ms cannot be negative")
        if target_slowdown_budget < 0:
            raise ValueError("target_slowdown_budget cannot be negative")
        self.rank_skew_budget_ms = float(rank_skew_budget_ms)
        self.target_slowdown_budget = float(target_slowdown_budget)

    def constrain(
        self,
        *,
        max_q: int,
        snapshot: TPStragglerSnapshot,
    ) -> MultiGPUConstraint:
        if not snapshot.samples:
            return MultiGPUConstraint(max_q)

        severe = (
            snapshot.rank_skew_ms > 2.0 * self.rank_skew_budget_ms
            or snapshot.target_slowdown > 2.0 * self.target_slowdown_budget
        )
        if severe:
            return MultiGPUConstraint(
                1,
                force_ordinary=True,
                coexec_mode=FALLBACK,
                reason="tp_straggler_fallback",
                fallback=True,
            )

        moderate = (
            snapshot.rank_skew_ms > self.rank_skew_budget_ms
            or snapshot.target_slowdown > self.target_slowdown_budget
        )
        if moderate:
            return MultiGPUConstraint(
                min(max_q, 2),
                force_ordinary=True,
                coexec_mode=SERIALIZE,
                reason="tp_straggler_throttle",
            )

        return MultiGPUConstraint(max_q)
