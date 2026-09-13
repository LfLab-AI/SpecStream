from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from sglang.srt.speculative.spectre.specstream.tp_straggler_monitor import (
    TPStragglerSnapshot,
)


COEXEC = "COEXEC"
SERIALIZE = "SERIALIZE"
FALLBACK = "FALLBACK"  # compatibility for external imports


@dataclass(frozen=True)
class MultiGPUConstraint:
    """Target protection changes overlap permission, never verification width."""

    max_q: int
    force_ordinary: bool = False
    coexec_mode: str = COEXEC
    reason: str = "minimum_estimated_cost"
    fallback: bool = False


@dataclass
class _ProtectionState:
    observation: tuple[int, int] = (-1, -1)
    violations: int = 0
    cooldown_remaining: int = 0


class MultiGPUTPPolicy:
    """Suspend Draft overlap on attributed regressions, then probe recovery.

    Serial multi-query rounds supply the counterfactual and useful output
    during warmup/cooldown. Repeated calls with one snapshot cannot exhaust
    cooldown, and a workload shape change cannot inherit another shape's ban.
    """

    def __init__(
        self,
        *,
        rank_skew_budget_ms: float = 1.0,
        target_slowdown_budget: float = 0.10,
        violation_samples: int = 2,
        cooldown_samples: int = 3,
        max_shapes: int = 128,
    ) -> None:
        if rank_skew_budget_ms < 0:
            raise ValueError("rank_skew_budget_ms cannot be negative")
        if target_slowdown_budget < 0:
            raise ValueError("target_slowdown_budget cannot be negative")
        if min(violation_samples, cooldown_samples, max_shapes) < 1:
            raise ValueError("protection sample counts must be positive")
        self.rank_skew_budget_ms = float(rank_skew_budget_ms)
        self.target_slowdown_budget = float(target_slowdown_budget)
        self.violation_samples = int(violation_samples)
        self.cooldown_samples = int(cooldown_samples)
        self.max_shapes = int(max_shapes)
        self._states: OrderedDict[tuple, _ProtectionState] = OrderedDict()

    @staticmethod
    def _serial(max_q: int, reason: str) -> MultiGPUConstraint:
        return MultiGPUConstraint(max_q, True, SERIALIZE, reason)

    def constrain(
        self,
        *,
        max_q: int,
        snapshot: TPStragglerSnapshot,
    ) -> MultiGPUConstraint:
        if not snapshot.baseline_ready:
            return self._serial(max_q, "tp_baseline_warmup")
        if snapshot.overlap_active is None:
            return self._serial(max_q, "tp_overlap_unattributed")
        key = tuple(snapshot.shape_key)
        state = self._states.setdefault(key, _ProtectionState())
        self._states.move_to_end(key)
        while len(self._states) > self.max_shapes:
            self._states.popitem(last=False)
        observation = (snapshot.round_id, snapshot.samples)
        if observation != state.observation:
            state.observation = observation
            if snapshot.overlap_active:
                violation = (
                    snapshot.excess_rank_skew_ms > self.rank_skew_budget_ms
                    or snapshot.target_slowdown > self.target_slowdown_budget
                )
                state.violations = state.violations + 1 if violation else 0
                if state.violations >= self.violation_samples:
                    state.cooldown_remaining = self.cooldown_samples
                    state.violations = 0
            else:
                state.violations = 0
                state.cooldown_remaining = max(0, state.cooldown_remaining - 1)
        if state.cooldown_remaining:
            return self._serial(max_q, "tp_overlap_cooldown")
        return MultiGPUConstraint(max_q)
