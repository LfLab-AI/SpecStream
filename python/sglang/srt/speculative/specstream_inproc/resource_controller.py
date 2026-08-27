from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CoexecutionMode(str, Enum):
    SERIAL = "serial"
    AHEAD_FREE = "ahead-free"
    AHEAD_THROTTLED = "ahead-throttled"


@dataclass(frozen=True)
class CoexecutionAction:
    mode: CoexecutionMode
    ahead_depth: int
    draft_tpcs: int | None = None


class InProcessResourceController:
    """Low-overhead controller for Phase-A in-process co-execution.

    TPC throttling is deliberately not activated here.  The enum and action
    carry the future control dimension, but Phase A selects only SERIAL or
    AHEAD_FREE until the no-TPC implementation passes its Go/No-Go gate.
    """

    def __init__(
        self,
        *,
        configured_mode: str,
        ahead_depth: int,
        min_reuse_ratio: float,
        target_slowdown_budget: float,
        ema_alpha: float = 0.2,
        warmup_rounds: int = 16,
        guard_ms: float = 0.1,
    ) -> None:
        if ahead_depth not in (1, 2, 4):
            raise ValueError("ahead_depth must be one of 1, 2, or 4")
        if not 0.0 <= min_reuse_ratio <= 1.0:
            raise ValueError("min_reuse_ratio must be in [0, 1]")
        if not 0.0 <= target_slowdown_budget <= 1.0:
            raise ValueError("target_slowdown_budget must be in [0, 1]")
        if configured_mode not in ("serial", "ahead-free", "auto"):
            raise ValueError("configured_mode must be serial, ahead-free, or auto")

        self.configured_mode = configured_mode
        self.ahead_depth = ahead_depth
        self.min_reuse_ratio = min_reuse_ratio
        self.target_slowdown_budget = target_slowdown_budget
        self.ema_alpha = ema_alpha
        self.warmup_rounds = warmup_rounds
        self.guard_ms = max(0.0, guard_ms)
        self.rounds = 0
        self.decisions = 0
        self.adaptive_depth = ahead_depth
        self.reuse_ema: float | None = None
        self.slowdown_ema: float | None = None
        self.verify_ms_ema: float | None = None
        self.ahead_token_ms_ema: float | None = None
        self.repair_ms_ema: float | None = None

    def choose(self) -> CoexecutionAction:
        self.decisions += 1
        if self.configured_mode == "serial":
            return CoexecutionAction(CoexecutionMode.SERIAL, 0)
        if self.configured_mode == "ahead-free":
            return CoexecutionAction(CoexecutionMode.AHEAD_FREE, self.ahead_depth)

        if self.rounds < self.warmup_rounds:
            return CoexecutionAction(CoexecutionMode.AHEAD_FREE, self.ahead_depth)
        if (self.reuse_ema is not None and self.reuse_ema < self.min_reuse_ratio) or (
            self.slowdown_ema is not None
            and self.slowdown_ema > self.target_slowdown_budget
        ):
            # Re-probe occasionally so AUTO can recover when the acceptance
            # distribution changes after a low-reuse interval.
            if self.decisions % 64 == 0:
                return CoexecutionAction(
                    CoexecutionMode.AHEAD_FREE, self.adaptive_depth
                )
            return CoexecutionAction(CoexecutionMode.SERIAL, 0)
        return CoexecutionAction(CoexecutionMode.AHEAD_FREE, self.adaptive_depth)

    def observe(
        self,
        *,
        reuse_ratio: float,
        target_slowdown: float | None,
        ahead_depth: int | None = None,
        verify_ms: float | None = None,
        ahead_ms: float | None = None,
        repair_ms: float | None = None,
    ) -> None:
        self.rounds += 1
        self.reuse_ema = self._ema(self.reuse_ema, reuse_ratio)
        if target_slowdown is not None:
            self.slowdown_ema = self._ema(self.slowdown_ema, target_slowdown)
        if verify_ms is not None:
            self.verify_ms_ema = self._ema(self.verify_ms_ema, verify_ms)
        if repair_ms is not None:
            self.repair_ms_ema = self._ema(self.repair_ms_ema, repair_ms)
        if ahead_ms is not None and ahead_depth:
            # The optimistic bonus is an anchor in addition to h candidate
            # steps, so charge its cost when estimating the available window.
            token_ms = ahead_ms / (ahead_depth + 1)
            self.ahead_token_ms_ema = self._ema(self.ahead_token_ms_ema, token_ms)
        self._update_adaptive_depth()

    def _update_adaptive_depth(self) -> None:
        if self.verify_ms_ema is None or self.ahead_token_ms_ema in (None, 0.0):
            return
        usable_ms = max(0.0, self.verify_ms_ema - self.guard_ms)
        # One time unit is reserved for the bonus anchor.  Excess repair cost
        # shrinks the next probe rather than increasing wasted speculative work.
        raw_depth = max(0, int(usable_ms / self.ahead_token_ms_ema) - 1)
        if self.repair_ms_ema and self.reuse_ema is not None:
            raw_depth = int(raw_depth * self.reuse_ema)
        choices = [depth for depth in (1, 2, 4) if depth <= self.ahead_depth]
        feasible = [depth for depth in choices if depth <= raw_depth]
        self.adaptive_depth = max(feasible, default=1)

    def observe_repair(self, repair_ms: float) -> None:
        self.repair_ms_ema = self._ema(self.repair_ms_ema, repair_ms)
        self._update_adaptive_depth()

    def _ema(self, old: float | None, new: float) -> float:
        if old is None:
            return float(new)
        return self.ema_alpha * float(new) + (1.0 - self.ema_alpha) * old
