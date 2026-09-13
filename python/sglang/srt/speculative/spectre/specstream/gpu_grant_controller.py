from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from sglang.srt.speculative.spectre.specstream.resource_profile import (
    ResourceProfile,
    ResourceProfileEntry,
)


class GrantState(str, Enum):
    TARGET_EXCLUSIVE = "TARGET_EXCLUSIVE"
    SLACK_FILL = "SLACK_FILL"
    DRAFT_CATCHUP = "DRAFT_CATCHUP"


@dataclass(frozen=True)
class GrantDecision:
    state: GrantState
    reason: str
    tpc_low: int = 0
    tpc_high: int = 0
    deadline_us: int | None = None
    profile_entry: ResourceProfileEntry | None = None
    draft_step_ms: float | None = None

    @property
    def allows_draft(self) -> bool:
        return self.state is not GrantState.TARGET_EXCLUSIVE


class GpuGrantController:
    """Target-priority spatial/temporal gate backed only by measurements.

    ``calibration_tpcs`` is retained as a CLI-compatible name, but it only
    fixes the Draft TPC range.  It does not bypass slack admission, a live
    slowdown latch, or the launch deadline.  A profile-free fixed-TPC overlap
    is fail-closed until both a Draft step and a Target-only baseline have been
    observed online.
    """

    def __init__(
        self,
        profile: ResourceProfile | None,
        *,
        target_slowdown_budget: float = 0.05,
        guard_us: float = 200.0,
        calibration_tpcs: int = 0,
        catchup_tpcs: int = 0,
        calibration_allow_overlap: bool = False,
        fixed_baseline_alpha: float = 0.2,
        fixed_baseline_min_samples: int = 2,
        catchup_token_quantum: int = 1,
    ) -> None:
        if not 0.0 <= target_slowdown_budget <= 1.0:
            raise ValueError("target_slowdown_budget must be in [0, 1]")
        if guard_us < 0:
            raise ValueError("guard_us cannot be negative")
        if calibration_tpcs < 0:
            raise ValueError("calibration_tpcs cannot be negative")
        if catchup_tpcs < 0:
            raise ValueError("catchup_tpcs cannot be negative")
        if catchup_tpcs and catchup_tpcs < calibration_tpcs:
            raise ValueError("catchup_tpcs cannot be smaller than calibration_tpcs")
        if not 0.0 < fixed_baseline_alpha <= 1.0:
            raise ValueError("fixed_baseline_alpha must be in (0, 1]")
        if fixed_baseline_min_samples < 1:
            raise ValueError("fixed_baseline_min_samples must be positive")
        if not 1 <= catchup_token_quantum <= 8:
            raise ValueError("catchup_token_quantum must be between 1 and 8")
        if profile is None and calibration_tpcs < 1:
            raise ValueError("a resource profile is required outside fixed-TPC mode")
        self.profile = profile
        self.target_slowdown_budget = float(target_slowdown_budget)
        self.guard_us = float(guard_us)
        self.calibration_tpcs = int(calibration_tpcs)
        self.catchup_tpcs = int(catchup_tpcs or calibration_tpcs)
        self.calibration_allow_overlap = bool(calibration_allow_overlap)
        self.fixed_baseline_alpha = float(fixed_baseline_alpha)
        self.fixed_baseline_min_samples = int(fixed_baseline_min_samples)
        self.catchup_token_quantum = int(catchup_token_quantum)
        self._force_exclusive = False
        self._force_reason = ""
        self._force_shape: str | None = None
        self._recovery_target_only_samples = 0
        self._fixed_target_baselines: dict[str, tuple[float, int]] = {}

    @property
    def fixed_tpc_mode(self) -> bool:
        return self.calibration_tpcs > 0

    def record_overlap_slowdown(
        self, slowdown: float, *, target_shape: str | None = None
    ) -> None:
        if float(slowdown) > self.target_slowdown_budget:
            self._force_exclusive = True
            self._force_reason = "observed_target_slowdown_over_budget"
            self._force_shape = target_shape
            self._recovery_target_only_samples = 0

    def clear_slowdown_latch(self) -> None:
        self._force_exclusive = False
        self._force_reason = ""
        self._force_shape = None
        self._recovery_target_only_samples = 0

    def record_target_only(self, target_shape: str) -> None:
        """A bounded cooldown restores a measured overlap probe after 3 rounds."""
        if self._force_exclusive and (
            self._force_shape is None or self._force_shape == str(target_shape)
        ):
            self._recovery_target_only_samples += 1
            if self._recovery_target_only_samples >= 3:
                self.clear_slowdown_latch()

    def record_fixed_target_forward(
        self,
        *,
        target_shape: str,
        elapsed_ms: float,
        possible_overlap: bool,
        confirmed_overlap: bool,
    ) -> None:
        """Learn Target-only timing or check a fixed-TPC overlap against it.

        Baselines are shape-specific and are updated only by rounds in which
        no ``SLACK_FILL`` grant was issued.  A sent-but-unconfirmed grant is
        excluded from both baseline learning and slowdown comparison because
        a delayed/lost ACK cannot prove whether its CUDA work launched.
        Upward changes are deliberately
        incorporated gradually, while a lower observation is accepted
        immediately.  This keeps the slowdown guard conservative instead of
        teaching it that an interfered Target is the new baseline.
        """

        if not self.fixed_tpc_mode:
            return
        elapsed_ms = float(elapsed_ms)
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            return
        shape = str(target_shape)
        baseline_ms, samples = self._fixed_target_baselines.get(shape, (0.0, 0))
        if confirmed_overlap:
            if samples < self.fixed_baseline_min_samples or baseline_ms <= 0:
                # This should be unreachable because decide() fails closed,
                # but latch if lifecycle wiring ever violates that invariant.
                self._force_exclusive = True
                self._force_reason = "fixed_tpc_missing_target_baseline"
                return
            slowdown = max(elapsed_ms / baseline_ms - 1.0, 0.0)
            self.record_overlap_slowdown(slowdown, target_shape=shape)
            return

        if possible_overlap:
            return

        self.record_target_only(shape)

        if baseline_ms <= 0:
            updated_ms = elapsed_ms
        else:
            ema_ms = (
                1.0 - self.fixed_baseline_alpha
            ) * baseline_ms + self.fixed_baseline_alpha * elapsed_ms
            updated_ms = ema_ms
        self._fixed_target_baselines[shape] = (updated_ms, samples + 1)

    def fixed_target_baseline_ms(self, target_shape: str) -> float | None:
        baseline_ms, samples = self._fixed_target_baselines.get(
            str(target_shape), (0.0, 0)
        )
        if samples < self.fixed_baseline_min_samples or baseline_ms <= 0:
            return None
        return baseline_ms

    def catchup_token_budget(
        self,
        *,
        remaining_tokens: int,
        draft_step_ms: float,
        deadline_us: int | None,
        now_us: int,
    ) -> int:
        """Bound one catchup lease by observed step cost and its wait deadline.

        Initial/unmeasured work remains one token. This function is never
        called to enlarge a physical History-H2D SLACK_FILL authorization.
        """
        cap = min(max(int(remaining_tokens), 0), self.catchup_token_quantum)
        if cap == 0:
            return 0
        if deadline_us is not None and deadline_us <= now_us:
            return 0
        if not math.isfinite(draft_step_ms) or draft_step_ms <= 0:
            return 1
        if deadline_us is None:
            return 1
        # Allow a single launch while its launch deadline remains valid, as
        # in the v1 protocol. Additional launches require a measured budget.
        available_us = max(deadline_us - now_us - self.guard_us, 0.0)
        measured_budget = max(1, int(available_us // (draft_step_ms * 1000.0)))
        return min(cap, measured_budget)

    def decide(
        self,
        *,
        target_shape: str,
        draft_bs: int,
        draft_ctx_bucket: str,
        predicted_slack_us: float,
        draft_step_ms: float = 0.0,
        slack_source: str = "target_forward",
        target_waiting: bool = False,
        deadline_us: int | None = None,
    ) -> GrantDecision:
        draft_step_ms = float(draft_step_ms)
        if not math.isfinite(draft_step_ms) or draft_step_ms <= 0:
            draft_step_ms = 0.0
        # Target wait is outside the overlap critical path.  It must remain
        # able to make Draft progress even after the overlap slowdown latch is
        # set, otherwise Target and Draft can deadlock waiting on each other.
        if self.fixed_tpc_mode:
            if target_waiting:
                return GrantDecision(
                    GrantState.DRAFT_CATCHUP,
                    "fixed_tpc_target_wait",
                    0,
                    self.catchup_tpcs,
                    deadline_us,
                    draft_step_ms=(draft_step_ms if draft_step_ms > 0 else None),
                )
        elif target_waiting:
            assert self.profile is not None
            entry = self.profile.select_catchup(
                target_shape=target_shape,
                draft_bs=draft_bs,
                draft_ctx_bucket=draft_ctx_bucket,
            )
            if entry is None:
                return GrantDecision(
                    GrantState.TARGET_EXCLUSIVE, "no_calibrated_catchup_entry"
                )
            return GrantDecision(
                GrantState.DRAFT_CATCHUP,
                "target_waiting_for_draft",
                0,
                entry.draft_tpcs,
                deadline_us,
                entry,
                entry.draft_step_ms,
            )

        if self._force_exclusive and (
            self._force_shape is None or self._force_shape == str(target_shape)
        ):
            return GrantDecision(GrantState.TARGET_EXCLUSIVE, self._force_reason)
        predicted_slack_us = float(predicted_slack_us)
        if not math.isfinite(predicted_slack_us) or predicted_slack_us <= 0:
            return GrantDecision(GrantState.TARGET_EXCLUSIVE, "no_predicted_slack")

        if self.fixed_tpc_mode:
            if not self.calibration_allow_overlap:
                return GrantDecision(GrantState.TARGET_EXCLUSIVE, "fixed_tpc_wait_only")
            if self.fixed_target_baseline_ms(target_shape) is None:
                return GrantDecision(
                    GrantState.TARGET_EXCLUSIVE,
                    "fixed_tpc_target_baseline_warmup",
                )
            if draft_step_ms <= 0:
                return GrantDecision(
                    GrantState.TARGET_EXCLUSIVE,
                    "fixed_tpc_draft_step_warmup",
                )
            required_us = draft_step_ms * 1000.0 + self.guard_us
            if required_us > predicted_slack_us:
                return GrantDecision(
                    GrantState.TARGET_EXCLUSIVE,
                    "fixed_tpc_window_too_short",
                )
            return GrantDecision(
                GrantState.SLACK_FILL,
                (
                    "fixed_tpc_safe_pcie_slack"
                    if slack_source == "history_h2d"
                    else "fixed_tpc_safe_slack"
                ),
                0,
                self.calibration_tpcs,
                deadline_us,
                draft_step_ms=draft_step_ms,
            )

        assert self.profile is not None
        entry = self.profile.select_safe(
            target_shape=target_shape,
            draft_bs=draft_bs,
            draft_ctx_bucket=draft_ctx_bucket,
            slowdown_budget=self.target_slowdown_budget,
            slack_us=predicted_slack_us,
            guard_us=self.guard_us,
            slack_source=slack_source,
        )
        if entry is None:
            return GrantDecision(
                GrantState.TARGET_EXCLUSIVE, "no_measured_safe_profile_entry"
            )
        return GrantDecision(
            GrantState.SLACK_FILL,
            (
                "measured_safe_pcie_slack"
                if slack_source == "history_h2d"
                else "measured_safe_slack"
            ),
            0,
            entry.draft_tpcs,
            deadline_us,
            entry,
            entry.draft_step_ms,
        )
