from __future__ import annotations

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

    @property
    def allows_draft(self) -> bool:
        return self.state is not GrantState.TARGET_EXCLUSIVE


class GpuGrantController:
    """Target-priority spatial/temporal gate backed only by measured entries."""

    def __init__(
        self,
        profile: ResourceProfile | None,
        *,
        target_slowdown_budget: float = 0.05,
        guard_us: float = 200.0,
        calibration_tpcs: int = 0,
        calibration_allow_overlap: bool = False,
    ) -> None:
        if not 0.0 <= target_slowdown_budget <= 1.0:
            raise ValueError("target_slowdown_budget must be in [0, 1]")
        if guard_us < 0:
            raise ValueError("guard_us cannot be negative")
        if calibration_tpcs < 0:
            raise ValueError("calibration_tpcs cannot be negative")
        if profile is None and calibration_tpcs < 1:
            raise ValueError("a resource profile is required outside calibration")
        self.profile = profile
        self.target_slowdown_budget = float(target_slowdown_budget)
        self.guard_us = float(guard_us)
        self.calibration_tpcs = int(calibration_tpcs)
        self.calibration_allow_overlap = bool(calibration_allow_overlap)
        self._force_exclusive = False
        self._force_reason = ""

    def record_overlap_slowdown(self, slowdown: float) -> None:
        if float(slowdown) > self.target_slowdown_budget:
            self._force_exclusive = True
            self._force_reason = "observed_target_slowdown_over_budget"

    def clear_slowdown_latch(self) -> None:
        self._force_exclusive = False
        self._force_reason = ""

    def decide(
        self,
        *,
        target_shape: str,
        draft_bs: int,
        draft_ctx_bucket: str,
        predicted_slack_us: float,
        slack_source: str = "target_forward",
        target_waiting: bool = False,
        deadline_us: int | None = None,
    ) -> GrantDecision:
        if self.calibration_tpcs > 0:
            if target_waiting:
                return GrantDecision(
                    GrantState.DRAFT_CATCHUP,
                    "calibration_fixed_tpc_target_wait",
                    0,
                    self.calibration_tpcs,
                    deadline_us,
                )
            if self.calibration_allow_overlap:
                return GrantDecision(
                    GrantState.SLACK_FILL,
                    "calibration_fixed_tpc_overlap",
                    0,
                    self.calibration_tpcs,
                    deadline_us,
                )
            return GrantDecision(GrantState.TARGET_EXCLUSIVE, "calibration_wait_only")

        if target_waiting:
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
            )

        if self._force_exclusive:
            return GrantDecision(GrantState.TARGET_EXCLUSIVE, self._force_reason)
        if predicted_slack_us <= 0:
            return GrantDecision(GrantState.TARGET_EXCLUSIVE, "no_predicted_slack")

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
        )
