from __future__ import annotations

from dataclasses import dataclass

from sglang.srt.speculative.spectre.specstream.cost_model import (
    SpecStreamCostProfile,
)
from sglang.srt.speculative.spectre.specstream.draft_load_tracker import (
    DraftLoadSnapshot,
)


COEXEC = "COEXEC"
THROTTLE = "THROTTLE"


@dataclass(frozen=True)
class SingleGPUConstraint:
    """The q/mode limit produced only by the single-GPU policy."""

    max_q: int
    coexec_mode: str = COEXEC
    reason: str = "minimum_estimated_cost"


class SingleGPUCoexecPolicy:
    """Step 2 policy: protect Target while Draft shares the same physical GPU.

    This module deliberately knows nothing about TP ranks or multi-GPU
    collectives.  It only looks at Drafter pressure and the local Target phase.
    """

    def __init__(
        self,
        *,
        draft_pressure_ratio: float = 0.80,
        draft_timeout_rate_threshold: float = 0.10,
        draft_pending_high_watermark: int = 16,
        compute_ratio_threshold: float = 0.90,
    ) -> None:
        if not 0.0 < draft_pressure_ratio <= 1.0:
            raise ValueError("draft_pressure_ratio must be in (0, 1]")
        if not 0.0 <= draft_timeout_rate_threshold <= 1.0:
            raise ValueError("draft_timeout_rate_threshold must be in [0, 1]")
        if draft_pending_high_watermark < 1:
            raise ValueError("draft_pending_high_watermark must be positive")
        if not 0.0 < compute_ratio_threshold <= 1.0:
            raise ValueError("compute_ratio_threshold must be in (0, 1]")
        self.draft_pressure_ratio = float(draft_pressure_ratio)
        self.draft_timeout_rate_threshold = float(draft_timeout_rate_threshold)
        self.draft_pending_high_watermark = int(draft_pending_high_watermark)
        self.compute_ratio_threshold = float(compute_ratio_threshold)

    def constrain(
        self,
        *,
        max_q: int,
        profile: SpecStreamCostProfile,
        draft_load: DraftLoadSnapshot,
    ) -> SingleGPUConstraint:
        pressure_p95 = draft_load.pressure_p95
        pending_p95 = draft_load.pending_p95
        timeout_rate = draft_load.timeout_rate
        reject_rate = draft_load.reject_rate

        if (
            (timeout_rate > 0.0 and timeout_rate >= self.draft_timeout_rate_threshold)
            or (reject_rate > 0.0 and reject_rate >= self.draft_timeout_rate_threshold)
            or pressure_p95 >= min(0.95, self.draft_pressure_ratio + 0.10)
            or (
                pending_p95 >= self.draft_pending_high_watermark
                and pressure_p95 >= 0.50
            )
        ):
            return SingleGPUConstraint(2, THROTTLE, "draft_pressure_limited")

        if (
            timeout_rate > 0.0
            or pressure_p95 >= self.draft_pressure_ratio
            or (
                pending_p95 >= max(2, self.draft_pending_high_watermark // 2)
                and pressure_p95 >= 0.50
            )
        ):
            return SingleGPUConstraint(
                max(2, (max_q + 1) // 2),
                THROTTLE,
                "draft_pressure_limited",
            )

        if (
            profile.target_compute_ratio >= self.compute_ratio_threshold
            and profile.exposed_copy_ms <= max(0.05, 0.1 * profile.target_other_ms)
        ):
            return SingleGPUConstraint(
                max(2, (max_q + 1) // 2),
                THROTTLE,
                "target_compute_heavy",
            )

        return SingleGPUConstraint(max_q)
