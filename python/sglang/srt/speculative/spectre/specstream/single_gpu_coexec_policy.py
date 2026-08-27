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
    """Compatibility wrapper for the retired MPS-centric Step-2 policy.

    Kept so old launch configurations and imports do not fail.  It must not
    convert Target compute ratio or Draft RTT into GPU execution permission;
    the measured-profile ``GpuGrantController`` owns that decision now.
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
        del profile, draft_load
        return SingleGPUConstraint(int(max_q), COEXEC, "deprecated_mps_policy_disabled")
