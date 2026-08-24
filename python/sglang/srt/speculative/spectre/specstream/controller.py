from __future__ import annotations

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
from sglang.srt.speculative.spectre.specstream.draft_load_tracker import (
    DraftLoadSnapshot,
    DraftLoadTracker,
)
from sglang.srt.speculative.spectre.specstream.multi_gpu_tp_policy import (
    MultiGPUTPPolicy,
)
from sglang.srt.speculative.spectre.specstream.single_gpu_coexec_policy import (
    SingleGPUCoexecPolicy,
)
from sglang.srt.speculative.spectre.specstream.tp_straggler_monitor import (
    TPStragglerSnapshot,
)


COEXEC = "COEXEC"
THROTTLE = "THROTTLE"
SERIALIZE = "SERIALIZE"
FALLBACK = "FALLBACK"


@dataclass(frozen=True)
class SpecStreamDecision:
    q: int
    mode: str
    reason: str
    estimated_cost: float
    coexec_mode: str = COEXEC
    rank_skew_ms: float = 0.0
    target_slowdown: float = 0.0


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
        *,
        draft_pressure_ratio: float = 0.80,
        draft_timeout_rate_threshold: float = 0.10,
        draft_pending_high_watermark: int = 16,
        phase_gating_enabled: bool = False,
        compute_ratio_threshold: float = 0.90,
        tp_straggler_budget_ms: float = 1.0,
        target_slowdown_budget: float = 0.10,
        single_gpu_policy: SingleGPUCoexecPolicy | None = None,
        multi_gpu_policy: MultiGPUTPPolicy | None = None,
    ) -> None:
        self.q_candidates = tuple(sorted(set(int(q) for q in q_candidates)))
        if not self.q_candidates or any(q < 1 for q in self.q_candidates):
            raise ValueError("q_candidates must contain positive integers")
        if not 0.0 <= switch_threshold < 1.0:
            raise ValueError("switch_threshold must be in [0, 1)")
        if not 0.0 < draft_pressure_ratio <= 1.0:
            raise ValueError("draft_pressure_ratio must be in (0, 1]")
        if not 0.0 <= draft_timeout_rate_threshold <= 1.0:
            raise ValueError("draft_timeout_rate_threshold must be in [0, 1]")
        if draft_pending_high_watermark < 1:
            raise ValueError("draft_pending_high_watermark must be positive")
        if not 0.0 < compute_ratio_threshold <= 1.0:
            raise ValueError("compute_ratio_threshold must be in (0, 1]")
        if tp_straggler_budget_ms < 0:
            raise ValueError("tp_straggler_budget_ms cannot be negative")
        if target_slowdown_budget < 0:
            raise ValueError("target_slowdown_budget cannot be negative")
        self.switch_threshold = switch_threshold
        self.draft_pressure_ratio = float(draft_pressure_ratio)
        self.draft_timeout_rate_threshold = float(draft_timeout_rate_threshold)
        self.draft_pending_high_watermark = int(draft_pending_high_watermark)
        self.phase_gating_enabled = bool(phase_gating_enabled)
        self.compute_ratio_threshold = float(compute_ratio_threshold)
        self.tp_straggler_budget_ms = float(tp_straggler_budget_ms)
        self.target_slowdown_budget = float(target_slowdown_budget)
        # Step 2 and Step 3 are separate optional policies.  The legacy
        # phase_gating flag still creates the Step 2 policy for compatibility.
        self.single_gpu_policy = single_gpu_policy
        if self.single_gpu_policy is None and phase_gating_enabled:
            self.single_gpu_policy = SingleGPUCoexecPolicy(
                draft_pressure_ratio=draft_pressure_ratio,
                draft_timeout_rate_threshold=draft_timeout_rate_threshold,
                draft_pending_high_watermark=draft_pending_high_watermark,
                compute_ratio_threshold=compute_ratio_threshold,
            )
        self.multi_gpu_policy = multi_gpu_policy
        self._last = SpecStreamDecision(
            1, "parallel", "cold_start", float("inf"), COEXEC
        )
        self.draft_load_tracker = DraftLoadTracker()
        self._draft_backoff_rounds = 0

    def record_draft_result(
        self,
        *,
        elapsed_ms: float,
        rtt_ms: float | None = None,
        timeout_ms: float,
        missing_count: int,
        total_count: int,
    ) -> None:
        """Feed observed Drafter queue/network pressure into future choices."""
        self.draft_load_tracker.record_result(
            elapsed_ms=elapsed_ms if rtt_ms is None else rtt_ms,
            timeout_ms=timeout_ms,
            missing_count=missing_count,
            total_count=total_count,
        )
        if missing_count > 0:
            # Batch-uniform verification makes one request-level miss a batch
            # fallback.  Four AR rounds drain work without permanently
            # disabling the Drafter; larger missing fractions back off longer.
            missing_ratio = max(int(missing_count), 0) / max(int(total_count), 1)
            penalty = 4 + min(12, int(round(12 * missing_ratio)))
            self._draft_backoff_rounds = max(self._draft_backoff_rounds, penalty)

    def record_draft_reject(self) -> None:
        self.draft_load_tracker.record_reject()

    @property
    def draft_timeout_rate(self) -> float:
        return self.draft_load_tracker.snapshot().timeout_rate

    @property
    def draft_pressure_p95(self) -> float:
        return self.draft_load_tracker.snapshot().pressure_p95

    @property
    def draft_pending_p95(self) -> int:
        return self.draft_load_tracker.snapshot().pending_p95

    def choose(
        self,
        batch_state: SpecStreamBatchState,
        profile: SpecStreamCostProfile,
        acceptance: AcceptanceSnapshot,
        *,
        allow_ordinary: bool = True,
        draft_load: DraftLoadSnapshot | None = None,
        tp_snapshot: TPStragglerSnapshot | None = None,
    ) -> SpecStreamDecision:
        draft_load = draft_load or self.draft_load_tracker.snapshot()
        tp_snapshot = tp_snapshot or TPStragglerSnapshot()
        rank_skew_ms = float(tp_snapshot.rank_skew_ms)
        target_slowdown = float(tp_snapshot.target_slowdown)
        if batch_state.rejected or batch_state.high_overhead:
            self._last = SpecStreamDecision(
                1,
                "ordinary",
                "safety_fallback",
                0.0,
                FALLBACK,
                rank_skew_ms,
                target_slowdown,
            )
            return self._last

        if self._draft_backoff_rounds > 0:
            self._draft_backoff_rounds -= 1
            self._last = SpecStreamDecision(
                1,
                "ordinary",
                "draft_timeout_backoff",
                0.0,
                FALLBACK,
                rank_skew_ms,
                target_slowdown,
            )
            return self._last

        max_q = max(self.q_candidates)
        coexec_mode = COEXEC
        force_ordinary = False
        reason = "minimum_estimated_cost"

        tp_constraint = (
            self.multi_gpu_policy.constrain(max_q=max_q, snapshot=tp_snapshot)
            if self.multi_gpu_policy is not None
            else None
        )
        if tp_constraint is not None and tp_constraint.fallback:
            self._last = SpecStreamDecision(
                1,
                "ordinary",
                "tp_straggler_fallback",
                0.0,
                FALLBACK,
                rank_skew_ms,
                target_slowdown,
            )
            return self._last
        if (
            tp_constraint is not None
            and tp_constraint.reason != "minimum_estimated_cost"
        ):
            max_q = tp_constraint.max_q
            force_ordinary = tp_constraint.force_ordinary
            coexec_mode = tp_constraint.coexec_mode
            reason = tp_constraint.reason
        elif self.single_gpu_policy is not None:
            single_constraint = self.single_gpu_policy.constrain(
                max_q=max_q,
                profile=profile,
                draft_load=draft_load,
            )
            max_q = single_constraint.max_q
            coexec_mode = single_constraint.coexec_mode
            reason = single_constraint.reason
        eligible_q = tuple(q for q in self.q_candidates if q <= max_q)
        if not eligible_q:
            eligible_q = (min(self.q_candidates),)
        pressure_limited = (
            max(eligible_q) < max(self.q_candidates) or coexec_mode != COEXEC
        )

        candidates: list[SpecStreamDecision] = []
        for q in eligible_q:
            cost = estimate_candidate_cost(q, batch_state, profile, acceptance)
            if not force_ordinary:
                candidates.append(
                    SpecStreamDecision(
                        q,
                        "parallel",
                        reason if pressure_limited else "minimum_estimated_cost",
                        cost.parallel_ms_per_useful_token,
                        coexec_mode,
                        rank_skew_ms,
                        target_slowdown,
                    )
                )
            if allow_ordinary:
                candidates.append(
                    SpecStreamDecision(
                        q,
                        "ordinary",
                        reason if pressure_limited else "minimum_estimated_cost",
                        cost.ordinary_ms_per_useful_token,
                        SERIALIZE,
                        rank_skew_ms,
                        target_slowdown,
                    )
                )

        best = min(candidates, key=lambda item: item.estimated_cost)
        previous = next(
            (
                item
                for item in candidates
                if item.q == self._last.q
                and item.mode == self._last.mode
                and item.coexec_mode == self._last.coexec_mode
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
                    previous.coexec_mode,
                    rank_skew_ms,
                    target_slowdown,
                )
        self._last = best
        return best
