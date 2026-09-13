from __future__ import annotations

from dataclasses import dataclass, replace
from collections import OrderedDict
from typing import Iterable

from sglang.srt.speculative.spectre.specstream.acceptance_tracker import (
    AcceptanceSnapshot,
)
from sglang.srt.speculative.spectre.specstream.cost_model import (
    SpecStreamBatchState,
    SpecStreamCostProfile,
    estimate_candidate_cost,
    workload_shape_key,
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
class CandidateControllerCost:
    q: int
    parallel_ms_per_useful_token: float
    ordinary_ms_per_useful_token: float
    verify_ms: float
    draft_rtt_ms: float


@dataclass(frozen=True)
class SpecStreamDecision:
    q: int
    mode: str
    reason: str
    estimated_cost: float
    coexec_mode: str = COEXEC
    rank_skew_ms: float = 0.0
    target_slowdown: float = 0.0
    parallel_cost: float = 0.0
    ordinary_cost: float = 0.0
    candidate_costs: tuple[CandidateControllerCost, ...] = ()
    planned_mode: str = ""


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
        draft_rtt_warmup_samples: int = 4,
        parallel_probe_interval: int = 8,
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
        if draft_rtt_warmup_samples < 1:
            raise ValueError("draft_rtt_warmup_samples must be positive")
        if parallel_probe_interval < 1:
            raise ValueError("parallel_probe_interval must be positive")
        self.switch_threshold = switch_threshold
        self.draft_pressure_ratio = float(draft_pressure_ratio)
        self.draft_timeout_rate_threshold = float(draft_timeout_rate_threshold)
        self.draft_pending_high_watermark = int(draft_pending_high_watermark)
        self.phase_gating_enabled = bool(phase_gating_enabled)
        self.compute_ratio_threshold = float(compute_ratio_threshold)
        self.tp_straggler_budget_ms = float(tp_straggler_budget_ms)
        self.target_slowdown_budget = float(target_slowdown_budget)
        self.draft_rtt_warmup_samples = int(draft_rtt_warmup_samples)
        self.parallel_probe_interval = int(parallel_probe_interval)
        self._parallel_probe_rounds: dict[tuple, int] = {}
        self._decision_round = 0
        # Negative admission outcomes belong to the requested q, not the q=1
        # execution bucket. Bound memory independently of context churn.
        self._parallel_failures: OrderedDict[tuple, tuple[int, int, str]] = OrderedDict()
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
            missing_ratio = max(int(missing_count), 0) / max(int(total_count), 1)
            # A minority miss is repaired per request by a full-context resync;
            # it must not force unrelated requests into global q=1 backoff.
            # Back off only for a majority outage, and cap the drain period so
            # one long-context tail event cannot create hundreds of AR rounds.
            if missing_ratio <= 0.5:
                return
            penalty = 2 + min(4, int(round(4 * missing_ratio)))
            self._draft_backoff_rounds = max(self._draft_backoff_rounds, penalty)

    def record_draft_reject(self) -> None:
        self.draft_load_tracker.record_reject()

    def record_parallel_result(self, batch_state, row) -> None:
        if row.controller_selected_mode != "parallel":
            return
        key = (workload_shape_key(batch_state), int(row.controller_selected_q))
        failed = row.q != row.controller_selected_q or row.slack_fill_issued_tokens == 0
        if not failed:
            self._parallel_failures.pop(key, None)
            return
        failures = min(self._parallel_failures.get(key, (0, 0, ""))[0] + 1, 6)
        # Two short recovery attempts; repeated failures get 128..512 rounds
        # of cooldown. A changed workload shape can be evaluated independently.
        delay = (8 << (failures - 1)) if failures < 3 else min(128 << (failures - 3), 512)
        reason = "parallel_no_slack_backoff"
        if row.q != row.controller_selected_q:
            reason = "parallel_horizon_backoff"
        elif row.draft_step_ms > 0 and row.h2d_window_remaining_us < row.draft_step_ms * 1000:
            reason = "parallel_window_too_short"
        self._parallel_failures[key] = (failures, self._decision_round + delay, reason)
        self._parallel_failures.move_to_end(key)
        while len(self._parallel_failures) > 128:
            self._parallel_failures.popitem(last=False)

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
        force_ordinary: bool = False,
        draft_load: DraftLoadSnapshot | None = None,
        tp_snapshot: TPStragglerSnapshot | None = None,
        tp_snapshots_by_q: dict[int, TPStragglerSnapshot] | None = None,
        parallel_ready: bool = True,
        allow_pipeline_seed: bool = False,
    ) -> SpecStreamDecision:
        self._decision_round += 1
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
        force_ordinary = bool(force_ordinary)
        reason = "minimum_estimated_cost"

        tp_constraint = (
            self.multi_gpu_policy.constrain(max_q=max_q, snapshot=tp_snapshot)
            if self.multi_gpu_policy is not None and tp_snapshots_by_q is None
            else None
        )
        if (
            tp_constraint is not None
            and tp_constraint.reason != "minimum_estimated_cost"
        ):
            max_q = tp_constraint.max_q
            force_ordinary = force_ordinary or tp_constraint.force_ordinary
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

        # PCIe-slack co-execution is only useful after at least one request in
        # the batch has published CPU-resident History KV.  Before that point
        # there is no History-H2D transfer window for the colocated Drafter to
        # fill.  Letting the generic cost model select ``parallel`` here adds
        # ZMQ/MPS scheduling overhead and can make the Drafter contend with the
        # Target despite there being no offload benefit.  Keep dynamic-q, but
        # restrict this round to SPECTRE ordinary (serial) execution.
        if batch_state.history_tokens <= 0:
            force_ordinary = True
            coexec_mode = SERIALIZE
            reason = "no_offloaded_history"
        eligible_q = tuple(q for q in self.q_candidates if q <= max_q)
        if not eligible_q:
            eligible_q = (min(self.q_candidates),)
        pressure_limited = (
            max(eligible_q) < max(self.q_candidates) or coexec_mode != COEXEC
        )

        candidate_estimates = {
            q: estimate_candidate_cost(q, batch_state, profile, acceptance)
            for q in eligible_q
        }
        candidate_costs = tuple(
            CandidateControllerCost(
                q=q,
                parallel_ms_per_useful_token=(
                    candidate_estimates[q].parallel_ms_per_useful_token
                ),
                ordinary_ms_per_useful_token=(
                    candidate_estimates[q].ordinary_ms_per_useful_token
                ),
                verify_ms=candidate_estimates[q].verify_ms,
                draft_rtt_ms=candidate_estimates[q].draft_ms,
            )
            for q in eligible_q
        )
        candidates: list[SpecStreamDecision] = []
        for q in eligible_q:
            cost = candidate_estimates[q]
            candidate_force_ordinary = force_ordinary
            candidate_reason = reason if pressure_limited else "minimum_estimated_cost"
            candidate_coexec_mode = coexec_mode
            candidate_skew = rank_skew_ms
            candidate_slowdown = target_slowdown
            failed_attempt = self._parallel_failures.get((workload_shape_key(batch_state), q))
            if failed_attempt is not None and self._decision_round <= failed_attempt[1]:
                candidate_force_ordinary = True
                candidate_reason = failed_attempt[2]
                candidate_coexec_mode = SERIALIZE
            if self.multi_gpu_policy is not None and tp_snapshots_by_q is not None:
                snapshot = tp_snapshots_by_q.get(q, TPStragglerSnapshot())
                constraint = self.multi_gpu_policy.constrain(max_q=q, snapshot=snapshot)
                candidate_skew = snapshot.rank_skew_ms
                candidate_slowdown = snapshot.target_slowdown
                if constraint.force_ordinary:
                    candidate_force_ordinary = True
                    candidate_reason = constraint.reason
                    candidate_coexec_mode = constraint.coexec_mode
            if not candidate_force_ordinary:
                candidates.append(
                    SpecStreamDecision(
                        q,
                        "parallel",
                        candidate_reason,
                        cost.parallel_ms_per_useful_token,
                        candidate_coexec_mode,
                        candidate_skew,
                        candidate_slowdown,
                        cost.parallel_ms_per_useful_token,
                        cost.ordinary_ms_per_useful_token,
                        candidate_costs,
                    )
                )
            if allow_ordinary or candidate_force_ordinary:
                candidates.append(
                    SpecStreamDecision(
                        q,
                        "ordinary",
                        candidate_reason,
                        cost.ordinary_ms_per_useful_token,
                        SERIALIZE,
                        candidate_skew,
                        candidate_slowdown,
                        cost.parallel_ms_per_useful_token,
                        cost.ordinary_ms_per_useful_token,
                        candidate_costs,
                    )
                )

        under_sampled_q = [
            q
            for q in eligible_q
            if q > 1
            and profile.draft_rtt_sample_count(q) < self.draft_rtt_warmup_samples
        ]
        if len(self.q_candidates) > 1 and under_sampled_q:
            # A candidate-specific cost model needs candidate-specific data.
            # Round-robin the least-sampled horizons for a bounded warmup,
            # bypassing hysteresis so every configured q becomes observable.
            exploratory_q = min(
                under_sampled_q,
                key=lambda q: (profile.draft_rtt_sample_count(q), q),
            )
            exploratory = [item for item in candidates if item.q == exploratory_q]
            selected = min(exploratory, key=lambda item: item.estimated_cost)
            best = SpecStreamDecision(
                selected.q,
                selected.mode,
                (
                    "draft_rtt_warmup"
                    if selected.reason == "minimum_estimated_cost"
                    else selected.reason
                ),
                selected.estimated_cost,
                selected.coexec_mode,
                selected.rank_skew_ms,
                selected.target_slowdown,
                selected.parallel_cost,
                selected.ordinary_cost,
                selected.candidate_costs,
            )
        else:
            max_eligible_q = max(eligible_q)
            max_q_samples = int(acceptance.samples.get(max_eligible_q, 0))
            if (
                batch_state.history_tokens > 0
                and not pressure_limited
                and max_q_samples < 8
            ):
                # Do not let one unlucky first q=max sample permanently lock the
                # controller into shorter blocks. History I/O is paid once per
                # verify round, so q=max needs a small, bounded evidence window
                # before measured acceptance can fairly rule it out.
                exploratory = [item for item in candidates if item.q == max_eligible_q]
                selected = min(exploratory, key=lambda item: item.estimated_cost)
                best = SpecStreamDecision(
                    selected.q,
                    selected.mode,
                    (
                        "max_q_acceptance_warmup"
                        if selected.reason == "minimum_estimated_cost"
                        else selected.reason
                    ),
                    selected.estimated_cost,
                    selected.coexec_mode,
                    selected.rank_skew_ms,
                    selected.target_slowdown,
                    selected.parallel_cost,
                    selected.ordinary_cost,
                    selected.candidate_costs,
                )
            else:
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
                            previous.rank_skew_ms,
                            previous.target_slowdown,
                            previous.parallel_cost,
                            previous.ordinary_cost,
                            previous.candidate_costs,
                        )
        # A serial round consumes all current drafts. An opt-in, bounded
        # seed obtains an aligned next-round draft using the existing retry
        # protocol; the next decision may then probe an admitted parallel q.
        if allow_pipeline_seed and parallel_ready and self._last.reason == "parallel_pipeline_seed":
            seeded = next((item for item in candidates
                           if item.q == self._last.q and item.mode == "parallel"), None)
            if seeded is not None:
                best = replace(seeded, reason="parallel_seeded_probe")
        if best.mode == "ordinary" and batch_state.history_tokens > 0 and (parallel_ready or allow_pipeline_seed):
            alternate = next(
                (
                    item
                    for item in candidates
                    if item.q == best.q and item.mode == "parallel"
                ),
                None,
            )
            if (
                alternate is not None
                and profile.empirical_cost(batch_state, best.q, "ordinary") is not None
                and profile.empirical_cost(batch_state, best.q, "parallel") is None
            ):
                # A conservative unmeasured overlap estimate can otherwise
                # suppress all observations of parallel mode. Probe only an
                # already-admitted q/shape, at bounded frequency, until its
                # measured candidate bucket is warm. This never overrides
                # Target protection, no-history gating or forced serial mode.
                key = (workload_shape_key(batch_state), best.q)
                count = self._parallel_probe_rounds.get(key, 0) + 1
                self._parallel_probe_rounds[key] = count
                while len(self._parallel_probe_rounds) > 128:
                    self._parallel_probe_rounds.pop(
                        next(iter(self._parallel_probe_rounds))
                    )
                if count % self.parallel_probe_interval == 0:
                    best = (
                        replace(alternate, reason="parallel_cost_probe")
                        if parallel_ready
                        else replace(best, reason="parallel_pipeline_seed", planned_mode="parallel")
                    )
        if best.mode == "parallel" and not parallel_ready:
            # Decide before sending Draft requests. Ordinary mode waits for the
            # requested horizon; entering parallel first would silently run q=1.
            best = replace(
                best, mode="ordinary", coexec_mode=SERIALIZE,
                reason="parallel_draft_not_ready", planned_mode="parallel",
                estimated_cost=best.ordinary_cost,
            )
        self._last = best
        return best
