from sglang.srt.speculative.spectre.specstream.acceptance_tracker import (
    AcceptanceSnapshot,
    AcceptanceTracker,
)
from sglang.srt.speculative.spectre.specstream.controller import IOAwareController
from sglang.srt.speculative.spectre.specstream.cost_model import (
    SpecStreamBatchState,
    SpecStreamCostProfile,
    estimate_candidate_cost,
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


def _state(**kwargs):
    base = dict(
        batch_size=2,
        context_tokens=32768,
        history_tokens=28672,
        history_bytes=2_000_000_000,
        num_chunks=14,
    )
    base.update(kwargs)
    return SpecStreamBatchState(**base)


def test_controller_returns_supported_batch_level_q():
    ctl = IOAwareController((1, 2, 4, 8), switch_threshold=0)
    decision = ctl.choose(
        _state(),
        SpecStreamCostProfile(h2d_gbps=8),
        AcceptanceTracker().snapshot((1, 2, 4, 8)),
    )
    assert decision.q in (1, 2, 4, 8)
    assert decision.mode in ("ordinary", "parallel")


def test_force_ordinary_keeps_dynamic_q_but_removes_parallel_candidates():
    ctl = IOAwareController((2, 4, 6, 8), switch_threshold=0)
    decision = ctl.choose(
        _state(),
        SpecStreamCostProfile(h2d_gbps=8),
        AcceptanceTracker().snapshot((2, 4, 6, 8)),
        force_ordinary=True,
    )
    assert decision.q in (2, 4, 6, 8)
    assert decision.mode == "ordinary"


def test_ordinary_cost_uses_candidate_specific_observed_draft_rtt():
    acceptance = AcceptanceTracker().snapshot((2, 8))
    profile = SpecStreamCostProfile(
        draft_rtt_ms_by_q=((2, 20.0), (8, 90.0)),
        draft_rtt_samples_by_q=((2, 8), (8, 8)),
    )

    state = _state(history_tokens=0, history_bytes=0, num_chunks=0)
    q2 = estimate_candidate_cost(2, state, profile, acceptance)
    q8 = estimate_candidate_cost(8, state, profile, acceptance)

    assert q2.draft_ms == 20.0
    assert q8.draft_ms == 90.0
    assert q8.ordinary_ms_per_useful_token > q2.ordinary_ms_per_useful_token


def test_dynamic_q_collects_each_rtt_bucket_before_cost_selection():
    ctl = IOAwareController(
        (2, 4, 6, 8),
        switch_threshold=0,
        draft_rtt_warmup_samples=1,
    )
    acceptance = AcceptanceTracker().snapshot((2, 4, 6, 8))
    samples = ()
    observed = ()
    for expected_q in (2, 4, 6, 8):
        decision = ctl.choose(
            _state(),
            SpecStreamCostProfile(
                draft_rtt_ms_by_q=observed,
                draft_rtt_samples_by_q=samples,
            ),
            acceptance,
            force_ordinary=True,
        )
        assert decision.q == expected_q
        assert decision.reason == "draft_rtt_warmup"
        samples = (*samples, (expected_q, 1))
        observed = (*observed, (expected_q, float(expected_q * 10)))


def test_dynamic_q_writes_all_candidate_costs_and_selects_measured_minimum():
    ctl = IOAwareController((2, 4, 6, 8), switch_threshold=0)
    acceptance = AcceptanceSnapshot(
        expected_useful_tokens={2: 2.0, 4: 4.0, 6: 6.0, 8: 8.0},
        rollback_probability={2: 0.0, 4: 0.0, 6: 0.0, 8: 0.0},
        samples={2: 16, 4: 16, 6: 16, 8: 16},
    )
    profile = SpecStreamCostProfile(
        draft_rtt_ms_by_q=((2, 10.0), (4, 40.0), (6, 80.0), (8, 140.0)),
        draft_rtt_samples_by_q=((2, 4), (4, 4), (6, 4), (8, 4)),
    )

    decision = ctl.choose(_state(), profile, acceptance, force_ordinary=True)

    assert decision.reason == "minimum_estimated_cost"
    assert decision.ordinary_cost > 0.0
    assert decision.estimated_cost == decision.ordinary_cost
    assert {item.q for item in decision.candidate_costs} == {2, 4, 6, 8}
    assert (
        decision.q
        == min(
            decision.candidate_costs,
            key=lambda item: item.ordinary_ms_per_useful_token,
        ).q
    )


def test_no_offloaded_history_forces_serial_execution_but_keeps_dynamic_q():
    ctl = IOAwareController((2, 4, 6, 8), switch_threshold=0)
    decision = ctl.choose(
        _state(history_tokens=0, history_bytes=0, num_chunks=0),
        SpecStreamCostProfile(h2d_gbps=8),
        AcceptanceTracker().snapshot((2, 4, 6, 8)),
    )
    assert decision.q in (2, 4, 6, 8)
    assert decision.mode == "ordinary"
    assert decision.coexec_mode == "SERIALIZE"
    assert decision.reason == "no_offloaded_history"


def test_safety_fallback_has_priority():
    ctl = IOAwareController((1, 4, 8))
    decision = ctl.choose(
        _state(rejected=True),
        SpecStreamCostProfile(),
        AcceptanceTracker().snapshot((1, 4, 8)),
    )
    assert (decision.q, decision.mode, decision.reason) == (
        1,
        "ordinary",
        "safety_fallback",
    )


def test_acceptance_tracker_bounds_useful_progress():
    tracker = AcceptanceTracker(alpha=1)
    tracker.update(4, [0, 1, 3])
    snapshot = tracker.snapshot((4,))
    assert snapshot.useful_tokens(4) == 7 / 3
    assert 0 <= snapshot.rollback(4) <= 1


def test_draft_timeout_forces_short_ar_backoff():
    ctl = IOAwareController((1, 2, 4, 8), switch_threshold=0)
    ctl.record_draft_result(
        elapsed_ms=5000,
        timeout_ms=5000,
        missing_count=9,
        total_count=16,
    )
    decision = ctl.choose(
        _state(batch_size=16),
        SpecStreamCostProfile(),
        AcceptanceTracker().snapshot((1, 2, 4, 8)),
    )
    assert (decision.q, decision.mode, decision.reason) == (
        1,
        "ordinary",
        "draft_timeout_backoff",
    )


def test_legacy_step2_policy_does_not_change_q_or_execution_permission():
    ctl = IOAwareController(
        (1, 2, 4, 8),
        switch_threshold=0,
        single_gpu_policy=SingleGPUCoexecPolicy(),
    )
    for _ in range(8):
        ctl.record_draft_result(
            elapsed_ms=4600,
            timeout_ms=5000,
            missing_count=0,
            total_count=16,
        )
    decision = ctl.choose(
        _state(batch_size=16),
        SpecStreamCostProfile(),
        AcceptanceTracker().snapshot((1, 2, 4, 8)),
    )
    assert decision.q in (1, 2, 4, 8)
    assert decision.reason != "draft_pressure_limited"
    assert decision.coexec_mode == "COEXEC"


def test_untagged_tp_sample_preserves_serial_multiquery_warmup():
    ctl = IOAwareController(
        (1, 2, 4, 8),
        switch_threshold=0,
        multi_gpu_policy=MultiGPUTPPolicy(
            rank_skew_budget_ms=1.0,
            target_slowdown_budget=0.10,
        ),
    )
    decision = ctl.choose(
        _state(),
        SpecStreamCostProfile(),
        AcceptanceTracker().snapshot((1, 2, 4, 8)),
        tp_snapshot=TPStragglerSnapshot(
            samples=8,
            colocated_rank=0,
            rank_forward_ms=(11.5, 10.0),
            rank_skew_ms=1.5,
            target_slowdown=0.05,
        ),
    )
    assert decision.q in (2, 4, 8)
    assert decision.mode == "ordinary"
    assert decision.coexec_mode == "SERIALIZE"
    assert decision.reason == "tp_baseline_warmup"


def test_severe_tp_straggler_preserves_fixed_q8_verification():
    ctl = IOAwareController(
        (8,),
        switch_threshold=0,
        multi_gpu_policy=MultiGPUTPPolicy(
            rank_skew_budget_ms=1.0,
            target_slowdown_budget=0.10,
            violation_samples=1,
        ),
    )
    decision = ctl.choose(
        _state(),
        SpecStreamCostProfile(),
        AcceptanceTracker().snapshot((1, 2, 4, 8)),
        tp_snapshot=TPStragglerSnapshot(
            samples=8,
            colocated_rank=0,
            rank_forward_ms=(14.0, 10.0),
            rank_skew_ms=4.0,
            target_slowdown=0.3,
            shape_key=(4, 8),
            baseline_ready=True,
            overlap_active=True,
            excess_rank_skew_ms=4.0,
        ),
    )
    assert (decision.q, decision.mode, decision.coexec_mode) == (
        8,
        "ordinary",
        "SERIALIZE",
    )
    assert decision.reason == "tp_overlap_cooldown"


def test_compute_ratio_is_not_used_as_gpu_execution_permission():
    ctl = IOAwareController(
        (1, 2, 4, 8),
        switch_threshold=0,
        single_gpu_policy=SingleGPUCoexecPolicy(
            compute_ratio_threshold=0.9,
        ),
    )
    decision = ctl.choose(
        _state(),
        SpecStreamCostProfile(
            target_compute_ratio=0.95,
            exposed_copy_ms=0.01,
            target_other_ms=1.0,
        ),
        AcceptanceTracker().snapshot((1, 2, 4, 8)),
    )
    assert decision.q in (1, 2, 4, 8)
    assert decision.coexec_mode == "COEXEC"
    assert decision.reason != "target_compute_heavy"


def test_tp_snapshot_is_ignored_when_step3_policy_is_disabled():
    ctl = IOAwareController((1, 2, 4, 8), switch_threshold=0)
    decision = ctl.choose(
        _state(),
        SpecStreamCostProfile(),
        AcceptanceTracker().snapshot((1, 2, 4, 8)),
        tp_snapshot=TPStragglerSnapshot(
            samples=8,
            colocated_rank=0,
            rank_forward_ms=(14.0, 10.0),
            rank_skew_ms=4.0,
            target_slowdown=0.3,
        ),
    )
    assert decision.reason != "tp_straggler_fallback"


def test_draft_pressure_is_ignored_when_step2_policy_is_disabled():
    ctl = IOAwareController((1, 2, 4, 8), switch_threshold=0)
    for _ in range(8):
        ctl.record_draft_result(
            elapsed_ms=4600,
            timeout_ms=5000,
            missing_count=0,
            total_count=16,
        )
    decision = ctl.choose(
        _state(batch_size=16),
        SpecStreamCostProfile(),
        AcceptanceTracker().snapshot((1, 2, 4, 8)),
    )
    assert decision.reason != "draft_pressure_limited"
