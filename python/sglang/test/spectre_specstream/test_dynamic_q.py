from sglang.srt.speculative.spectre.specstream.acceptance_tracker import (
    AcceptanceTracker,
)
from sglang.srt.speculative.spectre.specstream.controller import IOAwareController
from sglang.srt.speculative.spectre.specstream.cost_model import (
    SpecStreamBatchState,
    SpecStreamCostProfile,
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
        missing_count=3,
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


def test_tp_straggler_serializes_before_it_becomes_severe():
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
    assert decision.q <= 2
    assert decision.mode == "ordinary"
    assert decision.coexec_mode == "SERIALIZE"
    assert decision.reason == "tp_straggler_throttle"


def test_severe_tp_straggler_forces_q1_fallback():
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
            rank_forward_ms=(14.0, 10.0),
            rank_skew_ms=4.0,
            target_slowdown=0.3,
        ),
    )
    assert (decision.q, decision.mode, decision.coexec_mode) == (
        1,
        "ordinary",
        "FALLBACK",
    )
    assert decision.reason == "tp_straggler_fallback"


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
