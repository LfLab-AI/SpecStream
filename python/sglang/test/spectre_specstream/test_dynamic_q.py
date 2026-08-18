from sglang.srt.speculative.spectre.specstream.acceptance_tracker import (
    AcceptanceTracker,
)
from sglang.srt.speculative.spectre.specstream.controller import IOAwareController
from sglang.srt.speculative.spectre.specstream.cost_model import (
    SpecStreamBatchState,
    SpecStreamCostProfile,
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


def test_near_timeout_draft_pressure_caps_horizon():
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
    assert decision.q <= 2
    assert decision.reason == "draft_pressure_limited"
