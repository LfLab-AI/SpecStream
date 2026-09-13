from sglang.srt.speculative.spectre.specstream.multi_gpu_tp_policy import (
    MultiGPUTPPolicy,
)
from sglang.srt.speculative.spectre.specstream.tp_straggler_monitor import (
    TPStragglerSnapshot,
)


def test_unknown_baseline_serializes_without_changing_q():
    policy = MultiGPUTPPolicy(rank_skew_budget_ms=1.0)
    result = policy.constrain(
        max_q=8,
        snapshot=TPStragglerSnapshot(
            samples=8,
            rank_skew_ms=1.5,
            target_slowdown=0.05,
        ),
    )
    assert result.max_q == 8
    assert result.force_ordinary
    assert result.coexec_mode == "SERIALIZE"


def test_severe_unattributed_rank_skew_cannot_force_q1():
    policy = MultiGPUTPPolicy(rank_skew_budget_ms=1.0)
    result = policy.constrain(
        max_q=8,
        snapshot=TPStragglerSnapshot(
            samples=8,
            rank_skew_ms=3.0,
            target_slowdown=0.05,
        ),
    )
    assert result.max_q == 8
    assert not result.fallback
    assert result.coexec_mode == "SERIALIZE"


def _snapshot(round_id, *, overlap=True, slowdown=0.4, key=(4, 8)):
    return TPStragglerSnapshot(
        samples=round_id,
        round_id=round_id,
        shape_key=key,
        baseline_ready=True,
        baseline_samples=3,
        overlap_active=overlap,
        target_slowdown=slowdown,
    )


def test_real_overlap_regression_serializes_then_recovers_in_bounded_samples():
    policy = MultiGPUTPPolicy(cooldown_samples=3)
    assert not policy.constrain(max_q=8, snapshot=_snapshot(6)).force_ordinary
    bad = _snapshot(7)
    result = policy.constrain(max_q=8, snapshot=bad)
    assert result.force_ordinary and result.max_q == 8 and not result.fallback
    for _ in range(20):
        assert policy.constrain(max_q=8, snapshot=bad).force_ordinary
    for round_id in (8, 9):
        assert policy.constrain(
            max_q=8, snapshot=_snapshot(round_id, overlap=False, slowdown=0)
        ).force_ordinary
    result = policy.constrain(
        max_q=8, snapshot=_snapshot(10, overlap=False, slowdown=0)
    )
    assert not result.force_ordinary
    assert result.max_q == 8


def test_single_overlap_jitter_does_not_latch_and_shape_bans_do_not_leak():
    policy = MultiGPUTPPolicy()
    assert not policy.constrain(max_q=8, snapshot=_snapshot(6)).force_ordinary
    assert not policy.constrain(
        max_q=8, snapshot=_snapshot(7, slowdown=0)
    ).force_ordinary
    assert not policy.constrain(max_q=8, snapshot=_snapshot(8)).force_ordinary
    assert policy.constrain(max_q=8, snapshot=_snapshot(9)).force_ordinary
    assert not policy.constrain(
        max_q=8, snapshot=_snapshot(10, key=(1, 8), slowdown=0)
    ).force_ordinary
