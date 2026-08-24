from sglang.srt.speculative.spectre.specstream.multi_gpu_tp_policy import (
    MultiGPUTPPolicy,
)
from sglang.srt.speculative.spectre.specstream.tp_straggler_monitor import (
    TPStragglerSnapshot,
)


def test_multi_gpu_policy_serializes_on_moderate_rank_skew():
    policy = MultiGPUTPPolicy(rank_skew_budget_ms=1.0)
    result = policy.constrain(
        max_q=8,
        snapshot=TPStragglerSnapshot(
            samples=8,
            rank_skew_ms=1.5,
            target_slowdown=0.05,
        ),
    )
    assert result.max_q == 2
    assert result.force_ordinary
    assert result.coexec_mode == "SERIALIZE"


def test_multi_gpu_policy_falls_back_on_severe_rank_skew():
    policy = MultiGPUTPPolicy(rank_skew_budget_ms=1.0)
    result = policy.constrain(
        max_q=8,
        snapshot=TPStragglerSnapshot(
            samples=8,
            rank_skew_ms=3.0,
            target_slowdown=0.05,
        ),
    )
    assert result.max_q == 1
    assert result.fallback
    assert result.coexec_mode == "FALLBACK"
