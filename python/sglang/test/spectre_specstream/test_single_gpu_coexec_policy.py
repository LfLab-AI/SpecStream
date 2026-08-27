from sglang.srt.speculative.spectre.specstream.cost_model import (
    SpecStreamCostProfile,
)
from sglang.srt.speculative.spectre.specstream.draft_load_tracker import (
    DraftLoadSnapshot,
)
from sglang.srt.speculative.spectre.specstream.single_gpu_coexec_policy import (
    SingleGPUCoexecPolicy,
)


def test_legacy_single_gpu_policy_no_longer_converts_pressure_into_gpu_control():
    policy = SingleGPUCoexecPolicy()
    result = policy.constrain(
        max_q=8,
        profile=SpecStreamCostProfile(),
        draft_load=DraftLoadSnapshot(
            samples=8,
            rtt_ema_ms=4600.0,
            rtt_p95_ms=4600.0,
            pressure_p95=0.92,
        ),
    )
    assert result.max_q == 8
    assert result.coexec_mode == "COEXEC"
    assert result.reason == "deprecated_mps_policy_disabled"


def test_single_gpu_policy_does_not_read_tp_state():
    policy = SingleGPUCoexecPolicy()
    result = policy.constrain(
        max_q=8,
        profile=SpecStreamCostProfile(),
        draft_load=DraftLoadSnapshot(),
    )
    assert result.max_q == 8
    assert result.coexec_mode == "COEXEC"
    assert result.reason == "deprecated_mps_policy_disabled"
