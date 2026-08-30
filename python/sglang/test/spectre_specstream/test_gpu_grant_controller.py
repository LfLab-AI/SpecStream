from sglang.srt.speculative.spectre.specstream.gpu_grant_controller import (
    GpuGrantController,
    GrantState,
)
from sglang.srt.speculative.spectre.specstream.resource_profile import ResourceProfile


def _controller():
    profile = ResourceProfile.from_dict(
        {
            "total_tpcs": 54,
            "entries": [
                {
                    "target_shape": "shape",
                    "draft_bs": 2,
                    "draft_ctx_bucket": "8k",
                    "draft_tpcs": 4,
                    "draft_step_ms": 1.0,
                    "target_slowdown": 0.04,
                }
            ],
        }
    )
    return GpuGrantController(profile, target_slowdown_budget=0.05, guard_us=100)


def test_default_is_target_exclusive_without_proven_slack():
    decision = _controller().decide(
        target_shape="shape",
        draft_bs=2,
        draft_ctx_bucket="8k",
        predicted_slack_us=0,
    )
    assert decision.state is GrantState.TARGET_EXCLUSIVE


def test_safe_calibrated_window_uses_slack_fill():
    decision = _controller().decide(
        target_shape="shape",
        draft_bs=2,
        draft_ctx_bucket="8k",
        predicted_slack_us=1200,
    )
    assert decision.state is GrantState.SLACK_FILL
    assert (decision.tpc_low, decision.tpc_high) == (0, 4)


def test_pcie_slack_requires_a_history_h2d_calibration_entry():
    controller = _controller()
    decision = controller.decide(
        target_shape="shape",
        draft_bs=2,
        draft_ctx_bucket="8k",
        predicted_slack_us=1200,
        slack_source="history_h2d",
    )
    assert decision.state is GrantState.TARGET_EXCLUSIVE

    profile = ResourceProfile.from_dict(
        {
            "total_tpcs": 54,
            "entries": [
                {
                    "target_shape": "shape",
                    "draft_bs": 2,
                    "draft_ctx_bucket": "8k",
                    "draft_tpcs": 4,
                    "draft_step_ms": 1.0,
                    "target_slowdown": 0.04,
                    "slack_source": "history_h2d",
                }
            ],
        }
    )
    decision = GpuGrantController(
        profile, target_slowdown_budget=0.05, guard_us=100
    ).decide(
        target_shape="shape",
        draft_bs=2,
        draft_ctx_bucket="8k",
        predicted_slack_us=1200,
        slack_source="history_h2d",
    )
    assert decision.state is GrantState.SLACK_FILL
    assert decision.reason == "measured_safe_pcie_slack"


def test_target_wait_uses_catchup_but_still_requires_calibration():
    decision = _controller().decide(
        target_shape="shape",
        draft_bs=2,
        draft_ctx_bucket="8k",
        predicted_slack_us=0,
        target_waiting=True,
    )
    assert decision.state is GrantState.DRAFT_CATCHUP


def test_fixed_tpc_calibration_is_explicit_and_profile_free():
    controller = GpuGrantController(
        None, calibration_tpcs=6, calibration_allow_overlap=True
    )
    decision = controller.decide(
        target_shape="unmeasured",
        draft_bs=3,
        draft_ctx_bucket="16k",
        predicted_slack_us=0,
    )
    assert decision.state is GrantState.SLACK_FILL
    assert (decision.tpc_low, decision.tpc_high) == (0, 6)
