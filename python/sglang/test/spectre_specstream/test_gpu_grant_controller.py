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


def _warm_fixed_target_baseline(controller, *, shape="unmeasured", elapsed_ms=10.0):
    for _ in range(2):
        controller.record_fixed_target_forward(
            target_shape=shape,
            elapsed_ms=elapsed_ms,
            possible_overlap=False,
            confirmed_overlap=False,
        )


def test_fixed_tpc_is_profile_free_but_fails_closed_without_slack_or_timing():
    controller = GpuGrantController(
        None, calibration_tpcs=6, calibration_allow_overlap=True
    )
    decision = controller.decide(
        target_shape="unmeasured",
        draft_bs=3,
        draft_ctx_bucket="16k",
        predicted_slack_us=0,
    )
    assert decision.state is GrantState.TARGET_EXCLUSIVE
    assert decision.reason == "no_predicted_slack"

    decision = controller.decide(
        target_shape="unmeasured",
        draft_bs=3,
        draft_ctx_bucket="16k",
        predicted_slack_us=2_000,
    )
    assert decision.state is GrantState.TARGET_EXCLUSIVE
    assert decision.reason == "fixed_tpc_target_baseline_warmup"


def test_fixed_tpc_uses_online_timing_and_enforces_window_size():
    controller = GpuGrantController(
        None,
        calibration_tpcs=6,
        calibration_allow_overlap=True,
        guard_us=100,
    )
    _warm_fixed_target_baseline(controller)

    too_short = controller.decide(
        target_shape="unmeasured",
        draft_bs=3,
        draft_ctx_bucket="16k",
        predicted_slack_us=500,
        draft_step_ms=0.5,
    )
    assert too_short.state is GrantState.TARGET_EXCLUSIVE
    assert too_short.reason == "fixed_tpc_window_too_short"

    decision = controller.decide(
        target_shape="unmeasured",
        draft_bs=3,
        draft_ctx_bucket="16k",
        predicted_slack_us=600,
        draft_step_ms=0.5,
        slack_source="history_h2d",
    )
    assert decision.state is GrantState.SLACK_FILL
    assert decision.reason == "fixed_tpc_safe_pcie_slack"
    assert (decision.tpc_low, decision.tpc_high) == (0, 6)
    assert decision.draft_step_ms == 0.5


def test_fixed_tpc_slowdown_latch_blocks_overlap_but_not_catchup():
    controller = GpuGrantController(
        None,
        calibration_tpcs=6,
        calibration_allow_overlap=True,
        guard_us=0,
        target_slowdown_budget=0.05,
    )
    _warm_fixed_target_baseline(controller, elapsed_ms=10.0)
    controller.record_fixed_target_forward(
        target_shape="unmeasured",
        elapsed_ms=10.6,
        possible_overlap=True,
        confirmed_overlap=True,
    )

    overlap = controller.decide(
        target_shape="unmeasured",
        draft_bs=3,
        draft_ctx_bucket="16k",
        predicted_slack_us=2_000,
        draft_step_ms=0.5,
    )
    assert overlap.state is GrantState.TARGET_EXCLUSIVE
    assert overlap.reason == "observed_target_slowdown_over_budget"

    catchup = controller.decide(
        target_shape="unmeasured",
        draft_bs=3,
        draft_ctx_bucket="16k",
        predicted_slack_us=0,
        draft_step_ms=0.5,
        target_waiting=True,
        deadline_us=10**18,
    )
    assert catchup.state is GrantState.DRAFT_CATCHUP
    assert (catchup.tpc_low, catchup.tpc_high) == (0, 6)


def test_fixed_tpc_catchup_can_use_full_device_while_overlap_stays_bounded():
    controller = GpuGrantController(
        None,
        calibration_tpcs=34,
        catchup_tpcs=54,
        calibration_allow_overlap=True,
    )
    catchup = controller.decide(
        target_shape="verify_bs1_q4_ctx16k",
        draft_bs=1,
        draft_ctx_bucket="16k",
        predicted_slack_us=0,
        target_waiting=True,
    )
    assert catchup.state is GrantState.DRAFT_CATCHUP
    assert (catchup.tpc_low, catchup.tpc_high) == (0, 54)

    for _ in range(2):
        controller.record_fixed_target_forward(
            target_shape="verify_bs1_q4_ctx16k",
            elapsed_ms=10,
            possible_overlap=False,
            confirmed_overlap=False,
        )
    overlap = controller.decide(
        target_shape="verify_bs1_q4_ctx16k",
        draft_bs=1,
        draft_ctx_bucket="16k",
        predicted_slack_us=2000,
        draft_step_ms=1,
        slack_source="history_h2d",
    )
    assert overlap.state is GrantState.SLACK_FILL
    assert (overlap.tpc_low, overlap.tpc_high) == (0, 34)


def test_unconfirmed_possible_overlap_does_not_train_fixed_baseline():
    controller = GpuGrantController(
        None,
        calibration_tpcs=6,
        calibration_allow_overlap=True,
        fixed_baseline_min_samples=1,
    )
    controller.record_fixed_target_forward(
        target_shape="shape",
        elapsed_ms=10.0,
        possible_overlap=False,
        confirmed_overlap=False,
    )
    controller.record_fixed_target_forward(
        target_shape="shape",
        elapsed_ms=30.0,
        possible_overlap=True,
        confirmed_overlap=False,
    )

    assert controller.fixed_target_baseline_ms("shape") == 10.0
