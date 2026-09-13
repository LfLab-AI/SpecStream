import pytest

from sglang.srt.speculative.spectre.specstream import (
    coexec_runtime as coexec_runtime_module,
)
from sglang.srt.speculative.spectre.specstream.coexec_runtime import TargetGrantRuntime
from sglang.srt.speculative.spectre.specstream.gpu_grant_controller import (
    GpuGrantController,
)
from sglang.srt.speculative.spectre.specstream.resource_profile import ResourceProfile
from sglang.srt.speculative.spectre.spectre_protocol import (
    SpectreAction,
    SpectreRequest,
)


def _runtime():
    profile = ResourceProfile.from_dict(
        {
            "total_tpcs": 54,
            "entries": [
                {
                    "target_shape": "shape",
                    "draft_bs": 1,
                    "draft_ctx_bucket": "2k",
                    "draft_tpcs": 4,
                    "draft_step_ms": 0.5,
                    "target_slowdown": 0.01,
                    "target_baseline_ms": 10.0,
                }
            ],
        }
    )
    return TargetGrantRuntime(GpuGrantController(profile, guard_us=0))


def test_ack_is_required_before_next_one_token_grant():
    runtime = _runtime()
    runtime.register_round(
        request_id="r",
        spec_cnt=1,
        desired_q=2,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=1000,
    )
    first = runtime.initial_grants([("r", 1)])
    assert len(first) == 1
    assert first[0].deadline_us is not None
    assert runtime.waiting_grants([("r", 1)], deadline_us=10**18) == []

    assert runtime.acknowledge(
        SpectreRequest(
            request_id="r",
            spec_cnt=1,
            action=SpectreAction.GRANT_ACK,
            grant_epoch=first[0].grant_epoch,
            grant_tokens=1,
        )
    )
    second = runtime.waiting_grants([("r", 1)], deadline_us=10**18)
    assert len(second) == 1
    assert second[0].grant_epoch > first[0].grant_epoch


def test_ack_pipelines_all_next_round_tokens_before_target_wait():
    runtime = _runtime()
    runtime.register_round(
        request_id="r",
        spec_cnt=1,
        desired_q=3,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=100_000,
    )

    grants = runtime.initial_grants([("r", 1)])
    assert len(grants) == 1
    first_deadline = grants[0].deadline_us
    epochs = []
    for _ in range(2):
        grant = grants[0]
        epochs.append(grant.grant_epoch)
        assert runtime.acknowledge(
            SpectreRequest(
                request_id="r",
                spec_cnt=1,
                action=SpectreAction.GRANT_ACK,
                grant_epoch=grant.grant_epoch,
                grant_tokens=1,
            )
        )
        grants = runtime.overlap_grants([("r", 1)])
        assert len(grants) == 1
        assert grants[0].grant_state == "SLACK_FILL"
        assert grants[0].deadline_us == first_deadline

    epochs.append(grants[0].grant_epoch)
    assert epochs == sorted(epochs)
    assert len(set(epochs)) == 3
    assert runtime.acknowledge(
        SpectreRequest(
            request_id="r",
            spec_cnt=1,
            action=SpectreAction.GRANT_ACK,
            grant_epoch=grants[0].grant_epoch,
            grant_tokens=1,
        )
    )
    assert runtime.overlap_grants([("r", 1)]) == []


def test_round_preserves_pcie_slack_source_for_fail_closed_selection():
    profile = ResourceProfile.from_dict(
        {
            "total_tpcs": 54,
            "entries": [
                {
                    "target_shape": "shape",
                    "draft_bs": 1,
                    "draft_ctx_bucket": "2k",
                    "draft_tpcs": 4,
                    "draft_step_ms": 0.5,
                    "target_slowdown": 0.01,
                    "slack_source": "history_h2d",
                }
            ],
        }
    )
    runtime = TargetGrantRuntime(GpuGrantController(profile, guard_us=0))
    runtime.register_round(
        request_id="r",
        spec_cnt=1,
        desired_q=1,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=1000,
        slack_source="history_h2d",
        overlap_window_end_us=10**18,
    )

    state = runtime.state_for("r", 1)
    assert state is not None
    assert state.slack_source == "history_h2d"
    grant = runtime.initial_grants([("r", 1)])
    assert len(grant) == 1
    assert state.last_decision.reason == "measured_safe_pcie_slack"


def test_current_pcie_slack_admits_one_token_not_a_complete_q_horizon():
    profile = ResourceProfile.from_dict(
        {
            "total_tpcs": 54,
            "entries": [
                {
                    "target_shape": "shape",
                    "draft_bs": 1,
                    "draft_ctx_bucket": "2k",
                    "draft_tpcs": 4,
                    "draft_step_ms": 0.5,
                    "target_slowdown": 0.01,
                    "slack_source": "history_h2d",
                }
            ],
        }
    )

    too_short = TargetGrantRuntime(GpuGrantController(profile, guard_us=0))
    too_short.register_round(
        request_id="r",
        spec_cnt=1,
        desired_q=3,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=400,
        slack_source="history_h2d",
        overlap_window_end_us=10**18,
    )
    assert too_short.initial_grants([("r", 1)]) == []

    enough = TargetGrantRuntime(GpuGrantController(profile, guard_us=0))
    enough.register_round(
        request_id="r",
        spec_cnt=1,
        desired_q=3,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=1200,
        slack_source="history_h2d",
        overlap_window_end_us=10**18,
    )
    assert len(enough.initial_grants([("r", 1)])) == 1


def test_prefill_deferral_does_not_consume_token_budget():
    runtime = _runtime()
    runtime.register_round(
        request_id="r",
        spec_cnt=1,
        desired_q=1,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=1000,
    )
    first = runtime.initial_grants([("r", 1)])[0]
    assert runtime.acknowledge(
        SpectreRequest(
            request_id="r",
            spec_cnt=1,
            action=SpectreAction.GRANT_ACK,
            grant_epoch=first.grant_epoch,
            grant_tokens=0,
            grant_state="PREFILL_DEFERRED",
        )
    )
    catchup = runtime.waiting_grants([("r", 1)], deadline_us=10**18)
    assert len(catchup) == 1
    assert catchup[0].grant_state == "DRAFT_CATCHUP"


def test_profile_slowdown_baseline_is_activated_only_by_a_token_ack():
    deferred = _runtime()
    deferred.register_round(
        request_id="r",
        spec_cnt=1,
        desired_q=1,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=1000,
    )
    grant = deferred.initial_grants([("r", 1)])[0]
    assert deferred.acknowledge(
        SpectreRequest(
            request_id="r",
            spec_cnt=1,
            action=SpectreAction.GRANT_ACK,
            grant_epoch=grant.grant_epoch,
            grant_tokens=0,
            grant_state="EXPIRED",
        )
    )
    deferred.record_target_forward(20.0)
    assert not deferred.controller._force_exclusive

    executed = _runtime()
    executed.register_round(
        request_id="r",
        spec_cnt=1,
        desired_q=1,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=1000,
    )
    grant = executed.initial_grants([("r", 1)])[0]
    assert executed.acknowledge(
        SpectreRequest(
            request_id="r",
            spec_cnt=1,
            action=SpectreAction.GRANT_ACK,
            grant_epoch=grant.grant_epoch,
            grant_tokens=1,
            grant_state="SLACK_FILL",
        )
    )
    executed.record_target_forward(20.0)
    assert executed.controller._force_exclusive


def _fixed_runtime(*, guard_us=0, slowdown_budget=0.05):
    return TargetGrantRuntime(
        GpuGrantController(
            None,
            calibration_tpcs=4,
            calibration_allow_overlap=True,
            guard_us=guard_us,
            target_slowdown_budget=slowdown_budget,
        )
    )


def _record_fixed_target_only_round(runtime, *, spec_cnt, draft_step_ms=0.5):
    runtime.register_round(
        request_id="r",
        spec_cnt=spec_cnt,
        desired_q=1,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=0,
        draft_step_ms=draft_step_ms,
    )
    assert runtime.initial_grants([("r", spec_cnt)]) == []
    runtime.record_target_forward(10.0)


def test_fixed_tpc_current_h2d_requires_only_one_safe_token_window():
    runtime = _fixed_runtime()
    _record_fixed_target_only_round(runtime, spec_cnt=1)
    _record_fixed_target_only_round(runtime, spec_cnt=2)

    runtime.register_round(
        request_id="r",
        spec_cnt=3,
        desired_q=3,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=400,
        draft_step_ms=0.5,
        slack_source="history_h2d",
        overlap_window_end_us=10**18,
    )
    assert runtime.initial_grants([("r", 3)]) == []
    state = runtime.state_for("r", 3)
    assert state is not None
    assert state.last_decision.reason == "fixed_tpc_window_too_short"

    runtime.register_round(
        request_id="r",
        spec_cnt=4,
        desired_q=3,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=1_200,
        draft_step_ms=0.5,
        slack_source="history_h2d",
        overlap_window_end_us=10**18,
    )
    grants = runtime.initial_grants([("r", 4)])
    assert len(grants) == 1
    assert grants[0].grant_state == "SLACK_FILL"
    assert grants[0].deadline_us is not None


def test_same_current_h2d_window_never_extends_multi_token_deadline(monkeypatch):
    runtime = _fixed_runtime()
    _record_fixed_target_only_round(runtime, spec_cnt=1)
    _record_fixed_target_only_round(runtime, spec_cnt=2)
    now_us = [1_000_000]
    monkeypatch.setattr(
        coexec_runtime_module.time, "monotonic_ns", lambda: now_us[0] * 1000
    )
    runtime.register_round(
        request_id="r",
        spec_cnt=3,
        desired_q=3,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=2_000,
        draft_step_ms=0.5,
        slack_source="history_h2d",
        overlap_window_end_us=1_002_000,
    )

    grants = runtime.initial_grants([("r", 3)])
    assert len(grants) == 1
    fixed_deadline_us = grants[0].deadline_us
    assert fixed_deadline_us == 1_001_500
    for next_now_us in (1_000_500, 1_001_000):
        grant = grants[0]
        assert runtime.acknowledge(
            SpectreRequest(
                request_id="r",
                spec_cnt=3,
                action=SpectreAction.GRANT_ACK,
                grant_epoch=grant.grant_epoch,
                grant_tokens=1,
                draft_step_ms=0.5,
            )
        )
        now_us[0] = next_now_us
        grants = runtime.overlap_grants([("r", 3)])
        assert len(grants) == 1
        assert grants[0].deadline_us == fixed_deadline_us


def test_new_current_h2d_window_rearms_only_next_token(monkeypatch):
    runtime = _fixed_runtime()
    _record_fixed_target_only_round(runtime, spec_cnt=1)
    _record_fixed_target_only_round(runtime, spec_cnt=2)
    now_us = [1_000_000]
    monkeypatch.setattr(
        coexec_runtime_module.time, "monotonic_ns", lambda: now_us[0] * 1000
    )
    runtime.register_round(
        request_id="r",
        spec_cnt=3,
        desired_q=2,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=800,
        draft_step_ms=0.5,
        slack_source="history_h2d",
        overlap_window_end_us=1_000_800,
    )
    first = runtime.initial_grants([("r", 3)])[0]
    assert first.deadline_us == 1_000_300
    assert runtime.acknowledge(
        SpectreRequest(
            request_id="r",
            spec_cnt=3,
            action=SpectreAction.GRANT_ACK,
            grant_epoch=first.grant_epoch,
            grant_tokens=1,
            draft_step_ms=0.5,
        )
    )

    # This is the state update performed only after verifier.py observes a new
    # physical H2D window and confirms that no grant remains outstanding.
    state = runtime.state_for("r", 3)
    assert state is not None
    state.predicted_slack_us = 1_000
    state.overlap_window_end_us = 2_001_000
    now_us[0] = 2_000_000
    second = runtime.overlap_grants([("r", 3)])[0]
    assert second.deadline_us == 2_000_500
    assert second.deadline_us > first.deadline_us


def test_fixed_tpc_zero_slack_never_creates_undeadlined_overlap():
    runtime = TargetGrantRuntime(
        GpuGrantController(
            None,
            calibration_tpcs=4,
            calibration_allow_overlap=True,
        )
    )
    runtime.register_round(
        request_id="r",
        spec_cnt=1,
        desired_q=2,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=0,
        draft_step_ms=0.5,
    )
    grants = runtime.initial_grants([("r", 1)])
    assert grants == []
    state = runtime.state_for("r", 1)
    assert state is not None
    assert state.last_decision.reason == "no_predicted_slack"


def test_registering_new_spec_cnt_retires_old_round_and_clear_resets_state():
    runtime = _runtime()
    for spec_cnt in (1, 2):
        runtime.register_round(
            request_id="r",
            spec_cnt=spec_cnt,
            desired_q=1,
            target_shape="shape",
            draft_bs=1,
            draft_ctx_bucket="2k",
            predicted_slack_us=1_000,
        )

    assert runtime.state_for("r", 1) is None
    assert runtime.state_for("r", 2) is not None
    runtime.clear()
    assert runtime.state_for("r", 2) is None
    assert runtime._epochs == {}


def test_fixed_tpc_online_slowdown_latches_following_overlap():
    runtime = _fixed_runtime(slowdown_budget=0.05)
    _record_fixed_target_only_round(runtime, spec_cnt=1)
    _record_fixed_target_only_round(runtime, spec_cnt=2)

    runtime.register_round(
        request_id="r",
        spec_cnt=3,
        desired_q=1,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=2_000,
        draft_step_ms=0.5,
    )
    grant = runtime.initial_grants([("r", 3)])[0]
    assert runtime.acknowledge(
        SpectreRequest(
            request_id="r",
            spec_cnt=3,
            action=SpectreAction.GRANT_ACK,
            grant_epoch=grant.grant_epoch,
            grant_tokens=1,
            draft_step_ms=0.5,
        )
    )
    runtime.record_target_forward(10.6)

    runtime.register_round(
        request_id="r",
        spec_cnt=4,
        desired_q=1,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=2_000,
        draft_step_ms=0.5,
    )
    assert runtime.initial_grants([("r", 4)]) == []
    state = runtime.state_for("r", 4)
    assert state is not None
    assert state.last_decision.reason == "observed_target_slowdown_over_budget"


@pytest.mark.parametrize("completion", ["zero_ack", "expired"])
def test_fixed_tpc_unexecuted_grant_does_not_mark_real_overlap(completion):
    runtime = _fixed_runtime(slowdown_budget=0.05)
    _record_fixed_target_only_round(runtime, spec_cnt=1)
    _record_fixed_target_only_round(runtime, spec_cnt=2)
    runtime.register_round(
        request_id="r",
        spec_cnt=3,
        desired_q=1,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=2_000,
        draft_step_ms=0.5,
    )
    grant = runtime.initial_grants([("r", 3)])[0]
    if completion == "zero_ack":
        assert runtime.acknowledge(
            SpectreRequest(
                request_id="r",
                spec_cnt=3,
                action=SpectreAction.GRANT_ACK,
                grant_epoch=grant.grant_epoch,
                grant_tokens=0,
                grant_state="EXPIRED",
            )
        )
    else:
        runtime.pause_messages([("r", 3)])
        assert not runtime.acknowledge(
            SpectreRequest(
                request_id="r",
                spec_cnt=3,
                action=SpectreAction.GRANT_ACK,
                grant_epoch=grant.grant_epoch,
                grant_tokens=1,
                draft_step_ms=0.5,
            )
        )
    runtime.record_target_forward(10.6)
    assert runtime.controller.fixed_target_baseline_ms("shape") == 10.0

    runtime.register_round(
        request_id="r",
        spec_cnt=4,
        desired_q=1,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=2_000,
        draft_step_ms=0.5,
    )
    assert runtime.initial_grants([("r", 4)])


def test_ack_refreshes_fixed_tpc_in_round_draft_step_estimate():
    runtime = _fixed_runtime()
    runtime.register_round(
        request_id="r",
        spec_cnt=1,
        desired_q=1,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=0,
    )
    runtime.initial_grants([("r", 1)])
    grant = runtime.waiting_grants([("r", 1)], deadline_us=10**18)[0]
    assert runtime.acknowledge(
        SpectreRequest(
            request_id="r",
            spec_cnt=1,
            action=SpectreAction.GRANT_ACK,
            grant_epoch=grant.grant_epoch,
            grant_tokens=1,
            draft_step_ms=0.75,
        )
    )
    state = runtime.state_for("r", 1)
    assert state is not None
    assert state.draft_step_ms == 0.75


def test_pause_revokes_outstanding_target_state():
    runtime = _runtime()
    runtime.register_round(
        request_id="r",
        spec_cnt=1,
        desired_q=2,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=1000,
    )
    first = runtime.initial_grants([("r", 1)])[0]
    pause = runtime.pause_messages([("r", 1)])
    assert len(pause) == 1
    assert pause[0].action is SpectreAction.PAUSE
    assert not runtime.acknowledge(
        SpectreRequest(
            request_id="r",
            spec_cnt=1,
            action=SpectreAction.GRANT_ACK,
            grant_epoch=first.grant_epoch,
            grant_tokens=1,
        )
    )


def test_live_target_slowdown_latches_exclusive_mode():
    runtime = _runtime()
    runtime.register_round(
        request_id="r",
        spec_cnt=1,
        desired_q=2,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=1000,
    )
    grant = runtime.initial_grants([("r", 1)])[0]
    assert runtime.acknowledge(
        SpectreRequest(
            request_id="r",
            spec_cnt=1,
            action=SpectreAction.GRANT_ACK,
            grant_epoch=grant.grant_epoch,
            grant_tokens=1,
            grant_state="SLACK_FILL",
        )
    )
    runtime.record_target_forward(11.0)

    runtime.register_round(
        request_id="r",
        spec_cnt=2,
        desired_q=2,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=1000,
    )
    assert runtime.initial_grants([("r", 2)]) == []


def test_grant_protocol_round_trip_preserves_control_fields():
    message = SpectreRequest(
        request_id="r",
        spec_cnt=3,
        action=SpectreAction.GRANT,
        grant_epoch=7,
        grant_tokens=1,
        tpc_low=0,
        tpc_high=4,
        deadline_us=123456,
        placement_id=0,
        grant_state="SLACK_FILL",
    )
    restored = SpectreRequest.from_dict(message.to_dict())
    assert restored == message
