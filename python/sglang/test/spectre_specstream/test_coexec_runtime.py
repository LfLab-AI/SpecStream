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
    )

    state = runtime.state_for("r", 1)
    assert state is not None
    assert state.slack_source == "history_h2d"
    grant = runtime.initial_grants([("r", 1)])
    assert len(grant) == 1
    assert state.last_decision.reason == "measured_safe_pcie_slack"


def test_pcie_slack_admits_only_a_complete_next_round_horizon():
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
        predicted_slack_us=1200,
        slack_source="history_h2d",
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
        predicted_slack_us=3000,
        slack_source="history_h2d",
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


def test_calibration_overlap_does_not_create_one_microsecond_deadline():
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
    )
    grants = runtime.initial_grants([("r", 1)])
    assert len(grants) == 1
    assert grants[0].grant_state == "SLACK_FILL"
    assert grants[0].deadline_us is None


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
    assert runtime.initial_grants([("r", 1)])
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
