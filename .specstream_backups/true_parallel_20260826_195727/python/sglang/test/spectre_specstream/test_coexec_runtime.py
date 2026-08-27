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
