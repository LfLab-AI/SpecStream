from pathlib import Path

from sglang.srt.speculative.spectre.spectre_protocol import (
    SpectreAction,
    SpectreRequest,
    SpecType,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
CPP_PROTOCOL = (
    REPO_ROOT
    / "python/sglang/srt/speculative/spectre/cpp_zmq/include/spectre_protocol.hpp"
)
CPP_SERIALIZER = (
    REPO_ROOT
    / "python/sglang/srt/speculative/spectre/cpp_zmq/src/spectre_zmq_serialization.cpp"
)


def test_cpp_transport_supports_every_python_action():
    header = CPP_PROTOCOL.read_text(encoding="utf-8")

    for ordinal, action in enumerate(SpectreAction):
        assert f"{action.name} = {ordinal}" in header
        assert f'"{action.value}"' in header


def test_cpp_transport_contains_execution_grant_schema_v2():
    header = CPP_PROTOCOL.read_text(encoding="utf-8")
    serializer = CPP_SERIALIZER.read_text(encoding="utf-8")

    assert "kProtocolSchemaVersion = 2" in header
    assert "kSpectreRequestV1FieldCount = 15" in serializer
    assert "kSpectreRequestFieldCount = 23" in serializer
    for field in (
        "grant_epoch",
        "grant_tokens",
        "tpc_low",
        "tpc_high",
        "deadline_us",
        "placement_id",
        "grant_state",
        "draft_step_ms",
    ):
        assert field in header
        assert field in serializer


def test_python_execution_grant_fields_round_trip():
    request = SpectreRequest(
        request_id="rid",
        spec_cnt=3,
        action=SpectreAction.GRANT,
        spec_type=SpecType.DRAFT_REQUEST,
        grant_epoch=7,
        grant_tokens=1,
        tpc_low=0,
        tpc_high=4,
        deadline_us=1234567890123,
        placement_id=2,
        grant_state="SLACK_FILL",
        draft_step_ms=1.25,
    )

    assert SpectreRequest.from_dict(request.to_dict()) == request
