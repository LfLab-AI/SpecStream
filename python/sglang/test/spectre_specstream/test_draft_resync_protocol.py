from types import SimpleNamespace

from sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin import (
    SpectreDraftSchedulerMixin,
)
from sglang.srt.speculative.spectre.spectre_protocol import (
    SpectreAction,
    SpectreRequest,
    SpecType,
)
from sglang.srt.speculative.spectre.verifier.spectre_target_scheduler_mixin import (
    SchedulerSpectreTargetMixin,
    _draft_needs_full_context,
    _mark_draft_context_synced,
    _mark_draft_context_unsynced,
)


class _FakeCommunicator:
    def __init__(self):
        self.messages = []

    def send_objs(self, messages):
        self.messages.extend(messages)


def _req(rid="r0", spec_cnt=7, initialized=False, forced=False):
    return SimpleNamespace(
        rid=rid,
        spec_cnt=spec_cnt,
        spectre_draft_initialized=initialized,
        spectre_force_full_draft_context=forced,
    )


def test_spec_cnt_does_not_imply_remote_state_exists():
    req = _req(spec_cnt=7, initialized=False)
    assert _draft_needs_full_context(req)

    _mark_draft_context_synced(req)
    assert not _draft_needs_full_context(req)

    _mark_draft_context_unsynced(req)
    assert _draft_needs_full_context(req)


def test_half_open_probe_always_resynchronizes_full_context():
    req = _req(initialized=True)
    assert _draft_needs_full_context(req, is_half_open=True)


def test_need_context_protocol_round_trip():
    msg = SpectreRequest(
        request_id="r0",
        spec_cnt=9,
        action=SpectreAction.NEED_CONTEXT,
        spec_type=SpecType.DRAFT_RESPONSE,
    )
    restored = SpectreRequest.from_dict(msg.to_dict())
    assert restored.action is SpectreAction.NEED_CONTEXT
    assert restored.spec_cnt == 9


def test_drafter_replies_when_incremental_request_has_no_state():
    scheduler = object.__new__(SpectreDraftSchedulerMixin)
    scheduler.tp_size = 1
    scheduler.tp_rank = 0
    scheduler.zmq_communicator = _FakeCommunicator()
    request = SpectreRequest(
        request_id="missing",
        spec_cnt=4,
        action=SpectreAction.DRAFT,
        spec_type=SpecType.DRAFT_REQUEST,
        input_ids=None,
    )

    scheduler._send_need_context_response(request)

    assert len(scheduler.zmq_communicator.messages) == 1
    response = scheduler.zmq_communicator.messages[0]
    assert response.action is SpectreAction.NEED_CONTEXT
    assert (response.request_id, response.spec_cnt) == ("missing", 4)


def test_one_missing_request_does_not_downgrade_whole_batch():
    scheduler = object.__new__(SchedulerSpectreTargetMixin)
    scheduler.server_args = SimpleNamespace(spectre_no_draft_ratio=0.5)
    reqs = [_req(rid=f"r{i}", initialized=True) for i in range(4)]
    reqs[0].spectre_force_normal_decode = True
    batch = SimpleNamespace(reqs=reqs)

    assert not scheduler._consume_request_timeout_fallback(batch)
    assert all(not getattr(req, "spectre_force_normal_decode", False) for req in reqs)


def test_majority_missing_requests_keep_uniform_safety_round():
    scheduler = object.__new__(SchedulerSpectreTargetMixin)
    scheduler.server_args = SimpleNamespace(spectre_no_draft_ratio=0.5)
    reqs = [_req(rid=f"r{i}", initialized=True) for i in range(4)]
    for req in reqs[:3]:
        req.spectre_force_normal_decode = True
    batch = SimpleNamespace(reqs=reqs)

    assert scheduler._consume_request_timeout_fallback(batch)
