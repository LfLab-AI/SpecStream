import threading

import pytest

from sglang.srt.speculative.spectre.specstream.background_grant_pump import (
    BackgroundGrantPump,
)


def test_control_progresses_while_target_submission_thread_is_blocked():
    initialized = threading.Event()
    granted = threading.Event()
    target_done = threading.Event()
    observations = []
    def step():
        assert initialized.is_set()
        observations.append(target_done.is_set())
        granted.set()

    pump = BackgroundGrantPump(step, initialize=initialized.set).start()
    try:
        # Models a host-side staging wait during Target forward. A receive
        # callback that starts only after Target returns cannot satisfy this.
        assert granted.wait(2.0)
        assert observations and observations[0] is False
        target_done.set()
    finally:
        pump.stop()
    count = len(observations)
    assert pump.iterations == count
    assert pump.elapsed_ms >= 0


def test_stop_is_a_barrier_for_inflight_control_before_request_release():
    callback_entered = threading.Event()
    callback_can_finish = threading.Event()
    barrier_returned = threading.Event()
    callbacks = []
    def step():
        callback_entered.set()
        assert callback_can_finish.wait(2.0)
        callbacks.append("sent")

    pump = BackgroundGrantPump(step).start()
    assert callback_entered.wait(2.0)
    stopper = threading.Thread(target=lambda: (pump.stop(), barrier_returned.set()))
    stopper.start()
    try:
        assert not barrier_returned.wait(0.02)
        callback_can_finish.set()
        assert barrier_returned.wait(2.0)
        assert callbacks == ["sent"]
        # stop can safely be repeated by an exception cleanup path.
        pump.stop()
        assert callbacks == ["sent"]
    finally:
        callback_can_finish.set()
        stopper.join(2.0)
        pump.stop()


def test_pump_errors_are_propagated_at_join_and_do_not_continue():
    reached = threading.Event()
    def step():
        reached.set()
        raise ValueError("invalid grant generation")

    pump = BackgroundGrantPump(step).start()
    assert reached.wait(2.0)
    with pytest.raises(RuntimeError, match="background grant pump failed") as error:
        pump.stop()
    assert isinstance(error.value.__cause__, ValueError)
    pump.stop(raise_errors=False)


def test_completed_batch_ends_pump_without_issuing_later_grants():
    finished = threading.Event()
    def step():
        finished.set()
        return False

    pump = BackgroundGrantPump(step).start()
    assert finished.wait(2.0)
    pump.stop()
    assert pump.iterations == 1


@pytest.mark.parametrize(
    "staging_device, expected_index, expected_resolutions",
    [("cuda", 1, 1), ("cuda:0", 0, 0), ("cuda:1", 1, 0)],
)
def test_scheduler_pump_binds_device_resolved_on_parent_thread(
    monkeypatch, staging_device, expected_index, expected_resolutions
):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from sglang.srt.speculative.spectre.verifier.spectre_target_scheduler_mixin import (
        SchedulerSpectreTargetMixin,
    )

    parent_thread = threading.get_ident()
    resolutions = []
    bindings = []
    advanced = threading.Event()

    def current_device():
        assert threading.get_ident() == parent_thread
        resolutions.append(1)
        # A nonzero ordinal catches accidental hard-coding of CUDA device 0.
        return 1

    def set_device(index):
        assert threading.get_ident() != parent_thread
        assert isinstance(index, int)
        assert index == expected_index
        bindings.append(index)

    def drain(pending):
        assert bindings == [expected_index]
        advanced.set()
        return [], set(pending)

    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    monkeypatch.setenv("SPECSTREAM_BACKGROUND_GRANT_PUMP", "1")
    runtime = SimpleNamespace(
        grant_runtime=object(),
        config=SimpleNamespace(pcie_slack_coexec=True),
        staging=SimpleNamespace(device=torch.device(staging_device)),
    )
    scheduler = object.__new__(SchedulerSpectreTargetMixin)
    scheduler.tp_rank = 0
    scheduler._get_specstream_runtime = lambda: runtime
    scheduler._get_reqs_waiting_for_drafts = lambda batch: [
        SimpleNamespace(rid="request", spec_cnt=7)
    ]
    scheduler.req_to_draft_token = {"request": {7: None}}
    scheduler._drain_grant_acks_during_forward = drain
    batch = SimpleNamespace(
        specstream_mode="parallel", specstream_meta=SimpleNamespace(enabled=True)
    )

    pump = scheduler.start_specstream_grant_pump(batch)
    assert pump is not None
    try:
        assert advanced.wait(2.0)
    finally:
        pump.stop()
    assert bindings == [expected_index]
    assert len(resolutions) == expected_resolutions


def test_scheduler_ack_drain_preserves_drafts_and_excludes_completed_keys():
    pytest.importorskip("torch")
    from types import SimpleNamespace
    from sglang.srt.speculative.spectre.spectre_protocol import SpectreAction
    from sglang.srt.speculative.spectre.verifier.spectre_target_scheduler_mixin import (
        SchedulerSpectreTargetMixin,
    )
    scheduler = object.__new__(SchedulerSpectreTargetMixin)
    scheduler._msg_lock = threading.Lock()
    scheduler._data_ready = threading.Event()
    def message(action, rid, spec_cnt):
        return SimpleNamespace(action=action, request_id=rid, spec_cnt=spec_cnt)

    ack = message(SpectreAction.GRANT_ACK, "a", 7)
    future = message(SpectreAction.DRAFT, "a", 8)
    draft = message(SpectreAction.DRAFT, "b", 7)
    need_context = message(SpectreAction.NEED_CONTEXT, "c", 7)
    scheduler._msg_buffer = [future, ack, draft, need_context]
    scheduler._data_ready.set()
    acks, complete = scheduler._drain_grant_acks_during_forward(
        {("a", 7), ("b", 7), ("c", 7)})
    assert acks == [ack]
    assert complete == {("b", 7), ("c", 7)}
    assert scheduler._msg_buffer == [future, draft, need_context]
    assert scheduler._data_ready.is_set()
    scheduler._msg_buffer[:] = [ack]
    acks, complete = scheduler._drain_grant_acks_during_forward({("a", 7)})
    assert acks == [ack] and complete == set()
    assert not scheduler._data_ready.is_set()


def test_idle_harvest_records_late_ack_after_request_retirement_once():
    pytest.importorskip("torch")
    from types import SimpleNamespace
    from sglang.srt.speculative.spectre.spectre_protocol import SpectreAction
    from sglang.srt.speculative.spectre.verifier.spectre_target_scheduler_mixin import (
        SchedulerSpectreTargetMixin,
    )
    scheduler = object.__new__(SchedulerSpectreTargetMixin)
    scheduler.tp_rank = 0
    scheduler._msg_lock = threading.Lock()
    scheduler._data_ready = threading.Event()
    scheduler.req_to_draft_token = {}  # The final request has already retired.
    draft = SimpleNamespace(action=SpectreAction.DRAFT, request_id="old", spec_cnt=15)
    ack = SimpleNamespace(action=SpectreAction.GRANT_ACK, request_id="old", spec_cnt=15)
    scheduler._msg_buffer = [draft]
    assert scheduler._drain_msg_buffer() == [draft]
    scheduler._msg_buffer[:] = [ack, draft]
    scheduler._data_ready.set()
    recorded = []
    # No live grant state: acknowledge returns False, but the runtime still
    # writes this terminal disposition to its profile.
    runtime = SimpleNamespace(acknowledge_grant=lambda message: recorded.append(message) or False)
    scheduler._get_specstream_runtime = lambda: runtime
    assert scheduler._harvest_buffered_grant_acks() == 1
    assert recorded == [ack]
    assert scheduler._msg_buffer == [draft]
    assert scheduler._harvest_buffered_grant_acks() == 0
    assert recorded == [ack]
    scheduler.tp_rank = 1
    scheduler._msg_buffer.insert(0, ack)
    assert scheduler._harvest_buffered_grant_acks() == 0
    assert scheduler._msg_buffer == [ack, draft]


def test_cache_reset_preserves_ack_arriving_during_runtime_clear():
    pytest.importorskip("torch")
    from types import SimpleNamespace
    from sglang.srt.speculative.spectre.spectre_protocol import SpectreAction
    from sglang.srt.speculative.spectre.verifier.spectre_target_scheduler_mixin import (
        SchedulerSpectreTargetMixin,
    )
    scheduler = object.__new__(SchedulerSpectreTargetMixin)
    scheduler.tp_rank = 0
    scheduler._msg_lock = threading.Lock()
    scheduler._data_ready = threading.Event()
    early = SimpleNamespace(action=SpectreAction.GRANT_ACK, request_id="early", spec_cnt=1)
    late = SimpleNamespace(action=SpectreAction.GRANT_ACK, request_id="late", spec_cnt=2)
    draft = SimpleNamespace(action=SpectreAction.DRAFT, request_id="old", spec_cnt=1)
    scheduler._msg_buffer = [early, draft]
    recorded = []
    def clear_runtime():
        with scheduler._msg_lock:
            scheduler._msg_buffer.append(late)
            scheduler._data_ready.set()
    runtime = SimpleNamespace(acknowledge_grant=recorded.append, clear=clear_runtime)
    scheduler._get_specstream_runtime = lambda: runtime
    scheduler.reset_spectre_target_state()
    assert recorded == [early]
    assert scheduler._msg_buffer == [late]
    assert scheduler._data_ready.is_set()
    scheduler._harvest_buffered_grant_acks()
    assert recorded == [early, late]
    assert scheduler._msg_buffer == []
    assert not scheduler._data_ready.is_set()
