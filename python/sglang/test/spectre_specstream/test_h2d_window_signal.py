from collections import deque
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from sglang.srt.speculative.spectre.specstream import (
    staging_runtime as staging_runtime_module,
)
from sglang.srt.speculative.spectre.specstream.staging_runtime import (
    H2DWindowObservation,
    StagingWindowPool,
    _TrackedH2DWindow,
    _detect_pcie_bandwidth_ceiling_gbps,
)
from sglang.srt.speculative.spectre.specstream.verifier import (
    SpecStreamTargetRuntime,
    _require_pcie_slack_physical_ceiling,
)


class _FakeEvent:
    def __init__(self, complete: bool, elapsed_ms: float = 1.0):
        self.complete = complete
        self.elapsed_ms = elapsed_ms
        self.records = 0

    def record(self, stream=None):
        self.records += 1

    def query(self):
        return self.complete

    def elapsed_time(self, other):
        return other.elapsed_ms


def test_nvml_pcie_raw_link_rate_is_used_as_physical_ceiling(
    monkeypatch,
):
    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByUUID=lambda uuid: "gpu",
        nvmlDeviceGetMaxPcieLinkGeneration=lambda handle: 4,
        nvmlDeviceGetMaxPcieLinkWidth=lambda handle: 16,
    )
    monkeypatch.setitem(__import__("sys").modules, "pynvml", fake_nvml)
    monkeypatch.setattr(
        staging_runtime_module.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(uuid="GPU-test"),
    )

    ceiling, source = _detect_pcie_bandwidth_ceiling_gbps(
        __import__("torch").device("cuda")
    )
    assert ceiling == pytest.approx(35.2)
    assert source == "nvml_pcie_gen4_x16_raw_upper_bound"


def _tracked_cpu_staging(window: _TrackedH2DWindow) -> StagingWindowPool:
    staging = StagingWindowPool(2, "cpu")
    staging._track_h2d_windows = True
    staging._h2d_bandwidth_ceiling_gbps = 10.0
    staging._h2d_windows = deque([window])
    staging._h2d_windows_by_id = {window.window_id: window}
    return staging


def _mark_target_wait_exposed(window: _TrackedH2DWindow) -> _TrackedH2DWindow:
    window.target_wait_begin_event = _FakeEvent(True)
    window.target_wait_end_event = _FakeEvent(False)
    window.target_wait_gate_recorded = True
    return window


def test_current_h2d_event_transition_produces_bounded_budget(monkeypatch):
    begin = _FakeEvent(False)
    end = _FakeEvent(False)
    window = _mark_target_wait_exposed(
        _TrackedH2DWindow(
            round_id=7,
            window_id=11,
            nbytes=10_000_000,
            begin_event=begin,
            end_event=end,
            enqueued_ns=900_000,
        )
    )
    staging = _tracked_cpu_staging(window)

    monkeypatch.setattr(
        staging_runtime_module.time, "perf_counter_ns", lambda: 1_000_000
    )
    monkeypatch.setattr(staging_runtime_module.time, "monotonic_ns", lambda: 5_000_000)
    pending = staging.observe_h2d_window(7)
    assert not pending.active
    assert pending.reason == "current_h2d_pending"

    begin.complete = True
    monkeypatch.setattr(
        staging_runtime_module.time, "perf_counter_ns", lambda: 1_100_000
    )
    monkeypatch.setattr(staging_runtime_module.time, "monotonic_ns", lambda: 5_100_000)
    active = staging.observe_h2d_window(7)
    assert active.active
    # 10 MB / 10 GB/s = 1000 us; the observed transition bounds age at 100 us.
    assert active.remaining_us == pytest.approx(900.0)
    assert active.window_end_us == 6_000
    assert "target_wait_gate" in active.timing_source


def test_first_poll_active_uses_event_enqueue_time_as_safe_age_bound(monkeypatch):
    window = _mark_target_wait_exposed(
        _TrackedH2DWindow(
            round_id=3,
            window_id=5,
            nbytes=10_000_000,
            begin_event=_FakeEvent(True),
            end_event=_FakeEvent(False),
            enqueued_ns=1_000_000,
        )
    )
    staging = _tracked_cpu_staging(window)
    monkeypatch.setattr(
        staging_runtime_module.time, "perf_counter_ns", lambda: 1_250_000
    )
    monkeypatch.setattr(staging_runtime_module.time, "monotonic_ns", lambda: 8_000_000)

    observation = staging.observe_h2d_window(3)
    assert observation.active
    assert observation.remaining_us == pytest.approx(750.0)
    assert observation.window_end_us == 8_750


def test_prefetch_h2d_is_not_slack_until_target_compute_reaches_wait(monkeypatch):
    target_wait_begin = _FakeEvent(False)
    target_wait_end = _FakeEvent(False)
    window = _TrackedH2DWindow(
        round_id=5,
        window_id=6,
        nbytes=10_000_000,
        begin_event=_FakeEvent(True),
        end_event=_FakeEvent(False),
        enqueued_ns=1_000_000,
        target_wait_begin_event=target_wait_begin,
        target_wait_end_event=target_wait_end,
        target_wait_gate_recorded=True,
    )
    staging = _tracked_cpu_staging(window)
    monkeypatch.setattr(
        staging_runtime_module.time, "perf_counter_ns", lambda: 1_100_000
    )
    monkeypatch.setattr(staging_runtime_module.time, "monotonic_ns", lambda: 20_000_000)

    prefetch = staging.observe_h2d_window(5)
    assert not prefetch.active
    assert prefetch.reason == "target_compute_before_h2d_wait"

    target_wait_begin.complete = True
    exposed = staging.observe_h2d_window(5)
    assert exposed.active
    assert exposed.reason == "current_exposed_h2d_stall"

    target_wait_end.complete = True
    completed_wait = staging.observe_h2d_window(5)
    assert not completed_wait.active
    assert completed_wait.reason == "target_h2d_wait_completed"


def test_wait_ready_records_target_stream_stall_gate(monkeypatch):
    pre_wait = _FakeEvent(False)
    post_wait = _FakeEvent(False)
    window = _TrackedH2DWindow(
        round_id=2,
        window_id=3,
        nbytes=1024,
        begin_event=_FakeEvent(False),
        end_event=_FakeEvent(False),
        enqueued_ns=0,
        target_wait_begin_event=pre_wait,
        target_wait_end_event=post_wait,
    )
    staging = StagingWindowPool(2, "cpu")
    staging.copy_stream = object()
    staging._h2d_windows = deque([window])
    staging._h2d_windows_by_id = {window.window_id: window}
    ready_event = _FakeEvent(False)
    staging._ready_events[0] = ready_event

    class _FakeTargetStream:
        def __init__(self):
            self.waited = []

        def wait_event(self, event):
            self.waited.append(event)

    target_stream = _FakeTargetStream()
    monkeypatch.setattr(
        staging_runtime_module.torch.cuda,
        "current_stream",
        lambda device: target_stream,
    )
    transfer = SimpleNamespace(slot=0, h2d_window_id=3, tensor="payload")

    assert staging.wait_ready(transfer) == "payload"
    assert pre_wait.records == 1
    assert post_wait.records == 1
    assert target_stream.waited == [ready_event]
    assert window.target_wait_gate_recorded


def test_serialized_h2d_orders_copy_after_target_stream(monkeypatch):
    staging = StagingWindowPool(1, "cpu", serialize_h2d=True)

    class _FakeCopyStream:
        def __init__(self):
            self.waited = []

        def wait_stream(self, stream):
            self.waited.append(stream)

    target_stream = object()
    copy_stream = _FakeCopyStream()
    staging.copy_stream = copy_stream
    monkeypatch.setattr(
        staging_runtime_module.torch.cuda,
        "current_stream",
        lambda device: target_stream,
    )

    staging._serialize_after_target_compute()

    assert copy_stream.waited == [target_stream]


def test_async_h2d_does_not_add_target_to_copy_dependency(monkeypatch):
    staging = StagingWindowPool(2, "cpu", serialize_h2d=False)
    staging.copy_stream = SimpleNamespace(wait_stream=lambda stream: pytest.fail())
    monkeypatch.setattr(
        staging_runtime_module.torch.cuda,
        "current_stream",
        lambda device: pytest.fail(),
    )

    staging._serialize_after_target_compute()


def test_polling_active_window_primes_next_ring_entry_age(monkeypatch):
    first_end = _FakeEvent(False)
    second_begin = _FakeEvent(False)
    first = _mark_target_wait_exposed(
        _TrackedH2DWindow(
            round_id=6,
            window_id=1,
            nbytes=10_000_000,
            begin_event=_FakeEvent(True),
            end_event=first_end,
            enqueued_ns=100_000,
        )
    )
    second = _mark_target_wait_exposed(
        _TrackedH2DWindow(
            round_id=6,
            window_id=2,
            nbytes=10_000_000,
            begin_event=second_begin,
            end_event=_FakeEvent(False),
            enqueued_ns=100_000,
        )
    )
    staging = _tracked_cpu_staging(first)
    staging._h2d_windows.append(second)
    monkeypatch.setattr(
        staging_runtime_module.time, "perf_counter_ns", lambda: 1_000_000
    )
    monkeypatch.setattr(staging_runtime_module.time, "monotonic_ns", lambda: 10_000_000)

    assert staging.observe_h2d_window(6).window_id == 1
    assert second.last_not_started_ns == 1_000_000

    first_end.complete = True
    first.target_wait_end_event.complete = True
    second_begin.complete = True
    monkeypatch.setattr(
        staging_runtime_module.time, "perf_counter_ns", lambda: 1_100_000
    )
    monkeypatch.setattr(staging_runtime_module.time, "monotonic_ns", lambda: 10_100_000)
    observation = staging.observe_h2d_window(6)
    assert observation.window_id == 2
    assert observation.active
    # Harvesting the first 10 GB/s sample raises the optimistic bandwidth
    # ceiling to 12.5 GB/s: 800 us floor minus a 100 us age upper bound.
    assert observation.remaining_us == pytest.approx(700.0)


def test_completed_cuda_window_is_harvested_without_synchronize():
    window = _TrackedH2DWindow(
        round_id=4,
        window_id=9,
        nbytes=20_000_000,
        begin_event=_FakeEvent(True),
        end_event=_FakeEvent(True, elapsed_ms=2.0),
        enqueued_ns=0,
        target_wait_begin_event=_FakeEvent(True),
        target_wait_end_event=_FakeEvent(True, elapsed_ms=0.75),
        target_wait_gate_recorded=True,
    )
    staging = _tracked_cpu_staging(window)

    observation = staging.observe_h2d_window(4)
    samples = staging.drain_h2d_timing_samples()
    assert not observation.active
    assert len(samples) == 1
    assert samples[0].elapsed_ms == 2.0
    assert samples[0].target_wait_ms == 0.75
    assert samples[0].nbytes == 20_000_000
    assert len(staging._h2d_event_pool) == 1


def test_event_pool_grows_instead_of_dropping_unfinished_windows(monkeypatch):
    staging = StagingWindowPool(2, "cpu")
    staging.copy_stream = object()
    staging._track_h2d_windows = True
    event_pairs = [
        (
            _FakeEvent(False),
            _FakeEvent(False),
            _FakeEvent(False),
            _FakeEvent(False),
        )
        for _ in range(8)
    ]
    staging._h2d_event_pool = list(event_pairs)
    monkeypatch.setattr(
        staging_runtime_module.torch.cuda,
        "Event",
        lambda **_kwargs: _FakeEvent(False),
    )

    for round_id in range(1, 11):
        window = staging._begin_h2d_window(
            round_id=round_id, nbytes=round_id * 1000, asynchronous=True
        )
        assert window is not None
        staging._end_h2d_window(window)

    assert [window.round_id for window in staging._h2d_windows] == list(range(1, 11))
    assert len(staging._h2d_windows) == 10
    assert set(staging._h2d_windows_by_id) == {
        window.window_id for window in staging._h2d_windows
    }
    assert len(staging._h2d_event_pool) == 0
    # The original eight tuples are reused and two more are allocated. No
    # unfinished timing interval is overwritten, so coverage remains exact.
    assert len({id(window.begin_event) for window in staging._h2d_windows}) == 10
    assert (
        len({id(window.target_wait_end_event) for window in staging._h2d_windows}) == 10
    )


def test_copy_descriptor_is_published_only_after_end_event_is_recorded():
    staging = StagingWindowPool(2, "cpu")
    staging.copy_stream = object()
    staging._track_h2d_windows = True
    # Recycled events can report completed from a previous generation.
    staging._h2d_event_pool = [tuple(_FakeEvent(True) for _ in range(4))]
    window = staging._begin_h2d_window(round_id=4, nbytes=1000, asynchronous=True)
    assert window is not None
    assert not staging._h2d_windows
    assert not staging.observe_h2d_window(4).active
    assert staging.drain_h2d_timing_samples() == []
    staging._end_h2d_window(window)
    assert staging._h2d_windows_by_id[window.window_id] is window
    # Even a completed prefetch must retain events until wait_ready publishes
    # its Target wait markers, or discard explicitly cancels the consumer.
    assert staging.drain_h2d_timing_samples() == []
    assert len(staging._h2d_event_pool) == 0
    staging.discard(SimpleNamespace(h2d_window_id=window.window_id))
    samples = staging.drain_h2d_timing_samples()
    assert len(samples) == 1
    assert samples[0].window_id == window.window_id
    assert len(staging._h2d_event_pool) == 1


def test_discard_does_not_recycle_a_copy_that_is_still_in_flight():
    window = _TrackedH2DWindow(
        round_id=4, window_id=8, nbytes=1000,
        begin_event=_FakeEvent(True), end_event=_FakeEvent(False),
        enqueued_ns=0, consumer_pending=True,
    )
    staging = _tracked_cpu_staging(window)
    staging.discard(SimpleNamespace(h2d_window_id=8))
    assert staging.drain_h2d_timing_samples() == []
    assert 8 in staging._h2d_windows_by_id
    window.end_event.complete = True
    assert len(staging.drain_h2d_timing_samples()) == 1
    assert 8 not in staging._h2d_windows_by_id


class _FakeProfiler:
    def __init__(self):
        self.observations = []

    def record_h2d_event(self, *args, **kwargs):
        pass

    def record_h2d_window_observation(self, observation):
        self.observations.append(observation)

    def record_grant(self, *args, **kwargs):
        pass


class _FakeStaging:
    def __init__(self, observation):
        self.observation = observation

    def observe_h2d_window(self, round_id):
        assert round_id == self.observation.round_id
        return self.observation

    def drain_h2d_timing_samples(self):
        return []


class _FakeGrantRuntime:
    def __init__(self):
        self.state = SimpleNamespace(
            slack_source="history_h2d",
            outstanding_epoch=None,
            predicted_slack_us=0.0,
            overlap_window_end_us=123,
        )
        self.calls = 0

    def state_for(self, *key):
        return self.state

    def overlap_grants(self, keys):
        self.calls += 1
        return []


def _target_runtime_for_observation(observation):
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    runtime.config = SimpleNamespace(pcie_slack_coexec=True)
    runtime._round_id = observation.round_id
    runtime._h2d_grant_window_ids = {}
    runtime.staging = _FakeStaging(observation)
    runtime.profiler = _FakeProfiler()
    runtime.grant_runtime = _FakeGrantRuntime()
    return runtime


def test_overlap_grant_is_fail_closed_without_active_current_h2d():
    runtime = _target_runtime_for_observation(
        H2DWindowObservation(round_id=8, reason="current_h2d_pending")
    )

    assert runtime.overlap_grants([("r", 1)]) == []
    assert runtime.grant_runtime.calls == 0


def test_overlap_grant_is_armed_from_current_event_budget():
    runtime = _target_runtime_for_observation(
        H2DWindowObservation(
            round_id=8,
            window_id=17,
            active=True,
            remaining_us=2400.0,
            window_end_us=10_000,
            reason="current_h2d_window",
        )
    )

    assert runtime.overlap_grants([("r", 1)]) == []
    assert runtime.grant_runtime.calls == 1
    assert runtime.grant_runtime.state.predicted_slack_us == 2400.0
    assert runtime.grant_runtime.state.overlap_window_end_us == 10_000
    assert runtime._h2d_grant_window_ids[("r", 1)] == 17


def test_same_physical_h2d_window_does_not_rearm_or_extend_budget():
    runtime = _target_runtime_for_observation(
        H2DWindowObservation(
            round_id=8,
            window_id=17,
            active=True,
            remaining_us=2400.0,
            window_end_us=10_000,
            reason="current_h2d_window",
        )
    )
    runtime.overlap_grants([("r", 1)])
    runtime.grant_runtime.state.overlap_window_end_us = 123456
    runtime.staging.observation = H2DWindowObservation(
        round_id=8,
        window_id=17,
        active=True,
        remaining_us=800.0,
        window_end_us=10_000,
        reason="current_h2d_window",
    )

    runtime.overlap_grants([("r", 1)])
    assert runtime.grant_runtime.state.predicted_slack_us == 2400.0
    assert runtime.grant_runtime.state.overlap_window_end_us == 123456


def test_new_physical_h2d_window_rearms_after_outstanding_grant_clears():
    runtime = _target_runtime_for_observation(
        H2DWindowObservation(
            round_id=8,
            window_id=17,
            active=True,
            remaining_us=2400.0,
            window_end_us=10_000,
            reason="current_h2d_window",
        )
    )
    runtime.overlap_grants([("r", 1)])
    runtime.grant_runtime.state.overlap_window_end_us = 123456
    runtime.staging.observation = H2DWindowObservation(
        round_id=8,
        window_id=18,
        active=True,
        remaining_us=1600.0,
        window_end_us=20_000,
        reason="current_h2d_window",
    )

    runtime.overlap_grants([("r", 1)])
    assert runtime.grant_runtime.state.predicted_slack_us == 1600.0
    assert runtime.grant_runtime.state.overlap_window_end_us == 20_000
    assert runtime._h2d_grant_window_ids[("r", 1)] == 18


def test_new_physical_h2d_window_cannot_rearm_an_outstanding_token():
    runtime = _target_runtime_for_observation(
        H2DWindowObservation(
            round_id=8,
            window_id=17,
            active=True,
            remaining_us=2400.0,
            window_end_us=10_000,
            reason="current_h2d_window",
        )
    )
    runtime.overlap_grants([("r", 1)])
    runtime.grant_runtime.state.outstanding_epoch = 7
    runtime.staging.observation = H2DWindowObservation(
        round_id=8,
        window_id=18,
        active=True,
        remaining_us=1600.0,
        window_end_us=20_000,
        reason="current_h2d_window",
    )

    assert runtime.overlap_grants([("r", 1)]) == []
    assert runtime.grant_runtime.calls == 1
    assert runtime.grant_runtime.state.predicted_slack_us == 2400.0
    assert runtime._h2d_grant_window_ids[("r", 1)] == 17


class _RegisteringGrantRuntime:
    def __init__(self):
        self.registered = []
        self.initial_calls = 0

    def register_round(self, **kwargs):
        self.registered.append(kwargs)

    def initial_grants(self, keys):
        self.initial_calls += 1
        return []

    def state_for(self, *key):
        return None


def test_pcie_initial_path_registers_target_baseline_but_has_zero_slack():
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    runtime.config = SimpleNamespace(pcie_slack_coexec=True)
    runtime.grant_runtime = _RegisteringGrantRuntime()
    runtime.states = {}
    runtime._h2d_grant_window_ids = {("r", 1): 7}
    runtime.profiler = _FakeProfiler()
    runtime.slack_profiler = SimpleNamespace(
        snapshot=lambda *args, **kwargs: SimpleNamespace(
            predicted_slack_us=99_000.0, draft_step_ms=1.25
        )
    )
    req = SimpleNamespace(
        rid="r",
        spec_cnt=2,
        origin_input_ids=[1, 2],
        output_ids=[3],
    )

    assert runtime._register_grant_reqs([req], desired_q=4) == []
    assert runtime.grant_runtime.initial_calls == 1
    assert len(runtime.grant_runtime.registered) == 1
    assert ("r", 1) not in runtime._h2d_grant_window_ids
    registered = runtime.grant_runtime.registered[0]
    assert registered["predicted_slack_us"] == 0.0
    assert registered["slack_source"] == "history_h2d"
    assert registered["draft_step_ms"] == 1.25


def test_pcie_slack_startup_rejects_missing_physical_ceiling():
    staging = SimpleNamespace(
        h2d_bandwidth_ceiling_gbps=0.0,
        h2d_bandwidth_ceiling_source="nvml_unavailable",
    )

    with pytest.raises(RuntimeError, match="source=nvml_unavailable"):
        _require_pcie_slack_physical_ceiling(staging)


def test_target_runtime_clear_also_clears_grant_runtime():
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    runtime._poll_pending_seals = lambda wait: None
    runtime._pending_seals = {"r": object()}
    runtime.verifier = SimpleNamespace(discard_layer_prefetch=lambda: None)
    runtime.states = {"r": object()}
    runtime._h2d_grant_window_ids = {("r", 1): 7}
    runtime.history_store = SimpleNamespace(clear=lambda: None)
    cleared = []
    runtime.grant_runtime = SimpleNamespace(clear=lambda: cleared.append(True))

    runtime.clear()

    assert cleared == [True]
    assert runtime._h2d_grant_window_ids == {}
