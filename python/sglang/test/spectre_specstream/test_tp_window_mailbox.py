import time
from dataclasses import dataclass, replace

import pytest

from sglang.srt.speculative.spectre.specstream.gpu_grant import DraftExecutionGrant, DraftGrantTable
from sglang.srt.speculative.spectre.specstream.tp_window_mailbox import TPWindowMailbox


@dataclass(frozen=True)
class Window:
    round_id: int = 4
    window_id: int = 1
    window_end_us: int = 0
    active: bool = True
    remaining_us: float = 0.0
    reason: str = ""


def test_joint_window_requires_both_ranks_and_preserves_absolute_deadline(tmp_path):
    pytest.importorskip("fcntl")
    left = TPWindowMailbox(tmp_path, 0, 2)
    right = TPWindowMailbox(tmp_path, 1, 2)
    try:
        now = time.monotonic_ns() // 1000
        a = Window(window_end_us=now + 100000)
        b = Window(window_id=2, window_end_us=now + 80000)
        assert not left.intersect(a).active
        right.publish(b)
        merged = left.intersect(a)
        assert merged.active and merged.window_end_us == b.window_end_us
        assert 0 < merged.remaining_us <= 80000
        assert left.intersect(a).window_id == merged.window_id
        right.publish(replace(b, round_id=3))
        assert not left.intersect(a).active
        right.publish(replace(b, window_end_us=now - 1))
        assert not left.intersect(a).active
        right.clear()
        assert not left.intersect(a).active
    finally:
        left.close()
        right.close()


def test_rank_control_clock_does_not_replace_real_launch_expiry():
    table = DraftGrantTable()
    deadline = time.monotonic_ns() // 1000 - 100
    grant = DraftExecutionGrant("r", 0, 1, 1, 0, 4, deadline_us=deadline)
    table.apply(grant)
    table.control_now_us = deadline - 1
    assert table.active("r") is grant  # identical control decisions on both ranks
    assert grant.expired()  # launch readiness must still reject the stale lease
    table.control_now_us = deadline + 1
    assert table.pop_expired("r") is grant
    assert table.pop_expired("r") is None


def test_receive_issues_catchup_when_cuda_completes_between_poll_and_wait(monkeypatch):
    from types import SimpleNamespace
    from sglang.srt.speculative.spectre.verifier import spectre_target_scheduler_mixin as module
    from sglang.srt.speculative.spectre.spectre_protocol import SpectreAction

    now = [0.0]
    queries = [0]
    granted = [False]
    waits = []

    def completed():
        queries[0] += 1
        return queries[0] > 2

    def wait(timeout):
        waits.append(timeout)
        now[0] += timeout

    def catchup(keys, deadline_us):
        granted[0] = int(now[0] * 1e6) < deadline_us
        return []

    monkeypatch.setattr(module, "time", SimpleNamespace(
        perf_counter=lambda: now[0], monotonic_ns=lambda: int(now[0] * 1e9)))
    scheduler = object.__new__(module.SchedulerSpectreTargetMixin)
    scheduler.tp_rank = 0
    runtime = SimpleNamespace(grant_runtime=object(), overlap_grants=lambda keys: [],
                              waiting_grants=catchup)
    scheduler._get_specstream_runtime = lambda: runtime
    scheduler._data_ready = SimpleNamespace(wait=wait)
    message = SimpleNamespace(action=SpectreAction.DRAFT, request_id="r", spec_cnt=1)
    scheduler._drain_msg_buffer = lambda: [message] if granted[0] else []
    result = scheduler._collect_draft_messages(
        {"r"}, {"r": 1}, 5.0, target_forward_done_event=SimpleNamespace(query=completed))
    assert result == [message]
    assert waits and max(waits) <= 0.001
