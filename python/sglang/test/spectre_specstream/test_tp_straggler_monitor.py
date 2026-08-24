import pytest

from sglang.srt.speculative.spectre.specstream.tp_straggler_monitor import (
    TPRankSample,
    TPStragglerMonitor,
)


def _samples(shared_ms, peer_ms, round_id=1):
    return (
        TPRankSample(0, round_id, shared_ms),
        TPRankSample(1, round_id, peer_ms),
    )


def test_monitor_reports_colocated_rank_skew():
    monitor = TPStragglerMonitor(tp_size=2, colocated_rank=0, alpha=1.0)
    monitor.observe(_samples(12.0, 10.0))
    snapshot = monitor.snapshot()
    assert snapshot.samples == 1
    assert snapshot.rank_forward_ms == (12.0, 10.0)
    assert snapshot.rank_skew_ms == pytest.approx(2.0)
    assert snapshot.target_slowdown == 0.0


def test_monitor_uses_best_observed_critical_path_as_online_baseline():
    monitor = TPStragglerMonitor(tp_size=2, colocated_rank=0, alpha=1.0)
    monitor.observe(_samples(10.0, 10.0, round_id=1))
    monitor.observe(_samples(13.0, 12.0, round_id=2))
    snapshot = monitor.snapshot()
    assert snapshot.target_slowdown == pytest.approx(0.3)


def test_partial_rank_sample_does_not_publish_invalid_snapshot():
    monitor = TPStragglerMonitor(tp_size=2, colocated_rank=0)
    monitor.observe((TPRankSample(0, 1, 10.0), None))
    assert monitor.snapshot().samples == 0
