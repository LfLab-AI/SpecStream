import pytest

from sglang.srt.speculative.spectre.specstream.tp_straggler_monitor import (
    TPRankSample,
    TPStragglerMonitor,
)


def _samples(shared_ms, peer_ms, round_id=1, *, shape=(), overlap=None):
    return (
        TPRankSample(0, round_id, shared_ms, shape_key=shape, overlap_active=overlap),
        TPRankSample(1, round_id, peer_ms, shape_key=shape, overlap_active=overlap),
    )


def test_monitor_reports_colocated_rank_skew():
    monitor = TPStragglerMonitor(tp_size=2, colocated_rank=0, alpha=1.0)
    monitor.observe(_samples(12.0, 10.0))
    snapshot = monitor.snapshot()
    assert snapshot.samples == 1
    assert snapshot.rank_forward_ms == (12.0, 10.0)
    assert snapshot.rank_skew_ms == pytest.approx(2.0)
    assert snapshot.target_slowdown == 0.0


def test_untagged_global_workload_change_cannot_be_attributed_to_draft():
    monitor = TPStragglerMonitor(tp_size=2, colocated_rank=0, alpha=1.0)
    monitor.observe(_samples(10.0, 10.0, round_id=1))
    monitor.observe(_samples(13.0, 12.0, round_id=2))
    snapshot = monitor.snapshot()
    assert snapshot.target_slowdown == 0.0
    assert not snapshot.baseline_ready


def test_partial_rank_sample_does_not_publish_invalid_snapshot():
    monitor = TPStragglerMonitor(tp_size=2, colocated_rank=0)
    monitor.observe((TPRankSample(0, 1, 10.0), None))
    assert monitor.snapshot().samples == 0


def test_first_jitter_is_discarded_then_same_shape_baseline_warms():
    monitor = TPStragglerMonitor(tp_size=2)
    shape = ("decode", 4, 8)
    for round_id, values in enumerate(
        ((157, 130), (111, 102), (100, 100), (100, 100)), 1
    ):
        monitor.observe(_samples(*values, round_id, shape=shape, overlap=False))
        assert not monitor.snapshot().baseline_ready
    monitor.observe(_samples(100, 100, 5, shape=shape, overlap=False))
    assert monitor.snapshot().baseline_forward_ms == (100.0, 100.0)
    assert monitor.snapshot().baseline_samples == 3


def test_symmetric_batch_growth_uses_an_independent_baseline():
    monitor = TPStragglerMonitor(tp_size=2)
    for round_id in range(1, 6):
        monitor.observe(_samples(100, 100, round_id, shape=("batch", 1), overlap=False))
    for round_id in range(6, 11):
        monitor.observe(_samples(400, 400, round_id, shape=("batch", 4), overlap=False))
        assert monitor.snapshot().target_slowdown == 0.0
    assert monitor.snapshot(("batch", 1)).baseline_forward_ms == (100, 100)
    assert monitor.snapshot(("batch", 4)).baseline_forward_ms == (400, 400)
    assert not monitor.snapshot(("batch", 8)).baseline_ready


def test_only_confirmed_overlap_compares_to_frozen_target_only_baseline():
    monitor = TPStragglerMonitor(tp_size=2)
    for round_id in range(1, 6):
        monitor.observe(_samples(100, 100, round_id, shape=(4, 8), overlap=False))
    monitor.observe(_samples(150, 130, 6, shape=(4, 8), overlap=None))
    assert monitor.snapshot().target_slowdown == 0.0
    assert monitor.snapshot().baseline_forward_ms == (100, 100)
    monitor.observe(_samples(140, 130, 7, shape=(4, 8), overlap=True))
    snapshot = monitor.snapshot()
    assert snapshot.target_slowdown == pytest.approx(0.4)
    assert snapshot.excess_rank_skew_ms == 10
    assert snapshot.baseline_forward_ms == (100, 100)


def test_duplicate_round_and_mismatched_rank_round_are_not_training_samples():
    monitor = TPStragglerMonitor(tp_size=2)
    samples = _samples(100, 100, 1, shape=(4, 8), overlap=False)
    monitor.observe(samples)
    monitor.observe(samples)
    assert monitor.snapshot().samples == 1
    monitor.observe((samples[0], TPRankSample(1, 2, 100, shape_key=(4, 8))))
    assert monitor.snapshot().samples == 1


def test_persistent_target_only_rank_skew_is_subtracted_from_overlap_skew():
    monitor = TPStragglerMonitor(tp_size=2)
    for round_id in range(1, 6):
        monitor.observe(_samples(104, 100, round_id, shape=(4, 8), overlap=False))
    monitor.observe(_samples(104.5, 100, 6, shape=(4, 8), overlap=True))
    assert monitor.snapshot().rank_skew_ms == 4.5
    assert monitor.snapshot().excess_rank_skew_ms == 0.5
