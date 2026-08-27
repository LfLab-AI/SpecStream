from sglang.srt.speculative.spectre.specstream.draft_load_tracker import (
    DraftLoadTracker,
)


def test_tracker_reports_deadline_pressure_and_missing_ratio():
    tracker = DraftLoadTracker(window_size=8, alpha=1.0)
    tracker.record_result(
        elapsed_ms=800,
        timeout_ms=1000,
        missing_count=2,
        total_count=8,
    )
    snapshot = tracker.snapshot()
    assert snapshot.samples == 1
    assert snapshot.rtt_ema_ms == 800
    assert snapshot.pressure_p95 == 0.8
    assert snapshot.timeout_rate == 1.0
    assert snapshot.missing_ratio_ema == 0.25
    assert snapshot.pending_p95 == 8


def test_reject_rate_is_tracked_without_fabricating_rtt():
    tracker = DraftLoadTracker(window_size=4)
    tracker.record_reject()
    snapshot = tracker.snapshot()
    assert snapshot.samples == 0
    assert snapshot.rtt_p95_ms == 0
    assert snapshot.reject_rate == 1.0
