from sglang.srt.speculative.spectre.draft_delivery import (
    choose_ready_verify_horizon,
    should_fail_fast_on_draft_timeout,
    should_retry_missing_drafts,
)


def test_batch_size_one_can_retry_one_missing_draft():
    assert should_retry_missing_drafts(
        failed_count=1,
        batch_size=1,
        retry_fail_ratio=0.5,
        retry_min_count=1,
    )


def test_retry_min_count_counts_failures_not_batch_size():
    assert not should_retry_missing_drafts(
        failed_count=1,
        batch_size=8,
        retry_fail_ratio=0.0,
        retry_min_count=2,
    )
    assert should_retry_missing_drafts(
        failed_count=2,
        batch_size=8,
        retry_fail_ratio=0.25,
        retry_min_count=2,
    )


def test_retry_ratio_boundary_is_inclusive():
    assert should_retry_missing_drafts(
        failed_count=2,
        batch_size=4,
        retry_fail_ratio=0.5,
        retry_min_count=1,
    )


def test_no_retry_without_failures():
    assert not should_retry_missing_drafts(
        failed_count=0,
        batch_size=4,
        retry_fail_ratio=0.0,
        retry_min_count=1,
    )


def test_ordinary_mode_waits_for_requested_q_even_before_draft_is_attached():
    assert choose_ready_verify_horizon(
        mode="ordinary",
        requested_q=5,
        batch_size=1,
        no_draft_count=1,
        no_draft_ratio=0.5,
    ) == 5


def test_parallel_mode_uses_q1_until_pipelined_draft_is_attached():
    assert choose_ready_verify_horizon(
        mode="parallel",
        requested_q=5,
        batch_size=1,
        no_draft_count=1,
        no_draft_ratio=0.5,
    ) == 1


def test_parallel_mode_keeps_requested_q_when_draft_is_ready():
    assert choose_ready_verify_horizon(
        mode="parallel",
        requested_q=5,
        batch_size=1,
        no_draft_count=0,
        no_draft_ratio=0.5,
    ) == 5


def test_timeout_defaults_to_serving_safe_fallback_even_when_draft_is_required():
    assert not should_fail_fast_on_draft_timeout(
        require_draft=True, timeout_action="fallback"
    )


def test_calibration_can_explicitly_request_fail_fast():
    assert should_fail_fast_on_draft_timeout(
        require_draft=True, timeout_action="error"
    )
