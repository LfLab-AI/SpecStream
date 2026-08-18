from typing import Callable, Optional


def should_retry_missing_drafts(
    failed_count: int,
    batch_size: int,
    retry_fail_ratio: float,
    retry_min_count: int,
) -> bool:
    """Return whether missing remote drafts justify one synchronous retry.

    ``retry_min_count`` counts failed requests, not the total batch size.  This
    keeps retry policy usable for the common batch-size-one experiment.
    """
    if failed_count <= 0 or batch_size <= 0:
        return False
    return failed_count >= max(int(retry_min_count), 1) and (
        failed_count / batch_size >= float(retry_fail_ratio)
    )


def choose_ready_verify_horizon(
    *,
    mode: str,
    requested_q: int,
    batch_size: int,
    no_draft_count: int,
    no_draft_ratio: float,
) -> int:
    """Choose the horizon that the Target can verify in the current round.

    Ordinary mode receives the just-requested draft before constructing verify
    inputs, so an empty ``cur_drafts`` snapshot at scheduler time is expected.
    Parallel mode can only consume drafts already attached to the requests.
    """
    requested_q = max(int(requested_q), 1)
    if mode == "ordinary":
        return requested_q
    if batch_size > 0 and no_draft_count / batch_size > float(no_draft_ratio):
        return 1
    return requested_q


def should_fail_fast_on_draft_timeout(
    *, require_draft: bool, timeout_action: str
) -> bool:
    """Keep serving fallback and calibration fail-fast as separate policies."""
    if timeout_action not in ("fallback", "error"):
        raise ValueError(f"unsupported draft timeout action: {timeout_action}")
    return bool(require_draft and timeout_action == "error")


def finish_normal_decode_bookkeeping(
    *,
    req,
    server_role: str,
    verified_token: int,
    drafts,
    apply_drafts: Optional[Callable] = None,
) -> None:
    """Finish a normal-decode step while preserving remote round identity.

    ``spec_cnt`` is owned by the Target and identifies one Target-to-Drafter
    request/response pair. A remote Drafter can execute several autoregressive
    steps for that pair, so it must not advance ``spec_cnt`` once per token.
    """
    if server_role == "draft":
        req.len_output_ids = len(req.output_ids)
        return

    if apply_drafts is None:
        raise ValueError("Target bookkeeping requires apply_drafts")
    apply_drafts(
        req,
        verified_token=verified_token,
        drafts=drafts,
        skip_d0=True,
    )
    req.spec_cnt += 1
    req.len_output_ids = len(req.output_ids)
