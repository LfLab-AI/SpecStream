import pytest

from sglang.srt.speculative.specstream_inproc.ahead_state import (
    AheadPhase,
    AheadRequestState,
    longest_common_prefix,
)
from sglang.srt.speculative.specstream_inproc.resource_controller import (
    CoexecutionMode,
    InProcessResourceController,
)
from sglang.srt.speculative.specstream_inproc.rollback_manager import (
    DraftRollbackManager,
    RepairKind,
)


def _reconcile(target_tokens):
    state = AheadRequestState("request-0")
    state.prepare_round(
        committed_len=100,
        verification_tokens=[11, 12, 13, 14],
        checkpoint_kv_len=100,
    )
    state.launch_ahead([15, 16, 17, 18, 19])
    return state, state.reconcile(target_tokens)


def test_longest_common_prefix():
    assert longest_common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert longest_common_prefix([1, 2], [1, 2, 3]) == 2
    assert longest_common_prefix([], [1]) == 0


def test_all_candidates_and_bonus_match_promotes_ahead():
    state, result = _reconcile([11, 12, 13, 14, 15])

    assert result.promotable
    assert result.fork_point == 105
    assert result.ahead_reused == 5
    assert result.ahead_discarded == 0
    assert result.rollback_tokens == 0
    assert state.phase == AheadPhase.READY


@pytest.mark.parametrize(
    ("target_tokens", "fork_offset", "repair_tokens"),
    [
        ([91], 0, 1),  # reject the first candidate
        ([11, 12, 91], 2, 1),  # reject in the middle
        ([11, 12, 13, 14, 91], 4, 1),  # q accepted, bonus diverges
    ],
)
def test_divergence_discards_ahead_and_repairs(
    target_tokens, fork_offset, repair_tokens
):
    state, result = _reconcile(target_tokens)

    assert not result.promotable
    assert result.fork_offset == fork_offset
    assert result.ahead_reused == 0
    assert result.ahead_discarded == 5
    assert result.repair_tokens == repair_tokens
    assert state.phase == AheadPhase.REPAIRING


def test_shared_allocator_rollback_uses_overwrite_not_free():
    state, result = _reconcile([11, 12, 90])
    manager = DraftRollbackManager(page_size=1, shared_allocator=True)
    plan = manager.plan(
        result,
        checkpoint_kv_len=state.checkpoint_kv_len,
        speculative_end=state.ahead_end,
    )

    assert plan.kind == RepairKind.LOCAL_OVERWRITE
    assert not plan.release_physical_slots
    assert plan.rollback_start == result.fork_point


def test_page_size_greater_than_one_requires_reprefill():
    state, result = _reconcile([11, 90])
    manager = DraftRollbackManager(page_size=16, shared_allocator=True)
    plan = manager.plan(
        result,
        checkpoint_kv_len=state.checkpoint_kv_len,
        speculative_end=state.ahead_end,
    )

    assert plan.kind == RepairKind.REPREFILL


def test_auto_controller_falls_back_when_reuse_is_low():
    controller = InProcessResourceController(
        configured_mode="auto",
        ahead_depth=2,
        min_reuse_ratio=0.5,
        target_slowdown_budget=0.1,
        warmup_rounds=2,
    )
    assert controller.choose().mode == CoexecutionMode.AHEAD_FREE

    controller.observe(reuse_ratio=0.0, target_slowdown=0.0)
    controller.observe(reuse_ratio=0.0, target_slowdown=0.0)
    action = controller.choose()
    assert action.mode == CoexecutionMode.SERIAL
    assert action.ahead_depth == 0


def test_auto_controller_caps_depth_to_verify_window():
    controller = InProcessResourceController(
        configured_mode="auto",
        ahead_depth=4,
        min_reuse_ratio=0.0,
        target_slowdown_budget=1.0,
        ema_alpha=1.0,
        warmup_rounds=0,
        guard_ms=0.0,
    )
    controller.observe(
        reuse_ratio=1.0,
        target_slowdown=0.0,
        ahead_depth=4,
        verify_ms=3.0,
        ahead_ms=5.0,
    )

    assert controller.choose().ahead_depth == 2
