from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from sglang.srt.speculative.specstream_inproc.ahead_state import ReconcileResult


class RepairKind(str, Enum):
    NONE = "none"
    LOCAL_OVERWRITE = "local_overwrite"
    REPREFILL = "reprefill"


@dataclass(frozen=True)
class RollbackPlan:
    kind: RepairKind
    fork_point: int
    rollback_start: int
    rollback_end: int
    rollback_tokens: int
    repair_tokens: int
    release_physical_slots: bool


class DraftRollbackManager:
    """Plan Draft-only rollback without mutating Target KV ownership.

    STANDALONE V2 gives Target and Draft separate physical KV tensors but a
    shared request-to-token map and allocator.  Consequently an in-process
    Draft rollback must not return suffix slots to that shared allocator.  The
    normal repair extend overwrites the Draft tensor at those locations while
    Target's mapping remains valid.  Physical release is allowed only when a
    caller explicitly proves allocator ownership is exclusive.
    """

    def __init__(
        self,
        *,
        page_size: int,
        shared_allocator: bool = True,
    ) -> None:
        self.page_size = int(page_size)
        self.shared_allocator = bool(shared_allocator)

    def plan(
        self,
        result: ReconcileResult,
        *,
        checkpoint_kv_len: int,
        speculative_end: int,
    ) -> RollbackPlan:
        if result.promotable:
            return RollbackPlan(
                kind=RepairKind.NONE,
                fork_point=result.fork_point,
                rollback_start=result.fork_point,
                rollback_end=result.fork_point,
                rollback_tokens=0,
                repair_tokens=0,
                release_physical_slots=False,
            )

        local_safe = self.page_size == 1 and result.fork_point >= checkpoint_kv_len
        kind = RepairKind.LOCAL_OVERWRITE if local_safe else RepairKind.REPREFILL
        return RollbackPlan(
            kind=kind,
            fork_point=result.fork_point,
            rollback_start=result.fork_point,
            rollback_end=max(result.fork_point, speculative_end),
            rollback_tokens=result.rollback_tokens,
            repair_tokens=result.repair_tokens,
            release_physical_slots=local_safe and not self.shared_allocator,
        )

    def apply_exclusive_release(
        self,
        plan: RollbackPlan,
        release_callback: Optional[Callable[[int, int], None]],
    ) -> None:
        if not plan.release_physical_slots:
            return
        if release_callback is None:
            raise ValueError("exclusive rollback requires a release callback")
        release_callback(plan.rollback_start, plan.rollback_end)
