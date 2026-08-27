from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class AheadPhase(str, Enum):
    SYNCED = "synced"
    VERIFYING = "verifying"
    AHEAD = "ahead"
    RECONCILING = "reconciling"
    REPAIRING = "repairing"
    READY = "ready"


def longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the number of equal leading tokens in two token sequences."""

    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


@dataclass(frozen=True)
class ReconcileResult:
    request_id: str
    round_id: int
    fork_point: int
    fork_offset: int
    authoritative_tokens: tuple[int, ...]
    speculative_tokens: tuple[int, ...]
    ahead_generated: int
    ahead_reused: int
    ahead_discarded: int
    rollback_tokens: int
    repair_tokens: int
    promotable: bool

    @property
    def needs_repair(self) -> bool:
        return not self.promotable


@dataclass
class AheadRequestState:
    """Three-frontier state for one in-process speculative request.

    ``verification_tokens`` excludes the carried verified token and contains
    only the q candidates currently checked by Target. ``ahead_tokens`` starts
    with the optimistically predicted Target bonus token.  If Target commits
    the verification candidates and that anchor token, all following ahead
    work is based on the authoritative prefix and may be promoted.
    """

    request_id: str
    round_id: int = 0
    committed_len: int = 0
    verify_start: int = 0
    verify_end: int = 0
    ahead_start: int = 0
    ahead_end: int = 0
    checkpoint_kv_len: int = 0
    verification_tokens: list[int] = field(default_factory=list)
    ahead_tokens: list[int] = field(default_factory=list)
    fork_point: int = 0
    ahead_generated: int = 0
    ahead_reused: int = 0
    ahead_discarded: int = 0
    phase: AheadPhase = AheadPhase.SYNCED

    def prepare_round(
        self,
        *,
        committed_len: int,
        verification_tokens: Sequence[int],
        checkpoint_kv_len: int | None = None,
    ) -> None:
        if self.phase in {
            AheadPhase.VERIFYING,
            AheadPhase.AHEAD,
            AheadPhase.RECONCILING,
            AheadPhase.REPAIRING,
        }:
            raise RuntimeError(
                f"request {self.request_id} cannot start a round from {self.phase}"
            )
        if committed_len < 0:
            raise ValueError("committed_len cannot be negative")

        self.round_id += 1
        self.committed_len = committed_len
        self.verify_start = committed_len
        self.verification_tokens = [int(token) for token in verification_tokens]
        self.verify_end = self.verify_start + len(self.verification_tokens)
        self.ahead_start = self.verify_end
        self.ahead_end = self.ahead_start
        self.checkpoint_kv_len = (
            committed_len if checkpoint_kv_len is None else checkpoint_kv_len
        )
        self.ahead_tokens.clear()
        self.fork_point = self.verify_start
        self.phase = AheadPhase.VERIFYING

    def launch_ahead(self, tokens: Sequence[int]) -> None:
        if self.phase != AheadPhase.VERIFYING:
            raise RuntimeError(
                f"request {self.request_id} cannot launch ahead from {self.phase}"
            )
        self.ahead_tokens = [int(token) for token in tokens]
        self.ahead_generated += len(self.ahead_tokens)
        self.ahead_end = self.ahead_start + len(self.ahead_tokens)
        self.phase = AheadPhase.AHEAD

    def reconcile(self, authoritative_tokens: Sequence[int]) -> ReconcileResult:
        if self.phase != AheadPhase.AHEAD:
            raise RuntimeError(
                f"request {self.request_id} cannot reconcile from {self.phase}"
            )
        self.phase = AheadPhase.RECONCILING

        authoritative = [int(token) for token in authoritative_tokens]
        speculative = self.verification_tokens + self.ahead_tokens
        fork_offset = longest_common_prefix(speculative, authoritative)
        self.fork_point = self.verify_start + fork_offset

        # Target normally contributes at most q accepted candidates plus one
        # bonus token.  Matching through the first ahead token proves that the
        # remaining ahead suffix was generated from the correct prefix.
        anchor_end = len(self.verification_tokens) + 1
        promotable = (
            bool(self.ahead_tokens)
            and len(authoritative) >= anchor_end
            and fork_offset == len(authoritative)
            and authoritative == speculative[: len(authoritative)]
        )

        reused = len(self.ahead_tokens) if promotable else 0
        discarded = len(self.ahead_tokens) - reused
        if promotable:
            rollback_tokens = 0
            repair_tokens = 0
            self.phase = AheadPhase.READY
        else:
            rollback_tokens = max(0, len(speculative) - fork_offset)
            repair_tokens = max(0, len(authoritative) - fork_offset)
            self.phase = AheadPhase.REPAIRING

        self.ahead_reused += reused
        self.ahead_discarded += discarded

        return ReconcileResult(
            request_id=self.request_id,
            round_id=self.round_id,
            fork_point=self.fork_point,
            fork_offset=fork_offset,
            authoritative_tokens=tuple(authoritative),
            speculative_tokens=tuple(speculative),
            ahead_generated=len(self.ahead_tokens),
            ahead_reused=reused,
            ahead_discarded=discarded,
            rollback_tokens=rollback_tokens,
            repair_tokens=repair_tokens,
            promotable=promotable,
        )

    def mark_repaired(self, committed_len: int) -> None:
        if self.phase != AheadPhase.REPAIRING:
            raise RuntimeError(
                f"request {self.request_id} cannot finish repair from {self.phase}"
            )
        self.committed_len = int(committed_len)
        self.phase = AheadPhase.READY

    @property
    def reuse_ratio(self) -> float:
        if self.ahead_generated == 0:
            return 0.0
        return self.ahead_reused / self.ahead_generated
