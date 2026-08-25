from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable


@dataclass(frozen=True)
class DraftExecutionGrant:
    """One revocable Draft execution quantum.

    ``desired_q`` remains part of the SPECTRE request.  This object only grants
    permission to execute CUDA work and therefore intentionally carries no
    speculative-horizon semantics.
    """

    request_id: str
    spec_cnt: int
    grant_epoch: int
    grant_tokens: int
    tpc_low: int
    tpc_high: int
    deadline_us: int | None = None
    placement_id: int = 0
    grant_state: str = ""

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id cannot be empty")
        if self.spec_cnt < 0:
            raise ValueError("spec_cnt cannot be negative")
        if self.grant_epoch < 1:
            raise ValueError("grant_epoch must be positive")
        if self.grant_tokens != 1:
            raise ValueError("SpecStream v1 grants exactly one token")
        if self.tpc_low < 0 or self.tpc_high <= self.tpc_low:
            raise ValueError("TPC range must satisfy 0 <= low < high")
        if self.deadline_us is not None and self.deadline_us < 0:
            raise ValueError("deadline_us cannot be negative")
        if self.placement_id < 0:
            raise ValueError("placement_id cannot be negative")

    def expired(self, now_us: int | None = None) -> bool:
        if self.deadline_us is None:
            return False
        if now_us is None:
            now_us = time.monotonic_ns() // 1000
        return int(now_us) >= self.deadline_us


@dataclass(frozen=True)
class GrantApplyResult:
    accepted: bool
    reason: str


class DraftGrantTable:
    """Drafter-side epoch table with fail-closed token accounting."""

    def __init__(self) -> None:
        self._active: dict[str, DraftExecutionGrant] = {}
        self._latest_epoch: dict[str, int] = {}
        self._latest_spec_cnt: dict[str, int] = {}

    def apply(self, grant: DraftExecutionGrant) -> GrantApplyResult:
        latest_spec_cnt = self._latest_spec_cnt.get(grant.request_id, -1)
        latest_epoch = self._latest_epoch.get(grant.request_id, 0)
        if grant.spec_cnt < latest_spec_cnt:
            return GrantApplyResult(False, "stale_spec_cnt")
        if grant.spec_cnt == latest_spec_cnt and grant.grant_epoch <= latest_epoch:
            return GrantApplyResult(False, "stale_epoch")
        self._latest_spec_cnt[grant.request_id] = grant.spec_cnt
        self._latest_epoch[grant.request_id] = grant.grant_epoch
        self._active[grant.request_id] = grant
        return GrantApplyResult(True, "accepted")

    def pause(
        self,
        request_id: str,
        *,
        spec_cnt: int | None = None,
        grant_epoch: int | None = None,
    ) -> bool:
        current = self._active.get(request_id)
        if current is None:
            return False
        if spec_cnt is not None and spec_cnt < current.spec_cnt:
            return False
        if grant_epoch is not None and grant_epoch < current.grant_epoch:
            return False
        self._active.pop(request_id, None)
        return True

    def release(self, request_id: str) -> None:
        self._active.pop(request_id, None)
        self._latest_epoch.pop(request_id, None)
        self._latest_spec_cnt.pop(request_id, None)

    def active(
        self,
        request_id: str,
        *,
        spec_cnt: int | None = None,
        now_us: int | None = None,
    ) -> DraftExecutionGrant | None:
        grant = self._active.get(request_id)
        if grant is None:
            return None
        if spec_cnt is not None and grant.spec_cnt != spec_cnt:
            return None
        if grant.expired(now_us):
            return None
        return grant

    def pop_expired(
        self,
        request_id: str,
        *,
        spec_cnt: int | None = None,
        now_us: int | None = None,
    ) -> DraftExecutionGrant | None:
        """Remove and return one expired grant so Drafter can ACK deferral.

        Expiry is a normal fail-closed outcome, not a lost control message.
        Keeping the expired entry until the scheduler explicitly reaps it lets
        Drafter send a zero-token ACK and release Target's outstanding epoch.
        """
        grant = self._active.get(request_id)
        if grant is None:
            return None
        if spec_cnt is not None and grant.spec_cnt != spec_cnt:
            return None
        if not grant.expired(now_us):
            return None
        self._active.pop(request_id, None)
        return grant

    def active_for(
        self,
        request_ids: Iterable[tuple[str, int]],
        *,
        now_us: int | None = None,
    ) -> dict[str, DraftExecutionGrant]:
        result = {}
        for request_id, spec_cnt in request_ids:
            grant = self.active(request_id, spec_cnt=spec_cnt, now_us=now_us)
            if grant is not None:
                result[request_id] = grant
        return result

    def consume_one(
        self,
        request_id: str,
        *,
        spec_cnt: int,
        grant_epoch: int,
    ) -> bool:
        # The deadline authorizes launch.  Callers check ``active`` immediately
        # before submitting CUDA work; a valid launch may finish after that
        # deadline and must still consume exactly the epoch it launched under.
        grant = self._active.get(request_id)
        if (
            grant is None
            or grant.spec_cnt != spec_cnt
            or grant.grant_epoch != grant_epoch
        ):
            return False
        # v1 grants are exactly one token, so successful consumption always
        # removes the entry before another CUDA forward can be launched.
        self._active.pop(request_id, None)
        return True

    def __len__(self) -> int:
        return len(self._active)
