# Copyright (c) SpecStream contributors.
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DraftExecutionGrant:
    epoch: int
    tpc_low: int
    tpc_high: int
    token_budget: int = 1
    deadline_ns: Optional[int] = None
    request_ids: Tuple[str, ...] = ()
    reason: str = ""
    persistent: bool = False

    def validate(self, total_tpcs: Optional[int] = None) -> None:
        if self.epoch < 0:
            raise ValueError(f"grant epoch must be >=0, got {self.epoch}")
        if self.tpc_low < 0:
            raise ValueError(f"tpc_low must be >=0, got {self.tpc_low}")
        if self.tpc_high <= self.tpc_low:
            raise ValueError(
                f"tpc_high must be greater than tpc_low, got "
                f"[{self.tpc_low}, {self.tpc_high})"
            )
        if total_tpcs is not None and self.tpc_high > int(total_tpcs):
            raise ValueError(
                f"TPC range [{self.tpc_low}, {self.tpc_high}) exceeds "
                f"total_tpcs={total_tpcs}"
            )
        if self.token_budget <= 0:
            raise ValueError(
                f"token_budget must be positive, got {self.token_budget}"
            )

    def expired(self, now_ns: Optional[int] = None) -> bool:
        if self.deadline_ns is None:
            return False
        if now_ns is None:
            now_ns = time.monotonic_ns()
        return now_ns >= self.deadline_ns


class DraftGrantGate:
    """Local execution gate for the remote Drafter.

    Safety invariant:
        controlled Draft forward -> active grant -> active TPC range.

    Calibration uses a persistent fixed-TPC spatial grant.  Online mode
    consumes one token per grant.
    """

    def __init__(self, total_tpcs: Optional[int] = None) -> None:
        self.total_tpcs = int(total_tpcs) if total_tpcs is not None else None
        self._grant: Optional[DraftExecutionGrant] = None
        self._remaining_tokens: int = 0
        self._last_epoch: int = -1
        self._inflight: bool = False
        self._completed_steps: int = 0

    @property
    def active_grant(self) -> Optional[DraftExecutionGrant]:
        g = self._grant
        if g is None:
            return None
        if g.expired():
            self.revoke()
            return None
        if not g.persistent and self._remaining_tokens <= 0:
            self.revoke()
            return None
        return g

    @property
    def inflight(self) -> bool:
        return self._inflight

    def install_calibration_grant(
        self,
        tpcs: int,
        *,
        tpc_low: int = 0,
        epoch: int = 0,
    ) -> DraftExecutionGrant:
        tpcs = int(tpcs)
        if tpcs <= 0:
            raise ValueError(f"calibration tpcs must be positive, got {tpcs}")
        grant = DraftExecutionGrant(
            epoch=int(epoch),
            tpc_low=int(tpc_low),
            tpc_high=int(tpc_low) + tpcs,
            token_budget=1,
            reason="calibration_bootstrap",
            persistent=True,
        )
        grant.validate(self.total_tpcs)
        self._grant = grant
        self._remaining_tokens = 1
        self._last_epoch = max(self._last_epoch, grant.epoch)
        return grant

    def install_online_grant(self, grant: DraftExecutionGrant) -> bool:
        grant.validate(self.total_tpcs)
        if self._inflight or grant.expired() or grant.epoch <= self._last_epoch:
            return False
        self._grant = grant
        self._remaining_tokens = grant.token_budget
        self._last_epoch = grant.epoch
        return True

    def try_acquire(
        self,
        request_ids: Sequence[str] | None = None,
    ) -> Optional[DraftExecutionGrant]:
        if self._inflight:
            return None
        grant = self.active_grant
        if grant is None:
            return None

        if grant.request_ids and request_ids:
            req_set = set(str(x) for x in request_ids)
            allowed = set(grant.request_ids)
            if not req_set.issubset(allowed):
                return None

        self._inflight = True
        return grant

    def complete_one_step(self, *, success: bool = True) -> None:
        if not self._inflight:
            raise RuntimeError("complete_one_step called without inflight grant")
        self._inflight = False

        grant = self._grant
        if grant is None:
            return
        if success:
            self._completed_steps += 1
        if grant.persistent:
            return

        self._remaining_tokens -= 1
        if self._remaining_tokens <= 0:
            self.revoke()

    def revoke(self) -> None:
        if self._inflight:
            return
        self._grant = None
        self._remaining_tokens = 0


def normalize_external_grant(obj: Any) -> Optional[DraftExecutionGrant]:
    if obj is None:
        return None
    if isinstance(obj, DraftExecutionGrant):
        return obj

    def read(*names: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            for name in names:
                if name in obj:
                    return obj[name]
            return default
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    epoch = read("grant_epoch", "epoch", default=None)
    low = read("tpc_low", "draft_tpc_low", default=None)
    high = read("tpc_high", "draft_tpc_high", default=None)
    if epoch is None or low is None or high is None:
        return None

    budget = read("grant_tokens", "token_budget", "tokens", default=1)
    req_ids = read("request_ids", default=())
    if req_ids is None:
        req_ids = ()
    elif isinstance(req_ids, str):
        req_ids = (req_ids,)
    else:
        req_ids = tuple(str(x) for x in req_ids)

    return DraftExecutionGrant(
        epoch=int(epoch),
        tpc_low=int(low),
        tpc_high=int(high),
        token_budget=max(1, int(budget)),
        deadline_ns=read("deadline_ns", default=None),
        request_ids=req_ids,
        reason=str(read("reason", default="online")),
        persistent=bool(read("persistent", default=False)),
    )


def call_runtime_first_available(
    runtime: Any,
    method_names: Sequence[str],
    *,
    batch: Any = None,
    request_ids: Sequence[str] | None = None,
) -> Any:
    if runtime is None:
        return None

    for name in method_names:
        fn = getattr(runtime, name, None)
        if fn is None or not callable(fn):
            continue

        attempts = []
        if batch is not None and request_ids is not None:
            attempts += [
                ((batch,), {"request_ids": request_ids}),
                ((), {"batch": batch, "request_ids": request_ids}),
            ]
        if batch is not None:
            attempts += [((batch,), {}), ((), {"batch": batch})]
        if request_ids is not None:
            attempts += [
                ((request_ids,), {}),
                ((), {"request_ids": request_ids}),
            ]
        attempts += [((), {})]

        for args, kwargs in attempts:
            try:
                return fn(*args, **kwargs)
            except TypeError:
                continue

    return None
