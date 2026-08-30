from __future__ import annotations

from dataclasses import dataclass
import time

from sglang.srt.speculative.spectre.specstream.gpu_grant import (
    DraftExecutionGrant,
)
from sglang.srt.speculative.spectre.specstream.gpu_grant_controller import (
    GpuGrantController,
    GrantDecision,
    GrantState,
)
from sglang.srt.speculative.spectre.spectre_protocol import (
    SpectreAction,
    SpectreRequest,
    SpecType,
)


@dataclass
class _RoundGrantState:
    request_id: str
    spec_cnt: int
    desired_q: int
    target_shape: str
    draft_bs: int
    draft_ctx_bucket: str
    predicted_slack_us: float
    slack_source: str = "target_forward"
    overlap_window_end_us: int | None = None
    issued: int = 0
    acked: int = 0
    outstanding_epoch: int | None = None
    outstanding_since: float = 0.0
    last_grant_wait_ms: float = 0.0
    last_decision: GrantDecision | None = None


class TargetGrantRuntime:
    """Target-side one-token grant sequencer.

    At most one grant is outstanding per request.  A Drafter ACK is required
    before the next quantum can be issued, so queued ZMQ messages can never
    collapse into an accidental multi-token execution burst.
    """

    def __init__(self, controller: GpuGrantController) -> None:
        self.controller = controller
        self._rounds: dict[tuple[str, int], _RoundGrantState] = {}
        self._epochs: dict[str, int] = {}
        self._active_overlap_baselines_ms: list[float] = []

    def register_round(
        self,
        *,
        request_id: str,
        spec_cnt: int,
        desired_q: int,
        target_shape: str,
        draft_bs: int,
        draft_ctx_bucket: str,
        predicted_slack_us: float,
        slack_source: str = "target_forward",
    ) -> None:
        if desired_q < 1:
            raise ValueError("desired_q must be positive")
        self._rounds[(request_id, spec_cnt)] = _RoundGrantState(
            request_id=request_id,
            spec_cnt=int(spec_cnt),
            desired_q=int(desired_q),
            target_shape=str(target_shape),
            draft_bs=int(draft_bs),
            draft_ctx_bucket=str(draft_ctx_bucket),
            predicted_slack_us=max(float(predicted_slack_us), 0.0),
            slack_source=str(slack_source),
        )

    def _next_epoch(self, request_id: str) -> int:
        epoch = self._epochs.get(request_id, 0) + 1
        self._epochs[request_id] = epoch
        return epoch

    def _issue(
        self,
        state: _RoundGrantState,
        *,
        target_waiting: bool,
        deadline_us: int | None,
    ) -> SpectreRequest | None:
        if state.outstanding_epoch is not None or state.issued >= state.desired_q:
            return None
        now_us = time.monotonic_ns() // 1000
        available_slack_us = state.predicted_slack_us
        if not target_waiting:
            if state.overlap_window_end_us is None:
                state.overlap_window_end_us = now_us + int(state.predicted_slack_us)
            available_slack_us = max(state.overlap_window_end_us - now_us, 0.0)
        remaining_tokens = max(state.desired_q - state.issued, 1)
        decision_slack_us = available_slack_us
        if not target_waiting and state.slack_source == "history_h2d":
            # PCIe-Slack is admitted only when the measured window can hold the
            # complete next-round SPECTRE sequence, not merely its first token.
            # Re-evaluate after each ACK using the remaining absolute window.
            decision_slack_us = available_slack_us / remaining_tokens
        decision = self.controller.decide(
            target_shape=state.target_shape,
            draft_bs=state.draft_bs,
            draft_ctx_bucket=state.draft_ctx_bucket,
            predicted_slack_us=decision_slack_us,
            slack_source=state.slack_source,
            target_waiting=target_waiting,
            deadline_us=deadline_us,
        )
        state.last_decision = decision
        if not decision.allows_draft:
            return None
        effective_deadline_us = deadline_us
        if (
            decision.state is GrantState.SLACK_FILL
            and effective_deadline_us is None
            and not (
                self.controller.calibration_tpcs > 0
                and self.controller.calibration_allow_overlap
            )
        ):
            # A measured grant's deadline is the latest safe *launch* time,
            # not the end of the predicted window.  This lets ACK-driven
            # one-token grants pipeline the next SPECTRE candidate sequence
            # while Target verifies the current sequence, without restarting
            # the slack budget after every ACK.
            entry = decision.profile_entry
            if entry is None or state.overlap_window_end_us is None:
                state.last_decision = GrantDecision(
                    GrantState.TARGET_EXCLUSIVE, "missing_overlap_profile_entry"
                )
                return None
            quantum_us = entry.draft_step_ms * 1000.0 + self.controller.guard_us
            reserved_quanta = (
                remaining_tokens if state.slack_source == "history_h2d" else 1
            )
            effective_deadline_us = int(
                state.overlap_window_end_us - reserved_quanta * quantum_us
            )
            if effective_deadline_us <= now_us:
                state.last_decision = GrantDecision(
                    GrantState.TARGET_EXCLUSIVE, "overlap_window_exhausted"
                )
                return None
        # Calibration overlap is an explicit fixed-quota experiment rather
        # than an online slack decision.  In particular, the first measured
        # round has no slack history and therefore predicts zero microseconds.
        # Giving that round the generic one-microsecond fallback deadline makes
        # the grant expire in transit and leaves Target waiting forever for an
        # ACK that Drafter can never produce.  Keep the one-token boundary, but
        # let this explicitly requested calibration grant remain valid until it
        # is consumed or PAUSE revokes it.
        epoch = self._next_epoch(state.request_id)
        grant = DraftExecutionGrant(
            request_id=state.request_id,
            spec_cnt=state.spec_cnt,
            grant_epoch=epoch,
            grant_tokens=1,
            tpc_low=decision.tpc_low,
            tpc_high=decision.tpc_high,
            deadline_us=effective_deadline_us,
            grant_state=decision.state.value,
        )
        state.issued += 1
        state.outstanding_epoch = epoch
        state.outstanding_since = time.perf_counter()
        return SpectreRequest(
            request_id=grant.request_id,
            spec_cnt=grant.spec_cnt,
            action=SpectreAction.GRANT,
            spec_type=SpecType.DRAFT_REQUEST,
            grant_epoch=grant.grant_epoch,
            grant_tokens=grant.grant_tokens,
            tpc_low=grant.tpc_low,
            tpc_high=grant.tpc_high,
            deadline_us=grant.deadline_us,
            placement_id=grant.placement_id,
            grant_state=grant.grant_state,
        )

    def initial_grants(self, keys: list[tuple[str, int]]) -> list[SpectreRequest]:
        messages = []
        self._active_overlap_baselines_ms = []
        for key in keys:
            state = self._rounds.get(key)
            # Initial Draft prefill has no calibrated single-token slack model;
            # keep it Target-exclusive until the Target enters an explicit wait.
            if state is None:
                continue
            if state.spec_cnt <= 0:
                state.last_decision = GrantDecision(
                    GrantState.TARGET_EXCLUSIVE, "initial_prefill_wait_only"
                )
                continue
            message = self._issue(state, target_waiting=False, deadline_us=None)
            if message is not None:
                messages.append(message)
                entry = state.last_decision.profile_entry
                if entry is not None and entry.target_baseline_ms > 0:
                    self._active_overlap_baselines_ms.append(entry.target_baseline_ms)
        return messages

    def overlap_grants(self, keys: list[tuple[str, int]]) -> list[SpectreRequest]:
        """Issue the next ACK-gated token while Target forward is still active."""

        messages = []
        for key in keys:
            state = self._rounds.get(key)
            if state is None or state.spec_cnt <= 0:
                continue
            message = self._issue(
                state,
                target_waiting=False,
                deadline_us=None,
            )
            if message is not None:
                messages.append(message)
        return messages

    def record_target_forward(self, elapsed_ms: float) -> None:
        """Latch TARGET_EXCLUSIVE if live overlap exceeds its baseline budget."""
        if not self._active_overlap_baselines_ms:
            return
        baseline_ms = min(self._active_overlap_baselines_ms)
        slowdown = max(float(elapsed_ms) / baseline_ms - 1.0, 0.0)
        self.controller.record_overlap_slowdown(slowdown)
        self._active_overlap_baselines_ms = []

    def waiting_grants(
        self,
        keys: list[tuple[str, int]],
        *,
        deadline_us: int,
    ) -> list[SpectreRequest]:
        messages = []
        for key in keys:
            state = self._rounds.get(key)
            if state is None:
                continue
            message = self._issue(
                state, target_waiting=True, deadline_us=int(deadline_us)
            )
            if message is not None:
                messages.append(message)
        return messages

    def acknowledge(self, message: SpectreRequest) -> bool:
        if message.action is not SpectreAction.GRANT_ACK:
            return False
        key = (str(message.request_id), int(message.spec_cnt or 0))
        state = self._rounds.get(key)
        if state is None or state.outstanding_epoch is None:
            return False
        if int(message.grant_epoch or 0) != state.outstanding_epoch:
            return False
        ack_tokens = int(message.grant_tokens or 0)
        if ack_tokens not in (0, 1):
            return False
        state.last_grant_wait_ms = max(
            (time.perf_counter() - state.outstanding_since) * 1000.0, 0.0
        )
        state.outstanding_epoch = None
        state.outstanding_since = 0.0
        if ack_tokens == 0:
            # A Drafter can explicitly defer a re-prefill grant that was
            # calibrated only for decode.  It consumed no token budget, so the
            # next DRAFT_CATCHUP quantum must still be issuable.
            state.issued = max(state.issued - 1, state.acked)
            return True
        state.acked += 1
        return True

    def pause_messages(self, keys: list[tuple[str, int]]) -> list[SpectreRequest]:
        messages = []
        for request_id, spec_cnt in keys:
            state = self._rounds.get((request_id, spec_cnt))
            if state is None:
                continue
            messages.append(self.pause_message(request_id, spec_cnt))
            state.outstanding_epoch = None
            state.outstanding_since = 0.0
        return messages

    def pause_message(self, request_id: str, spec_cnt: int) -> SpectreRequest:
        return SpectreRequest(
            request_id=request_id,
            spec_cnt=spec_cnt,
            action=SpectreAction.PAUSE,
            spec_type=SpecType.DRAFT_REQUEST,
            grant_epoch=self._epochs.get(request_id, 0),
        )

    def release_request(self, request_id: str) -> None:
        for key in [key for key in self._rounds if key[0] == request_id]:
            self._rounds.pop(key, None)
        self._epochs.pop(request_id, None)

    def state_for(self, request_id: str, spec_cnt: int) -> _RoundGrantState | None:
        return self._rounds.get((request_id, spec_cnt))
