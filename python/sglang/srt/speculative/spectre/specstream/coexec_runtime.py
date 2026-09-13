from __future__ import annotations

import math
import time
from dataclasses import dataclass

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
    draft_step_ms: float = 0.0
    slack_source: str = "target_forward"
    overlap_window_end_us: int | None = None
    issued: int = 0
    acked: int = 0
    outstanding_epoch: int | None = None
    outstanding_tokens: int = 0
    outstanding_grant_state: str = ""
    outstanding_target_baseline_ms: float = 0.0
    outstanding_since: float = 0.0
    last_grant_wait_ms: float = 0.0
    last_decision: GrantDecision | None = None


class TargetGrantRuntime:
    """Target-side bounded grant sequencer.

    At most one grant is outstanding per request.  A Drafter ACK is required
    before the next quantum can be issued, so queued ZMQ messages can never
    collapse into an accidental execution burst. Only explicit Target wait
    may issue a measured multi-token lease; overlap remains one token.
    """

    def __init__(self, controller: GpuGrantController) -> None:
        self.controller = controller
        self._rounds: dict[tuple[str, int], _RoundGrantState] = {}
        self._epochs: dict[str, int] = {}
        self._active_overlap_baselines_ms: list[float] = []
        self._active_target_shapes: set[str] = set()
        self._active_fixed_possible_overlap_shapes: set[str] = set()
        self._active_fixed_overlap_shapes: set[str] = set()
        self._active_possible_overlap = False
        self._active_confirmed_overlap = False

    def overlap_status(self) -> bool | None:
        """Read attribution before record_target_forward clears the interval."""
        if self._active_confirmed_overlap:
            return True
        if self._active_possible_overlap:
            return None
        return False

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
        draft_step_ms: float = 0.0,
        slack_source: str = "target_forward",
        overlap_window_end_us: int | None = None,
    ) -> None:
        if desired_q < 1:
            raise ValueError("desired_q must be positive")
        # A request has only one live SPECTRE round.  Retaining every completed
        # spec_cnt until terminal request release grows without bound on long
        # generations and leaves stale states available to late messages.
        for key in [
            key
            for key in self._rounds
            if key[0] == request_id and key[1] != int(spec_cnt)
        ]:
            self._rounds.pop(key, None)
        predicted_slack_us = float(predicted_slack_us)
        if not math.isfinite(predicted_slack_us) or predicted_slack_us <= 0:
            predicted_slack_us = 0.0
        draft_step_ms = float(draft_step_ms)
        if not math.isfinite(draft_step_ms) or draft_step_ms <= 0:
            draft_step_ms = 0.0
        self._rounds[(request_id, spec_cnt)] = _RoundGrantState(
            request_id=request_id,
            spec_cnt=int(spec_cnt),
            desired_q=int(desired_q),
            target_shape=str(target_shape),
            draft_bs=int(draft_bs),
            draft_ctx_bucket=str(draft_ctx_bucket),
            predicted_slack_us=predicted_slack_us,
            draft_step_ms=draft_step_ms,
            slack_source=str(slack_source),
            overlap_window_end_us=(
                int(overlap_window_end_us)
                if overlap_window_end_us is not None and int(overlap_window_end_us) > 0
                else None
            ),
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
                if state.slack_source == "history_h2d":
                    state.last_decision = GrantDecision(
                        GrantState.TARGET_EXCLUSIVE,
                        "missing_current_h2d_deadline",
                    )
                    return None
                state.overlap_window_end_us = now_us + int(state.predicted_slack_us)
            available_slack_us = min(
                available_slack_us,
                max(state.overlap_window_end_us - now_us, 0.0),
            )
        remaining_tokens = max(state.desired_q - state.issued, 1)
        decision_slack_us = available_slack_us
        reserve_complete_horizon = bool(
            self.controller.fixed_tpc_mode and state.slack_source != "history_h2d"
        )
        if not target_waiting and reserve_complete_horizon:
            # A non-H2D fixed window may reserve the complete next-round
            # SPECTRE sequence.  A current history-H2D observation is a single
            # physical transfer window: admit one token, then query the same
            # or a later CUDA-event window again after its ACK.
            # Re-evaluate after each ACK using the remaining absolute window.
            decision_slack_us = available_slack_us / remaining_tokens
        decision = self.controller.decide(
            target_shape=state.target_shape,
            draft_bs=state.draft_bs,
            draft_ctx_bucket=state.draft_ctx_bucket,
            predicted_slack_us=decision_slack_us,
            draft_step_ms=state.draft_step_ms,
            slack_source=state.slack_source,
            target_waiting=target_waiting,
            deadline_us=deadline_us,
        )
        state.last_decision = decision
        if not decision.allows_draft:
            return None
        effective_deadline_us = deadline_us
        if decision.state is GrantState.SLACK_FILL and effective_deadline_us is None:
            # A measured grant's deadline is the latest safe *launch* time,
            # not the end of the predicted window.  This lets ACK-driven
            # one-token grants pipeline the next SPECTRE candidate sequence
            # while Target verifies the current sequence, without restarting
            # the slack budget after every ACK.
            draft_step_ms = decision.draft_step_ms
            if draft_step_ms is None or state.overlap_window_end_us is None:
                state.last_decision = GrantDecision(
                    GrantState.TARGET_EXCLUSIVE, "missing_overlap_timing"
                )
                return None
            quantum_us = draft_step_ms * 1000.0 + self.controller.guard_us
            reserved_quanta = remaining_tokens if reserve_complete_horizon else 1
            effective_deadline_us = int(
                state.overlap_window_end_us - reserved_quanta * quantum_us
            )
            if effective_deadline_us <= now_us:
                state.last_decision = GrantDecision(
                    GrantState.TARGET_EXCLUSIVE, "overlap_window_exhausted"
                )
                return None
        epoch = self._next_epoch(state.request_id)
        grant_tokens = 1
        if decision.state is GrantState.DRAFT_CATCHUP and state.spec_cnt > 0:
            grant_tokens = self.controller.catchup_token_budget(
                remaining_tokens=remaining_tokens,
                draft_step_ms=float(decision.draft_step_ms or 0.0),
                deadline_us=effective_deadline_us,
                now_us=now_us,
            )
            if grant_tokens <= 0:
                return None
        grant = DraftExecutionGrant(
            request_id=state.request_id,
            spec_cnt=state.spec_cnt,
            grant_epoch=epoch,
            grant_tokens=grant_tokens,
            tpc_low=decision.tpc_low,
            tpc_high=decision.tpc_high,
            deadline_us=effective_deadline_us,
            grant_state=decision.state.value,
        )
        state.issued += grant_tokens
        state.outstanding_epoch = epoch
        state.outstanding_tokens = grant_tokens
        state.outstanding_grant_state = decision.state.value
        if decision.state is GrantState.SLACK_FILL:
            self._active_possible_overlap = True
        if self.controller.fixed_tpc_mode and decision.state is GrantState.SLACK_FILL:
            # Sending a grant means overlap was possible, not that Draft work
            # necessarily launched.  Keep it separate from ACK-confirmed work
            # so an expired/lost grant cannot train a Target-only baseline.
            self._active_fixed_possible_overlap_shapes.add(state.target_shape)
        state.outstanding_target_baseline_ms = (
            float(decision.profile_entry.target_baseline_ms)
            if decision.profile_entry is not None
            and decision.profile_entry.target_baseline_ms > 0
            else 0.0
        )
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
        self._active_target_shapes = set()
        self._active_fixed_possible_overlap_shapes = set()
        self._active_fixed_overlap_shapes = set()
        self._active_possible_overlap = False
        self._active_confirmed_overlap = False
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
            self._active_target_shapes.add(state.target_shape)
            message = self._issue(state, target_waiting=False, deadline_us=None)
            if message is not None:
                messages.append(message)
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
        for target_shape in self._active_target_shapes:
            if not self.controller.fixed_tpc_mode and self.overlap_status() is False:
                self.controller.record_target_only(target_shape)
            self.controller.record_fixed_target_forward(
                target_shape=target_shape,
                elapsed_ms=elapsed_ms,
                possible_overlap=(
                    target_shape in self._active_fixed_possible_overlap_shapes
                ),
                confirmed_overlap=(target_shape in self._active_fixed_overlap_shapes),
            )
        if self._active_overlap_baselines_ms:
            baseline_ms = min(self._active_overlap_baselines_ms)
            slowdown = max(float(elapsed_ms) / baseline_ms - 1.0, 0.0)
            self.controller.record_overlap_slowdown(slowdown)
        self._active_overlap_baselines_ms = []
        self._active_target_shapes = set()
        self._active_fixed_possible_overlap_shapes = set()
        self._active_fixed_overlap_shapes = set()
        self._active_possible_overlap = False
        self._active_confirmed_overlap = False

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
        outstanding_tokens = state.outstanding_tokens
        if not 0 <= ack_tokens <= outstanding_tokens:
            return False
        state.last_grant_wait_ms = max(
            (time.perf_counter() - state.outstanding_since) * 1000.0, 0.0
        )
        completed_grant_state = state.outstanding_grant_state
        completed_target_baseline_ms = state.outstanding_target_baseline_ms
        state.outstanding_epoch = None
        state.outstanding_tokens = 0
        state.outstanding_grant_state = ""
        state.outstanding_target_baseline_ms = 0.0
        state.outstanding_since = 0.0
        # Expiry, PAUSE, early terminal output or re-prefill can finish only
        # a prefix of the lease. Return its unused token budget exactly once.
        state.issued = max(
            state.issued - (outstanding_tokens - ack_tokens), state.acked
        )
        if ack_tokens == 0:
            # A Drafter can explicitly defer a re-prefill grant that was
            # calibrated only for decode.  It consumed no token budget, so the
            # next DRAFT_CATCHUP quantum must still be issuable.
            return True
        measured_step_ms = float(message.draft_step_ms or 0.0)
        if (
            str(message.grant_state or "") != "PREFILL_COMPLETE"
            and math.isfinite(measured_step_ms)
            and measured_step_ms > 0
        ):
            if state.draft_step_ms <= 0:
                state.draft_step_ms = measured_step_ms
            else:
                # React immediately to a slower step, and decay cautiously
                # after a faster observation so later grants do not use an
                # optimistic in-round estimate.
                ema_ms = 0.8 * state.draft_step_ms + 0.2 * measured_step_ms
                state.draft_step_ms = max(measured_step_ms, ema_ms)
        if completed_grant_state == GrantState.SLACK_FILL.value:
            self._active_confirmed_overlap = True
        if (
            self.controller.fixed_tpc_mode
            and completed_grant_state == GrantState.SLACK_FILL.value
        ):
            # A sent grant is not evidence of interference: it may expire in
            # transit or be explicitly deferred.  Only a successful one-token
            # ACK proves that fixed-TPC Draft work actually executed during
            # this Target-forward observation.
            self._active_fixed_overlap_shapes.add(state.target_shape)
        if (
            completed_grant_state == GrantState.SLACK_FILL.value
            and completed_target_baseline_ms > 0.0
        ):
            # A profile entry becomes an interference sample only after the
            # Drafter confirms that one token actually ran.  Expired or
            # deferred grants must leave this Target round as Target-only.
            self._active_overlap_baselines_ms.append(completed_target_baseline_ms)
        state.acked += ack_tokens
        return True

    def pause_messages(self, keys: list[tuple[str, int]]) -> list[SpectreRequest]:
        messages = []
        for request_id, spec_cnt in keys:
            state = self._rounds.get((request_id, spec_cnt))
            if state is None:
                continue
            messages.append(self.pause_message(request_id, spec_cnt))
            state.outstanding_epoch = None
            state.outstanding_tokens = 0
            state.outstanding_grant_state = ""
            state.outstanding_target_baseline_ms = 0.0
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

    def clear(self) -> None:
        self._rounds.clear()
        self._epochs.clear()
        self._active_overlap_baselines_ms.clear()
        self._active_target_shapes.clear()
        self._active_fixed_possible_overlap_shapes.clear()
        self._active_fixed_overlap_shapes.clear()
        self._active_possible_overlap = False
        self._active_confirmed_overlap = False

    def state_for(self, request_id: str, spec_cnt: int) -> _RoundGrantState | None:
        return self._rounds.get((request_id, spec_cnt))
