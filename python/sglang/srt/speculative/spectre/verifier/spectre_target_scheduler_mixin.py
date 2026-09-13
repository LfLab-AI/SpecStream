import logging
import os
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spectre.draft_delivery import (
    choose_ready_verify_horizon,
    should_fail_fast_on_draft_timeout,
)
from sglang.srt.speculative.spectre.spectre_protocol import (
    SpectreAction,
    SpectreRequest,
    SpecType,
)
from sglang.srt.speculative.spectre.spectre_protocol import (
    is_health_check_req as _is_health_check,
)
from sglang.srt.speculative.spectre.specstream.background_grant_pump import (
    BackgroundGrantPump,
)
from sglang.srt.utils import DynamicGradMode, broadcast_pyobj

logger = logging.getLogger(__name__)


def _spectre_now_us() -> float:
    return time.time() * 1e6


def _draft_needs_full_context(req: Req, *, is_half_open: bool = False) -> bool:
    """Return whether this round must bootstrap or repair Drafter state.

    ``spec_cnt`` is a wire round identifier, not proof that the remote Drafter
    has state. It can advance during q=1 fallback rounds even when no remote
    request was sent, so successful bootstrap must be tracked explicitly.
    """

    return bool(
        is_half_open
        or int(getattr(req, "spec_cnt", 0) or 0) <= 0
        or not bool(getattr(req, "spectre_draft_initialized", False))
        or bool(getattr(req, "spectre_force_full_draft_context", False))
    )


def _mark_draft_context_pending(req: Req, needs_full_context: bool) -> None:
    req.spectre_full_context_pending = bool(needs_full_context)


def _mark_draft_context_synced(req: Req) -> None:
    req.spectre_draft_initialized = True
    req.spectre_force_full_draft_context = False
    req.spectre_full_context_pending = False


def _mark_draft_context_unsynced(req: Req) -> None:
    req.spectre_force_full_draft_context = True
    req.spectre_full_context_pending = False


class DraftCircuitBreaker:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 30,
        cooldown_rounds: int = 100,
        tp_rank: int = 0,
    ):
        self.state = self.CLOSED
        self.consecutive_failures = 0
        self.failure_threshold = failure_threshold
        self.cooldown_rounds = cooldown_rounds
        self.rounds_in_open = 0
        self.tp_rank = tp_rank

    def should_send(self) -> bool:
        if self.state == self.CLOSED:
            return True
        if self.state == self.OPEN:
            self.rounds_in_open += 1
            if self.rounds_in_open >= self.cooldown_rounds:
                self.state = self.HALF_OPEN
                if self.tp_rank == 0:
                    logger.info(
                        "\033[34m [CircuitBreaker] OPEN -> HALF_OPEN, probing draft... \033[0m"
                    )
                return True
            return False
        if self.state == self.HALF_OPEN:
            return True
        return False

    def record_success(self):
        if self.state != self.CLOSED and self.tp_rank == 0:
            logger.info(
                f"\033[34m [CircuitBreaker] {self.state} -> CLOSED, draft recovered \033[0m"
            )
        self.consecutive_failures = 0
        self.state = self.CLOSED
        self.rounds_in_open = 0

    def record_failure(self):
        self.consecutive_failures += 1
        if self.state == self.HALF_OPEN:
            self.state = self.OPEN
            self.rounds_in_open = 0
            if self.tp_rank == 0:
                logger.info(
                    "\033[34m [CircuitBreaker] HALF_OPEN -> OPEN, probe failed \033[0m"
                )
        elif self.consecutive_failures >= self.failure_threshold:
            if self.state != self.OPEN and self.tp_rank == 0:
                logger.info(
                    f"\033[34m [CircuitBreaker] CLOSED -> OPEN after "
                    f"{self.consecutive_failures} consecutive timeouts, "
                    f"cooldown {self.cooldown_rounds} rounds \033[0m"
                )
            self.state = self.OPEN
            self.rounds_in_open = 0


class SchedulerSpectreTargetMixin:
    def _init_draft_recv_infra(self):
        self._recv_timeout_s = (
            float(
                os.environ.get(
                    "SPECTRE_RECV_TIMEOUT_MS",
                    str(self.server_args.spectre_recv_timeout_ms),
                )
            )
            / 1000.0
        )
        self._initial_recv_timeout_s = (
            float(
                os.environ.get(
                    "SPECTRE_INITIAL_RECV_TIMEOUT_MS",
                    str(self.server_args.spectre_initial_recv_timeout_ms),
                )
            )
            / 1000.0
        )
        self._msg_buffer: List[SpectreRequest] = []
        self._msg_lock = threading.Lock()
        self._data_ready = threading.Event()
        self._bg_running = True
        self._spectre_flush_at_us = 0.0
        self._accept_reject_messages = True

        failure_threshold = int(
            os.environ.get(
                "SPECTRE_FAILURE_THRESHOLD",
                str(self.server_args.spectre_failure_threshold),
            )
        )
        cooldown_rounds = int(
            os.environ.get(
                "SPECTRE_COOLDOWN_ROUNDS",
                str(self.server_args.spectre_cooldown_rounds),
            )
        )
        self.draft_circuit_breaker = DraftCircuitBreaker(
            failure_threshold=failure_threshold,
            cooldown_rounds=cooldown_rounds,
            tp_rank=self.tp_rank,
        )
        if self.tp_rank == 0:
            logger.info(
                "SPECTRE draft circuit breaker: failure_threshold=%d, "
                "cooldown_rounds=%d",
                failure_threshold,
                cooldown_rounds,
            )

        if self.tp_size == 1 or self.tp_rank == 0:
            self._bg_recv_thread = threading.Thread(
                target=self._bg_recv_loop, daemon=True, name="draft_recv_bg"
            )
            self._bg_recv_thread.start()

    def _bg_recv_loop(self):
        while self._bg_running:
            try:
                if (
                    hasattr(self, "zmq_communicator")
                    and self.zmq_communicator is not None
                ):
                    msgs = self.zmq_communicator.recv_all_objs()
                    if msgs:
                        with self._msg_lock:
                            self._msg_buffer.extend(msgs)
                            self._data_ready.set()
                    else:
                        time.sleep(0.0005)
                else:
                    time.sleep(0.0005)
            except Exception as e:
                logger.error(f"\033[34m [Target][BgRecv] Error: {e} \033[0m")
                time.sleep(0.0005)

    def _drain_msg_buffer(self) -> List[SpectreRequest]:
        with self._msg_lock:
            msgs = list(self._msg_buffer)
            self._msg_buffer.clear()
            self._data_ready.clear()
        return msgs

    def _drain_grant_acks_during_forward(self, keys):
        """Extract only ACKs; leave Draft/REJECT/context messages for TP receive."""
        acknowledgements = []
        completed = set()
        with self._msg_lock:
            retained = []
            for msg in self._msg_buffer:
                if msg.action == SpectreAction.GRANT_ACK:
                    acknowledgements.append(msg)
                    continue
                retained.append(msg)
                if msg.action == SpectreAction.REJECT:
                    completed.update(keys)
                elif msg.action in (SpectreAction.DRAFT, SpectreAction.NEED_CONTEXT):
                    key = (str(msg.request_id), int(msg.spec_cnt or 0))
                    if key in keys:
                        completed.add(key)
            self._msg_buffer[:] = retained
            if not retained:
                self._data_ready.clear()
        return acknowledgements, completed

    def _harvest_buffered_grant_acks(self) -> int:
        """Account for late ACKs even after the last Draft response/request.

        The receiver thread remains the sole communicator reader. The main
        scheduler only removes ACKs from its protected message buffer; complete
        Draft/control messages still follow the normal TP receive path. Runtime
        records terminal ACKs even if their request state has already retired.
        """
        if self.tp_rank != 0 or not hasattr(self, "_msg_buffer"):
            return 0
        runtime = self._get_specstream_runtime()
        if runtime is None:
            return 0
        acknowledgements, _ = self._drain_grant_acks_during_forward(())
        for ack in acknowledgements:
            runtime.acknowledge_grant(ack)
        return len(acknowledgements)

    def start_specstream_grant_pump(self, batch: ScheduleBatch):
        """Advance bounded overlap grants while Target is still submitting layers.

        All TP broadcasts and complete Draft consumption stay on the scheduler
        thread. Runtime methods serialize grant state; send_objs only enqueues
        to the C++ communicator's mutex-protected outbound queue.
        """
        runtime = self._get_specstream_runtime()
        if (
            runtime is None
            or runtime.grant_runtime is None
            or (self.tp_rank != 0 and getattr(runtime, "tp_window_mailbox", None) is None)
            or getattr(batch, "specstream_mode", "parallel") != "parallel"
            or os.environ.get("SPECSTREAM_BACKGROUND_GRANT_PUMP", "1") == "0"
        ):
            return None
        if runtime.config.pcie_slack_coexec and not bool(
            getattr(getattr(batch, "specstream_meta", None), "enabled", False)
        ):
            # No CPU History means no physical H2D grant window to observe.
            # Avoid a thread launch on the short-context/native GPU fast path.
            return None
        if self.tp_rank != 0:
            device = runtime.staging.device
            device_index = device.index if device.index is not None else torch.cuda.current_device()

            def publish_window():
                runtime.tp_window_mailbox.publish(
                    runtime.staging.observe_h2d_window(runtime._round_id)
                )
                return True

            return BackgroundGrantPump(
                publish_window,
                initialize=lambda: torch.cuda.set_device(device_index),
                finalize=runtime.tp_window_mailbox.clear,
            ).start()
        keys = {
            (str(req.rid), int(req.spec_cnt))
            for req in self._get_reqs_waiting_for_drafts(batch)
            if req.spec_cnt in self.req_to_draft_token.get(req.rid, {})
            and self.req_to_draft_token[req.rid][req.spec_cnt] is None
        }
        if not keys:
            return None
        pending = set(keys)
        device = runtime.staging.device
        # Staging can use an unindexed torch.device("cuda"). Resolve it on
        # the scheduler thread: a new thread may start on a different device,
        # and set_device() rejects CUDA devices without an explicit index.
        device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )

        def initialize():
            torch.cuda.set_device(device_index)

        def step():
            acknowledgements, completed = self._drain_grant_acks_during_forward(pending)
            for ack in acknowledgements:
                runtime.acknowledge_grant(ack)
            pending.difference_update(completed)
            if not pending:
                return False
            grants = runtime.overlap_grants(tuple(pending))
            if grants:
                self._zmq_send(grants)
            return True

        return BackgroundGrantPump(step, initialize=initialize).start()

    def reset_spectre_target_state(self) -> None:
        # Account for already received dispositions before retiring runtime
        # state. ACKs arriving while clear() waits for pending seals are kept
        # below and harvested by the next (possibly idle) scheduler iteration.
        self._harvest_buffered_grant_acks()
        self._spectre_flush_at_us = _spectre_now_us()
        self._accept_reject_messages = False

        # flush_cache() clears the generic request/token pools immediately
        # after this hook. SpecStream must retire pending D2H work and forget
        # GPU-History page-table ownership first.
        runtime = self._get_specstream_runtime()
        if runtime is not None:
            runtime.clear()

        if hasattr(self, "req_to_draft_token"):
            self.req_to_draft_token.clear()

        if hasattr(self, "_msg_buffer"):

            def clear_data_preserving_acks():
                self._msg_buffer[:] = [
                    msg
                    for msg in self._msg_buffer
                    if msg.action == SpectreAction.GRANT_ACK
                ]
                if hasattr(self, "_data_ready"):
                    if self._msg_buffer:
                        self._data_ready.set()
                    else:
                        self._data_ready.clear()

            if hasattr(self, "_msg_lock"):
                with self._msg_lock:
                    clear_data_preserving_acks()
            else:
                clear_data_preserving_acks()
        elif hasattr(self, "_data_ready"):
            self._data_ready.clear()

        if hasattr(self, "draft_circuit_breaker"):
            self.draft_circuit_breaker.state = DraftCircuitBreaker.CLOSED
            self.draft_circuit_breaker.consecutive_failures = 0
            self.draft_circuit_breaker.rounds_in_open = 0

        if hasattr(self, "is_rejected"):
            self.is_rejected = False
        if hasattr(self, "rejected_forward_ct"):
            self.rejected_forward_ct = 0

    def _should_store_draft_message(self, msg: SpectreRequest) -> bool:
        rid_cache = getattr(self, "req_to_draft_token", {}).get(msg.request_id)
        if rid_cache is None or msg.spec_cnt not in rid_cache:
            return False

        flush_at_us = getattr(self, "_spectre_flush_at_us", 0.0)
        if (
            flush_at_us > 0.0
            and msg.target_send_time is not None
            and msg.target_send_time >= 0.0
            and msg.target_send_time < flush_at_us
        ):
            return False

        return True

    @DynamicGradMode()
    def event_loop_normal_spectre_target(self):
        self.req_to_draft_token: Dict[
            str, Dict[int, Optional[Tuple[List[int], List[float]]]]
        ] = defaultdict(dict)
        self.is_rejected: bool = False
        self.rejected_forward_ct: int = 0

        self._init_draft_recv_infra()

        while True:
            # _collect_draft_messages stops when its final DRAFT arrives. Its
            # ACK can arrive later, when there is no subsequent receive call.
            # Drain here before idle/paused branches and before any cache flush.
            self._harvest_buffered_grant_acks()
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                batch.spectre_draft_timeout = False
                batch.spectre_policy_fallback = False
                batch.spectre_missing_draft_rids = []
                batch.spectre_fallback_reason = ""
                if self._is_self_high_overhead_target(batch):
                    self._configure_q1_fallback(batch, "target_high_overhead")
                elif not self.draft_circuit_breaker.should_send():
                    self._configure_q1_fallback(batch, "draft_circuit_open")
                elif self._consume_request_timeout_fallback(batch):
                    self._configure_q1_fallback(batch, "previous_draft_timeout")
                else:
                    draft_num_tokens = self._decide_speculative_num_draft_tokens(batch)
                    # Keep the controller/configured horizon separate from the
                    # horizon that can be verified immediately.  Parallel mode
                    # may temporarily verify q=1 while waiting for its first
                    # pipelined response; diagnostics must still know that the
                    # requested q was greater than one.
                    batch.spectre_requested_q = draft_num_tokens
                    if draft_num_tokens <= 1:
                        reason = getattr(
                            getattr(batch, "specstream_decision", None),
                            "reason",
                            "controller_q1",
                        )
                        self._configure_q1_fallback(batch, reason)
                    else:
                        self.send_batch_draft_requests(batch, draft_num_tokens)
                        batch.draft_num_tokens = self._decide_verify_num_draft_tokens(
                            batch
                        )
                        batch.recv_draft_fn = self.recv_drafts_for_batch
                        batch.retry_fn = self.retry_drafts_for_reqs
                        batch.retry_fail_ratio = (
                            self.server_args.spectre_retry_fail_ratio
                        )
                        batch.retry_min_count = self.server_args.spectre_retry_min_count

                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.self_check_during_idle()

            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

    def _configure_q1_fallback(self, batch: ScheduleBatch, reason: str) -> None:
        """Configure one batch-uniform AR/q=1 round without contacting Draft."""
        batch.spectre_requested_q = 1
        batch.draft_num_tokens = 1
        batch.recv_draft_fn = None
        batch.retry_fn = None
        batch.retry_fail_ratio = 0.0
        batch.retry_min_count = 1
        batch.specstream_mode = "ordinary"
        batch.spectre_policy_fallback = True
        batch.spectre_fallback_reason = str(reason)

    def _consume_request_timeout_fallback(self, batch: ScheduleBatch) -> bool:
        reqs = [req for req in batch.reqs if not _is_health_check(req)]
        forced_count = 0
        for req in reqs:
            if getattr(req, "spectre_force_normal_decode", False):
                forced_count += 1
                req.spectre_force_normal_decode = False
        if not reqs:
            return False
        # One delayed RID must not downgrade every ready request. Keep the
        # uniform q=1 safety round only when the missing share exceeds the same
        # policy threshold used by ready-horizon selection.
        threshold = float(self.server_args.spectre_no_draft_ratio)
        return forced_count / len(reqs) > threshold

    def _collect_draft_messages(
        self,
        pending_rids: Set[str],
        pending_spec_cnts: Dict[str, int],
        timeout_s: float,
        target_forward_done_event=None,
    ) -> List[SpectreRequest]:
        all_messages: List[SpectreRequest] = []
        deadline = time.perf_counter() + timeout_s
        grant_deadline_us = time.monotonic_ns() // 1000 + int(timeout_s * 1e6)
        grant_keys = [
            (str(rid), int(pending_spec_cnts[rid]))
            for rid in pending_rids
            if rid in pending_spec_cnts
        ]
        runtime = self._get_specstream_runtime()

        def target_forward_complete() -> bool:
            if target_forward_done_event is None:
                return True
            try:
                return bool(target_forward_done_event.query())
            except RuntimeError:
                return False

        def pump_grants() -> None:
            if runtime is None or not pending_rids:
                return
            keys = [key for key in grant_keys if key[0] in pending_rids]
            if target_forward_complete():
                grant_messages = runtime.waiting_grants(
                    keys, deadline_us=grant_deadline_us
                )
            else:
                # Preserve SPECTRE's pipeline: while Target verifies round n,
                # every ACK can unlock one more token for round n+1.  The
                # runtime reuses the original absolute PCIe-slack window, so
                # ACKs never restart or extend the overlap budget.
                grant_messages = runtime.overlap_grants(keys)
            if grant_messages:
                self._zmq_send(grant_messages)

        # SLACK_FILL advances the next-round Draft sequence while asynchronous
        # Target verification is active.  DRAFT_CATCHUP begins only after the
        # Target CUDA event completes.
        pump_grants()

        while pending_rids:
            pump_grants()
            msgs = self._drain_msg_buffer()
            if msgs:
                all_messages.extend(msgs)
                for msg in msgs:
                    if msg.action == SpectreAction.GRANT_ACK:
                        if runtime is not None:
                            runtime.acknowledge_grant(msg)
                        continue
                    if msg.action == SpectreAction.NEED_CONTEXT:
                        if msg.request_id not in pending_rids:
                            continue
                        expected_sc = pending_spec_cnts.get(msg.request_id)
                        if expected_sc is None or msg.spec_cnt == expected_sc:
                            pending_rids.discard(msg.request_id)
                        continue
                    if msg.action != SpectreAction.DRAFT:
                        continue
                    if msg.request_id not in pending_rids:
                        continue
                    expected_sc = pending_spec_cnts.get(msg.request_id)
                    if expected_sc is None or msg.spec_cnt == expected_sc:
                        pending_rids.discard(msg.request_id)
                if not pending_rids:
                    break
                pump_grants()

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                if pending_rids and self.tp_rank == 0:
                    logger.info(
                        f"\033[32m [Target] Recv timeout, {len(pending_rids)} rids still pending: "
                        f"{list(pending_rids)} \033[0m"
                    )
                break
            # GPU completion can flip after pump_grants() observed an active
            # forward. Sleeping the full receive timeout on a second query
            # would leave DRAFT_CATCHUP unissued until its deadline expires.
            # Keep progress bounded whenever this receiver owns grant issuance.
            wait_timeout = (
                min(remaining, 0.001)
                if runtime is not None and runtime.grant_runtime is not None
                else remaining if target_forward_complete() else min(remaining, 0.001)
            )
            self._data_ready.wait(timeout=wait_timeout)

        return all_messages

    def _tp_broadcast_messages(
        self, messages: Optional[List[SpectreRequest]]
    ) -> List[SpectreRequest]:
        if self.tp_size > 1:
            return broadcast_pyobj(
                messages if messages else [],
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
        return messages or []

    def _store_messages(self, messages: List[SpectreRequest]) -> bool:
        has_draft = False
        for msg in messages:
            assert isinstance(
                msg, SpectreRequest
            ), f"Expected SpectreRequest, got {type(msg)}"
            if msg.action == SpectreAction.REJECT:
                if getattr(self, "_accept_reject_messages", True):
                    self.process_reject_action()
                    if self.tp_rank == 0:
                        logger.info(
                            "\033[32m [Target] Received REJECT from draft \033[0m"
                        )
            elif msg.action == SpectreAction.DRAFT:
                if not self._should_store_draft_message(msg):
                    continue
                self.req_to_draft_token[msg.request_id][msg.spec_cnt] = (
                    msg.draft_token_ids,
                    msg.draft_logprobs,
                )
                has_draft = True
                self._accept_reject_messages = True
        return has_draft

    def _build_result_from_cache(
        self, reqs: List[Req]
    ) -> Dict[str, Tuple[List[int], List[float]]]:
        result = {}
        for req in reqs:
            if _is_health_check(req):
                continue
            rid, sc = req.rid, req.spec_cnt
            rid_cache = self.req_to_draft_token.get(rid)
            if rid_cache is None:
                continue

            entry = rid_cache.get(sc)
            if entry is not None:
                result[rid] = entry
                del rid_cache[sc]

            stale_keys = [k for k in list(rid_cache.keys()) if k < sc]
            for k in stale_keys:
                del rid_cache[k]

        return result

    def _get_extend_decode_req_ids(self, batch: ScheduleBatch) -> Set[int]:
        decoding_reqs = getattr(batch, "decoding_reqs", None)
        if not decoding_reqs:
            return set()
        return {id(req) for req in decoding_reqs if not _is_health_check(req)}

    def _get_reqs_waiting_for_drafts(self, batch: ScheduleBatch) -> List[Req]:
        forward_mode = getattr(batch, "forward_mode", None)
        is_extend_batch = (
            forward_mode is not None and forward_mode.is_extend()
        ) or getattr(batch, "is_extend_in_batch", False)
        if not is_extend_batch:
            return [req for req in batch.reqs if not _is_health_check(req)]

        decoding_req_ids = self._get_extend_decode_req_ids(batch)
        return [
            req
            for req in batch.reqs
            if not _is_health_check(req)
            and (id(req) in decoding_req_ids or getattr(req, "is_chunked", 0) <= 0)
        ]

    def recv_drafts_for_batch(self, batch: ScheduleBatch) -> dict:
        specstream_started = time.perf_counter()
        batch.spectre_draft_timeout = False
        batch.spectre_missing_draft_rids = []
        reqs_waiting_for_drafts = self._get_reqs_waiting_for_drafts(batch)
        is_initial_round = any(
            req.spec_cnt <= 0
            or bool(getattr(req, "spectre_full_context_pending", False))
            for req in reqs_waiting_for_drafts
        )
        timeout_s = (
            self._initial_recv_timeout_s if is_initial_round else self._recv_timeout_s
        )

        if self.tp_size == 1 or self.tp_rank == 0:
            pending_rids = {
                req.rid
                for req in reqs_waiting_for_drafts
                if req.spec_cnt in self.req_to_draft_token.get(req.rid, {})
                and self.req_to_draft_token[req.rid][req.spec_cnt] is None
            }
            messages = (
                self._collect_draft_messages(
                    pending_rids=pending_rids,
                    pending_spec_cnts={
                        req.rid: req.spec_cnt
                        for req in reqs_waiting_for_drafts
                        if req.spec_cnt in self.req_to_draft_token.get(req.rid, {})
                        and self.req_to_draft_token[req.rid][req.spec_cnt] is None
                    },
                    timeout_s=timeout_s,
                    target_forward_done_event=getattr(
                        batch, "spectre_target_forward_done_event", None
                    ),
                )
                if pending_rids
                else []
            )
        else:
            messages = None

        messages = self._tp_broadcast_messages(messages)

        waiting_keys = {(req.rid, req.spec_cnt) for req in reqs_waiting_for_drafts}
        self._store_messages(messages)
        resync_rids = {
            str(msg.request_id)
            for msg in messages
            if msg.action == SpectreAction.NEED_CONTEXT
            and (msg.request_id, msg.spec_cnt) in waiting_keys
        }
        recv_now_us = _spectre_now_us()
        delivered_rtt_ms = [
            max(0.0, (recv_now_us - float(msg.target_send_time)) / 1000.0)
            for msg in messages
            if msg.action == SpectreAction.DRAFT
            and (msg.request_id, msg.spec_cnt) in waiting_keys
            and msg.target_send_time is not None
            and msg.target_send_time > 0.0
        ]
        requested_q = int(
            getattr(
                batch,
                "spectre_requested_q",
                getattr(batch, "draft_num_tokens", 1),
            )
            or 1
        )

        result = self._build_result_from_cache(reqs_waiting_for_drafts)
        for req in reqs_waiting_for_drafts:
            if req.rid in result:
                _mark_draft_context_synced(req)
            elif req.rid in resync_rids:
                _mark_draft_context_unsynced(req)
        missing_reqs: List[Req] = []
        timed_out_reqs: List[Req] = []
        fail_fast_message: Optional[str] = None
        if reqs_waiting_for_drafts:
            missing_reqs = [
                req for req in reqs_waiting_for_drafts if req.rid not in result
            ]
            if not missing_reqs:
                self.draft_circuit_breaker.record_success()
            else:
                timed_out_reqs = [
                    req for req in missing_reqs if req.rid not in resync_rids
                ]
                # NEED_CONTEXT proves that Drafter is responsive. A partial
                # response is likewise not a process-wide outage and must not
                # open the global circuit breaker.
                if timed_out_reqs and len(timed_out_reqs) == len(
                    reqs_waiting_for_drafts
                ):
                    self.draft_circuit_breaker.record_failure()
                else:
                    self.draft_circuit_breaker.record_success()
                missing = [req.rid for req in missing_reqs]
                batch.spectre_draft_timeout = bool(timed_out_reqs)
                batch.spectre_missing_draft_rids = missing
                batch.spectre_fallback_reason = (
                    "remote_draft_timeout" if timed_out_reqs else "remote_draft_resync"
                )
                if self.tp_size == 1 or self.tp_rank == 0:
                    runtime = self._get_specstream_runtime()
                    if runtime is not None:
                        pause_messages = runtime.pause_grants(
                            [(str(req.rid), int(req.spec_cnt)) for req in missing_reqs]
                        )
                        if pause_messages:
                            self._zmq_send(pause_messages)
                for req in missing_reqs:
                    req.spectre_full_context_pending = False
                    if req.rid in resync_rids or not bool(
                        getattr(req, "spectre_draft_initialized", False)
                    ):
                        _mark_draft_context_unsynced(req)
                    # Ordinary mode consumes this marker in the current worker
                    # call. Parallel/extend mode consumes it on the next batch.
                    # A responsive resync request can recover directly on the
                    # next q>1 round without downgrading unrelated requests.
                    req.spectre_force_normal_decode = req.rid not in resync_rids
                if self.tp_rank == 0:
                    logger.warning(
                        "[Target][DraftFallback] q=%d missing=%d/%d resync=%d "
                        "after %.0f ms; rids=%s",
                        requested_q,
                        len(missing_reqs),
                        len(reqs_waiting_for_drafts),
                        len(resync_rids),
                        timeout_s * 1000,
                        missing,
                    )
                if (
                    timed_out_reqs
                    and requested_q > 1
                    and should_fail_fast_on_draft_timeout(
                        require_draft=self.server_args.spectre_require_draft,
                        timeout_action=self.server_args.spectre_draft_timeout_action,
                    )
                ):
                    fail_fast_message = (
                        "SPECTRE required a remote draft for q="
                        f"{requested_q}, but no valid response arrived within "
                        f"{timeout_s * 1000:.0f} ms; "
                        f"missing_rids={[req.rid for req in timed_out_reqs]}. "
                        "Check Drafter readiness/ZMQ and increase "
                        "--spectre-recv-timeout-ms or "
                        "--spectre-initial-recv-timeout-ms."
                    )
        elapsed_ms = (time.perf_counter() - specstream_started) * 1000
        observed_rtt_ms = max(
            max(delivered_rtt_ms, default=elapsed_ms),
            timeout_s * 1000 if timed_out_reqs else 0.0,
        )
        runtime = self._get_specstream_runtime()
        if runtime is not None:
            runtime.record_network_wait(elapsed_ms)
            runtime.record_draft_result(
                q=requested_q,
                elapsed_ms=elapsed_ms,
                rtt_ms=observed_rtt_ms,
                timeout_ms=timeout_s * 1000,
                # A NEED_CONTEXT response is a fast protocol repair, not a
                # timeout sample, and must not poison adaptive backoff.
                missing_count=sum(req.rid not in resync_rids for req in missing_reqs),
                total_count=len(reqs_waiting_for_drafts),
            )
        if fail_fast_message is not None:
            raise RuntimeError(fail_fast_message)
        return result

    def retry_drafts_for_reqs(self, failed_reqs: List[Req]) -> dict:
        if not failed_reqs:
            return {}

        num_draft_tokens = self.server_args.speculative_num_steps + 1

        for req in failed_reqs:
            if _is_health_check(req):
                continue
            self.req_to_draft_token[req.rid][req.spec_cnt] = None

        if self.tp_size == 1 or self.tp_rank == 0:
            self._send_retry_requests(failed_reqs, max(1, num_draft_tokens - 1))

        if self.tp_size == 1 or self.tp_rank == 0:
            pending_rids = {req.rid for req in failed_reqs if not _is_health_check(req)}
            pending_spec_cnts = {
                req.rid: req.spec_cnt
                for req in failed_reqs
                if not _is_health_check(req)
            }
            messages = self._collect_draft_messages(
                pending_rids=pending_rids,
                pending_spec_cnts=pending_spec_cnts,
                timeout_s=self._recv_timeout_s * 0.5,
            )
        else:
            messages = None

        messages = self._tp_broadcast_messages(messages)

        self._store_messages(messages)
        result = self._build_result_from_cache(failed_reqs)
        retry_resync_rids = {
            str(msg.request_id)
            for msg in messages
            if msg.action == SpectreAction.NEED_CONTEXT
        }
        for req in failed_reqs:
            if req.rid in result:
                _mark_draft_context_synced(req)
            elif req.rid in retry_resync_rids:
                _mark_draft_context_unsynced(req)
        return result

    def send_batch_draft_requests(
        self, batch: ScheduleBatch, speculative_num_draft_tokens: int
    ) -> None:
        if (
            self.is_rejected
            and self.server_args.spectre_reject_interval > 0
            and (
                (self.forward_ct - self.rejected_forward_ct + 1)
                % self.server_args.spectre_reject_interval
                != 0
            )
        ):
            return

        self.is_rejected = False

        reqs_to_send: List[Req] = []

        for req in batch.reqs:
            if _is_health_check(req):
                continue
            rid_cache = self.req_to_draft_token[req.rid]
            if req.spec_cnt in rid_cache:
                continue
            rid_cache[req.spec_cnt] = None
            reqs_to_send.append(req)

        if self.tp_size == 1 or self.tp_rank == 0:
            if hasattr(self, "zmq_communicator") and self.zmq_communicator is not None:
                draft_reqs = []
                is_half_open = (
                    self.draft_circuit_breaker.state == DraftCircuitBreaker.HALF_OPEN
                )
                for req in reqs_to_send:
                    needs_full_context = _draft_needs_full_context(
                        req, is_half_open=is_half_open
                    )
                    _mark_draft_context_pending(req, needs_full_context)
                    draft_reqs.append(
                        SpectreRequest(
                            request_id=req.rid,
                            spec_cnt=req.spec_cnt,
                            action=SpectreAction.DRAFT,
                            spec_type=SpecType.DRAFT_REQUEST,
                            input_ids=(
                                req.origin_input_ids if needs_full_context else None
                            ),
                            output_ids=req.output_ids,
                            draft_token_ids=req.cur_drafts,
                            num_draft_tokens=speculative_num_draft_tokens,
                            sampling_params=(
                                req.sampling_params if needs_full_context else None
                            ),
                            grammar=None,
                        )
                    )
                if draft_reqs:
                    runtime = self._get_specstream_runtime()
                    grant_reqs = (
                        runtime.prepare_initial_grants(
                            batch, speculative_num_draft_tokens
                        )
                        if runtime is not None
                        else []
                    )
                    batch.spectre_draft_request_sent = self._zmq_send(
                        draft_reqs + grant_reqs,
                        wait_for_identity_s=(
                            self._initial_recv_timeout_s
                            if any(
                                getattr(req, "spectre_full_context_pending", False)
                                for req in reqs_to_send
                            )
                            else 0.0
                        ),
                    )

    def _send_retry_requests(
        self, failed_reqs: List[Req], num_draft_tokens: int
    ) -> None:
        if not (
            hasattr(self, "zmq_communicator") and self.zmq_communicator is not None
        ):
            return
        reqs_to_send = []
        for req in failed_reqs:
            if _is_health_check(req):
                continue
            needs_full_context = _draft_needs_full_context(req)
            _mark_draft_context_pending(req, needs_full_context)
            reqs_to_send.append(
                SpectreRequest(
                    request_id=req.rid,
                    spec_cnt=req.spec_cnt,
                    action=SpectreAction.DRAFT,
                    spec_type=SpecType.DRAFT_REQUEST,
                    input_ids=(req.origin_input_ids if needs_full_context else None),
                    output_ids=req.output_ids,
                    draft_token_ids=(req.cur_drafts if not needs_full_context else []),
                    num_draft_tokens=num_draft_tokens,
                    sampling_params=(
                        req.sampling_params if needs_full_context else None
                    ),
                )
            )
        if reqs_to_send:
            runtime = self._get_specstream_runtime()
            grant_reqs = (
                runtime.prepare_retry_grants(failed_reqs, num_draft_tokens)
                if runtime is not None
                else []
            )
            self._zmq_send(reqs_to_send + grant_reqs)

    def _zmq_send(
        self, reqs: List[SpectreRequest], wait_for_identity_s: float = 0.0
    ) -> bool:
        deadline = time.perf_counter() + max(float(wait_for_identity_s), 0.0)
        all_drafts_identity = self.zmq_communicator.get_all_drafts_identity()
        while not all_drafts_identity and time.perf_counter() < deadline:
            time.sleep(0.01)
            all_drafts_identity = self.zmq_communicator.get_all_drafts_identity()
        if not all_drafts_identity:
            logger.warning(
                "\033[32m [Target] No draft available, check draft status! \033[0m"
            )
            return False
        if wait_for_identity_s > 0.0 and self.tp_rank == 0:
            logger.info(
                "[Target][DraftLink] registered=%s; sending %d initial " "request(s)",
                all_drafts_identity[0],
                len(reqs),
            )
        send_time_us = _spectre_now_us()
        for req in reqs:
            if req.action == SpectreAction.DRAFT:
                req.target_send_time = send_time_us
        self.zmq_communicator.send_objs(reqs, all_drafts_identity[0])
        return True

    def notify_draft_request_finished_or_aborted(
        self, req: Req, action: SpectreAction
    ) -> None:
        if _is_health_check(req):
            return

        msg = SpectreRequest(
            request_id=req.rid,
            spec_cnt=req.spec_cnt,
            action=action,
            spec_type=SpecType.DRAFT_REQUEST,
            input_ids=[],
            output_ids=[],
            draft_token_ids=[],
            num_draft_tokens=0,
        )

        if self.tp_size == 1 or self.tp_rank == 0:
            if hasattr(self, "zmq_communicator") and self.zmq_communicator is not None:
                self._zmq_send([msg])

        try:
            if req.rid in self.req_to_draft_token:
                del self.req_to_draft_token[req.rid]
        except Exception as e:
            if self.tp_rank == 0:
                logger.error(
                    f"\033[34m [Target][Notify] Failed to cleanup req_to_draft_token "
                    f"for {req.rid}: {e} \033[0m"
                )
        runtime = self._get_specstream_runtime()
        if runtime is not None:
            runtime.release_request(req.rid)

    def prepare_specstream_request_release(self, req: Req) -> None:
        """Finish an in-flight D2H seal before generic KV cache release.

        This is a terminal safety boundary, not part of the decode critical
        path.  During normal inference seals are retired only through
        nonblocking CUDA-event polling.
        """

        runtime = self._get_specstream_runtime()
        if runtime is not None:
            runtime.prepare_request_release(req.rid)

    def _is_self_high_overhead_target(self, batch: ScheduleBatch) -> bool:
        current_bsz = max(batch.batch_size(), self.running_batch.batch_size())
        if current_bsz > self.server_args.spectre_max_batch_size:
            batch.is_high_overhead = True
            return True
        batch.is_high_overhead = False
        return False

    def _decide_speculative_num_draft_tokens(self, batch: ScheduleBatch) -> int:
        batch.specstream_rejected = bool(self.is_rejected)
        if self.is_rejected:
            batch.specstream_mode = "ordinary"
            return 1
        runtime = self._get_specstream_runtime()
        if runtime is not None and runtime.controller is not None:
            # Local samples can differ while asynchronous seals/timing retire.
            # Every rank must receive the same gate before any conditional
            # collective, or one can all-gather while another broadcasts q.
            sync_tp_profile = (
                bool(runtime.should_sync_tp_profile()) if self.tp_rank == 0 else False
            )
            if self.tp_size > 1:
                sync_payload = broadcast_pyobj(
                    [sync_tp_profile] if self.tp_rank == 0 else [],
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
                if len(sync_payload) != 1 or type(sync_payload[0]) is not bool:
                    raise RuntimeError(
                        "TP profile sync broadcast returned an invalid gate"
                    )
                sync_tp_profile = sync_payload[0]
            if sync_tp_profile:
                samples = self.tp_group.all_gather_object(
                    runtime.local_tp_rank_sample()
                )
                runtime.record_tp_rank_samples(samples)
            decision = runtime.choose_decision(batch) if self.tp_rank == 0 else None
            if self.tp_size > 1:
                # broadcast_pyobj serializes a list payload.  Passing the
                # dataclass directly works in TP=1 (where no broadcast is
                # needed) but crashes TP>1 dynamic-q on the source rank when
                # broadcast_pyobj calls len(data).
                decision_payload = broadcast_pyobj(
                    [decision] if self.tp_rank == 0 else [],
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
                if len(decision_payload) != 1 or decision_payload[0] is None:
                    raise RuntimeError(
                        "TP dynamic-q broadcast returned an invalid decision payload"
                    )
                decision = decision_payload[0]
            runtime.record_decision(decision)
            batch.specstream_decision = decision
            batch.specstream_mode = decision.mode
            return int(decision.q)
        batch.specstream_mode = self.server_args.spectre_fixed_q_mode
        return self.server_args.speculative_num_steps + 1

    def process_reject_action(self) -> None:
        self.is_rejected = True
        self.rejected_forward_ct = self.forward_ct
        runtime = self._get_specstream_runtime()
        if runtime is not None:
            runtime.record_draft_reject()

    def _decide_verify_num_draft_tokens(self, batch: ScheduleBatch) -> int:
        if batch.forward_mode == ForwardMode.EXTEND:
            return self.server_args.speculative_num_draft_tokens

        if self.is_rejected:
            if self.tp_rank == 0:
                logger.info("\033[34m [Target] draft_num_tokens=1 (rejected) \033[0m")
            return 1

        # Ordinary mode intentionally waits for the draft in SpectreWorker
        # immediately before constructing TARGET_VERIFY.  At this point the
        # just-requested draft is therefore not yet present in req.cur_drafts;
        # applying the parallel-mode no-draft gate here would force every
        # ordinary round to q=1 and make the later wait unreachable.
        no_draft_reqs = sum(
            1 for req in batch.reqs if not _is_health_check(req) and not req.cur_drafts
        )
        bs = batch.batch_size()
        return choose_ready_verify_horizon(
            mode=getattr(batch, "specstream_mode", "parallel"),
            requested_q=getattr(
                batch,
                "spectre_requested_q",
                self.server_args.speculative_num_draft_tokens,
            ),
            batch_size=bs,
            no_draft_count=no_draft_reqs,
            no_draft_ratio=self.server_args.spectre_no_draft_ratio,
        )

    def _get_specstream_runtime(self):
        draft_worker = getattr(self, "draft_worker", None)
        return getattr(draft_worker, "specstream_runtime", None)
