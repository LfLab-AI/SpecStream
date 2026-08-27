import logging
import os
import time
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.sampler import SamplingBatchInfo
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spectre.drafter.spectre_kv_rollbacker import (
    SpectreKVRollbacker,
)
from sglang.srt.speculative.spectre.drafter.spectre_state_manager import (
    SpectreDraftState,
    SpectreDraftStateManager,
)
from sglang.srt.speculative.spectre.specstream.gpu_grant import (
    DraftExecutionGrant,
    DraftGrantTable,
)
from sglang.srt.speculative.spectre.spectre_protocol import (
    SpectreAction,
    SpectreRequest,
    SpecType,
)
from sglang.srt.utils import DynamicGradMode, broadcast_pyobj
from sglang.srt.speculative.spectre.specstream.draft_grant_runtime import (
    DraftGrantGate,
    call_runtime_first_available,
    normalize_external_grant,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DraftReqLocation(str, Enum):
    DRAFT_WAITING = "draft_waiting"
    DRAFT_BATCH = "draft_batch"
    PAUSED = "paused"


def _fix_sampling_params_stop_strs(sp) -> None:
    if not hasattr(sp, "stop_strs") or sp.stop_strs is None:
        sp.stop_strs = []
    elif isinstance(sp.stop_strs, str):
        sp.stop_strs = [sp.stop_strs]

    if not hasattr(sp, "stop_regex_strs") or sp.stop_regex_strs is None:
        sp.stop_regex_strs = []
    elif isinstance(sp.stop_regex_strs, str):
        sp.stop_regex_strs = [sp.stop_regex_strs]


class SpectreDraftSchedulerMixin:
    def _init_draft_components(self) -> None:
        self.draft_state_manager = SpectreDraftStateManager(timeout_threshold=60.0)
        self.draft_kv_manager = SpectreKVRollbacker(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            tree_cache=self.tree_cache,
            page_size=self.server_args.page_size or 1,
            tp_rank=self.tp_rank,
        )

        self.draft_waiting_queue: List[Req] = []
        self.draft_batch: ScheduleBatch = ScheduleBatch(reqs=[])
        self.draft_paused_reqs: List[Req] = []
        self.last_draft_batch: Optional[ScheduleBatch] = None
        self._draft_batch_pending_adds: List[Req] = []
        self.draft_forward_cycle: int = 0
        self._specstream_grants_enabled = bool(
            getattr(self.server_args, "specstream_smctrl_enabled", False)
        )
        if (
            self._specstream_grants_enabled
            and not self.server_args.spectre_draft_priority
        ):
            raise RuntimeError(
                "SpecStream SM control requires --spectre-draft-priority so "
                "one-token grant boundaries cannot be bypassed"
            )
        if self._specstream_grants_enabled and self.enable_overlap:
            raise RuntimeError(
                "SpecStream SM control requires --disable-overlap-schedule so "
                "a grant closes only after synchronous Draft sampling"
            )
        if self._specstream_grants_enabled:
            worker = getattr(self, "tp_worker", getattr(self, "model_worker", None))
            readiness = getattr(worker, "get_specstream_draft_smctrl_info", None)
            if readiness is None:
                raise RuntimeError(
                    "SpecStream remote Drafter worker does not expose libsmctrl "
                    "startup readiness"
                )
            mask_scope, total_tpcs = readiness()
            logger.info(
                "SpecStream remote Drafter libsmctrl ready: scope=%s total_tpcs=%d",
                mask_scope,
                total_tpcs,
            )
        self._grant_table = DraftGrantTable()
        self.draft_cleanup_interval: int = int(
            os.environ.get("SGLANG_DRAFT_CLEANUP_INTERVAL", "500")
        )
        self._specstream_init_first_grant_gate()


    def _specstream_smctrl_enabled_for_draft(self) -> bool:
        return bool(getattr(self.server_args, "specstream_smctrl_enabled", False))

    def _specstream_calibration_tpcs(self) -> int:
        return int(
            getattr(self.server_args, "specstream_smctrl_calibration_tpcs", 0) or 0
        )

    def _specstream_underlying_tp_worker(self):
        worker = getattr(self, "model_worker", None)
        if worker is None:
            raise RuntimeError("SpecStream cannot find Scheduler.model_worker")
        return getattr(worker, "target_worker", worker)

    def _specstream_init_first_grant_gate(self) -> None:
        if not self._specstream_smctrl_enabled_for_draft():
            self._specstream_draft_grant_gate = None
            return

        if not bool(getattr(self.server_args, "spectre_draft_priority", False)):
            raise RuntimeError(
                "SpecStream SM control requires --spectre-draft-priority so "
                "one-token grant boundaries cannot be bypassed"
            )

        worker = self._specstream_underlying_tp_worker()
        total_tpcs = None
        fn = getattr(worker, "specstream_total_tpcs", None)
        if callable(fn):
            total_tpcs = fn()

        self._specstream_draft_grant_gate = DraftGrantGate(
            total_tpcs=total_tpcs
        )

        calibration_tpcs = self._specstream_calibration_tpcs()
        if calibration_tpcs > 0:
            grant = self._specstream_draft_grant_gate.install_calibration_grant(
                calibration_tpcs
            )
            self._specstream_apply_grant_to_worker(grant)
            if getattr(self, "tp_rank", 0) == 0:
                logger.info(
                    "SpecStream calibration bootstrap grant active before first "
                    "Draft forward: epoch=%d tpc=[%d,%d)",
                    grant.epoch,
                    grant.tpc_low,
                    grant.tpc_high,
                )

    def _specstream_req_ids(self, batch) -> tuple[str, ...]:
        if batch is None:
            return ()
        if isinstance(batch, (list, tuple)):
            reqs = batch
        else:
            reqs = getattr(batch, "reqs", ())
        return tuple(
            str(r.rid)
            for r in reqs
            if hasattr(r, "rid")
        )

    def _specstream_existing_coexec_runtime(self):
        names = (
            "specstream_coexec_runtime",
            "_specstream_coexec_runtime",
            "coexec_runtime",
            "target_grant_runtime",
            "grant_runtime",
        )
        for owner in (self, getattr(self, "model_worker", None)):
            if owner is None:
                continue
            for name in names:
                obj = getattr(owner, name, None)
                if obj is not None:
                    return obj
        return None

    def _specstream_poll_online_grant(self, batch):
        gate = getattr(self, "_specstream_draft_grant_gate", None)
        if gate is None:
            return None

        if self._specstream_calibration_tpcs() > 0:
            return gate.try_acquire(self._specstream_req_ids(batch))

        runtime = self._specstream_existing_coexec_runtime()
        req_ids = self._specstream_req_ids(batch)
        external = call_runtime_first_available(
            runtime,
            (
                "try_acquire_one_token",
                "try_acquire_draft_grant",
                "acquire_draft_grant",
                "get_active_draft_grant",
                "current_draft_grant",
            ),
            batch=batch,
            request_ids=req_ids,
        )
        grant = normalize_external_grant(external)
        if grant is not None:
            gate.install_online_grant(grant)

        return gate.try_acquire(req_ids)

    def _specstream_apply_grant_to_worker(self, grant) -> None:
        worker = self._specstream_underlying_tp_worker()
        fn = getattr(worker, "specstream_apply_draft_tpc_range", None)
        if fn is None:
            raise RuntimeError(
                "TpModelWorker lacks specstream_apply_draft_tpc_range(); "
                "tp_worker patch was not applied"
            )
        fn(int(grant.tpc_low), int(grant.tpc_high))

    def _specstream_begin_controlled_forward(self) -> None:
        worker = self._specstream_underlying_tp_worker()
        fn = getattr(worker, "specstream_begin_controlled_draft_forward", None)
        if fn is not None:
            fn()

    def _specstream_end_controlled_forward(self) -> None:
        worker = self._specstream_underlying_tp_worker()
        fn = getattr(worker, "specstream_end_controlled_draft_forward", None)
        if fn is not None:
            fn()

    def _specstream_finish_one_granted_step(
        self, batch, *, success: bool
    ) -> None:
        gate = getattr(self, "_specstream_draft_grant_gate", None)
        if gate is None:
            return
        gate.complete_one_step(success=success)

        if self._specstream_calibration_tpcs() > 0:
            return

        runtime = self._specstream_existing_coexec_runtime()
        call_runtime_first_available(
            runtime,
            (
                "complete_draft_step",
                "ack_draft_grant",
                "complete_one_token",
                "mark_draft_step_complete",
            ),
            batch=batch,
            request_ids=self._specstream_req_ids(batch),
        )

    def _get_draft_state(self, req_id: str) -> Optional[SpectreDraftState]:
        return self.draft_state_manager.get_state(req_id)

    def _set_draft_state(self, req_id: str, state: SpectreDraftState) -> None:
        self.draft_state_manager.set_state(req_id, state)

    def _delete_draft_state(self, req_id: str) -> bool:
        return self.draft_state_manager.delete(req_id)

    def _exists_draft_state(self, req_id: str) -> bool:
        return self.draft_state_manager.exists(req_id)

    @DynamicGradMode()
    def event_loop_normal_spectre_draft(self) -> None:
        self.last_batch = None
        self._init_draft_components()
        draft_priority: bool = self.server_args.spectre_draft_priority

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.recv_and_process_draft_requests()
            saved_last_batch = self.last_batch
            if self.draft_waiting_queue:
                self._prefill_draft_reqs()
            self.last_batch = saved_last_batch

            if draft_priority:
                self._run_draft_priority_phase()
            else:
                self._merge_draft_into_running()
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
                self.draft_forward_cycle += 1
            else:
                self.self_check_during_idle()

            self.last_batch = batch

            if not draft_priority:
                self._extract_paused_drafts_from_running()

            if self.draft_forward_cycle % self.draft_cleanup_interval == 0:
                self._cleanup_stale_draft_states()


    def _run_draft_priority_phase(self) -> None:
        saved_last_batch = self.last_batch
        self._filter_draft_batch()

        if self.draft_batch.is_empty():
            self.last_batch = saved_last_batch
            return

        if not self._specstream_smctrl_enabled_for_draft():
            remaining_steps = max(
                (
                    r.draft_tokens_target
                    - (len(r.output_ids) - r.draft_generation_start_len)
                )
                for r in self.draft_batch.reqs
                if not getattr(r, "draft_is_paused", False)
            )
            max_steps = self.server_args.spectre_max_draft_priority_steps
            if max_steps <= 0:
                max_steps = remaining_steps
            steps_taken = max(1, min(remaining_steps, max_steps))

            for _step in range(steps_taken):
                self._filter_draft_batch()
                if self.draft_batch.is_empty():
                    break
                if not self.draft_batch.check_decode_mem():
                    self._handle_draft_batch_oom()
                    break
                self.draft_batch.prepare_for_decode()
                result = self.run_batch(self.draft_batch)
                self._process_draft_decode_result(
                    self.draft_batch, result
                )
                self._update_draft_batch_after_decode()

            self.last_batch = saved_last_batch
            return

        # I2: no grant -> no GPU forward.
        grant = self._specstream_poll_online_grant(self.draft_batch)
        if grant is None:
            self.last_batch = saved_last_batch
            return

        if not self.draft_batch.check_decode_mem():
            gate = self._specstream_draft_grant_gate
            if gate is not None and gate.inflight:
                gate.complete_one_step(success=False)
            self._handle_draft_batch_oom()
            self.last_batch = saved_last_batch
            return

        self._specstream_apply_grant_to_worker(grant)
        self.draft_batch.prepare_for_decode()

        success = False
        self._specstream_begin_controlled_forward()
        try:
            result = self.run_batch(self.draft_batch)
            self._process_draft_decode_result(
                self.draft_batch, result
            )
            self._update_draft_batch_after_decode()
            self.draft_forward_cycle += 1
            success = True
        finally:
            self._specstream_end_controlled_forward()
            self._specstream_finish_one_granted_step(
                self.draft_batch, success=success
            )

        self.last_batch = saved_last_batch

    def _run_granted_draft_step(self) -> None:
        """Run at most one token and only for requests with an active grant."""
        saved_last_batch = self.last_batch
        self._pause_ungranted_draft_reqs()
        self._filter_draft_batch()
        if self.draft_batch.is_empty():
            self.last_batch = saved_last_batch
            return

        grants = self._active_grants_for_batch(self.draft_batch)
        if len(grants) != len(self.draft_batch.reqs):
            # Fail closed if request/grant state changed between filtering and
            # launch preparation.
            self._pause_ungranted_draft_reqs()
            self._filter_draft_batch()
            self.last_batch = saved_last_batch
            return
        if not self.draft_batch.check_decode_mem():
            self._handle_draft_batch_oom()
            self.last_batch = saved_last_batch
            return

        self._apply_batch_grant_mask(grants.values())
        runnable_reqs = list(self.draft_batch.reqs)
        started = time.perf_counter()
        self.draft_batch.prepare_for_decode()
        result = self.run_batch(self.draft_batch)
        self._synchronize_specstream_draft_stream()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        for req in runnable_reqs:
            grant = grants.get(req.rid)
            if grant is None:
                continue
            consumed = self._grant_table.consume_one(
                req.rid,
                spec_cnt=int(req.spec_cnt),
                grant_epoch=grant.grant_epoch,
            )
            if not consumed:
                raise RuntimeError(
                    f"Draft grant changed during forward for request {req.rid}"
                )
            self._send_grant_ack(req, grant, elapsed_ms)

        # ACK is emitted after GPU completion but before a terminal DRAFT
        # response.  This preserves the protocol invariant that the next grant
        # cannot overtake completion of the previous quantum.
        self._process_draft_decode_result(self.draft_batch, result)
        for req in runnable_reqs:
            if not req.finished():
                req.draft_is_paused = True
                state = self._get_draft_state(req.rid)
                if state is not None:
                    state.location = DraftReqLocation.PAUSED
                    state.last_updated_time = time.time()
                if req not in self.draft_paused_reqs:
                    self.draft_paused_reqs.append(req)

        self._update_draft_batch_after_decode()
        self.last_batch = saved_last_batch

    def _active_grants_for_batch(
        self, batch: ScheduleBatch
    ) -> Dict[str, DraftExecutionGrant]:
        return self._grant_table.active_for(
            ((req.rid, int(req.spec_cnt)) for req in batch.reqs)
        )

    def _pause_ungranted_draft_reqs(self) -> None:
        if self.draft_batch.is_empty():
            return
        for req in self.draft_batch.reqs:
            if (
                self._grant_table.active(req.rid, spec_cnt=int(req.spec_cnt))
                is not None
            ):
                continue
            self._ack_expired_grant(req)
            req.draft_is_paused = True
            state = self._get_draft_state(req.rid)
            if state is not None:
                state.location = DraftReqLocation.PAUSED
                state.last_updated_time = time.time()
            if req not in self.draft_paused_reqs:
                self.draft_paused_reqs.append(req)

    def _ack_expired_grant(self, req: Req) -> bool:
        expired = self._grant_table.pop_expired(
            req.rid, spec_cnt=int(req.spec_cnt)
        )
        if expired is None:
            return False
        # A deadline can expire while the message is in ZMQ or while Target
        # owns the GPU.  Failing closed is correct, but silently deleting the
        # grant strands Target's outstanding epoch and prevents it from issuing
        # DRAFT_CATCHUP.  A zero-token ACK reports that no CUDA work launched
        # and safely reopens the one-token sequencer.
        self._send_grant_ack(
            req,
            expired,
            0.0,
            grant_tokens=0,
            grant_state="EXPIRED",
        )
        if self.tp_rank == 0:
            logger.debug(
                "[Draft][Grant] expired rid=%s spec_cnt=%s epoch=%s; "
                "sent zero-token ACK",
                req.rid,
                req.spec_cnt,
                expired.grant_epoch,
            )
        return True

    def _apply_batch_grant_mask(self, grants) -> tuple[int, int]:
        grants = tuple(grants)
        if not grants:
            raise RuntimeError("cannot launch Draft CUDA work without a grant")
        low = max(grant.tpc_low for grant in grants)
        high = min(grant.tpc_high for grant in grants)
        if high <= low:
            raise RuntimeError("active Draft grants have disjoint TPC ranges")
        worker = getattr(self, "tp_worker", getattr(self, "model_worker", None))
        setter = getattr(worker, "set_specstream_draft_tpc_mask", None)
        if setter is None:
            raise RuntimeError(
                "SpecStream Drafter worker does not expose stream-level TPC masking"
            )
        setter(low, high)
        return low, high

    def _synchronize_specstream_draft_stream(self) -> None:
        worker = getattr(self, "tp_worker", getattr(self, "model_worker", None))
        synchronize = getattr(worker, "synchronize_specstream_draft_stream", None)
        if synchronize is None:
            raise RuntimeError(
                "SpecStream Drafter worker does not expose stream synchronization"
            )
        synchronize()

    def _send_grant_ack(
        self,
        req: Req,
        grant: DraftExecutionGrant,
        elapsed_ms: float,
        *,
        grant_tokens: int = 1,
        grant_state: Optional[str] = None,
    ) -> None:
        if self.tp_size > 1 and self.tp_rank != 0:
            return
        if not hasattr(self, "zmq_communicator") or self.zmq_communicator is None:
            return
        self.zmq_communicator.send_objs(
            [
                SpectreRequest(
                    request_id=req.rid,
                    spec_cnt=int(req.spec_cnt),
                    action=SpectreAction.GRANT_ACK,
                    spec_type=SpecType.DRAFT_RESPONSE,
                    grant_epoch=grant.grant_epoch,
                    grant_tokens=int(grant_tokens),
                    tpc_low=grant.tpc_low,
                    tpc_high=grant.tpc_high,
                    draft_step_ms=float(elapsed_ms),
                    grant_state=grant_state or grant.grant_state,
                )
            ]
        )
        if self.tp_rank == 0:
            logger.debug(
                "[Draft][Grant] ACK rid=%s spec_cnt=%s epoch=%s tokens=%s "
                "state=%s step_ms=%.3f",
                req.rid,
                req.spec_cnt,
                grant.grant_epoch,
                grant_tokens,
                grant_state or grant.grant_state,
                elapsed_ms,
            )

    def _process_draft_decode_result(self, batch: ScheduleBatch, result) -> None:
        self.process_batch_result_decode(batch, result)
        self._filter_draft_batch()

    def _update_draft_batch_after_decode(self) -> None:
        self._filter_draft_batch()

    def _handle_draft_batch_oom(self) -> None:
        if self.tp_rank == 0:
            logger.warning("[Draft] Draft batch OOM, aborting all draft reqs in batch")
        for req in list(self.draft_batch.reqs):
            try:
                self._finish_draft_request(req.rid)
            except Exception as e:
                if self.tp_rank == 0:
                    logger.error(f"[Draft] Failed to finish {req.rid} during OOM: {e}")
        self.draft_batch = ScheduleBatch(reqs=[])

    def _filter_draft_batch(self) -> None:
        if self.draft_batch.is_empty():
            return

        keep_indices: List[int] = []
        for i, req in enumerate(self.draft_batch.reqs):
            if getattr(req, "draft_is_paused", False):
                if req not in self.draft_paused_reqs:
                    self.draft_paused_reqs.append(req)
            elif req.finished():
                pass
            else:
                keep_indices.append(i)

        if len(keep_indices) < len(self.draft_batch.reqs):
            if keep_indices:
                self.draft_batch.filter_batch(keep_indices=keep_indices)
            else:
                self.draft_batch = ScheduleBatch(reqs=[])

    def _merge_draft_into_running(self) -> None:
        self._filter_draft_batch()
        if self.draft_batch.is_empty():
            return

        if self.running_batch.is_empty():
            self.running_batch = self.draft_batch
        else:
            self.running_batch.merge_batch(self.draft_batch)

        self.draft_batch = ScheduleBatch(reqs=[])

    def _extract_paused_drafts_from_running(self) -> None:
        if self.running_batch.is_empty():
            return

        paused_indices: List[int] = []
        for i, req in enumerate(self.running_batch.reqs):
            if not getattr(req, "draft_is_paused", False):
                continue
            if req not in self.draft_paused_reqs:
                self.draft_paused_reqs.append(req)
            state = self._get_draft_state(req.rid)
            if state:
                state.location = DraftReqLocation.PAUSED
            paused_indices.append(i)

        if paused_indices:
            paused_set = set(paused_indices)
            keep = [
                i for i in range(len(self.running_batch.reqs)) if i not in paused_set
            ]
            self.running_batch.filter_batch(keep_indices=keep)

    def _prefill_draft_reqs(self) -> None:
        if not self.draft_waiting_queue:
            return

        candidate_reqs = list(self.draft_waiting_queue)
        if self._specstream_grants_enabled:
            granted_reqs = []
            for req in candidate_reqs:
                grant = self._grant_table.active(req.rid, spec_cnt=int(req.spec_cnt))
                if grant is None:
                    self._ack_expired_grant(req)
                    continue
                if grant.grant_state != "DRAFT_CATCHUP":
                    # The offline SLACK_FILL table is calibrated for one-token
                    # decode.  A re-prefill may be arbitrarily longer, so it is
                    # deferred until Target explicitly enters its wait phase.
                    if self._grant_table.consume_one(
                        req.rid,
                        spec_cnt=int(req.spec_cnt),
                        grant_epoch=grant.grant_epoch,
                    ):
                        self._send_grant_ack(
                            req,
                            grant,
                            0.0,
                            grant_tokens=0,
                            grant_state="PREFILL_DEFERRED",
                        )
                    continue
                granted_reqs.append(req)
            candidate_reqs = granted_reqs
            if not candidate_reqs:
                return

        for req in candidate_reqs:
            req.init_next_round_input(self.tree_cache)

        adder = self._build_prefill_adder_for_draft()
        for req in candidate_reqs:
            res = adder.add_one_req(
                req,
                has_chunked_req=False,
                truncation_align_size=getattr(self, "truncation_align_size", None),
            )
            if res != AddReqResult.CONTINUE:
                break

        admitted: List[Req] = adder.can_run_list
        grants = None
        if self._specstream_grants_enabled:
            grants = self._grant_table.active_for(
                ((req.rid, int(req.spec_cnt)) for req in admitted)
            )
            if len(grants) != len(admitted):
                # Leave every request in draft_waiting_queue.  In particular,
                # do not allocate a prefill batch after a deadline expires.
                return
        admitted_set = set(id(r) for r in admitted)
        self.draft_waiting_queue = [
            r for r in self.draft_waiting_queue if id(r) not in admitted_set
        ]
        if not admitted:
            return

        if self._specstream_smctrl_enabled_for_draft():
            _specstream_prefill_grant = self._specstream_poll_online_grant(admitted)
            if _specstream_prefill_grant is None:
                self.draft_waiting_queue = admitted + self.draft_waiting_queue
                return
            self._specstream_apply_grant_to_worker(_specstream_prefill_grant)
        else:
            _specstream_prefill_grant = None

        draft_prefill_batch = ScheduleBatch.init_new(
            admitted,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        draft_prefill_batch.prepare_for_extend()
        if self._specstream_grants_enabled:
            assert grants is not None
            self._apply_batch_grant_mask(grants.values())
        started = time.perf_counter()
        if _specstream_prefill_grant is None:
            result = self.run_batch(draft_prefill_batch)
        else:
            _specstream_prefill_success = False
            self._specstream_begin_controlled_forward()
            try:
                result = self.run_batch(draft_prefill_batch)
                _specstream_prefill_success = True
            finally:
                self._specstream_end_controlled_forward()
                self._specstream_finish_one_granted_step(
                    draft_prefill_batch, success=_specstream_prefill_success
                )
        if self._specstream_grants_enabled:
            self._synchronize_specstream_draft_stream()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            for req in admitted:
                grant = grants.get(req.rid)
                if grant is None:
                    continue
                if not self._grant_table.consume_one(
                    req.rid,
                    spec_cnt=int(req.spec_cnt),
                    grant_epoch=grant.grant_epoch,
                ):
                    raise RuntimeError(
                        f"Draft grant changed during prefill for request {req.rid}"
                    )
                self._send_grant_ack(req, grant, elapsed_ms)

        self._process_draft_prefill_result(draft_prefill_batch, result)
        if self._specstream_grants_enabled:
            for req in admitted:
                if not req.finished():
                    req.draft_is_paused = True
                    state = self._get_draft_state(req.rid)
                    if state is not None:
                        state.location = DraftReqLocation.PAUSED
                        state.last_updated_time = time.time()
                    if req not in self.draft_paused_reqs:
                        self.draft_paused_reqs.append(req)

        draft_prefill_batch.filter_batch()

        if not draft_prefill_batch.is_empty():
            if self.draft_batch.is_empty():
                self.draft_batch = draft_prefill_batch
            else:
                self.draft_batch.merge_batch(draft_prefill_batch)

            for req in draft_prefill_batch.reqs:
                state = self._get_draft_state(req.rid)
                if state:
                    state.location = (
                        DraftReqLocation.PAUSED
                        if getattr(req, "draft_is_paused", False)
                        else DraftReqLocation.DRAFT_BATCH
                    )

    def _process_draft_prefill_result(self, batch: ScheduleBatch, result) -> None:
        self.process_batch_result_prefill(batch, result)

    def _build_prefill_adder_for_draft(self) -> PrefillAdder:
        return PrefillAdder(
            page_size=self.page_size,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            running_batch=self.running_batch,
            new_token_ratio=self.new_token_ratio,
            rem_input_tokens=self.max_prefill_tokens,
            rem_chunk_tokens=None,
            mixed_with_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
        )

    def _add_req_to_draft_batch(self, req: Req) -> None:
        tmp_batch = self._build_decode_batch_from_reqs([req])
        if self.draft_batch.is_empty():
            self.draft_batch = tmp_batch
        else:
            self.draft_batch.merge_batch(tmp_batch)

    def _build_decode_batch_from_reqs(self, reqs: List[Req]) -> ScheduleBatch:
        try:
            device = self.tp_group.device
        except Exception:
            device = "cuda"

        def _seq_len(r: Req) -> int:
            return len(r.origin_input_ids) + len(r.output_ids) - 1

        seq_lens_list = [_seq_len(r) for r in reqs]

        try:
            from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator

            is_hybrid_swa = isinstance(
                self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
            )
        except ImportError:
            is_hybrid_swa = False

        batch = ScheduleBatch(
            reqs=list(reqs),
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            is_hybrid_swa=is_hybrid_swa,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
            forward_mode=ForwardMode.DECODE,
            device=device,
        )

        batch.req_pool_indices = torch.tensor(
            [r.req_pool_idx for r in reqs], dtype=torch.int64, device=device
        )
        batch.seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(
            seq_lens_list, dtype=torch.int32, device=device
        )
        batch.out_cache_loc = None
        batch.seq_lens_sum = sum(seq_lens_list)
        batch.output_ids = torch.tensor(
            [
                r.output_ids[-1] if r.output_ids else r.origin_input_ids[-1]
                for r in reqs
            ],
            dtype=torch.int64,
            device=device,
        )
        batch.return_logprob = any(r.return_logprob for r in reqs)
        batch.top_logprobs_nums = [
            r.top_logprobs_num if r.return_logprob else 0 for r in reqs
        ]
        batch.token_ids_logprobs = [
            r.token_ids_logprob if r.return_logprob else None for r in reqs
        ]
        batch.multimodal_inputs = [r.multimodal_inputs for r in reqs]
        batch.has_stream = any(r.stream for r in reqs)
        batch.has_grammar = any(r.grammar for r in reqs)
        batch.return_hidden_states = any(r.return_hidden_states for r in reqs)
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.model_config.vocab_size
        )

        return batch

    def recv_and_process_draft_requests(self) -> None:
        if self.tp_size == 1:
            if not hasattr(self, "zmq_communicator") or self.zmq_communicator is None:
                return

        if self._is_self_high_overhead_draft():
            if self.tp_size == 1 or self.tp_rank == 0:
                self._send_reject_message()
            return

        if self.tp_size == 1 or self.tp_rank == 0:
            messages = self._recv_draft_requests()
            if messages:
                logger.debug(
                    f"\033[32m [Draft][Recv] {len(messages)} messages from Target \033[0m"
                )
        else:
            messages = None

        if self.tp_size > 1:
            messages = broadcast_pyobj(
                messages if messages else [],
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )

        if not messages:
            return

        control_msgs, latest_msgs = self.deduplicate_draft_requests(messages)

        if not control_msgs and not latest_msgs:
            return

        self.token_to_kv_pool_allocator.free_group_begin()
        self._process_control_message(control_msgs)
        self._process_draft_requests(latest_msgs)
        self.token_to_kv_pool_allocator.free_group_end()
        self._flush_draft_batch_pending_adds()

    def _recv_draft_requests(self) -> List[SpectreRequest]:
        try:
            msgs: List[SpectreRequest] = []
            if hasattr(self, "zmq_communicator") and self.zmq_communicator is not None:
                msgs = self.zmq_communicator.recv_all_objs()
                if msgs:
                    more = self.zmq_communicator.recv_all_objs()
                    if more:
                        msgs.extend(more)
            return msgs
        except (ConnectionError, OSError) as e:
            if self.tp_rank == 0:
                logger.error(f"[Draft] Network error in recv: {e}", exc_info=True)
            return []
        except Exception as e:
            if self.tp_rank == 0:
                logger.error(f"[Draft] Unexpected recv error: {e}", exc_info=True)
            return []

    def deduplicate_draft_requests(
        self,
        messages: List[SpectreRequest],
    ) -> Tuple[List[SpectreRequest], Dict[str, SpectreRequest]]:
        latest_msgs: Dict[str, SpectreRequest] = {}
        control_msgs: List[SpectreRequest] = []

        for draft_req in messages:
            req_id = draft_req.request_id
            action = getattr(draft_req, "action", SpectreAction.DRAFT)

            if action in (
                SpectreAction.FINISH,
                SpectreAction.ABORT,
                SpectreAction.GRANT,
                SpectreAction.PAUSE,
            ):
                control_msgs.append(draft_req)
                continue

            if (
                req_id not in latest_msgs
                or draft_req.spec_cnt > latest_msgs[req_id].spec_cnt
            ):
                if req_id in latest_msgs and draft_req.input_ids is None:
                    draft_req.input_ids = latest_msgs[req_id].input_ids
                    draft_req.sampling_params = (
                        draft_req.sampling_params or latest_msgs[req_id].sampling_params
                    )
                latest_msgs[req_id] = draft_req

        total = len(messages)
        kept = len(latest_msgs) + len(control_msgs)
        if total > kept and self.tp_rank == 0:
            logger.debug(f"\033[36m [Draft][Dedup] {total} → {kept} \033[0m")

        return control_msgs, latest_msgs

    def _process_control_message(self, control_msgs: List[SpectreRequest]) -> None:
        for draft_req in control_msgs:
            action = draft_req.action
            if action in (SpectreAction.FINISH, SpectreAction.ABORT):
                if self.tp_rank == 0:
                    logger.debug(
                        f"[Draft] Received {action} for {draft_req.request_id}"
                    )
                self._grant_table.release(draft_req.request_id)
                self._finish_draft_request(draft_req.request_id)
            elif action == SpectreAction.GRANT:
                if not self._specstream_grants_enabled:
                    continue
                try:
                    grant = DraftExecutionGrant(
                        request_id=str(draft_req.request_id),
                        spec_cnt=int(draft_req.spec_cnt or 0),
                        grant_epoch=int(draft_req.grant_epoch or 0),
                        grant_tokens=int(draft_req.grant_tokens or 0),
                        tpc_low=int(draft_req.tpc_low or 0),
                        tpc_high=int(draft_req.tpc_high or 0),
                        deadline_us=(
                            int(draft_req.deadline_us)
                            if draft_req.deadline_us is not None
                            else None
                        ),
                        placement_id=int(draft_req.placement_id or 0),
                        grant_state=str(draft_req.grant_state or ""),
                    )
                except (TypeError, ValueError) as exc:
                    if self.tp_rank == 0:
                        logger.warning("[Draft][Grant] invalid grant ignored: %s", exc)
                    continue
                applied = self._grant_table.apply(grant)
                if not applied.accepted:
                    if self.tp_rank == 0:
                        logger.debug(
                            "[Draft][Grant] ignored %s grant rid=%s epoch=%s",
                            applied.reason,
                            grant.request_id,
                            grant.grant_epoch,
                        )
                    continue
                state = self._get_draft_state(grant.request_id)
                if (
                    state is not None
                    and state.req_object is not None
                    and state.location == DraftReqLocation.PAUSED
                    and int(state.req_object.spec_cnt) == grant.spec_cnt
                ):
                    self._resume_draft_req(state.req_object, state)
            elif action == SpectreAction.PAUSE:
                if not self._specstream_grants_enabled:
                    continue
                self._grant_table.pause(
                    str(draft_req.request_id),
                    spec_cnt=(
                        int(draft_req.spec_cnt)
                        if draft_req.spec_cnt is not None
                        else None
                    ),
                    grant_epoch=(
                        int(draft_req.grant_epoch)
                        if draft_req.grant_epoch is not None
                        else None
                    ),
                )
                state = self._get_draft_state(str(draft_req.request_id))
                if state is not None and state.req_object is not None:
                    self._pause_req(state.req_object, state)

    def _process_draft_requests(self, latest_msgs: Dict[str, SpectreRequest]) -> None:
        for req_id, draft_req in latest_msgs.items():
            try:
                if self.tp_rank == 0 and (draft_req.spec_cnt or 0) <= 0:
                    logger.info(
                        "[Draft][DraftLink] received initial request rid=%s "
                        "q=%s input_len=%d",
                        req_id,
                        draft_req.num_draft_tokens,
                        len(draft_req.input_ids or []),
                    )
                state = self._get_draft_state(req_id)

                if state is None:
                    if draft_req.input_ids is None:
                        if self.tp_rank == 0:
                            logger.warning(
                                f"[Draft] {req_id}: no state and no input_ids, skipping"
                            )
                        continue
                    self._create_new_draft_req(draft_req)
                    continue

                state.last_updated_time = time.time()
                req = state.req_object

                if req is None:
                    if self.tp_rank == 0:
                        logger.warning(f"[Draft] {req_id} has None req_object")
                    self._finish_draft_request(req_id)
                    continue

                req.target_send_time = draft_req.target_send_time
                req.draft_recv_time = draft_req.draft_recv_time

                if draft_req.input_ids is not None:
                    target_input = draft_req.input_ids
                    state.target_origin_input_ids = list(target_input)
                else:
                    target_input = state.target_origin_input_ids or []

                target_fill_ids: List[int] = (
                    target_input
                    + (draft_req.output_ids or [])
                    + (draft_req.draft_token_ids or [])
                )
                draft_fill_ids: List[int] = (req.origin_input_ids or []) + (
                    req.output_ids or []
                )

                if not target_fill_ids:
                    continue

                skip = len(target_input)
                if (
                    skip > 0
                    and len(draft_fill_ids) >= skip
                    and len(target_fill_ids) >= skip
                ):
                    is_identical, fork_offset = self._find_fork_point(
                        draft_fill_ids[skip:], target_fill_ids[skip:]
                    )
                    fork_point = skip + fork_offset
                else:
                    is_identical, fork_point = self._find_fork_point(
                        draft_fill_ids, target_fill_ids
                    )

                if is_identical:
                    self._handle_identical_tokens(req, draft_req, state)
                else:
                    self._handle_divergence(
                        req, target_fill_ids, fork_point, draft_req, state
                    )

            except Exception as e:
                if self.tp_rank == 0:
                    logger.error(
                        f"\033[31m [Draft] Error processing {req_id}: {e} \033[0m",
                        exc_info=True,
                    )
                try:
                    self._finish_draft_request(req_id)
                except Exception:
                    pass

    def _find_fork_point(
        self,
        draft_ids: List[int],
        target_ids: List[int],
    ) -> Tuple[bool, int]:
        min_len = min(len(draft_ids), len(target_ids))
        for i in range(min_len):
            if draft_ids[i] != target_ids[i]:
                return (False, i)
        return (len(draft_ids) == len(target_ids), min_len)

    def _handle_identical_tokens(
        self,
        req: Req,
        draft_req: SpectreRequest,
        state: SpectreDraftState,
    ) -> None:
        if self.tp_rank == 0:
            logger.debug(
                f"\033[34m [Draft][NoChange] {req.rid}, "
                f"spec_cnt={draft_req.spec_cnt}, location={state.location} \033[0m"
            )
        self._update_req_state(req, draft_req, state)
        self._resume_or_update(req, state, tokens_changed=False)

    def _handle_divergence(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        draft_req: SpectreRequest,
        state: SpectreDraftState,
    ) -> None:
        state.last_updated_time = time.time()

        current_len = len(req.origin_input_ids) + len(req.output_ids)
        target_len = len(target_fill_ids)
        current_kv_len = current_len - 1
        needs_kv_release = fork_point < current_kv_len

        if current_len == target_len:
            self._handle_equal_length(
                req,
                target_fill_ids,
                fork_point,
                current_len,
                current_kv_len,
                needs_kv_release,
                draft_req,
                state,
            )
        elif current_len > target_len:
            self._handle_draft_ahead(
                req,
                target_fill_ids,
                fork_point,
                current_kv_len,
                needs_kv_release,
                draft_req,
                state,
            )
        else:
            self._handle_target_ahead(
                req,
                target_fill_ids,
                fork_point,
                current_len,
                current_kv_len,
                needs_kv_release,
                draft_req,
                state,
            )

    def _handle_equal_length(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        current_len: int,
        current_kv_len: int,
        needs_kv_release: bool,
        draft_req: SpectreRequest,
        state: SpectreDraftState,
    ) -> None:
        if fork_point == current_len - 1:
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[36m [Case 1.1] {req.rid=}, {draft_req.spec_cnt=}, "
                    f"replace last token \033[0m"
                )
            self._update_tokens(req, fork_point, target_fill_ids[fork_point:])
            self._update_req_state(req, draft_req, state)
            self._resume_or_update(req, state)
        else:
            self._handle_multi_token_divergence(
                req,
                target_fill_ids,
                fork_point,
                current_kv_len,
                needs_kv_release,
                draft_req,
                state,
                "1.2",
            )

    def _handle_draft_ahead(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        current_kv_len: int,
        needs_kv_release: bool,
        draft_req: SpectreRequest,
        state: SpectreDraftState,
    ) -> None:
        target_len = len(target_fill_ids)

        if fork_point == target_len:
            target_output_len = target_len - len(req.origin_input_ids)
            draft_output_len = len(req.output_ids)
            tokens_ahead = draft_output_len - target_output_len

            if self.tp_rank == 0:
                logger.debug(
                    f"\033[36m [Case 2.1] {req.rid=}, {draft_req.spec_cnt=}, "
                    f"draft ahead by {tokens_ahead} token(s) \033[0m"
                )

            req.draft_generation_start_len = target_output_len
            req.spec_cnt = draft_req.spec_cnt
            req.draft_tokens_target = draft_req.num_draft_tokens
            req.len_output_ids = draft_output_len
            state.last_updated_time = time.time()

            if tokens_ahead >= draft_req.num_draft_tokens:
                if self.tp_rank == 0:
                    logger.debug(
                        f"\033[33m [Case 2.1] {req.rid=}: already {tokens_ahead} ahead, "
                        f"sending immediately \033[0m"
                    )
                self._send_draft_response(req)
                self._pause_req(req, state)
            else:
                req.draft_is_paused = False
                self._resume_or_update(req, state)
        else:
            self._handle_multi_token_divergence(
                req,
                target_fill_ids,
                fork_point,
                current_kv_len,
                needs_kv_release,
                draft_req,
                state,
                "2.2",
            )

    def _handle_target_ahead(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        current_len: int,
        current_kv_len: int,
        needs_kv_release: bool,
        draft_req: SpectreRequest,
        state: SpectreDraftState,
    ) -> None:
        if fork_point == current_len:
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[36m [Case 3.1] {req.rid=}, {draft_req.spec_cnt=}, "
                    f"re-prefill for extend \033[0m"
                )
        else:
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[36m [Case 3.2] {req.rid=}, {draft_req.spec_cnt=}, "
                    f"re-prefill for extend+rollback \033[0m"
                )
        self._prepare_for_reprefill(req, target_fill_ids, draft_req, state)

    def _handle_multi_token_divergence(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        current_kv_len: int,
        needs_kv_release: bool,
        draft_req: SpectreRequest,
        state: SpectreDraftState,
        case_name: str,
    ) -> None:
        new_len = len(target_fill_ids)
        can_decode_after_rollback = new_len - 1 <= fork_point

        if can_decode_after_rollback and self.draft_kv_manager.can_local_rollback(
            req, fork_point
        ):
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[35m [Case {case_name}] {req.rid=}, {draft_req.spec_cnt=} "
                    f"→ local rollback + decode \033[0m"
                )
            if needs_kv_release:
                self.draft_kv_manager.local_rollback(req, fork_point, current_kv_len)
            self._update_tokens(req, fork_point, target_fill_ids[fork_point:])
            self._update_req_state(req, draft_req, state)
            self._resume_or_update(req, state)
        else:
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[36m [Case {case_name}] {req.rid=}, {draft_req.spec_cnt=} "
                    f"→ re-prefill \033[0m"
                )
            self._prepare_for_reprefill(req, target_fill_ids, draft_req, state)

    def _update_tokens(
        self, req: Req, fork_point: int, delta_tokens: List[int]
    ) -> None:
        truncate_point = fork_point - len(req.origin_input_ids)
        req.output_ids = req.output_ids[: max(0, truncate_point)]
        req.output_ids.extend(delta_tokens)
        req.fill_ids = req.origin_input_ids + req.output_ids

    def _update_req_state(
        self,
        req: Req,
        draft_req: SpectreRequest,
        state: SpectreDraftState,
    ) -> None:
        req.spec_cnt = draft_req.spec_cnt
        req.draft_tokens_target = draft_req.num_draft_tokens
        req.draft_generation_start_len = len(req.output_ids)
        req.draft_is_paused = False
        req.len_output_ids = len(req.output_ids)
        state.last_updated_time = time.time()

    def _resume_or_update(
        self,
        req: Req,
        state: SpectreDraftState,
        tokens_changed: bool = True,
    ) -> None:
        location = state.location

        if location == DraftReqLocation.PAUSED:
            self._resume_draft_req(req, state)
        elif location == DraftReqLocation.DRAFT_BATCH:
            if tokens_changed:
                self._rebuild_req_in_draft_batch(req, state)

    def _resume_draft_req(self, req: Req, state: SpectreDraftState) -> None:
        if req in self.draft_paused_reqs:
            self.draft_paused_reqs.remove(req)

        if req.req_pool_idx is None:
            if self.tp_rank == 0:
                logger.warning(
                    f"[Draft][Resume] {req.rid} has no KV pool slot, "
                    f"falling back to re-prefill"
                )
            req.draft_is_paused = False
            state.location = DraftReqLocation.DRAFT_WAITING
            if req not in self.draft_waiting_queue:
                self.draft_waiting_queue.append(req)
            return

        req.draft_is_paused = False
        state.location = DraftReqLocation.DRAFT_BATCH
        if req not in self._draft_batch_pending_adds:
            self._draft_batch_pending_adds.append(req)

        if self.tp_rank == 0:
            logger.debug(f"[Draft][Resume] {req.rid} → draft_batch (pending)")

    def _rebuild_req_in_draft_batch(self, req: Req, state: SpectreDraftState) -> None:
        if not self.draft_batch.is_empty() and req in self.draft_batch.reqs:
            self.draft_batch.filter_batch(chunked_req_to_exclude=[req])

        if req.req_pool_idx is not None:
            req.draft_is_paused = False
            state.location = DraftReqLocation.DRAFT_BATCH
            if req not in self._draft_batch_pending_adds:
                self._draft_batch_pending_adds.append(req)
        else:
            state.location = DraftReqLocation.DRAFT_WAITING
            if req not in self.draft_waiting_queue:
                self.draft_waiting_queue.append(req)

    def _flush_draft_batch_pending_adds(self) -> None:
        if not self._draft_batch_pending_adds:
            return

        new_batch = self._build_decode_batch_from_reqs(self._draft_batch_pending_adds)
        self._draft_batch_pending_adds.clear()

        if self.draft_batch.is_empty():
            self.draft_batch = new_batch
        else:
            self.draft_batch.merge_batch(new_batch)

    def _pause_req(self, req: Req, state: SpectreDraftState) -> None:
        if not self.draft_batch.is_empty() and req in self.draft_batch.reqs:
            self.draft_batch.filter_batch(chunked_req_to_exclude=[req])

        req.draft_is_paused = True
        if req not in self.draft_paused_reqs:
            self.draft_paused_reqs.append(req)
        state.location = DraftReqLocation.PAUSED
        state.last_updated_time = time.time()

    def _reset_req_logprob_fields(self, req: Req) -> None:
        req.input_token_logprobs_val = None
        req.input_token_logprobs_idx = None
        req.input_top_logprobs_val = None
        req.input_top_logprobs_idx = None
        req.input_token_ids_logprobs_val = None
        req.input_token_ids_logprobs_idx = None
        req.input_token_logprobs = None
        req.temp_input_top_logprobs_val = None
        req.temp_input_top_logprobs_idx = None
        req.temp_input_token_ids_logprobs_val = None
        req.temp_input_token_ids_logprobs_idx = None
        req.input_logprob_sent = False

        req.output_token_logprobs_val = []
        req.output_token_logprobs_idx = []
        req.output_top_logprobs_val = []
        req.output_top_logprobs_idx = []
        req.output_token_ids_logprobs_val = []
        req.output_token_ids_logprobs_idx = []

    def _remove_draft_req(self, req: Req) -> None:
        if req in self.draft_paused_reqs:
            self.draft_paused_reqs.remove(req)

        if not self.draft_batch.is_empty() and req in self.draft_batch.reqs:
            self.draft_batch.filter_batch(chunked_req_to_exclude=[req])

        if req in self.draft_waiting_queue:
            self.draft_waiting_queue.remove(req)

        if req in self._draft_batch_pending_adds:
            self._draft_batch_pending_adds.remove(req)

    def _prepare_for_reprefill(
        self,
        req: Req,
        target_fill_ids: List[int],
        draft_req: SpectreRequest,
        state: SpectreDraftState,
    ) -> None:
        if self.tp_rank == 0:
            logger.debug(
                f"[Draft][RePrefill] {req.rid=}, {draft_req.spec_cnt=}, "
                f"new_len={len(target_fill_ids)}"
            )

        self._remove_draft_req(req)
        if req.req_pool_idx is not None:
            self.draft_kv_manager.release_all_kv_for_reprefill_req(req)

        req.fill_ids = target_fill_ids
        req.origin_input_ids = list(target_fill_ids)
        req.output_ids = []
        req.prefix_indices = []
        req.extend_input_len = len(req.fill_ids)

        req.spec_cnt = draft_req.spec_cnt
        req.draft_tokens_target = draft_req.num_draft_tokens
        req.draft_generation_start_len = 0
        req.draft_is_paused = False
        req.len_output_ids = 0

        req.last_node = None
        req.kv_committed_len = 0
        req.kv_committed_freed = False
        req.kv_overallocated_freed = False

        state.location = DraftReqLocation.DRAFT_WAITING
        req.logprob_start_len = len(req.origin_input_ids) - 1
        self._reset_req_logprob_fields(req)

        self.draft_waiting_queue.append(req)

    def _check_and_pause_draft_req(self, req: Req) -> bool:
        if getattr(req, "spec_type", None) != SpecType.DRAFT_REQUEST:
            return False

        if req.draft_is_paused:
            return True

        tokens_generated = len(req.output_ids) - req.draft_generation_start_len

        if tokens_generated >= req.draft_tokens_target:
            self._send_draft_response(req)

            req.draft_is_paused = True
            if req not in self.draft_paused_reqs:
                self.draft_paused_reqs.append(req)

            state = self._get_draft_state(req.rid)
            if state:
                state.location = DraftReqLocation.PAUSED
                state.last_updated_time = time.time()

            return True

        return False

    def _send_draft_response(self, req: Req) -> None:
        draft_tokens = req.output_ids[req.draft_generation_start_len :]

        draft_logits: List[float] = []
        if hasattr(req, "output_token_logprobs_val") and req.output_token_logprobs_val:
            start = req.draft_generation_start_len
            end = start + len(draft_tokens)
            if len(req.output_token_logprobs_val) >= end:
                draft_logits = req.output_token_logprobs_val[start:end]

        response = SpectreRequest(
            request_id=req.rid,
            spec_cnt=req.spec_cnt,
            action=SpectreAction.DRAFT,
            spec_type=SpecType.DRAFT_RESPONSE,
            draft_token_ids=draft_tokens,
            draft_logprobs=draft_logits or [],
            target_send_time=req.target_send_time,
            draft_recv_time=req.draft_recv_time,
        )

        if self.tp_size == 1 or self.tp_rank == 0:
            if hasattr(self, "zmq_communicator") and self.zmq_communicator is not None:
                self.zmq_communicator.send_objs([response])
                if (req.spec_cnt or 0) <= 0:
                    logger.info(
                        "[Draft][DraftLink] sent initial response rid=%s "
                        "spec_cnt=%s tokens=%d",
                        req.rid,
                        req.spec_cnt,
                        len(draft_tokens),
                    )

        req.draft_generation_start_len = len(req.output_ids)

    def _create_new_draft_req(self, draft_req: SpectreRequest) -> None:
        req_id = draft_req.request_id

        if self._exists_draft_state(req_id):
            self._finish_draft_request(req_id)

        input_ids: List[int] = (
            (draft_req.input_ids or [])
            + (draft_req.output_ids or [])
            + (draft_req.draft_token_ids or [])
        )

        if draft_req.sampling_params is None:
            from sglang.srt.sampling.sampling_params import SamplingParams

            sampling_params = SamplingParams()
        else:
            sampling_params = draft_req.sampling_params

        if hasattr(sampling_params, "normalize"):
            try:
                sampling_params.normalize(self.tokenizer)
            except Exception as e:
                if self.tp_rank == 0:
                    logger.warning(
                        f"[Draft] Failed to normalize SamplingParams for {req_id}: {e}, "
                        f"applying manual fix"
                    )
                _fix_sampling_params_stop_strs(sampling_params)
        else:
            _fix_sampling_params_stop_strs(sampling_params)

        req = Req(
            rid=req_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=True,
            top_logprobs_num=1,
            token_ids_logprob=None,
            stream=False,
            lora_id=None,
            input_embeds=None,
            custom_logit_processor=None,
            return_hidden_states=False,
            eos_token_ids=self.model_config.hf_eos_token_id,
            bootstrap_host=None,
            bootstrap_port=8998,
            bootstrap_room=None,
            vocab_size=self.model_config.vocab_size,
        )

        req.spec_cnt = draft_req.spec_cnt
        req.spec_type = SpecType.DRAFT_REQUEST
        req.tokenizer = self.tokenizer
        req.logprob_start_len = len(req.origin_input_ids) - 1
        req.draft_tokens_target = draft_req.num_draft_tokens
        req.draft_generation_start_len = 0
        req.draft_is_paused = False
        req.len_output_ids = 0
        req.target_send_time = draft_req.target_send_time
        req.draft_recv_time = draft_req.draft_recv_time

        self.draft_waiting_queue.append(req)

        self._set_draft_state(
            req_id,
            SpectreDraftState(
                req_id=req_id,
                spec_cnt=draft_req.spec_cnt,
                req_object=req,
                location=DraftReqLocation.DRAFT_WAITING,
                target_origin_input_ids=(
                    list(draft_req.input_ids) if draft_req.input_ids else []
                ),
                last_prefix_length=len(input_ids),
                last_output_length=0,
            ),
        )

        if self.tp_rank == 0:
            logger.debug(
                f"[Draft][New] {req_id=}, {req.spec_cnt=}, input_len={len(input_ids)}"
            )

    def _finish_draft_request(self, req_id: str) -> None:
        if hasattr(self, "_grant_table"):
            self._grant_table.release(req_id)
        state = self._get_draft_state(req_id)
        if state is None:
            return

        req = state.req_object
        self._remove_draft_req(req)

        if not req.finished():
            req.to_abort = True
            req.finished_reason = FINISH_ABORT("Target request finished")

        if req.req_pool_idx is not None and not getattr(
            req, "kv_committed_freed", False
        ):
            self.draft_kv_manager.release_all_kv_for_finished_req(req)

        self._delete_draft_state(req_id)
        if self.tp_rank == 0:
            logger.debug(f"[Draft][Finish] {req_id=}")

    def _cleanup_stale_draft_states(self) -> None:
        for req_id in self.draft_state_manager.cleanup_stale_states():
            try:
                self._finish_draft_request(req_id)
            except Exception as e:
                if self.tp_rank == 0:
                    logger.warning(f"[Draft] Cleanup failed for {req_id=}: {e}")
            finally:
                try:
                    self._delete_draft_state(req_id)
                except Exception:
                    pass

    def _is_self_high_overhead_draft(self) -> bool:
        if not hasattr(self, "running_batch") or self.running_batch is None:
            return False
        return self.running_batch.batch_size() > self.server_args.spectre_max_batch_size

    def _send_reject_message(self) -> None:
        if self.tp_size > 1 and self.tp_rank != 0:
            return
        if not hasattr(self, "zmq_communicator") or self.zmq_communicator is None:
            return

        reject_msg = SpectreRequest(
            request_id="system",
            spec_cnt=0,
            action=SpectreAction.REJECT,
            spec_type=SpecType.DRAFT_RESPONSE,
            draft_token_ids=[],
            draft_logprobs=[],
        )
        self.zmq_communicator.send_objs([reject_msg])
        if self.tp_rank == 0:
            logger.debug("[Draft] Sent REJECT to Target (high load)")

    def get_num_allocatable_reqs(self, running_bs: int) -> int:
        paused_id_set: set = set()
        paused_reqs_lock = getattr(self, "paused_reqs_lock", None)
        if paused_reqs_lock is not None and hasattr(self, "paused_reqs"):
            try:
                with paused_reqs_lock:
                    paused_id_set.update(id(r) for r in self.paused_reqs)
            except Exception:
                pass
        if hasattr(self, "draft_paused_reqs"):
            paused_id_set.update(id(r) for r in self.draft_paused_reqs)

        paused_count = len(paused_id_set)
        total_occupied = running_bs + paused_count

        res = self.server_args.pp_max_micro_batch_size - total_occupied
        if self.pp_size > 1:
            res = min(res, self.req_to_token_pool.available_size())
        return res
