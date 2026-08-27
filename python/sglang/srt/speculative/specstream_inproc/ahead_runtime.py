from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.specstream_inproc.ahead_state import (
    AheadRequestState,
    ReconcileResult,
)
from sglang.srt.speculative.specstream_inproc.profiler import (
    InProcessAheadProfiler,
)
from sglang.srt.speculative.specstream_inproc.resource_controller import (
    CoexecutionAction,
    CoexecutionMode,
    InProcessResourceController,
)
from sglang.srt.speculative.specstream_inproc.rollback_manager import (
    DraftRollbackManager,
    RollbackPlan,
)
from sglang.srt.speculative.specstream_inproc.stream_runtime import (
    InProcessStreamRuntime,
    RoundEvents,
    RoundTiming,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleVerifyInput
    from sglang.srt.speculative.standalone_worker_v2 import StandaloneDraftWorker

logger = logging.getLogger(__name__)


@dataclass
class DraftAheadArtifact:
    next_draft_input: Any
    next_batch: Any
    assumed_bonus: torch.Tensor
    ahead_tokens: torch.Tensor
    verify_input: Any = None
    partial_progress: Any = None

    @property
    def is_complete(self) -> bool:
        return self.verify_input is not None


@dataclass
class AheadRound:
    batch_key: tuple[str, ...]
    expected_next_seq_lens: tuple[int, ...]
    batch_snapshot: Any
    verify_input: Any
    current_candidates: torch.Tensor
    states: list[AheadRequestState]
    action: CoexecutionAction
    events: RoundEvents
    artifact: Optional[DraftAheadArtifact] = None
    results: list[ReconcileResult] = field(default_factory=list)
    rollback_plans: list[RollbackPlan] = field(default_factory=list)
    timing: Optional[RoundTiming] = None


@dataclass(frozen=True)
class BatchReconcileOutcome:
    promoted: bool
    results: tuple[ReconcileResult, ...]
    timing: RoundTiming


class InProcessAheadRuntime:
    """Coordinate optimistic same-request Draft-ahead inside one worker."""

    def __init__(self, server_args: Any) -> None:
        self.server_args = server_args
        self.streams = InProcessStreamRuntime(server_args.device)
        self.controller = InProcessResourceController(
            configured_mode=server_args.specstream_inproc_mode,
            ahead_depth=server_args.specstream_inproc_ahead_depth,
            min_reuse_ratio=server_args.specstream_inproc_min_reuse_ratio,
            target_slowdown_budget=(
                server_args.specstream_inproc_target_slowdown_budget
            ),
        )
        self.rollback = DraftRollbackManager(
            page_size=server_args.page_size,
            shared_allocator=True,
        )
        self.profiler = InProcessAheadProfiler(
            profile_path=server_args.specstream_inproc_profile_path,
            flush_interval=server_args.specstream_inproc_profile_interval,
        )
        self._states: dict[str, AheadRequestState] = {}
        self._cached: Optional[AheadRound] = None

    @staticmethod
    def _batch_key(batch: ModelWorkerBatch) -> tuple[str, ...] | None:
        if not batch.reqs:
            return None
        return tuple(str(req.rid) for req in batch.reqs)

    def can_attempt(self, batch: ModelWorkerBatch) -> bool:
        if batch.forward_mode.is_idle() or batch.has_grammar:
            return False
        if not batch.sampling_info.is_all_greedy:
            return False
        return self._batch_key(batch) is not None

    def prepare_round(
        self,
        batch: ModelWorkerBatch,
        verify_input: EagleVerifyInput,
    ) -> AheadRound | None:
        action = self.controller.choose()
        if action.mode == CoexecutionMode.SERIAL or not self.can_attempt(batch):
            return None

        batch_key = self._batch_key(batch)
        assert batch_key is not None
        batch_size = len(batch.seq_lens)
        token_rows = verify_input.draft_token.reshape(
            batch_size, verify_input.draft_token_num
        )
        if token_rows.shape[1] != self.server_args.speculative_num_steps + 1:
            logger.warning(
                "SpecStream in-process round fell back to serial because the "
                "linear verify shape changed: shape=%s",
                tuple(token_rows.shape),
            )
            return None

        # The first token is the carried verified token; the remaining q tokens
        # are the current verification frontier.
        candidates = token_rows[:, 1:]
        snapshot = copy.copy(batch)
        snapshot.forward_mode = ForwardMode.DECODE
        snapshot.seq_lens = batch.seq_lens.clone()
        snapshot.seq_lens_cpu = batch.seq_lens_cpu.clone()
        snapshot.seq_lens_sum = int(batch.seq_lens_sum)

        states: list[AheadRequestState] = []
        committed_lens = snapshot.seq_lens_cpu.tolist()
        for index, request_id in enumerate(batch_key):
            state = self._states.setdefault(
                request_id, AheadRequestState(request_id=request_id)
            )
            # Candidate tensors are intentionally copied only at reconciliation,
            # after candidate_ready has completed, to avoid a pre-Verify sync.
            state.prepare_round(
                committed_len=int(committed_lens[index]),
                verification_tokens=[],
                checkpoint_kv_len=int(committed_lens[index]),
            )
            states.append(state)

        return AheadRound(
            batch_key=batch_key,
            expected_next_seq_lens=tuple(
                int(length) + verify_input.draft_token_num for length in committed_lens
            ),
            batch_snapshot=snapshot,
            verify_input=verify_input,
            current_candidates=candidates,
            states=states,
            action=action,
            events=self.streams.new_round(),
        )

    def launch_ahead(
        self,
        round_state: AheadRound,
        draft_worker: StandaloneDraftWorker,
    ) -> None:
        with self.streams.draft_context(round_state.events):
            round_state.artifact = draft_worker.launch_specstream_ahead(
                round_state.batch_snapshot,
                round_state.verify_input,
                round_state.action.ahead_depth,
            )

    def reconcile(
        self,
        round_state: AheadRound,
        batch_result: GenerationBatchResult,
    ) -> BatchReconcileOutcome:
        artifact = round_state.artifact
        if artifact is None:
            raise RuntimeError("ahead round has no launched artifact")
        accept_indices = getattr(batch_result, "specstream_accept_indices", None)
        if accept_indices is None:
            raise RuntimeError("Target verify result did not retain accept indices")

        verify_done = batch_result.next_draft_input.verify_done
        self.streams.wait_for_reconciliation(round_state.events, verify_done)

        current_candidates = round_state.current_candidates.detach().cpu().tolist()
        ahead_tokens = artifact.ahead_tokens.detach().cpu().tolist()
        accept_indices_cpu = accept_indices.detach().cpu().tolist()
        predict_cpu = batch_result.next_token_ids.detach().cpu().tolist()

        results: list[ReconcileResult] = []
        plans: list[RollbackPlan] = []
        for index, state in enumerate(round_state.states):
            state.verification_tokens = [int(x) for x in current_candidates[index]]
            state.verify_end = state.verify_start + len(state.verification_tokens)
            state.ahead_start = state.verify_end
            state.launch_ahead(ahead_tokens[index])

            authoritative = [
                int(predict_cpu[token_index])
                for token_index in accept_indices_cpu[index]
                if token_index >= 0
            ]
            result = state.reconcile(authoritative)
            results.append(result)
            plans.append(
                self.rollback.plan(
                    result,
                    checkpoint_kv_len=state.checkpoint_kv_len,
                    speculative_end=state.ahead_end,
                )
            )

        round_state.results = results
        round_state.rollback_plans = plans
        promoted = bool(results) and all(result.promotable for result in results)
        if promoted:
            self._cached = round_state

        timing = self.streams.timing(round_state.events, verify_done)
        round_state.timing = timing
        generated = sum(result.ahead_generated for result in results)
        reused = sum(result.ahead_reused for result in results)
        reuse_ratio = reused / generated if generated else 0.0
        self.controller.observe(
            reuse_ratio=reuse_ratio,
            target_slowdown=None,
            ahead_depth=round_state.action.ahead_depth,
            verify_ms=timing.verify_ms,
            ahead_ms=timing.ahead_ms,
        )
        if promoted:
            self._record_profile(round_state, timing)
        return BatchReconcileOutcome(
            promoted=promoted,
            results=tuple(results),
            timing=timing,
        )

    def consume_cached(
        self,
        batch: ModelWorkerBatch,
        draft_worker: StandaloneDraftWorker,
    ) -> EagleVerifyInput | None:
        cached = self._cached
        self._cached = None
        if cached is None or cached.artifact is None:
            return None
        current_seq_lens = tuple(int(length) for length in batch.seq_lens_cpu.tolist())
        if (
            self._batch_key(batch) != cached.batch_key
            or current_seq_lens != cached.expected_next_seq_lens
            or not self.can_attempt(batch)
        ):
            return None

        artifact = cached.artifact
        if artifact.is_complete:
            self.streams.wait_target_for_candidates(cached.events.ahead_done)
            return artifact.verify_input

        # Complete only the non-overlapped suffix of an h<q draft.  It remains
        # on the Draft stream; Target waits on one candidate-ready event.
        with self.streams.device_module.stream(self.streams.draft_stream):
            self.streams.draft_stream.wait_event(cached.events.ahead_done)
            verify_input = draft_worker.complete_specstream_ahead(artifact)
            candidate_ready = self.streams.device_module.Event(enable_timing=False)
            candidate_ready.record(self.streams.draft_stream)
        self.streams.wait_target_for_candidates(candidate_ready)
        return verify_input

    def mark_repaired(self, round_state: AheadRound) -> None:
        for state, result in zip(round_state.states, round_state.results):
            state.mark_repaired(state.verify_start + len(result.authoritative_tokens))
        self.streams.record_repair_done(round_state.events)
        if round_state.timing is None:
            raise RuntimeError("repair finished before round timing was recorded")
        timing = self.streams.repair_timing(round_state.events, round_state.timing)
        round_state.timing = timing
        self.controller.observe_repair(timing.repair_ms)
        self._record_profile(round_state, timing)

    def begin_repair(self, round_state: AheadRound) -> None:
        self.streams.record_repair_start(round_state.events)

    def _record_profile(self, round_state: AheadRound, timing: RoundTiming) -> None:
        self.profiler.record(
            batch_size=len(round_state.states),
            ahead_depth=round_state.action.ahead_depth,
            results=round_state.results,
            timing=timing,
            target_slowdown=None,
        )

    def discard_cached(self) -> None:
        self._cached = None

    def clear(self) -> None:
        self._cached = None
        self._states.clear()
        self.profiler.flush()
