"""Native speculative lookahead with the same split Draft attention as decoding."""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass

import numpy as np
import torch

from sglang.srt.distributed.parallel_state import make_lookahead_tp_group, use_lookahead_tp_group
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionMetadata
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient
from sglang.srt.speculative.native_draft_spill import DraftAttentionPlan
from sglang.srt.speculative.native_lookahead_admission import LookaheadAdmission
from sglang.srt.speculative.spec_utils import fast_topk


@dataclass
class _ReadyCandidates:
    request_state: tuple
    tokens: torch.Tensor


@dataclass
class _PartialCandidates:
    tokens: torch.Tensor
    physical: np.ndarray


@dataclass
class _Trim:
    keep: list[int]
    tail: torch.Tensor
    final: torch.Tensor
    live_reqs: list
    primed: dict


def _request_state(reqs):
    return tuple((req.rid, len(req.output_ids)) for req in reqs)


def resolve_native_suffix(current, bonus, verified, *, finished):
    """Return the safe KV prefix length and whether the entire branch matched."""
    matched = list(verified) == [*current, bonus]
    if finished:
        return 0, matched
    if not verified or len(verified) > len(current) + 1:
        raise ValueError("Invalid output suffix after verification")
    committed = 0
    # The final verified token is a correction or bonus. Leave it for Draft
    # extend unless both the original candidates and predicted bonus matched.
    for accept, candidate in zip(verified[:-1], current):
        if accept != candidate:
            break
        committed += 1
    return committed, matched


class NativeRoundParallel:
    def __init__(self, worker, enabled, trace):
        self.worker = worker
        self.spill = worker._native_draft_spill
        self.enabled = enabled
        self.device = worker.draft_model_runner.device
        self.stream = torch.cuda.Stream(device=self.device) if enabled else None
        sm_limit = int(os.getenv("SPECSTREAM_NATIVE_LOOKAHEAD_SM_LIMIT", "0"))
        if enabled and sm_limit:
            from sgl_kernel import spatial

            device_id = torch.cuda.current_device()
            available = spatial.get_sm_available(device_id)
            if sm_limit < 4 or sm_limit % 2 or available - sm_limit < 4:
                raise ValueError("Expected a valid Ampere Draft SM limit")
            self._spatial_streams = spatial.create_greenctx_stream_by_value(
                available - sm_limit, sm_limit, device_id,
            )
            self.stream = self._spatial_streams[1]
        self.tp_group = make_lookahead_tp_group(worker.target_worker.model_runner.tp_group) if enabled else None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="draft-lookahead") if enabled else None
        self.process_executor = None
        if enabled and os.getenv("SPECSTREAM_NATIVE_LOOKAHEAD_PROCESS", "0") == "1":
            from sglang.srt.speculative.native_mps_draft_executor import NativeMPSDraftExecutor

            if sm_limit or os.getenv("SPECSTREAM_NATIVE_JOINT_KV_GIB"):
                raise ValueError("Native MPS lookahead requires unlimited SMs and a fixed KV layout")
            self.process_executor = NativeMPSDraftExecutor(worker.draft_model_runner, self.tp_group)
        self.trace = trace
        self.round_id = 0
        self.last_round_metrics = None
        target_only = os.getenv("SPECSTREAM_NATIVE_TARGET_FIRST_LOOKAHEAD", "0")
        if target_only not in ("0", "1"):
            raise ValueError("SPECSTREAM_NATIVE_TARGET_FIRST_LOOKAHEAD must be 0 or 1")
        self.target_only_enabled = target_only == "1"
        self.target_only_mapping = False
        self.ready = None
        self.trim = None
        self.cached_candidates_consumed = 0
        self.seed_candidates_consumed = 0
        self.partial_candidates_consumed = 0
        self.seeded_requests = set()
        profile_path = os.getenv("SPECSTREAM_NATIVE_LOOKAHEAD_ADMISSION")
        self.partial_enabled = enabled and bool(profile_path)
        if self.partial_enabled and worker.hot_token_id is not None:
            raise ValueError("Partial native lookahead requires the full Draft vocabulary")
        self.admission = LookaheadAdmission(profile_path) if self.partial_enabled and self.tp_group.rank_in_group == 0 else None
        if self.admission is not None and worker._native_segment_trace is not None:
            worker._native_segment_trace.lookahead_admission = self.admission
        self.partial = {}
        self.partial_spill_map = None
        diagnostic = os.getenv("SPECSTREAM_NATIVE_ROUND_DIAGNOSTIC_MAX_STAGES")
        self.diagnostic_max_stages = None
        if diagnostic is not None:
            self.diagnostic_max_stages = int(diagnostic)
            if not enabled or not 1 <= self.diagnostic_max_stages <= 8:
                raise ValueError("Diagnostic lookahead stage limit must be 1..8 with parallel enabled")

    def use_ready(self, batch):
        self.cached_candidates_consumed = 0
        self.partial_candidates_consumed = 0
        request_state = _request_state(batch.reqs)
        active = set(request_state)
        self.seed_candidates_consumed = len(active & self.seeded_requests)
        self.seeded_requests.difference_update(active)
        ready, self.ready = self.ready, None
        worker = self.worker
        steps = worker.speculative_num_steps
        if ready is not None and request_state == ready.request_state and ready.tokens.shape[1] >= steps:
            self.clear_partial()
            for req in batch.reqs:
                req.decode_batch_idx += 1
            spec = batch.spec_info
            if batch.sampling_info.penalizer_orchestrator.is_required:
                batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                    spec.verified_id.to(torch.int64)
                )
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
            self.cached_candidates_consumed = batch.batch_size() * steps
            return self.build_verify_input(batch, ready.tokens[:, :steps])
        return self.continue_partial(batch, request_state)

    def build_verify_input(self, batch, candidate_tokens):
        worker = self.worker
        steps = worker.speculative_num_steps
        seq_lens_sum = batch.seq_lens_sum
        indices = torch.arange(steps, device=self.device, dtype=torch.int64)
        parents = indices.sub(1).expand(batch.batch_size(), -1).contiguous()
        if steps == 1:
            parents = parents[:, :0].contiguous()
        tree, positions, retrieve, child, sibling, tokens = build_tree_kernel_efficient(
            batch.spec_info.verified_id, parents, indices.expand(batch.batch_size(), -1).contiguous(),
            candidate_tokens.contiguous(), batch.seq_lens, seq_lens_sum, 1,
            steps, steps + 1,
        )
        worker._native_proposal_graph_replayed = False
        return EagleVerifyInput(
            draft_token=tokens, custom_mask=tree, positions=positions,
            retrive_index=retrieve, retrive_next_token=child, retrive_next_sibling=sibling,
            retrive_cum_len=None, spec_steps=steps, topk=1, draft_token_num=steps + 1,
            capture_hidden_mode=CaptureHiddenMode.FULL, seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def clear_partial(self):
        if self.partial_spill_map is not None and self.partial_spill_map is self.spill.gpu_map:
            for proposal in self.partial.values():
                self.spill.gpu_free.extend(proposal.physical.tolist())
        self.partial.clear()
        self.partial_spill_map = None

    def continue_partial(self, batch, request_state):
        if not self.partial:
            return None
        spill = self.spill
        worker = self.worker
        steps = worker.speculative_num_steps
        if steps < 2 or not spill.active or self.partial_spill_map is not spill.gpu_map:
            self.clear_partial()
            return None
        retained = {i: self.partial[key] for i, key in enumerate(request_state) if key in self.partial}
        if not retained:
            self.clear_partial()
            return None
        # Allocate the normal tentative logical suffix. A detached GPU page
        # has no logical ID until adopted into these slots, so it cannot be
        # read by Target or evicted to Draft's committed CPU history.
        try:
            worker._draft_preprocess_decode(batch)
            logical = batch.out_cache_loc.cpu().numpy().reshape(batch.batch_size(), steps)
            if np.any(spill.gpu_map[logical]) or np.any(spill.cpu_map[logical]):
                raise RuntimeError("Partial Draft proposal has an occupied destination")
            spill.tentative = logical.reshape(-1)
            tokens = batch.spec_info.topk_index.reshape(-1, 1).expand(-1, steps).clone()
            ready_lengths = np.ones(batch.batch_size(), dtype=np.int64)
            for row, proposal in retained.items():
                count = min(steps, proposal.tokens.numel())
                tokens[row, :count] = proposal.tokens[:count]
                ready_lengths[row] = count
                # The last emitted candidate doesn't need its KV materialized
                # until we run it to obtain its successor.
                spill.gpu_map[logical[row, :count - 1]] = proposal.physical[:count - 1]
                proposal.physical = proposal.physical[count - 1:]
                self.partial_candidates_consumed += count - 1
            self.clear_partial()
            return self.finish_partial(batch, logical, tokens, ready_lengths)
        finally:
            self.clear_partial()
            spill.end_tentative()

    def finish_partial(self, batch, logical, tokens, ready_lengths):
        spill = self.spill
        device = self.device
        runner = self.worker.draft_model_runner
        backend = runner.attn_backend
        lengths = batch.seq_lens_cpu.numpy()
        table = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, :int(lengths.max())
        ].cpu().numpy()
        history = [row[:n] for row, n in zip(table, lengths)]
        protected = np.concatenate([row[-128:] for row in history])
        spill._allocate_writes(logical.reshape(-1), protected)
        gpu_prefix, cpu_prefix = [], []
        for row in history:
            gpu, cpu = spill.gpu_map[row], spill.cpu_map[row]
            if np.any(row <= 0) or np.any((gpu > 0) == (cpu > 0)):
                raise RuntimeError("Partial Draft proposal found invalid committed history")
            gpu_prefix.append(gpu[gpu > 0])
            cpu_prefix.append(cpu[cpu > 0])
        physical = spill.gpu_map[logical]
        prefix_gpu_sizes = np.array([len(row) for row in gpu_prefix], dtype=np.int32)
        page_table = np.zeros((len(history), int(prefix_gpu_sizes.max()) + logical.shape[1]), dtype=np.int32)
        for row, pages, sizes, suffix in zip(page_table, gpu_prefix, prefix_gpu_sizes, physical):
            row[:sizes] = pages
            row[sizes:sizes + len(suffix)] = suffix
        table = torch.as_tensor(page_table, device=device)
        destinations = torch.as_tensor(physical, device=device)
        state = backend.forward_metadata
        try:
            for step in range(tokens.shape[1] - 1):
                rows = np.flatnonzero(ready_lengths <= step + 1)
                if not len(rows):
                    continue
                picks = torch.as_tensor(rows, device=device)
                seq_lens_cpu = torch.as_tensor(lengths[rows] + step + 1, dtype=torch.int32)
                seq_lens = seq_lens_cpu.to(device)
                writes = destinations[picks, step]
                cpu_indices = [i for i, r in enumerate(rows) if len(cpu_prefix[r])]
                plan = DraftAttentionPlan(
                    write_slots=writes, page_table=table.index_select(0, picks),
                    lengths=torch.as_tensor(prefix_gpu_sizes[rows] + step + 1, device=device),
                    cpu_rows=[cpu_prefix[rows[i]] for i in cpu_indices],
                    cpu_queries=[np.array([i], dtype=np.int64) for i in cpu_indices],
                )
                if cpu_indices:
                    plan.cpu_groups = list(spill._cpu_groups(plan, device))
                offsets = torch.arange(len(rows) + 1, dtype=torch.int32, device=device)
                metadata = FlashAttentionMetadata(
                    max_seq_len_q=1, max_seq_len_k=int(seq_lens_cpu.max()), cu_seqlens_q=offsets,
                )
                metadata.native_draft_spill_plan = plan
                backend.forward_metadata = metadata
                frame = ForwardBatch(
                    forward_mode=ForwardMode.DRAFT_EXTEND, batch_size=len(rows),
                    input_ids=tokens[picks, step], positions=seq_lens.to(torch.int64),
                    req_pool_indices=None, seq_lens=seq_lens, out_cache_loc=writes,
                    seq_lens_sum=int(seq_lens_cpu.sum()), seq_lens_cpu=seq_lens_cpu,
                    extend_num_tokens=len(rows), extend_seq_lens=torch.ones_like(seq_lens),
                    extend_start_loc=offsets[:-1], extend_prefix_lens=seq_lens - 1,
                    extend_prefix_lens_cpu=(lengths[rows] + step).tolist(),
                    extend_seq_lens_cpu=[1] * len(rows), extend_logprob_start_lens_cpu=[0] * len(rows),
                    token_to_kv_pool=spill.pool, attn_backend=backend,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                    spec_algorithm=self.worker.speculative_algorithm,
                )
                logits = runner.forward(frame, skip_attn_backend_init=True).logits_output
                tokens[picks, step + 1] = fast_topk(
                    torch.softmax(logits.next_token_logits, dim=-1), 1, dim=-1,
                )[1].reshape(-1)
        finally:
            backend.forward_metadata = state
        return self.build_verify_input(batch, tokens)

    def prepare(self, batch, verify_info):
        self.clear_partial()
        self.round_id += 1
        self.last_round_metrics = None
        self.trim = None
        self.ready = None
        state = _Round(self, batch, verify_info)
        state.prepare(batch, verify_info)
        return state

    def extend(self, batch, baseline_extend):
        trim, self.trim = self.trim, None
        if trim is None:
            return baseline_extend(batch)
        spec = batch.spec_info
        if trim.primed:
            self.merge_primed(batch, baseline_extend, trim)
            return None
        # Keep verified lengths and public slots for the scheduler. Only the
        # packed Draft query input omits the already materialized prefix.
        old_slots, old_ids, old_lengths = batch.out_cache_loc, spec.verified_id, spec.accept_length
        old_lengths_cpu, old_hidden = spec.accept_length_cpu, spec.hidden_states
        batch.out_cache_loc = old_slots.index_select(0, trim.tail)
        spec.verified_id = old_ids.index_select(0, trim.tail)
        spec.accept_length_cpu = trim.keep
        spec.accept_length = torch.tensor(trim.keep, dtype=old_lengths.dtype, device=self.device)
        if old_hidden is not None:
            spec.hidden_states = old_hidden.index_select(0, trim.tail)
        try:
            return baseline_extend(batch)
        finally:
            batch.out_cache_loc = old_slots
            spec.accept_length = old_lengths
            spec.accept_length_cpu = old_lengths_cpu

    def merge_primed(self, batch, baseline_extend, trim):
        """Replay only rows whose real bonus/correction was not precomputed."""
        spec = batch.spec_info
        live = len(trim.live_reqs)
        pending = [i for i in range(live) if i not in trim.primed]
        groups = []
        for index, values in sorted(trim.primed.items()):
            groups.append(([index], values))
        if pending:
            pending_index = torch.tensor(pending, dtype=torch.int64, device=self.device)
            pending_cpu = torch.tensor(pending, dtype=torch.int64)
            # ScheduleBatch.copy() omits fields used by draft extend. Shallow
            # copy, then replace every field mutated by the native forward.
            shadow = copy(batch)
            local = copy(spec)
            shadow.spec_info = local
            shadow.reqs = [trim.live_reqs[i] for i in pending]
            shadow.out_cache_loc = batch.out_cache_loc.index_select(0, trim.tail)
            shadow.prefix_lens = [
                int(spec.seq_lens_for_draft_extend_cpu[i]) - trim.keep[j] - 1
                for j, i in enumerate(pending)
            ]
            shadow.extend_logprob_start_lens = [0] * len(pending)
            local.verified_id = spec.verified_id.index_select(0, trim.tail)
            local.accept_length_cpu = trim.keep
            local.accept_length = torch.tensor(
                trim.keep, dtype=spec.accept_length.dtype, device=self.device,
            )
            if spec.hidden_states is not None:
                local.hidden_states = spec.hidden_states.index_select(0, trim.tail)
            local.seq_lens_for_draft_extend = spec.seq_lens_for_draft_extend.index_select(0, pending_index)
            local.seq_lens_for_draft_extend_cpu = spec.seq_lens_for_draft_extend_cpu.index_select(0, pending_cpu)
            local.req_pool_indices_for_draft_extend = spec.req_pool_indices_for_draft_extend.index_select(0, pending_index)
            baseline_extend(shadow)
            groups.append((pending, (local.topk_p, local.topk_index, local.hidden_states)))

        spec.verified_id = spec.verified_id.index_select(0, trim.final)
        merged = []
        for field in range(3):
            values = [q[field] for _, q in groups]
            if all(t is None for t in values):
                merged.append(None)
                continue
            if any(t is None for t in values):
                raise RuntimeError("Mixed Draft rows disagree on hidden state capture")
            rows = torch.tensor([idx for indices, _ in groups for idx in indices],
                                dtype=torch.int64, device=self.device)
            source = torch.cat(values)
            if len(source) != live:
                raise RuntimeError("Draft seed output count differs from unfinished rows")
            merged.append(torch.empty_like(source).index_copy_(0, rows, source))
        spec.topk_p, spec.topk_index, spec.hidden_states = merged

    def write_trace(self, row):
        if self.trace:
            with open(self.trace, "a", encoding="utf8") as stream:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")


class _Round:
    def __init__(self, owner, batch, info):
        self.owner = owner
        self.worker = owner.worker
        self.width = info.draft_token_num
        self.batch_size = len(batch.reqs)
        self.rows = []
        self.cohort_size = 0
        self.output_starts = [len(req.output_ids) for req in batch.reqs]
        self.event_start = self.event_end = None
        self.process_target_start_ns = None
        self.process_target_end_ns = None
        self.target_forward_submitted = threading.Event()
        self.cancelled = threading.Event()
        self.future = None
        self.target_host_start_ns = self.target_host_submitted_ns = None
        self.draft_host_start_ns = self.draft_host_end_ns = None
        self.admission_decision = None
        self.stop_reason = None
        self.timestamp = time.time()
        self.extra = None
        self.verify_slots = None
        self.released_future_slots = 0
        self.work = []
        self.samples = []
        self.steps = []
        self.skip = "disabled"
        self.activated_gpu_draft_mapping = False
        self.gpu_only_scratch = False

    def prepare(self, batch, info):
        if not self.owner.enabled:
            return
        spill = self.owner.spill
        if not spill.active:
            self.owner.target_only_mapping = False
        elif spill.cpu_resident_tokens:
            self.owner.target_only_mapping = False
        target_pool = self.worker.target_worker.model_runner.token_to_kv_pool
        target_spill = getattr(target_pool, "native_target_spill", None)
        target_only = (
            not spill.active and self.owner.target_only_enabled
            and target_spill is not None
            and target_spill.batch_history_tokens(batch.reqs) > 0
        )
        if not target_only and (not spill.active or self.worker._native_saved_draft_graphs is None):
            self.skip = "gpu_only"
            return
        if batch.forward_mode.is_idle() or batch.has_grammar or not batch.reqs:
            self.skip = "unsupported_batch"
            return
        # Establishing a remap changes ordinary Draft Graph eligibility. Do
        # this only after the strict window admission has approved the work.
        if target_only and not self.owner.partial_enabled:
            self.skip = "target_only_requires_admission"
            return
        if self.owner.partial_enabled:
            decision = None
            if self.owner.admission is not None:
                pool = self.worker.target_worker.model_runner.token_to_kv_pool
                target_host = pool.native_target_spill.cpu_resident_tokens
                decision = self.owner.admission.decide(self.batch_size, self.width, target_host)
            self.admission_decision = self.owner.tp_group.broadcast_object(decision, src=0)
            self.skip = self.admission_decision["skip"]
            if self.skip is not None:
                return
        device = self.owner.device
        allocator = self.worker.token_to_kv_pool_allocator
        lengths = batch.seq_lens_cpu.tolist()
        context_limit = self.worker.draft_model_runner.model_config.context_len
        # Future writes use otherwise free logical IDs but are written to a
        # separate Draft pool. They are never published to the shared table.
        table = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, :max(lengths)
        ].cpu().numpy()
        history = [row[:n] for row, n in zip(table, lengths)]
        if any(np.any(row <= 0) for row in history):
            raise RuntimeError("Round lookahead has an incomplete committed Draft prefix")
        self.rows = [
            i for i, n in enumerate(lengths)
            if n + 2 * self.width - 2 < context_limit
        ]
        self.cohort_size = len(self.rows)
        if not self.rows:
            self.skip = "context_limit"
            return
        if target_only:
            # Existing live Draft pages keep their original physical IDs.
            # No committed KV is transferred to CPU by activate().
            spill.activate(allocator.free_pages)
            if spill.cpu_resident_tokens:
                raise RuntimeError("Draft mapping initialization unexpectedly moved KV to CPU")
            self.owner.target_only_mapping = True
            self.activated_gpu_draft_mapping = True
            self.worker._update_native_draft_graphs()
        self.gpu_only_scratch = self.owner.target_only_mapping and spill.cpu_resident_tokens == 0
        rows = torch.tensor(self.rows, dtype=torch.int64, device=device)
        current = batch.out_cache_loc.view(self.batch_size, self.width).index_select(0, rows).cpu().numpy()
        protected = np.concatenate([row[-128:] for row in history])
        unprotected_gpu = max(0, spill.gpu_slots - len(spill.gpu_free) - int(
            np.count_nonzero(spill.gpu_map[protected])
        ))
        available = len(spill.gpu_free) + min(len(spill.cpu_free), unprotected_gpu)
        if self.gpu_only_scratch:
            # The current Target provisional slots were already reserved when
            # the identity map was established. Future work can use free GPU
            # pages, but may not evict any committed Draft history.
            available = len(spill.gpu_free) + int(np.count_nonzero(spill.gpu_map[current]))
        stages = min(self.width - 2, len(allocator.free_pages) // self.cohort_size)
        # Speculation can spill only history which already belongs to a live request.
        # Reduce the scratch horizon if the fixed GPU+CPU budget is tight.
        stages = min(stages, max(0, available // self.cohort_size - self.width))
        count = self.cohort_size * (self.width + stages)
        if count > available:
            self.skip = "no_scratch_capacity"
            return
        # rank-independent sizes come from the same logical allocation and the
        # same deterministic spill policy on every TP rank.
        extra = allocator.free_pages[: stages * self.cohort_size].cpu().numpy()
        targets = np.concatenate((current, extra.reshape(self.cohort_size, stages)), axis=1)
        if self.gpu_only_scratch and np.count_nonzero(spill.gpu_map[targets] == 0) > len(spill.gpu_free):
            self.skip = "no_gpu_scratch_capacity"
            return
        self.verify_slots = current.reshape(-1)
        self.extra = extra
        self.current = info.draft_token.view(self.batch_size, self.width).index_select(0, rows)[:, 1:].contiguous()
        try:
            spill._allocate_writes(targets.reshape(-1), protected)
            chosen_history = [history[i] for i in self.rows]
            chosen_lengths = [lengths[i] for i in self.rows]
            self.make_batches(chosen_history, targets, chosen_lengths, device)
        except BaseException:
            self.abort()
            raise
        self.skip = None

    def make_batches(self, history, targets, lengths, device):
        spill = self.owner.spill
        runner = self.worker.draft_model_runner
        gpu_prefix = []
        cpu_prefix = []
        for row in history:
            gpu, cpu = spill.gpu_map[row], spill.cpu_map[row]
            if np.any((gpu > 0) == (cpu > 0)):
                raise RuntimeError("Draft history must have exactly one KV owner per token")
            gpu_prefix.append(gpu[gpu > 0])
            cpu_prefix.append(cpu[cpu > 0])
        cpu_rows = [row for row in cpu_prefix if len(row)]
        cpu_indices = [i for i, row in enumerate(cpu_prefix) if len(row)]
        destinations = spill.gpu_map[targets]
        prefix_gpu_sizes = np.array([len(x) for x in gpu_prefix], dtype=np.int32)
        table = np.zeros((self.cohort_size, int(prefix_gpu_sizes.max()) + targets.shape[1]), dtype=np.int32)
        for row, (prefix, loc) in enumerate(zip(gpu_prefix, destinations)):
            table[row, :len(prefix)] = prefix
            table[row, len(prefix):len(prefix)+len(loc)] = loc
        page_table = torch.as_tensor(table, device=device)
        destination_tensor = torch.as_tensor(destinations, device=device)
        starts = torch.as_tensor(lengths, dtype=torch.int64, device=device)
        placeholder = torch.empty(self.cohort_size, dtype=torch.int64, device=device)
        stages = [(0, self.width - 1), (self.width - 1, self.width)]
        stages.extend((i, i + 1) for i in range(self.width, targets.shape[1]))
        for left, right in stages:
            queries = right - left
            cpu_queries = [
                np.arange(i * queries, (i + 1) * queries, dtype=np.int64)
                for i in cpu_indices
            ]
            lengths_cpu = torch.tensor([n + right for n in lengths], dtype=torch.int32)
            sequence_lengths = starts.to(torch.int32) + right
            # The current candidate prefix and bonus use canonical extend
            # positions; future candidates keep the native proposal convention.
            positions = starts[:, None] + torch.arange(left, right, device=device)
            if left >= self.width:
                positions = positions + 1
            writes = destination_tensor[:, left:right].reshape(-1).contiguous()
            frame = ForwardBatch(
                forward_mode=ForwardMode.DRAFT_EXTEND, batch_size=self.cohort_size,
                input_ids=self.current.reshape(-1) if left == 0 else placeholder,
                positions=positions.reshape(-1), req_pool_indices=None,
                seq_lens=sequence_lengths, out_cache_loc=writes,
                seq_lens_sum=sum(lengths) + self.cohort_size * right,
                seq_lens_cpu=lengths_cpu, extend_num_tokens=self.cohort_size * queries,
                extend_seq_lens=torch.full_like(sequence_lengths, queries),
                extend_start_loc=torch.arange(self.cohort_size, dtype=torch.int32, device=device) * queries,
                extend_prefix_lens=starts.to(torch.int32) + left,
                extend_prefix_lens_cpu=[n + left for n in lengths],
                extend_seq_lens_cpu=[queries] * self.cohort_size,
                extend_logprob_start_lens_cpu=[0] * self.cohort_size,
                token_to_kv_pool=spill.pool, attn_backend=runner.attn_backend,
                capture_hidden_mode=CaptureHiddenMode.LAST,
                spec_algorithm=self.worker.speculative_algorithm,
            )
            plan = DraftAttentionPlan(
                write_slots=writes, page_table=page_table,
                lengths=torch.as_tensor(prefix_gpu_sizes + right, device=device),
                cpu_rows=cpu_rows, cpu_queries=cpu_queries,
            )
            if cpu_queries and len(cpu_queries) * queries <= 128:
                plan.cpu_groups = list(spill._cpu_groups(plan, page_table.device))
            metadata = FlashAttentionMetadata(
                max_seq_len_q=queries, max_seq_len_k=int(lengths_cpu.max().item()),
                cu_seqlens_q=torch.arange(0, (self.cohort_size + 1) * queries, queries,
                                          device=device, dtype=torch.int32),
            )
            metadata.native_draft_spill_plan = plan
            self.work.append((frame, metadata))

    def start(self):
        if self.skip is None:
            self.event_start = torch.cuda.Event(enable_timing=True)
            self.event_end = torch.cuda.Event(enable_timing=True)
            self.event_start.record()
            self.target_host_start_ns = time.monotonic_ns()
            self.future = self.owner.executor.submit(self.run)

    def finish_target(self):
        if self.future is None:
            return
        self.event_end.record()
        self.target_host_submitted_ns = time.monotonic_ns()
        self.target_forward_submitted.set()
        observer = None
        if self.owner.process_executor is not None:
            def observe_target_completion():
                with torch.cuda.device(self.owner.device):
                    self.event_end.synchronize()
                self.process_target_end_ns = time.monotonic_ns()

            observer = threading.Thread(target=observe_target_completion, daemon=True)
            observer.start()
        try:
            self.join()
        finally:
            if observer is not None:
                observer.join()

    def join(self):
        if self.future is None:
            return
        future, self.future = self.future, None
        try:
            future.result()
        finally:
            # Only the scheduler thread changes logical-to-physical slot ownership.
            if not self.owner.partial_enabled:
                self.release_future()

    def run(self):
        self.draft_host_start_ns = time.monotonic_ns()
        stream = self.owner.stream
        runner = self.worker.draft_model_runner
        backend = runner.attn_backend
        state = backend.forward_metadata
        group = self.owner.tp_group
        try:
            with (
                # Split attention keeps mutable pinned buffers across rounds
                # and also uses them from the scheduler's no-grad context.
                torch.no_grad(),
                torch.cuda.device(stream.device),
                torch.cuda.stream(stream),
                use_lookahead_tp_group(group),
            ):
                process = self.owner.process_executor
                if process is None:
                    stream.wait_event(self.event_start)
                else:
                    # Protect IPC reads of the parent-created descriptor and
                    # provisional pages with the same Target-start dependency.
                    self.event_start.synchronize()
                    self.process_target_start_ns = time.monotonic_ns()
                for i, (frame, metadata) in enumerate(self.work):
                    stop = None
                    if group.rank_in_group == 0:
                        limit = self.owner.diagnostic_max_stages
                        if self.cancelled.is_set():
                            stop = "cancelled"
                        elif self.target_forward_submitted.is_set() and self.event_end.query():
                            stop = "target_finished"
                        elif limit is not None and i >= limit:
                            stop = "diagnostic_limit"
                        elif i > 0 and self.admission_decision is not None and not self.owner.admission.next_fits(
                            self.admission_decision, self.target_host_start_ns
                        ):
                            stop = "next_step_would_overrun"
                    self.stop_reason = group.broadcast_object(stop, src=0)
                    if self.stop_reason is not None:
                        break
                    backend.forward_metadata = metadata
                    if i:
                        frame.input_ids = self.samples[-1][1].reshape(-1)
                    if process is not None:
                        begin_ns, end_ns, probabilities, tokens, hidden = process.forward(
                            frame, metadata, reuse_previous=bool(i)
                        )
                        self.steps.append((begin_ns, end_ns, 0))
                        self.samples.append((probabilities, tokens, hidden))
                        continue
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    cpu_start = self.owner.spill.cpu_attention_ns
                    begin.record()
                    logits = runner.forward(frame, skip_attn_backend_init=True).logits_output
                    probabilities, tokens = fast_topk(torch.softmax(logits.next_token_logits, dim=-1), 1, dim=-1)
                    end.record()
                    self.steps.append((begin, end, self.owner.spill.cpu_attention_ns - cpu_start))
                    self.samples.append((probabilities, tokens, logits.hidden_states))
                    # Recheck the Target deadline after each complete stage.
                    # A single stage may run beyond it and is timed separately.
                    end.synchronize()
        finally:
            backend.forward_metadata = state
            # An exception inside a stage can leave queued reads or writes.
            stream.synchronize()
            self.draft_host_end_ns = time.monotonic_ns()
        if not self.steps:
            self.skip = "target_already_finished"

    def release_future(self):
        if self.extra is not None:
            if len(self.extra):
                self.released_future_slots = int(np.count_nonzero(self.owner.spill.gpu_map[self.extra]))
                self.owner.spill._free(self.extra)
            self.extra = None

    def abort(self):
        self.cancelled.set()
        try:
            self.join()
        finally:
            self.release_future()
            if self.verify_slots is not None:
                self.owner.spill._free(self.verify_slots)
                self.verify_slots = None

    def keep_partial(self, req, selected, candidates):
        if not self.owner.partial_enabled or len(candidates) <= 1:
            return
        spill = self.owner.spill
        count = len(candidates) - 1
        logical = self.extra.reshape(self.cohort_size, -1)[selected, :count]
        held = spill.gpu_map[logical].copy()
        if len(held) != count or np.any(held <= 0) or np.any(spill.cpu_map[logical]):
            raise RuntimeError("Partial Draft KV must remain in tentative GPU slots")
        tokens = torch.cat([s[1][selected].reshape(-1) for s in self.samples[1:]])
        self.owner.partial[(req.rid, len(req.output_ids))] = _PartialCandidates(tokens, held)
        self.owner.partial_spill_map = spill.gpu_map
        spill.gpu_map[logical] = 0

    def reconcile(self, batch, result):
        try:
            trace = self.reconcile_branches(batch, result)
        finally:
            self.release_future()
        trace["released_future_slots"] = self.released_future_slots
        self.owner.last_round_metrics = dict(
            target_overlap_ms=trace.get("target_overlap_ms", 0.0),
            target_window_completed=trace["target_window_completed"],
            cached_candidates_consumed=trace["cached_candidates_consumed"],
            partial_candidates_consumed=trace["partial_candidates_consumed"],
            seed_candidates_consumed=trace["seed_candidates_consumed"],
        )
        self.owner.write_trace(trace)

    def reconcile_branches(self, batch, result):
        trace = dict(
            timestamp=self.timestamp, round_id=self.owner.round_id,
            batch_size=self.batch_size, q=self.width, enabled=self.owner.enabled,
            skip_reason=self.skip, target_window_completed=0, branches=[],
            cached_candidates_consumed=self.owner.cached_candidates_consumed,
            seed_candidates_consumed=self.owner.seed_candidates_consumed,
            partial_candidates_consumed=self.owner.partial_candidates_consumed,
            admission=self.admission_decision,
            stop_reason=self.stop_reason,
            released_future_slots=self.released_future_slots,
            selected_batch_size=self.cohort_size,
            activated_gpu_draft_mapping=self.activated_gpu_draft_mapping,
            gpu_only_scratch=self.gpu_only_scratch,
            draft_cpu_resident_tokens=self.owner.spill.cpu_resident_tokens,
        )
        if self.event_start is None:
            return trace
        target_end = round(self.event_start.elapsed_time(self.event_end) * 1e6)
        trace.update(target_start_ns=0, target_end_ns=target_end,
                     target_resolve_monotonic_ns=time.monotonic_ns(),
                     target_host_submit_ms=(self.target_host_submitted_ns - self.target_host_start_ns) / 1e6,
                     draft_host_start_offset_ms=(self.draft_host_start_ns - self.target_host_start_ns) / 1e6,
                     draft_host_execution_ms=(self.draft_host_end_ns - self.draft_host_start_ns) / 1e6)
        if not self.steps:
            return trace
        specs = result.draft_input
        device = self.owner.device
        current = self.current.tolist()
        bonus, *next_candidates = [tokens.reshape(-1).tolist() for _, tokens, _ in self.samples]
        phases = []
        is_process = self.owner.process_executor is not None
        anchor = None
        if is_process:
            # Same-host monotonic clock with each GPU range reconstructed
            # from its own CUDA events and observed device completion.
            start_anchor = self.process_target_start_ns
            end_anchor = self.process_target_end_ns - target_end
            anchor = min(start_anchor, end_anchor)
            trace["overlap_timing_source"] = "cuda_events_anchored_to_host_completion"
            trace["overlap_anchor_difference_ms"] = abs(start_anchor - end_anchor) / 1e6
            trace["draft_process_pid"] = self.owner.process_executor.pid
        for i, (begin, end, cpu_ns) in enumerate(self.steps):
            phases.append(dict(
                kind="bonus" if i == 0 else "candidate", candidate_count=int(i != 0),
                start_ns=begin - anchor if is_process else round(self.event_start.elapsed_time(begin) * 1e6),
                end_ns=end - anchor if is_process else round(self.event_start.elapsed_time(end) * 1e6),
                draft_cpu_attention_ms=cpu_ns / 1e6,
            ))
        trace["target_overlap_ms"] = sum(
            max(0, min(phase["end_ns"], target_end) - max(0, phase["start_ns"]))
            for phase in phases
        ) / 1e6
        if self.owner.admission is not None:
            self.owner.admission.observe_steps(self.admission_decision, phases)
        ready_count = sum(s["candidate_count"] for s in phases if s["end_ns"] <= target_end)
        trace["target_window_completed"] = ready_count * self.cohort_size
        keep, tail, final, live = [], [], [], []
        primed = {}
        selected_index = {original: selected for selected, original in enumerate(self.rows)}
        ready_rows = []
        offset = 0
        for i, req in enumerate(batch.reqs):
            verified = req.output_ids[self.output_starts[i]:]
            finished = req.finished()
            selected = selected_index.get(i)
            committed = 0
            matched = False
            reuse_bonus = False
            if selected is not None:
                committed, matched = resolve_native_suffix(
                    current[selected], bonus[selected], verified, finished=finished,
                )
                reuse_bonus = bool(next_candidates and matched and not finished)
            if not finished:
                length = len(verified)
                if reuse_bonus:
                    primed[len(live)] = tuple(t[selected:selected + 1] if t is not None else None
                                              for t in self.samples[1])
                    self.keep_partial(req, selected, next_candidates)
                    ready_rows.append(selected)
                else:
                    keep.append(length - committed - 1)
                    tail.extend(range(offset + committed, offset + length))
                final.append(offset + length - 1)
                offset += length
                live.append(req)
            if selected is not None:
                trace["branches"].append(dict(
                    row=i, current=current[selected], verified=verified, bonus=bonus[selected],
                    next_candidates=[step[selected] for step in next_candidates],
                    retained=(len(next_candidates) if reuse_bonus and self.owner.partial_enabled else int(reuse_bonus)),
                    committed=committed, finished=finished,
                    forked=not matched, steps=phases,
                    discarded_slots=(len(current[selected]) - committed + len(next_candidates)
                                     - (len(next_candidates) if reuse_bonus and self.owner.partial_enabled else int(reuse_bonus))),
                    reused_bonus_slot=reuse_bonus,
                ))
        if offset != specs.verified_id.numel():
            raise RuntimeError("Lookahead suffix offsets do not match Draft input")
        if offset:
            if len(primed) == len(live):
                positions = torch.tensor(ready_rows, dtype=torch.int64, device=device)
                next_ids = torch.cat([s[1] for s in self.samples[1:]], dim=1).index_select(0, positions)
                self.owner.ready = _ReadyCandidates(_request_state(live), next_ids)
                for branch in trace["branches"]:
                    if branch["reused_bonus_slot"]:
                        branch["retained"] = len(next_candidates)
            if primed or len(tail) < offset:
                self.owner.trim = _Trim(
                    keep, torch.as_tensor(tail, dtype=torch.int64, device=device),
                    torch.as_tensor(final, dtype=torch.int64, device=device), live, primed,
                )
            self.owner.seeded_requests = set(_request_state(live[i] for i in primed))
        return trace
