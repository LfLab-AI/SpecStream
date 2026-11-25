"""Target-first physical KV repartition at native worker round boundaries."""

import json
import math
import os
from pathlib import Path
import time

import torch

from sglang.srt.speculative.native_target_kv_loader import (
    native_target_stage_bytes, native_target_stage_tokens,
)


MOVE_SCRATCH_BYTES = 8 * 2**20


def create_joint_arena(runner):
    args = runner.server_args
    if (runner.device != "cuda" or runner.page_size != 1 or not runner.spec_algorithm.is_standalone()
            or runner.model_config.hf_config.model_type != "qwen3" or args.tp_size != 2
            or runner.kv_cache_dtype != torch.bfloat16 or args.pp_size != 1 or args.enable_dp_attention
            or not args.disable_radix_cache or not args.disable_overlap_schedule
            or args.attention_backend != "fa3" or args.disaggregation_mode != "null"
            or args.enable_memory_saver or not os.getenv("SPECSTREAM_NATIVE_DRAFT_GPU_SLOTS")):
        raise ValueError("Joint KV arena requires ordinary TP2 BF16 Qwen3 STANDALONE with FA3")
    if runner.is_draft_worker:
        arena = getattr(runner.token_to_kv_pool_allocator, "native_joint_kv_arena", None)
        if arena is None:
            raise ValueError("Draft cannot find the shared Target KV arena")
        return arena
    if not hasattr(runner, "_native_target_gpu_slots"):
        raise ValueError("Joint KV arena requires a Target overflow pool")
    gib = float(os.environ["SPECSTREAM_NATIVE_JOINT_KV_GIB"])
    if not math.isfinite(gib) or gib <= 0:
        raise ValueError("Joint KV budget must be positive and finite")
    from sglang.srt.speculative.native_joint_kv_arena import NativeJointKVArena
    from sglang.srt.layers.dp_attention import get_attention_tp_size

    total_budget = int(gib * 2**30)
    staging = native_target_stage_bytes(
        native_target_stage_tokens(), runner.model_config.get_num_kv_heads(get_attention_tp_size()),
        runner.model_config.head_dim, runner.kv_cache_dtype,
    )
    if total_budget <= staging + MOVE_SCRATCH_BYTES:
        raise ValueError("Joint KV budget does not cover staging and move scratch")
    arena = NativeJointKVArena(runner.device, total_budget - staging,
                              scratch_bytes=MOVE_SCRATCH_BYTES)
    arena.total_budget_bytes = total_budget
    arena.target_staging_bytes = staging
    return arena


def _bytes_per_slot(pool):
    return pool.layer_num * pool.head_num * (pool.head_dim + pool.v_head_dim) * pool.store_dtype.itemsize


class NativeTargetFirstResize:
    def __init__(self, worker, arena):
        self.worker = worker
        self.arena = arena
        self.allocator = worker.token_to_kv_pool_allocator
        self.target = self.allocator.native_target_spill
        self.draft = self.allocator.native_draft_spill
        self.target_bytes = _bytes_per_slot(self.target.pool)
        self.draft_bytes = _bytes_per_slot(self.draft.pool)
        self.grow_tokens = int(os.getenv("SPECSTREAM_NATIVE_JOINT_KV_GROW_TOKENS", "8192"))
        if self.grow_tokens < 1:
            raise ValueError("Draft KV growth quantum must be positive")
        requests = worker.server_args.max_running_requests
        prefill = worker.server_args.chunked_prefill_size
        if requests is None or prefill is None or prefill < 1:
            raise ValueError("Joint KV resizing requires bounded requests and chunked prefill")
        if self.target.min_gpu_tail < 512:
            raise ValueError("Joint KV resizing requires a 512-token Target tail per request")
        self.target_floor = max(512, requests * (self.target.min_gpu_tail + 2 * worker.speculative_num_draft_tokens) + prefill)
        if min(self.target.gpu_slots, self.draft.gpu_slots) < 1 or self.target.gpu_slots < self.target_floor:
            raise ValueError("Initial Target KV pool does not fit its required residency floor")
        data_bytes = arena.budget_bytes - arena.scratch_bytes
        self.full_residency_slots = data_bytes // (self.target_bytes + self.draft_bytes) - 1
        self.max_draft_slots = min(self.allocator.size - 1,
            (data_bytes - (self.target_floor + 1) * self.target_bytes) // self.draft_bytes - 1)
        cap = os.getenv("SPECSTREAM_NATIVE_DRAFT_GPU_CAP_BYTES")
        if cap is not None:
            # The physical pool includes reserved slot 0.
            cap_slots = int(cap) // self.draft_bytes - 1
            if cap_slots < self.draft.gpu_slots:
                raise ValueError("Draft GPU KV cap is below the initial pool size")
            self.max_draft_slots = min(self.max_draft_slots, cap_slots)
        self.path = None
        path = os.getenv("SPECSTREAM_NATIVE_JOINT_KV_TRACE")
        if path:
            path = Path(path)
            self.path = path.with_name(f"{path.stem}.tp{worker.tp_rank}{path.suffix or '.jsonl'}")
        self.moved = False
        self._record("initial", dict(target_slots=self.target.gpu_slots, draft_slots=self.draft.gpu_slots,
                                     max_draft_slots=self.max_draft_slots))

    def _record(self, kind, data):
        if self.path is None:
            return
        with self.path.open("a", encoding="utf8") as trace:
            trace.write(json.dumps(dict(event=kind, timestamp=time.time(), tp_rank=self.worker.tp_rank,
                                        total_budget_bytes=self.arena.total_budget_bytes,
                                        arena_budget_bytes=self.arena.budget_bytes,
                                        target_floor_slots=self.target_floor,
                                        staging_bytes=self.arena.target_staging_bytes,
                                        **data)) + "\n")

    def maybe_resize(self, batch):
        if not batch.reqs or batch.forward_mode.is_idle():
            return
        # For prefill, the scheduler already reserved the chunk in the logical
        # allocator. Decode still needs space for its next speculative writes.
        live = self.allocator.size - self.allocator.available_size()
        reserve = 2 * self.worker.speculative_num_draft_tokens * len(batch.reqs) if batch.forward_mode.is_decode() else 0
        requested = live + reserve
        old_draft = self.draft.gpu_slots
        if requested <= old_draft or old_draft >= self.max_draft_slots:
            return
        # Move chunks of capacity to avoid copying all live KV once per token.
        grow = ((requested - old_draft + self.grow_tokens - 1) // self.grow_tokens) * self.grow_tokens
        new_draft = min(self.max_draft_slots, old_draft + grow)
        # Preserve both complete histories while they still fit. Rounding up
        # Draft's growth must not consume space needed by current Target KV.
        if requested <= self.full_residency_slots:
            new_draft = min(new_draft, self.full_residency_slots)
        old_target = self.target.gpu_slots
        new_target = min(old_target, (self.arena.budget_bytes - self.arena.scratch_bytes
                                      - (new_draft + 1) * self.draft_bytes) // self.target_bytes - 1)
        assert self.target_floor <= new_target <= old_target and new_draft > old_draft
        group = self.worker.target_worker.model_runner.tp_group
        plan = (old_target, old_draft, new_target, new_draft)
        if group.broadcast_object(plan if self.worker.tp_rank == 0 else None, src=0) != plan:
            raise RuntimeError("TP ranks disagree on joint KV pool sizes")
        start = time.perf_counter()
        parallel = self.worker._native_round_parallel
        if parallel is not None:
            # Never move storage while an uncommitted lookahead owns detached pages.
            parallel.clear_partial()
            parallel.ready = None
            parallel.trim = None
            parallel.seeded_requests.clear()
        torch.cuda.synchronize()
        segment = self.worker._native_segment_trace
        if segment is not None:
            segment.flush()
        if not self.target.active:
            self.target.activate(self.allocator.free_pages)
        if not self.draft.active:
            self.draft.activate(self.allocator.free_pages)
        # Once the physical addresses change, replaying any previously captured
        # q/batch graph could read the other model's KV.
        if not self.moved:
            self._invalidate_graphs()
        protected = batch.out_cache_loc if batch.forward_mode.is_extend() else None
        protected = () if protected is None else protected.cpu().numpy()
        moves = self.target.shrink_gpu(new_target, protected_logical=protected,
                                       protected_tail=self.target.min_gpu_tail)
        torch.cuda.synchronize()
        layout = self.arena.resize(new_target, new_draft, target_preserved=new_target,
                                   draft_preserved=old_draft)
        self.draft.grow_gpu(new_draft)
        torch.cuda.synchronize()
        self.moved = True
        if parallel is not None and not self.draft.cpu_resident_tokens:
            parallel.target_only_mapping = True
        self._record("resize", dict(
            old_target_slots=old_target, new_target_slots=new_target,
            old_draft_slots=old_draft, new_draft_slots=new_draft,
            target_cpu_slots=self.target.cpu_resident_tokens,
            draft_cpu_slots=self.draft.cpu_resident_tokens,
            demand_tokens=requested, full_residency_slots=self.full_residency_slots,
            elapsed_ms=(time.perf_counter() - start) * 1e3,
            arena_storage_ptr=layout["arena_storage_ptr"],
            arena_layout=layout, **{k: v for k, v in moves.items() if k.endswith("_tokens")},
        ))

    def _invalidate_graphs(self):
        worker = self.worker
        for runner in (worker.target_worker.model_runner, worker.draft_model_runner):
            runner.graph_runner = None
            runner.piecewise_cuda_graph_runner = None
        worker.cuda_graph_runner = None
        worker.cuda_graph_runner_for_draft_extend = None
        worker._native_dynamic_graphs = None
        # The native eager spill path uses this marker to admit lookahead. It
        # also makes sure an idle reset cannot restore captured graphs.
        worker._native_saved_draft_graphs = (None, None, None)
