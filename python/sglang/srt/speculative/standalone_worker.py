import json
import logging
import os
import time
from typing import Optional

import torch

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context, load_token_map
from sglang.srt.utils import empty_context, get_bool_env_var, is_cuda

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")


class StandaloneWorker(EAGLEWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        if server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init draft worker
        with empty_context(), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            TpModelWorker.__init__(
                self,
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )

        # Init attention backend and cuda graphs
        self._native_draft_spill = None
        self._native_pressure_trace = os.getenv("SPECSTREAM_PRESSURE_TRACE") if tp_rank == 0 else None
        self._native_segment_trace = None
        self._native_saved_draft_graphs = None
        if os.getenv("SPECSTREAM_NATIVE_DRAFT_GPU_SLOTS"):
            from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
            from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
            from sglang.srt.speculative.native_draft_spill import NativeDraftSpill

            pool = self.draft_model_runner.token_to_kv_pool
            allocator = self.token_to_kv_pool_allocator
            if (
                type(pool) is not MHATokenToKVPool
                or type(allocator) is not TokenToKVPoolAllocator
                or self.page_size != 1 or allocator.need_sort
                or self.topk != 1 or self.speculative_num_steps < 2
                or not server_args.disable_radix_cache
                or not server_args.disable_overlap_schedule
                or server_args.disaggregation_mode != "null"
                or server_args.attention_backend != "fa3"
                or server_args.prefill_attention_backend not in (None, "fa3")
                or server_args.decode_attention_backend not in (None, "fa3")
                or pool.dtype != torch.bfloat16 or pool.head_dim != 128
            ):
                raise ValueError(
                    "Native Draft pressure requires BF16 FA3 STANDALONE, "
                    "topk=1, page_size=1, and ordinary scheduling without radix cache"
                )
            if pool.size < allocator.size:
                self._native_draft_spill = NativeDraftSpill(
                    pool, allocator.size, tp_rank,
                    tp_threads=int(os.getenv("SPECSTREAM_CPU_ATTN_THREADS", "16")),
                )
                pool.native_draft_spill = self._native_draft_spill
                allocator.native_draft_spill = self._native_draft_spill

        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)
        self._native_round_parallel = None
        parallel_setting = os.getenv("SPECSTREAM_NATIVE_ROUND_PARALLEL", "0")
        segment_path = os.getenv("SPECSTREAM_NATIVE_SEGMENT_TRACE")
        if segment_path:
            from sglang.srt.speculative.native_segment_timing import NativeSegmentTiming

            target_runner = self.target_worker.model_runner
            if (self._native_draft_spill is None
                    or not getattr(target_runner.token_to_kv_pool, "native_target_spill", None)
                    or os.getenv("SPECSTREAM_CPU_ATTN_TIMING") != "1"
                    or os.getenv("SPECSTREAM_NATIVE_DYNAMIC_Q") != "1"):
                raise ValueError("Segment timing requires dynamic-q Target/Draft spill with CPU timing")
            self._native_segment_trace = NativeSegmentTiming(
                segment_path, tp_rank, int(os.getenv("SPECSTREAM_NATIVE_SEGMENT_STRIDE", "3")),
            )
            target_runner._native_segment_trace = self._native_segment_trace
        parallel_trace = os.getenv("SPECSTREAM_NATIVE_ROUND_TRACE")
        if parallel_setting not in ("0", "1"):
            raise ValueError("SPECSTREAM_NATIVE_ROUND_PARALLEL must be 0 or 1")
        if parallel_setting == "1" or parallel_trace:
            from sglang.srt.speculative.native_round_parallel import NativeRoundParallel

            if (self._native_draft_spill is None or server_args.enable_dp_attention
                    or server_args.tp_size != 2 or server_args.pp_size != 1
                    or self.draft_model_runner.model_config.hf_config.architectures
                    != ["Qwen3ForCausalLM"]):
                raise ValueError("Native round lookahead requires colocated TP2 Qwen3 and Draft spill")
            if parallel_trace is not None:
                if "{rank}" in parallel_trace:
                    parallel_trace = parallel_trace.format(rank=tp_rank)
                else:
                    root, extension = os.path.splitext(parallel_trace)
                    parallel_trace = f"{root}.tp{tp_rank}{extension or '.jsonl'}"
            self._native_round_parallel = NativeRoundParallel(
                self, parallel_setting == "1", parallel_trace,
            )

    def _update_native_draft_graphs(self):
        spill = self._native_draft_spill
        if spill is None:
            return
        if spill.active and self._native_saved_draft_graphs is None:
            self._native_saved_draft_graphs = (
                self.cuda_graph_runner, self.cuda_graph_runner_for_draft_extend,
                self.draft_model_runner.piecewise_cuda_graph_runner,
            )
            self.cuda_graph_runner = None
            self.cuda_graph_runner_for_draft_extend = None
            self.draft_model_runner.piecewise_cuda_graph_runner = None
        elif not spill.active and self._native_saved_draft_graphs is not None:
            (
                self.cuda_graph_runner, self.cuda_graph_runner_for_draft_extend,
                self.draft_model_runner.piecewise_cuda_graph_runner,
            ) = self._native_saved_draft_graphs
            self._native_saved_draft_graphs = None

    def _draft_preprocess_decode(self, batch):
        super()._draft_preprocess_decode(batch)
        self._update_native_draft_graphs()

    def draft(self, batch):
        if self._native_round_parallel is not None and not batch.forward_mode.is_idle():
            result = self._native_round_parallel.use_ready(batch)
            if result is not None:
                return result
        try:
            return super().draft(batch)
        finally:
            if self._native_draft_spill is not None:
                self._native_draft_spill.end_tentative()

    def forward_draft_extend(self, *args, **kwargs):
        if self._native_round_parallel is not None:
            self._native_round_parallel.clear_partial()
        self._update_native_draft_graphs()
        return super().forward_draft_extend(*args, **kwargs)

    def forward_draft_extend_after_decode(self, batch):
        self._update_native_draft_graphs()
        if self._native_round_parallel is not None:
            return self._native_round_parallel.extend(
                batch, super().forward_draft_extend_after_decode,
            )
        return super().forward_draft_extend_after_decode(batch)

    def forward_batch_generation(self, batch):
        trace = self._native_pressure_trace and batch.forward_mode.is_decode()
        batch_size = batch.batch_size() if trace else 0
        spill = self._native_draft_spill
        cpu_start_ns = spill.cpu_attention_ns if trace and spill is not None else 0
        target_pool = self.target_worker.model_runner.token_to_kv_pool
        target_spill = getattr(target_pool, "native_target_spill", None)
        if target_spill is not None:
            target_spill.register_requests(batch)
        if self._native_segment_trace is not None:
            self._native_segment_trace.flush()
        result = super().forward_batch_generation(batch)
        if trace:
            can_run_target_graph = result.can_run_cuda_graph
            row = {
                "time_ns": time.time_ns(),
                "batch_size": batch_size,
                "q": getattr(self, "_native_realized_q", self.speculative_num_draft_tokens),
                "draft_spill_tokens": spill.cpu_resident_tokens if spill else 0,
                "draft_spilled_records": spill.spilled_records if spill else 0,
                "target_spill_tokens": target_spill.cpu_resident_tokens if target_spill else 0,
                "target_spilled_records": target_spill.spilled_records if target_spill else 0,
                "target_cpu_attention_calls": target_spill.cpu_attention_calls if target_spill else 0,
                "target_h2d_bytes": target_spill.h2d_bytes if target_spill else 0,
                "target_h2d_cross_layer_bytes": target_spill.h2d_cross_layer_bytes if target_spill else 0,
                "target_h2d_chunks": target_spill.h2d_chunks if target_spill else 0,
                "target_h2d_copy_batches": target_spill.h2d_copy_batches if target_spill else 0,
                "target_gpu_online_merge_calls": target_spill.gpu_online_merge_calls if target_spill else 0,
                "target_gpu_cohort_calls": target_spill.gpu_cohort_calls if target_spill else 0,
                "cpu_attention_calls": spill.cpu_attention_calls if spill else 0,
                "cpu_attention_round_ns": (spill.cpu_attention_ns - cpu_start_ns) if spill else 0,
                "last_cpu_history_lengths_and_queries": spill.last_cpu_work if spill and spill.active else None,
                "prefill_own_evicted": getattr(spill, "prefill_own_evicted", 0),
                "prefill_host_queries": getattr(spill, "prefill_host_queries", 0),
                "prefill_host_query_keys": getattr(spill, "prefill_host_query_keys", 0),
                "draft_gpu_slots": self.draft_model_runner.token_to_kv_pool.size,
                "target_gpu_slots": target_pool.size,
                "target_logical_slots": self.token_to_kv_pool_allocator.size,
                "draft_graph_bypassed": self._native_saved_draft_graphs is not None,
                "target_graph_replayed": bool(can_run_target_graph),
            }
            with open(self._native_pressure_trace, "a", encoding="utf-8") as output:
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
        return result
