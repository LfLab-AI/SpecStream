# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A tensor parallel worker."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.managers.io_struct import (
    DestroyWeightsUpdateGroupReqInput,
    GetWeightsByNameReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    SendWeightsToRemoteInstanceReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spectre.specstream.config import (
    should_initialize_drafter_smctrl,
)
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj, set_random_seed
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import MemoryPoolConfig

logger = logging.getLogger(__name__)


class BaseTpWorker(ABC):
    @abstractmethod
    def forward_batch_generation(self, forward_batch: ForwardBatch):
        pass

    @property
    @abstractmethod
    def model_runner(self) -> "ModelRunner":
        pass

    @property
    def sliding_window_size(self) -> Optional[int]:
        return self.model_runner.sliding_window_size

    @property
    def is_hybrid_swa(self) -> bool:
        return self.model_runner.is_hybrid_swa

    def get_tokens_per_layer_info(self):
        return (
            self.model_runner.full_max_total_num_tokens,
            self.model_runner.swa_max_total_num_tokens,
        )

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.model_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.model_runner.init_weights_update_group(
            recv_req.master_address,
            recv_req.master_port,
            recv_req.rank_offset,
            recv_req.world_size,
            recv_req.group_name,
            recv_req.backend,
        )
        return success, message

    def destroy_weights_update_group(self, recv_req: DestroyWeightsUpdateGroupReqInput):
        success, message = self.model_runner.destroy_weights_update_group(
            recv_req.group_name,
        )
        return success, message

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        success, message = (
            self.model_runner.init_weights_send_group_for_remote_instance(
                recv_req.master_address,
                recv_req.ports,
                recv_req.group_rank,
                recv_req.world_size,
                recv_req.group_name,
                recv_req.backend,
            )
        )
        return success, message

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        success, message = self.model_runner.send_weights_to_remote_instance(
            recv_req.master_address,
            recv_req.ports,
            recv_req.group_name,
        )
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        success, message = self.model_runner.update_weights_from_distributed(
            recv_req.names,
            recv_req.dtypes,
            recv_req.shapes,
            recv_req.group_name,
            recv_req.load_format,
        )
        return success, message

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):

        monkey_patch_torch_reductions()
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=MultiprocessingSerializer.deserialize(
                recv_req.serialized_named_tensors[self.tp_rank]
            ),
            load_format=recv_req.load_format,
        )
        return success, message

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update weights from IPC for checkpoint-engine integration."""
        success, message = self.model_runner.update_weights_from_ipc(recv_req)
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.model_runner.get_weights_by_name(
            recv_req.name, recv_req.truncate_size
        )
        return parameter

    def load_lora_adapter(self, recv_req: LoadLoRAAdapterReqInput):
        result = self.model_runner.load_lora_adapter(recv_req.to_ref())
        return result

    def unload_lora_adapter(self, recv_req: UnloadLoRAAdapterReqInput):
        result = self.model_runner.unload_lora_adapter(recv_req.to_ref())
        return result

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ):
        # The LoRA code handles TP sharding internally using slice_lora_a_weights
        # and slice_lora_b_weights methods (see lora/layers.py:46-49, mem_pool.py:437-440).
        if recv_req.load_format == "flattened_bucket":
            flattened_data = MultiprocessingSerializer.deserialize(
                recv_req.serialized_tensors
            )
            bucket = FlattenedTensorBucket(
                flattened_tensor=flattened_data["flattened_tensor"],
                metadata=flattened_data["metadata"],
            )
            tensors = dict(bucket.reconstruct_tensors())
        else:
            tensors = MultiprocessingSerializer.deserialize(recv_req.serialized_tensors)
        result = self.model_runner.load_lora_adapter_from_tensors(
            recv_req.to_ref(),
            tensors,
            recv_req.config_dict,
            recv_req.added_tokens_config,
        )
        return result

    def forward_batch_embedding(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch).logits_output
        embeddings = logits_output.embeddings
        return embeddings


class TpModelWorker(BaseTpWorker):
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        is_multi_layer_eagle: bool = False,
    ):
        # Parse args
        self.server_args = server_args
        self.tp_size = server_args.tp_size
        self.ep_size = server_args.ep_size
        self.pp_size = server_args.pp_size
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        self.is_draft_worker = is_draft_worker
        self.is_multi_layer_eagle = is_multi_layer_eagle
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.memory_pool_config = memory_pool_config
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank

        # MTP model runners
        self.model_runner_list: List[ModelRunner] = []

        self._init_model_config()
        self._init_model_runner()

        if is_multi_layer_eagle:
            self._init_multi_layer_eagle_model_runners()

        self._init_dllm_algorithm()

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
        self.device = self.model_runner.device
        self.specstream_draft_stream = None
        self.specstream_smctrl = None
        self._last_specstream_draft_tpc_range = None
        # A remote SPECTRE Drafter is a normal TpModelWorker whose role is
        # selected by --spectre-role draft.  ``is_draft_worker`` instead marks
        # an in-process EAGLE/MTP draft model and is false for this server.
        # Using that flag here skipped libsmctrl initialization until the first
        # grant attempted to apply a mask.
        if should_initialize_drafter_smctrl(server_args):
            self._init_specstream_draft_smctrl()

        # Init nccl groups
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = self.model_runner.max_running_requests
        assert self.max_running_requests > 0, "max_running_request is zero"
        self.max_queued_requests = server_args.max_queued_requests
        assert (
            self.max_queued_requests is None or self.max_queued_requests >= 1
        ), "If configured, max_queued_requests must be at least 1 for any work to be scheduled."
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.model_runner.max_token_pool_size - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        # Sync random seed across TP workers
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_size * self.pp_rank + tp_rank,
            self.world_group.cpu_group,
            src=self.world_group.ranks[0],
        )[0]
        set_random_seed(self.random_seed)

        self.enable_overlap = not server_args.disable_overlap_schedule
        self.enable_spec = server_args.speculative_algorithm is not None
        self.hicache_layer_transfer_counter = None

    def _init_model_config(self):
        from sglang.srt.configs.model_config import ModelConfig

        self.model_config = ModelConfig.from_server_args(
            self.server_args,
            model_path=(
                self.server_args.model_path
                if not self.is_draft_worker
                else self.server_args.speculative_draft_model_path
            ),
            model_revision=(
                self.server_args.revision
                if not self.is_draft_worker
                else self.server_args.speculative_draft_model_revision
            ),
            is_draft_model=self.is_draft_worker,
        )

    def _init_model_runner(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        self._model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=self.server_args.mem_fraction_static,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.ep_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port,
            dp_rank=self.dp_rank,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
            draft_model_idx=0 if self.is_multi_layer_eagle else None,
        )

    def _init_multi_layer_eagle_model_runners(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        self.model_runner_list.append(self.model_runner)
        for i in range(1, self.server_args.speculative_num_steps):
            self.model_runner_list.append(
                ModelRunner(
                    model_config=self.model_config,
                    mem_fraction_static=self.server_args.mem_fraction_static,
                    gpu_id=self.gpu_id,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                    moe_ep_rank=self.moe_ep_rank,
                    moe_ep_size=self.ep_size,
                    pp_rank=self.pp_rank,
                    pp_size=self.pp_size,
                    nccl_port=self.nccl_port,
                    dp_rank=self.dp_rank,
                    server_args=self.server_args,
                    is_draft_worker=self.is_draft_worker,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    memory_pool_config=self.memory_pool_config,
                    draft_model_idx=i,
                )
            )

    def _init_dllm_algorithm(self):
        from sglang.srt.dllm.algorithm.base import DllmAlgorithm

        if self.server_args.dllm_algorithm is not None:
            self.dllm_algorithm = DllmAlgorithm.from_server_args(self.server_args)
        else:
            self.dllm_algorithm = None

    @property
    def model_runner(self) -> "ModelRunner":
        return self._model_runner

    def register_hicache_layer_transfer_counter(self, counter: LayerDoneCounter):
        self.hicache_layer_transfer_counter = counter

    def set_hicache_consumer(self, consumer_index: int):
        if self.hicache_layer_transfer_counter is not None:
            self.hicache_layer_transfer_counter.set_consumer(consumer_index)

    def register_hisparse_coordinator(self, coordinator):
        self.model_runner.hisparse_coordinator = coordinator

    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            self.model_runner.forward_stream,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def is_dllm(self):
        return self.dllm_algorithm is not None

    def _init_specstream_draft_smctrl(self) -> None:
        from sglang.srt.speculative.spectre.specstream.sm_controller import (
            SMController,
        )

        with torch.cuda.device(self.device):
            self.specstream_draft_stream = torch.cuda.Stream(
                device=self.device, priority=0
            )
            self.specstream_smctrl = SMController(
                getattr(self.server_args, "specstream_smctrl_library", "") or None,
                device_index=self.gpu_id,
                mask_scope=getattr(
                    self.server_args, "specstream_smctrl_mask_scope", "stream"
                ),
            )
            logger.info(
                "SpecStream Draft TPC control initialized: scope=%s total_tpcs=%d",
                self.specstream_smctrl.mask_scope,
                self.specstream_smctrl.total_tpcs,
            )

    def set_specstream_draft_tpc_mask(self, tpc_low: int, tpc_high: int) -> None:
        if self.specstream_smctrl is None or self.specstream_draft_stream is None:
            raise RuntimeError(
                "SpecStream Draft TPC mask requested but libsmctrl is not initialized"
            )
        requested = (int(tpc_low), int(tpc_high))
        if requested == self._last_specstream_draft_tpc_range:
            return
        self.specstream_smctrl.set_stream_mask(self.specstream_draft_stream, *requested)
        self._last_specstream_draft_tpc_range = requested

    def get_specstream_draft_smctrl_info(self) -> tuple[str, int]:
        if self.specstream_smctrl is None or self.specstream_draft_stream is None:
            raise RuntimeError(
                "SpecStream SM control is enabled on the remote Drafter, but "
                "libsmctrl was not initialized during worker startup"
            )
        return (
            str(self.specstream_smctrl.mask_scope),
            int(self.specstream_smctrl.total_tpcs),
        )

    def synchronize_specstream_draft_stream(self) -> None:
        """Close a grant only after all forward and sampling work completes."""
        if self.specstream_draft_stream is None:
            raise RuntimeError("SpecStream Draft stream is not initialized")
        self.specstream_draft_stream.synchronize()

    def _forward_batch_generation_dllm(
        self, forward_batch: ForwardBatch
    ) -> GenerationBatchResult:
        logits_output, next_token_ids, can_run_cuda_graph = self.dllm_algorithm.run(
            self.model_runner, forward_batch
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=can_run_cuda_graph,
        )


    def specstream_begin_controlled_draft_forward(self) -> None:
        self._specstream_controlled_draft_forward = True

    def specstream_end_controlled_draft_forward(self) -> None:
        self._specstream_controlled_draft_forward = False

    def specstream_draft_mask_active(self) -> bool:
        """Return whether a real Draft TPC range has been installed."""
        return self._last_specstream_draft_tpc_range is not None

    def specstream_total_tpcs(self):
        """Return the physical TPC count reported by the native SMController."""
        if self.specstream_smctrl is None:
            return None
        return int(self.specstream_smctrl.total_tpcs)

    def specstream_apply_draft_tpc_range(
        self,
        tpc_low: int,
        tpc_high: int,
    ) -> None:
        """Install a Draft TPC range through SpecStream's native SM-control path.

        SMController intentionally exposes set_stream_mask() for both mask scopes.
        With mask_scope="global", set_stream_mask() internally calls
        libsmctrl_set_global_mask().  Therefore we must reuse
        set_specstream_draft_tpc_mask() instead of looking for nonexistent Python
        methods such as set_global_mask() or set_global_tpc_range().
        """
        low = int(tpc_low)
        high = int(tpc_high)

        if low < 0 or high <= low:
            raise ValueError(
                f"invalid Draft TPC range [{low}, {high})"
            )

        if self.specstream_smctrl is None:
            raise RuntimeError(
                "SpecStream Draft SMController is not initialized"
            )

        total_tpcs = int(self.specstream_smctrl.total_tpcs)
        if high > total_tpcs:
            raise ValueError(
                f"Draft TPC range [{low}, {high}) exceeds "
                f"total_tpcs={total_tpcs}"
            )

        # This is the native and correct implementation:
        #
        #   set_specstream_draft_tpc_mask()
        #       -> self.specstream_smctrl.set_stream_mask(
        #              self.specstream_draft_stream, low, high
        #          )
        #
        # When mask_scope == "global", SMController itself invokes
        # libsmctrl_set_global_mask().
        self.set_specstream_draft_tpc_mask(low, high)

        # Compatibility marker used by the first-grant bootstrap guard.
        self._specstream_active_draft_tpc_range = (low, high)

    def specstream_bootstrap_calibration_mask_if_needed(self) -> None:
        if self.specstream_draft_mask_active():
            return
        if not bool(getattr(self.server_args, "specstream_smctrl_enabled", False)):
            return
        tpcs = int(
            getattr(self.server_args, "specstream_smctrl_calibration_tpcs", 0) or 0
        )
        if tpcs <= 0:
            return
        self.specstream_apply_draft_tpc_range(0, tpcs)

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init=False,
    ) -> GenerationBatchResult:
        if self.specstream_draft_stream is None:
            return self._forward_batch_generation_impl(
                model_worker_batch,
                forward_batch=forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                is_verify=is_verify,
                skip_attn_backend_init=skip_attn_backend_init,
            )

        # Only a scheduler-marked remote Draft forward is grant-controlled.
        # Ordinary/local/warmup forwards keep the native SGLang path.
        if getattr(self, "_specstream_controlled_draft_forward", False):
            # Fixed-TPC calibration must establish a real spatial mask before the
            # first controlled forward.
            self.specstream_bootstrap_calibration_mask_if_needed()
            if not self.specstream_draft_mask_active():
                raise RuntimeError(
                    "SpecStream refused an ungranted Draft forward: "
                    "no TPC mask is active"
                )

        caller_stream = torch.cuda.current_stream(self.device)
        self.specstream_draft_stream.wait_stream(caller_stream)
        with torch.cuda.stream(self.specstream_draft_stream):
            result = self._forward_batch_generation_impl(
                model_worker_batch,
                forward_batch=forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                is_verify=is_verify,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            done = torch.cuda.Event()
            done.record(self.specstream_draft_stream)
        caller_stream.wait_event(done)

        delayed = result.delay_sample_func
        if delayed is not None:
            draft_stream = self.specstream_draft_stream

            def sample_on_draft_stream():
                current = torch.cuda.current_stream(self.device)
                draft_stream.wait_stream(current)
                with torch.cuda.stream(draft_stream):
                    sampled = delayed()
                    sample_done = torch.cuda.Event()
                    sample_done.record(draft_stream)
                current.wait_event(sample_done)
                return sampled

            result.delay_sample_func = sample_on_draft_stream
        return result

    def _forward_batch_generation_impl(
        self,
        model_worker_batch: ModelWorkerBatch,
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init=False,
    ) -> GenerationBatchResult:
        # FIXME(lsyin): maybe remove skip_attn_backend_init in forward_batch_generation,
        #               which requires preparing replay to always be in this function

        # Get forward batch from model worker batch
        if model_worker_batch is not None:
            # update the consumer index of hicache to the running batch
            self.set_hicache_consumer(model_worker_batch.hicache_consumer_index)

            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        else:
            # FIXME(lsyin): unify the interface of forward_batch
            assert forward_batch is not None

        if self.is_dllm():
            return self._forward_batch_generation_dllm(forward_batch)

        if self.pp_group.is_last_rank:
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            batch_result = GenerationBatchResult(
                logits_output=logits_output,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
            )

            if is_verify:
                # Skip sampling and return logits for target forward
                return batch_result

            if (
                self.enable_overlap
                and not self.enable_spec
                and model_worker_batch.sampling_info.grammars is not None
            ):

                def sample_batch_func():
                    batch_result.next_token_ids = self.model_runner.sample(
                        logits_output, forward_batch
                    )
                    return batch_result

                batch_result.delay_sample_func = sample_batch_func
                return batch_result

            if not model_worker_batch.is_prefill_only:
                # For normal requests, sample the next token ids.
                batch_result.next_token_ids = self.model_runner.sample(
                    logits_output, forward_batch
                )
            else:
                # For prefill-only requests, create dummy token IDs on CPU
                # The size should match the batch size (number of sequences), not total tokens
                batch_result.next_token_ids = torch.zeros(
                    len(model_worker_batch.seq_lens),
                    dtype=torch.long,
                    device=model_worker_batch.input_ids.device,
                )
                if (
                    model_worker_batch.return_logprob
                    and logits_output.next_token_logits is not None
                ):
                    # NOTE: Compute logprobs without full sampling
                    self.model_runner.compute_logprobs_only(
                        logits_output, model_worker_batch
                    )

            return batch_result
        else:
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            pp_proxy_tensors, can_run_cuda_graph = out.logits_output, out.can_run_graph
            return GenerationBatchResult(
                pp_hidden_states_proxy_tensors=pp_proxy_tensors,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
            )

    def forward_batch_split_prefill(self, batch: ScheduleBatch):
        if batch.split_index == 0:
            model_worker_batch = batch.get_model_worker_batch()
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
            batch.split_forward_batch = forward_batch
            batch.seq_lens_cpu_cache = model_worker_batch.seq_lens_cpu
        else:
            model_worker_batch = batch.get_model_worker_batch(batch.seq_lens_cpu_cache)

        out = self.model_runner.forward(
            batch.split_forward_batch, split_forward_count=batch.split_forward_count
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        if logits_output:
            next_token_ids = self.model_runner.sample(logits_output, model_worker_batch)
        else:
            next_token_ids = None
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            expert_distribution_metrics=out.expert_distribution_metrics,
        )
        batch_result.next_token_ids = next_token_ids
        return batch_result
