"""Run native GPU Draft lookahead in a fresh CUDA process for each TP rank.

The parent retains ownership of the exported CUDA storage and slot mappings.
Use torch.multiprocessing to send the exported tensors. The caller must order
CUDA access across processes and drain the child before recycling any slot.
"""

from copy import copy
from types import SimpleNamespace

import torch


_POOL_FIELDS = (
    "size", "page_size", "dtype", "head_num", "head_dim", "v_head_dim",
    "layer_num", "start_layer", "end_layer",
)


def export_draft_model_state(runner):
    """Describe one loaded TP shard and export its original CUDA tensor views."""
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    args = runner.server_args
    model_config = runner.model_config
    pool = runner.token_to_kv_pool
    if (
        not runner.is_draft_worker
        or runner.tp_size != 2
        or runner.pp_size != 1
        or args.device != "cuda"
        or args.enable_dp_attention
        or args.attn_cp_size != 1
        or args.attention_backend != "fa3"
        or args.speculative_algorithm != "STANDALONE"
        or args.speculative_eagle_topk != 1
        or not args.disable_cuda_graph
        or not args.disable_piecewise_cuda_graph
        or model_config.hf_config.architectures != ["Qwen3ForCausalLM"]
        or model_config.quantization is not None
        or model_config.dtype != torch.bfloat16
        or type(pool) is not MHATokenToKVPool
        or pool.dtype != torch.bfloat16
        or pool.page_size != 1
        or pool.layer_transfer_counter is not None
        or getattr(pool, "native_joint_arena", None) is not None
    ):
        raise ValueError("MPS lookahead needs BF16 Qwen3 TP2, FA3, no graphs and static GPU Draft KV")

    # Rotary caches are nonpersistent buffers. A state_dict alone omits them.
    bindings = {}

    def export_tensors(named_tensors):
        for name, tensor in named_tensors:
            if id(tensor) not in bindings:
                bindings[id(tensor)] = tensor.detach()
            yield name, bindings[id(tensor)]

    buffers = dict(runner.model.named_buffers(remove_duplicate=False))
    moved_buffers = {}
    for name, buffer in buffers.items():
        if buffer.device.type == "cpu":
            key = id(buffer)
            if key not in moved_buffers:
                moved_buffers[key] = buffer.to(device=runner.device)
            module_name, _, member = name.rpartition(".")
            buffers[name] = moved_buffers[key]
            runner.model.get_submodule(module_name)._buffers[member] = buffers[name]
    return {
        "server_args": copy(args),
        "model_config": copy(model_config),
        "gpu_id": runner.gpu_id,
        "tp_rank": runner.tp_rank,
        "parameters": dict(export_tensors(runner.model.named_parameters(remove_duplicate=False))),
        "buffers": dict(export_tensors(buffers.items())),
        "pool_fields": {name: getattr(pool, name) for name in _POOL_FIELDS},
        "pool_alt_stream": pool.alt_stream is not None,
        "k_buffer": tuple(pool.k_buffer),
        "v_buffer": tuple(pool.v_buffer),
    }


def _bind_module_tensors(model, parameters, buffers, device):
    parameter_placeholders = dict(model.named_parameters(remove_duplicate=False))
    buffer_placeholders = dict(model.named_buffers(remove_duplicate=False))
    installed = {}
    for sources, placeholders, is_parameter in (
        (parameters, parameter_placeholders, True),
        (buffers, buffer_placeholders, False),
    ):
        if sources.keys() != placeholders.keys():
            mismatch = sorted(sources.keys() ^ placeholders.keys())
            raise ValueError(f"Draft model export does not match the child module: {mismatch}")
        for name, source in sources.items():
            placeholder = placeholders[name]
            if not is_parameter and name.endswith(".rotary_emb.cos_sin_cache"):
                # The loaded parent may already have grown or cast its cache.
                shape_matches = source.ndim == placeholder.ndim and source.shape[1:] == placeholder.shape[1:]
                dtype_matches = source.is_floating_point()
            else:
                shape_matches = source.shape == placeholder.shape
                dtype_matches = source.dtype == placeholder.dtype
            if not shape_matches or not dtype_matches or source.device != device:
                raise ValueError(
                    f"Invalid export for {name}: parent {source.shape}/{source.dtype}/{source.device}, "
                    f"child {placeholder.shape}/{placeholder.dtype}, expected device {device}"
                )
            module_name, _, member = name.rpartition(".")
            module = model.get_submodule(module_name)
            if is_parameter:
                key = id(source)
                if key not in installed:
                    parameter = torch.nn.Parameter(source, requires_grad=False)
                    parameter.__dict__.update(placeholder.__dict__)
                    installed[key] = parameter
                module._parameters[member] = installed[key]
            else:
                module._buffers[member] = source


def _import_kv_pool(state, device):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    fields = state["pool_fields"]
    for name, head_dim in (("k_buffer", "head_dim"), ("v_buffer", "v_head_dim")):
        tensors = state[name]
        expected = (fields["size"] + fields["page_size"], fields["head_num"], fields[head_dim])
        if len(tensors) != fields["layer_num"] or any(
            t.shape != expected or t.dtype != fields["dtype"] or t.device != device
            for t in tensors
        ):
            raise ValueError(f"Imported Draft {name} differs from the parent KV geometry")

    class ImportedKVPool(MHATokenToKVPool):
        def _create_buffers(self):
            self.k_buffer = list(state["k_buffer"])
            self.v_buffer = list(state["v_buffer"])

    # Lookahead only reads existing pages and writes scratch pages. The parent owns
    # remapping and rollback, so the child does not initialize KV relocation kernels.
    return ImportedKVPool(
        **fields,
        device=device,
        enable_memory_saver=False,
        enable_alt_stream=state["pool_alt_stream"],
    )


class NativeMPSDraftModel:
    """The model and attention side of a spawned, rank-local Draft worker."""

    def __init__(self, state, nccl_port):
        from sglang.srt.distributed.parallel_state import (
            get_tp_group,
            init_distributed_environment,
            initialize_model_parallel,
            make_lookahead_tp_group,
            set_custom_all_reduce,
            set_mscclpp_all_reduce,
            set_torch_symm_mem_all_reduce,
        )
        from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
        from sglang.srt.layers.dp_attention import initialize_dp_attention
        from sglang.srt.model_loader.utils import set_default_torch_dtype
        from sglang.srt.models.qwen3 import Qwen3ForCausalLM
        from sglang.srt.server_args import set_global_server_args_for_scheduler
        from sglang.srt.speculative.native_draft_spill import NativeDraftSpill

        if torch.distributed.is_initialized():
            raise RuntimeError("Start the MPS Draft worker in a fresh spawned process")
        self.rank = state["tp_rank"]
        self.device = torch.device("cuda", state["gpu_id"])
        self.tp_group = None
        args = state["server_args"]
        model_config = state["model_config"]
        torch.cuda.set_device(self.device)
        set_global_server_args_for_scheduler(args)
        args.use_mla_backend = False
        set_custom_all_reduce(False)
        set_mscclpp_all_reduce(False)
        set_torch_symm_mem_all_reduce(False)
        try:
            init_distributed_environment(
                backend="nccl", world_size=2, rank=self.rank,
                local_rank=self.device.index,
                distributed_init_method=f"tcp://127.0.0.1:{nccl_port}",
                timeout=45,
            )
            initialize_model_parallel(tensor_model_parallel_size=2)
            initialize_dp_attention(server_args=args, model_config=model_config)
            self.tp_group = make_lookahead_tp_group(get_tp_group())

            with set_default_torch_dtype(model_config.dtype), torch.device("meta"):
                self.model = Qwen3ForCausalLM(model_config.hf_config)
            _bind_module_tensors(self.model, state["parameters"], state["buffers"], self.device)
            self.model.eval()

            self.pool = _import_kv_pool(state, self.device)
            self.pool.native_draft_spill = NativeDraftSpill(self.pool, self.pool.size, self.rank, 1)
            self.runner = SimpleNamespace(
                server_args=args,
                model_config=model_config,
                model=self.model,
                device=self.device,
                req_to_token_pool=SimpleNamespace(req_to_token=None),
                token_to_kv_pool=self.pool,
                kv_cache_dtype=self.pool.dtype,
                page_size=self.pool.page_size,
                attn_cp_size=1,
                sliding_window_size=None,
            )
            self.backend = FlashAttentionBackend(self.runner, fa_impl_ver=3)
            self.runner.attn_backend = self.backend
        except BaseException:
            self.close()
            raise

    @torch.no_grad()
    def forward(self, frame_values, metadata_values, plan_values):
        """Use a parent-built phase descriptor and return local Draft logits."""
        from sglang.srt.distributed.parallel_state import use_lookahead_tp_group
        from sglang.srt.layers.attention.flashattention_backend import FlashAttentionMetadata
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
        from sglang.srt.speculative.native_draft_spill import DraftAttentionPlan

        if plan_values.get("cpu_rows") or plan_values.get("cpu_queries") or plan_values.get("cpu_groups"):
            raise ValueError("The MPS Draft worker only accepts all-GPU lookahead plans")
        if frame_values["forward_mode"] != ForwardMode.DRAFT_EXTEND:
            raise ValueError("The MPS Draft worker expects native DRAFT_EXTEND lookahead")

        def device_values(values):
            return {
                name: value.to(self.device, non_blocking=True)
                if isinstance(value, torch.Tensor) and not name.endswith("_cpu") else value
                for name, value in values.items()
            }

        plan = DraftAttentionPlan(
            **device_values({"cpu_rows": [], "cpu_queries": [], **plan_values})
        )
        frame = ForwardBatch(**device_values({
            **frame_values, "token_to_kv_pool": self.pool, "attn_backend": self.backend,
        }))
        self.backend.forward_metadata = FlashAttentionMetadata(**device_values(metadata_values))
        self.backend.forward_metadata.native_draft_spill_plan = plan
        try:
            with use_lookahead_tp_group(self.tp_group):
                frame.prepare_attn_tp_scatter_input(self.runner)
                return self.model(frame.input_ids, frame.positions, frame)
        finally:
            self.backend.forward_metadata = None

    def close(self):
        """Drain CUDA work before the caller invalidates parent-owned storage."""
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        torch.cuda.synchronize(self.device)
        if self.tp_group is not None:
            self.tp_group.destroy()
            self.tp_group = None
        if torch.distributed.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
