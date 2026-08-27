import contextlib
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
)
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    draft_tp_context,
    maybe_detect_nan,
    maybe_detect_oob,
    select_top_k_tokens,
)
from sglang.srt.utils import empty_context, fast_topk, get_bool_env_var, is_cuda

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")


@dataclass
class _LinearDraftProgress:
    model_batch: Any
    draft_input: EagleDraftInput
    forward_batch: ForwardBatch
    out_cache_loc: torch.Tensor
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    hidden_states: torch.Tensor
    scores: Optional[torch.Tensor] = None
    score_list: list[torch.Tensor] = field(default_factory=list)
    token_list: list[torch.Tensor] = field(default_factory=list)
    parents_list: list[torch.Tensor] = field(default_factory=list)
    step: int = 0


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class StandaloneDraftWorker(EagleDraftWorker):
    """Custom EagleDraftWorker that doesn't share embeddings/lm_head with target model."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank

        # Args for easy access
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Set constant
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        # Do not capture cuda graph in `TpModelWorker` init,
        # will capture later with init_cuda_graphs()
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        with empty_context():
            # Init draft worker
            self.draft_worker = TpModelWorker(
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

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner

        self.init_token_map()
        self.init_lm_head()

        # Init attention backend and cuda graphs
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(
            self.draft_runner.tp_group
        ), speculative_moe_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    def init_lm_head(self):
        """Override to prevent sharing embeddings and lm_head with target model."""
        # For standalone worker, we don't share embeddings and lm_head
        # The draft model uses its own embeddings and lm_head
        pass

    def _specstream_extend(
        self,
        batch,
        predict: torch.Tensor,
        num_tokens: int,
    ):
        """Run a local Draft extend without touching Target execution state."""

        batch.forward_mode = ForwardMode.DECODE
        # ForwardBatch pads every EagleDraftInput hidden-state tensor even for
        # a normal standalone CausalLM that never consumes it.  Keep a tiny
        # shape-compatible placeholder instead of copying Target hidden states.
        hidden_placeholder = torch.empty(
            (predict.numel(), 1),
            dtype=self.draft_runner.model_config.dtype,
            device=predict.device,
        )
        draft_input = EagleDraftInput(
            hidden_states=hidden_placeholder,
            num_tokens_per_req=num_tokens,
            num_tokens_for_logprob_per_req=num_tokens,
            accept_length=torch.full(
                (len(batch.seq_lens),),
                num_tokens,
                dtype=torch.int32,
                device=batch.seq_lens.device,
            ),
        )
        forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
            batch,
            predict,
            num_tokens,
            self.draft_runner,
            None,
        )
        logits_output = self.draft_runner.forward(
            forward_batch, skip_attn_backend_init=True
        ).logits_output
        maybe_detect_nan(logits_output.next_token_logits, "specstream_ahead_extend")
        return logits_output

    def launch_specstream_ahead(
        self,
        batch,
        verify_input: EagleVerifyInput,
        ahead_depth: int,
    ):
        """Launch optimistic bonus prediction and h next-round Draft steps.

        The method is called under the low-priority Draft stream.  It assumes
        all q current candidates are accepted, predicts the Target bonus with
        the Drafter, extends that anchor, and then generates up to h tokens of
        the next verification frontier.
        """

        from sglang.srt.speculative.specstream_inproc.ahead_runtime import (
            DraftAheadArtifact,
        )

        batch_size = len(batch.seq_lens)
        token_rows = verify_input.draft_token.reshape(
            batch_size, verify_input.draft_token_num
        )
        current_candidates = token_rows[:, 1:]
        num_candidates = current_candidates.shape[1]

        # Fill Draft KV for the optimistic q-token verification path and use
        # its final logits to predict the Target bonus anchor.
        prefix_batch = self._clone_specstream_batch(batch)
        prefix_logits = self._specstream_extend(
            prefix_batch,
            current_candidates.reshape(-1),
            num_candidates,
        )
        last_indices = (
            torch.arange(batch_size, device=self.device) * num_candidates
            + num_candidates
            - 1
        )
        assumed_bonus = torch.argmax(
            prefix_logits.next_token_logits[last_indices], dim=-1
        ).to(torch.int32)

        # One-token extend establishes the exact Draft state after the assumed
        # bonus.  It writes only Draft KV at the already reserved shared slots.
        bonus_batch = self._clone_specstream_batch(batch)
        bonus_batch.seq_lens = bonus_batch.seq_lens + num_candidates
        bonus_batch.seq_lens_cpu = bonus_batch.seq_lens_cpu + num_candidates
        bonus_batch.seq_lens_sum += batch_size * num_candidates
        bonus_logits = self._specstream_extend(
            bonus_batch,
            assumed_bonus,
            1,
        )
        next_logits = bonus_logits.next_token_logits
        probs = torch.softmax(next_logits, dim=-1)
        topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        next_seq_lens = batch.seq_lens + verify_input.draft_token_num
        next_draft_input = EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=bonus_logits.hidden_states,
            verified_id=assumed_bonus,
            new_seq_lens=next_seq_lens,
        )

        next_batch = self._clone_specstream_batch(batch)
        next_batch.forward_mode = ForwardMode.DECODE
        next_batch.seq_lens = next_seq_lens
        next_batch.seq_lens_cpu = next_batch.seq_lens_cpu + verify_input.draft_token_num
        next_batch.seq_lens_sum += batch_size * verify_input.draft_token_num
        # Ahead generation mutates spec_info.hidden_states while it advances.
        # Preserve the relay input so a later batch-key/cache miss can still
        # execute the ordinary serial Draft path from the correct bonus state.
        progress_input = copy.copy(next_draft_input)
        next_batch.spec_info = progress_input

        if ahead_depth >= self.speculative_num_steps:
            next_verify_input = self._specstream_full_draft(next_batch, progress_input)
            ahead_tokens = next_verify_input.draft_token.reshape(
                batch_size, next_verify_input.draft_token_num
            )
            return DraftAheadArtifact(
                next_draft_input=next_draft_input,
                next_batch=next_batch,
                assumed_bonus=assumed_bonus,
                ahead_tokens=ahead_tokens,
                verify_input=next_verify_input,
            )

        progress = self._start_specstream_linear_draft(next_batch, progress_input)
        self._advance_specstream_linear_draft(progress, ahead_depth)
        generated = torch.cat(progress.token_list, dim=1)
        ahead_tokens = torch.cat((assumed_bonus[:, None], generated), dim=1)
        return DraftAheadArtifact(
            next_draft_input=next_draft_input,
            next_batch=next_batch,
            assumed_bonus=assumed_bonus,
            ahead_tokens=ahead_tokens,
            partial_progress=progress,
        )

    def _specstream_full_draft(
        self,
        batch,
        draft_input: EagleDraftInput,
    ) -> EagleVerifyInput:
        """Generate a full next-round tree with private Target-mask buffers.

        Draft CUDA Graph replay remains enabled, but tree construction does not
        write the Target attention backend's shared graph buffers while the
        current Target Verify may still be reading them.
        """

        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )
        if can_cuda_graph:
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch
            )
        else:
            if not batch.forward_mode.is_idle() and self.speculative_num_steps > 1:
                self.draft_attn_backend.init_forward_metadata(forward_batch)
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
            )

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            None,
            None,
        )
        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def complete_specstream_ahead(self, artifact):
        progress = artifact.partial_progress
        if progress is None:
            if artifact.verify_input is None:
                raise RuntimeError("incomplete ahead artifact has no draft progress")
            return artifact.verify_input
        self._advance_specstream_linear_draft(
            progress, self.speculative_num_steps - progress.step
        )
        artifact.verify_input = self._finish_specstream_linear_draft(progress)
        artifact.partial_progress = None
        return artifact.verify_input

    @staticmethod
    def _clone_specstream_batch(batch):
        cloned = copy.copy(batch)
        cloned.seq_lens = batch.seq_lens.clone()
        cloned.seq_lens_cpu = batch.seq_lens_cpu.clone()
        cloned.seq_lens_sum = int(batch.seq_lens_sum)
        cloned.lora_ids = list(batch.lora_ids)
        return cloned

    def _start_specstream_linear_draft(
        self,
        batch,
        draft_input: EagleDraftInput,
    ) -> _LinearDraftProgress:
        batch.spec_info = draft_input
        forward_batch, _ = draft_input.prepare_for_v2_draft(
            self.req_to_token_pool,
            batch,
            None,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )
        if not batch.forward_mode.is_idle() and self.speculative_num_steps > 1:
            self.draft_attn_backend.init_forward_metadata(forward_batch)

        out_cache_loc = forward_batch.out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )
        topk_index = draft_input.topk_index
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]
        return _LinearDraftProgress(
            model_batch=batch,
            draft_input=draft_input,
            forward_batch=forward_batch,
            out_cache_loc=out_cache_loc,
            topk_p=draft_input.topk_p,
            topk_index=topk_index,
            hidden_states=draft_input.hidden_states,
        )

    def _advance_specstream_linear_draft(
        self,
        progress: _LinearDraftProgress,
        num_steps: int,
    ) -> None:
        stop = min(self.speculative_num_steps, progress.step + num_steps)
        while progress.step < stop:
            step = progress.step
            (
                input_ids,
                hidden_states,
                progress.scores,
                tree_info,
            ) = select_top_k_tokens(
                step,
                progress.topk_p,
                progress.topk_index,
                progress.hidden_states,
                progress.scores,
                self.topk,
            )
            progress.score_list.append(tree_info[0])
            progress.token_list.append(tree_info[1])
            progress.parents_list.append(tree_info[2])
            progress.step += 1

            if step == self.speculative_num_steps - 1:
                break

            forward_batch = progress.forward_batch
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = progress.out_cache_loc[step]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[step]
            forward_batch.spec_info.hidden_states = hidden_states
            logits_output = self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            maybe_detect_nan(
                logits_output.next_token_logits,
                f"specstream_ahead_draft step {step}",
            )
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            progress.topk_p, progress.topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                progress.topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"specstream_ahead_draft step {step}: topk index",
            )
            if self.hot_token_id is not None:
                progress.topk_index = self.hot_token_id[progress.topk_index]
            progress.hidden_states = logits_output.hidden_states

    def _finish_specstream_linear_draft(
        self, progress: _LinearDraftProgress
    ) -> EagleVerifyInput:
        if progress.step != self.speculative_num_steps:
            raise RuntimeError(
                "cannot finish partial Draft before all configured steps complete"
            )
        score_list = torch.cat(progress.score_list, dim=1).flatten(1)
        token_list = torch.cat(progress.token_list, dim=1)
        top_scores_index = torch.topk(
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        ).indices
        top_scores_index = torch.sort(top_scores_index).values
        draft_tokens = torch.gather(token_list, index=top_scores_index, dim=1)
        if len(progress.parents_list) > 1:
            parent_list = torch.cat(progress.parents_list[:-1], dim=1)
        else:
            parent_list = torch.empty(token_list.shape[0], 0, device=token_list.device)

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            progress.draft_input.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            progress.model_batch.seq_lens,
            progress.model_batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            None,
            None,
        )
        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )


class StandaloneWorkerV2(EAGLEWorkerV2):

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
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Create our custom draft worker that doesn't share embeddings/lm_head
        self._draft_worker = StandaloneDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

        self.specstream_inproc_runtime = None
        if server_args.specstream_inproc_enabled:
            from sglang.srt.speculative.specstream_inproc.ahead_runtime import (
                InProcessAheadRuntime,
            )

            self.specstream_inproc_runtime = InProcessAheadRuntime(server_args)
            logger.info(
                "Enabled SpecStream in-process optimistic Draft-ahead: "
                "mode=%s depth=%d",
                server_args.specstream_inproc_mode,
                server_args.specstream_inproc_ahead_depth,
            )

    def clear_cache_pool(self):
        if self.specstream_inproc_runtime is not None:
            self.specstream_inproc_runtime.clear()
        return super().clear_cache_pool()

    def forward_batch_generation(self, model_worker_batch):
        runtime = self.specstream_inproc_runtime
        if runtime is None or (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            if runtime is not None:
                runtime.discard_cached()
            return super().forward_batch_generation(model_worker_batch)

        if model_worker_batch.spec_info is None:
            model_worker_batch.spec_info = EagleDraftInput.create_idle_input(
                device=self.device,
                hidden_size=self.target_worker.model_config.hidden_size,
                dtype=self.target_worker.model_config.dtype,
                topk=self.topk,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )

        verify_input = runtime.consume_cached(model_worker_batch, self.draft_worker)
        if verify_input is None:
            with self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                verify_input = self.draft_worker.draft(model_worker_batch)

        assert verify_input.is_verify_input()
        model_worker_batch.spec_info = verify_input
        ahead_round = runtime.prepare_round(model_worker_batch, verify_input)
        batch_output = self.verify(model_worker_batch)

        if ahead_round is not None:
            with self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                runtime.launch_ahead(ahead_round, self.draft_worker)
            outcome = runtime.reconcile(ahead_round, batch_output)
            if outcome.promoted:
                assert ahead_round.artifact is not None
                next_draft_input = ahead_round.artifact.next_draft_input
                next_draft_input.verify_done = batch_output.next_draft_input.verify_done
                batch_output.next_draft_input = next_draft_input
                return batch_output

        if ahead_round is not None:
            runtime.begin_repair(ahead_round)
        with self.draft_worker.draft_tp_context(
            self.draft_worker.draft_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.draft_worker._draft_extend_for_decode(model_worker_batch, batch_output)
        if ahead_round is not None:
            runtime.mark_repaired(ahead_round)
        return batch_output
