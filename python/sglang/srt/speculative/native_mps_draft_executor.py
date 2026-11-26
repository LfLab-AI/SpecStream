"""Process transport for native GPU-only Draft lookahead on a fixed KV pool."""

from dataclasses import MISSING, fields
import atexit
import os
import socket
import time
import traceback

import torch
from torch import multiprocessing as mp

from sglang.srt.speculative.spec_utils import fast_topk


def _draft_process(pipe, port):
    try:
        from sglang.srt.speculative.native_mps_draft_model import NativeMPSDraftModel

        draft = NativeMPSDraftModel(pipe.recv(), nccl_port=port)
        pipe.send(("ok", os.getpid()))
        tokens = None
        while True:
            command = pipe.recv()
            if command is None:
                break
            frame, metadata, plan, reuse_previous = command
            if reuse_previous:
                frame["input_ids"] = tokens.reshape(-1)
            device = next(draft.runner.model.parameters()).device
            with torch.no_grad(), torch.cuda.device(device):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                logits = draft.forward(frame, metadata, plan)
                probs, tokens = fast_topk(torch.softmax(logits.next_token_logits, dim=-1), 1, dim=-1)
                end.record()
                end.synchronize()
                end_ns = time.monotonic_ns()
                start_ns = end_ns - round(begin.elapsed_time(end) * 1e6)
            pipe.send(("ok", (start_ns, end_ns, probs, tokens, logits.hidden_states)))
        draft.close()
    except EOFError:
        return
    except BaseException:
        try:
            pipe.send(("error", traceback.format_exc()))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        pipe.close()


def _descriptor(value, skip=()):
    return {
        field.name: getattr(value, field.name)
        for field in fields(value)
        if field.init and field.name not in skip and (
            getattr(value, field.name) is not None
            or field.default is MISSING and field.default_factory is MISSING
        )
    }


class NativeMPSDraftExecutor:
    """Keep scheduling/rollback in the native rank, dispatch only its Draft GPU forward."""

    def __init__(self, runner, group):
        from sglang.srt.speculative.native_mps_draft_model import export_draft_model_state

        if not os.getenv("CUDA_MPS_PIPE_DIRECTORY"):
            raise ValueError("The process backend requires a private CUDA MPS service")
        if mp.current_process().daemon:
            raise RuntimeError("The scheduler must allow spawned Draft processes")
        port = None
        if group.rank_in_group == 0:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
        port = group.broadcast_object(port, src=0)
        context = mp.get_context("spawn")
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=_draft_process, args=(child, port), daemon=True)
        self.process.start()
        child.close()
        try:
            self.pipe.send(export_draft_model_state(runner))
            self.pid = self._receive(420)
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _receive(self, seconds):
        if not self.pipe.poll(seconds):
            raise TimeoutError(f"MPS Draft worker {self.process.pid} returned no response")
        status, value = self.pipe.recv()
        if status != "ok":
            raise RuntimeError(f"MPS Draft worker {self.process.pid} failed:\n{value}")
        return value

    def forward(self, frame, metadata, *, reuse_previous):
        plan = metadata.native_draft_spill_plan
        if plan.cpu_rows or plan.cpu_queries:
            raise RuntimeError("MPS comparison requires a fully GPU-resident Draft prefix")
        skip = ("token_to_kv_pool", "req_to_token_pool", "attn_backend")
        if reuse_previous:
            # An imported IPC tensor cannot be exported back to its producer.
            skip += ("input_ids",)
        self.pipe.send((_descriptor(frame, skip), _descriptor(metadata),
                        _descriptor(plan), reuse_previous))
        return self._receive(120)

    def close(self):
        try:
            if not self.pipe.closed:
                if self.process.is_alive():
                    self.pipe.send(None)
                    self.process.join(timeout=5)
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            self.pipe.close()
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
