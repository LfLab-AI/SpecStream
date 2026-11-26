"""Process transport for native Draft lookahead on fixed GPU and shared CPU KV."""

from dataclasses import MISSING, fields
import atexit
import os
import socket
import time
import traceback
import weakref

import torch
from torch import multiprocessing as mp

from sglang.srt.speculative.spec_utils import fast_topk


def _draft_process(pipe, port):
    draft = None
    try:
        from sglang.srt.speculative.native_mps_draft_model import NativeMPSDraftModel

        draft = NativeMPSDraftModel(pipe.recv(), nccl_port=port)
        pipe.send(("ok", os.getpid()))
        tokens = None
        cpu_rows = None
        while True:
            command = pipe.recv()
            if command is None:
                break
            if command[0] == "history":
                draft.set_cpu_history(command[1])
                cpu_rows = None
                pipe.send(("ok", None))
                continue

            mode, frame, metadata, plan, reuse_previous, reuse_cpu_rows = command
            if mode != "forward":
                raise ValueError(f"Unknown MPS Draft command: {mode}")
            if reuse_previous:
                frame["input_ids"] = tokens.reshape(-1)
            if reuse_cpu_rows:
                if cpu_rows is None:
                    raise RuntimeError("MPS Draft has no CPU slot rows to reuse")
                plan["cpu_rows"] = cpu_rows
            else:
                cpu_rows = plan.get("cpu_rows")
            device = next(draft.runner.model.parameters()).device
            cpu_intervals = []
            draft.set_cpu_attention_intervals(cpu_intervals)
            try:
                with torch.no_grad(), torch.cuda.device(device):
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    cpu_before = draft.cpu_attention_ns
                    begin.record()
                    logits = draft.forward(frame, metadata, plan)
                    probs, tokens = fast_topk(torch.softmax(logits.next_token_logits, dim=-1), 1, dim=-1)
                    end.record()
                    end.synchronize()
                    end_ns = time.monotonic_ns()
                    start_ns = end_ns - round(begin.elapsed_time(end) * 1e6)
                    cpu_ns = draft.cpu_attention_ns - cpu_before
            finally:
                draft.set_cpu_attention_intervals(None)
            pipe.send(("ok", (start_ns, end_ns, probs, tokens, logits.hidden_states, cpu_ns, cpu_intervals)))
    except EOFError:
        return
    except BaseException:
        try:
            pipe.send(("error", traceback.format_exc()))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        try:
            if draft is not None:
                draft.close()
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
        self._cpu_history = None
        self._cpu_rows = None
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

    def attach_cpu_history(self, history):
        """Install the current history generation before a speculative round starts."""
        if history is None:
            raise ValueError("Cannot attach a missing Draft CPU history")
        if self._cpu_history is not None and self._cpu_history() is history:
            return
        self.pipe.send(("history", history.export_descriptor()))
        self._receive(120)
        self._cpu_history = weakref.ref(history)
        self._cpu_rows = None

    def detach_cpu_history(self):
        """Release the child view after the parent has drained the round."""
        if self._cpu_history is not None:
            self.pipe.send(("history", None))
            self._receive(120)
            self._cpu_history = None
            self._cpu_rows = None

    def forward(self, frame, metadata, *, reuse_previous):
        plan = metadata.native_draft_spill_plan
        if plan.cpu_rows and (self._cpu_history is None or self._cpu_history() is None):
            raise RuntimeError("Attach the current CPU Draft history before processing a mixed plan")
        skip = ("token_to_kv_pool", "req_to_token_pool", "attn_backend")
        if reuse_previous:
            # An imported IPC tensor cannot be exported back to its producer.
            skip += ("input_ids",)
        # All phases share the long CPU slot lists. Only their query offsets
        # change. Materialize the grouped tensors in the child process.
        reuse_cpu_rows = bool(reuse_previous and plan.cpu_rows and plan.cpu_rows is self._cpu_rows)
        skip_plan = ("cpu_groups", "cpu_rows") if reuse_cpu_rows else ("cpu_groups",)
        self.pipe.send(("forward", _descriptor(frame, skip), _descriptor(metadata),
                        _descriptor(plan, skip_plan), reuse_previous, reuse_cpu_rows))
        self._cpu_rows = plan.cpu_rows
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
