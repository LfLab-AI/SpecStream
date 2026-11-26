"""Causal Draft attention over disjoint CPU and GPU KV pages."""

import os
import time

import torch

from sglang.srt.layers.attention.triton_ops.merge_state import merge_state_triton


class SplitDraftAttention:
    """Overlap host attention and a GPU attention partial within a single layer.

    The caller supplies the post-RoPE query, CPU slots allowed by the causal
    mask, and a callable that enqueues the GPU partial on the current stream.
    The two slot sets must be disjoint and together cover all visible keys.
    A query with no host slots still needs a nonempty GPU partial.
    """

    def __init__(self, history, device):
        self.history = history
        self.device = torch.device(device)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.download_done = torch.cuda.Event()
        self.upload_done = torch.cuda.Event()
        self.upload_in_flight = False
        self.buffers = {}
        self.time_cpu_attention = os.getenv("SPECSTREAM_CPU_ATTN_TIMING", "0") == "1"
        self.cpu_attention_ns = 0
        self.cpu_attention_intervals = None

    def _pinned(self, name, shape, dtype):
        size = shape.numel()
        buffer = self.buffers.get(name)
        if buffer is None or buffer.numel() < size:
            buffer = torch.empty(size, dtype=dtype, device="cpu", pin_memory=True)
            self.buffers[name] = buffer
        return buffer[:size].view(shape)

    def forward(self, layer, q, cpu_slots, cpu_lens, scale, gpu_partial, *, query_offsets=None):
        """Return full attention in Q's shape and dtype.

        GPU states carry a normalized output [queries, heads, head_dim] and
        natural-log LSE [queries, heads]. The layer ID is local to the CPU pool.
        If query_offsets is supplied, CPU slot rows are per request. Its
        consecutive queries all see the same committed CPU prefix.
        """
        if q.ndim != 3 or q.dtype != torch.bfloat16 or not q.is_cuda:
            raise ValueError("Split Draft attention expects CUDA BF16 Q[queries, heads, head_dim]")
        if not bool(cpu_lens.any()):
            return gpu_partial()[0]
        current = torch.cuda.current_stream(q.device)
        contiguous_q = q.contiguous()
        q_host = self._pinned("q", q.shape, q.dtype)
        self.copy_stream.wait_stream(current)
        with torch.cuda.stream(self.copy_stream):
            q_host.copy_(contiguous_q, non_blocking=True)
            self.download_done.record(self.copy_stream)
        try:
            gpu_out, gpu_lse = gpu_partial()
        finally:
            self.download_done.synchronize()
        if gpu_out.shape != q.shape or gpu_lse.shape != q.shape[:2] or gpu_out.dtype != q.dtype:
            raise ValueError("GPU attention state differs from the Draft query shape or dtype")

        if self.upload_in_flight:
            self.upload_done.synchronize()
        host_out = self._pinned("out", q.shape, torch.float32)
        host_lse = self._pinned("lse", q.shape[:2], torch.float32)
        intervals = self.cpu_attention_intervals
        if self.time_cpu_attention or intervals is not None:
            start_ns = time.monotonic_ns()
        if query_offsets is None:
            self.history.decode_state(layer, q_host, cpu_slots, cpu_lens, scale, out=host_out, lse=host_lse)
        else:
            self.history.decode_grouped_state(
                layer, q_host, cpu_slots, cpu_lens, query_offsets, scale, out=host_out, lse=host_lse,
            )
        if self.time_cpu_attention or intervals is not None:
            end_ns = time.monotonic_ns()
            if self.time_cpu_attention:
                self.cpu_attention_ns += end_ns - start_ns
            if intervals is not None:
                intervals.append((start_ns, end_ns))
        with torch.cuda.stream(self.copy_stream):
            cpu_out = host_out.to(q.device, non_blocking=True)
            cpu_lse = host_lse.to(q.device, non_blocking=True)
            self.upload_done.record(self.copy_stream)
            self.upload_in_flight = True
        current.wait_event(self.upload_done)
        cpu_out.record_stream(current)
        cpu_lse.record_stream(current)
        result, _ = merge_state_triton(
            gpu_out.contiguous(), gpu_lse.float().contiguous(), cpu_out, cpu_lse,
        )
        return result
