"""CPU-resident BF16 KV with paged attention for the Draft worker."""
from __future__ import annotations

import contextlib
import ctypes
import functools
import hashlib
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading

import torch


@functools.lru_cache(maxsize=1)
def _native_decode():
    source = Path(__file__).with_suffix(".cpp")
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        raise RuntimeError("Building CPU Draft attention requires a C++17 compiler with OpenMP")
    windows = os.name == "nt"
    tag = hashlib.sha256(source.read_bytes() + compiler.encode() +
                         platform.node().encode() + platform.machine().encode()).hexdigest()[:20]
    private_dir = Path(tempfile.gettempdir()) / (
        "specstream-cpu-attention" if windows else f"specstream-cpu-attention-{os.getuid()}"
    )
    private_dir.mkdir(mode=0o700, exist_ok=True)
    if not windows and (private_dir.is_symlink() or private_dir.stat().st_uid != os.getuid()
                        or private_dir.stat().st_mode & 0o077):
        raise RuntimeError(f"CPU attention compiler cache must be private: {private_dir}")
    libpath = private_dir / (tag + (".dll" if windows else ".so"))
    with open(private_dir / (tag + ".lock"), "a+b") as lock:
        if not windows:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        if not libpath.exists():
            with tempfile.TemporaryDirectory(dir=private_dir) as builddir:
                output = Path(builddir) / libpath.name
                flags = ["-O3", "-march=native", "-std=c++17", "-fopenmp", "-shared"]
                flags += ["-static-libgcc", "-static-libstdc++"] if windows else ["-fPIC"]
                command = [compiler, *flags, str(source), "-o", str(output)]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(f"Building CPU attention failed: {result.stderr[-4000:]}")
                os.replace(output, libpath)
    dll_search = (os.add_dll_directory(str(Path(compiler).parent))
                  if windows and hasattr(os, "add_dll_directory") else contextlib.nullcontext())
    with dll_search:
        library = ctypes.CDLL(str(libpath))
    calls = (library.cpu_history_decode, library.cpu_history_decode_state,
             library.cpu_history_decode_grouped_state)
    for call, pointers, dimensions in zip(calls, (6, 7, 8), (6, 6, 7)):
        call.argtypes = ([ctypes.c_void_p] * pointers + [ctypes.c_int] * dimensions
                         + [ctypes.c_float, ctypes.c_int])
        call.restype = ctypes.c_int
    return calls


def _cpu(tensor, dtype):
    if tensor.device.type != "cpu" or tensor.dtype != dtype:
        raise ValueError(f"Expected CPU {dtype}, received {tensor.device}/{tensor.dtype}")


def _bind_draft_worker(tp_rank):
    """Optionally bind this Draft process before the host KV pool is first-touched."""
    masks = os.environ.get("SPECSTREAM_CPU_ATTN_CPUSETS")
    if not masks:
        return None
    if platform.system() != "Linux":
        raise RuntimeError("SPECSTREAM_CPU_ATTN_CPUSETS requires Linux")
    choices = masks.split(";")
    if tp_rank is None or not 0 <= tp_rank < len(choices):
        raise ValueError("Specify a CPU list for each Draft TP rank in SPECSTREAM_CPU_ATTN_CPUSETS")
    allowed = os.sched_getaffinity(0)
    selected = set()
    for spec in choices[tp_rank].split(","):
        bounds = spec.strip().split("-")
        if len(bounds) not in (1, 2) or not all(x.isdecimal() for x in bounds):
            raise ValueError(f"Invalid CPU range {spec!r} for Draft TP rank {tp_rank}")
        low, high = int(bounds[0]), int(bounds[-1])
        if high < low or high > max(allowed):
            raise ValueError(f"Unavailable CPU range {spec!r} for Draft TP rank {tp_rank}")
        selected.update(range(low, high+1))
    if not selected <= allowed:
        raise ValueError(f"Requested CPUs outside current worker affinity: {sorted(selected-allowed)}")
    current_tid = threading.get_native_id()
    changed = []
    preserved = 0
    try:
        # Bind the caller first so threads it subsequently creates inherit the requested mask.
        # Narrow existing OpenMP teams too. Keep externally pinned, disjoint helpers in place.
        tids = [current_tid] + [
            int(entry.name) for entry in Path("/proc/self/task").iterdir()
            if int(entry.name) != current_tid
        ]
        for tid in tids:
            try:
                old = os.sched_getaffinity(tid)
                narrowed = old & selected
                if not narrowed:
                    preserved += 1
                elif narrowed != old:
                    os.sched_setaffinity(tid, narrowed)
                    changed.append((tid, old))
            except ProcessLookupError:
                continue
        if os.sched_getaffinity(0) != selected:
            raise RuntimeError("Worker affinity changed while binding Draft attention")
    except BaseException:
        for tid, old in reversed(changed):
            with contextlib.suppress(ProcessLookupError):
                os.sched_setaffinity(tid, old)
        raise
    logging.getLogger(__name__).info(
        "Draft CPU affinity: tp_rank=%s cpus=%s narrowed=%s preserved_disjoint=%s",
        tp_rank, sorted(selected), len(changed), preserved,
    )
    return selected


class CpuHistoryAttention:
    """The scheduler owns slot lifetimes. Unpublished slots never participate in attention."""

    def __init__(self, num_layers, capacity, num_kv_heads, head_dim=128,
                 dtype=torch.bfloat16, num_threads=None, tp_rank=None):
        self.capacity = int(capacity)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.num_layers = int(num_layers)
        self.dtype = dtype
        if min(self.num_layers, self.capacity, self.num_kv_heads, self.head_dim) < 1:
            raise ValueError("All CPU attention dimensions must be positive")
        if dtype != torch.bfloat16:
            raise ValueError("CPU history attention requires BF16")
        threads = int(os.environ.get("SPECSTREAM_CPU_ATTN_THREADS", "16")
                      if num_threads is None else num_threads)
        if threads < 1:
            raise ValueError("CPU attention requires at least one thread")
        self.implementation = os.environ.get("SPECSTREAM_CPU_ATTN_IMPLEMENTATION", "native")
        if self.implementation not in ("native", "torch"):
            raise ValueError(f"Unknown CPU attention implementation: {self.implementation}")
        if self.implementation == "native" and self.head_dim != 128:
            raise ValueError("The native CPU kernel expects head_dim=128")
        self.affinity = _bind_draft_worker(tp_rank)
        self.num_threads = min(threads, len(self.affinity)) if self.affinity else threads
        self.native_decode, self.native_decode_state, self.native_grouped_state = (
            _native_decode() if self.implementation == "native" else (None, None, None)
        )
        # Ordinary pageable CPU DRAM. Pin only transfer staging, outside this pool.
        # Each CPU worker streams one KV head across all tokens of one request.
        # Keep that axis physically contiguous and expose the scheduler's [layer, slot, head, dim] view.
        shape = (self.num_layers, self.num_kv_heads, self.capacity, self.head_dim)
        self.keys = torch.empty(shape, dtype=dtype, device="cpu").transpose(1, 2)
        self.values = torch.empty_like(self.keys)

    def _check_layer(self, layer):
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"Invalid CPU KV layer {layer}")

    def store(self, layer, slots_cpu, k_cpu, v_cpu, *, contiguous_first=None):
        self._check_layer(layer)
        _cpu(slots_cpu, torch.int64)
        _cpu(k_cpu, self.dtype)
        _cpu(v_cpu, self.dtype)
        expected = (slots_cpu.numel(), self.num_kv_heads, self.head_dim)
        if slots_cpu.ndim != 1 or k_cpu.shape != expected or v_cpu.shape != expected:
            raise ValueError("Invalid slot or KV shape for CPU history store")
        if contiguous_first is not None:
            # The immutable slots were verified once when the forward was prepared.
            n = slots_cpu.numel()
            if contiguous_first < 1 or contiguous_first + n > self.capacity:
                raise ValueError("Contiguous CPU Draft prefill exceeds its physical slot budget")
            self.keys[layer].narrow(0, contiguous_first, n).copy_(k_cpu)
            self.values[layer].narrow(0, contiguous_first, n).copy_(v_cpu)
            return
        self.keys[layer].index_copy_(0, slots_cpu, k_cpu)
        self.values[layer].index_copy_(0, slots_cpu, v_cpu)

    def decode(self, layer, q_cpu, slots_cpu, lengths_cpu, scale, out=None):
        return self._decode(layer, q_cpu, slots_cpu, lengths_cpu, scale, out, None)

    def decode_state(self, layer, q_cpu, slots_cpu, lengths_cpu, scale, *, out=None, lse=None):
        """Return FP32 partial attention and natural-log LSE for disjoint CPU slots.

        A row with zero CPU slots yields a zero output and a negative-infinity LSE.
        The caller supplies the causal prefix length of each post-RoPE query.
        """
        if lse is None:
            lse = torch.empty(q_cpu.shape[:2], dtype=torch.float32, device="cpu")
        return self._decode(layer, q_cpu, slots_cpu, lengths_cpu, scale, out, lse), lse

    def decode_grouped_state(self, layer, q_cpu, slots_cpu, lengths_cpu, query_offsets,
                             scale, *, out=None, lse=None):
        """Attend several consecutive queries to the same immutable CPU history.

        slots and lengths are indexed by request. Query rows for request i span
        [query_offsets[i], query_offsets[i+1]); newer causal KV stays on the GPU.
        """
        if lse is None:
            lse = torch.empty(q_cpu.shape[:2], dtype=torch.float32, device="cpu")
        out = self._decode(layer, q_cpu, slots_cpu, lengths_cpu, scale, out, lse, query_offsets)
        return out, lse

    def _decode(self, layer, q_cpu, slots_cpu, lengths_cpu, scale, out, lse, query_offsets=None):
        self._check_layer(layer)
        _cpu(q_cpu, self.dtype)
        _cpu(slots_cpu, torch.int64)
        _cpu(lengths_cpu, torch.int64)
        if q_cpu.ndim != 3 or slots_cpu.ndim != 2 or lengths_cpu.ndim != 1:
            raise ValueError("Expected Q[B,H,D], slots[B,T], lengths[B]")
        batch, q_heads, dim = q_cpu.shape
        requests = len(lengths_cpu)
        if (not batch or not requests or not q_heads or dim != self.head_dim
                or q_heads % self.num_kv_heads or requests != len(slots_cpu)):
            raise ValueError("Inconsistent shapes for CPU grouped-query attention")
        if query_offsets is None:
            if batch != requests:
                raise ValueError("Expected one query per CPU history row")
        else:
            _cpu(query_offsets, torch.int64)
            if query_offsets.shape != (requests + 1,):
                raise ValueError("Expected one query offset per request and a final end offset")
            query_offsets = query_offsets.contiguous()
        dtype = self.dtype if lse is None else torch.float32
        if out is None:
            out = torch.empty(q_cpu.shape, dtype=dtype, device="cpu")
        _cpu(out, dtype)
        if out.shape != q_cpu.shape or not out.is_contiguous():
            raise ValueError("CPU attention output must be contiguous with the shape of Q")
        if lse is not None:
            _cpu(lse, torch.float32)
            if lse.shape != q_cpu.shape[:2] or not lse.is_contiguous():
                raise ValueError("CPU attention LSE must be contiguous with shape [B,H]")
        if self.native_decode is None:
            if query_offsets is not None:
                offsets = query_offsets.tolist()
                if (offsets[0] != 0 or offsets[-1] != batch
                        or any(a > b for a, b in zip(offsets, offsets[1:]))):
                    raise ValueError("Invalid CPU attention query offsets")
                rows = torch.repeat_interleave(torch.arange(requests, device="cpu"),
                                               query_offsets.diff(), output_size=batch)
                slots_cpu = slots_cpu.index_select(0, rows)
                lengths_cpu = lengths_cpu.index_select(0, rows)
            self._reference(layer, q_cpu, slots_cpu, lengths_cpu, scale, out, lse)
            return out
        q_cpu = q_cpu.contiguous()
        if slots_cpu.stride(1) != 1:
            slots_cpu = slots_cpu.contiguous()
        lengths_cpu = lengths_cpu.contiguous()
        pointers = [q_cpu.data_ptr(), self.keys[layer].data_ptr(), self.values[layer].data_ptr(),
                    slots_cpu.data_ptr(), lengths_cpu.data_ptr()]
        if query_offsets is not None:
            pointers.append(query_offsets.data_ptr())
        pointers.append(out.data_ptr())
        call = self.native_decode
        if lse is not None:
            call = self.native_decode_state if query_offsets is None else self.native_grouped_state
            pointers.append(lse.data_ptr())
        sizes = [batch] if query_offsets is None else [requests, batch]
        result = call(
            *pointers, *sizes, q_heads, self.num_kv_heads, slots_cpu.stride(0), slots_cpu.shape[1],
            self.capacity, scale, self.num_threads,
        )
        if result:
            errors = {1: "unsupported GQA ratio", 2: "invalid history length", 3: "invalid KV slot"}
            raise ValueError(errors.get(result, f"CPU attention returned {result}"))
        return out

    def _reference(self, layer, q, slots, lengths, scale, out, lse):
        groups = q.shape[1] // self.num_kv_heads
        for b, length in enumerate(lengths.tolist()):
            if length == 0 and lse is not None:
                out[b].zero_()
                lse[b].fill_(-float("inf"))
                continue
            if not 0 < length <= slots.shape[1]:
                raise ValueError("invalid history length")
            live = slots[b, :length]
            keys = self.keys[layer].index_select(0, live).float()
            values = self.values[layer].index_select(0, live).float()
            query = q[b].float().reshape(self.num_kv_heads, groups, self.head_dim)
            scores = torch.einsum("hgd,thd->hgt", query, keys) * scale
            result = torch.einsum("hgt,thd->hgd", scores.softmax(-1), values)
            out[b].copy_(result.reshape(-1, self.head_dim))
            if lse is not None:
                lse[b].copy_(scores.logsumexp(-1).reshape(-1))
