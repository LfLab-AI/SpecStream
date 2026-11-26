"""Share rank-local Draft KV in DRAM between a scheduler and its MPS worker."""

import mmap
import os
import warnings
import weakref

import torch

from sglang.srt.speculative.spectre.drafter.cpu_history_attention import CpuHistoryAttention


def _tensor_views(mapping, shape, dtype, *, read_only):
    count = 1
    for size in shape:
        count *= size
    with warnings.catch_warnings():
        if read_only:
            # The child kernel only reads this mapping. PyTorch's frombuffer
            # wrapper cannot represent that restriction in the Tensor API.
            warnings.filterwarnings("ignore", message="The given buffer is not writable.*")
        backing = torch.frombuffer(mapping, dtype=dtype, count=2 * count)
    keys, values = backing.view(2, *shape).unbind(0)
    return keys.transpose(1, 2), values.transpose(1, 2)


class SharedCpuHistoryAttention(CpuHistoryAttention):
    """A memfd-backed native CPU attention pool with one scheduler writer."""

    def __init__(self, num_layers, capacity, num_kv_heads, head_dim=128,
                 dtype=torch.bfloat16, num_threads=None, tp_rank=None, *, fd=None):
        writer = fd is None
        try:
            capacity = int(capacity)
            if capacity < 1:
                raise ValueError("Shared Draft history capacity must be positive")
            # Initialize the existing affinity, kernel, and layout logic with a
            # one-slot placeholder. The full backing is supplied by the memfd.
            super().__init__(num_layers, 1, num_kv_heads, head_dim, dtype, num_threads, tp_rank)
            self.capacity = capacity
            self._shape = (self.num_layers, self.num_kv_heads, capacity, self.head_dim)
            nbytes = 2 * self.num_layers * self.num_kv_heads * capacity * self.head_dim * dtype.itemsize
            self._owner_fd = None
            if writer:
                fd = os.memfd_create("specstream-draft-history", os.MFD_CLOEXEC)
                # Sparse allocation keeps physical memory proportional to the
                # pages the scheduler actually uses and avoids /dev/shm quota.
                os.ftruncate(fd, nbytes)
            elif os.fstat(fd).st_size != nbytes:
                raise ValueError("Draft history descriptor disagrees with the mapping size")
            access = mmap.ACCESS_WRITE if writer else mmap.ACCESS_READ
            mapping = mmap.mmap(fd, nbytes, access=access)
            self.keys, self.values = _tensor_views(mapping, self._shape, dtype, read_only=not writer)
            # frombuffer's storage holds a reference to mmap. Never close the
            # mapping explicitly while tensor views could still be live.
            if writer:
                self._owner_fd = fd
                weakref.finalize(self, os.close, fd)
                fd = None
        finally:
            if fd is not None:
                os.close(fd)

    def export_descriptor(self):
        if self._owner_fd is None:
            raise RuntimeError("Only the scheduler can export shared CPU Draft history")
        from multiprocessing.reduction import DupFd

        return {
            "num_layers": self.num_layers,
            "capacity": self.capacity,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "dtype": self.dtype,
            "num_threads": self.num_threads,
            "fd_handle": DupFd(self._owner_fd),
        }

    @classmethod
    def import_descriptor(cls, descriptor, tp_rank):
        args = dict(descriptor)
        fd = args.pop("fd_handle").detach()
        return cls(**args, tp_rank=tp_rank, fd=fd)

    def store(self, *args, **kwargs):
        if self._owner_fd is None:
            raise RuntimeError("The MPS Draft process cannot write committed CPU history")
        return super().store(*args, **kwargs)


def create_shared_cpu_history(*args, **kwargs):
    """Factory compatible with the native Draft spill allocator."""
    return SharedCpuHistoryAttention(*args, **kwargs)
