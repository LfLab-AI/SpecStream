"""One physical GPU allocation shared by the native Target and Draft KV pools.

The caller owns stream synchronization, KV eviction and logical slot maps.
Before resizing, preserved GPU slots must occupy each pool's prefix 1..N.
Every captured graph referring to the previous views must be invalidated.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _Geometry:
    layers: int
    heads: int
    key_dim: int
    value_dim: int
    dtype: torch.dtype
    itemsize: int
    padding: int

    @classmethod
    def from_pool(cls, pool):
        dtype = pool.store_dtype
        if dtype != pool.dtype or dtype not in (
            torch.bfloat16, torch.float16, torch.float32
        ):
            raise ValueError("Joint KV arena requires an ordinary unquantized MHA pool")
        if pool.page_size != 1 or getattr(pool, "cpu_draft_history", None) is not None:
            raise ValueError("Joint KV arena requires page_size=1 GPU KV storage")
        geometry = cls(
            int(pool.layer_num), int(pool.head_num), int(pool.head_dim),
            int(getattr(pool, "v_head_dim", pool.head_dim)), dtype,
            dtype.itemsize, int(pool.page_size),
        )
        if min(geometry.layers, geometry.heads, geometry.key_dim, geometry.value_dim) < 1:
            raise ValueError("Joint KV arena dimensions must be positive")
        return geometry

    @property
    def bytes_per_token(self):
        return self.layers * self.heads * (self.key_dim + self.value_dim) * self.itemsize

    def buffer_bytes(self, slots, value=False):
        dim = self.value_dim if value else self.key_dim
        return (slots + self.padding) * self.heads * dim * self.itemsize

    def storage_bytes(self, slots):
        return (slots + self.padding) * self.bytes_per_token


@dataclass(frozen=True)
class _BufferRegion:
    role: str
    value: bool
    layer: int
    offset: int
    nbytes: int
    row_bytes: int


class NativeJointKVArena:
    """Reassign physical bytes from Target KV to Draft KV at a safe boundary."""

    def __init__(self, device, budget_bytes, scratch_bytes=8 * 1024**2):
        self.device = torch.device(device)
        self.budget_bytes = int(budget_bytes)
        self.scratch_bytes = int(scratch_bytes)
        if not 0 < self.scratch_bytes < self.budget_bytes:
            raise ValueError("Joint KV scratch must be positive and smaller than its budget")
        self.data_capacity_bytes = self.budget_bytes - self.scratch_bytes
        self.storage = torch.empty(self.budget_bytes, dtype=torch.uint8, device=self.device)
        self.device = self.storage.device
        self._scratch = self.storage[self.data_capacity_bytes:]
        self.pools = {}
        self._geometry = {}
        self._slots = {}
        self.resize_count = 0
        self.bytes_moved = 0

    @staticmethod
    def bytes_per_pool_token(pool):
        return _Geometry.from_pool(pool).bytes_per_token

    @staticmethod
    def bytes_for_pool(pool, slots):
        return _Geometry.from_pool(pool).storage_bytes(int(slots))

    def bytes_per_token(self, role):
        return self._geometry[role].bytes_per_token

    def _regions(self, sizes):
        regions = []
        offset = 0
        for role in ("target", "draft"):
            if role not in sizes:
                continue
            geometry = self._geometry[role]
            slots = sizes[role]
            for value in (False, True):
                dim = geometry.value_dim if value else geometry.key_dim
                for layer in range(geometry.layers):
                    nbytes = geometry.buffer_bytes(slots, value)
                    regions.append(_BufferRegion(
                        role, value, layer, offset, nbytes,
                        geometry.heads * dim * geometry.itemsize,
                    ))
                    offset += nbytes
        if offset > self.data_capacity_bytes:
            raise ValueError("Joint KV layout exceeds the physical data budget")
        return regions

    def _views(self, role, sizes, regions):
        geometry = self._geometry[role]
        keys, values = [], []
        for region in regions:
            if region.role != role:
                continue
            dim = geometry.value_dim if region.value else geometry.key_dim
            view = self.storage[region.offset:region.offset + region.nbytes].view(
                geometry.dtype
            ).view(sizes[role] + geometry.padding, geometry.heads, dim)
            (values if region.value else keys).append(view)
        return keys, values

    @staticmethod
    def _pointer_metadata(keys, values, device):
        k_ptrs = torch.tensor([tensor.data_ptr() for tensor in keys],
                              dtype=torch.uint64, device=device)
        v_ptrs = torch.tensor([tensor.data_ptr() for tensor in values],
                              dtype=torch.uint64, device=device)
        pointers = torch.cat((k_ptrs, v_ptrs))
        strides = torch.tensor(
            [tensor.shape[1] * tensor.shape[2] * tensor.element_size()
             for tensor in keys + values], dtype=torch.int64, device=device,
        )
        return k_ptrs, v_ptrs, pointers, strides

    def _binding(self, role, sizes, regions):
        keys, values = self._views(role, sizes, regions)
        metadata = self._pointer_metadata(keys, values, self.device)
        return keys, values, metadata

    def _publish(self, role, sizes, binding):
        pool = self.pools[role]
        pool.k_buffer, pool.v_buffer, metadata = binding
        pool.k_data_ptrs, pool.v_data_ptrs, pool.data_ptrs, pool.data_strides = metadata
        pool.size = sizes[role]
        pool.native_joint_kv_arena = self
        # The ordinary MHA metric uses GiB for the summed K/V allocation.
        pool.mem_usage = self._geometry[role].storage_bytes(sizes[role]) / 1024**3

    def bind_pool(self, role, pool, slots):
        """Replace a pool's initial buffer allocation with views of this arena.

        Invoke inside MHA._create_buffers, before standalone KV tensors exist.
        Target must bind first so the Draft segment has a defined start address.
        """
        if role not in ("target", "draft") or role in self.pools:
            raise ValueError("A joint KV role must be target or draft and bind once")
        if role == "draft" and "target" not in self.pools:
            raise ValueError("Bind Target before Draft")
        if hasattr(pool, "k_buffer") or hasattr(pool, "v_buffer"):
            raise ValueError("Bind joint storage before allocating standalone KV buffers")
        slots = int(slots)
        pool_device = torch.device(pool.device)
        if pool_device.type == "cuda" and pool_device.index is None:
            pool_device = torch.device("cuda", torch.cuda.current_device())
        if slots < 1 or pool_device != self.device:
            raise ValueError("Joint KV pool size or device disagrees with its arena")
        self._geometry[role] = _Geometry.from_pool(pool)
        sizes = dict(self._slots, **{role: slots})
        try:
            regions = self._regions(sizes)
            binding = self._binding(role, sizes, regions)
        except BaseException:
            del self._geometry[role]
            raise
        self.pools[role] = pool
        self._slots = sizes
        self._publish(role, sizes, binding)
        # Slot zero is the stable dummy destination used by padded kernels.
        for tensor in pool.k_buffer + pool.v_buffer:
            tensor[0].zero_()
        return self.describe()

    def _memmove(self, source, destination, length):
        if not length or source == destination:
            return 0, 0
        storage = self.storage
        if destination + length <= source or source + length <= destination:
            storage[destination:destination + length].copy_(storage[source:source + length])
            return length, 1
        operations = 0
        backwards = destination > source
        for offset in range(0, length, self.scratch_bytes):
            count = min(self.scratch_bytes, length - offset)
            begin = length - offset - count if backwards else offset
            self._scratch[:count].copy_(storage[source + begin:source + begin + count])
            storage[destination + begin:destination + begin + count].copy_(self._scratch[:count])
            operations += 2
        return length, operations

    def resize(self, target_slots, draft_slots, *, target_preserved, draft_preserved):
        """Move compact live prefixes, then publish resized physical KV views.

        Prefix lengths exclude dummy slot zero. All copies use the caller's
        current stream. Returning means copies were enqueued, so the caller
        must honor its synchronization boundary before using another stream.
        """
        if set(self.pools) != {"target", "draft"}:
            raise RuntimeError("Both joint KV pools must bind before resizing")
        sizes = {"target": int(target_slots), "draft": int(draft_slots)}
        preserved = {"target": int(target_preserved), "draft": int(draft_preserved)}
        if sizes["target"] > self._slots["target"] or sizes["draft"] < self._slots["draft"]:
            raise ValueError("Joint KV resize supports Target shrinking and Draft growth")
        for role in sizes:
            if sizes[role] < 1 or not 0 <= preserved[role] <= min(sizes[role], self._slots[role]):
                raise ValueError("Preserved KV prefix does not fit both pool layouts")
        old_regions = self._regions(self._slots)
        new_regions = self._regions(sizes)
        # Prepare views and metadata before modifying any live byte range.
        bindings = {role: self._binding(role, sizes, new_regions) for role in sizes}
        before = self.describe()
        moves = []
        for old, new in zip(old_regions, new_regions, strict=True):
            rows = preserved[old.role] + self._geometry[old.role].padding
            moves.append((old.offset, new.offset, rows * old.row_bytes))

        moved = operations = 0
        if all(destination <= source for source, destination, _ in moves):
            # The normal Target-to-Draft exchange shifts every live range left.
            for source, destination, length in moves:
                copied, calls = self._memmove(source, destination, length)
                moved += copied
                operations += calls
        else:
            # Rounding can let Draft also consume previously unused arena bytes.
            # Compact left first, then scatter right in reverse buffer order.
            compact = []
            offset = 0
            for source, destination, length in moves:
                copied, calls = self._memmove(source, offset, length)
                moved += copied
                operations += calls
                compact.append((offset, destination, length))
                offset += length
            for source, destination, length in reversed(compact):
                copied, calls = self._memmove(source, destination, length)
                moved += copied
                operations += calls

        for role in sizes:
            self._publish(role, sizes, bindings[role])
        self._slots = sizes
        self.resize_count += 1
        self.bytes_moved += moved
        after = self.describe()
        return dict(
            before=before, after=after, bytes_moved=moved, copy_ops=operations,
            old_target_slots=before["target_slots"], old_draft_slots=before["draft_slots"],
            target_slots=sizes["target"], draft_slots=sizes["draft"],
            arena_storage_ptr=self.storage.data_ptr(),
        )

    def describe(self):
        result = dict(
            budget_bytes=self.budget_bytes, scratch_bytes=self.scratch_bytes,
            data_capacity_bytes=self.data_capacity_bytes,
            allocated_storage_bytes=self.storage.nbytes,
            arena_storage_ptr=self.storage.data_ptr(), resize_count=self.resize_count,
            cumulative_bytes_moved=self.bytes_moved,
        )
        for role, pool in self.pools.items():
            result[f"{role}_slots"] = self._slots[role]
            result[f"{role}_bytes"] = sum(tensor.nbytes for tensor in pool.k_buffer + pool.v_buffer)
            result[f"{role}_first_key_ptr"] = pool.k_buffer[0].data_ptr()
            result[f"{role}_last_value_ptr"] = pool.v_buffer[-1].data_ptr()
        result["unused_data_bytes"] = self.data_capacity_bytes - sum(
            self._geometry[role].storage_bytes(slots) for role, slots in self._slots.items()
        )
        return result
