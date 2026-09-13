import pytest

torch = pytest.importorskip("torch")

from sglang.srt.speculative.spectre.specstream.cpu_history_store import (  # noqa: E402
    CPUHistoryStore,
    SealTicket,
)


class _Pool:
    def __init__(self):
        self.key = {0: torch.arange(12 * 2 * 4).view(12, 2, 4).float()}
        self.value = {0: self.key[0] + 1000}

    def get_key_buffer(self, layer):
        return self.key[layer]

    def get_value_buffer(self, layer):
        return self.value[layer]


def test_layer_major_chunks_are_contiguous_and_releasable():
    store = CPUHistoryStore(
        max_memory_bytes=1 << 20,
        chunk_tokens=4,
        layer_ids=(0,),
        kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    ticket = store.seal_slots_async(
        rid="r", abs_start=0, slots=torch.arange(8), token_to_kv_pool=_Pool()
    )
    assert ticket.is_ready()
    assert store.complete_seal(ticket)
    chunks = list(store.iter_layer_chunks("r", 0, history_end=8))
    assert [chunk.length for chunk in chunks] == [4, 4]
    assert all(chunk.tensor.is_contiguous() for chunk in chunks)
    assert torch.equal(chunks[0].tensor[:, 0], _Pool().key[0][:4])
    suffix = list(
        store.iter_layer_chunks("r", 0, history_start=2, history_end=7)
    )
    assert [(chunk.abs_start, chunk.length) for chunk in suffix] == [(2, 2), (4, 3)]
    assert torch.equal(
        torch.cat([chunk.tensor[:, 0] for chunk in suffix]), _Pool().key[0][2:7]
    )
    store.release("r")
    assert store.bytes_used == store.bytes_reserved == 0


def test_failed_seal_rolls_back_new_cpu_slabs():
    class _FailingPool(_Pool):
        def get_key_buffer(self, layer):
            if layer == 1:
                raise RuntimeError("injected gather failure")
            return super().get_key_buffer(layer)

        def get_value_buffer(self, layer):
            if layer == 1:
                raise RuntimeError("injected gather failure")
            return super().get_value_buffer(layer)

    pool = _FailingPool()
    pool.key[1] = pool.key[0]
    pool.value[1] = pool.value[0]
    store = CPUHistoryStore(
        max_memory_bytes=1 << 20,
        chunk_tokens=4,
        layer_ids=(0, 1),
        kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="injected gather failure"):
        store.seal_slots_async(
            rid="r", abs_start=0, slots=torch.arange(4), token_to_kv_pool=pool
        )

    assert store.request_block_ids("r") == []
    assert store.bytes_used == store.bytes_reserved == 0


class _Event:
    def __init__(self):
        self.ready = False
        self.synchronized = False

    def query(self):
        return self.ready

    def synchronize(self):
        self.synchronized = True
        self.ready = True


def test_seal_ticket_poll_does_not_synchronize():
    event = _Event()
    source = torch.tensor([1])
    ticket = SealTicket([1], event, [source])
    assert not ticket.is_ready()
    assert not event.synchronized

    event.ready = True
    assert ticket.is_ready()
    ticket.wait_safe_to_free()
    assert ticket.completed
    assert not ticket.pending_sources


def test_grouped_backing_keeps_logical_chunks_and_layer_contiguity():
    from sglang.srt.speculative.spectre.specstream.staging_runtime import StagingWindowPool

    pool = _Pool()
    pool.key[1], pool.value[1] = pool.key[0] + 2000, pool.value[0] + 2000
    store = CPUHistoryStore(
        max_memory_bytes=1 << 20, chunk_tokens=4, allocation_group_chunks=3,
        layer_ids=(0, 1), kv_heads=2, head_dim=4, dtype=torch.float32, device="cpu",
    )
    ticket = store.seal_slots_async(
        rid="r", abs_start=0, slots=torch.arange(12), token_to_kv_pool=pool,
    )
    store.complete_seal(ticket)
    assert len(ticket.block_ids) == 3
    for layer in (0, 1):
        chunks = list(store.iter_layer_chunks("r", layer, history_start=2, history_end=11))
        assert [(c.abs_start, c.length) for c in chunks] == [(2, 2), (4, 4), (8, 3)]
        assert len({c.tensor.untyped_storage().data_ptr() for c in chunks}) == 1
        transfer = StagingWindowPool(1, "cpu").submit_many([c.tensor for c in chunks], 0)
        assert transfer.source_count == 3
        assert transfer.dma_count == 1
        torch.testing.assert_close(transfer.tensor[:, 0], pool.key[layer][2:11])
        torch.testing.assert_close(transfer.tensor[:, 1], pool.value[layer][2:11])
    assert store.bytes_used == store.bytes_reserved == 12 * 2 * 2 * 2 * 4 * 4
    store.release("r")
    assert store.bytes_used == store.bytes_reserved == 0


def test_group_allocation_failure_preserves_existing_request_and_accounting():
    slab_bytes = 4 * 2 * 2 * 4 * 4
    store = CPUHistoryStore(
        max_memory_bytes=3 * slab_bytes, chunk_tokens=4, allocation_group_chunks=2,
        layer_ids=(0,), kv_heads=2, head_dim=4, dtype=torch.float32, device="cpu",
    )
    existing = store.seal_slots_async(
        rid="existing", abs_start=0, slots=torch.arange(4), token_to_kv_pool=_Pool(),
    )
    with pytest.raises(MemoryError, match="budget exceeded"):
        store.seal_slots_async(
            rid="new", abs_start=0, slots=torch.arange(12), token_to_kv_pool=_Pool(),
        )
    assert store.request_block_ids("new") == []
    assert store.request_block_ids("existing") == existing.block_ids
    assert store.bytes_reserved == store.bytes_used == slab_bytes
    store.clear()
    assert store.bytes_reserved == store.bytes_used == 0
