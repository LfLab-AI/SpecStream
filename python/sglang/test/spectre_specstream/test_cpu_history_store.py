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
    store.release("r")
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
