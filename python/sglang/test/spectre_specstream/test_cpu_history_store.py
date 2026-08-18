import pytest

torch = pytest.importorskip("torch")

from sglang.srt.speculative.spectre.specstream.cpu_history_store import (  # noqa: E402
    CPUHistoryStore,
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
    ticket.wait_safe_to_free()
    chunks = list(store.iter_layer_chunks("r", 0, history_end=8))
    assert [chunk.length for chunk in chunks] == [4, 4]
    assert all(chunk.tensor.is_contiguous() for chunk in chunks)
    assert torch.equal(chunks[0].tensor[:, 0], _Pool().key[0][:4])
    store.release("r")
    assert store.bytes_used == store.bytes_reserved == 0
