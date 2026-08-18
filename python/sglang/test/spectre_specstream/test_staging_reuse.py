import pytest

torch = pytest.importorskip("torch")

from sglang.srt.speculative.spectre.specstream.staging_runtime import (  # noqa: E402
    StagingWindowPool,
)


def test_bounded_staging_reuses_slot_allocation_on_cpu():
    staging = StagingWindowPool(2, "cpu")
    first = staging.submit(torch.ones(8, 2), 0)
    staging.mark_consumed(first)
    storage = first.tensor.untyped_storage().data_ptr()
    second = staging.submit(torch.zeros(4, 2), 0)
    assert second.tensor.untyped_storage().data_ptr() == storage
    assert (
        staging.allocated_bytes
        == 8 * 2 * torch.ones((), dtype=torch.float32).element_size()
    )


def test_grouped_submit_concatenates_without_extra_staging_slots():
    staging = StagingWindowPool(2, "cpu")
    sources = [
        torch.full((3, 2, 2), 1.0),
        torch.full((2, 2, 2), 2.0),
        torch.full((1, 2, 2), 3.0),
    ]
    transfer = staging.submit_many(sources, 1)
    actual = staging.wait_ready(transfer)
    expected = torch.cat(sources, dim=0)
    torch.testing.assert_close(actual, expected)
    assert transfer.source_count == 3
    assert transfer.nbytes == sum(source.nbytes for source in sources)
    assert staging.allocated_bytes == expected.nbytes


def test_reserve_avoids_first_submit_reallocation():
    staging = StagingWindowPool(2, "cpu")
    staging.reserve((16, 2), torch.float16)
    pointers = [tensor.untyped_storage().data_ptr() for tensor in staging._buffers]
    transfers = [
        staging.submit(torch.ones(4, 2, dtype=torch.float16), slot) for slot in range(2)
    ]
    assert [
        transfer.tensor.untyped_storage().data_ptr() for transfer in transfers
    ] == pointers
