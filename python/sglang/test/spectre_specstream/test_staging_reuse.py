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


def test_cohort_group_submit_uses_one_packed_window_and_reuses_host_storage():
    staging = StagingWindowPool(2, "cpu")
    staging.reserve((2, 8, 2, 2), torch.float32)
    staging.reserve_cohort_pack((2, 8, 2, 2), torch.float32)
    first_groups = [
        [torch.full((3, 2, 2), 1.0), torch.full((2, 2, 2), 2.0)],
        [torch.full((4, 2, 2), 3.0)],
    ]
    first = staging.submit_cohort_groups(first_groups, 0)
    packed = staging.wait_ready(first)
    assert packed.shape == (2, 5, 2, 2)
    assert first.source_count == 3
    assert first.valid_lengths == (5, 4)
    torch.testing.assert_close(packed[0, :3], first_groups[0][0])
    torch.testing.assert_close(packed[0, 3:5], first_groups[0][1])
    torch.testing.assert_close(packed[1, :4], first_groups[1][0])
    torch.testing.assert_close(
        first.valid_tokens, torch.tensor([5, 4], dtype=torch.int32)
    )

    host_ptr = staging._host_buffers[0].untyped_storage().data_ptr()
    staging.mark_consumed(first)
    second = staging.submit_cohort_groups([[torch.zeros((2, 2, 2))], []], 0)
    assert staging._host_buffers[0].untyped_storage().data_ptr() == host_ptr
    assert second.tensor.shape == (2, 2, 2, 2)
    assert second.valid_lengths == (2, 0)
    assert staging.allocated_host_bytes == 2 * 2 * 8 * 2 * 2 * 4


def test_direct_async_cohort_prefetch_skips_large_host_pack_buffer():
    staging = StagingWindowPool(2, "cpu")
    staging.reserve((2, 8, 2, 2), torch.float32)
    groups = [
        [torch.full((3, 2, 2), 1.0), torch.full((2, 2, 2), 2.0)],
        [torch.full((4, 2, 2), 3.0)],
    ]

    transfer = staging.submit_cohort_groups_direct_async(groups, 0)
    actual = staging.wait_ready(transfer)

    assert staging._host_buffers[0] is None
    assert transfer.source_count == 3
    assert transfer.valid_lengths == (5, 4)
    assert (
        transfer.nbytes
        == sum(source.nbytes for group in groups for source in group)
        + transfer.valid_tokens.nbytes
    )
    torch.testing.assert_close(actual[0, :3], groups[0][0])
    torch.testing.assert_close(actual[0, 3:5], groups[0][1])
    torch.testing.assert_close(actual[1, :4], groups[1][0])
    torch.testing.assert_close(
        transfer.valid_tokens, torch.tensor([5, 4], dtype=torch.int32)
    )


def test_h2d_calibration_is_disabled_for_cpu_staging():
    staging = StagingWindowPool(2, "cpu")
    staging.reserve((8, 2, 1, 2), torch.float32)

    assert staging.calibrate_h2d_gbps() == 0.0


def test_adjacent_history_views_coalesce_without_copying_padding():
    staging = StagingWindowPool(2, "cpu")
    storage = torch.arange(12 * 4).reshape(12, 2, 2).float()
    groups = [[storage[:3], storage[3:8]], [storage[8:11]]]
    transfer = staging.submit_cohort_groups_direct_async(groups, 0)
    assert transfer.source_count == 3
    assert transfer.dma_count == 3  # two contiguous KV regions + metadata
    assert transfer.source_nbytes == storage[:11].nbytes
    assert transfer.padding_nbytes == 0
    assert transfer.nbytes == storage[:11].nbytes + 8
    torch.testing.assert_close(transfer.tensor[0, :8], storage[:8])
    torch.testing.assert_close(transfer.tensor[1, :3], storage[8:11])


def test_packed_ragged_cohort_does_not_transfer_masked_padding():
    staging = StagingWindowPool(2, "cpu")
    groups = [[torch.ones(8, 2, 2)], [torch.ones(2, 2, 2)], []]
    transfer = staging.submit_cohort_groups(groups, 0)
    assert transfer.nbytes == 10 * 2 * 2 * 4 + 3 * 4
    assert transfer.nbytes < transfer.tensor.nbytes
    assert transfer.dma_count == 3
    assert transfer.padding_nbytes == 0


def test_immutable_metadata_reuse_survives_slot_reuse_and_bounded_eviction():
    staging = StagingWindowPool(2, "cpu", metadata_cache_entries=2)
    first = staging.submit_cohort_groups_direct_async(
        [[torch.ones(4, 2)], [torch.ones(3, 2)]], 0
    )
    staging.mark_consumed(first)
    same = staging.submit_cohort_groups_direct_async(
        [[torch.ones(4, 2)], [torch.ones(3, 2)]], 1
    )
    assert same.metadata_cache_hit
    assert same.valid_tokens is first.valid_tokens
    assert same.host_wait_ms == 0
    assert same.nbytes == same.source_nbytes
    for length in (2, 1, 4):
        staging.submit_cohort_groups_direct_async([[torch.ones(length, 2)], []], 0)
    assert len(staging._metadata_cache) == 2
    # Older transfer references retain immutable values after eviction.
    torch.testing.assert_close(first.valid_tokens, torch.tensor([4, 3], dtype=torch.int32))


def test_metadata_eviction_waits_only_for_evicted_host_upload():
    staging = StagingWindowPool(2, "cpu", metadata_cache_entries=1)
    first, _, _ = staging._immutable_valid_lengths((4, 3))

    class PendingUpload:
        waited = False

        def query(self):
            return False

        def synchronize(self):
            self.waited = True

    upload = PendingUpload()
    first.ready_event = upload
    hit, cached, wait_ms = staging._immutable_valid_lengths((4, 3))
    assert hit is first and cached and wait_ms == 0
    assert not upload.waited
    staging._immutable_valid_lengths((3, 2))
    assert upload.waited


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_direct_metadata_and_gpu_slot_lifetimes_under_queued_consumers():
    staging = StagingWindowPool(2, "cuda", metadata_cache_entries=2)
    staging.reserve((2, 16, 2, 2), torch.float32)
    snapshots = []
    sources = []
    for step in range(24):
        first_len, second_len = 12 + step % 3, 3 + step % 5
        source = torch.full((first_len + second_len, 2, 2), float(step), pin_memory=True)
        sources.append(source)
        transfer = staging.submit_cohort_groups_direct_async(
            [[source[:first_len]], [source[first_len:]]], step % 2
        )
        packed = staging.wait_ready(transfer)
        # Delay consumers enough to exercise allocator/cache eviction while
        # previous uses of immutable device metadata remain queued.
        torch.cuda._sleep(100_000)
        snapshots.append((
            packed[0, :first_len].clone(), packed[1, :second_len].clone(),
            transfer.valid_tokens.clone(), first_len, second_len, step,
        ))
        staging.mark_consumed(transfer)
    torch.cuda.synchronize()
    for first, second, lengths, nfirst, nsecond, step in snapshots:
        torch.testing.assert_close(first.cpu(), torch.full((nfirst, 2, 2), float(step)))
        torch.testing.assert_close(second.cpu(), torch.full((nsecond, 2, 2), float(step)))
        torch.testing.assert_close(lengths.cpu(), torch.tensor([nfirst, nsecond], dtype=torch.int32))
