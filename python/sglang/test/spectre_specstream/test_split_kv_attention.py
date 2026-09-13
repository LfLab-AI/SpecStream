"""One GPU correctness matrix and an opt-in CUDA-event performance comparison.

Run correctness:
  pytest -q python/sglang/test/spectre_specstream/test_split_kv_attention.py
Run performance (same kernels, output includes medians and exact shapes):
  SPECSTREAM_ATTENTION_BENCH_JSON=/tmp/attention.json pytest -qs \
    python/sglang/test/spectre_specstream/test_split_kv_attention.py -k performance
"""

import json
import os
from pathlib import Path
import statistics

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.speculative.spectre.specstream.online_softmax import (
    finalize_online_softmax_state,
    init_online_softmax_state,
    update_online_softmax_state,
)
from sglang.srt.speculative.spectre.specstream.triton_stream_attn import (
    choose_split_kv_count,
    finalize_batched_online_softmax_state,
    init_batched_online_softmax_state,
    split_packed_history_cohort_state,
    triton_fused_available,
    update_gpu_paged_state_batched,
    update_gpu_tail_state,
    update_packed_history_cohort_batched,
    update_packed_history_state,
)


def test_split_selection_bounds_small_and_already_parallel_work():
    common = dict(query_count=4, num_query_heads=32, num_kv_heads=4, sm_count=108)
    assert choose_split_kv_count(batch_size=1, key_count=513, **common) == 1
    assert choose_split_kv_count(batch_size=1, key_count=8192, **common) > 1
    assert choose_split_kv_count(batch_size=128, key_count=8192, **common) == 1
    assert choose_split_kv_count(batch_size=1, key_count=1024, **common) <= 2


@pytest.mark.skipif(not triton_fused_available(), reason="requires CUDA and Triton")
def test_single_request_split_history_reuses_scratch_and_merges_each_chunk_once():
    torch.manual_seed(711)
    queries = torch.randn(4, 32, 128, device="cuda", dtype=torch.bfloat16)
    first = torch.randn(2053, 2, 4, 128, device="cuda", dtype=torch.bfloat16)
    second = torch.randn_like(first)
    candidate = init_online_softmax_state(queries, 4)
    candidate, _ = update_packed_history_state(candidate, queries, first, num_splits=8)
    scratch = tuple(id(t) for tensors in candidate.workspace.values() for t in tensors)
    candidate, _ = update_packed_history_state(candidate, queries, second, num_splits=8)
    assert tuple(id(t) for tensors in candidate.workspace.values() for t in tensors) == scratch
    assert scratch
    reference = init_online_softmax_state(queries, 4)
    both = torch.cat((first, second))
    reference = update_online_softmax_state(
        reference, queries, both[:, 0], both[:, 1], range(4), range(4106), causal=False)
    output, lse = finalize_online_softmax_state(candidate)
    ref_output, ref_lse = finalize_online_softmax_state(reference)
    torch.testing.assert_close(output, ref_output, rtol=2e-2, atol=8e-3)
    torch.testing.assert_close(lse, ref_lse, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_empty_paged_regions_finalize_to_zero_without_nan(device):
    if device == "cuda" and not triton_fused_available():
        pytest.skip("requires CUDA and Triton")
    dtype = torch.float16 if device == "cuda" else torch.float32
    query = torch.randn(2, 4, 8, 64, device=device, dtype=dtype)
    key = torch.randn(32, 2, 64, device=device, dtype=dtype)
    value = torch.randn_like(key)
    table = torch.arange(32, device=device).repeat(2, 1)
    # First item has zero keys; every key in the second is causally excluded.
    metadata = torch.tensor([[0, 0, 0, 0], [1, 8, 7, 0]], device=device)
    state = init_batched_online_softmax_state(query, 2)
    state, _ = update_gpu_paged_state_batched(
        state, query, key, value, table, metadata, max_key_count=7,
        require_fused_cuda=device == "cuda", num_splits=8)
    output, lse = finalize_batched_online_softmax_state(state)
    assert torch.count_nonzero(output).item() == 0
    assert torch.isneginf(lse).all().item()
    assert not torch.isnan(state.weighted_value).any().item()


@pytest.mark.skipif(not triton_fused_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("query_count", [1, 4, 8])
@pytest.mark.parametrize("batch", [1, 4, 8])
def test_split_history_paged_history_and_causal_tail(dtype, query_count, batch):
    """Mixed lengths, empty splits, GQA, non-contiguous cache and live state merge."""
    torch.manual_seed(709)
    heads, kvheads, dim = 32, 4, 128
    gpu_lengths = [257, 129, 0, 65] * 2
    cpu_lengths = [2053, 1025, 0, 511] * 2
    tail_lengths = [query_count + 17, query_count + 5, query_count, query_count + 3] * 2
    gpu_lengths, cpu_lengths, tail_lengths = (
        lengths[:batch] for lengths in (gpu_lengths, cpu_lengths, tail_lengths))
    queries = torch.randn(batch, query_count, heads, dim, device="cuda", dtype=dtype)
    packed = torch.randn(batch, 2053, 2, kvheads, dim, device="cuda", dtype=dtype)
    valid = torch.tensor(cpu_lengths, device="cuda", dtype=torch.int32)
    table_width = 4096
    # Interleaved K/V also tests independent non-contiguous token strides.
    cache = torch.randn(batch * 512, 2, kvheads, dim, device="cuda", dtype=dtype)
    table = torch.zeros(batch + 1, table_width, device="cuda", dtype=torch.int64)
    history_meta, tail_meta, slots_per_item = [], [], []
    for item in range(batch):
        hg, hc, tail = gpu_lengths[item], cpu_lengths[item], tail_lengths[item]
        slots = torch.randperm(512, device="cuda")[:hg + tail] + item * 512
        slots_per_item.append(slots)
        req_row = batch - item  # Metadata rows deliberately differ from batch order.
        table[req_row, :hg] = slots[:hg]
        table[req_row, hg + hc:hg + hc + tail] = slots[hg:]
        qstart = hg + hc + tail - query_count
        history_meta.append([req_row, 0, hg, qstart])
        tail_meta.append([req_row, hg + hc, tail, qstart])
    history_meta = torch.tensor(history_meta, device="cuda", dtype=torch.int32)
    tail_meta = torch.tensor(tail_meta, device="cuda", dtype=torch.int64)
    state = init_batched_online_softmax_state(queries, kvheads)
    state, fused = update_gpu_paged_state_batched(
        state, queries, cache[:, 0], cache[:, 1], table, history_meta,
        max_key_count=max(gpu_lengths), num_splits=4)
    assert fused
    # Force more splits than the shortest regions need. Empty partials must
    # leave a previously accumulated GPU History state unchanged.
    state, fused = update_packed_history_cohort_batched(
        state, queries, packed, valid, num_splits=8)
    assert fused
    state, fused = update_gpu_paged_state_batched(
        state, queries, cache[:, 0], cache[:, 1], table, tail_meta,
        max_key_count=max(tail_lengths), num_splits=1)
    assert fused
    output, lse = finalize_batched_online_softmax_state(state)
    cast_output, _ = finalize_batched_online_softmax_state(state, output_dtype=dtype)
    torch.testing.assert_close(cast_output.float(), output, rtol=1e-2, atol=1e-3)
    for item in range(batch):
        hg, hc, tail = gpu_lengths[item], cpu_lengths[item], tail_lengths[item]
        slots = slots_per_item[item]
        keys = torch.cat((cache[slots[:hg], 0], packed[item, :hc, 0],
                          cache[slots[hg:], 0]))
        values = torch.cat((cache[slots[:hg], 1], packed[item, :hc, 1],
                            cache[slots[hg:], 1]))
        total = hg + hc + tail
        ref = update_online_softmax_state(
            init_online_softmax_state(queries[item], kvheads), queries[item], keys,
            values, range(total - query_count, total), range(total), causal=True)
        ref_output, ref_lse = finalize_online_softmax_state(ref)
        torch.testing.assert_close(output[item], ref_output, rtol=2e-2, atol=8e-3)
        torch.testing.assert_close(lse[item], ref_lse, rtol=3e-3, atol=3e-3)


def _event_median_ms(fn, warmup=10, repeats=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


@pytest.mark.skipif(not triton_fused_available(), reason="requires CUDA and Triton")
@pytest.mark.skipif(not os.environ.get("SPECSTREAM_ATTENTION_BENCH_JSON"),
                    reason="opt-in GPU performance test")
def test_split_kv_and_paged_batch_performance():
    torch.manual_seed(710)
    results = []
    for batch in (1, 4, 8):
        queries = torch.randn(batch, 4, 32, 128, device="cuda", dtype=torch.bfloat16)
        packed = torch.randn(batch, 8192, 2, 4, 128, device="cuda", dtype=torch.bfloat16)
        valid = torch.full((batch,), 8192, device="cuda", dtype=torch.int32)
        old = init_batched_online_softmax_state(queries, 4)
        new = init_batched_online_softmax_state(queries, 4)
        old_ms = _event_median_ms(lambda: update_packed_history_cohort_batched(
            old, queries, packed, valid, num_splits=1))
        new_ms = _event_median_ms(lambda: update_packed_history_cohort_batched(
            new, queries, packed, valid))
        results.append(dict(kind="packed_history", batch=batch, q=4, tokens=8192,
                            baseline_ms=old_ms, optimized_ms=new_ms,
                            speedup=old_ms / new_ms))
        del packed
        cache = torch.randn(batch * 2048, 2, 4, 128, device="cuda", dtype=torch.bfloat16)
        table = torch.arange(batch * 2048, device="cuda").view(batch, 2048)
        metadata = torch.tensor([[i, 0, 2048, 2044] for i in range(batch)], device="cuda")
        old_states = split_packed_history_cohort_state(old)
        def sequential_tail():
            for i in range(batch):
                update_gpu_tail_state(old_states[i], queries[i], cache[:, 0], cache[:, 1],
                                      token_indices=table[i], query_position_start=2044,
                                      key_position_start=0)
        old_ms = _event_median_ms(sequential_tail)
        new_ms = _event_median_ms(lambda: update_gpu_paged_state_batched(
            new, queries, cache[:, 0], cache[:, 1], table, metadata,
            max_key_count=2048))
        results.append(dict(kind="paged_tail", batch=batch, q=4, tokens=2048,
                            baseline_ms=old_ms, optimized_ms=new_ms,
                            speedup=old_ms / new_ms))
    report = dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                  dtype="bfloat16", repetitions=25, timing="CUDA event median", cases=results)
    destination = Path(os.environ["SPECSTREAM_ATTENTION_BENCH_JSON"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    assert all(case["optimized_ms"] > 0 for case in results)
