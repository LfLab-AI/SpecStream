"""Exercise real multi-layer H2D staging and verifier dispatch against full KV."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.speculative.spectre.specstream.config import SpecStreamConfig
from sglang.srt.speculative.spectre.specstream.cpu_history_store import CPUHistoryStore
from sglang.srt.speculative.spectre.specstream.diagnostics import SpecStreamDiagnostics
from sglang.srt.speculative.spectre.specstream.online_softmax import full_attention_reference
from sglang.srt.speculative.spectre.specstream.round_meta import (
    SpecStreamRequestMeta, SpecStreamRoundMeta,
)
from sglang.srt.speculative.spectre.specstream.staging_runtime import StagingWindowPool
from sglang.srt.speculative.spectre.specstream.triton_stream_attn import triton_fused_available
from sglang.srt.speculative.spectre.specstream.verifier import SpecStreamVerifier


@pytest.mark.skipif(not triton_fused_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("cohort,buffers,q_len,softcap", [
    (True, 2, 4, 0.0), (True, 4, 8, 0.0),
    (False, 2, 4, 0.0), (True, 2, 4, 5.0),
])
def test_verifier_layers_ragged_history_prefetch_cancel_and_round_reorder(
    cohort, buffers, q_len, softcap, tmp_path,
):
    torch.manual_seed(712)
    batch, hq, hkv, dim, table_width = 4, 32, 4, 128, 3072
    layer_ids = (0, 1, 2)
    device, dtype = torch.device("cuda"), torch.bfloat16
    caches = [torch.randn(batch * table_width, 2, hkv, dim, device=device, dtype=dtype)
              for _ in layer_ids]
    pool = SimpleNamespace(
        get_kv_buffer=lambda layer: (caches[layer][:, 0], caches[layer][:, 1]),
        get_key_buffer=lambda layer: caches[layer][:, 0],
        get_value_buffer=lambda layer: caches[layer][:, 1],
    )
    page_table = torch.stack([
        torch.randperm(table_width, device=device) + i * table_width
        for i in range(batch)])
    history_lengths, gpu_lengths = (2304, 1024, 512, 0), (512, 256, 512, 0)
    tail_lengths = (q_len + 17, q_len + 33, q_len + 3, q_len + 257)
    store = CPUHistoryStore(
        max_memory_bytes=128 * 1024 ** 2, chunk_tokens=256,
        layer_ids=layer_ids, kv_heads=hkv, head_dim=dim, dtype=dtype, device=device,
        allocation_group_chunks=2)
    for i, count in enumerate(history_lengths):
        ticket = store.seal_slots_async(
            rid=f"r{i}", abs_start=0, slots=page_table[i, :count], token_to_kv_pool=pool)
        assert store.complete_seal(ticket, wait=True)
    staging = StagingWindowPool(buffers, device)
    config = SpecStreamConfig(
        enabled=True, reference_attention=False, chunk_tokens=256,
        num_buffers=buffers, chunks_per_transfer=2, layer_prefetch=True,
        cohort_enabled=cohort, max_cohort_size=2)
    verifier = SpecStreamVerifier(
        config=config, history_store=store, staging=staging,
        profiler=SimpleNamespace(record_h2d=lambda *a, **k: None,
                                 record_attention=lambda *a, **k: None),
        diagnostics=SpecStreamDiagnostics(str(tmp_path / "profile.csv"), False))
    forward_batch = SimpleNamespace(
        token_to_kv_pool=pool, req_to_token_pool=SimpleNamespace(req_to_token=page_table))

    try:
        for round_id, request_order in ((1, (0, 1, 2, 3)), (2, (3, 2, 1, 0))):
            items = tuple(SpecStreamRequestMeta(
                rid=f"r{rid}", req_pool_idx=rid,
                query_begin=index * q_len, query_end=(index + 1) * q_len,
                committed_len=history_lengths[rid] + tail_lengths[rid] - q_len,
                history_len=history_lengths[rid], gpu_history_len=gpu_lengths[rid],
                logical_len=history_lengths[rid] + tail_lengths[rid],
                stream_enabled=history_lengths[rid] > 0,
            ) for index, rid in enumerate(request_order))
            meta = SpecStreamRoundMeta(
                round_id=round_id, q_len=q_len, mode="ordinary", items=items,
                enabled=True, cohort_enabled=cohort)
            for layer_id in layer_ids:
                query = torch.randn(batch * q_len, hq, dim, device=device, dtype=dtype)
                layer = SimpleNamespace(
                    layer_id=layer_id, tp_q_head_num=hq, head_dim=dim,
                    scaling=dim ** -0.5, logit_cap=softcap,
                    is_cross_attention=False, sliding_window_size=-1)
                candidate = verifier.forward(
                    q=query, k_new=None, v_new=None, forward_batch=forward_batch,
                    meta=meta, layer=layer).view(batch, q_len, hq, dim)
                for index, item in enumerate(items):
                    slots = page_table[item.req_pool_idx, :item.logical_len]
                    reference, _ = full_attention_reference(
                        query[item.query_begin:item.query_end],
                        caches[layer_id][slots, 0], caches[layer_id][slots, 1],
                        range(item.committed_len, item.logical_len),
                        range(item.logical_len), scale=layer.scaling,
                        causal=True, softcap=softcap)
                    torch.testing.assert_close(
                        candidate[index].float(), reference, rtol=3e-2, atol=1e-2)
                if layer_id == 0:
                    # Cancellation invalidates the shared next-layer queue;
                    # the next forward must prime it again without stale views.
                    verifier.discard_layer_prefetch("r1")
            assert not verifier._single_layer_prefetch
            assert not verifier._batched_layer_prefetch
        torch.cuda.synchronize()
    finally:
        verifier.discard_layer_prefetch()
        torch.cuda.synchronize()
        store.clear()
