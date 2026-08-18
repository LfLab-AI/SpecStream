import pytest

torch = pytest.importorskip("torch")

from sglang.srt.speculative.spectre.specstream.online_softmax import (  # noqa: E402
    finalize_online_softmax_state,
    init_online_softmax_state,
    update_online_softmax_state,
)
from sglang.srt.speculative.spectre.specstream.triton_stream_attn import (  # noqa: E402
    triton_fused_available,
    update_gpu_tail_state,
    update_packed_history_state,
)


@pytest.mark.skipif(not triton_fused_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("query_count", [1, 5])
def test_multi_query_tiled_kernel_matches_full_history(dtype, query_count):
    torch.manual_seed(17)
    device = torch.device("cuda")
    num_query_heads = 28
    num_kv_heads = 4
    head_dim = 128
    key_count = 513
    query = torch.randn(
        query_count,
        num_query_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    packed = torch.randn(
        key_count,
        2,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )

    candidate = init_online_softmax_state(query, num_kv_heads, head_dim)
    candidate, fused = update_packed_history_state(
        candidate,
        query,
        packed,
        require_fused_cuda=True,
    )
    assert fused
    candidate_output, candidate_lse = finalize_online_softmax_state(candidate)

    reference = init_online_softmax_state(query, num_kv_heads, head_dim)
    reference = update_online_softmax_state(
        reference,
        query,
        packed[:, 0],
        packed[:, 1],
        tuple(range(query_count)),
        tuple(range(key_count)),
        causal=False,
    )
    reference_output, reference_lse = finalize_online_softmax_state(reference)

    # Tensor-Core dot/accumulation order intentionally differs from the FP32
    # Torch oracle.  These bounds validate full-history semantics, not bitwise
    # identity.
    torch.testing.assert_close(candidate_output, reference_output, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(candidate_lse, reference_lse, rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not triton_fused_available(), reason="requires CUDA and Triton")
def test_tiled_history_and_indirect_tail_merge_matches_reference():
    torch.manual_seed(23)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    query_count, num_query_heads, num_kv_heads, head_dim = 5, 28, 4, 128
    history_count, tail_count = 257, 37
    query = torch.randn(
        query_count, num_query_heads, head_dim, dtype=dtype, device=device
    )
    history = torch.randn(
        history_count, 2, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    cache = torch.randn(
        2 * tail_count, 2, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    slots = torch.arange(0, 2 * tail_count, 2, dtype=torch.int64, device=device)

    candidate = init_online_softmax_state(query, num_kv_heads, head_dim)
    candidate, _ = update_packed_history_state(candidate, query, history)
    candidate, _ = update_gpu_tail_state(
        candidate,
        query,
        cache[:, 0],
        cache[:, 1],
        query_position_start=history_count + tail_count - query_count,
        key_position_start=history_count,
        token_indices=slots,
    )
    candidate_output, _ = finalize_online_softmax_state(candidate)

    reference = init_online_softmax_state(query, num_kv_heads, head_dim)
    all_key = torch.cat((history[:, 0], cache[slots, 0]), dim=0)
    all_value = torch.cat((history[:, 1], cache[slots, 1]), dim=0)
    reference = update_online_softmax_state(
        reference,
        query,
        all_key,
        all_value,
        tuple(
            range(
                history_count + tail_count - query_count,
                history_count + tail_count,
            )
        ),
        tuple(range(history_count + tail_count)),
        causal=True,
    )
    reference_output, _ = finalize_online_softmax_state(reference)
    torch.testing.assert_close(candidate_output, reference_output, rtol=3e-2, atol=3e-2)
