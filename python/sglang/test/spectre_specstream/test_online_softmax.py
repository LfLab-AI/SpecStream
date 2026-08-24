import pytest

torch = pytest.importorskip("torch")

from sglang.srt.speculative.spectre.specstream.online_softmax import (  # noqa: E402
    finalize_online_softmax_state,
    init_online_softmax_state,
    update_online_softmax_state,
)


@pytest.mark.parametrize("q_len", [1, 4])
def test_chunked_online_softmax_matches_full_attention(q_len):
    torch.manual_seed(7)
    q = torch.randn(q_len, 4, 16)
    k = torch.randn(37, 2, 16)
    v = torch.randn(37, 2, 16)
    state = init_online_softmax_state(q, 2, 16)
    for begin in range(0, 37, 11):
        state = update_online_softmax_state(
            state,
            q,
            k[begin : begin + 11],
            v[begin : begin + 11],
            range(q_len),
            range(begin, min(begin + 11, 37)),
            scale=16**-0.5,
            causal=False,
        )
    actual, _ = finalize_online_softmax_state(state)
    k_gqa = k.repeat_interleave(2, dim=1)
    v_gqa = v.repeat_interleave(2, dim=1)
    expected = torch.einsum(
        "qhd,khd->qhk",
        q.float(),
        k_gqa.float(),
    ) * (16**-0.5)
    expected = torch.einsum("qhk,khd->qhd", expected.softmax(-1), v_gqa.float())
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
