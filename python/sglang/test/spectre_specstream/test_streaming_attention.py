import pytest

torch = pytest.importorskip("torch")

from sglang.srt.speculative.spectre.specstream.online_softmax import (  # noqa: E402
    full_attention_reference,
    online_attention_reference,
)


def test_causal_absolute_positions_across_history_and_tail():
    torch.manual_seed(11)
    q = torch.randn(4, 4, 8)
    k = torch.randn(21, 2, 8)
    v = torch.randn(21, 2, 8)
    q_pos = range(17, 21)
    k_pos = range(21)
    actual, _ = online_attention_reference(
        q, k, v, q_pos, k_pos, chunk_size=7, causal=True
    )
    expected, _ = full_attention_reference(q, k, v, q_pos, k_pos, causal=True)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
