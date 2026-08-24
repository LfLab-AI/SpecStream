import pytest

from sglang.srt.speculative.spectre.specstream.state import TargetTieredKVState


def test_sealed_history_is_committed_and_chunk_aligned():
    state = TargetTieredKVState("r", committed_len=8192, logical_len=8192)
    state.start_seal()
    state.mark_sealed(4096, [1, 2])
    state.check(2048)
    assert state.stream_enabled
    assert state.history_len == state.tail_start == 4096


def test_rollback_cannot_cross_sealed_history():
    state = TargetTieredKVState(
        "r",
        committed_len=8192,
        history_len=4096,
        logical_len=8192,
        tail_start=4096,
        stream_enabled=True,
    )
    with pytest.raises(AssertionError, match="sealed CPU history"):
        state.finish_round(4095)


def test_only_one_seal_can_be_in_flight():
    state = TargetTieredKVState("r", committed_len=4096, logical_len=4096)
    state.start_seal()
    with pytest.raises(AssertionError, match="only one seal"):
        state.start_seal()
