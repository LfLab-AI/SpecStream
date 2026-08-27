import pytest

from sglang.srt.speculative.spectre.specstream.gpu_grant import (
    DraftExecutionGrant,
    DraftGrantTable,
)


def _grant(epoch=1, *, spec_cnt=2, deadline_us=None):
    return DraftExecutionGrant(
        request_id="r1",
        spec_cnt=spec_cnt,
        grant_epoch=epoch,
        grant_tokens=1,
        tpc_low=0,
        tpc_high=4,
        deadline_us=deadline_us,
    )


def test_one_token_quantum_is_enforced():
    with pytest.raises(ValueError, match="exactly one token"):
        DraftExecutionGrant("r1", 0, 1, 2, 0, 4)


def test_grant_is_consumable_exactly_once():
    table = DraftGrantTable()
    assert table.apply(_grant()).accepted
    assert table.consume_one("r1", spec_cnt=2, grant_epoch=1)
    assert not table.consume_one("r1", spec_cnt=2, grant_epoch=1)
    assert table.active("r1", spec_cnt=2) is None


def test_stale_epoch_and_spec_count_are_rejected():
    table = DraftGrantTable()
    assert table.apply(_grant(4, spec_cnt=3)).accepted
    assert table.apply(_grant(3, spec_cnt=3)).reason == "stale_epoch"
    assert table.apply(_grant(5, spec_cnt=2)).reason == "stale_spec_cnt"


def test_expired_grant_fails_closed():
    table = DraftGrantTable()
    table.apply(_grant(deadline_us=100))
    assert table.active("r1", spec_cnt=2, now_us=100) is None
    expired = table.pop_expired("r1", spec_cnt=2, now_us=100)
    assert expired is not None
    assert expired.grant_epoch == 1
    assert table.pop_expired("r1", spec_cnt=2, now_us=100) is None


def test_launched_grant_can_finish_after_its_deadline():
    table = DraftGrantTable()
    table.apply(_grant(deadline_us=100))
    assert table.active("r1", spec_cnt=2, now_us=99) is not None
    # Once launch was authorized, post-launch consumption must not re-check
    # wall time and turn a completed CUDA step into a scheduler failure.
    assert table.consume_one("r1", spec_cnt=2, grant_epoch=1)
