import time

from sglang.srt.speculative.spectre.specstream.draft_grant_runtime import (
    DraftExecutionGrant,
    DraftGrantGate,
)


def test_no_grant_no_forward_permission():
    gate = DraftGrantGate(total_tpcs=54)
    assert gate.try_acquire(["r0"]) is None


def test_calibration_bootstrap_before_first_forward():
    gate = DraftGrantGate(total_tpcs=54)
    gate.install_calibration_grant(4)
    g = gate.try_acquire(["r0"])
    assert g is not None
    assert (g.tpc_low, g.tpc_high) == (0, 4)
    gate.complete_one_step()
    assert gate.try_acquire(["r0"]) is not None


def test_online_one_token_consumed():
    gate = DraftGrantGate(total_tpcs=54)
    assert gate.install_online_grant(
        DraftExecutionGrant(
            epoch=1,
            tpc_low=0,
            tpc_high=4,
            token_budget=1,
            request_ids=("r0",),
        )
    )
    assert gate.try_acquire(["r0"]) is not None
    gate.complete_one_step()
    assert gate.try_acquire(["r0"]) is None


def test_stale_epoch_rejected():
    gate = DraftGrantGate(total_tpcs=54)
    assert gate.install_online_grant(
        DraftExecutionGrant(epoch=2, tpc_low=0, tpc_high=4)
    )
    assert gate.try_acquire() is not None
    gate.complete_one_step()
    assert not gate.install_online_grant(
        DraftExecutionGrant(epoch=2, tpc_low=0, tpc_high=4)
    )


def test_expired_grant_rejected():
    gate = DraftGrantGate(total_tpcs=54)
    assert not gate.install_online_grant(
        DraftExecutionGrant(
            epoch=1,
            tpc_low=0,
            tpc_high=4,
            deadline_ns=time.monotonic_ns() - 1,
        )
    )
