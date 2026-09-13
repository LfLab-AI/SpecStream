import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.srt.speculative.spectre.specstream import coexec_runtime as runtime_module
from sglang.srt.speculative.spectre.specstream.coexec_runtime import TargetGrantRuntime
from sglang.srt.speculative.spectre.specstream.gpu_grant import (
    DraftExecutionGrant,
    DraftGrantTable,
)
from sglang.srt.speculative.spectre.specstream.gpu_grant_controller import (
    GpuGrantController,
)
from sglang.srt.speculative.spectre.spectre_protocol import (
    SpectreAction,
    SpectreRequest,
)


def _grant(tokens=4, *, epoch=1, spec_cnt=2, deadline_us=10**18):
    return DraftExecutionGrant(
        "r",
        spec_cnt,
        epoch,
        tokens,
        0,
        4,
        deadline_us=deadline_us,
        grant_state="DRAFT_CATCHUP",
    )


def _runtime(monkeypatch, *, quantum=4, desired_q=7, spec_cnt=2, step_ms=1.0):
    monkeypatch.setattr(runtime_module.time, "monotonic_ns", lambda: 1_000_000)
    controller = GpuGrantController(
        None, calibration_tpcs=4, guard_us=200, catchup_token_quantum=quantum
    )
    runtime = TargetGrantRuntime(controller)
    runtime.register_round(
        request_id="r",
        spec_cnt=spec_cnt,
        desired_q=desired_q,
        target_shape="shape",
        draft_bs=1,
        draft_ctx_bucket="2k",
        predicted_slack_us=0,
        draft_step_ms=step_ms,
    )
    return runtime


def _ack(grant, tokens):
    return SpectreRequest(
        request_id=grant.request_id,
        spec_cnt=grant.spec_cnt,
        action=SpectreAction.GRANT_ACK,
        grant_epoch=grant.grant_epoch,
        grant_tokens=tokens,
        draft_step_ms=1.0,
    )


def test_catchup_budget_is_opt_in_bounded_by_deadline_and_remaining(monkeypatch):
    runtime = _runtime(monkeypatch)
    # now=1000us, deadline=4500us, guard=200us -> three measured 1ms steps.
    grant = runtime.waiting_grants([("r", 2)], deadline_us=4500)[0]
    assert grant.grant_tokens == 3
    assert runtime.waiting_grants([("r", 2)], deadline_us=10**18) == []
    assert runtime.acknowledge(_ack(grant, 3))
    grant2 = runtime.waiting_grants([("r", 2)], deadline_us=10**18)[0]
    assert grant2.grant_tokens == 4
    assert runtime.acknowledge(_ack(grant2, 4))
    assert runtime.waiting_grants([("r", 2)], deadline_us=10**18) == []


def test_partial_lease_ack_returns_unused_budget_and_rejects_replay(monkeypatch):
    runtime = _runtime(monkeypatch, desired_q=5)
    grant = runtime.waiting_grants([("r", 2)], deadline_us=10**18)[0]
    assert grant.grant_tokens == 4
    assert not runtime.acknowledge(_ack(grant, 5))
    assert runtime.state_for("r", 2).outstanding_tokens == 4
    assert runtime.acknowledge(_ack(grant, 2))
    assert not runtime.acknowledge(_ack(grant, 2))
    state = runtime.state_for("r", 2)
    assert state.issued == state.acked == 2
    grant2 = runtime.waiting_grants([("r", 2)], deadline_us=10**18)[0]
    assert grant2.grant_tokens == 3


@pytest.mark.parametrize("kwargs", [{"quantum": 1}, {"step_ms": 0.0}, {"spec_cnt": 0}])
def test_single_token_fallback_for_default_unmeasured_and_initial_prefill(
    monkeypatch, kwargs
):
    runtime = _runtime(monkeypatch, **kwargs)
    spec_cnt = kwargs.get("spec_cnt", 2)
    grant = runtime.waiting_grants([("r", spec_cnt)], deadline_us=10**18)[0]
    assert grant.grant_tokens == 1


def test_overlap_grant_cannot_have_multiple_tokens():
    with pytest.raises(ValueError, match="exactly one token"):
        DraftExecutionGrant("r", 2, 1, 2, 0, 4, grant_state="SLACK_FILL")


def test_partial_lease_preserves_deadline_and_epoch_at_every_launch():
    table = DraftGrantTable()
    grant = _grant(deadline_us=100)
    table.apply(grant)
    assert table.active("r", spec_cnt=2, now_us=99) == grant
    assert table.consume_one("r", spec_cnt=2, grant_epoch=1)
    assert table.remaining("r", spec_cnt=2, grant_epoch=1) == 3
    assert table.active("r", spec_cnt=2, now_us=100) is None
    assert table.pop_expired("r", spec_cnt=2, now_us=100) == grant
    assert not table.consume_one("r", spec_cnt=2, grant_epoch=1)
    table.apply(_grant(epoch=2))
    assert not table.consume_one("r", spec_cnt=2, grant_epoch=1)
    assert table.remaining("r", spec_cnt=2, grant_epoch=2) == 4


def _drafter_methods():
    # Exercise actual scheduler lifecycle methods without requiring CUDA,
    # torch, SGLang frontend or a tokenizer in a source-only CPU checkout.
    path = (
        Path(__file__).parents[2]
        / "srt/speculative/spectre/drafter/spectre_draft_scheduler_mixin.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {
        "_record_completed_grant_step",
        "_close_grant_lease",
        "_ack_expired_grant",
    }
    methods = [
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            ast.ClassDef(
                name="ActualLeaseMethods",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    namespace = {"logger": SimpleNamespace(debug=lambda *args: None)}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    instance = namespace["ActualLeaseMethods"]()
    instance._grant_table = DraftGrantTable()
    instance.tp_rank = 0
    instance.acks = []
    instance._send_grant_ack = (
        lambda req, grant, elapsed_ms, **kwargs: instance.acks.append(
            (grant, elapsed_ms, kwargs)
        )
    )
    return instance


def test_drafter_ack_is_aggregated_once_after_actual_gpu_completions():
    drafter = _drafter_methods()
    req = SimpleNamespace(rid="r", spec_cnt=2)
    grant = _grant(tokens=3)
    drafter._grant_table.apply(grant)
    for elapsed_ms in (1.0, 2.0):
        drafter._record_completed_grant_step(req, grant, elapsed_ms)
        assert drafter.acks == []
    drafter._record_completed_grant_step(req, grant, 1.0)
    assert len(drafter.acks) == 1
    assert drafter.acks[0][2]["grant_tokens"] == 3
    assert drafter.acks[0][1] == 2.0  # conservative per-step cost, not sum
    drafter._close_grant_lease(req, grant, grant_state="EARLY_FINISH")
    assert len(drafter.acks) == 1


def test_drafter_early_finish_and_expiry_ack_only_completed_prefix(monkeypatch):
    for disposition in ("EARLY_FINISH", "EXPIRED"):
        drafter = _drafter_methods()
        req = SimpleNamespace(rid="r", spec_cnt=2)
        grant = _grant(tokens=4, deadline_us=2000)
        drafter._grant_table.apply(grant)
        drafter._record_completed_grant_step(req, grant, 0.5)
        if disposition == "EXPIRED":
            monkeypatch.setattr(runtime_module.time, "monotonic_ns", lambda: 3_000_000)
            assert drafter._ack_expired_grant(req)
        else:
            drafter._close_grant_lease(req, grant, grant_state=disposition)
        assert len(drafter.acks) == 1
        assert drafter.acks[0][2]["grant_tokens"] == 1
        assert drafter.acks[0][2]["grant_state"] == disposition
        assert drafter._grant_table.current("r") is None


def test_fixed_tpc_protection_recovers_after_three_target_only_samples():
    controller = GpuGrantController(
        None, calibration_tpcs=4, calibration_allow_overlap=True
    )
    for _ in range(2):
        controller.record_fixed_target_forward(
            target_shape="shape",
            elapsed_ms=10,
            possible_overlap=False,
            confirmed_overlap=False,
        )
    controller.record_fixed_target_forward(
        target_shape="shape",
        elapsed_ms=20,
        possible_overlap=True,
        confirmed_overlap=True,
    )
    assert controller._force_exclusive
    for _ in range(3):
        controller.record_fixed_target_forward(
            target_shape="shape",
            elapsed_ms=10,
            possible_overlap=False,
            confirmed_overlap=False,
        )
    assert not controller._force_exclusive
