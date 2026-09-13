from dataclasses import replace
from types import SimpleNamespace

from sglang.srt.speculative.spectre.specstream.acceptance_tracker import AcceptanceTracker
from sglang.srt.speculative.spectre.specstream.controller import IOAwareController, SpecStreamDecision
from sglang.srt.speculative.spectre.specstream.cost_model import SpecStreamBatchState, SpecStreamCostProfile, workload_shape_key
from sglang.srt.speculative.spectre.specstream.profiler import SpecStreamProfiler, SpecStreamProfileRow
from sglang.srt.speculative.spectre.specstream.tp_straggler_monitor import TPStragglerSnapshot


def state():
    return SpecStreamBatchState(4, 65536, 57344, 8_000_000_000, 28, no_draft_ratio=1.0)


def choose(controller, batch=None, **kwargs):
    return controller.choose(batch or state(), SpecStreamCostProfile(), AcceptanceTracker().snapshot((8,)), **kwargs)


def failed_row(q=1, issued=0):
    return SpecStreamProfileRow(
        0, 0, "r", 1, "parallel", q,
        controller_selected_q=8, controller_selected_mode="parallel",
        slack_fill_issued_tokens=issued, h2d_window_remaining_us=1300,
        draft_step_ms=7.2,
    )


def test_unready_pipeline_preserves_q_and_does_not_repeat_q1_probes():
    controller = IOAwareController((8,), switch_threshold=0)
    for _ in range(300):
        decision = choose(controller, parallel_ready=False)
        assert (decision.q, decision.mode) == (8, "ordinary")
    assert controller._parallel_probe_rounds == {}
    # Admission can recover as soon as drafts are ready; this is not a global
    # switch disabling overlap or changing the configured horizon.
    controller._last = SpecStreamDecision(1, "parallel", "reset", 0)
    assert choose(controller, parallel_ready=True).mode == "parallel"


def test_failed_actual_q1_is_attributed_to_planned_q8_and_cools_down():
    controller = IOAwareController((8,), switch_threshold=0)
    choose(controller)
    controller.record_parallel_result(state(), failed_row())
    key = (workload_shape_key(state()), 8)
    assert key in controller._parallel_failures
    assert (workload_shape_key(state()), 1) not in controller._parallel_failures
    for _ in range(8):
        decision = choose(controller)
        assert (decision.q, decision.mode) == (8, "ordinary")
    assert choose(controller).mode == "parallel"


def test_short_window_and_repeated_no_slack_get_long_backoff():
    controller = IOAwareController((8,), switch_threshold=0)
    for _ in range(3):
        controller.record_parallel_result(state(), failed_row(q=8))
    key = (workload_shape_key(state()), 8)
    assert controller._parallel_failures[key] == (3, 128, "parallel_window_too_short")
    for _ in range(128):
        assert choose(controller).mode == "ordinary"
    assert choose(controller).mode == "parallel"


def test_backoff_is_shape_local_bounded_and_success_clears_it():
    controller = IOAwareController((8,), switch_threshold=0)
    for n in range(200):
        controller.record_parallel_result(replace(state(), context_tokens=65536+n*8192), failed_row())
    assert len(controller._parallel_failures) == 128
    controller.record_parallel_result(state(), failed_row())
    controller.record_parallel_result(state(), failed_row(q=8, issued=1))
    assert (workload_shape_key(state()), 8) not in controller._parallel_failures
    assert choose(controller, replace(state(), batch_size=1)).mode == "parallel"


def test_profiler_feeds_requested_q_without_relabeling_actual_q(tmp_path):
    controller = IOAwareController((8,), switch_threshold=0)
    profiler = SpecStreamProfiler(str(tmp_path / "rounds.csv"), 0, 1)
    profiler.parallel_feedback = controller.record_parallel_result
    batch = state()
    for rid in range(1, 6):
        profiler.record_decision(SpecStreamDecision(8, "parallel", "parallel_cost_probe", 1), controller.draft_load_tracker.snapshot(), TPStragglerSnapshot())
        meta = SimpleNamespace(round_id=rid, mode="parallel", q_len=1, context_tokens=65536, items=())
        profiler.begin_round(meta, batch_state=batch)
        profiler.record_target_forward(rid, 800)
        profiler.finish_round(rid, 4, 0, 0)
    assert (workload_shape_key(batch), 8) in controller._parallel_failures
    assert profiler.snapshot().empirical_cost(batch, 8, "parallel") is None
    assert profiler.snapshot().empirical_cost(batch, 1, "parallel") is not None


def test_forced_serial_stays_serial_with_ready_drafts():
    controller = IOAwareController((8,), switch_threshold=0)
    assert choose(controller, force_ordinary=True, parallel_ready=True).mode == "ordinary"


def test_catchup_does_not_erase_evidence_of_slack_issuance(tmp_path):
    profiler = SpecStreamProfiler(str(tmp_path / "round.csv"), 0, 1)
    profiler.begin_round(SimpleNamespace(round_id=1, mode="parallel", q_len=8, context_tokens=65536, items=()))
    profiler.record_grant(SimpleNamespace(grant_state="SLACK_FILL", grant_tokens=1), target_phase="target_forward")
    profiler.record_grant(SimpleNamespace(grant_state="DRAFT_CATCHUP", grant_tokens=1), target_phase="target_wait")
    assert profiler._active[1].slack_fill_issued_tokens == 1
