import csv
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from sglang.srt.speculative.spectre.specstream.acceptance_tracker import (
    AcceptanceTracker,
)
from sglang.srt.speculative.spectre.specstream.cost_model import (
    EmpiricalCandidateCost,
    SpecStreamBatchState,
    SpecStreamCostProfile,
    estimate_candidate_cost,
    execution_shape_key,
    workload_shape_key,
)
from sglang.srt.speculative.spectre.specstream.controller import IOAwareController
from sglang.srt.speculative.spectre.specstream.multi_gpu_tp_policy import (
    MultiGPUTPPolicy,
)
from sglang.srt.speculative.spectre.specstream.profiler import SpecStreamProfiler
from sglang.srt.speculative.spectre.specstream.tp_straggler_monitor import (
    TPStragglerSnapshot,
)


def _state():
    return SpecStreamBatchState(
        batch_size=4,
        context_tokens=65536,
        history_tokens=57344,
        history_bytes=8_000_000_000,
        num_chunks=28,
        gpu_history_tokens=8192,
        attention_impl="cohort_splitkv",
        tp_size=2,
    )


def _meta(round_id=1, mode="parallel", q=4):
    return SimpleNamespace(
        round_id=round_id,
        mode=mode,
        q_len=q,
        context_tokens=65536,
        items=tuple(
            SimpleNamespace(
                rid=f"r{i}",
                committed_len=16384,
                history_len=14336,
                gpu_history_len=2048,
                tail_tokens=2048,
            )
            for i in range(4)
        ),
    )


def test_shape_isolation_includes_batch_q_residency_phase_and_implementation():
    batch = _state()
    key = execution_shape_key(batch, 4)
    for changed in (
        replace(batch, batch_size=8),
        replace(batch, history_bytes=10_000_000_000),
        replace(batch, context_tokens=131072),
        replace(batch, gpu_history_tokens=0),
        replace(batch, attention_impl="different"),
        replace(batch, tp_size=1),
        replace(batch, phase="prefill"),
    ):
        assert execution_shape_key(changed, 4) != key
    assert execution_shape_key(batch, 8) != key


def test_cpu_attention_submission_does_not_train_gpu_cost(tmp_path):
    profiler = SpecStreamProfiler(str(tmp_path / "profile.csv"), 0, 1)
    before = profiler.snapshot()
    profiler.begin_round(_meta())
    profiler.record_attention(1, 780.0)
    profiler.record_attention(1, 100.0, tail=True)
    profiler.record_target_forward(1, 784.0, enqueue_ms=780.0)
    profiler.finish_round(1, 12, 0, 0)
    after = profiler.snapshot()
    assert after.attention_chunk_ms_q1 == before.attention_chunk_ms_q1
    assert after.target_other_ms == before.target_other_ms
    with profiler.path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["attention_timing_source"] == "cpu_enqueue"
    assert row["copy_compute_overlap"] == "0.0"


def test_true_round_walltime_does_not_add_overlapping_receive(tmp_path, monkeypatch):
    profiler = SpecStreamProfiler(str(tmp_path / "profile.csv"), 0, 1)
    profiler.start_round_clock(started_ns=1_000_000_000)
    profiler.begin_round(_meta())
    profiler.record_target_forward(1, 100.0)
    profiler.record_network_wait(80.0)
    profiler_module = sys.modules[SpecStreamProfiler.__module__]
    monkeypatch.setattr(profiler_module.time, "perf_counter_ns", lambda: 1_120_000_000)
    profiler.finish_round(1, 12, 0, 0)
    with profiler.path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["round_ms"]) == 120.0
    assert row["round_timing_source"] == "outer_round_wall"


def test_empirical_round_cost_uses_warmed_same_shape_and_actual_progress(tmp_path):
    profiler = SpecStreamProfiler(str(tmp_path / "profile.csv"), 0, 1)
    batch = _state()
    for round_id, verify_ms in enumerate((900, 600, 100, 100, 100), 1):
        profiler.begin_round(_meta(round_id, "ordinary"), batch_state=batch)
        profiler.record_target_forward(round_id, verify_ms)
        profiler.record_round_timing(round_id, wall_ms=verify_ms + 20)
        profiler.finish_round(round_id, 12, 0, 0)
    profile = profiler.snapshot()
    point = profile.empirical_cost(batch, 4, "ordinary")
    assert point is not None and point.samples == 3
    assert point.verify_ms == pytest.approx(100)
    assert point.round_ms == pytest.approx(120)
    assert point.useful_tokens == pytest.approx(3)
    cost = estimate_candidate_cost(
        4, batch, profile, AcceptanceTracker().snapshot((4,))
    )
    assert cost.verify_ms == pytest.approx(100)
    assert cost.ordinary_ms_per_useful_token == pytest.approx(40)
    assert profile.empirical_cost(replace(batch, batch_size=8), 4, "ordinary") is None


def test_unmeasured_overlap_is_not_assumed_and_q1_has_no_draft_rtt():
    batch = _state()
    acceptance = AcceptanceTracker().snapshot((1, 4))
    profile = SpecStreamCostProfile(draft_rtt_ms_by_q=((1, 1000), (4, 90)))
    q4 = estimate_candidate_cost(4, batch, profile, acceptance)
    assert q4.parallel_ms_per_useful_token == q4.ordinary_ms_per_useful_token
    q1 = estimate_candidate_cost(1, batch, profile, acceptance)
    assert q1.draft_ms == 0.0


def test_profile_only_partial_round_never_becomes_zero_cost():
    batch = _state()
    profile = SpecStreamCostProfile(
        empirical_candidates=(
            EmpiricalCandidateCost(
                workload_shape_key(batch), 4, "ordinary", 100, 0, 3, 8
            ),
        )
    )
    cost = estimate_candidate_cost(
        4, batch, profile, AcceptanceTracker().snapshot((4,))
    )
    assert cost.verify_ms == 100
    assert cost.ordinary_ms_per_useful_token > 0


def test_controller_checks_the_selected_q_baseline_instead_of_previous_q():
    controller = IOAwareController(
        (4, 8), switch_threshold=0, multi_gpu_policy=MultiGPUTPPolicy()
    )
    profile = SpecStreamCostProfile(draft_rtt_samples_by_q=((4, 8), (8, 8)))
    snapshots = {
        4: TPStragglerSnapshot(
            samples=5, shape_key=(4,), baseline_ready=True, overlap_active=False
        ),
        8: TPStragglerSnapshot(shape_key=(8,)),
    }
    # Cold acceptance forces q8 exploration; q4's warmed permission must not
    # allow the uncalibrated q8 to overlap.
    decision = controller.choose(
        _state(),
        profile,
        AcceptanceTracker().snapshot((4, 8)),
        tp_snapshots_by_q=snapshots,
    )
    assert decision.q == 8
    assert decision.mode == "ordinary"
    assert decision.reason == "tp_baseline_warmup"


def test_parallel_cost_probes_are_bounded_and_never_override_tp_gate():
    batch = _state()
    profile = SpecStreamCostProfile(
        draft_rtt_ms_by_q=((4, 90),),
        draft_rtt_samples_by_q=((4, 8),),
        empirical_candidates=(
            EmpiricalCandidateCost(
                workload_shape_key(batch), 4, "ordinary", 100, 120, 3, 8
            ),
        ),
    )
    controller = IOAwareController(
        (4,),
        switch_threshold=0,
        parallel_probe_interval=3,
        multi_gpu_policy=MultiGPUTPPolicy(),
    )
    acceptance = AcceptanceTracker().snapshot((4,))
    ready = TPStragglerSnapshot(
        samples=5, shape_key=(4,), baseline_ready=True, overlap_active=False
    )
    for _ in range(2):
        assert (
            controller.choose(batch, profile, acceptance, tp_snapshot=ready).mode
            == "ordinary"
        )
    assert (
        controller.choose(batch, profile, acceptance, tp_snapshot=ready).reason
        == "parallel_cost_probe"
    )
    for _ in range(6):
        result = controller.choose(
            batch, profile, acceptance, tp_snapshot=TPStragglerSnapshot()
        )
        assert result.mode == "ordinary"
        assert result.reason != "parallel_cost_probe"
