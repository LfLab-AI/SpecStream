import csv
import time
from types import SimpleNamespace

import pytest

from sglang.srt.speculative.spectre.specstream.config import (
    SpecStreamConfig,
    should_initialize_drafter_smctrl,
)
from sglang.srt.speculative.spectre.specstream.controller import (
    CandidateControllerCost,
    SpecStreamDecision,
)
from sglang.srt.speculative.spectre.specstream.draft_load_tracker import (
    DraftLoadSnapshot,
)
from sglang.srt.speculative.spectre.specstream.mps_env import MPSEnvironment
from sglang.srt.speculative.spectre.specstream.profiler import SpecStreamProfiler
from sglang.srt.speculative.spectre.specstream.slack_profiler import SlackProfiler
from sglang.srt.speculative.spectre.specstream.tp_straggler_monitor import (
    TPStragglerSnapshot,
)


def test_smctrl_initializes_for_remote_spectre_drafter_role():
    args = SimpleNamespace(
        specstream_smctrl_enabled=True,
        speculative_algorithm="SPECTRE",
        spectre_role="draft",
    )
    assert should_initialize_drafter_smctrl(args)

    args.spectre_role = "target"
    assert not should_initialize_drafter_smctrl(args)

    args.spectre_role = "draft"
    args.specstream_smctrl_enabled = False
    assert not should_initialize_drafter_smctrl(args)


def test_online_draft_timing_is_shape_specific_and_conservative():
    profiler = SlackProfiler(alpha=0.2)
    profiler.record_draft_step(0.5, draft_bs=1, draft_ctx_bucket="16k")
    assert profiler.snapshot(draft_bs=1, draft_ctx_bucket="16k").draft_step_ms == 0.5
    assert profiler.snapshot(draft_bs=8, draft_ctx_bucket="16k").draft_step_ms == 0.0

    profiler.record_draft_step(0.4, draft_bs=1, draft_ctx_bucket="16k")
    assert profiler.snapshot(
        draft_bs=1, draft_ctx_bucket="16k"
    ).draft_step_ms == pytest.approx(0.48)
    profiler.record_draft_step(0.8, draft_bs=1, draft_ctx_bucket="16k")
    assert profiler.snapshot(draft_bs=1, draft_ctx_bucket="16k").draft_step_ms == 0.8


def test_control_and_tp_metrics_are_written_to_profile(tmp_path):
    path = tmp_path / "profile.csv"
    profiler = SpecStreamProfiler(
        str(path),
        tp_rank=0,
        tp_size=1,
        mps_environment=MPSEnvironment(active_thread_percentage=80),
    )
    decision = SpecStreamDecision(
        q=2,
        mode="ordinary",
        reason="tp_straggler_throttle",
        estimated_cost=1.25,
        coexec_mode="SERIALIZE",
        rank_skew_ms=1.5,
        target_slowdown=0.08,
        parallel_cost=0.75,
        ordinary_cost=1.25,
        candidate_costs=(
            CandidateControllerCost(2, 0.75, 1.25, 0.5, 2.0),
            CandidateControllerCost(4, 0.70, 1.10, 0.7, 3.0),
        ),
    )
    load = DraftLoadSnapshot(
        samples=8,
        rtt_ema_ms=12.0,
        rtt_p95_ms=15.0,
        pressure_p95=0.75,
        timeout_rate=0.125,
        missing_ratio_ema=0.02,
        pending_p95=16,
    )
    tp = TPStragglerSnapshot(
        samples=4,
        colocated_rank=0,
        rank_forward_ms=(11.5,),
        rank_collective_wait_ms=(0.0,),
        rank_skew_ms=1.5,
        target_slowdown=0.08,
    )
    profiler.record_decision(decision, load, tp)
    profiler.record_draft_rtt(2, 12.5)
    profiler.record_grant_decision(
        SimpleNamespace(
            state=SimpleNamespace(value="TARGET_EXCLUSIVE"),
            tpc_low=0,
            tpc_high=0,
        ),
        target_phase="history_h2d",
        predicted_slack_us=750.0,
    )
    item = SimpleNamespace(
        rid="r0", committed_len=16384, history_len=8192, tail_tokens=512
    )
    meta = SimpleNamespace(
        round_id=1,
        mode="ordinary",
        q_len=2,
        context_tokens=16384,
        items=(item,),
        fallback=False,
        fallback_reason="",
        missing_draft_count=0,
    )
    profiler.begin_round(meta, chunk_tokens=2048)
    profiler.record_target_forward(1, 11.5, enqueue_ms=0.4)
    profiler.finish_round(1, accepted_tokens=2, staging_bytes=1024, cpu_bytes=4096)

    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["controller_selected_q"] == "2"
    assert row["coexec_mode"] == "SERIALIZE"
    assert row["draft_rtt_p95_ms"] == "15.0"
    assert row["mps_active_thread_percentage"] == "80"
    assert row["tp_rank_skew_ms"] == "1.5"
    assert row["target_forward_ms"] == "11.5"
    assert row["controller_cost_parallel"] == "0.75"
    assert row["controller_cost_ordinary"] == "1.25"
    assert '"q":2' in row["controller_candidate_costs"]
    assert row["draft_rtt_by_q"] == '{"2":12.5}'
    assert row["draft_rtt_samples_by_q"] == '{"2":1}'
    assert row["target_phase"] == "history_h2d"
    assert row["predicted_slack_us"] == "750.0"


def test_profile_records_rejection_without_crossing_sealed_history(tmp_path):
    path = tmp_path / "lifecycle.csv"
    profiler = SpecStreamProfiler(str(path), tp_rank=0, tp_size=1)
    items = (
        SimpleNamespace(
            rid="r0", committed_len=16384, history_len=8192, tail_tokens=512
        ),
        SimpleNamespace(
            rid="r1", committed_len=12288, history_len=8192, tail_tokens=512
        ),
    )
    meta = SimpleNamespace(
        round_id=7,
        mode="ordinary",
        q_len=8,
        context_tokens=28672,
        items=items,
        fallback=False,
        fallback_reason="",
        missing_draft_count=0,
    )
    post_states = [
        SimpleNamespace(
            committed_len=16387,
            history_len=10240,
            logical_len=16387,
            seal_inflight=False,
        ),
        SimpleNamespace(
            committed_len=12296,
            history_len=8192,
            logical_len=12296,
            seal_inflight=False,
        ),
    ]
    profiler.begin_round(meta, chunk_tokens=2048)
    profiler.finish_round(
        7,
        accepted_tokens=11,
        staging_bytes=1024,
        cpu_bytes=4096,
        accepted_per_req=[3, 8],
        post_states=post_states,
        sealed_history_floors=[8192, 8192],
    )

    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["rejected_requests"] == "1"
    assert row["rollback_tokens"] == "5"
    assert row["rollback_ratio"] == str(5 / 16)
    assert row["history_advanced_tokens"] == "2048"
    assert row["post_committed_tokens_total"] == str(16387 + 12296)
    assert row["rollback_crossed_history"] == "False"
    assert int(row["rollback_floor_min"]) >= 0


@pytest.mark.parametrize(
    "control_flags",
    (
        {"dynamic_q": True},
        {"coexec_enabled": True},
        {"tp_straggler_control": True},
        {
            "dynamic_q": True,
            "coexec_enabled": True,
            "tp_straggler_control": True,
        },
    ),
)
def test_native_gpu_kv_profile_only_mode_accepts_control_flags(control_flags):
    config = SpecStreamConfig(profile_only=True, **control_flags)

    assert config.control_runtime_enabled
    assert not config.enabled


@pytest.mark.parametrize(
    "control_flags",
    (
        {"dynamic_q": True},
        {"coexec_enabled": True},
        {"tp_straggler_control": True},
    ),
)
def test_control_flags_require_tiered_or_native_gpu_kv_runtime(control_flags):
    with pytest.raises(ValueError, match="profile-only"):
        SpecStreamConfig(**control_flags)


def test_native_gpu_kv_mode_cannot_enable_offload_only_features():
    with pytest.raises(ValueError, match="cohort scheduling"):
        SpecStreamConfig(profile_only=True, cohort_enabled=True)
    with pytest.raises(ValueError, match="Full-Restore"):
        SpecStreamConfig(profile_only=True, full_restore_baseline=True)


def test_pcie_slack_coexec_requires_tiered_kv_and_sm_control():
    with pytest.raises(ValueError, match="requires --specstream-enabled"):
        SpecStreamConfig(profile_only=True, pcie_slack_coexec=True)
    with pytest.raises(ValueError, match="requires --specstream-smctrl-enabled"):
        SpecStreamConfig(enabled=True, pcie_slack_coexec=True)
    with pytest.raises(ValueError, match="requires --specstream-coexec-require-mps"):
        SpecStreamConfig(
            enabled=True,
            pcie_slack_coexec=True,
            smctrl_enabled=True,
        )

    config = SpecStreamConfig(
        spectre_role="target",
        enabled=True,
        pcie_slack_coexec=True,
        smctrl_enabled=True,
        coexec_require_mps=True,
    )
    assert config.pcie_slack_coexec


def test_pcie_slack_coexec_rejects_full_restore_baseline():
    with pytest.raises(ValueError, match="incompatible"):
        SpecStreamConfig(
            enabled=True,
            full_restore_baseline=True,
            pcie_slack_coexec=True,
            smctrl_enabled=True,
            coexec_require_mps=True,
        )


def test_pcie_grant_poll_interval_is_bounded():
    assert SpecStreamConfig().pcie_grant_poll_us == 200
    with pytest.raises(ValueError, match="pcie_grant_poll_us"):
        SpecStreamConfig(pcie_grant_poll_us=24)
    with pytest.raises(ValueError, match="pcie_grant_poll_us"):
        SpecStreamConfig(pcie_grant_poll_us=10_001)


def test_serialized_h2d_requires_single_buffer_without_prefetch():
    config = SpecStreamConfig(
        num_buffers=1,
        layer_prefetch=False,
        serialize_h2d=True,
    )
    assert config.serialize_h2d
    with pytest.raises(ValueError, match="exactly one buffer"):
        SpecStreamConfig(
            num_buffers=2,
            layer_prefetch=False,
            serialize_h2d=True,
        )
    with pytest.raises(ValueError, match="incompatible with layer prefetch"):
        SpecStreamConfig(
            num_buffers=1,
            layer_prefetch=True,
            serialize_h2d=True,
        )


def test_tiered_kv_and_native_gpu_kv_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        SpecStreamConfig(enabled=True, profile_only=True)


def test_target_sm_control_requires_a_target_control_runtime():
    with pytest.raises(ValueError, match="SM control requires"):
        SpecStreamConfig(spectre_role="target", smctrl_enabled=True)

    config = SpecStreamConfig(
        spectre_role="target", profile_only=True, smctrl_enabled=True
    )
    assert config.smctrl_enabled


def test_drafter_sm_control_does_not_require_target_profile_runtime():
    config = SpecStreamConfig(spectre_role="draft", smctrl_enabled=True)
    assert config.smctrl_enabled
    assert not config.control_runtime_enabled


def test_sm_control_rejects_unbounded_quantum_and_ambiguous_calibration():
    with pytest.raises(ValueError, match="between 1 and 8"):
        SpecStreamConfig(grant_token_quantum=9)
    with pytest.raises(ValueError, match="fixed-TPC overlap requires"):
        SpecStreamConfig(smctrl_calibration_allow_overlap=True)


def test_profile_schema_change_uses_a_new_file_instead_of_shifting_columns(tmp_path):
    path = tmp_path / "profile.csv"
    path.write_text("timestamp,tp_rank\n1.0,0\n", encoding="utf-8")
    profiler = SpecStreamProfiler(str(path), tp_rank=0, tp_size=1)
    item = SimpleNamespace(rid="r0", committed_len=4, history_len=0, tail_tokens=4)
    meta = SimpleNamespace(
        round_id=1,
        mode="ordinary",
        q_len=1,
        context_tokens=4,
        items=(item,),
        fallback=False,
        fallback_reason="",
        missing_draft_count=0,
    )

    profiler.begin_round(meta, chunk_tokens=2)
    profiler.record_target_forward(1, 2.5)
    profiler.finish_round(1, accepted_tokens=1, staging_bytes=0, cpu_bytes=0)

    assert path.read_text(encoding="utf-8") == "timestamp,tp_rank\n1.0,0\n"
    schema_paths = list(tmp_path.glob("profile.schema-*.csv"))
    assert len(schema_paths) == 1
    with schema_paths[0].open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["target_forward_ms"] == "2.5"
    assert row["accepted_tokens"] == "1"


def test_grant_events_record_monotonic_clock_for_deadline_checks(tmp_path):
    profiler = SpecStreamProfiler(str(tmp_path / "profile.csv"), tp_rank=0, tp_size=1)
    deadline_us = time.monotonic_ns() // 1000 + 1_000_000
    message = SimpleNamespace(
        request_id="r0",
        spec_cnt=1,
        grant_epoch=1,
        grant_state="DRAFT_CATCHUP",
        grant_tokens=1,
        deadline_us=deadline_us,
        tpc_low=0,
        tpc_high=54,
        draft_step_ms=0.5,
    )

    profiler.record_grant(message, target_phase="draft_wait")

    with profiler.grant_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert int(row["timestamp_ns"]) > 0
    assert 0 < int(row["monotonic_ns"]) // 1000 < deadline_us


def test_async_queue_residence_does_not_corrupt_calibrated_h2d_bandwidth(tmp_path):
    path = tmp_path / "profile.csv"
    profiler = SpecStreamProfiler(
        str(path),
        tp_rank=0,
        tp_size=1,
        calibrated_h2d_gbps=24.0,
    )
    item = SimpleNamespace(rid="r0", committed_len=16, history_len=8, tail_tokens=8)
    meta = SimpleNamespace(
        round_id=1,
        mode="ordinary",
        q_len=1,
        context_tokens=16,
        items=(item,),
        fallback=False,
        fallback_reason="",
        missing_draft_count=0,
    )

    profiler.begin_round(meta, chunk_tokens=8)
    # A Python interval of 0.01 ms would imply an impossible 2400 GB/s.  It is
    # queue residence/enqueue timing and must not replace CUDA-event calibration.
    profiler.record_h2d(1, 24_000_000, 0.01)
    profiler.record_target_forward(1, 2.0)
    profiler.finish_round(1, accepted_tokens=1, staging_bytes=0, cpu_bytes=0)

    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["h2d_gbps"] == "24.0"
    assert row["copy_floor_ms"] == "1.0"
    assert row["h2d_timing_source"] == "cuda_event_startup_calibration"


def test_complete_runtime_cuda_event_coverage_replaces_queue_residence(tmp_path):
    path = tmp_path / "profile.csv"
    profiler = SpecStreamProfiler(
        str(path),
        tp_rank=0,
        tp_size=1,
        calibrated_h2d_gbps=20.0,
    )
    item = SimpleNamespace(rid="r0", committed_len=16, history_len=8, tail_tokens=8)
    meta = SimpleNamespace(
        round_id=1,
        mode="parallel",
        q_len=1,
        context_tokens=16,
        items=(item,),
        fallback=False,
        fallback_reason="",
        missing_draft_count=0,
    )

    profiler.begin_round(meta, chunk_tokens=8)
    # Host queue residence includes unrelated work, while the event brackets
    # the actual transfer on the copy stream.
    profiler.record_h2d(1, 10_000_000, 100.0)
    profiler.record_h2d_event(1, 10_000_000, 1.0, target_wait_ms=0.25)
    profiler.record_target_forward(1, 10.0)
    profiler.finish_round(1, accepted_tokens=1, staging_bytes=0, cpu_bytes=0)

    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["h2d_event_ops"] == "1"
    assert row["h2d_event_bytes"] == "10000000"
    assert row["h2d_event_ms"] == "1.0"
    assert row["h2d_wait_event_ops"] == "1"
    assert row["staging_wait_ms"] == "0.25"
    assert row["h2d_timing_source"] == "cuda_event_runtime+target_wait_event"
    assert row["exposed_copy_ms"] == "0.25"
    assert row["copy_compute_overlap"] == "0.75"
