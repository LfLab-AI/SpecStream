import csv
from types import SimpleNamespace

import pytest

from sglang.srt.speculative.spectre.specstream.config import (
    SpecStreamConfig,
    should_initialize_drafter_smctrl,
)
from sglang.srt.speculative.spectre.specstream.controller import SpecStreamDecision
from sglang.srt.speculative.spectre.specstream.draft_load_tracker import (
    DraftLoadSnapshot,
)
from sglang.srt.speculative.spectre.specstream.mps_env import MPSEnvironment
from sglang.srt.speculative.spectre.specstream.profiler import SpecStreamProfiler
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


def test_sm_control_rejects_multi_token_and_ambiguous_calibration():
    with pytest.raises(ValueError, match="must equal 1"):
        SpecStreamConfig(grant_token_quantum=2)
    with pytest.raises(ValueError, match="calibration overlap requires"):
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
