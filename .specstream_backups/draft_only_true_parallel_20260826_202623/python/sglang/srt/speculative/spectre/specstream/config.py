from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def should_initialize_drafter_smctrl(server_args) -> bool:
    """True only for the standalone remote SPECTRE Drafter process."""

    return bool(
        getattr(server_args, "specstream_smctrl_enabled", False)
        and getattr(server_args, "speculative_algorithm", None) == "SPECTRE"
        and getattr(server_args, "spectre_role", None) == "draft"
    )


def parse_q_candidates(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",") if item.strip()]
        try:
            candidates = tuple(int(item) for item in raw_values)
        except ValueError as exc:
            raise ValueError(
                "specstream_q_candidates must be a comma-separated integer list"
            ) from exc
    else:
        candidates = tuple(int(item) for item in value)

    if not candidates:
        raise ValueError("specstream_q_candidates cannot be empty")
    if any(item < 1 for item in candidates):
        raise ValueError("all SpecStream q candidates must be positive")
    if len(set(candidates)) != len(candidates):
        raise ValueError("specstream_q_candidates cannot contain duplicates")
    return tuple(sorted(candidates))


@dataclass(frozen=True)
class SpecStreamConfig:
    spectre_role: str | None = None
    enabled: bool = False
    profile_only: bool = False
    full_restore_baseline: bool = False
    reference_attention: bool = True
    chunk_tokens: int = 2048
    num_buffers: int = 2
    chunks_per_transfer: int = 4
    layer_prefetch: bool = True
    active_tail_tokens: int = 512
    min_history_tokens: int = 8192
    cpu_memory_gb: int = 128
    default_q: int = 5
    dynamic_q: bool = False
    q_candidates: tuple[int, ...] = (1, 2, 4, 6, 8)
    q_switch_threshold: float = 0.08
    coexec_enabled: bool = False
    coexec_draft_pressure_ratio: float = 0.80
    coexec_timeout_rate_threshold: float = 0.10
    coexec_pending_high_watermark: int = 16
    coexec_compute_ratio_threshold: float = 0.90
    coexec_require_mps: bool = False
    smctrl_enabled: bool = False
    grant_token_quantum: int = 1
    coexec_target_slowdown_budget: float = 0.05
    coexec_guard_us: float = 200.0
    coexec_resource_profile_path: str = "specstream_resource_profile.json"
    smctrl_library: str = ""
    smctrl_mask_scope: str = "stream"
    smctrl_calibration_tpcs: int = 0
    smctrl_calibration_allow_overlap: bool = False
    smctrl_complementary_partition: bool = False
    tp_straggler_control: bool = False
    colocated_tp_rank: int = 0
    tp_straggler_budget_ms: float = 1.0
    target_slowdown_budget: float = 0.10
    tp_monitor_interval: int = 8
    cohort_enabled: bool = False
    max_cohort_size: int = 8
    max_cohort_delay_us: float = 200.0
    profile_path: str = "specstream_profile.csv"
    shadow_attention: bool = False
    strict_invariants: bool = True

    def __post_init__(self) -> None:
        if self.enabled and self.profile_only:
            raise ValueError(
                "--specstream-enabled and --specstream-profile-only are mutually "
                "exclusive"
            )
        if self.chunk_tokens < 1:
            raise ValueError("specstream_chunk_tokens must be positive")
        if self.num_buffers < 1:
            raise ValueError("specstream_num_buffers must be positive")
        if self.chunks_per_transfer < 1:
            raise ValueError("specstream_chunks_per_transfer must be positive")
        if self.active_tail_tokens < 0:
            raise ValueError("specstream_active_tail_tokens cannot be negative")
        if self.min_history_tokens < 0:
            raise ValueError("specstream_min_history_tokens cannot be negative")
        if self.cpu_memory_gb < 1:
            raise ValueError("specstream_cpu_memory_gb must be positive")
        if self.default_q < 1:
            raise ValueError("specstream default q must be positive")
        if not 0.0 <= self.q_switch_threshold < 1.0:
            raise ValueError("specstream_q_switch_threshold must be in [0, 1)")
        if not 0.0 < self.coexec_draft_pressure_ratio <= 1.0:
            raise ValueError("specstream_coexec_draft_pressure_ratio must be in (0, 1]")
        if not 0.0 <= self.coexec_timeout_rate_threshold <= 1.0:
            raise ValueError(
                "specstream_coexec_timeout_rate_threshold must be in [0, 1]"
            )
        if self.coexec_pending_high_watermark < 1:
            raise ValueError(
                "specstream_coexec_pending_high_watermark must be positive"
            )
        if not 0.0 < self.coexec_compute_ratio_threshold <= 1.0:
            raise ValueError(
                "specstream_coexec_compute_ratio_threshold must be in (0, 1]"
            )
        if self.grant_token_quantum != 1:
            raise ValueError("SpecStream v1 grant_token_quantum must equal 1")
        if not 0.0 <= self.coexec_target_slowdown_budget <= 1.0:
            raise ValueError(
                "specstream_coexec_target_slowdown_budget must be in [0, 1]"
            )
        if self.coexec_guard_us < 0:
            raise ValueError("specstream_coexec_guard_us cannot be negative")
        if self.smctrl_calibration_tpcs < 0:
            raise ValueError("specstream_smctrl_calibration_tpcs cannot be negative")
        if self.smctrl_mask_scope not in {"stream", "global"}:
            raise ValueError("specstream_smctrl_mask_scope must be stream or global")
        if self.smctrl_complementary_partition and not self.smctrl_enabled:
            raise ValueError(
                "complementary TPC partition requires --specstream-smctrl-enabled"
            )
        if (
            self.smctrl_complementary_partition
            and self.smctrl_mask_scope != "global"
        ):
            raise ValueError(
                "complementary TPC partition requires "
                "--specstream-smctrl-mask-scope global"
            )
        if self.smctrl_calibration_allow_overlap and self.smctrl_calibration_tpcs < 1:
            raise ValueError(
                "calibration overlap requires --specstream-smctrl-calibration-tpcs"
            )
        if (
            self.smctrl_enabled
            and self.smctrl_calibration_tpcs < 1
            and not self.coexec_resource_profile_path
        ):
            raise ValueError(
                "SpecStream SM control requires a calibrated resource profile path"
            )
        if self.colocated_tp_rank < 0:
            raise ValueError("specstream_colocated_tp_rank cannot be negative")
        if self.tp_straggler_budget_ms < 0:
            raise ValueError("specstream_tp_straggler_budget_ms cannot be negative")
        if self.target_slowdown_budget < 0:
            raise ValueError("specstream_target_slowdown_budget cannot be negative")
        if self.tp_monitor_interval < 1:
            raise ValueError("specstream_tp_monitor_interval must be positive")
        if self.max_cohort_size < 1:
            raise ValueError("specstream_max_cohort_size must be positive")
        if self.max_cohort_delay_us < 0:
            raise ValueError("specstream_max_cohort_delay_us cannot be negative")
        parse_q_candidates(self.q_candidates)
        if self.cohort_enabled and not self.enabled:
            raise ValueError(
                "SpecStream cohort scheduling requires --specstream-enabled"
            )
        if self.dynamic_q and not self.control_runtime_enabled:
            raise ValueError(
                "SpecStream dynamic q requires --specstream-enabled or "
                "--specstream-profile-only"
            )
        if self.coexec_enabled and not self.control_runtime_enabled:
            raise ValueError(
                "SpecStream co-execution requires --specstream-enabled or "
                "--specstream-profile-only"
            )
        if (
            self.smctrl_enabled
            and self.spectre_role != "draft"
            and not self.control_runtime_enabled
        ):
            raise ValueError(
                "SpecStream SM control requires --specstream-enabled or "
                "--specstream-profile-only on the Target"
            )
        if self.tp_straggler_control and not self.control_runtime_enabled:
            raise ValueError(
                "SpecStream TP straggler control requires --specstream-enabled or "
                "--specstream-profile-only"
            )
        if self.full_restore_baseline and not self.enabled:
            raise ValueError(
                "SpecStream Full-Restore baseline requires --specstream-enabled"
            )

    @property
    def cpu_memory_bytes(self) -> int:
        return int(self.cpu_memory_gb) * 1024**3

    @property
    def control_runtime_enabled(self) -> bool:
        """Whether profiling/control may run around the SPECTRE Target.

        ``enabled`` selects tiered CPU/GPU KV. ``profile_only`` deliberately
        leaves native SPECTRE KV resident on GPU while exposing the same
        co-execution and TP-straggler controllers for isolated experiments.
        """

        return bool(self.enabled or self.profile_only)

    @classmethod
    def from_server_args(cls, server_args) -> "SpecStreamConfig":
        return cls(
            spectre_role=getattr(server_args, "spectre_role", None),
            enabled=bool(server_args.specstream_enabled),
            profile_only=bool(server_args.specstream_profile_only),
            full_restore_baseline=bool(server_args.specstream_full_restore_baseline),
            reference_attention=bool(server_args.specstream_reference_attention),
            chunk_tokens=int(server_args.specstream_chunk_tokens),
            num_buffers=int(server_args.specstream_num_buffers),
            chunks_per_transfer=int(server_args.specstream_chunks_per_transfer),
            layer_prefetch=bool(server_args.specstream_layer_prefetch),
            active_tail_tokens=int(server_args.specstream_active_tail_tokens),
            min_history_tokens=int(server_args.specstream_min_history_tokens),
            cpu_memory_gb=int(server_args.specstream_cpu_memory_gb),
            default_q=int(server_args.speculative_num_steps) + 1,
            dynamic_q=bool(server_args.specstream_dynamic_q),
            q_candidates=parse_q_candidates(server_args.specstream_q_candidates),
            q_switch_threshold=float(server_args.specstream_q_switch_threshold),
            coexec_enabled=bool(server_args.specstream_coexec_enabled),
            coexec_draft_pressure_ratio=float(
                server_args.specstream_coexec_draft_pressure_ratio
            ),
            coexec_timeout_rate_threshold=float(
                server_args.specstream_coexec_timeout_rate_threshold
            ),
            coexec_pending_high_watermark=int(
                server_args.specstream_coexec_pending_high_watermark
            ),
            coexec_compute_ratio_threshold=float(
                server_args.specstream_coexec_compute_ratio_threshold
            ),
            coexec_require_mps=bool(server_args.specstream_coexec_require_mps),
            smctrl_enabled=bool(server_args.specstream_smctrl_enabled),
            grant_token_quantum=int(server_args.specstream_grant_token_quantum),
            coexec_target_slowdown_budget=float(
                server_args.specstream_coexec_target_slowdown_budget
            ),
            coexec_guard_us=float(server_args.specstream_coexec_guard_us),
            coexec_resource_profile_path=str(
                server_args.specstream_coexec_resource_profile_path
            ),
            smctrl_library=str(server_args.specstream_smctrl_library),
            smctrl_mask_scope=str(server_args.specstream_smctrl_mask_scope),
            smctrl_calibration_tpcs=int(server_args.specstream_smctrl_calibration_tpcs),
            smctrl_calibration_allow_overlap=bool(
                server_args.specstream_smctrl_calibration_allow_overlap
            ),
            smctrl_complementary_partition=bool(
                server_args.specstream_smctrl_complementary_partition
            ),
            tp_straggler_control=bool(server_args.specstream_tp_straggler_control),
            colocated_tp_rank=int(server_args.specstream_colocated_tp_rank),
            tp_straggler_budget_ms=float(server_args.specstream_tp_straggler_budget_ms),
            target_slowdown_budget=float(server_args.specstream_target_slowdown_budget),
            tp_monitor_interval=int(server_args.specstream_tp_monitor_interval),
            cohort_enabled=bool(server_args.specstream_cohort_enabled),
            max_cohort_size=int(server_args.specstream_max_cohort_size),
            max_cohort_delay_us=float(server_args.specstream_max_cohort_delay_us),
            profile_path=str(server_args.specstream_profile_path),
            shadow_attention=bool(server_args.specstream_shadow_attention),
            strict_invariants=bool(server_args.specstream_strict_invariants),
        )
