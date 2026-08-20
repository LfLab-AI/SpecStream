from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


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
    dynamic_q: bool = False
    q_candidates: tuple[int, ...] = (1, 2, 4, 6, 8)
    q_switch_threshold: float = 0.08
    cohort_enabled: bool = False
    max_cohort_size: int = 8
    max_cohort_delay_us: float = 200.0
    profile_path: str = "specstream_profile.csv"
    shadow_attention: bool = False
    strict_invariants: bool = True

    def __post_init__(self) -> None:
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
        if not 0.0 <= self.q_switch_threshold < 1.0:
            raise ValueError("specstream_q_switch_threshold must be in [0, 1)")
        if self.max_cohort_size < 1:
            raise ValueError("specstream_max_cohort_size must be positive")
        if self.max_cohort_delay_us < 0:
            raise ValueError("specstream_max_cohort_delay_us cannot be negative")
        parse_q_candidates(self.q_candidates)
        if self.cohort_enabled and not self.enabled:
            raise ValueError(
                "SpecStream cohort scheduling requires --specstream-enabled"
            )
        if self.dynamic_q and not self.enabled:
            raise ValueError("SpecStream dynamic q requires --specstream-enabled")
        if self.full_restore_baseline and not self.enabled:
            raise ValueError(
                "SpecStream Full-Restore baseline requires --specstream-enabled"
            )

    @property
    def cpu_memory_bytes(self) -> int:
        return int(self.cpu_memory_gb) * 1024**3

    @classmethod
    def from_server_args(cls, server_args) -> "SpecStreamConfig":
        return cls(
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
            dynamic_q=bool(server_args.specstream_dynamic_q),
            q_candidates=parse_q_candidates(server_args.specstream_q_candidates),
            q_switch_threshold=float(server_args.specstream_q_switch_threshold),
            cohort_enabled=bool(server_args.specstream_cohort_enabled),
            max_cohort_size=int(server_args.specstream_max_cohort_size),
            max_cohort_delay_us=float(server_args.specstream_max_cohort_delay_us),
            profile_path=str(server_args.specstream_profile_path),
            shadow_attention=bool(server_args.specstream_shadow_attention),
            strict_invariants=bool(server_args.specstream_strict_invariants),
        )
