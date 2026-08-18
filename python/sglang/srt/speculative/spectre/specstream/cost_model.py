from __future__ import annotations

from dataclasses import dataclass
import math

from sglang.srt.speculative.spectre.specstream.acceptance_tracker import (
    AcceptanceSnapshot,
)


@dataclass(frozen=True)
class SpecStreamBatchState:
    batch_size: int
    context_tokens: int
    history_tokens: int
    history_bytes: int
    num_chunks: int
    no_draft_ratio: float = 0.0
    rejected: bool = False
    high_overhead: bool = False


@dataclass(frozen=True)
class SpecStreamCostProfile:
    h2d_gbps: float = 12.0
    fill_drain_ms: float = 0.05
    target_other_ms: float = 0.25
    attention_chunk_ms_q1: float = 0.08
    tail_ms_q1: float = 0.04
    draft_base_ms: float = 0.15
    draft_per_token_ms: float = 0.08
    network_ms: float = 0.05
    repair_ms: float = 0.20


@dataclass(frozen=True)
class CandidateCost:
    q: int
    parallel_ms_per_useful_token: float
    ordinary_ms_per_useful_token: float
    verify_ms: float
    draft_ms: float


def estimate_candidate_cost(
    q: int,
    batch: SpecStreamBatchState,
    profile: SpecStreamCostProfile,
    acceptance: AcceptanceSnapshot,
) -> CandidateCost:
    if q < 1:
        raise ValueError("q must be positive")
    bandwidth_bytes_per_ms = max(profile.h2d_gbps, 1e-6) * 1e6
    copy_total_ms = batch.history_bytes / bandwidth_bytes_per_ms
    copy_chunk_ms = copy_total_ms / max(batch.num_chunks, 1)
    # Multi-query attention grows sublinearly for the small q candidate set;
    # this term is replaced by measured buckets as profiling warms up.
    attention_chunk_ms = profile.attention_chunk_ms_q1 * (0.65 + 0.35 * q)
    verify_ms = (
        profile.target_other_ms
        + profile.fill_drain_ms
        + max(batch.num_chunks, 1) * max(copy_chunk_ms, attention_chunk_ms)
        + profile.tail_ms_q1 * q
    )
    draft_ms = profile.draft_base_ms + profile.draft_per_token_ms * q
    useful = acceptance.useful_tokens(q)
    rollback_cost = acceptance.rollback(q) * profile.repair_ms
    parallel = max(draft_ms + profile.network_ms, verify_ms) + rollback_cost
    ordinary = draft_ms + profile.network_ms + verify_ms
    if not math.isfinite(parallel) or not math.isfinite(ordinary):
        raise ValueError("SpecStream cost model produced a non-finite estimate")
    return CandidateCost(q, parallel / useful, ordinary / useful, verify_ms, draft_ms)
