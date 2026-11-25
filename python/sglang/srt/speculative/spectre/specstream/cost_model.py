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
    # Sorted ``(q, milliseconds)`` and ``(q, samples)`` tuples keep this frozen
    # profile cheap to broadcast while allowing candidate-specific Draft RTT.
    # RTT already includes Drafter compute, queueing, and transport.
    draft_rtt_ms_by_q: tuple[tuple[int, float], ...] = ()
    draft_rtt_samples_by_q: tuple[tuple[int, int], ...] = ()
    repair_ms: float = 0.20
    target_compute_ratio: float = 0.0
    exposed_copy_ms: float = 0.0
    verification_correction_ms_by_q: tuple[tuple[int, float], ...] = ()

    def observed_draft_rtt_ms(self, q: int) -> float | None:
        for candidate_q, elapsed_ms in self.draft_rtt_ms_by_q:
            if int(candidate_q) == int(q):
                return max(float(elapsed_ms), 0.0)
        return None

    def draft_rtt_sample_count(self, q: int) -> int:
        for candidate_q, samples in self.draft_rtt_samples_by_q:
            if int(candidate_q) == int(q):
                return max(int(samples), 0)
        return 0

    def estimated_draft_rtt_ms(self, q: int) -> float:
        """Return an exact per-q RTT, or a bounded cold-start estimate."""

        observed = self.observed_draft_rtt_ms(q)
        if observed is not None:
            return observed
        points = [
            (int(candidate_q), max(float(elapsed_ms), 0.0))
            for candidate_q, elapsed_ms in self.draft_rtt_ms_by_q
        ]
        if len(points) >= 2:
            mean_q = sum(item[0] for item in points) / len(points)
            mean_ms = sum(item[1] for item in points) / len(points)
            denominator = sum((item[0] - mean_q) ** 2 for item in points)
            slope = (
                max(
                    0.0,
                    sum(
                        (candidate_q - mean_q) * (elapsed_ms - mean_ms)
                        for candidate_q, elapsed_ms in points
                    )
                    / denominator,
                )
                if denominator > 0
                else 0.0
            )
            intercept = max(0.0, mean_ms - slope * mean_q)
            return intercept + slope * int(q)
        if len(points) == 1:
            measured_q, measured_ms = points[0]
            return max(
                0.0,
                measured_ms + self.draft_per_token_ms * (int(q) - measured_q),
            )
        return max(
            0.0,
            self.draft_base_ms + self.draft_per_token_ms * int(q) + self.network_ms,
        )


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
    correction_ms = dict(profile.verification_correction_ms_by_q).get(q, 0.0)
    verify_ms = max(0.0, verify_ms + correction_ms)
    draft_ms = profile.estimated_draft_rtt_ms(q)
    useful = acceptance.useful_tokens(q)
    rollback_cost = acceptance.rollback(q) * profile.repair_ms
    # Candidate-specific Draft RTT already includes transport.  Adding the
    # global network EMA here would double-count the ordinary wait.
    parallel = max(draft_ms, verify_ms) + rollback_cost
    ordinary = draft_ms + verify_ms + rollback_cost
    if not math.isfinite(parallel) or not math.isfinite(ordinary):
        raise ValueError("SpecStream cost model produced a non-finite estimate")
    return CandidateCost(q, parallel / useful, ordinary / useful, verify_ms, draft_ms)
