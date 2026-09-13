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
    gpu_history_tokens: int = 0
    phase: str = "decode"
    attention_impl: str = ""
    tp_size: int = 1


def _bucket(value: int, quantum: int) -> int:
    """Bound workload drift within a bucket without per-token fragmentation."""
    value = max(int(value), 0)
    return (value + quantum - 1) // quantum


def workload_shape_key(batch: SpecStreamBatchState) -> tuple:
    # Context uses the batch sum (not only the longest request); residency is
    # represented independently so increasing batch cannot reuse a warm cache
    # baseline. Fine 64 MiB miss buckets bound transfer-cost drift.
    return (
        str(batch.phase),
        int(batch.batch_size),
        _bucket(batch.context_tokens, 2048 * max(batch.batch_size, 1)),
        _bucket(batch.history_bytes, 64 * 1024 * 1024),
        _bucket(batch.gpu_history_tokens, 1024),
        str(batch.attention_impl),
        int(batch.tp_size),
    )


def execution_shape_key(
    batch: SpecStreamBatchState,
    q: int,
    *,
    attention_impl: str | None = None,
    tp_size: int | None = None,
) -> tuple:
    key = workload_shape_key(batch)
    return (
        *key[:5],
        key[5] if attention_impl is None else str(attention_impl),
        key[6] if tp_size is None else int(tp_size),
        int(q),
    )


@dataclass(frozen=True)
class EmpiricalCandidateCost:
    shape_key: tuple
    q: int
    mode: str
    verify_ms: float
    round_ms: float
    useful_tokens: float
    samples: int


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
    empirical_candidates: tuple[EmpiricalCandidateCost, ...] = ()
    empirical_min_samples: int = 3
    # Zero until measured. An optimistic max(Draft, Target) incorrectly
    # assumes all Draft work can run while Target is busy.
    draft_overlap_fraction: float = 0.0

    def empirical_cost(
        self, batch: SpecStreamBatchState, q: int, mode: str
    ) -> EmpiricalCandidateCost | None:
        key = workload_shape_key(batch)
        return next(
            (
                point
                for point in self.empirical_candidates
                if point.shape_key == key
                and point.q == int(q)
                and point.mode == mode
                and point.samples >= self.empirical_min_samples
            ),
            None,
        )

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
    ordinary_observed = profile.empirical_cost(batch, q, "ordinary")
    parallel_observed = profile.empirical_cost(batch, q, "parallel")
    observed_verify = ordinary_observed or parallel_observed
    if observed_verify is not None:
        # A measured whole forward already contains H2D and stream waits.
        # Never add the analytic copy term on top of this observation.
        verify_ms = max(observed_verify.verify_ms, 0.0)
    draft_ms = 0.0 if q == 1 else profile.estimated_draft_rtt_ms(q)
    useful = acceptance.useful_tokens(q)
    rollback_cost = acceptance.rollback(q) * profile.repair_ms
    # Candidate-specific Draft RTT already includes transport.  Adding the
    # global network EMA here would double-count the ordinary wait.
    overlap_fraction = min(max(profile.draft_overlap_fraction, 0.0), 1.0)
    parallel = (
        draft_ms
        + verify_ms
        - overlap_fraction * min(draft_ms, verify_ms)
        + rollback_cost
    )
    ordinary = draft_ms + verify_ms + rollback_cost
    parallel_useful = ordinary_useful = useful
    if parallel_observed is not None and parallel_observed.round_ms > 0.0:
        parallel = parallel_observed.round_ms
        parallel_useful = max(parallel_observed.useful_tokens, 1.0)
    if ordinary_observed is not None and ordinary_observed.round_ms > 0.0:
        ordinary = ordinary_observed.round_ms
        ordinary_useful = max(ordinary_observed.useful_tokens, 1.0)
    if not math.isfinite(parallel) or not math.isfinite(ordinary):
        raise ValueError("SpecStream cost model produced a non-finite estimate")
    return CandidateCost(
        q, parallel / parallel_useful, ordinary / ordinary_useful, verify_ms, draft_ms
    )
