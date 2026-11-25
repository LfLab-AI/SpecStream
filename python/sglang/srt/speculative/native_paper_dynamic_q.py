"""Adapt the paper's I/O-aware q controller to the native local Draft worker."""

from collections import deque
from dataclasses import asdict, dataclass, field, replace
import math
from statistics import median

from sglang.srt.speculative.spectre.specstream.acceptance_tracker import AcceptanceTracker
from sglang.srt.speculative.spectre.specstream.controller import IOAwareController, SpecStreamDecision
from sglang.srt.speculative.spectre.specstream.cost_model import (
    SpecStreamBatchState,
    SpecStreamCostProfile,
    estimate_candidate_cost,
)


@dataclass
class _TimingBucket:
    controller: IOAwareController
    acceptance: AcceptanceTracker = field(default_factory=AcceptanceTracker)
    draft_ms: dict = field(default_factory=dict)
    timing_samples: dict = field(default_factory=dict)
    verification_correction_ms: dict = field(default_factory=dict)
    target_forward_ms: dict = field(default_factory=dict)
    round_overhead_ms: dict = field(default_factory=dict)
    h2d_gbps: float = 12.0
    decisions: int = 0
    pending_samples: dict = field(default_factory=dict)
    last_timing_round: int = -1
    resident_samples: dict = field(default_factory=dict)
    resident_cold_widths: set = field(default_factory=set)
    resident_width: int = 4
    resident_probe_index: int = 0
    resident_probe_width: int | None = None


class NativePaperWidth:
    """Select q with the existing paper controller and delayed CUDA timings.

    Proposal plus post-verify timing measures exposed Draft work, after any
    reuse of work started in the previous round. Its sum with Target time is
    the ordinary cost input. Cross-round overlap remains a scheduler decision.
    """

    widths = (2, 4, 6, 8)

    def __init__(self, switch_threshold=0.08, warmup_samples=4, timing_stride=16,
                 measured_resident=True):
        if warmup_samples < 1 or timing_stride < 1:
            raise ValueError("Timing warmup and stride must be positive")
        self.switch_threshold = float(switch_threshold)
        self.warmup_samples = int(warmup_samples)
        self.timing_stride = int(timing_stride)
        self.measured_resident = measured_resident
        self.buckets = {}
        self._last_description = {}

    @staticmethod
    def bucket_key(state):
        batch_size = int(state["batch_size"])
        context_tokens = int(state["context_tokens"])
        if batch_size < 1 or context_tokens < 0:
            raise ValueError("Paper q requires a nonempty batch and valid context")
        return (
            batch_size,
            (context_tokens + 4095) // 4096,
            bool(state["draft_cpu"]),
            bool(state["target_cpu"]),
        )

    def _bucket(self, state):
        key = self._timing_key(state)
        if key not in self.buckets:
            self.buckets[key] = _TimingBucket(
                IOAwareController(
                    self.widths,
                    self.switch_threshold,
                    draft_rtt_warmup_samples=self.warmup_samples,
                )
            )
        return self.buckets[key]

    @staticmethod
    def _batch_state(state):
        history_tokens = int(state["history_tokens"])
        history_bytes = int(state["history_bytes"])
        num_chunks = int(state["num_chunks"])
        if min(history_tokens, history_bytes, num_chunks) < 0:
            raise ValueError("Paper q history counters must be nonnegative")
        return SpecStreamBatchState(
            batch_size=int(state["batch_size"]),
            context_tokens=int(state["context_tokens"]),
            history_tokens=history_tokens,
            history_bytes=history_bytes,
            num_chunks=num_chunks,
        )

    @staticmethod
    def _profile(bucket):
        return SpecStreamCostProfile(
            h2d_gbps=bucket.h2d_gbps,
            draft_rtt_ms_by_q=tuple(sorted(bucket.draft_ms.items())),
            draft_rtt_samples_by_q=tuple(sorted(bucket.timing_samples.items())),
            verification_correction_ms_by_q=tuple(
                sorted(bucket.verification_correction_ms.items())
            ),
            # The measured post-verify stage already includes Draft repair.
            repair_ms=0.0,
        )

    def uses_measured_resident_timing(self, state):
        return self.measured_resident and not (state["draft_cpu"] or state["target_cpu"])

    def _timing_key(self, state):
        batch, context, draft_cpu, target_cpu = self.bucket_key(state)
        if self.uses_measured_resident_timing(state):
            # Nearby resident lengths share timing data. A 16K prompt growing
            # by a few hundred output tokens should not restart calibration.
            context = (int(state["context_tokens"]) + 4096) // 8192
        return batch, context, draft_cpu, target_cpu

    @staticmethod
    def _resident_cost(samples):
        if not samples:
            return None
        ceiling = 3.0 * median(elapsed for elapsed, _ in samples)
        usable = [(elapsed, tokens) for elapsed, tokens in samples if elapsed <= ceiling]
        return sum(elapsed for elapsed, _ in usable) / sum(tokens for _, tokens in usable)

    def _resident_decision(self, state, bucket):
        order = (4, 2, 6, 8)
        costs = {q: self._resident_cost(bucket.resident_samples.get(q)) for q in order}
        bucket.resident_probe_width = None
        current = bucket.resident_width
        ready = [q for q in order if len(bucket.resident_samples.get(q, ())) >= self.warmup_samples]
        if current in ready:
            best = min(ready, key=costs.__getitem__)
            if costs[best] < 0.98 * costs[current]:
                bucket.resident_width = current = best
        remaining = [
            q for q in order
            if q != current and len(bucket.resident_samples.get(q, ())) + bucket.pending_samples.get(q, 0)
            < self.warmup_samples + (q not in bucket.resident_cold_widths)
        ]
        q = current
        reason = "resident_measured_cost" if current in ready else "resident_timing_pending"
        if remaining and bucket.decisions % 8 == 0:
            q = min(remaining, key=lambda width: len(bucket.resident_samples.get(width, ()))
                    + bucket.pending_samples.get(width, 0) + (width in bucket.resident_cold_widths))
            reason = "resident_timing_warmup"
        elif len(ready) == len(order) and bucket.decisions % 64 == 0:
            alternatives = [q for q in order if q != current]
            q = alternatives[bucket.resident_probe_index % len(alternatives)]
            bucket.resident_probe_index += 1
            bucket.resident_probe_width = q
            reason = "resident_periodic_probe"
        batch_size = int(state["batch_size"])
        batch_cost = costs[q]
        # Keep the public per-request cost convention in the decision object.
        decision = SpecStreamDecision(q, "ordinary", reason,
                                      batch_size * batch_cost if batch_cost is not None else 0.0)
        self._last_description = {
            "controller": "paper", "resident_control": "measured",
            "bucket": list(self._timing_key(state)), "q": q, "reason": reason,
            "cost_mode": "ordinary", "scheduler_parallel_allowed": bool(state["parallel_allowed"]),
            "estimated_ms_per_batch_output_token": batch_cost,
            "resident_samples_by_q": {q: len(bucket.resident_samples.get(q, ())) for q in order},
            "candidate_costs": [
                {"q": q, "ordinary_ms_per_batch_output_token": costs[q]} for q in sorted(order)
            ],
        }
        return decision

    def choose(self, state):
        bucket = self._bucket(state)
        bucket.decisions += 1
        if self.uses_measured_resident_timing(state):
            return self._resident_decision(state, bucket)
        batch_state = self._batch_state(state)
        profile = self._profile(bucket)
        acceptance = bucket.acceptance.snapshot(self.widths)
        decision = bucket.controller.choose(
            batch_state,
            profile,
            acceptance,
            allow_ordinary=True,
            force_ordinary=True,
        )
        if decision.q not in self.widths:
            raise RuntimeError("Paper q returned a width outside the native candidates")
        batch_size = int(state["batch_size"])
        candidate_costs = []
        for candidate in decision.candidate_costs:
            row = asdict(candidate)
            row.update(
                expected_useful_tokens_per_request=acceptance.useful_tokens(candidate.q),
                ordinary_ms_per_batch_output_token=(
                    candidate.ordinary_ms_per_useful_token / batch_size
                ),
                parallel_ms_per_batch_output_token=(
                    candidate.parallel_ms_per_useful_token / batch_size
                ),
            )
            candidate_costs.append(row)
        self._last_description = {
            "controller": "paper",
            "bucket": list(self.bucket_key(state)),
            "q": decision.q,
            "reason": decision.reason,
            "cost_mode": decision.mode,
            "scheduler_parallel_allowed": bool(state["parallel_allowed"]),
            "draft_timing_basis": "exposed_proposal_plus_post_verify",
            "estimated_ms_per_useful_token": decision.estimated_cost,
            "cost_denominator": "mean_useful_tokens_per_request",
            "expected_useful_tokens_per_request": acceptance.useful_tokens(decision.q),
            "estimated_ms_per_batch_output_token": decision.estimated_cost / batch_size,
            "candidate_costs": candidate_costs,
            "target_forward_ms_by_q": dict(bucket.target_forward_ms),
            "round_overhead_ms_by_q": dict(bucket.round_overhead_ms),
            "acceptance": asdict(acceptance),
            "profile": asdict(profile),
        }
        return decision

    def record_acceptance(self, state, q, accepted_lengths):
        if q not in self.widths:
            raise ValueError("Unsupported native paper width")
        values = list(accepted_lengths)
        if not values:
            return False
        try:
            valid = all(
                math.isfinite(float(value)) and int(value) == value and 0 <= value < q
                for value in values
            )
        except (TypeError, ValueError, OverflowError):
            return False
        if not valid:
            return False
        self._bucket(state).acceptance.update(q, values)
        return True

    def needs_timing(self, state, q):
        if q not in self.widths:
            raise ValueError("Unsupported native paper width")
        bucket = self._bucket(state)
        pending = bucket.pending_samples.get(q, 0)
        if self.uses_measured_resident_timing(state):
            samples = len(bucket.resident_samples.get(q, ()))
            target = self.warmup_samples + (q not in bucket.resident_cold_widths)
            collect = (samples + pending < target or bucket.resident_probe_width == q
                       or (pending == 0 and bucket.decisions % self.timing_stride == 0))
            if collect:
                bucket.pending_samples[q] = pending + 1
            return collect
        samples = bucket.timing_samples.get(q, 0)
        collect = (
            samples + pending < self.warmup_samples
            or (pending == 0 and bucket.decisions % self.timing_stride == 0)
        )
        if collect:
            bucket.pending_samples[q] = pending + 1
        return collect

    def is_warming_up(self, state):
        bucket = self._bucket(state)
        if self.uses_measured_resident_timing(state):
            return any(len(bucket.resident_samples.get(q, ())) < self.warmup_samples for q in self.widths)
        samples = bucket.timing_samples
        return any(samples.get(q, 0) < self.warmup_samples for q in self.widths)

    @staticmethod
    def _ema(values, key, value):
        previous = values.get(key)
        values[key] = value if previous is None else 0.8 * previous + 0.2 * value

    def observe_timing(self, sample):
        state = sample["state"]
        q = int(sample["q"])
        if q not in self.widths:
            raise ValueError("Unsupported native paper timing width")
        bucket = self._bucket(state)
        round_id = sample.get("round_id")
        if round_id is not None and int(round_id) <= bucket.last_timing_round:
            return False
        pending = bucket.pending_samples.get(q, 0)
        bucket.pending_samples[q] = max(pending - 1, 0)
        actual_state = sample.get("actual_state")
        if sample.get("placement_changed", False) or (
            actual_state is not None and any(
                bool(state[key]) != bool(actual_state[key])
                for key in ("draft_cpu", "target_cpu")
            )
        ):
            return False
        if int(sample.get("batch_size", state["batch_size"])) != int(state["batch_size"]):
            return False
        if self.uses_measured_resident_timing(state):
            try:
                elapsed = float(sample.get("round_host_ms", sample.get("round_wall_ms")))
                output = float(sample["committed_output_tokens"])
            except (KeyError, TypeError, ValueError, OverflowError):
                return False
            if not (math.isfinite(elapsed) and elapsed > 0 and math.isfinite(output) and output > 0):
                return False
            if q in bucket.resident_cold_widths:
                bucket.resident_samples.setdefault(q, deque(maxlen=max(16, self.warmup_samples))).append((elapsed, output))
            else:
                # The first use of a width may include graph/backend/allocator initialization.
                bucket.resident_cold_widths.add(q)
            if round_id is not None:
                bucket.last_timing_round = int(round_id)
            return True
        try:
            proposal_ms = float(sample["draft_proposal_ms"])
            post_verify_ms = float(sample["post_verify_draft_ms"])
            target_ms = float(sample["target_forward_cuda_ms"])
            h2d_ms = float(sample.get("target_h2d_event_ms", 0.0))
            h2d_bytes = int(sample.get("target_h2d_bytes", 0))
            output_tokens = float(sample.get("committed_output_tokens", 1))
            round_cuda_ms = sample.get("round_elapsed_cuda_ms")
            if round_cuda_ms is not None:
                round_cuda_ms = float(round_cuda_ms)
        except (TypeError, ValueError, OverflowError):
            return False
        draft_ms = proposal_ms + post_verify_ms
        if any(
            not math.isfinite(value) or value < 0
            for value in (proposal_ms, post_verify_ms, draft_ms, target_ms, h2d_ms)
        ) or target_ms == 0 or h2d_bytes < 0 or (
            not math.isfinite(output_tokens) or output_tokens <= 0
        ) or (
            round_cuda_ms is not None
            and (not math.isfinite(round_cuda_ms) or round_cuda_ms <= 0)
        ):
            return False
        overhead_ms = (
            max(0.0, round_cuda_ms - draft_ms - target_ms)
            if round_cuda_ms is not None else 0.0
        )
        calibrated_target_ms = target_ms + overhead_ms

        next_h2d_gbps = bucket.h2d_gbps
        if h2d_bytes > 0 and h2d_ms > 0:
            measured_gbps = h2d_bytes / (h2d_ms * 1e6)
            if not math.isfinite(measured_gbps) or measured_gbps <= 0:
                return False
            next_h2d_gbps = (
                measured_gbps
                if not bucket.timing_samples
                else 0.8 * bucket.h2d_gbps + 0.2 * measured_gbps
            )

        base_profile = replace(
            self._profile(bucket),
            h2d_gbps=next_h2d_gbps,
            verification_correction_ms_by_q=(),
        )
        predicted_target_ms = estimate_candidate_cost(
            q,
            self._batch_state(state),
            base_profile,
            bucket.acceptance.snapshot(self.widths),
        ).verify_ms
        self._ema(
            bucket.verification_correction_ms, q,
            calibrated_target_ms - predicted_target_ms,
        )
        self._ema(bucket.target_forward_ms, q, target_ms)
        self._ema(bucket.round_overhead_ms, q, overhead_ms)
        bucket.h2d_gbps = next_h2d_gbps
        self._ema(bucket.draft_ms, q, draft_ms)
        bucket.timing_samples[q] = bucket.timing_samples.get(q, 0) + 1
        if round_id is not None:
            bucket.last_timing_round = int(round_id)
        return True

    def describe_decision(self):
        return self._last_description.copy()
