from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import math
import statistics
from typing import Iterable


@dataclass(frozen=True)
class TPRankSample:
    rank: int
    round_id: int
    target_forward_ms: float
    collective_wait_ms: float = 0.0
    stream_attn_ms: float = 0.0
    exposed_copy_ms: float = 0.0
    # Empty/untagged samples are diagnostic only. False must mean that the
    # runtime excluded Draft work for the whole measured forward.
    shape_key: tuple = ()
    overlap_active: bool | None = None


@dataclass(frozen=True)
class TPStragglerSnapshot:
    samples: int = 0
    colocated_rank: int = 0
    rank_forward_ms: tuple[float, ...] = ()
    rank_collective_wait_ms: tuple[float, ...] = ()
    rank_skew_ms: float = 0.0
    target_slowdown: float = 0.0
    shape_key: tuple = ()
    round_id: int = -1
    baseline_ready: bool = False
    baseline_samples: int = 0
    baseline_forward_ms: tuple[float, ...] = ()
    excess_rank_skew_ms: float = 0.0
    overlap_active: bool | None = None


@dataclass
class _ShapeStats:
    samples: int = 0
    target_only_seen: int = 0
    baseline_samples: int = 0
    baseline: tuple[float, ...] = ()
    warm_values: list[tuple[float, ...]] = field(default_factory=list)
    forward: tuple[float, ...] = ()
    collective: tuple[float, ...] = ()
    overlap_active: bool | None = None
    round_id: int = -1


class TPStragglerMonitor:
    """Compare attributed overlap against warmed, same-shape Target-only work.

    Forward CUDA events include collective and transfer waits. A global best
    latency is therefore not a valid counterfactual when batch, q or residency
    changes. Each bounded shape bucket has its own warmed baseline, updated
    only by explicitly non-overlapping work; overlap can never train it.
    """

    def __init__(
        self,
        *,
        tp_size: int,
        colocated_rank: int = 0,
        alpha: float = 0.2,
        warmup_samples: int = 2,
        baseline_samples: int = 3,
        max_shapes: int = 128,
    ) -> None:
        if tp_size < 1:
            raise ValueError("tp_size must be positive")
        if not 0 <= colocated_rank < tp_size:
            raise ValueError("colocated_rank must be a valid TP rank")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if warmup_samples < 0 or baseline_samples < 1 or max_shapes < 1:
            raise ValueError("invalid baseline warmup or shape capacity")
        self.tp_size = int(tp_size)
        self.colocated_rank = int(colocated_rank)
        self.alpha = float(alpha)
        self.warmup_samples = int(warmup_samples)
        self.baseline_samples = int(baseline_samples)
        self.max_shapes = int(max_shapes)
        self._shapes: OrderedDict[tuple, _ShapeStats] = OrderedDict()
        self._last_shape: tuple = ()
        self._local_sample: TPRankSample | None = None

    def record_local(self, sample: TPRankSample) -> None:
        self._local_sample = sample

    def local_sample(self) -> TPRankSample | None:
        return self._local_sample

    def observe(self, samples: Iterable[TPRankSample | None]) -> None:
        by_rank = {
            int(sample.rank): sample
            for sample in samples
            if sample is not None
            and math.isfinite(float(sample.target_forward_ms))
            and float(sample.target_forward_ms) > 0.0
        }
        if set(by_rank) != set(range(self.tp_size)):
            return
        ordered = [by_rank[rank] for rank in range(self.tp_size)]
        if len({sample.round_id for sample in ordered}) != 1:
            return
        keys = {tuple(sample.shape_key) for sample in ordered}
        if len(keys) != 1:
            return
        key = keys.pop()
        state = self._shapes.setdefault(key, _ShapeStats())
        # Object all-gather may run repeatedly before another forward.
        if ordered[0].round_id <= state.round_id:
            return
        self._shapes.move_to_end(key)
        while len(self._shapes) > self.max_shapes:
            self._shapes.popitem(last=False)
        self._last_shape = key
        state.round_id = int(ordered[0].round_id)
        forward = tuple(float(sample.target_forward_ms) for sample in ordered)
        state.forward = forward
        state.collective = tuple(
            max(float(sample.collective_wait_ms), 0.0) for sample in ordered
        )
        state.samples += 1
        # Any True proves work on the colocated rank; False requires every
        # rank to explicitly exclude overlap. None covers issued but unacked
        # grants, which must not train a Target-only baseline.
        flags = [sample.overlap_active for sample in ordered]
        state.overlap_active = (
            True
            if any(flag is True for flag in flags)
            else False if all(flag is False for flag in flags) else None
        )
        if not key or state.overlap_active is not False:
            return
        state.target_only_seen += 1
        if state.target_only_seen <= self.warmup_samples:
            return
        state.baseline_samples += 1
        if not state.baseline:
            state.warm_values.append(forward)
            if len(state.warm_values) >= self.baseline_samples:
                state.baseline = tuple(
                    statistics.median(values[rank] for values in state.warm_values)
                    for rank in range(self.tp_size)
                )
                state.warm_values.clear()
        else:
            state.baseline = tuple(
                (1.0 - self.alpha) * old + self.alpha * current
                for old, current in zip(state.baseline, forward)
            )

    def _skew(self, values: tuple[float, ...]) -> float:
        peers = [v for rank, v in enumerate(values) if rank != self.colocated_rank]
        return values[self.colocated_rank] - (
            statistics.median(peers) if peers else values[0]
        )

    def snapshot(self, shape_key: tuple | None = None) -> TPStragglerSnapshot:
        key = self._last_shape if shape_key is None else tuple(shape_key)
        state = self._shapes.get(key)
        if state is None:
            return TPStragglerSnapshot(
                colocated_rank=self.colocated_rank, shape_key=key
            )
        skew = self._skew(state.forward)
        attributed = bool(state.baseline and state.overlap_active is True)
        slowdown = (
            max(0.0, max(state.forward) / max(max(state.baseline), 1e-6) - 1.0)
            if attributed
            else 0.0
        )
        return TPStragglerSnapshot(
            samples=state.samples,
            colocated_rank=self.colocated_rank,
            rank_forward_ms=state.forward,
            rank_collective_wait_ms=state.collective,
            rank_skew_ms=skew,
            target_slowdown=slowdown,
            shape_key=key,
            round_id=state.round_id,
            baseline_ready=bool(state.baseline),
            baseline_samples=state.baseline_samples,
            baseline_forward_ms=state.baseline,
            excess_rank_skew_ms=(
                max(0.0, skew - self._skew(state.baseline)) if attributed else 0.0
            ),
            overlap_active=state.overlap_active,
        )
