from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class TPStragglerSnapshot:
    samples: int = 0
    colocated_rank: int = 0
    rank_forward_ms: tuple[float, ...] = ()
    rank_collective_wait_ms: tuple[float, ...] = ()
    rank_skew_ms: float = 0.0
    target_slowdown: float = 0.0


class TPStragglerMonitor:
    """EMA-based TP critical-path monitor for colocated Drafter experiments.

    Full-forward GPU timings include any time spent waiting inside TP
    collectives.  Consequently ``rank_skew_ms`` is a conservative signal, not
    a replacement for an Nsight/NCCL breakdown.  ``target_slowdown`` compares
    the current slowest-rank EMA with the best observed slowest-rank EMA and is
    useful for an online safety gate even when collective wait is unavailable.
    """

    def __init__(
        self,
        *,
        tp_size: int,
        colocated_rank: int = 0,
        alpha: float = 0.2,
    ) -> None:
        if tp_size < 1:
            raise ValueError("tp_size must be positive")
        if not 0 <= colocated_rank < tp_size:
            raise ValueError("colocated_rank must be a valid TP rank")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.tp_size = int(tp_size)
        self.colocated_rank = int(colocated_rank)
        self.alpha = float(alpha)
        self._forward_ema: dict[int, float] = {}
        self._collective_ema: dict[int, float] = {}
        self._best_slowest_ms: float | None = None
        self._samples = 0
        self._local_sample: TPRankSample | None = None

    def record_local(self, sample: TPRankSample) -> None:
        self._local_sample = sample

    def local_sample(self) -> TPRankSample | None:
        return self._local_sample

    def observe(self, samples: Iterable[TPRankSample | None]) -> None:
        by_rank = {
            int(sample.rank): sample
            for sample in samples
            if sample is not None and float(sample.target_forward_ms) >= 0.0
        }
        if len(by_rank) != self.tp_size:
            return
        for rank in range(self.tp_size):
            sample = by_rank[rank]
            forward_ms = max(float(sample.target_forward_ms), 0.0)
            collective_ms = max(float(sample.collective_wait_ms), 0.0)
            if rank not in self._forward_ema:
                self._forward_ema[rank] = forward_ms
                self._collective_ema[rank] = collective_ms
            else:
                alpha = self.alpha
                self._forward_ema[rank] = (1.0 - alpha) * self._forward_ema[
                    rank
                ] + alpha * forward_ms
                self._collective_ema[rank] = (1.0 - alpha) * self._collective_ema[
                    rank
                ] + alpha * collective_ms
        self._samples += 1
        slowest_ms = max(self._forward_ema.values())
        if self._best_slowest_ms is None or slowest_ms < self._best_slowest_ms:
            self._best_slowest_ms = slowest_ms

    def snapshot(self) -> TPStragglerSnapshot:
        if len(self._forward_ema) != self.tp_size:
            return TPStragglerSnapshot(colocated_rank=self.colocated_rank)
        forward = tuple(self._forward_ema[rank] for rank in range(self.tp_size))
        collective = tuple(
            self._collective_ema.get(rank, 0.0) for rank in range(self.tp_size)
        )
        peers = [
            forward[rank] for rank in range(self.tp_size) if rank != self.colocated_rank
        ]
        peer_median = statistics.median(peers) if peers else forward[0]
        rank_skew_ms = forward[self.colocated_rank] - peer_median
        baseline = max(float(self._best_slowest_ms or 0.0), 1e-6)
        target_slowdown = max(0.0, max(forward) / baseline - 1.0)
        return TPStragglerSnapshot(
            samples=self._samples,
            colocated_rank=self.colocated_rank,
            rank_forward_ms=forward,
            rank_collective_wait_ms=collective,
            rank_skew_ms=rank_skew_ms,
            target_slowdown=target_slowdown,
        )
