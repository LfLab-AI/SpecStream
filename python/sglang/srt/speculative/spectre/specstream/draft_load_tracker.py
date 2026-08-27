from __future__ import annotations

from collections import deque
from dataclasses import dataclass


def _percentile(values: deque[float] | deque[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, int(quantile * len(ordered) + 0.999999) - 1)
    return ordered[min(index, len(ordered) - 1)]


@dataclass(frozen=True)
class DraftLoadSnapshot:
    samples: int = 0
    rtt_ema_ms: float = 0.0
    rtt_p95_ms: float = 0.0
    pressure_p95: float = 0.0
    timeout_rate: float = 0.0
    missing_ratio_ema: float = 0.0
    pending_p95: int = 0
    reject_rate: float = 0.0


class DraftLoadTracker:
    """Bounded online view of remote-Drafter pressure.

    Target only observes end-to-end request/response time, the number of
    requests waiting in the current batch, and delivery failures.  These are
    deliberately treated as serving-level pressure signals rather than as a
    claim about the Drafter's internal CUDA queue.
    """

    def __init__(self, *, window_size: int = 32, alpha: float = 0.2) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._rtt_ms: deque[float] = deque(maxlen=window_size)
        self._pressure: deque[float] = deque(maxlen=window_size)
        self._timeout: deque[int] = deque(maxlen=window_size)
        self._missing_ratio: deque[float] = deque(maxlen=window_size)
        self._pending: deque[int] = deque(maxlen=window_size)
        self._reject: deque[int] = deque(maxlen=window_size)
        self._rtt_ema_ms = 0.0
        self._missing_ratio_ema = 0.0

    def record_result(
        self,
        *,
        elapsed_ms: float,
        timeout_ms: float,
        missing_count: int,
        total_count: int,
    ) -> None:
        elapsed_ms = max(float(elapsed_ms), 0.0)
        timeout_ms = max(float(timeout_ms), 1e-6)
        missing_count = max(int(missing_count), 0)
        total_count = max(int(total_count), 0)
        missing_count = min(missing_count, total_count)
        missing_ratio = missing_count / max(total_count, 1)

        self._rtt_ms.append(elapsed_ms)
        self._pressure.append(elapsed_ms / timeout_ms)
        self._timeout.append(int(missing_count > 0))
        self._missing_ratio.append(missing_ratio)
        self._pending.append(total_count)
        self._reject.append(0)
        if len(self._rtt_ms) == 1:
            self._rtt_ema_ms = elapsed_ms
            self._missing_ratio_ema = missing_ratio
        else:
            alpha = self.alpha
            self._rtt_ema_ms = (1.0 - alpha) * self._rtt_ema_ms + alpha * elapsed_ms
            self._missing_ratio_ema = (
                1.0 - alpha
            ) * self._missing_ratio_ema + alpha * missing_ratio

    def record_reject(self) -> None:
        # A REJECT can arrive independently of a receive timeout.  Keep the
        # sample window aligned without fabricating RTT or pending values.
        self._reject.append(1)

    def snapshot(self) -> DraftLoadSnapshot:
        samples = len(self._rtt_ms)
        return DraftLoadSnapshot(
            samples=samples,
            rtt_ema_ms=self._rtt_ema_ms,
            rtt_p95_ms=_percentile(self._rtt_ms, 0.95),
            pressure_p95=_percentile(self._pressure, 0.95),
            timeout_rate=(
                (sum(self._timeout) / len(self._timeout)) if self._timeout else 0.0
            ),
            missing_ratio_ema=self._missing_ratio_ema,
            pending_p95=int(_percentile(self._pending, 0.95)),
            reject_rate=(
                (sum(self._reject) / len(self._reject)) if self._reject else 0.0
            ),
        )
