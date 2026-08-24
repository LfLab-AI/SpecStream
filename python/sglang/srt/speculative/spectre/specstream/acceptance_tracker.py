from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class AcceptanceSnapshot:
    expected_useful_tokens: dict[int, float]
    rollback_probability: dict[int, float]
    samples: dict[int, int]

    def useful_tokens(self, q: int) -> float:
        return max(1.0, min(float(q), self.expected_useful_tokens.get(q, 1.0)))

    def rollback(self, q: int) -> float:
        return min(1.0, max(0.0, self.rollback_probability.get(q, 0.0)))


@dataclass
class AcceptanceTracker:
    alpha: float = 0.2
    _useful: dict[int, float] = field(default_factory=dict)
    _rollback: dict[int, float] = field(default_factory=dict)
    _samples: dict[int, int] = field(default_factory=dict)

    def update(self, q: int, accepted_lengths: Iterable[int]) -> None:
        values = [max(1, min(q, int(value) + 1)) for value in accepted_lengths]
        if not values:
            return
        useful = sum(values) / len(values)
        rollback = sum(value < q for value in values) / len(values)
        if q in self._useful:
            a = self.alpha
            self._useful[q] = (1.0 - a) * self._useful[q] + a * useful
            self._rollback[q] = (1.0 - a) * self._rollback[q] + a * rollback
        else:
            self._useful[q] = useful
            self._rollback[q] = rollback
        self._samples[q] = self._samples.get(q, 0) + len(values)

    def snapshot(self, candidates: Iterable[int]) -> AcceptanceSnapshot:
        expected = {}
        rollback = {}
        samples = {}
        for q in candidates:
            # Conservative cold-start prior: roughly half of the speculative
            # block is useful, while q=1 always advances one token.
            expected[q] = self._useful.get(q, 1.0 if q == 1 else 1.0 + 0.5 * (q - 1))
            rollback[q] = self._rollback.get(q, 0.0 if q == 1 else 0.5)
            samples[q] = self._samples.get(q, 0)
        return AcceptanceSnapshot(expected, rollback, samples)
