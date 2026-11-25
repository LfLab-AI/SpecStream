"""Admit a speculative prefix only when at least one new candidate should fit."""

from collections import defaultdict
import json
import time


def batch_bucket(batch_size):
    return next((size for size in (4, 8, 16, 24, 31, 32) if batch_size <= size), batch_size)


def key_for(batch_size, width):
    return f"{batch_bucket(batch_size)}:{width}"


class LookaheadAdmission:
    def __init__(self, path):
        with open(path, encoding="utf-8") as stream:
            self.profiles = json.load(stream)["buckets"]
        self.observed = {}
        self.over_budget = defaultdict(int)
        self.cooldown = defaultdict(int)

    def decide(self, batch_size, q, target_host_tokens):
        key = key_for(batch_size, q)
        if not target_host_tokens:
            return {"key": key, "skip": "no_target_kv_spill"}
        profile = self.profiles.get(key)
        if profile is None:
            return {"key": key, "skip": "no_window_profile"}
        if batch_size != profile.get("required_batch_size", batch_size):
            return {"key": key, "skip": "profile_batch_mismatch"}
        if target_host_tokens < profile.get("min_target_host_tokens", 0):
            return {"key": key, "skip": "below_calibrated_target_history"}
        if self.cooldown[key]:
            self.cooldown[key] -= 1
            return {"key": key, "skip": "h2d_slowdown_backoff"}

        initial = profile["initial_ms"]
        candidate = profile["candidate_ms"]
        measured = self.observed.get(key)
        if measured:
            initial = max(initial, measured["initial_ms"])
            candidate = max(candidate, measured["candidate_ms"])
        # Prior window is an estimate; the GPU completion event is still the
        # authority for stopping. Leave room for synchronization and jitter.
        window = profile["window_ms"]
        guard = max(75.0, 0.08 * window)
        return {
            "key": key, "initial_ms": initial, "candidate_ms": candidate,
            "window_ms": window, "guard_ms": guard,
            "skip": "first_candidate_would_overrun" if initial + candidate + guard > window else None,
        }

    @staticmethod
    def next_fits(decision, target_start_ns):
        elapsed = (time.monotonic_ns() - target_start_ns) / 1e6
        return elapsed + decision["candidate_ms"] + decision["guard_ms"] <= decision["window_ms"]

    def observe_steps(self, decision, phases):
        key = decision["key"]
        profile = self.profiles[key]
        stats = self.observed.setdefault(key, {
            name: profile[name] for name in ("initial_ms", "candidate_ms")
        })
        for name, selected in (("initial_ms", phases[:1]), ("candidate_ms", phases[1:])):
            if selected:
                elapsed = max((step["end_ns"] - step["start_ns"]) / 1e6 for step in selected)
                stats[name] = 0.8 * stats[name] + 0.2 * elapsed

    def observe_transfer(self, row):
        if not row.get("lookahead_admitted") or row["target_profiled_h2d_bytes"] < 4 * 2**30:
            return
        key = key_for(row["batch_size"], row["q"])
        baseline = self.profiles.get(key, {}).get("h2d_ms_per_gib")
        if baseline is None:
            return
        actual = row["target_h2d_dma_ms"] * 2**30 / row["target_profiled_h2d_bytes"]
        if actual <= baseline * 1.12:
            self.over_budget[key] = 0
            return
        self.over_budget[key] += 1
        if self.over_budget[key] >= 2:
            self.over_budget[key] = 0
            self.cooldown[key] = 3
