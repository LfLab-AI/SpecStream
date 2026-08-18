#!/usr/bin/env python3
"""Aggregate SpecStream per-round CSV profiles with only the Python stdlib."""

from __future__ import annotations

import argparse
import csv
import glob
from collections import Counter
from pathlib import Path


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0) or 0)
    except ValueError:
        return 0.0


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * ratio + 0.5))
    return ordered[index]


def summarize(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("timestamp") not in (None, "", "timestamp")
        ]
    h2d_bytes = sum(number(row, "h2d_bytes") for row in rows)
    h2d_ms = sum(number(row, "h2d_ms") for row in rows)
    modes = Counter(row.get("mode", "") for row in rows)
    qs = Counter(str(int(number(row, "q"))) for row in rows)
    round_ms = [number(row, "round_ms") for row in rows]
    network_wait_ms = [number(row, "network_wait_ms") for row in rows]
    cohort_sizes = [number(row, "cohort_size") for row in rows]
    fallback_reasons = Counter(
        row.get("fallback_reason", "")
        for row in rows
        if str(row.get("fallback", "")).lower() in ("true", "1")
    )
    return {
        "file": str(path),
        "rows": len(rows),
        "stream_rows": sum(number(row, "h2d_bytes") > 0 for row in rows),
        "q_dist": ",".join(f"{key}:{qs[key]}" for key in sorted(qs, key=int)),
        "mode_dist": ",".join(f"{key}:{value}" for key, value in sorted(modes.items())),
        "max_history": int(max((number(row, "history_len") for row in rows), default=0)),
        "h2d_gib": h2d_bytes / (1024**3),
        "h2d_ops": int(sum(number(row, "h2d_ops") for row in rows)),
        "effective_h2d_gbps": h2d_bytes / h2d_ms / 1e6 if h2d_ms else 0.0,
        "stream_attn_s": sum(number(row, "stream_attn_ms") for row in rows) / 1000,
        "target_forward_s": sum(number(row, "target_forward_ms") for row in rows)
        / 1000,
        "mean_round_ms": sum(round_ms) / len(round_ms) if round_ms else 0.0,
        "p95_round_ms": percentile(round_ms, 0.95),
        "mean_network_wait_ms": (
            sum(network_wait_ms) / len(network_wait_ms) if network_wait_ms else 0.0
        ),
        "p95_network_wait_ms": percentile(network_wait_ms, 0.95),
        "accepted_tokens": int(sum(number(row, "accepted_tokens") for row in rows)),
        "mean_cohort": sum(cohort_sizes) / len(cohort_sizes)
        if cohort_sizes
        else 0.0,
        "max_cohort": int(max(cohort_sizes, default=0)),
        "fallback_rows": sum(
            str(row.get("fallback", "")).lower() in ("true", "1") for row in rows
        ),
        "missing_drafts": int(
            sum(number(row, "missing_draft_count") for row in rows)
        ),
        "fallback_reasons": ",".join(
            f"{key}:{value}" for key, value in sorted(fallback_reasons.items())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="CSV files or glob patterns")
    args = parser.parse_args()
    paths: list[Path] = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        paths.extend(Path(item) for item in (matches or [pattern]))

    columns = (
        "file",
        "rows",
        "stream_rows",
        "q_dist",
        "mode_dist",
        "max_history",
        "h2d_gib",
        "h2d_ops",
        "effective_h2d_gbps",
        "stream_attn_s",
        "target_forward_s",
        "mean_round_ms",
        "p95_round_ms",
        "mean_network_wait_ms",
        "p95_network_wait_ms",
        "accepted_tokens",
        "mean_cohort",
        "max_cohort",
        "fallback_rows",
        "missing_drafts",
        "fallback_reasons",
    )
    print("\t".join(columns))
    for path in sorted(dict.fromkeys(paths)):
        if not path.is_file():
            print(f"warning: missing file: {path}")
            continue
        result = summarize(path)
        values = []
        for column in columns:
            value = result[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        print("\t".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
