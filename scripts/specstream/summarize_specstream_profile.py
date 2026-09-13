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


def summarize(path: Path, *, h2d_gbps: float = 0.0) -> dict[str, object]:
    with path.open(encoding="utf-8", newline="") as handle:
        parsed_rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("timestamp") not in (None, "", "timestamp")
        ]
    schema_mismatch_rows = sum(bool(row.get(None)) for row in parsed_rows)
    rows = [row for row in parsed_rows if not row.get(None)]
    h2d_bytes = sum(number(row, "h2d_bytes") for row in rows)
    h2d_ms = sum(number(row, "h2d_ms") for row in rows)
    modes = Counter(row.get("mode", "") for row in rows)
    coexec_modes = Counter(row.get("coexec_mode", "") for row in rows)
    qs = Counter(str(int(number(row, "q"))) for row in rows)
    round_ms = [number(row, "round_ms") for row in rows]
    network_wait_ms = [number(row, "network_wait_ms") for row in rows]
    cohort_sizes = [number(row, "cohort_size") for row in rows]
    draft_rtt_p95 = [number(row, "draft_rtt_p95_ms") for row in rows]
    timeout_rates = [number(row, "draft_timeout_rate") for row in rows]
    rank_skews = [number(row, "tp_rank_skew_ms") for row in rows]
    target_slowdowns = [number(row, "tp_target_slowdown") for row in rows]
    accepted_tokens = sum(number(row, "accepted_tokens") for row in rows)
    h2d_ops = sum(number(row, "h2d_ops") for row in rows)
    stream_attn_ops = sum(number(row, "stream_attn_ops") for row in rows)
    target_forward_ms = sum(number(row, "target_forward_ms") for row in rows)
    copy_floor_ms = h2d_bytes / (h2d_gbps * 1e6) if h2d_gbps > 0 else 0.0
    fallback_reasons = Counter(
        row.get("fallback_reason", "")
        for row in rows
        if str(row.get("fallback", "")).lower() in ("true", "1")
    )
    return {
        "file": str(path),
        "rows": len(rows),
        "schema_mismatch_rows": schema_mismatch_rows,
        "stream_rows": sum(number(row, "h2d_bytes") > 0 for row in rows),
        "q_dist": ",".join(f"{key}:{qs[key]}" for key in sorted(qs, key=int)),
        "mode_dist": ",".join(f"{key}:{value}" for key, value in sorted(modes.items())),
        "coexec_dist": ",".join(
            f"{key}:{value}" for key, value in sorted(coexec_modes.items()) if key
        ),
        "max_history": int(
            max((number(row, "history_len") for row in rows), default=0)
        ),
        "h2d_gib": h2d_bytes / (1024**3),
        "h2d_ops": int(h2d_ops),
        "h2d_ops_per_accepted": h2d_ops / accepted_tokens if accepted_tokens else 0.0,
        "h2d_mib_per_accepted": (
            h2d_bytes / (1024**2) / accepted_tokens if accepted_tokens else 0.0
        ),
        "effective_h2d_gbps": h2d_bytes / h2d_ms / 1e6 if h2d_ms else 0.0,
        "copy_floor_ms_per_accepted": (
            copy_floor_ms / accepted_tokens if accepted_tokens else 0.0
        ),
        "copy_floor_target_fraction": (
            copy_floor_ms / target_forward_ms if target_forward_ms else 0.0
        ),
        "stream_attn_s": sum(number(row, "stream_attn_ms") for row in rows) / 1000,
        "stream_attn_ops": int(stream_attn_ops),
        "stream_attn_ops_per_accepted": (
            stream_attn_ops / accepted_tokens if accepted_tokens else 0.0
        ),
        "target_forward_s": target_forward_ms / 1000,
        "mean_round_ms": sum(round_ms) / len(round_ms) if round_ms else 0.0,
        "p95_round_ms": percentile(round_ms, 0.95),
        "mean_network_wait_ms": (
            sum(network_wait_ms) / len(network_wait_ms) if network_wait_ms else 0.0
        ),
        "p95_network_wait_ms": percentile(network_wait_ms, 0.95),
        "max_draft_rtt_p95_ms": max(draft_rtt_p95, default=0.0),
        "draft_rtt_by_q": rows[-1].get("draft_rtt_by_q", "") if rows else "",
        "controller_candidate_costs": (
            rows[-1].get("controller_candidate_costs", "") if rows else ""
        ),
        "max_draft_timeout_rate": max(timeout_rates, default=0.0),
        "max_tp_rank_skew_ms": max(rank_skews, default=0.0),
        "max_tp_target_slowdown": max(target_slowdowns, default=0.0),
        "accepted_tokens": int(accepted_tokens),
        "mean_cohort": sum(cohort_sizes) / len(cohort_sizes) if cohort_sizes else 0.0,
        "max_cohort": int(max(cohort_sizes, default=0)),
        "fallback_rows": sum(
            str(row.get("fallback", "")).lower() in ("true", "1") for row in rows
        ),
        "missing_drafts": int(sum(number(row, "missing_draft_count") for row in rows)),
        "fallback_reasons": ",".join(
            f"{key}:{value}" for key, value in sorted(fallback_reasons.items())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="CSV files or glob patterns")
    parser.add_argument(
        "--h2d-gbps",
        type=float,
        default=0.0,
        help=(
            "Measured effective pinned H2D bandwidth. When set, report the "
            "unavoidable serial-copy lower bound; do not use link-rate marketing GB/s."
        ),
    )
    args = parser.parse_args()
    if args.h2d_gbps < 0:
        parser.error("--h2d-gbps cannot be negative")
    paths: list[Path] = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        paths.extend(Path(item) for item in (matches or [pattern]))

    columns = (
        "file",
        "rows",
        "schema_mismatch_rows",
        "stream_rows",
        "q_dist",
        "mode_dist",
        "coexec_dist",
        "max_history",
        "h2d_gib",
        "h2d_ops",
        "h2d_ops_per_accepted",
        "h2d_mib_per_accepted",
        "effective_h2d_gbps",
        "copy_floor_ms_per_accepted",
        "copy_floor_target_fraction",
        "stream_attn_s",
        "stream_attn_ops",
        "stream_attn_ops_per_accepted",
        "target_forward_s",
        "mean_round_ms",
        "p95_round_ms",
        "mean_network_wait_ms",
        "p95_network_wait_ms",
        "max_draft_rtt_p95_ms",
        "draft_rtt_by_q",
        "controller_candidate_costs",
        "max_draft_timeout_rate",
        "max_tp_rank_skew_ms",
        "max_tp_target_slowdown",
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
        result = summarize(path, h2d_gbps=args.h2d_gbps)
        values = []
        for column in columns:
            value = result[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        print("\t".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
