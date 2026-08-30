#!/usr/bin/env python3
"""Build a fail-closed SpecStream TPC profile from measured JSONL samples."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


REQUIRED = {
    "target_shape",
    "draft_bs",
    "draft_ctx_bucket",
    "draft_tpcs",
    "draft_step_ms",
    "target_latency_ms",
    "target_baseline_ms",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path, help="Measured JSONL samples")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--total-tpcs", type=int, required=True)
    parser.add_argument("--min-repetitions", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = defaultdict(list)
    for line_number, line in enumerate(
        args.samples.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = REQUIRED - set(row)
        if missing:
            raise ValueError(f"line {line_number} missing fields: {sorted(missing)}")
        key = (
            str(row["target_shape"]),
            int(row["draft_bs"]),
            str(row["draft_ctx_bucket"]),
            int(row["draft_tpcs"]),
            str(row.get("slack_source", "target_forward")),
        )
        grouped[key].append(row)

    entries = []
    for key, rows in sorted(grouped.items()):
        if len(rows) < args.min_repetitions:
            raise ValueError(
                f"{key} has {len(rows)} repetitions; need {args.min_repetitions}"
            )
        draft_step_ms = statistics.median(float(r["draft_step_ms"]) for r in rows)
        target_latency_ms = statistics.median(
            float(r["target_latency_ms"]) for r in rows
        )
        target_baseline_ms = statistics.median(
            float(r["target_baseline_ms"]) for r in rows
        )
        if (
            not all(
                math.isfinite(value)
                for value in (draft_step_ms, target_latency_ms, target_baseline_ms)
            )
            or draft_step_ms <= 0
            or target_latency_ms <= 0
            or target_baseline_ms <= 0
        ):
            raise ValueError(f"{key} contains a non-positive latency")
        entries.append(
            {
                "target_shape": key[0],
                "draft_bs": key[1],
                "draft_ctx_bucket": key[2],
                "draft_tpcs": key[3],
                "slack_source": key[4],
                "draft_step_ms": draft_step_ms,
                "target_slowdown": max(
                    0.0, target_latency_ms / target_baseline_ms - 1.0
                ),
                "target_baseline_ms": target_baseline_ms,
                "repetitions": len(rows),
            }
        )

    if not entries:
        raise ValueError("no measurements found")
    if args.total_tpcs < max(entry["draft_tpcs"] for entry in entries):
        raise ValueError("--total-tpcs is smaller than a measured draft_tpcs")

    payload = {
        "schema_version": 1,
        "gpu": args.gpu,
        "draft_model": args.draft_model,
        "target_model": args.target_model,
        "total_tpcs": args.total_tpcs,
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
