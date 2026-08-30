#!/usr/bin/env python3
"""Extract one aggregate interference sample from two SpecStream CSV runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


def context_bucket(context_tokens: int) -> str:
    context_tokens = max(int(context_tokens), 0)
    for limit, label in (
        (2048, "2k"),
        (4096, "4k"),
        (8192, "8k"),
        (16384, "16k"),
        (32768, "32k"),
        (65536, "64k"),
    ):
        if context_tokens <= limit:
            return label
    return "64k+"


def target_q(target_shape: str) -> int | None:
    match = re.search(r"(?:^|_)q(\d+)(?:_|$)", target_shape)
    return int(match.group(1)) if match else None


def row_batch_size(row: dict[str, str]) -> int:
    if row.get("batch_size"):
        return int(row["batch_size"])
    # Compatibility with profiles written before batch_size was explicit.
    request_ids = [value for value in row.get("rid", "").split("|") if value]
    return max(len(request_ids), 1)


def row_matches_shape(
    row: dict[str, str],
    *,
    target_shape: str,
    draft_bs: int,
    draft_ctx_bucket: str,
    draft_tpcs: int | None = None,
    target_phase: str | None = None,
) -> bool:
    expected_q = target_q(target_shape)
    if expected_q is not None and int(float(row.get("q", 0) or 0)) != expected_q:
        return False
    if row_batch_size(row) != draft_bs:
        return False
    if context_bucket(int(float(row.get("context_tokens", 0) or 0))) != (
        draft_ctx_bucket.lower()
    ):
        return False
    if draft_tpcs is not None:
        tpc_low = int(float(row.get("draft_tpc_low", -1) or -1))
        tpc_high = int(float(row.get("draft_tpc_high", -1) or -1))
        if tpc_low < 0 or tpc_high - tpc_low != draft_tpcs:
            return False
    if target_phase is not None and row.get("target_phase") != target_phase:
        return False
    if target_phase == "history_h2d":
        if int(float(row.get("history_len", 0) or 0)) <= 0:
            return False
        if int(float(row.get("h2d_ops", 0) or 0)) <= 0:
            return False
        if float(row.get("exposed_copy_ms", 0) or 0) <= 0:
            return False
    return True


def positive_values(
    path: Path,
    field: str,
    *,
    target_shape: str,
    draft_bs: int,
    draft_ctx_bucket: str,
    draft_tpcs: int | None = None,
    target_phase: str | None = None,
) -> list[float]:
    with path.open(encoding="utf-8", newline="") as handle:
        values = [
            float(row[field])
            for row in csv.DictReader(handle)
            if row_matches_shape(
                row,
                target_shape=target_shape,
                draft_bs=draft_bs,
                draft_ctx_bucket=draft_ctx_bucket,
                draft_tpcs=draft_tpcs,
                target_phase=target_phase,
            )
            and row.get(field)
            and float(row[field]) > 0
        ]
    if not values:
        raise ValueError(
            f"{path} has no positive {field} samples for "
            f"shape={target_shape}, draft_bs={draft_bs}, "
            f"draft_ctx_bucket={draft_ctx_bucket}"
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--overlap-profile", type=Path, required=True)
    parser.add_argument("--target-shape", required=True)
    parser.add_argument("--draft-bs", type=int, required=True)
    parser.add_argument("--draft-ctx-bucket", required=True)
    parser.add_argument("--draft-tpcs", type=int, required=True)
    parser.add_argument(
        "--slack-source",
        choices=("target_forward", "history_h2d"),
        default="target_forward",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    row = {
        "target_shape": args.target_shape,
        "draft_bs": args.draft_bs,
        "draft_ctx_bucket": args.draft_ctx_bucket,
        "draft_tpcs": args.draft_tpcs,
        "slack_source": args.slack_source,
        "draft_step_ms": statistics.median(
            positive_values(
                args.overlap_profile,
                "draft_step_ms",
                target_shape=args.target_shape,
                draft_bs=args.draft_bs,
                draft_ctx_bucket=args.draft_ctx_bucket,
                draft_tpcs=args.draft_tpcs,
                target_phase=(
                    "history_h2d" if args.slack_source == "history_h2d" else None
                ),
            )
        ),
        "target_latency_ms": statistics.median(
            positive_values(
                args.overlap_profile,
                "target_forward_ms",
                target_shape=args.target_shape,
                draft_bs=args.draft_bs,
                draft_ctx_bucket=args.draft_ctx_bucket,
                draft_tpcs=args.draft_tpcs,
                target_phase=(
                    "history_h2d" if args.slack_source == "history_h2d" else None
                ),
            )
        ),
        "target_baseline_ms": statistics.median(
            positive_values(
                args.baseline_profile,
                "target_forward_ms",
                target_shape=args.target_shape,
                draft_bs=args.draft_bs,
                draft_ctx_bucket=args.draft_ctx_bucket,
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
