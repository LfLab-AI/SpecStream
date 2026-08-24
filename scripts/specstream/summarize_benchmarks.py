#!/usr/bin/env python3
"""Print a compact TSV summary for SGLang bench_serving JSONL files."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


COLUMNS = (
    "file",
    "tag",
    "dataset_name",
    "requested_output_len",
    "mean_actual_output_len",
    "completed",
    "request_rate",
    "max_concurrency",
    "request_throughput",
    "output_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "mean_accept_length",
    "specstream_enabled",
    "specstream_reference_attention",
    "specstream_layer_prefetch",
    "specstream_num_buffers",
    "specstream_cohort_enabled",
    "disable_cuda_graph",
    "disable_overlap_schedule",
    "spectre_fixed_q_mode",
    "error_count",
)


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(Path(item) for item in (matches or [pattern]))
    return sorted(dict.fromkeys(paths))


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\t", " ").replace("\n", " ")


def mean(values: object) -> float:
    if isinstance(values, (int, float)):
        return float(values)
    if not isinstance(values, list) or not values:
        return 0.0
    numeric = [float(value) for value in values]
    return sum(numeric) / len(numeric)


def normalize(row: dict[str, object]) -> None:
    sharegpt_output_len = row.get("sharegpt_output_len")
    row["requested_output_len"] = (
        sharegpt_output_len
        if sharegpt_output_len is not None
        else row.get("random_output_len")
    )
    completed = int(row.get("completed") or 0)
    row["mean_actual_output_len"] = (
        float(row.get("total_output_tokens") or 0) / completed if completed else 0.0
    )
    row["mean_accept_length"] = mean(row.get("accept_length"))
    row["error_count"] = sum(bool(error) for error in row.get("errors", []) or [])
    server_info = row.get("server_info") or {}
    if not isinstance(server_info, dict):
        server_info = {}
    for key in (
        "specstream_enabled",
        "specstream_reference_attention",
        "specstream_layer_prefetch",
        "specstream_num_buffers",
        "specstream_cohort_enabled",
        "disable_cuda_graph",
        "disable_overlap_schedule",
        "spectre_fixed_q_mode",
    ):
        row[key] = server_info.get(key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="JSONL files or glob patterns")
    args = parser.parse_args()

    print("\t".join(COLUMNS))
    for path in expand_inputs(args.files):
        if not path.is_file():
            print(f"warning: missing file: {path}", flush=True)
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["file"] = str(path)
                normalize(row)
                print("\t".join(fmt(row.get(column)) for column in COLUMNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
