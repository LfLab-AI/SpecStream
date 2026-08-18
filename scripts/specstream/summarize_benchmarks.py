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
    "accept_length",
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
                print("\t".join(fmt(row.get(column)) for column in COLUMNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
