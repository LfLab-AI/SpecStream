#!/usr/bin/env python3
"""Run deterministic, sample-addressable public-dataset accuracy requests."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import requests


def load_manifest(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("accuracy manifest must be a JSON list")
    return rows


def balanced_mrcr(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or limit >= len(rows):
        return rows
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("n_needles", "unknown")), []).append(row)
    chosen: list[dict] = []
    keys = sorted(groups)
    while len(chosen) < limit and any(groups.values()):
        for key in keys:
            if groups[key] and len(chosen) < limit:
                chosen.append(groups[key].pop(0))
    return chosen


def generated_text(payload: object) -> str:
    if isinstance(payload, dict):
        value = payload.get("text", payload.get("generated_text", ""))
        if isinstance(value, list):
            return str(value[0]) if value else ""
        return str(value)
    if isinstance(payload, list) and payload:
        return generated_text(payload[0])
    return ""


def run_request(
    *,
    ordinal: int,
    row: dict,
    dataset: str,
    url: str,
    max_new_tokens: int,
    seed: int,
    timeout_s: float,
) -> dict:
    conversations = row.get("conversations", [])
    if not conversations:
        raise ValueError(f"row {ordinal} has no conversations")
    prompt = str(conversations[0].get("value", ""))
    record = {
        "dataset": dataset,
        "ordinal": ordinal,
        "sample_id": row.get("sample_id", f"{dataset}-{ordinal:05d}"),
        "prompt_tokens": row.get("prompt_tokens"),
        "reference_answer": row.get(
            "reference_answer", conversations[-1].get("value", "")
        ),
        "metric": row.get("metric"),
        "random_string_to_prepend": row.get("random_string_to_prepend", ""),
        "n_needles": row.get("n_needles", ""),
        "category": row.get("category", row.get("domain", "")),
        "sub_domain": row.get("sub_domain", ""),
        "difficulty": row.get("difficulty", ""),
        "length_bucket": row.get("length_bucket", row.get("length", "")),
    }
    response = None
    started = time.perf_counter()
    try:
        response = requests.post(
            url,
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "top_p": 1,
                    "max_new_tokens": max_new_tokens,
                    "sampling_seed": seed,
                },
            },
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        record["generated_text"] = generated_text(payload)
        record["meta_info"] = (
            payload.get("meta_info", {}) if isinstance(payload, dict) else {}
        )
        record["error"] = None
    except Exception as exc:  # Preserve evidence before failing the case.
        record["generated_text"] = ""
        record["meta_info"] = {}
        body = ""
        if response is not None:
            body = response.text[:2000].replace("\n", " ")
        record["error"] = f"{type(exc).__name__}: {exc}; response={body}"
    record["latency_s"] = time.perf_counter() - started
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=["gsm8k", "longbench_v2", "mrcr"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30000/generate")
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of simultaneous /generate requests.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=1,
        help="Stop after this many request failures instead of flooding the server log.",
    )
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    rows = balanced_mrcr(rows, args.limit) if args.dataset == "mrcr" else rows[: args.limit or None]
    if not rows:
        raise RuntimeError("no accuracy rows selected")
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    completed = 0
    with args.output.open("w", encoding="utf-8") as handle:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.max_concurrency, len(rows))
        )
        pending: dict[concurrent.futures.Future, int] = {}
        next_ordinal = 0

        def submit(ordinal: int) -> None:
            pending[
                executor.submit(
                    run_request,
                    ordinal=ordinal,
                    row=rows[ordinal],
                    dataset=args.dataset,
                    url=args.url,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                    timeout_s=args.timeout_s,
                )
            ] = ordinal

        for _ in range(min(args.max_concurrency, len(rows))):
            submit(next_ordinal)
            next_ordinal += 1

        try:
            while pending and failures < args.max_errors:
                done, _ = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in sorted(done, key=lambda item: pending[item]):
                    pending.pop(future)
                    record = future.result()
                    completed += 1
                    if record["error"] is not None:
                        failures += 1
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(
                        f"[{completed}/{len(rows)}] {record['sample_id']} "
                        f"ordinal={record['ordinal']} "
                        f"error={record['error'] is not None}",
                        flush=True,
                    )
                    if failures >= args.max_errors:
                        break
                    if next_ordinal < len(rows):
                        submit(next_ordinal)
                        next_ordinal += 1
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        if failures >= args.max_errors:
            print(
                f"FAIL_FAST: reached max_errors={args.max_errors}; "
                "inspect the recorded response and server log before retrying",
                flush=True,
            )
    if failures:
        raise SystemExit(f"accuracy client recorded {failures} failed requests")


if __name__ == "__main__":
    main()
