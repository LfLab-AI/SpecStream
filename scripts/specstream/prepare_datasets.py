#!/usr/bin/env python3
"""Inspect and normalize the local ShareGPT-style datasets used by SpecStream.

The SGLang ``random`` and ``sharegpt`` benchmark loaders consume only the
first prompt/completion pair.  ``merge-sharegpt`` therefore validates and
normalizes the first two turns of every record, and deliberately discards
later turns.  This makes the prepared file robust to ShareGPT V3 records with
metadata or incomplete later conversation turns, without silently changing
the pair that is actually benchmarked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def read_json_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("data", "train", "records", "instances"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return candidate
        raise ValueError(
            f"{path} is a JSON object, not a sample array; top-level keys="
            f"{list(value)[:20]}"
        )
    raise ValueError(f"{path} has unsupported top-level type {type(value).__name__}")


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            yield line_number, value


def conversation_list(record: dict) -> list:
    value = record.get("conversations", record.get("conversation", []))
    return value if isinstance(value, list) else []


def classify_record(record: dict) -> str:
    turns = conversation_list(record)
    if not turns or not isinstance(turns[0], dict):
        return "unknown"
    keys = set(turns[0])
    if "human" in keys and "assistant" in keys:
        return "paired-human-assistant"
    if "value" in keys or "content" in keys:
        return "turn-list"
    return "unknown"


def inspect_path(path: Path) -> int:
    if path.suffix.lower() == ".jsonl":
        line_number, sample = next(iter_jsonl(path))
        count = sum(1 for _ in iter_jsonl(path))
        source = f"JSONL rows={count}, first_nonempty_line={line_number}"
    else:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, list):
            sample = raw[0] if raw else {}
            source = f"JSON array samples={len(raw)}"
        elif isinstance(raw, dict):
            source = f"JSON object keys={list(raw)[:20]}"
            sample = raw
        else:
            source = f"JSON top-level type={type(raw).__name__}"
            sample = {}
    turns = conversation_list(sample) if isinstance(sample, dict) else []
    print(f"path={path}")
    print(source)
    print(f"record_type={classify_record(sample) if isinstance(sample, dict) else 'unknown'}")
    print(f"top_keys={list(sample)[:20] if isinstance(sample, dict) else []}")
    print(f"conversation_items={len(turns)}")
    print(
        "first_item_keys="
        f"{list(turns[0])[:20] if turns and isinstance(turns[0], dict) else []}"
    )
    return 0


def turn_text(turn: dict) -> str | None:
    value = turn.get("value", turn.get("content"))
    return value if isinstance(value, str) and value.strip() else None


def normalize_sharegpt_record(record: dict, source: Path, index: int) -> dict:
    turns = conversation_list(record)
    if len(turns) < 2 or not all(isinstance(turn, dict) for turn in turns[:2]):
        raise ValueError(f"{source}: record {index} has fewer than two valid turns")
    normalized = []
    # SGLang's ShareGPT and random-text loaders use only conversations[0:2].
    # Do not reject an otherwise usable record because a later multi-turn
    # message is absent or is metadata rather than text.
    for turn_index, turn in enumerate(turns[:2]):
        text = turn_text(turn)
        if text is None:
            raise ValueError(
                f"{source}: record {index} benchmark turn {turn_index} "
                "has no non-empty value/content"
            )
        role = turn.get("from", turn.get("role"))
        if role is None:
            role = "human" if turn_index % 2 == 0 else "gpt"
        normalized.append({"from": role, "value": text})
    result = dict(record)
    result.pop("conversation", None)
    result["conversations"] = normalized
    return result


def merge_sharegpt(inputs: list[Path], output: Path, strict: bool = False) -> int:
    merged: list[dict] = []
    total_records = 0
    skipped_records = 0
    warning_limit = 20
    for path in inputs:
        records = read_json_records(path)
        file_kept = 0
        file_skipped = 0
        for index, record in enumerate(records):
            total_records += 1
            if not isinstance(record, dict):
                error = ValueError(f"{path}: record {index} is not an object")
            else:
                try:
                    normalized = normalize_sharegpt_record(record, path, index)
                except ValueError as exc:
                    error = exc
                else:
                    merged.append(normalized)
                    file_kept += 1
                    continue

            if strict:
                raise error
            skipped_records += 1
            file_skipped += 1
            if skipped_records <= warning_limit:
                print(f"warning: skipped unusable record: {error}")

        print(
            f"loaded {len(records)} records from {path}; "
            f"kept={file_kept}, skipped={file_skipped}"
        )
    if not merged:
        raise ValueError("no usable ShareGPT records were found in the input files")
    if skipped_records > warning_limit:
        print(
            f"warning: suppressed {skipped_records - warning_limit} additional "
            "unusable-record messages"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False)
    print(
        f"wrote {len(merged)} ShareGPT records to {output}; "
        f"source_records={total_records}, skipped={skipped_records}; "
        "each record contains the first two benchmark turns only"
    )
    return 0


def convert_paired_jsonl(input_path: Path, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    source_rows = 0
    with output.open("w", encoding="utf-8") as destination:
        for line_number, record in iter_jsonl(input_path):
            source_rows += 1
            pairs = conversation_list(record)
            if not pairs:
                raise ValueError(f"{input_path}:{line_number}: missing conversation")
            for turn_index, pair in enumerate(pairs):
                if not isinstance(pair, dict):
                    raise ValueError(
                        f"{input_path}:{line_number}: conversation item is not an object"
                    )
                human = pair.get("human")
                assistant = pair.get("assistant")
                if not isinstance(human, str) or not isinstance(assistant, str):
                    raise ValueError(
                        f"{input_path}:{line_number}: expected human/assistant strings"
                    )
                if not human.strip() or not assistant.strip():
                    continue
                converted = {
                    "conversation_id": record.get("conversation_id", line_number),
                    "category": record.get("category"),
                    "turn_index": turn_index,
                    "conversations": [
                        {"role": "user", "content": human},
                        {"role": "assistant", "content": assistant},
                    ],
                }
                destination.write(json.dumps(converted, ensure_ascii=False) + "\n")
                rows_written += 1
    print(
        f"converted {source_rows} source rows into {rows_written} SGLang custom rows: "
        f"{output}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("path", type=Path)

    merge_parser = subparsers.add_parser("merge-sharegpt")
    merge_parser.add_argument("inputs", nargs="+", type=Path)
    merge_parser.add_argument("--output", required=True, type=Path)
    merge_parser.add_argument(
        "--strict",
        action="store_true",
        help="fail on the first unusable record instead of skipping it",
    )

    convert_parser = subparsers.add_parser("convert-paired-jsonl")
    convert_parser.add_argument("--input", required=True, type=Path)
    convert_parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "inspect":
        return inspect_path(args.path)
    if args.command == "merge-sharegpt":
        return merge_sharegpt(args.inputs, args.output, strict=args.strict)
    if args.command == "convert-paired-jsonl":
        return convert_paired_jsonl(args.input, args.output)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
