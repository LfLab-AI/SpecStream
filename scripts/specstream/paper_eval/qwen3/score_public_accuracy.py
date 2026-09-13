#!/usr/bin/env python3
"""Score public accuracy outputs and compare SpecStream with native SD."""

from __future__ import annotations

import argparse
import collections
import json
import re
import string
from decimal import Decimal, InvalidOperation
from pathlib import Path

def normalize(text: str) -> str:
    text = text.lower().replace(",", "")
    text = "".join(" " if c in string.punctuation else c for c in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def gsm_number(text: str) -> str:
    hits = re.findall(r"####\s*([-+]?\d+(?:\.\d+)?)", text.replace(",", ""))
    if not hits:
        hits = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if not hits:
        return ""
    try:
        value = Decimal(hits[-1])
    except InvalidOperation:
        return hits[-1]
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def longbench_choice(text: str) -> str:
    """Apply the repository's official LongBench-v2 extraction rules.

    A generic "last A-D token" regex is incorrect for LongBench-v2 because a
    chain-of-thought response can mention several options before giving its
    final answer.  This is deliberately kept lightweight so the result scorer
    does not import the full SGLang runtime merely to parse a text file.
    """

    response = str(text).replace("*", "")
    patterns = (
        r"The correct answer is \(([A-D])\)",
        r"The correct answer is ([A-D])",
        r"Answer\s*:\s*([A-D])",
        r"answer\s+is\s*\(?([A-D])\)?",
    )
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def longbench_reference(text: str) -> str:
    value = str(text).strip().upper()
    if value in {"A", "B", "C", "D"}:
        return value
    return longbench_choice(value)


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize(prediction).split()
    ref = normalize(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = collections.Counter(pred) & collections.Counter(ref)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score(dataset: str, rows: list[dict]) -> dict:
    valid = [row for row in rows if not row.get("error")]
    if dataset == "gsm8k":
        values = [gsm_number(r["generated_text"]) == gsm_number(r["reference_answer"]) for r in valid]
        return {
            "task_metric": "numeric_exact_match",
            "task_score": sum(values) / max(len(rows), 1),
            "correct": sum(values),
            "scored_samples": len(rows),
        }
    if dataset == "longbench_v2":
        predictions = [longbench_choice(r["generated_text"]) for r in valid]
        references = [longbench_reference(r["reference_answer"]) for r in valid]
        values = [bool(pred) and pred == ref for pred, ref in zip(predictions, references)]
        parsed = sum(bool(pred) for pred in predictions)
        return {
            "task_metric": "official_choice_accuracy",
            "task_score": sum(values) / max(len(rows), 1),
            "correct": sum(values),
            "scored_samples": len(rows),
            "parsed_predictions": parsed,
            "invalid_predictions": len(rows) - parsed,
        }
    f1 = [token_f1(r["generated_text"], r["reference_answer"]) for r in valid]
    markers = [
        bool(r.get("random_string_to_prepend"))
        and normalize(r["random_string_to_prepend"]) in normalize(r["generated_text"])
        for r in valid
    ]
    return {
        "task_metric": "token_f1",
        "task_score": sum(f1) / max(len(rows), 1),
        "marker_recall": sum(markers) / max(len(rows), 1),
        "scored_samples": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "longbench_v2", "mrcr"], required=True)
    parser.add_argument("--input", action="append", required=True, metavar="METHOD=JSONL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    by_method: dict[str, list[dict]] = {}
    for item in args.input:
        method, path = item.split("=", 1)
        by_method[method] = load(Path(path))
    report = {"dataset": args.dataset, "methods": {}}
    for method, rows in by_method.items():
        report["methods"][method] = {"samples": len(rows), "failures": sum(bool(r.get("error")) for r in rows), **score(args.dataset, rows)}
    specstream_key = next(
        (key for key in ("SPECSTREAM_1GPU", "SPECSTREAM") if key in by_method),
        None,
    )
    if "SGLANG_SD" in by_method and specstream_key is not None:
        left = {r["sample_id"]: r for r in by_method["SGLANG_SD"]}
        right = {r["sample_id"]: r for r in by_method[specstream_key]}
        ids = sorted(set(left) & set(right))
        paired = {
            "samples": len(ids),
            "normalized_output_agreement": sum(normalize(left[i]["generated_text"]) == normalize(right[i]["generated_text"]) for i in ids) / max(len(ids), 1),
        }
        if args.dataset == "gsm8k":
            paired["task_answer_agreement"] = sum(
                gsm_number(left[i]["generated_text"])
                == gsm_number(right[i]["generated_text"])
                for i in ids
            ) / max(len(ids), 1)
        elif args.dataset == "longbench_v2":
            paired["task_answer_agreement"] = sum(
                bool(longbench_choice(left[i]["generated_text"]))
                and longbench_choice(left[i]["generated_text"])
                == longbench_choice(right[i]["generated_text"])
                for i in ids
            ) / max(len(ids), 1)
        report["paired"] = paired
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
