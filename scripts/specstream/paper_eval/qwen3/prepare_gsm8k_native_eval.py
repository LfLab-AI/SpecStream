#!/usr/bin/env python3
"""Prepare leakage-free GSM8K files for the native SGLang evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        from datasets import load_dataset

        return [
            dict(row)
            for row in load_dataset("parquet", data_files=str(path), split="train")
        ]
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if path.suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            obj = obj.get("data", [])
        if isinstance(obj, list):
            return [dict(row) for row in obj]
    raise ValueError(f"unsupported GSM8K source: {path}")


def validate_rows(rows: list[dict], name: str) -> None:
    if not rows:
        raise ValueError(f"{name} is empty")
    for index, row in enumerate(rows):
        if not str(row.get("question", "")).strip() or not str(row.get("answer", "")).strip():
            raise ValueError(f"{name} row {index} lacks question/answer")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {"question": str(row["question"]), "answer": str(row["answer"])},
                    ensure_ascii=False,
                )
                + "\n"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chat_template_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    # transformers 5.x returns BatchEncoding here, while older versions may
    # return the input-id list directly.
    if isinstance(rendered, dict) or hasattr(rendered, "keys"):
        rendered = rendered["input_ids"]
    if rendered and isinstance(rendered[0], list):
        if len(rendered) != 1:
            raise RuntimeError("unexpected batched chat-template output")
        rendered = rendered[0]
    return [int(token) for token in rendered]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-source", type=Path, required=True)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--expected-test-rows", type=int, default=1319)
    args = parser.parse_args()

    test_rows = load_rows(args.test_source)
    train_rows = load_rows(args.train_source)
    validate_rows(test_rows, "GSM8K main/test")
    validate_rows(train_rows, "GSM8K main/train")
    if len(test_rows) != args.expected_test_rows:
        raise RuntimeError(
            f"GSM8K test row mismatch: got {len(test_rows)}, expected {args.expected_test_rows}"
        )
    if len(train_rows) < args.num_shots:
        raise RuntimeError(f"GSM8K train has only {len(train_rows)} rows")

    target = AutoTokenizer.from_pretrained(
        args.target_model, trust_remote_code=True, local_files_only=True
    )
    draft = AutoTokenizer.from_pretrained(
        args.draft_model, trust_remote_code=True, local_files_only=True
    )
    if target.get_vocab() != draft.get_vocab():
        raise RuntimeError("Target and Draft token-to-id mappings differ")

    demos = train_rows[: args.num_shots]
    demonstration = "".join(
        f"Question: {row['question']}\nAnswer: {row['answer']}\n\n" for row in demos
    )
    probe_content = demonstration + f"Question: {test_rows[0]['question']}\nAnswer:"
    messages = [{"role": "user", "content": probe_content}]
    target_ids = chat_template_ids(target, messages)
    draft_ids = chat_template_ids(draft, messages)
    if target_ids != draft_ids:
        raise RuntimeError("Target and Draft render different Qwen3 non-thinking prompts")

    args.output_root.mkdir(parents=True, exist_ok=True)
    test_output = args.output_root / "gsm8k_main_test.jsonl"
    fewshot_output = args.output_root / f"gsm8k_main_train_fewshot{args.num_shots}.jsonl"
    metadata_output = args.output_root / "gsm8k_native_eval_manifest.json"
    write_jsonl(test_output, test_rows)
    write_jsonl(fewshot_output, demos)
    metadata = {
        "test_source": str(args.test_source),
        "train_source": str(args.train_source),
        "test_rows": len(test_rows),
        "few_shot_rows": len(demos),
        "few_shot_source_split": "main/train",
        "test_split_leakage": False,
        "target_model": str(args.target_model),
        "draft_model": str(args.draft_model),
        "enable_thinking": False,
        "probe_prompt_tokens": len(target_ids),
        "test_sha256": sha256(test_output),
        "fewshot_sha256": sha256(fewshot_output),
    }
    metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checksum_output = args.output_root / "gsm8k_native_eval_sha256.txt"
    with checksum_output.open("w", encoding="utf-8") as handle:
        for path in (test_output, fewshot_output, metadata_output):
            handle.write(f"{sha256(path)}  {path.name}\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"GSM8K_TEST={test_output}")
    print(f"GSM8K_FEWSHOT={fewshot_output}")
    print("GSM8K_NATIVE_DATA_GATE=PASS")


if __name__ == "__main__":
    main()
