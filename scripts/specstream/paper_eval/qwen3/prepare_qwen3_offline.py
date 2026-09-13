#!/usr/bin/env python3
"""Prepare the local public datasets for Qwen3-8B/0.6B experiments.

The script is intentionally offline-only.  Every input must resolve to a local
file or directory, and both tokenizers are loaded with ``local_files_only``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm
from transformers import AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load a local HF save_to_disk directory or ordinary JSON/Parquet tree."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if path.suffix.lower() == ".jsonl":
            return read_jsonl(path)
        if path.suffix.lower() == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                obj = obj.get("data", obj.get("train", []))
            if not isinstance(obj, list):
                raise ValueError(f"JSON source is not a row list: {path}")
            return [dict(row) for row in obj]
        if path.suffix.lower() == ".parquet":
            from datasets import load_dataset

            ds = load_dataset("parquet", data_files=str(path), split="train")
            return [dict(row) for row in ds]
        raise ValueError(f"unsupported input file: {path}")

    try:
        from datasets import Dataset, DatasetDict, load_from_disk

        obj = load_from_disk(str(path))
        if isinstance(obj, DatasetDict):
            for split in ("test", "train", "validation"):
                if split in obj:
                    return [dict(row) for row in obj[split]]
            obj = next(iter(obj.values()))
        if isinstance(obj, Dataset):
            return [dict(row) for row in obj]
    except Exception:
        pass

    parquet = sorted(path.rglob("*.parquet"))
    if parquet:
        from datasets import load_dataset

        ds = load_dataset(
            "parquet", data_files=[str(item) for item in parquet], split="train"
        )
        return [dict(row) for row in ds]

    rows: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*.jsonl")):
        rows.extend(read_jsonl(item))
    if rows:
        return rows
    for item in sorted(path.rglob("*.json")):
        try:
            obj = json.loads(item.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, list) and (not obj or isinstance(obj[0], dict)):
            rows.extend(dict(row) for row in obj)
    if rows:
        return rows
    raise RuntimeError(f"no supported local dataset files found below {path}")


def write_json(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def sharegpt_row(prompt: str, answer: str, tokens: int, **metadata: Any) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer or "N/A"},
        ],
        "prompt_tokens": tokens,
        "template": "qwen3_nonthinking",
        **metadata,
    }


def prepare_gsm8k(tokenizer, source: Path, output: Path, max_prompt: int) -> dict:
    rows = load_rows(source)
    prepared = []
    lengths = []
    for index, row in enumerate(tqdm(rows, desc="GSM8K -> Qwen3", unit="row")):
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        prompt = render(
            tokenizer,
            [{
                "role": "user",
                "content": (
                    "Solve the following problem. Give a concise derivation and finish "
                    "with `#### <number>`.\n\n" + question
                ),
            }],
        )
        length = token_len(tokenizer, prompt)
        if length <= max_prompt:
            answer = str(row.get("answer", "N/A"))
            prepared.append(
                sharegpt_row(
                    prompt,
                    answer,
                    length,
                    sample_id=f"gsm8k-{index:05d}",
                    reference_answer=answer,
                    metric="gsm8k_numeric_exact_match",
                )
            )
            lengths.append(length)
    if not prepared:
        raise RuntimeError("GSM8K produced no usable rows")
    write_json(output, prepared)
    return {"source_rows": len(rows), "rows": len(prepared), "min": min(lengths), "max": max(lengths)}


def longbench_question(row: dict[str, Any], context: str) -> str:
    from sglang.test.simple_eval_longbench_v2 import format_longbench_v2_question

    item = dict(row)
    item["context"] = context
    return format_longbench_v2_question(item)


def fit_longbench(tokenizer, row: dict[str, Any], max_prompt: int) -> tuple[str, int, bool]:
    context = str(row.get("context", ""))
    if not context or not row.get("question"):
        raise ValueError("LongBench row lacks raw context/question")
    full = render(tokenizer, [{"role": "user", "content": longbench_question(row, context)}])
    full_len = token_len(tokenizer, full)
    if full_len <= max_prompt:
        return full, full_len, False

    ids = tokenizer.encode(context, add_special_tokens=False)
    marker = "\n\n...[middle truncated to fit the 32K evaluation window]...\n\n"
    low, high = 256, len(ids)
    best: tuple[str, int] | None = None
    while low <= high:
        keep = (low + high) // 2
        head = keep // 2
        tail = keep - head
        shortened = (
            tokenizer.decode(ids[:head], skip_special_tokens=False)
            + marker
            + tokenizer.decode(ids[-tail:], skip_special_tokens=False)
        )
        prompt = render(
            tokenizer,
            [{"role": "user", "content": longbench_question(row, shortened)}],
        )
        length = token_len(tokenizer, prompt)
        if length <= max_prompt:
            best = prompt, length
            low = keep + 1
        else:
            high = keep - 1
    if best is None:
        raise RuntimeError("LongBench prompt cannot fit the configured context")
    return best[0], best[1], True


def prepare_longbench(
    tokenizer, source: Path, output: Path, metadata_path: Path, max_prompt: int, min_prompt: int
) -> dict:
    rows = load_rows(source)
    prepared, metadata, lengths = [], [], []
    truncated = 0
    skipped = 0
    for index, row in enumerate(tqdm(rows, desc="LongBench-v2 -> Qwen3", unit="row")):
        context = str(row.get("context", ""))
        if not context or not row.get("question"):
            skipped += 1
            continue
        prompt = render(
            tokenizer,
            [{"role": "user", "content": longbench_question(row, context)}],
        )
        length = token_len(tokenizer, prompt)
        # Accuracy and performance share the exact unmodified task prompt.
        # Over-context rows are excluded instead of middle-truncated because
        # truncation can delete the evidence that determines the answer.
        if length > max_prompt:
            skipped += 1
            continue
        was_truncated = False
        if length < min_prompt:
            skipped += 1
            continue
        answer = str(row.get("answer", "N/A"))
        sample_id = str(row.get("_id", row.get("id", f"longbench-{index:05d}")))
        prepared.append(
            sharegpt_row(
                prompt,
                answer,
                length,
                source_index=index,
                sample_id=sample_id,
                reference_answer=answer,
                metric="longbench_v2_choice_accuracy",
                category=row.get("category", row.get("domain", "")),
                sub_domain=row.get("sub_domain", ""),
                difficulty=row.get("difficulty", ""),
                length_bucket=row.get("length", ""),
            )
        )
        metadata.append({
            "source_index": index,
            "id": row.get("_id", row.get("id", "")),
            "prompt_tokens": length,
            "truncated": was_truncated,
            "answer": answer,
            "category": row.get("category", row.get("domain", "")),
            "sub_domain": row.get("sub_domain", ""),
            "difficulty": row.get("difficulty", ""),
            "length_bucket": row.get("length", ""),
        })
        lengths.append(length)
        truncated += int(was_truncated)
    if not prepared:
        raise RuntimeError("LongBench-v2 produced no 8K-32K rows")
    write_json(output, prepared)
    write_jsonl(metadata_path, metadata)
    return {
        "source_rows": len(rows), "rows": len(prepared), "skipped": skipped,
        "truncated": truncated, "min": min(lengths), "max": max(lengths),
    }


def normalize_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    raw = row.get("prompt", row.get("messages", row.get("conversation", [])))
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
            raw = decoded if isinstance(decoded, list) else raw
        except json.JSONDecodeError:
            pass
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    if not isinstance(raw, list):
        return []
    messages = []
    for message in raw:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", message.get("from", ""))).lower()
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        content = str(message.get("content", message.get("value", "")))
        if role in {"system", "user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages


def prepare_mrcr(tokenizer, source: Path, output: Path, metadata_path: Path, max_prompt: int) -> dict:
    rows = load_rows(source)
    prepared, metadata, lengths = [], [], []
    skipped = 0
    for index, row in enumerate(tqdm(rows, desc="MRCR -> Qwen3", unit="row")):
        # The local MRCR files contain contexts from a few thousand to several
        # hundred thousand characters.  This conservative character gate only
        # avoids obviously out-of-bucket rows; the authoritative 16K/32K
        # decision is still made with the Qwen3 Target tokenizer below.
        n_chars = int(row.get("n_chars", 0) or 0)
        if n_chars and (n_chars < 40000 or n_chars > 180000):
            skipped += 1
            continue
        messages = normalize_messages(row)
        if not messages:
            skipped += 1
            continue
        prompt = render(tokenizer, messages)
        length = token_len(tokenizer, prompt)
        if not 16384 < length <= max_prompt:
            skipped += 1
            continue
        answer = str(row.get("answer", row.get("response", "N/A")))
        marker = str(row.get("random_string_to_prepend", ""))
        prepared.append(
            sharegpt_row(
                prompt,
                answer,
                length,
                source_index=index,
                sample_id=f"mrcr-{index:05d}",
                reference_answer=answer,
                metric="mrcr_token_f1_and_marker_recall",
                random_string_to_prepend=marker,
                n_needles=row.get("n_needles", ""),
            )
        )
        metadata.append({
            "source_index": index, "prompt_tokens": length,
            "n_needles": row.get("n_needles", ""), "answer": answer,
            "random_string_to_prepend": marker,
        })
        lengths.append(length)
    if not prepared:
        raise RuntimeError("MRCR produced no 16K-32K rows")
    write_json(output, prepared)
    write_jsonl(metadata_path, metadata)
    return {
        "source_rows": len(rows), "rows": len(prepared), "skipped": skipped,
        "min": min(lengths), "max": max(lengths),
    }


def verify_manifest(tokenizer, path: Path, max_prompt: int, minimum_rows: int) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) < minimum_rows:
        raise RuntimeError(f"{path} has {len(rows)} rows; need {minimum_rows}")
    lengths = []
    for row in tqdm(rows, desc=f"Verify {path.name}", unit="row"):
        conv = row.get("conversations", [])
        if len(conv) < 2 or conv[0].get("from") != "human":
            raise ValueError(f"invalid ShareGPT row in {path}")
        length = token_len(tokenizer, str(conv[0].get("value", "")))
        if length + 256 > max_prompt + 256:
            raise ValueError(f"context overflow in {path}: {length}")
        lengths.append(length)
    return {"rows": len(rows), "min": min(lengths), "max": max(lengths)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--gsm8k-source", type=Path, required=True)
    parser.add_argument("--longbench-source", type=Path, required=True)
    parser.add_argument("--mrcr-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--reserved-output-length", type=int, default=256)
    parser.add_argument("--minimum-rows", type=int, default=64)
    args = parser.parse_args()

    for path in (args.target_model, args.draft_model, args.gsm8k_source, args.longbench_source, args.mrcr_source):
        if not path.exists():
            raise FileNotFoundError(path)
    target = AutoTokenizer.from_pretrained(
        args.target_model, trust_remote_code=True, use_fast=True, local_files_only=True
    )
    draft = AutoTokenizer.from_pretrained(
        args.draft_model, trust_remote_code=True, use_fast=True, local_files_only=True
    )
    if target.get_vocab() != draft.get_vocab():
        raise RuntimeError("Target and Draft token-to-id mappings differ")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(target, key) != getattr(draft, key):
            raise RuntimeError(f"special token mismatch: {key}")

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    max_prompt = args.context_length - args.reserved_output_length
    outputs = {
        "gsm8k": root / "gsm8k_qwen3_nothink_sharegpt.json",
        "longbench": root / "longbench_v2_qwen3_8b_8k32k_sharegpt.json",
        "longbench_meta": root / "longbench_v2_qwen3_8b_8k32k_metadata.jsonl",
        "mrcr": root / "mrcr_qwen3_16k32k_sharegpt.json",
        "mrcr_meta": root / "mrcr_qwen3_16k32k_metadata.jsonl",
    }
    stats = {
        "gsm8k": prepare_gsm8k(target, args.gsm8k_source, outputs["gsm8k"], max_prompt),
        "longbench": prepare_longbench(
            target, args.longbench_source, outputs["longbench"], outputs["longbench_meta"],
            max_prompt, 8192,
        ),
        "mrcr": prepare_mrcr(target, args.mrcr_source, outputs["mrcr"], outputs["mrcr_meta"], max_prompt),
    }
    stats["verified"] = {
        name: verify_manifest(
            target,
            outputs[name],
            max_prompt,
            max(128, args.minimum_rows) if name == "gsm8k" else args.minimum_rows,
        )
        for name in ("gsm8k", "longbench", "mrcr")
    }
    vocab_json = json.dumps(sorted(target.get_vocab().items()), ensure_ascii=False).encode()
    manifest = {
        "target_model": str(args.target_model), "draft_model": str(args.draft_model),
        "vocab_sha256": hashlib.sha256(vocab_json).hexdigest(),
        "context_length": args.context_length,
        "reserved_output_length": args.reserved_output_length,
        "enable_thinking": False,
    }
    (root / "tokenizer_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checksum_files = [*outputs.values(), root / "tokenizer_manifest.json", root / "dataset_stats.json"]
    with (root / "dataset_sha256.txt").open("w", encoding="utf-8") as handle:
        for path in checksum_files:
            # The prepared directory is atomically renamed after validation;
            # relative paths keep this manifest usable after that rename.
            handle.write(f"{sha256_file(path)}  {path.relative_to(root)}\n")
    print("QWEN3_OFFLINE_DATA_GATE=PASS")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
