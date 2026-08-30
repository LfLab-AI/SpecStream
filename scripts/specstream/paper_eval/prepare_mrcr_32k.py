#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def render_messages(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        parts = []
        for m in messages:
            parts.append(
                f"<|{m.get('role', 'user')}|>\n"
                f"{m.get('content', '')}\n"
            )
        parts.append("<|assistant|>\n")
        return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--server-context", type=int, default=32768)
    ap.add_argument("--output-len", type=int, default=256)
    args = ap.parse_args()

    max_prompt = args.server_context - args.output_len

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        use_fast=True,
    )

    ds = load_dataset("openai/mrcr", split="train")

    buckets = {
        "8k16k": [],
        "16k32k": [],
    }
    meta = []

    for source_index, row in enumerate(ds):
        messages = json.loads(row["prompt"])
        prompt = render_messages(tok, messages)

        prompt_tokens = len(
            tok.encode(prompt, add_special_tokens=False)
        )

        if prompt_tokens + args.output_len > args.server_context:
            continue

        answer = str(row["answer"])

        item = {
            "source_index": source_index,
            "prompt_tokens": prompt_tokens,
            "n_needles": int(row["n_needles"]),
            "desired_msg_index": int(row["desired_msg_index"]),
            "total_messages": int(row["total_messages"]),
        }

        share = {
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": answer},
            ]
        }

        if 8192 <= prompt_tokens <= 16384:
            buckets["8k16k"].append(share)
            item["bucket"] = "8k16k"
            meta.append(item)

        elif 16384 < prompt_tokens <= max_prompt:
            buckets["16k32k"].append(share)
            item["bucket"] = "16k32k"
            meta.append(item)

    p1 = out_dir / "mrcr_qwen25_8k16k_sharegpt.json"
    p2 = out_dir / "mrcr_qwen25_16k32k_sharegpt.json"
    pm = out_dir / "mrcr_qwen25_metadata.jsonl"

    p1.write_text(
        json.dumps(buckets["8k16k"], ensure_ascii=False),
        encoding="utf-8",
    )
    p2.write_text(
        json.dumps(buckets["16k32k"], ensure_ascii=False),
        encoding="utf-8",
    )

    with pm.open("w", encoding="utf-8") as f:
        for x in meta:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print("8K-16K rows =", len(buckets["8k16k"]))
    print("16K-32K rows =", len(buckets["16k32k"]))

    cnt = Counter((x["bucket"], x["n_needles"]) for x in meta)

    print("bucket x needles:")
    for k in sorted(cnt):
        print(k, cnt[k])

    assert buckets["8k16k"]
    assert buckets["16k32k"]


if __name__ == "__main__":
    main()
