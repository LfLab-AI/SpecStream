#!/usr/bin/env python3

import argparse
import hashlib
import json
import random
from pathlib import Path

import requests
from transformers import AutoTokenizer


def extract_token_ids(body):
    # SGLang 原生 /generate 响应把完整生成序列放在顶层 output_ids。
    # output_token_logprobs 在某些 SPECTRE 路径中可能只含最后一个 token，
    # 因而不能用它判断实际生成长度。
    output_ids = body.get("output_ids")
    if isinstance(output_ids, list) and all(
        isinstance(token_id, int) for token_id in output_ids
    ):
        return output_ids

    meta = body.get("meta_info") or {}
    result = []
    for item in meta.get("output_token_logprobs") or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            result.append(int(item[1]))
        elif isinstance(item, dict) and "token_id" in item:
            result.append(int(item["token_id"]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-len", type=int, default=16384)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    session = requests.Session()
    health = session.get(args.base_url + "/health", timeout=30)
    health.raise_for_status()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    seed_text = (
        "A systems paper evaluates long-context inference, tiered KV storage, "
        "PCIe transfer, and speculative decoding. Preserve every identifier "
        "and continue the technical explanation deterministically. "
    )
    seed_ids = tok.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError("tokenizer produced an empty seed")
    repeat = args.prompt_len // len(seed_ids) + 4
    base = seed_ids * repeat

    rows = []
    for index in range(args.num_prompts):
        rng = random.Random(args.seed + index)
        max_offset = len(base) - args.prompt_len
        offset = rng.randrange(max_offset + 1) if max_offset > 0 else 0
        input_ids = base[offset : offset + args.prompt_len]
        if len(input_ids) != args.prompt_len:
            raise RuntimeError("failed to construct exact-length input_ids")

        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.output_len,
                "sampling_seed": args.seed,
            },
            "return_logprob": True,
            "top_logprobs_num": 0,
        }
        response = session.post(
            args.base_url + "/generate", json=payload, timeout=1200
        )
        if not response.ok:
            raise RuntimeError(
                f"POST /generate returned HTTP {response.status_code}: "
                f"{response.text[:4000]}"
            )
        body = response.json()
        meta = body.get("meta_info") or {}
        token_ids = extract_token_ids(body)
        if len(token_ids) != args.output_len:
            raise RuntimeError(
                f"request {index}: expected {args.output_len} output token ids, "
                f"got {len(token_ids)}; finish_reason={meta.get('finish_reason')}"
            )
        prompt_bytes = json.dumps(input_ids, separators=(",", ":")).encode()
        rows.append(
            {
                "index": index,
                "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                "output_token_ids": token_ids,
                "output_text": body.get("text", ""),
                "finish_reason": meta.get("finish_reason"),
            }
        )
        print(f"completed {index + 1}/{args.num_prompts}", flush=True)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("WROTE", destination, "rows=", len(rows))


if __name__ == "__main__":
    main()
