#!/usr/bin/env python3
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def normalize_label(x):
    s = str(x).strip().upper()
    m = re.search(r'(?<![A-Z])([ABCD])(?![A-Z])', s)
    return m.group(1) if m else ""


def infer(base_url, prompt, max_new_tokens):
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "top_p": 1,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": False,
        },
        "stream": False,
    }

    r = requests.post(
        base_url.rstrip("/") + "/generate",
        json=payload,
        timeout=1800,
    )
    r.raise_for_status()

    obj = r.json()
    text = obj.get("text", "")
    return text, normalize_label(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-prompt-tokens", type=int, default=0)
    ap.add_argument("--only-stream-eligible", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    src = load_jsonl(args.dataset)
    selected = []

    for i, x in enumerate(src):
        if "eval_prompt" not in x:
            raise ValueError(
                "Prepared LongBench file must contain eval_prompt"
            )

        if (
            args.only_stream_eligible
            and not bool(x.get("expected_stream_eligible", False))
        ):
            continue

        n = int(x.get("prompt_tokens", 0) or 0)

        if n < args.min_prompt_tokens:
            continue

        selected.append((i, x, x["eval_prompt"], n))

        if args.limit and len(selected) >= args.limit:
            break

    print("selected examples =", len(selected))

    out = [None] * len(selected)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                infer,
                args.base_url,
                prompt,
                args.max_new_tokens,
            ): j
            for j, (_, _, prompt, _) in enumerate(selected)
        }

        for fut in as_completed(futs):
            j = futs[fut]
            source_index, x, _, prompt_tokens = selected[j]

            try:
                text, pred = fut.result()
                gold = normalize_label(x["answer"])

                out[j] = {
                    "source_index": source_index,
                    "_id": x.get("_id", ""),
                    "prompt_tokens": prompt_tokens,
                    "truncated": bool(x.get("truncated", False)),
                    "expected_stream_eligible": bool(
                        x.get("expected_stream_eligible", False)
                    ),
                    "gold": gold,
                    "pred": pred,
                    "correct": pred == gold,
                    "output": text,
                    "error": "",
                }

            except Exception as e:
                out[j] = {
                    "source_index": source_index,
                    "_id": x.get("_id", ""),
                    "prompt_tokens": prompt_tokens,
                    "gold": normalize_label(x.get("answer", "")),
                    "pred": "",
                    "correct": False,
                    "output": "",
                    "error": repr(e),
                }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    ok = [x for x in out if not x["error"]]

    acc = (
        sum(bool(x["correct"]) for x in ok) / len(ok)
        if ok
        else 0.0
    )

    print(
        f"total={len(out)} "
        f"success={len(ok)} "
        f"errors={len(out)-len(ok)} "
        f"accuracy={acc:.6f}"
    )


if __name__ == "__main__":
    main()
