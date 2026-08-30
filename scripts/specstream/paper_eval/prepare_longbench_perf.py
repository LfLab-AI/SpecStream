#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--server-context", type=int, default=32768)
ap.add_argument("--output-len", type=int, default=256)
args = ap.parse_args()

max_prompt = args.server_context - args.output_len

src = [
    json.loads(x)
    for x in open(args.input, encoding="utf-8")
    if x.strip()
]

out = []
lens = []

for x in src:
    n = int(x["prompt_tokens"])

    if n > max_prompt:
        continue

    out.append(
        {
            "conversations": [
                {"from": "human", "value": x["eval_prompt"]},
                {"from": "gpt", "value": str(x["answer"])},
            ],
            "_id": x.get("_id", ""),
            "prompt_tokens": n,
        }
    )
    lens.append(n)

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(
    json.dumps(out, ensure_ascii=False),
    encoding="utf-8",
)

print("source rows =", len(src))
print("kept rows   =", len(out))
print("min prompt  =", min(lens))
print("max prompt  =", max(lens))
print("WROTE", args.output)

assert out
assert max(lens) + args.output_len <= args.server_context
