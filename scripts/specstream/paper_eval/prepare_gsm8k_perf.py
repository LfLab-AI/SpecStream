#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
args = ap.parse_args()

rows = [
    json.loads(x)
    for x in open(args.input, encoding="utf-8")
    if x.strip()
]

out = []

for x in rows:
    prompt = (
        "Solve the following math problem carefully and show the reasoning.\n\n"
        f"Question: {x['question']}\n\n"
        "Answer:"
    )

    out.append(
        {
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": str(x["answer"])},
            ]
        }
    )

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(
    json.dumps(out, ensure_ascii=False),
    encoding="utf-8",
)

print("rows =", len(out))
print("WROTE", args.output)
