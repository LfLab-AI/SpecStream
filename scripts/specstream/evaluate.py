#!/usr/bin/env python3
"""Prepare a reproducible workload and evaluate a running generation server."""

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prompt_ids(tokenizer, prompt):
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    return encoded["input_ids"] if isinstance(encoded, Mapping) else encoded


def load_rows(kind, source):
    if source:
        path = Path(source)
        if path.is_file() and path.suffix in (".json", ".jsonl"):
            with path.open(encoding="utf-8") as stream:
                return ([json.loads(line) for line in stream if line.strip()]
                        if path.suffix == ".jsonl" else json.load(stream))
        from datasets import load_dataset, load_from_disk
        if path.is_dir():
            data = load_from_disk(str(path))
            return data if hasattr(data, "column_names") and isinstance(data.column_names, list) else data["test" if kind == "gsm8k" else "train"]
        if path.suffix == ".parquet" and path.is_file():
            return load_dataset("parquet", data_files=str(path), split="train")
        raise ValueError(f"Unsupported dataset path: {path}")
    from datasets import load_dataset
    if kind == "gsm8k":
        return load_dataset("openai/gsm8k", "main", split="test")
    return load_dataset("zai-org/LongBench-v2", split="train")


def make_prompt(kind, row):
    if kind == "gsm8k":
        return (row["question"] + "\nSolve the problem and end with '#### <answer>'.",
                row["answer"].rsplit("####", 1)[-1].strip())
    choices = "\n".join(f"{letter}. {row['choice_' + letter]}" for letter in "ABCD")
    return (f"Read the document and answer the question.\n\n{row['context']}\n\n"
            f"{row['question']}\n{choices}\nReply with only the letter A, B, C, or D.",
            row["answer"].strip().upper())


def score_answer(kind, text, reference):
    if kind == "synthetic":
        return None
    if kind == "longbench-v2":
        match = re.fullmatch(r"\s*(?:[Aa]nswer\s*:\s*)?\(?([ABCD])\)?[.。]?\s*", text)
        return bool(match and match.group(1) == reference)
    candidate = text.rsplit("####", 1)[-1] if "####" in text else text
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", candidate)
    if not numbers:
        return False
    try:
        return Decimal(numbers[-1].replace(",", "")) == Decimal(reference.replace(",", ""))
    except ArithmeticError:
        return False


def prepare(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)
    selected = []
    if args.kind == "synthetic":
        if args.input_tokens > args.max_input_tokens:
            raise ValueError("input-tokens exceeds max-input-tokens")
        base = tokenizer.encode("The document describes a sequence of measurements and observations. ", add_special_tokens=False)
        for index in range(args.samples):
            prefix = tokenizer.encode(f"Document {index}: ", add_special_tokens=False)
            ids = (prefix + base * (args.input_tokens // len(base) + 1))[:args.input_tokens]
            selected.append({"id": str(index), "input_ids": ids, "reference": None})
    else:
        rows = load_rows(args.kind, args.source)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        for index in indices:
            row = rows[index]
            prompt, reference = make_prompt(args.kind, row)
            ids = prompt_ids(tokenizer, prompt)
            if args.min_input_tokens <= len(ids) <= args.max_input_tokens:
                selected.append({"id": str(row.get("_id", index)), "input_ids": ids, "reference": reference})
            if len(selected) == args.samples:
                break
        if len(selected) != args.samples:
            raise ValueError(f"Found {len(selected)} examples in the token range; requested {args.samples}")
    if any(len(item["input_ids"]) + args.output_tokens > args.context_length for item in selected):
        raise ValueError("Input plus output exceeds context-length")
    vocab = json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False).encode()
    workload = {
        "kind": args.kind, "source": args.source, "seed": args.seed,
        "tokenizer": args.tokenizer, "vocab_sha256": hashlib.sha256(vocab).hexdigest(),
        "chat_template": tokenizer.chat_template, "enable_thinking": False,
        "output_tokens": args.output_tokens, "context_length": args.context_length,
        "ignore_eos": args.kind == "synthetic", "samples": selected,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    write_json(output, workload)
    print(f"Prepared {len(selected)} examples; input tokens={sum(len(s['input_ids']) for s in selected)}; {output}")


def request_json(url, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    handlers = ([urllib.request.ProxyHandler({})]
                if urllib.parse.urlsplit(url).hostname in ("127.0.0.1", "localhost", "::1") else [])
    with urllib.request.build_opener(*handlers).open(request, timeout=timeout) as response:
        return json.load(response)


def generate(base_url, workload, sample, timeout):
    start = time.perf_counter()
    try:
        result = request_json(base_url + "/generate", {
            "input_ids": sample["input_ids"], "stream": False,
            "sampling_params": {"temperature": 0, "top_p": 1,
                                "max_new_tokens": workload["output_tokens"],
                                "ignore_eos": workload["ignore_eos"]},
        }, timeout)
        meta = result["meta_info"]
        reason = meta.get("finish_reason", {})
        if reason.get("type") not in ("stop", "length"):
            raise RuntimeError(f"Unexpected finish reason: {reason}")
        if meta["prompt_tokens"] != len(sample["input_ids"]):
            raise RuntimeError("Server prompt token count differs from prepared input")
        if workload["ignore_eos"] and meta["completion_tokens"] != workload["output_tokens"]:
            raise RuntimeError("Fixed-length generation ended early")
        return {"id": sample["id"], "text": result["text"], "reference": sample["reference"],
                "correct": score_answer(workload["kind"], result["text"], sample["reference"]),
                "latency_s": time.perf_counter() - start, "meta_info": meta, "error": None}
    except Exception as exc:
        return {"id": sample["id"], "error": f"{type(exc).__name__}: {exc}",
                "latency_s": time.perf_counter() - start}


def run(args):
    payload = Path(args.workload).read_bytes()
    workload = json.loads(payload)
    samples = workload["samples"]
    if not samples:
        raise ValueError("Empty workload")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    base_url = args.base_url.rstrip("/")
    server = request_json(base_url + "/get_model_info", timeout=30)
    write_json(output / "config.json", {**vars(args), "server": server,
               "workload_sha256": hashlib.sha256(payload).hexdigest()})
    for index in range(args.warmup):
        result = generate(base_url, workload, samples[index % len(samples)], args.timeout)
        if result["error"]:
            write_json(output / "warmup_error.json", result)
            raise RuntimeError(f"Warmup failed: {result['error']}")
    start = time.perf_counter()
    results = []
    with (output / "generations.jsonl").open("w", encoding="utf-8") as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(generate, base_url, workload, sample, args.timeout) for sample in samples]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"Completed {len(results)}/{len(samples)}", flush=True)
    elapsed = time.perf_counter() - start
    passed = [result for result in results if result["error"] is None]
    tokens = sum(result["meta_info"]["completion_tokens"] for result in passed)
    summary = {
        "kind": workload["kind"], "requested": len(samples), "completed": len(passed),
        "errors": len(results) - len(passed), "concurrency": args.concurrency,
        "output_tokens": tokens, "wall_time_s": elapsed, "output_tokens_per_s": tokens / elapsed,
        "requests_per_s": len(passed) / elapsed,
        "mean_request_latency_s": sum(r["latency_s"] for r in passed) / len(passed) if passed else None,
        "accuracy": (sum(r["correct"] for r in passed) / len(samples)
                     if workload["kind"] != "synthetic" else None),
        "warmup_requests": args.warmup,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["errors"]:
        raise SystemExit(1)
    (output / "complete.marker").write_text("PASS\n", encoding="utf-8")


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--kind", choices=["synthetic", "gsm8k", "longbench-v2"], required=True)
    prep.add_argument("--tokenizer", required=True)
    prep.add_argument("--source", help="Raw JSON, JSONL, Parquet, or a datasets.save_to_disk directory")
    prep.add_argument("--samples", type=positive, default=16)
    prep.add_argument("--seed", type=int, default=1)
    prep.add_argument("--input-tokens", type=positive, default=8192)
    prep.add_argument("--min-input-tokens", type=int, default=0)
    prep.add_argument("--max-input-tokens", type=positive, default=15360)
    prep.add_argument("--output-tokens", type=positive, default=128)
    prep.add_argument("--context-length", type=positive, default=16384)
    prep.add_argument("--output", required=True)
    test = commands.add_parser("run")
    test.add_argument("--workload", required=True)
    test.add_argument("--base-url", default="http://127.0.0.1:30000")
    test.add_argument("--concurrency", type=positive, default=4)
    test.add_argument("--warmup", type=int, default=1)
    test.add_argument("--timeout", type=positive, default=1800)
    test.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if getattr(args, "warmup", 0) < 0 or getattr(args, "min_input_tokens", 0) < 0:
        parser.error("warmup and min-input-tokens must be nonnegative")
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
