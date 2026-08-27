#!/usr/bin/env python3

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import hashlib
import json
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer


def load_rows(path):
    path = str(path)

    if path.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")

    with open(path, encoding="utf-8") as f:
        if path.endswith(".json"):
            obj = json.load(f)

            if isinstance(obj, list):
                return obj

            if isinstance(obj, dict):
                if isinstance(obj.get("data"), list):
                    return obj["data"]
                if isinstance(obj.get("examples"), list):
                    return obj["examples"]

            raise ValueError(
                f"Unsupported JSON structure: {type(obj).__name__}"
            )

        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def prompt_prefix():
    return (
        "Please read the following text and answer the question below.\n\n"
        "<text>\n"
    )


def prompt_suffix(item):
    return (
        "\n</text>\n\n"
        f"What is the correct answer to this question: {item['question']}\n"
        "Choices:\n"
        f"(A) {item['choice_A']}\n"
        f"(B) {item['choice_B']}\n"
        f"(C) {item['choice_C']}\n"
        f"(D) {item['choice_D']}\n\n"
        'Format your response as follows: '
        '"The correct answer is (insert answer here)".'
    )


def encode(tokenizer, text):
    return tokenizer.encode(
        text,
        add_special_tokens=False,
    )


def build_fast_middle_truncated_prompt(
    item,
    tokenizer,
    max_prompt_tokens,
):
    """
    Efficient LongBench-style middle truncation.

    Important:
    We DO NOT tokenize the entire multi-hundred-thousand-token document.

    For an overlong context we tokenize only sufficiently large text windows
    from the beginning and end, then retain the required token budget.
    """

    prefix = prompt_prefix()
    suffix = prompt_suffix(item)
    context = item["context"]

    prefix_ids = encode(tokenizer, prefix)
    suffix_ids = encode(tokenizer, suffix)

    # Reserve a little room for encode/decode boundary effects.
    safety_tokens = 64

    context_budget = (
        max_prompt_tokens
        - len(prefix_ids)
        - len(suffix_ids)
        - safety_tokens
    )

    if context_budget <= 1024:
        raise ValueError(
            "Question/options consume too much of the prompt budget."
        )

    head_budget = context_budget // 2
    tail_budget = context_budget - head_budget

    # ------------------------------------------------------
    # Critical optimization:
    #
    # English text is commonly several characters per token.
    # Start from 6 chars/token and enlarge only when necessary.
    #
    # For a 32K prompt this normally means tokenizing only roughly
    # the first ~100K chars + last ~100K chars, instead of a
    # million-token document.
    # ------------------------------------------------------

    head_chars = min(
        len(context),
        max(8192, head_budget * 6),
    )

    tail_chars = min(
        len(context),
        max(8192, tail_budget * 6),
    )

    # If the two windows already cover the whole document, tokenize
    # it once. This is only the relatively short-document case.
    if head_chars + tail_chars >= len(context):
        context_ids = encode(
            tokenizer,
            context,
        )

        if len(context_ids) <= context_budget:
            candidate_context_ids = context_ids
            truncated = False
        else:
            candidate_context_ids = (
                context_ids[:head_budget]
                + context_ids[-tail_budget:]
            )
            truncated = True

    else:
        # Long document:
        # never tokenize the discarded middle.
        head_text = context[:head_chars]
        tail_text = context[-tail_chars:]

        head_ids = encode(
            tokenizer,
            head_text,
        )

        tail_ids = encode(
            tokenizer,
            tail_text,
        )

        # Rare case: highly ASCII/code-heavy text may yield fewer
        # tokens than the initial estimate. Enlarge geometrically.
        while (
            len(head_ids) < head_budget
            and head_chars < len(context) // 2
        ):
            head_chars = min(
                head_chars * 2,
                len(context) // 2,
            )

            head_ids = encode(
                tokenizer,
                context[:head_chars],
            )

        while (
            len(tail_ids) < tail_budget
            and tail_chars < len(context) // 2
        ):
            tail_chars = min(
                tail_chars * 2,
                len(context) // 2,
            )

            tail_ids = encode(
                tokenizer,
                context[-tail_chars:],
            )

        candidate_context_ids = (
            head_ids[:head_budget]
            + tail_ids[-tail_budget:]
        )

        truncated = True

    candidate_context = tokenizer.decode(
        candidate_context_ids,
        skip_special_tokens=True,
    )

    prompt = (
        prefix
        + candidate_context
        + suffix
    )

    prompt_ids = encode(
        tokenizer,
        prompt,
    )

    # Boundary effects after decode -> encode may add a few tokens.
    # Enforce the hard 32K limit deterministically.
    while len(prompt_ids) > max_prompt_tokens:
        overflow = (
            len(prompt_ids)
            - max_prompt_tokens
            + 32
        )

        new_context_budget = (
            len(candidate_context_ids)
            - overflow
        )

        if new_context_budget <= 1024:
            raise RuntimeError(
                "Unable to satisfy max prompt length."
            )

        new_head = new_context_budget // 2
        new_tail = new_context_budget - new_head

        candidate_context_ids = (
            candidate_context_ids[:new_head]
            + candidate_context_ids[-new_tail:]
        )

        candidate_context = tokenizer.decode(
            candidate_context_ids,
            skip_special_tokens=True,
        )

        prompt = (
            prefix
            + candidate_context
            + suffix
        )

        prompt_ids = encode(
            tokenizer,
            prompt,
        )

    return (
        prompt,
        len(prompt_ids),
        truncated,
        len(context),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=32000,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    # We perform context management ourselves.
    # This only suppresses HF's warning.
    tokenizer.model_max_length = 10**9

    rows = load_rows(
        args.dataset
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    truncated_count = 0
    stream_eligible_count = 0
    max_seen_tokens = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for source_index, item in enumerate(
            tqdm(
                rows,
                desc="Preparing LongBench-v2 32K",
                unit="sample",
            )
        ):
            (
                prompt,
                prompt_tokens,
                truncated,
                original_context_chars,
            ) = build_fast_middle_truncated_prompt(
                item,
                tokenizer,
                args.max_prompt_tokens,
            )

            if truncated:
                truncated_count += 1

            # Current SpecStream first sealing point is ~8704 tokens.
            expected_stream_eligible = (
                prompt_tokens >= 8704
            )

            if expected_stream_eligible:
                stream_eligible_count += 1

            max_seen_tokens = max(
                max_seen_tokens,
                prompt_tokens,
            )

            row = {
                "source_index":
                    source_index,

                "_id":
                    item.get("_id", ""),

                "domain":
                    item.get("domain", ""),

                "sub_domain":
                    item.get("sub_domain", ""),

                "difficulty":
                    item.get("difficulty", ""),

                "length":
                    item.get("length", ""),

                "answer":
                    item["answer"],

                "eval_prompt":
                    prompt,

                "prompt_tokens":
                    prompt_tokens,

                "truncated":
                    truncated,

                "original_context_chars":
                    original_context_chars,

                "expected_stream_eligible":
                    expected_stream_eligible,

                "prompt_sha256":
                    hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
            }

            fout.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            # Important: make progress durable.
            fout.flush()

    print()
    print("=" * 72)
    print("LongBench-v2 preparation complete")
    print("=" * 72)
    print("samples =", len(rows))
    print("truncated =", truncated_count)
    print(
        "stream_eligible =",
        stream_eligible_count,
    )
    print(
        "max_prompt_tokens_seen =",
        max_seen_tokens,
    )
    print(
        "output =",
        output_path,
    )


if __name__ == "__main__":
    main()
