import argparse
import ast
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset

from sglang.lang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import download_and_cache_file, dump_state_text, read_jsonl

INVALID = -9999999


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def main(args):
    # Select backend
    set_default_backend(select_sglang_backend(args))

    # Qwen3 correctness runs must explicitly apply the model chat template in
    # either thinking or non-thinking mode.  Other models retain the legacy raw
    # completion behavior when neither flag is set.
    if args.enable_thinking and args.disable_thinking:
        raise ValueError("--enable-thinking and --disable-thinking are mutually exclusive")
    tokenizer = None
    if args.enable_thinking or args.disable_thinking:
        from transformers import AutoTokenizer

        assert (
            args.tokenizer_path is not None
        ), "--tokenizer-path is required when --enable-thinking is set"
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )

    # Read data
    if args.platinum:
        print("Loading GSM8K Platinum dataset from HuggingFace...")
        dataset = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
        lines = [
            {"question": item["question"], "answer": item["answer"]} for item in dataset
        ]
    else:
        data_path = args.data_path
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        if not os.path.isfile(data_path):
            data_path = download_and_cache_file(url)
        lines = list(read_jsonl(data_path))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_lines = lines
    if args.few_shot_data_path:
        if not os.path.isfile(args.few_shot_data_path):
            raise FileNotFoundError(args.few_shot_data_path)
        few_shot_lines = list(read_jsonl(args.few_shot_data_path))
    if len(few_shot_lines) < num_shots:
        raise ValueError(
            f"few-shot source has {len(few_shot_lines)} rows; need {num_shots}"
        )
    few_shot_examples = get_few_shot_examples(few_shot_lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        raw_question = few_shot_examples + get_one_example(lines, i, False)
        if tokenizer is not None:
            messages = [{"role": "user", "content": raw_question}]
            raw_question = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        questions.append(raw_question)
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def few_shot_gsm8k(s, question):
        s += question
        s += sgl.gen(
            "answer",
            max_tokens=args.max_new_tokens,
            stop=["Question", "Assistant:", "<|separator|>"],
        )

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch(
        arguments,
        temperature=args.temperature,
        top_p=args.top_p,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    preds = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]["answer"]))

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Compute speed
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")

    # Dump results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)
    if args.raw_result_file:
        raw_path = Path(args.raw_result_file)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_rows = []
        for i, state in enumerate(states):
            output = state["answer"]
            meta = state.get_meta_info("answer")
            raw_rows.append(
                {
                    "dataset": "gsm8k",
                    "ordinal": i,
                    "sample_id": f"gsm8k-{i:05d}",
                    "reference_answer": lines[i]["answer"],
                    "metric": "gsm8k_numeric_exact_match",
                    "generated_text": output,
                    "predicted_value": preds[i],
                    "reference_value": labels[i],
                    "correct": bool(preds[i] == labels[i]),
                    "error": None,
                    "meta_info": meta,
                }
            )
        raw_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in raw_rows) + "\n",
            encoding="utf-8",
        )
        print(f"GSM8K detailed results saved to {raw_path}")

    if args.result_file:
        result_path = Path(args.result_file)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w", encoding="utf-8") as fout:
            value = {
                "task": "gsm8k-platinum" if args.platinum else "gsm8k",
                "method": args.method,
                "model": args.tokenizer_path or "raw-completion",
                "backend": args.backend,
                "num_gpus": 1,
                "latency": round(latency, 3),
                "accuracy": round(acc, 3),
                "invalid_fraction": round(float(invalid), 6),
                "request_errors": 0,
                "num_requests": len(questions),
                "correct": int(np.sum(np.array(preds) == np.array(labels))),
                "other": {
                    "num_questions": len(questions),
                    "parallel": args.parallel,
                    "num_shots": num_shots,
                    "few_shot_data_path": args.few_shot_data_path,
                    "enable_thinking": args.enable_thinking,
                },
            }
            fout.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        print(f"GSM8K summary saved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument(
        "--few-shot-data-path",
        type=str,
        default=None,
        help="Optional train-split JSONL used only for few-shot demonstrations.",
    )
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode by wrapping prompts with chat template",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Apply the tokenizer chat template with enable_thinking=False (Qwen3).",
    )
    parser.add_argument("--method", type=str, default="")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer (required when --enable-thinking is set)",
    )
    parser.add_argument(
        "--platinum",
        action="store_true",
        help="Use GSM8K Platinum dataset (drop-in replacement with corrected labels)",
    )
    args = add_common_sglang_args_and_parse(parser)
    main(args)
