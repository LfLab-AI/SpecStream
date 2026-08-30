#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from transformers import AutoTokenizer


def load_rows(path):
    if path.endswith('.parquet'):
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient='records')

    # Preferred paper path is the prepared JSONL file.  Keep raw data.json
    # compatibility as a safeguard.
    with open(path, encoding='utf-8') as f:
        if path.endswith('.json'):
            obj = json.load(f)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and isinstance(obj.get('data'), list):
                return obj['data']
            if isinstance(obj, dict) and isinstance(obj.get('examples'), list):
                return obj['examples']
            raise ValueError(f'Unsupported JSON structure: {type(obj).__name__}')
        return [json.loads(x) for x in f if x.strip()]


def prompt_of(x):
    return (
        f"{x['context']}\n\n"
        f"Question: {x['question']}\n"
        f"A. {x['choice_A']}\n"
        f"B. {x['choice_B']}\n"
        f"C. {x['choice_C']}\n"
        f"D. {x['choice_D']}\n"
        "Answer:"
    )


def normalize_label(x):
    s=str(x).strip().upper()
    m=re.search(r'\b([ABCD])\b', s)
    return m.group(1) if m else ''


def infer(base_url, prompt, max_new_tokens):
    payload={
        'text': prompt,
        'sampling_params': {
            'temperature': 0,
            'top_p': 1,
            'max_new_tokens': max_new_tokens,
            'ignore_eos': False,
        },
        'stream': False,
    }
    r=requests.post(base_url.rstrip('/') + '/generate', json=payload, timeout=1800)
    r.raise_for_status()
    obj=r.json()
    text=obj.get('text','')
    return text, normalize_label(text)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--base-url', default='http://127.0.0.1:30000')
    ap.add_argument('--answer-field', default='answer')
    ap.add_argument('--max-new-tokens', type=int, default=32)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--min-prompt-tokens', type=int, default=0)
    ap.add_argument('--output', required=True)
    args=ap.parse_args()

    tok=AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    src=load_rows(args.dataset)
    selected=[]
    for i,x in enumerate(src):
        p=prompt_of(x)
        n=len(tok(p).input_ids)
        if n < args.min_prompt_tokens:
            continue
        selected.append((i,x,p,n))
        if args.limit and len(selected)>=args.limit:
            break

    print('selected examples =', len(selected))
    out=[None]*len(selected)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs={ex.submit(infer,args.base_url,p,args.max_new_tokens):j
              for j,(_,_,p,_) in enumerate(selected)}
        for fut in as_completed(futs):
            j=futs[fut]
            i,x,p,n=selected[j]
            try:
                text,pred=fut.result()
                gold=normalize_label(x[args.answer_field])
                out[j]={
                    'source_index': i,
                    'prompt_tokens': n,
                    'gold': gold,
                    'pred': pred,
                    'correct': pred==gold,
                    'output': text,
                    'error': '',
                }
            except Exception as e:
                out[j]={
                    'source_index': i,
                    'prompt_tokens': n,
                    'gold': normalize_label(x.get(args.answer_field,'')),
                    'pred': '',
                    'correct': False,
                    'output': '',
                    'error': repr(e),
                }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output,'w',encoding='utf-8') as f:
        for x in out:
            f.write(json.dumps(x,ensure_ascii=False)+'\n')

    ok=[x for x in out if not x['error']]
    acc=sum(x['correct'] for x in ok)/len(ok) if ok else 0.0
    print(f'total={len(out)} success={len(ok)} errors={len(out)-len(ok)} accuracy={acc:.6f}')

if __name__=='__main__':
    main()
