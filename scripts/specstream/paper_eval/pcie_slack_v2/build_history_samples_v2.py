#!/usr/bin/env python3
import argparse, csv, json, statistics
from collections import defaultdict
from pathlib import Path

def num(row, key):
    try: return float(row.get(key) or 0)
    except Exception: return 0.0

def bucket(tokens):
    for limit,label in ((2048,'2k'),(4096,'4k'),(8192,'8k'),(16384,'16k'),(32768,'32k'),(65536,'64k')):
        if tokens <= limit: return label
    return '64k+'

def load(path):
    return list(csv.DictReader(Path(path).open(encoding='utf-8')))

def aggregate(base_path, overlap_path):
    base=defaultdict(list)
    for r in load(base_path):
        q=int(num(r,'q')); t=num(r,'target_forward_ms')
        if q<=0 or t<=0: continue
        bs=max(int(num(r,'batch_size') or 1),1)
        key=(bs,q,bucket(int(num(r,'context_tokens'))))
        base[key].append(t)

    overlap=defaultdict(lambda:{'draft':[],'target':[]})
    for r in load(overlap_path):
        q=int(num(r,'q')); bs=max(int(num(r,'batch_size') or 1),1)
        low=int(num(r,'draft_tpc_low')); high=int(num(r,'draft_tpc_high')); tpcs=high-low
        if (r.get('target_phase')!='history_h2d' or
            r.get('grant_state')!='SLACK_FILL' or
            num(r,'history_len')<=0 or num(r,'h2d_ops')<=0 or
            num(r,'exposed_copy_ms')<=0 or num(r,'draft_step_ms')<=0 or
            num(r,'target_forward_ms')<=0 or q<=0 or tpcs<=0):
            continue
        key=(bs,q,bucket(int(num(r,'context_tokens'))),tpcs)
        overlap[key]['draft'].append(num(r,'draft_step_ms'))
        overlap[key]['target'].append(num(r,'target_forward_ms'))

    out={}
    for key,v in overlap.items():
        base_key=key[:3]
        if base_key not in base: continue
        bs,q,ctx,tpcs=key
        out[key]={
            'target_shape':f'verify_bs{bs}_q{q}_ctx{ctx}',
            'draft_bs':bs,
            'draft_ctx_bucket':ctx,
            'draft_tpcs':tpcs,
            'slack_source':'history_h2d',
            'draft_step_ms':statistics.median(v['draft']),
            'target_latency_ms':statistics.median(v['target']),
            'target_baseline_ms':statistics.median(base[base_key]),
        }
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pair', nargs=2, action='append', required=True, metavar=('BASE','OVERLAP'))
    ap.add_argument('--output', required=True)
    a=ap.parse_args()
    if len(a.pair)<3: raise SystemExit('need >=3 matched repetitions')
    groups=[aggregate(x,y) for x,y in a.pair]
    common=set(groups[0])
    for g in groups[1:]: common &= set(g)
    if not common: raise SystemExit('no common history_h2d shape across repetitions')
    dst=Path(a.output); dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open('w',encoding='utf-8') as h:
        for key in sorted(common):
            for g in groups: h.write(json.dumps(g[key])+'\n')
    print('common keys:')
    for key in sorted(common): print(' ',key)
    print('WROTE',dst)
if __name__=='__main__': main()
