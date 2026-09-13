#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PATTERN = re.compile(
    r'\[Draft\]\[Grant\] ACK '
    r'rid=(\S+) spec_cnt=(\d+) epoch=(\d+) tokens=(\d+) '
    r'state=(\S+) step_ms=([0-9.]+)'
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft-log', required=True)
    parser.add_argument('--q', type=int, required=True)
    parser.add_argument('--fixed-tpcs', type=int, default=27)
    parser.add_argument('--output', required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()

    counts = Counter()
    slack_by_round = defaultdict(int)
    last_epoch = {}
    positive_step_ms = []

    for line in Path(args.draft_log).open(encoding='utf-8', errors='replace'):
        match = PATTERN.search(line)
        if not match:
            continue
        rid, spec_cnt, epoch, tokens, state, step_ms = match.groups()
        key = (rid, int(spec_cnt))
        epoch = int(epoch)
        tokens = int(tokens)
        step_ms = float(step_ms)

        if epoch <= last_epoch.get(key, -1):
            raise SystemExit(f'non-monotonic epoch for {key}: {epoch}')
        last_epoch[key] = epoch
        counts[state] += 1

        if state == 'SLACK_FILL' and tokens == 1 and step_ms > 0:
            slack_by_round[key] += 1
            positive_step_ms.append(step_ms)

    complete = sum(value >= args.q for value in slack_by_round.values())
    payload = {
        'fixed_draft_tpcs': args.fixed_tpcs,
        'q': args.q,
        'ack_states': dict(counts),
        'slack_rounds': len(slack_by_round),
        'complete_q_sequences': complete,
        'positive_slack_acks': sum(slack_by_round.values()),
        'draft_step_ms_min': min(positive_step_ms) if positive_step_ms else None,
        'draft_step_ms_max': max(positive_step_ms) if positive_step_ms else None,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))

    if args.require_complete:
        if payload['positive_slack_acks'] <= 0:
            raise SystemExit('no positive SLACK_FILL ACK')
        if complete < 1:
            raise SystemExit(f'no complete q={args.q} SLACK_FILL sequence')

if __name__ == '__main__':
    main()
