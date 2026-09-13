#!/usr/bin/env python3
"""Documented TP2 experiments; invokes the existing runner without GPU-side work."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from collections import Counter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['public', 'i3-deploy', 'i3-fixed'], required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', choices=['longbench_v2', 'mrcr16_32'],
                   default=['longbench_v2', 'mrcr16_32', 'gsm8k']) ##gsm8k
    p.add_argument('--concurrencies', nargs='+', type=int, choices=[1, 4, 8], default=[8])
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    repo = Path(os.environ.get('REPO', '/root/lifei/SpecStream'))
    data = Path(os.environ.get('QWEN3_DATA_ROOT', str(repo / 'specstream_prepared/qwen3_offline')))
    env = os.environ.copy()
    assert env.get('TARGET_TP_SIZE') == '2', 'Source the current TP2 preflight runtime_env.sh'
    assert len(set(env.get('TARGET_UUIDS', '').split(','))) == 2, 'Need two distinct Target UUIDs'
    assert env.get('SPECSTREAM_SMCTRL_VALIDATED') == '1', 'Run the current-machine preflight first'
    # Freeze the public/I3 contract even when launched after I1/I2 in the same shell.
    env.update(MODEL_TAG='qwen3_0p6b_32b', SERVER_CONTEXT_LEN='40960', FINAL_DRAFT_TPCS='34',
               SPECSTREAM_TARGET_MEM_FRACTION='0.62', SPECSTREAM_DRAFT_MEM_FRACTION='0.80',
               SPECSTREAM_TARGET_MAX_TOTAL_TOKENS='131072', SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS='196608',
               SPECSTREAM_TARGET_MIN_KV_TOKENS='131072', SPECSTREAM_DRAFT_MIN_KV_TOKENS='196608',
               SPECSTREAM_PREFILL_MAX_REQUESTS='1', SPECSTREAM_GPU_HISTORY_CACHE_TOKENS='8192',
               SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS='0', SPECSTREAM_SPLIT_KV='auto',
               SPECSTREAM_BACKGROUND_GRANT_PUMP='1', SPECSTREAM_NUM_BUFFERS='2',
               SPECSTREAM_GRANT_TOKEN_QUANTUM='1', SPECSTREAM_REQUIRE_SLACK_FILL='0',
               SPECSTREAM_DRY_RUN=str(int(a.dry_run)), CLIENT_MODE='performance',
               REQUEST_RATE='inf', SEED='1', OUTPUT_LEN='256', WARMUP_REQUESTS='4',
               CASE_TIMEOUT_S='43200', PYTHONUNBUFFERED='1', TQDM_MININTERVAL='0.5',
               PYTHONPATH=str(repo / 'python') + os.pathsep + env.get('PYTHONPATH', ''))
    paths = {'gsm8k': data / 'gsm8k_qwen3_nothink_sharegpt.json',
             'longbench_v2': data / 'longbench_v2_qwen3_8b_8k32k_sharegpt.json',
             'mrcr16_32': data / 'mrcr_qwen3_16k32k_sharegpt.json'}
    plan = []

    def add(group, method, dataset, concurrency, tp=2, mode='auto', q=0, input_len=0):
        path = paths.get(dataset)
        n = len(json.loads(path.read_text())) if path else 64
        assert n > 0
        tag = dataset if a.kind == 'public' else f'{group}_{dataset}'
        plan.append(dict(case=f'{method}_{tag}_c{concurrency}', group=group, method=method,
                         tag=tag, dataset=dataset, path=str(path) if path else '',
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path else '',
                         concurrency=concurrency, prompts=n, draft_tp=tp, mode=mode, q=q,
                         input_len=input_len))

    if a.kind == 'public':
        for d in a.datasets:
            for m in ['SPECSTREAM_1GPU', 'SGLANG_SD', 'AR', 'SGLANG_SD_KV_OFFLOAD']:
                add(m, m, d, 8 if d == 'gsm8k' else 4,
                    tp=0 if m == 'AR' else 2)
    elif a.kind == 'i3-deploy':
        for g, mode in [('D2S', 'serial'), ('D2P', 'auto')]:
            add(g, 'SPECSTREAM_1GPU', 'longbench_v2', 4, mode=mode)
    else:
        for length in [16384, 32000]:
            for c in a.concurrencies:
                for g, tp, mode in [('S1', 1, 'serial'), ('P1', 1, 'auto'),
                                    ('S2', 2, 'serial'), ('P2', 2, 'auto')]:
                    add(g, 'SPECSTREAM_1GPU', f'random{length}', c, tp, mode, 8, length)
    assert len({x['case'] for x in plan}) == len(plan), 'Duplicate cells requested'
    a.root = a.root.resolve()
    a.root.mkdir(parents=True, exist_ok=True)
    for sub in ['bench', 'logs', 'profiles', 'summary', 'env']:
        (a.root / sub).mkdir(exist_ok=True)
    plan_path = a.root / 'env/matrix_plan.json'
    assert not plan_path.exists(), 'Use a new result root; partial runs are preserved'
    plan_path.write_text(json.dumps(dict(kind=a.kind, dry_run=a.dry_run, cells=plan), indent=2))
    frozen = {k: v for k, v in env.items() if k.startswith(('SPECSTREAM_', 'TARGET_', 'DRAFT_', 'COLOCATED_'))
              or k in ['MODEL_TAG', 'SERVER_CONTEXT_LEN', 'FINAL_DRAFT_TPCS', 'PYTHONPATH']}
    (a.root / 'env/frozen_config.json').write_text(json.dumps(frozen, indent=2))
    for name, cmd in [('git_commit.txt', ['git', 'rev-parse', 'HEAD']),
                      ('git_status.txt', ['git', 'status', '--short']),
                      ('source.diff', ['git', 'diff', 'HEAD', '--', 'python/sglang', 'scripts/specstream'])]:
        (a.root / 'env' / name).write_bytes(subprocess.check_output(cmd, cwd=repo))
    script = repo / 'scripts/specstream/paper_eval/qwen3/run_public_once.sh'
    (a.root / 'env/runner.sha256').write_text(hashlib.sha256(script.read_bytes()).hexdigest() + '\n')
    reports = []
    for index, cell in enumerate(plan, 1):
        knobs = (f"mode={cell['mode']} q={cell['q']}" if cell['method'] == 'SPECSTREAM_1GPU'
                 else 'native/fixed runner preset; SpecStream mode/q switches do not apply')
        print(f"\n[{index}/{len(plan)}] {cell['case']} DraftTP={cell['draft_tp']} {knobs}", flush=True)
        case_env = dict(env, METHOD=cell['method'], DATASET_TAG=cell['tag'],
                        DATASET_NAME='sharegpt' if cell['path'] else 'random-ids',
                        DATASET_PATH=cell['path'], INPUT_LEN=str(cell['input_len']),
                        NUM_PROMPTS=str(cell['prompts']), MAX_CONCURRENCY=str(cell['concurrency']),
                        SPECSTREAM_DRAFT_TP_SIZE=str(cell['draft_tp']), SPECSTREAM_OVERLAP_MODE=cell['mode'],
                        SPECSTREAM_FIXED_Q=str(cell['q']), RESULT_ROOT=str(a.root))
        subprocess.run(['bash', str(script)], cwd=repo, env=case_env, check=True)
        case_root = a.root / 'logs' / cell['case']
        cfg = dict(line.split('=', 1) for line in (case_root / 'config.env').read_text().splitlines() if '=' in line)
        assert cfg['TARGET_TP_SIZE'] == '2'
        if cell['method'] == 'SPECSTREAM_1GPU':
            assert cfg['DRAFT_TP_SIZE'] == str(cell['draft_tp'])
            assert cfg['SPECSTREAM_OVERLAP_MODE'] == cell['mode']
            assert cfg['SPECSTREAM_FIXED_Q'] == str(cell['q'])
            assert cfg['DRAFT_VISIBLE'] == (cfg['TARGET_VISIBLE'] if cell['draft_tp'] == 2 else env['COLOCATED_UUID'])
        if a.dry_run:
            continue
        assert (case_root / 'case_complete.marker').is_file(), cell['case']
        for log_name in ['target.log', 'draft.log']:
            log_path = case_root / log_name
            if log_path.exists():
                with log_path.open(errors='replace') as stream:
                    for line in stream:
                        assert 'DraftFallback' not in line and 'RecvTimeout' not in line, (cell['case'], line.strip())
        bench = [json.loads(line) for line in (a.root / 'bench' / (cell['case'] + '.jsonl')).read_text().splitlines() if line]
        assert len(bench) == 1, cell['case']
        result = bench[0]
        assert result['completed'] == cell['prompts'], (cell['case'], result['completed'], cell['prompts'])
        assert not any(result.get('errors', [])), cell['case']
        assert result['total_output_tokens'] > 0
        report = {k: result.get(k) for k in ['output_throughput', 'request_throughput', 'accept_length',
                  'mean_ttft_ms', 'mean_tpot_ms', 'p99_e2e_latency_ms', 'total_output_tokens', 'completed']}
        report.update(case=cell['case'], group=cell['group'])
        if cell['method'] == 'SPECSTREAM_1GPU':
            gate = json.loads((case_root / 'grant_event_gate.json').read_text())
            assert gate['status'] == 'PASS', cell['case']
            rows = list(csv.DictReader((a.root / 'profiles' / (cell['case'] + '.csv')).open()))
            assert rows, cell['case']
            report.update(slack_fill_success=gate['slack_fill_success'],
                          overlap_evidence='ACTIVATED' if gate['slack_fill_success'] else 'NOT_ACTIVATED',
                          q_histogram=dict(Counter(r.get('q', '') for r in rows)),
                          mode_histogram=dict(Counter(r.get('mode', '') for r in rows)),
                          coexec_reasons=dict(Counter(r.get('coexec_reason', '') for r in rows)),
                          planned_mode_histogram=dict(Counter(r.get('controller_planned_mode', '') for r in rows)))
            if cell['mode'] == 'serial':
                assert gate['slack_fill_success'] == 0, 'Serial control unexpectedly executed SLACK_FILL'
        reports.append(report)
        (a.root / 'summary/results.json').write_text(json.dumps(reports, indent=2))
    if a.dry_run:
        print(f'TP2_MATRIX_DRY_RUN=PASS cells={len(plan)}', flush=True)
        return
    expected = {x['case'] for x in plan}
    assert expected == {x.parent.name for x in (a.root / 'logs').glob('*/case_complete.marker')}
    assert expected == {x.stem for x in (a.root / 'bench').glob('*.jsonl')}
    (a.root / 'matrix_complete.marker').write_text(f'kind={a.kind}\ncells={len(plan)}\n')
    print(f'TP2_MATRIX_COMPLETE=PASS cells={len(plan)} root={a.root}', flush=True)


if __name__ == '__main__':
    main()
