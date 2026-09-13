# SpecStream Qwen3 数据集重生成、准确性标注与验证手册

> 原始数据必须来自服务器本地：`/root/autodl-tmp/dataset/{gsm8k,LongBench-v2,mrcr}`。  
> 使用 Qwen3-8B Target tokenizer 和官方 Qwen3 chat template，`enable_thinking=false`。  
> 产物同时服务于 01 的准确性和性能实验；任何源数据、tokenizer 或格式脚本改变都必须整体重生成。

> 不要在交互终端设置 `set -euo pipefail`，本手册不使用 `exit`。Gate 失败时只打印 `ERROR`，用户保留在当前 SSH shell 中进行修复。

## 1. 产物和设计原则

输出目录：`/root/lifei/SpecStream/specstream_prepared/qwen3_offline`。

| 文件 | 用途 |
|---|---|
| `gsm8k_main_test.jsonl` | GSM8K 原生 evaluator 的 1319 条 main/test |
| `gsm8k_main_train_fewshot5.jsonl` | GSM8K 5-shot main/train 示例，与 test 隔离 |
| `gsm8k_native_eval_manifest.json` | split、模型、thinking 和 tokenizer 验证记录 |
| `gsm8k_native_eval_sha256.txt` | GSM8K 原生 evaluator 输入校验 |
| `gsm8k_qwen3_nothink_sharegpt.json` | GSM8K 性能/smoke manifest，不用于正式准确率 |
| `longbench_v2_qwen3_8b_8k32k_sharegpt.json` | LongBench-v2 8K–约39K |
| `longbench_v2_qwen3_8b_8k32k_metadata.jsonl` | 原始索引、答案、长度、截断标记 |
| `mrcr_qwen3_16k32k_sharegpt.json` | MRCR 16K–约39K |
| `mrcr_qwen3_16k32k_metadata.jsonl` | needle、marker、答案和长度 |
| `tokenizer_manifest.json` | 模型路径、vocab SHA256、context/output 口径 |
| `dataset_stats.json` | 行数和 token 长度统计 |
| `dataset_sha256.txt` | 所有正式输入的不可变校验 |

每行必须保留 `sample_id`、`reference_answer`、`metric` 和 `prompt_tokens`。MRCR 额外保留 `n_needles` 和 `random_string_to_prepend`。LongBench 额外保留 `category/domain`、`sub_domain`、`difficulty` 和 `length_bucket`，用于分层审计；超过上下文的行直接排除，不允许中间截断后作为准确性题目，因为截断可能删除决定答案的证据。LongBench 正确性生成上限固定为 256 tokens，评分严格复现 `extract_longbench_v2_answer()` 的规则，不能使用“最后一个 A-D”通用正则。

## 2. 环境与原始数据快照

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/miniconda3/envs/spectre
cd /root/lifei/SpecStream
export REPO=$PWD
export SPECSTREAM_PYTHON=/root/miniconda3/envs/spectre/bin/python
export PYTHONPATH=$REPO/python:${PYTHONPATH:-}
export TARGET_MODEL=/root/autodl-tmp/model/Qwen3-8B
export DRAFT_MODEL=/root/autodl-tmp/model/Qwen3-0.6B
export GSM8K_SOURCE=/root/autodl-tmp/dataset/gsm8k/main/test-00000-of-00001.parquet
export GSM8K_TRAIN_SOURCE=/root/autodl-tmp/dataset/gsm8k/main/train-00000-of-00001.parquet
export LONGBENCH_SOURCE=/root/autodl-tmp/dataset/LongBench-v2
export MRCR_SOURCE=/root/autodl-tmp/dataset/mrcr
export FINAL_ROOT=$REPO/specstream_prepared/qwen3_offline
export BUILD_ROOT=$REPO/specstream_prepared/qwen3_offline.build.$(date +%Y%m%d_%H%M%S)

for p in "$TARGET_MODEL" "$DRAFT_MODEL" "$GSM8K_SOURCE" "$GSM8K_TRAIN_SOURCE" "$LONGBENCH_SOURCE" "$MRCR_SOURCE"; do
  test -e "$p" || echo "ERROR: missing $p" >&2
done
find /root/autodl-tmp/dataset -maxdepth 3 -type f -printf '%p\t%s\n' | sort \
  | tee /tmp/qwen3_raw_dataset_files.tsv
```

目的：记录真正使用的本地文件，而不是只记录目录名。GSM8K 必须明确使用官方 `main/test` 的全部 1319 条，不能把 `main/train` 或 `socratic` split 合并进测试集。服务器离线时禁止自动下载缺失 split。

## 3. tokenizer 兼容性硬 Gate

```bash
"$SPECSTREAM_PYTHON" - <<'PY'
import hashlib, json, os
from transformers import AutoTokenizer
paths=[os.environ['TARGET_MODEL'],os.environ['DRAFT_MODEL']]
t=[AutoTokenizer.from_pretrained(p,trust_remote_code=True,use_fast=True,local_files_only=True) for p in paths]
assert t[0].get_vocab()==t[1].get_vocab(), 'Target/Draft vocab mapping differs'
for key in ['bos_token_id','eos_token_id','pad_token_id']:
    assert getattr(t[0],key)==getattr(t[1],key), key
sample=[{'role':'user','content':'Return only OK.'}]
for tok in t:
    text=tok.apply_chat_template(sample,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    print(text)
raw=json.dumps(sorted(t[0].get_vocab().items()),ensure_ascii=False).encode()
print('vocab_sha256',hashlib.sha256(raw).hexdigest())
print('TOKENIZER_GATE=PASS')
PY
```

只有完整 token-to-id mapping 和特殊 token 一致才可继续。仅仅词表大小相同不够。

## 4. 原始 schema 审计

```bash
"$SPECSTREAM_PYTHON" - <<'PY'
import importlib.util, os
from pathlib import Path
spec=importlib.util.spec_from_file_location('prepare',Path('scripts/specstream/paper_eval/qwen3/prepare_qwen3_offline.py'))
prepare=importlib.util.module_from_spec(spec); spec.loader.exec_module(prepare)
for name,env in [('gsm8k','GSM8K_SOURCE'),('longbench','LONGBENCH_SOURCE'),('mrcr','MRCR_SOURCE')]:
    rows=prepare.load_rows(Path(os.environ[env]))
    assert rows
    keys=sorted(set().union(*(r.keys() for r in rows[:50])))
    print(name,'rows=',len(rows),'keys=',keys)
PY
```

期望字段：GSM8K 至少有 `question,answer`；LongBench 至少有 `context,question,answer` 以及 A-D/choices；MRCR 至少有 `prompt,answer,n_needles,random_string_to_prepend`。字段不匹配时先修改转换器并审查，不得靠空字符串继续。

## 5. 原子式重生成

```bash
rm -rf -- "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT"

"$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/prepare_qwen3_offline.py \
  --target-model "$TARGET_MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --gsm8k-source "$GSM8K_SOURCE" \
  --longbench-source "$LONGBENCH_SOURCE" \
  --mrcr-source "$MRCR_SOURCE" \
  --output-root "$BUILD_ROOT" \
  --context-length 40960 \
  --reserved-output-length 1024 \
  --minimum-rows 64 \
  2>&1 | tee "$BUILD_ROOT/prepare_console.log"

grep -q 'QWEN3_OFFLINE_DATA_GATE=PASS' "$BUILD_ROOT/prepare_console.log"

"$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/prepare_gsm8k_native_eval.py \
  --test-source "$GSM8K_SOURCE" \
  --train-source "$GSM8K_TRAIN_SOURCE" \
  --target-model "$TARGET_MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --output-root "$BUILD_ROOT" \
  --num-shots 5 --expected-test-rows 1319 \
  2>&1 | tee "$BUILD_ROOT/prepare_gsm8k_native_console.log"

grep -q 'GSM8K_NATIVE_DATA_GATE=PASS' "$BUILD_ROOT/prepare_gsm8k_native_console.log"
test "$(grep -cve '^$' "$BUILD_ROOT/gsm8k_main_test.jsonl")" = 1319
test "$(grep -cve '^$' "$BUILD_ROOT/gsm8k_main_train_fewshot5.jsonl")" = 5
(cd "$BUILD_ROOT" && sha256sum -c gsm8k_native_eval_sha256.txt)
```

说明：40960 是两个模型配置允许的上下文上限；保留 1024 给 MRCR 最长输出。LongBench 保留 8K 以上且未截断的行，MRCR 保留 16K 以上且不溢出的行。输出文件名中的 `8k32k/16k32k` 为历史命名，权威范围以每行 `prompt_tokens` 和 `tokenizer_manifest.json` 为准；论文中应写实际 min/max，不能只照文件名。

## 6. 内容、长度和标注完整性验证

```bash
"$SPECSTREAM_PYTHON" - <<'PY'
import json, os, re
from collections import Counter
root=os.environ['BUILD_ROOT']
files={
 'gsm8k':'gsm8k_qwen3_nothink_sharegpt.json',
 'longbench':'longbench_v2_qwen3_8b_8k32k_sharegpt.json',
 'mrcr':'mrcr_qwen3_16k32k_sharegpt.json'}
for ds,name in files.items():
    rows=json.load(open(os.path.join(root,name),encoding='utf-8'))
    assert len(rows)>=64
    ids=[r['sample_id'] for r in rows]
    assert len(ids)==len(set(ids)), f'duplicate ids in {ds}'
    for r in rows:
        assert r['reference_answer'] not in ['', 'N/A']
        assert 0 < int(r['prompt_tokens']) + 1024 <= 40960
        assert r['conversations'][0]['from']=='human'
    print(ds,'rows',len(rows),'range',min(r['prompt_tokens'] for r in rows),max(r['prompt_tokens'] for r in rows))
    if ds=='gsm8k':
        assert all(re.search(r'####\s*[-+]?\d',r['reference_answer'].replace(',','')) for r in rows)
    if ds=='longbench':
        assert all(str(r['reference_answer']).strip().upper() in {'A','B','C','D'} for r in rows)
        assert all({'category','sub_domain','difficulty','length_bucket'} <= set(r) for r in rows)
    if ds=='mrcr':
        counts=Counter(str(r['n_needles']) for r in rows)
        assert all(r['random_string_to_prepend'] for r in rows)
        print('mrcr needle counts',dict(counts))
print('ANNOTATION_GATE=PASS')
PY
```

若本地 LongBench `answer` 不是单字符 A-D，而是其它官方 schema，必须先核对 formatter/评分器，不能通过放宽断言掩盖 schema 错误。

## 7. tokenizer 重新计数验证

转换器已经记录 token 数，但正式冻结前必须独立重算：

```bash
"$SPECSTREAM_PYTHON" - <<'PY'
import json, os
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(os.environ['TARGET_MODEL'],trust_remote_code=True,use_fast=True,local_files_only=True)
root=os.environ['BUILD_ROOT']
for name in ['gsm8k_qwen3_nothink_sharegpt.json','longbench_v2_qwen3_8b_8k32k_sharegpt.json','mrcr_qwen3_16k32k_sharegpt.json']:
    rows=json.load(open(os.path.join(root,name),encoding='utf-8'))
    for i,r in enumerate(rows):
        actual=len(tok.encode(r['conversations'][0]['value'],add_special_tokens=False))
        assert actual==int(r['prompt_tokens']), (name,i,actual,r['prompt_tokens'])
        assert actual+1024<=40960
    print(name,'verified',len(rows))
print('TOKEN_COUNT_GATE=PASS')
PY
```

## 8. MRCR 全量合格样本组成审计

准确性和性能主实验使用 MRCR manifest 中所有通过长度、schema 和标注 Gate 的样本，不再平衡抽取 96 条。下面只统计全量样本的 needle 组成，不执行采样：

```bash
"$SPECSTREAM_PYTHON" - <<'PY'
import json, os
from collections import Counter
p=os.path.join(os.environ['BUILD_ROOT'],'mrcr_qwen3_16k32k_sharegpt.json')
rows=json.load(open(p,encoding='utf-8'))
assert rows and len(rows)==len({r['sample_id'] for r in rows})
print(Counter(str(r['n_needles']) for r in rows))
print('all_qualified_mrcr_rows',len(rows))
PY
```

将 2/4/8 needle 的实际数量写入实验结果。各组不均衡时保持原始合格数据分布，不得为了好看删除多数组样本；可以额外报告按 needle 分层指标，但总体指标必须覆盖全量合格样本。

## 9. checksum 与冻结

```bash
cat "$BUILD_ROOT/dataset_stats.json"
cat "$BUILD_ROOT/tokenizer_manifest.json"
(cd "$BUILD_ROOT" && sha256sum -c dataset_sha256.txt)
test -s "$BUILD_ROOT/prepare_console.log"

if test -e "$FINAL_ROOT"; then
  export BACKUP_ROOT=${FINAL_ROOT}.backup.$(date +%Y%m%d_%H%M%S)
  mv -- "$FINAL_ROOT" "$BACKUP_ROOT"
  echo "old dataset moved to $BACKUP_ROOT"
fi
mv -- "$BUILD_ROOT" "$FINAL_ROOT"
(cd "$FINAL_ROOT" && sha256sum -c dataset_sha256.txt)
```

这里先在独立 build 目录完成所有 Gate，最后才原子切换。旧目录移动到时间戳备份，可恢复；确认新数据完成论文实验之前不要删除备份。

## 10. 最小服务 smoke

数据内容 Gate 通过后再做服务 smoke；它不产生论文性能结果。

```bash
export QWEN3_DATA_ROOT=$FINAL_ROOT
export GSM8K_QWEN3=$FINAL_ROOT/gsm8k_qwen3_nothink_sharegpt.json
export SMOKE_ROOT=$REPO/results/qwen3_data_smoke_$(date +%Y%m%d_%H%M%S)

METHOD=SGLANG_SD DATASET_TAG=data_smoke DATASET_NAME=sharegpt \
DATASET_PATH="$GSM8K_QWEN3" NUM_PROMPTS=8 OUTPUT_LEN=32 \
MAX_CONCURRENCY=2 WARMUP_REQUESTS=1 REQUEST_RATE=inf SEED=1 \
RESULT_ROOT="$SMOKE_ROOT" CASE_TIMEOUT_S=1200 \
bash scripts/specstream/paper_eval/qwen3/run_public_once.sh

test -s "$SMOKE_ROOT/bench/SGLANG_SD_data_smoke_c2.jsonl"
grep -Ei 'Traceback|CUDA out of memory|500 Internal' \
  "$SMOKE_ROOT/logs/SGLANG_SD_data_smoke_c2"/*.log \
  && echo 'ERROR: smoke log contains a fatal pattern' >&2 || true
echo DATA_SERVICE_SMOKE=PASS
```

## 11. 论文复现记录

将以下内容复制到 01/02 的每个结果根目录：`dataset_sha256.txt`、`dataset_stats.json`、`tokenizer_manifest.json`、原始文件清单、转换命令、Git commit/status。论文准确性表写实际样本数、筛选区间、输出上限和评分规则；性能表写同一 manifest SHA256。

以下情况必须整体重生成：更换 Target/Draft 模型或 tokenizer；改变 chat template/thinking；改变 context/output reserve；改变 LongBench formatter；改变 MRCR needle/marker 字段；修改 `prepare_qwen3_offline.py`。只改请求数或并发不需要重生成，但仍必须使用冻结目录的 SHA256。
