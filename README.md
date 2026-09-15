# SpecStream

## Method overview

SpecStream extends [SGLang](https://github.com/sgl-project/sglang) with
memory-efficient speculative decoding on shared GPUs. It keeps the active KV
frontier on the GPU and streams committed history from host memory. Grouped
transfers, staging buffers, and online softmax combine the attention results
across history blocks. The runtime adjusts verification width and schedules
Draft computation in available Target transfer windows, using timing estimates
and GPU resource limits to decide which work to admit.

The implementation is in
`python/sglang/srt/speculative/spectre/specstream/`, with GPU resource control in
`csrc/specstream_smctrl/`.

## Environment

The reference setup uses two NVIDIA A800 80 GB GPUs, with both Target and Draft
running at tensor-parallel size 2 on the same devices.

| Component | Reference version |
| --- | --- |
| OS | Linux x86-64 |
| Python | 3.12 |
| PyTorch | 2.9.1 + CUDA 12.8 |
| Transformers | 5.3.0 |
| SGLang base | 0.5.17 |
| GPU architecture | SM80 |


## Installation and compilation

Run these commands from the repository root in a Python 3.12 environment:

```bash
conda create -n specstream python=3.12 -y
conda activate specstream
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libzmq3-dev cppzmq-dev libmsgpack-dev
python -m pip install --upgrade pip
SETUPTOOLS_SCM_PRETEND_VERSION=0.5.17 python -m pip install -e ./python
python -m pip install pytest pybind11 msgpack cmake ninja datasets
python scripts/specstream/check_environment.py
```

For offline installation, place compatible wheels in a local directory and add
`--no-index --find-links /path/to/wheels` to the pip commands.

Build the communication and GPU resource-control extensions:

```bash
bash python/sglang/srt/speculative/spectre/cpp_zmq/scripts/build_cpp_zmq.sh
make -C csrc/specstream_smctrl config CUDA_ARCH=80
make -C csrc/specstream_smctrl build
python scripts/specstream/check_environment.py --gpu
```

Set `CUDA_ARCH` to your GPU's compute capability. The communication extension
uses the active Python environment; `PYTHON`, `ZMQ_INCLUDE_DIR`, and
`ZMQ_LIBRARY_DIR` can select another interpreter or native-library location.

## Small-scale inference tests

These tests run a real Target/Draft pair and save generated answers, request
counts, token counts, latency, and throughput. The public-data examples also
report answer accuracy on the selected subset.

### 1. Start the model pair

In the first terminal, activate the environment and set the local model paths:

```bash
conda activate specstream
export TARGET_MODEL=/path/to/Qwen3-32B
export DRAFT_MODEL=/path/to/Qwen3-0.6B
export GPU_IDS=0,1
export DRAFT_TPCS=34
bash scripts/specstream/serve_pair.sh
```

The script validates the global TPC mask on each selected GPU, starts a private
CUDA MPS session, and loads Target followed by Draft. When it prints
`SPECSTREAM_SERVER_READY=http://127.0.0.1:30000`, run the tests from a second
terminal. Press Ctrl-C in the first terminal to stop the pair and its MPS session.
Server commands, profiles, and logs are saved under `outputs/server_*`.

Runtime admission
and verification-width selection remain active. Set `DRAFT_TPCS` to the quota
chosen for your GPU and model pair. The global mask applies to the separate
Draft processes.

Override `CONTEXT_LENGTH`, `TARGET_KV_TOKENS`, `DRAFT_KV_TOKENS`, and
`CPU_MEMORY_GB` as needed. `TARGET_MEM_FRACTION` and `DRAFT_MEM_FRACTION` set the
per-process memory ceilings.
`TARGET_PORT`, `DRAFT_PORT`, and `ZMQ_PORT` default to 30000, 30001, and 5557;
use `--base-url` in the evaluation command when changing the Target port.

### 2. Run a long-input workload

In the second terminal, from the repository root:

```bash
conda activate specstream
export TARGET_MODEL=/path/to/Qwen3-32B
export TEST_ROOT=outputs/example_$(date +%Y%m%d_%H%M%S)
mkdir -p "$TEST_ROOT"

python scripts/specstream/evaluate.py prepare \
  --kind synthetic --tokenizer "$TARGET_MODEL" \
  --samples 16 --input-tokens 12288 --output-tokens 128 \
  --output "$TEST_ROOT/synthetic.json"

python scripts/specstream/evaluate.py run \
  --workload "$TEST_ROOT/synthetic.json" --concurrency 4 \
  --output-dir "$TEST_ROOT/synthetic_run"
cat "$TEST_ROOT/synthetic_run/summary.json"
```


### 3. Test on GSM8K

The [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k) provides
short mathematical word problems. This example selects 32 questions from the
`main` configuration's test split with seed 1, allows up to 512 output tokens,
and scores the final numerical answer.

```bash
python scripts/specstream/evaluate.py prepare \
  --kind gsm8k --tokenizer "$TARGET_MODEL" \
  --samples 32 --seed 1 --output-tokens 512 \
  --output "$TEST_ROOT/gsm8k.json"

python scripts/specstream/evaluate.py run \
  --workload "$TEST_ROOT/gsm8k.json" --concurrency 4 \
  --output-dir "$TEST_ROOT/gsm8k_run"
cat "$TEST_ROOT/gsm8k_run/summary.json"
```


### 4. Test on LongBench v2

[LongBench v2](https://huggingface.co/datasets/zai-org/LongBench-v2) provides
long-document multiple-choice questions. This example selects 16 complete
prompts between 8,192 and 15,360 tokens from its `train` split with seed 1.
Selection uses the Target tokenizer and its chat template; documents outside
the range are skipped. The model returns an answer letter for each question.

```bash
python scripts/specstream/evaluate.py prepare \
  --kind longbench-v2 --tokenizer "$TARGET_MODEL" \
  --samples 16 --seed 1 --min-input-tokens 8192 --max-input-tokens 15360 \
  --output-tokens 128 --output "$TEST_ROOT/longbench.json"

python scripts/specstream/evaluate.py run \
  --workload "$TEST_ROOT/longbench.json" --concurrency 4 \
  --output-dir "$TEST_ROOT/longbench_run"
cat "$TEST_ROOT/longbench_run/summary.json"
```


### 5. Read the outputs

Each prepared workload records sample IDs, token IDs, references, seed, and
vocabulary hash. Each run writes:

| File | Contents |
| --- | --- |
| `config.json` | Run arguments, server model information, and workload hash |
| `generations.jsonl` | Answers, references, scoring, token counts, latency, and request errors |
| `summary.json` | Completed requests, errors, output tokens, throughput, mean latency, and subset accuracy |
| `complete.marker` | Written after every measured request finishes successfully |

Throughput is generated tokens divided by the measured batch wall time,
including prefill and queueing. Request latency covers submission through the
full response.
  python -m pytest -q python/sglang/test/spectre_specstream
```
