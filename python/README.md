# SpecStream

## Method overview

SpecStream extends SGLang with memory-efficient speculative decoding on shared
GPUs. It separates committed history KV from the rollback-sensitive GPU frontier,
streams history blocks through grouped transfers and staging buffers, and combines
partial attention results with online softmax. A controller adjusts verification
width and admits Draft work subject to Target progress, timing, and resource limits.

This release contains the language-model runtime and the SpecStream extensions.
The implementation is under `python/sglang/srt/speculative/spectre/specstream/`.
The surrounding `spectre` module and command-line identifiers are retained for
runtime and protocol compatibility. The SM-control extension is under
`csrc/specstream_smctrl/`. Copyright and third-party notices are retained in
`LICENSE`, `THIRD_PARTY_NOTICES.txt`, and the original source headers.

## Environment

The reference environment uses Linux x86-64, Python 3.12, PyTorch 2.9.1 with
CUDA 12.8, Transformers 5.3.0, and two NVIDIA A800 80 GB GPUs for the shared-GPU
configuration. The SGLang base version is 0.5.17. The GPU-independent smoke tests
also run without CUDA. A compatible compiler, CUDA toolkit, and NVIDIA driver
are required to build and validate the GPU extensions.

Use local model directories and paths belonging to your own machine. No model
weights, prepared datasets, result files, or machine-specific benchmark matrices
are included. Tensor-parallel size, memory budgets, GPU selection, and TPC limits
must match the deployment. Model pairs must have compatible token IDs, special
tokens, and prompt templates; belonging to the same model family is insufficient.

## Installation and compilation

Run the commands from this repository's root. Activate an isolated Python 3.12
environment first. On Debian/Ubuntu, install the native build prerequisites:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libzmq3-dev cppzmq-dev libmsgpack-dev
python -m pip install --upgrade pip
SETUPTOOLS_SCM_PRETEND_VERSION=0.5.17 python -m pip install -e ./python
python -m pip install pytest pybind11 msgpack cmake ninja
python scripts/specstream/check_environment.py
```

The package metadata specifies runtime dependencies. For an offline machine,
prepare a compatible wheel directory in advance and use pip's `--no-index` and
`--find-links` options for both installation commands. Install a PyTorch build
compatible with the intended CUDA environment; avoid replacing an existing
validated environment merely to run these commands.

Build the inter-process protocol extension with the active environment's Python:

```bash
bash python/sglang/srt/speculative/spectre/cpp_zmq/scripts/build_cpp_zmq.sh
```

The build script checks prerequisites and does not install system packages.
For nonstandard native-library locations, set `ZMQ_INCLUDE_DIR` and
`ZMQ_LIBRARY_DIR`. Use `PYTHON` to select a different interpreter if needed.

Build the optional GPU resource-control extension when testing shared-GPU grants:

```bash
# 80 is the compute capability used by the reference A800; change for your GPU.
make -C csrc/specstream_smctrl config CUDA_ARCH=80
make -C csrc/specstream_smctrl build
python scripts/specstream/check_environment.py --gpu
```

Validate the mask backend on the deployment GPU before enabling it in a server:

```bash
make -C csrc/specstream_smctrl validate TPC_LOW=0 TPC_HIGH=4
```

If stream masking is unsupported by the driver, test the independent Draft
process backend explicitly:

```bash
make -C csrc/specstream_smctrl validate-global TPC_LOW=0 TPC_HIGH=4
```

Use `--specstream-smctrl-mask-scope global` only after that validator passes.
This backend affects the whole Draft process and must not be applied to a
process that also executes Target kernels. A successful build alone does not
establish mask compatibility. The example of four TPCs is a small validation
case, not a tuned performance setting.

## Minimal tests

The default smoke test uses synthetic tensors and protocol fixtures. It does
not download models or datasets, start a service, or run a performance matrix:

```bash
bash scripts/specstream/smoke_test.sh
```

Expected result: pytest reports passing tests and the script prints
`SPECSTREAM_SMOKE=PASS`. This checks online-softmax equivalence, resource-profile
selection, grant decisions, and GPU-history budget accounting on CPU.

To exercise CUDA streaming attention and the verifier with synthetic KV tensors:

```bash
bash scripts/specstream/smoke_test.sh --gpu
```

The GPU mode fails before testing if CUDA or the required attention kernels are
unavailable. It is a component correctness test, not an end-to-end throughput
claim. For the full retained component suite:

```bash
PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" \
  python -m pytest -q python/sglang/test/spectre_specstream
```

The remaining utilities are generic: `scripts/specstream/analyze_grant_events.py`
checks captured grant events, and `scripts/specstream/smctrl/` assembles resource
profiles from user-supplied measurements. Neither contains measured results or
automatically tunes a TPC quota. Generated files belong outside the source tree
or in an ignored output directory.
