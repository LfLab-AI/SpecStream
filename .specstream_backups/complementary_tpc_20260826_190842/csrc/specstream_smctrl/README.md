# SpecStream libsmctrl vendor

This directory vendors the minimal stream-mask library and validator used by
SpecStream innovation point 2.  The mask implementation is vendored from
`zejia-lin/BulletServe` commit
`445afae2abc15d578d3107d37a9b57ebc49e5e46` (`csrc/`), which in turn credits
Joshua Bakita and James H. Anderson's libsmctrl work.  The parent repository is
Apache-2.0 licensed; retain the upstream copyright notices in the sources.  The
validator was hardened to use the current CUDA device, derive its SM-to-TPC
ratio, validate CLI arguments, and return a failing process status on mismatch.

Build and validate on the exact experiment host before enabling runtime grants:

```bash
make config
make build
make validate TPC_LOW=0 TPC_HIGH=4
```

CUDA 13 / Driver 580 may move the private per-stream mask field.  If the
ordinary validator reports that SMs outside the requested range were used, do
not set `MASK_OFF` in a service process and do not add an unvalidated version
case.  On a dedicated Drafter process with at most 64 TPCs, validate the
offset-independent QMD/TMD callback backend instead:

```bash
make validate-global TPC_LOW=0 TPC_HIGH=4
```

Only after this target passes on the exact GPU/driver combination may the
Drafter be launched with `--specstream-smctrl-mask-scope global`.  This scope
is process-wide: it is correct for SpecStream's dedicated Drafter process, but
must not be enabled in a process that also runs unmasked Target kernels.

The runtime intentionally fails closed if `libsmctrl.so` is absent, the CUDA
version is unsupported, the requested TPC range is invalid, or validation
fails.  Do not use MPS active-thread percentage as a substitute for this
validator.
