---
title: "qwen3-jax-v057-g-tokamax-layer-ckpt-fix-splash"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "Fixing Splash attention VMEM OOM (35.63M > 32.00M limit) by increasing the VMEM limit via LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536, and keeping BQ=2048 and BKV=2048."
status: refuted
created: 2026-06-15
origin: retrospective-recommendation
---

# qwen3-jax-v057-g-tokamax-layer-ckpt-fix-splash

## Hypothesis under test

**Hypothesis**: In `v057-f`, setting the Splash attention block sizes to 2048 to satisfy the tracing assertions caused the generated kernel's VMEM requirements to exceed the default 32.00 MB limit (it needed 35.63 MB). By increasing the limit to 64 MB (65536 KiB) using `LIBTPU_INIT_ARGS`, the Splash kernel will compile successfully.

**Mechanism**:
1. Keep the `v057-d` docker image which has the `Qwen3DecoderLayer` `jax.checkpoint` fix.
2. Update the `launch_cmd` to include `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536` in addition to the `SPLASH_*` block size variables.

**Predicted signal**: The model will compile and train successfully at `batch_size=8`, `seqlen=8192` with Tokamax CE and Splash Attention.

**Falsification criterion**: HBM OOM, or a VMEM OOM that exceeds 64MB, or MFU drops unexpectedly.

## Setup
Using image `<your-registry>/torchtitan-images/v6e8-qwen3-8b-jax-20260615-exp057-d-tokamax-layer-ckpt:latest`.

## Results
The experiment failed during XLA compilation with an HBM OOM:

```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 35.83G of 31.25G hbm. Exceeded hbm capacity by 4.59G.

Total hbm usage >= 36.09G:
    reserved        260.00M
    program          30.11G
    arguments         5.72G
```

**Verdict: Refuted.**
The VMEM fix for the Splash kernel worked (it bypassed the previous VMEM exhaustion), but compiling the model with Tokamax CE and Splash attention required >35.8 GB of HBM during compilation, which exceeded the 32GB capacity of the TPU v6e lite node. The program size (30.11G) is massively bloated.
