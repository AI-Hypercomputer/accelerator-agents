---
title: "qwen3-jax-v057-f-tokamax-layer-ckpt-fix-splash"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "Fixing Splash attention bkv_compute error by explicitly setting SPLASH_BQ=2048, SPLASH_BKV=2048, SPLASH_BQ_DKV=2048, and SPLASH_BKV_DKV=2048 to match the default bkv_compute=2048 in the backward pass, while keeping the jax.checkpoint layer fix from v057-d."
status: filed
created: 2026-06-15
origin: retrospective-recommendation
---

# qwen3-jax-v057-f-tokamax-layer-ckpt-fix-splash

## Hypothesis under test

**Hypothesis**: In `v057-e`, the workload crashed during tracing with `ValueError: bkv_compute=2048 should not be greater than bkv=1024`. By setting `SPLASH_BKV=2048`, `SPLASH_BQ=2048`, `SPLASH_BKV_DKV=2048`, and `SPLASH_BQ_DKV=2048`, the Splash kernel will trace and compile successfully, and the layer-wise `jax.checkpoint` fix from `v057-d` will prevent the 327 GB HBM OOM.

**Mechanism**:
1. Keep the `v057-d` docker image which has the `Qwen3DecoderLayer` `jax.checkpoint` fix.
2. Update the `launch_cmd` environment variables to `SPLASH_BQ=2048 SPLASH_BKV=2048 SPLASH_BQ_DKV=2048 SPLASH_BKV_DKV=2048` to satisfy the backward pass `bkv_compute=2048` assertion.

**Predicted signal**: The model will compile and train successfully at `batch_size=8`, `seqlen=8192` with Tokamax CE and Splash Attention.

**Falsification criterion**: HBM or VMEM OOM, or MFU drops unexpectedly.

## Setup
Using image `<your-registry>/torchtitan-images/v6e8-qwen3-8b-jax-20260615-exp057-d-tokamax-layer-ckpt:latest`.

## Results
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
