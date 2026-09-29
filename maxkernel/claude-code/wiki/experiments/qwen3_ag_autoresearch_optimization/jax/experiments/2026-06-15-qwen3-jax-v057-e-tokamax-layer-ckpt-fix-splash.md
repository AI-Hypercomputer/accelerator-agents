---
title: "qwen3-jax-v057-e-tokamax-layer-ckpt-fix-splash"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "Fixing Splash attention bkv_compute error by explicitly setting SPLASH_BQ=1024 and SPLASH_BKV=1024 to match the default bkv_compute=1024, while keeping the jax.checkpoint layer fix from v057-d."
status: filed
created: 2026-06-15
origin: retrospective-recommendation
---

# qwen3-jax-v057-e-tokamax-layer-ckpt-fix-splash

## Hypothesis under test

**Hypothesis**: In `v057-d`, the workload crashed during tracing with `ValueError: bkv=512 must be a multiple of bkv_compute=1024`. By setting `SPLASH_BKV=1024`, `SPLASH_BQ=1024`, `SPLASH_BKV_DKV=1024`, and `SPLASH_BQ_DKV=1024`, the Splash kernel will trace and compile successfully, and the layer-wise `jax.checkpoint` fix from `v057-d` will prevent the 327 GB HBM OOM.

**Mechanism**:
1. Keep the `v057-d` docker image which has the `Qwen3DecoderLayer` `jax.checkpoint` fix.
2. Update the `launch_cmd` environment variables to `SPLASH_BQ=1024 SPLASH_BKV=1024 SPLASH_BQ_DKV=1024 SPLASH_BKV_DKV=1024`.

**Predicted signal**: The model will compile and train successfully at `batch_size=8`, `seqlen=8192` with Tokamax CE and Splash Attention.

**Falsification criterion**: HBM or VMEM OOM, or MFU drops unexpectedly.

## Setup
Using image `<your-registry>/torchtitan-images/v6e8-qwen3-8b-jax-20260615-exp057-d-tokamax-layer-ckpt:latest`.

## Results
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
