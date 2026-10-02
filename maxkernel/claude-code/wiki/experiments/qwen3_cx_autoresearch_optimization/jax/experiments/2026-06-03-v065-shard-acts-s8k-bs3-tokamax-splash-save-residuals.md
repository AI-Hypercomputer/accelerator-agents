---
title: "Qwen3 JAX v065 - shard acts s8k bs3 tokamax splash save residuals"
type: experiment
hypothesis: "Saving Tokamax Splash forward residuals may reduce the valid qseq1 forward-residual custom-call bucket."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v065-save-residuals
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, save-residuals, remat, maxtext-ce, activation-sharding, long-seq, invalid, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v065 - shard acts s8k bs3 tokamax splash save residuals

## Hypothesis under test

**Hypothesis**: The v056 frontier still spends about 11.1% of device time in
Tokamax Splash forward residual custom-calls. Enabling Tokamax
`save_residuals=True` on the valid qseq1 path may trade the 2.9 GiB HBM
headroom for lower residual recompute time and improve wall-clock MFU.

**Mechanism**: Keep the v056 frontier fixed and change only the Tokamax Splash
wrapper to pass `save_residuals=True` when `TOKAMAX_SAVE_RESIDUALS=1`.
Runtime sets `TOKAMAX_SAVE_RESIDUALS=1` while keeping
`TOKAMAX_Q_SEQ_SHARDS=1` and `TOKAMAX_MAX_LOGIT_CONST=30.0`.

**Predicted signal**: Loss stays near the v056 trajectory, ending around 12.05,
peak HBM remains under 31.25 GiB, and throughput exceeds v056's 50,383 tok/s /
36.1% MFU by reducing the forward residual custom-call bucket.

**Falsification criterion**: Compile/runtime failure, HBM OOM, NaN/invalid loss
trajectory, or clean completion at or below v056's 50,383 tok/s / 36.1% MFU.

## Setup

GKE workload `alekseyv-qwen3-v065`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v065-save-residuals`
- **Base image**: `v041-shard-acts`
- **Code change**: in `/app/trainer/splash_attn.py`, add the
  `TOKAMAX_SAVE_RESIDUALS` env gate and pass `save_residuals=...` to
  Tokamax `make_splash_mha`.
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v065-shard-acts-s8k-bs3-tokamax-splash-save-residuals`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0 TOKAMAX_SAVE_RESIDUALS=1`
- **XLA flag**: keep v056's `--xla_latency_hiding_scheduler_rerun=3`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Invalid before the first train step on both workers.

The patched wrapper successfully reached the Tokamax path and logged
`save_residuals=True`, but tracing failed before compilation:

`ValueError: shard_map ... out_specs is PartitionSpec('fsdp', 'tp', None, None)
which has length 4, but wrap_flash_attention(*args)[1]['logsumexp'] has shape
float32[3,32,8192], which has rank 3`

Tokamax `save_residuals=True` changes the kernel return from just the attention
output to `(output, stats)`. The wrapper still returned that full tuple through
a shard_map whose `out_specs` only described the 4D attention output.

## Profile

No profile. The job failed before train step 0 and before the profiling window.

## HLO Dump

No HLO dump. The job failed during tracing before compilation.

## Verdict

**Invalid.** This run tested the API wiring, not the performance hypothesis.
The next minimal correction is to unwrap the `(output, stats)` tuple inside the
Tokamax wrapper, returning only the attention output through shard_map while
keeping `save_residuals=True`.
