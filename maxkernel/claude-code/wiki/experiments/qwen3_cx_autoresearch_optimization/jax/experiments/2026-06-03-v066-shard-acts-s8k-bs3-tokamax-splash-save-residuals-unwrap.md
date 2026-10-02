---
title: "Qwen3 JAX v066 - shard acts s8k bs3 tokamax splash save residuals unwrap"
type: experiment
hypothesis: "Unwrapping Tokamax save_residuals output may allow the valid qseq1 residual-saving probe to compile and reduce forward residual time."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v066-save-residuals-unwrap
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, save-residuals, remat, maxtext-ce, activation-sharding, long-seq, invalid, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v066 - shard acts s8k bs3 tokamax splash save residuals unwrap

## Hypothesis under test

**Hypothesis**: v065 failed because the wrapper returned Tokamax
`save_residuals=True` stats through `shard_map`. Returning only the attention
output should make the residual-saving probe compile; if the saved residuals
are still usable by Tokamax's VJP, this may reduce the v056 forward-residual
custom-call bucket.

**Mechanism**: Keep v065 fixed but unwrap the Tokamax result inside
`wrap_flash_attention`: when `save_residuals=True`, return `result[0]` through
shard_map rather than the full `(output, stats)` tuple.

**Predicted signal**: The run compiles, loss stays near the v056 trajectory,
peak HBM remains under 31.25 GiB, and throughput exceeds v056's 50,383 tok/s /
36.1% MFU with lower forward residual custom-call time.

**Falsification criterion**: Compile/runtime failure, HBM OOM, NaN/invalid loss
trajectory, or clean completion at or below v056's 50,383 tok/s / 36.1% MFU.

## Setup

GKE workload `alekseyv-qwen3-v066`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v066-save-residuals-unwrap`
- **Base image**: `v041-shard-acts`
- **Code change**: add `TOKAMAX_SAVE_RESIDUALS` gate, pass
  `save_residuals=...` to Tokamax `make_splash_mha`, and return only
  `result[0]` from the wrapper when residual stats are produced.
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v066-shard-acts-s8k-bs3-tokamax-splash-save-residuals-unwrap`
- **Mesh**: `fsdp=8,tp=1`
- **Environment change**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0 TOKAMAX_SAVE_RESIDUALS=1`
- **XLA flag**: keep v056's `--xla_latency_hiding_scheduler_rerun=3`
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Invalid before the first train step.

v066 got past v065's shard_map `out_specs` mismatch, but Tokamax's backward
path then failed during tracing:

`AttributeError: 'tuple' object has no attribute 'astype'`

The failing line is in Tokamax `_splash_attention_bwd`, where it expects `do`
to be the attention-output cotangent, but the wrapper/API combination produced
a tuple cotangent after unwrapping `(output, stats)`.

## Profile

No profile. The job failed before step 0 and before the profiling window.

## HLO Dump

No HLO dump. The job failed during tracing before compilation.

## Verdict

**Invalid.** `save_residuals=True` is not a drop-in replacement in this wrapper:
returning the full tuple breaks shard_map output specs, while returning only the
attention output breaks Tokamax's backward path. Do not continue this axis
without a deeper custom-VJP wrapper.
