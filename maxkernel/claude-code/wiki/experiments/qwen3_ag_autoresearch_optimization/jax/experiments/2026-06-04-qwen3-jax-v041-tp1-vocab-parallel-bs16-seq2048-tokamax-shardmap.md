---
title: "v041: TP=1 Vocab-Parallel Tokamax Pallas CE with ShardMap (bs=16)"
date: "2026-06-04"
model: "qwen3-8b-jax"
lane: "jax"
---

# Experiment: v041 - TP=1 Vocab-Parallel Tokamax Pallas CE with ShardMap (bs=16)

## Hypothesis under test
- **Hypothesis**: In `v040`, we attempted to use Tokamax's Pallas (`"mosaic_tpu"`) cross-entropy loss, but it crashed because XLA cannot auto-partition custom Mosaic kernels under pure FSDP (`TP=1`). To fix this, we must wrap the loss computation in `jax.experimental.shard_map` to manually partition the `B*L` dimension over the `fsdp` mesh axis, and instruct XLA to all-gather the fully sharded parameter (`lm_head.weight`) to compute the local logits slice without materializing the global `[B*L, V]` activation tensor.
- **Mechanism**: Set Tokamax implementation to `"pallas"` and wrap it in `shard_map`. Use `bs=16` and `L=2048`.
- **Predicted signal**: The model compiles without OOM, executes without crashing, and achieves MFU >35% due to avoiding the heavy HBM materialization of dense CE loss.
- **Falsification criterion**: Pallas CE with shard_map crashes, OOMs, or is slower than dense XLA CE.

## Configuration
- **Model**: Qwen3 8B (JAX)
- **Batch Size**: 16
- **Tensor Parallelism (TP)**: 1
- **Mechanisms Enabled**:
  - Splash Attention
  - Tokamax `implementation="pallas"`
  - `jax.experimental.shard_map` manual partitioning
  - Inner `nnx.remat`
  - Vocab-Parallel FSDP CE

## Setup Steps
1. Fork codebase from `v040`.
2. Edit `train.py` to add `jax.experimental.shard_map.shard_map` wrapping around `tokamax.linear_softmax_cross_entropy_loss`.
3. Set `in_specs=(P('fsdp', None), P('fsdp'), P(None, None))` so XLA handles the all-gathering of `lm_head.weight`.
4. Launch via `gke-cluster-runner`.

## Results
- **Status**: Pending
- **Throughput**: N/A
- **Approx MFU**: N/A

## Verdict
Pending.

## Profile

**Source**: gs://<your-bucket>/autoresearch/qwen3-jax-v041-profile
**xprof URL**: N/A (Server environment unavailable)
**Run name**: qwen3-jax-v041-profile
**Steps captured**: 10 steps (approx)
**Description**: xprof trace (TPU device + host) manually aggregated from trace.json.gz

**Bucket attribution** (total step time ~6.89s, 41.3% in XLA device execution wrapper, heavily bottlenecked by unrolled layer overhead):

| Bucket | % of step | Self time (total across 12 measured steps) |
|---|---|---|
| jit_train_step (Wrapper) | 41.3% | 82.6s |
| fusion (Dots/Math) | 25.3% | 50.6s |
| while (Loop overhead) | 17.4% | 34.9s |
| splash_mha (Custom Kernel) | 4.0% | 8.0s |
| tokamax_ce_loss (Custom Kernel) | 3.0% | 5.9s |
| collective-permute / copy | 4.5% | 8.3s |

**Dominant observations**:
1. **while loop overhead**: 34.9 seconds across 12 steps (17.4% of total time). 9,504 while calls. Caused by Python `for` loop over layers instead of `jax.lax.scan`.
2. **fusion**: 50.6 seconds. This represents all the matrix multiplications and element-wise ops inside the layers. The high duration indicates `jax.checkpoint` on the outer `loss_fn` is causing full forward-pass recomputation during the backward pass.
3. **linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu**: 4.18 seconds total (2.09% of time) for 12 steps. The custom kernel is highly efficient and operating successfully.
4. **splash_mha**: 8.0 seconds total. Working effectively.

**Step time**: ~6.89 s/step · **MFU**: ~24.5% (calculated via global BS 128 throughput 38033 tok/s)

## HLO Dump

**Source**: gs://<your-bucket>/autoresearch/qwen3-jax-v041/hlo
**Modules**: Extracted from xplane.pb

**Hypothesis-firing audit** (Phase 3):
- Mechanism claimed: tokamax mosaic_tpu CE kernel via shard_map and splash attention.
- Found in HLO: **YES** — `linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu` and `splash_mha_fwd_residuals` are present and taking appropriate fractions of compute.
- Result: **HYPOTHESIS FIRING CONFIRMED**.

**Notable patterns**:
- **SEVERE**: Massive `while` loop overhead detected due to lack of `scan-over-layers`.
- **SEVERE**: `fusion` time indicates excessive recomputation from `jax.checkpoint` applied across the entire `loss_fn`.
