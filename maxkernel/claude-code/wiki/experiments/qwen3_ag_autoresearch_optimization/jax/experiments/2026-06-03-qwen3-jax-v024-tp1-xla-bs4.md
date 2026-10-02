# Experiment: v024 TP=1 Vocab-Parallel CE using XLA reference (bs=4)

**Status:** Pending
**Lane:** jax
**Date:** 2026-06-03

## Hypothesis under test
- **Hypothesis**: The previous `v023` hung during autotuning of `ChunkedXlaLinearSoftmaxCrossEntropyLoss` because its inner `jax.lax.fori_loop` triggered `jax.custom_vjp` autotuning on the massive unpartitioned global shapes before GSPMD had a chance to shard the tensors. By using `implementation="xla"`, we map directly to standard JAX math operations (`x @ w` and `jax.nn.logsumexp`) through the reference implementation. Because these are pure standard ops, JAX passes them cleanly to the GSPMD partitioner, which will automatically insert an `all-gather` on the activations and perfectly shard the resulting `[65536, 18992]` logits tensor (2.48 GB per chip) across the vocab axis on FSDP, fully escaping TP overhead while safely fitting into the 31.25 GB HBM.
- **Mechanism**: Use `implementation="xla"` in `train.py`.
- **Predicted signal**: The experiment should bypass compilation hangs and OOM limits, run at `bs=4`, and yield a significant boost in MFU (>35%) over `v017` by completely removing TP communication.
- **Falsification criterion**: The model continues to OOM or performance remains < 32% MFU.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 4
- **Seq Length**: 2048
- **Remat Policy**: `nnx.scan(remat=True)`
- **Attention**: Splash Attention
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260603-v024-tp1-xla-bs4`
- **Flags**: `--tp_parallelism=1 --batch_size=4 --use_splash=True`

## Setup
Derived from `v021`. Uses the default `v021` image but explicitly sets `implementation="xla"`.

## Results
- **Status**: Pending
- **Throughput**: TBA
- **Approx MFU**: TBA

## Profile Analysis
TBA

## Verdict
TBA

## Next Steps
TBA
