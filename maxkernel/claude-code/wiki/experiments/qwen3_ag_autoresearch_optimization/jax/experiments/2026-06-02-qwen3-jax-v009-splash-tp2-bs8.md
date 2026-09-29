# Experiment: v009 Splash + TP=2 (bs=8)

**Status:** Completed
**Lane:** jax
**Date:** 2026-06-02

## Hypothesis under test
- **Hypothesis**: The Splash Attention + TP=2 configuration structurally bypasses the massive attention score materialization and halves the per-chip KV cache overhead. This architectural layout provides enough HBM headroom to continue scaling the batch size. Doubling the batch size from 4 to 8 will increase the arithmetic intensity of the FFN and QKV projection GEMMs, amortizing memory bandwidth overheads and pushing the MFU closer to our 35% goal.
- **Mechanism**: Use the `v008` Splash Attention + TP=2 image and set `--batch_size=8` in the launch command.
- **Predicted signal**: The workload will compile and run successfully at `bs=8` without OOM, pushing MFU beyond the 28.6% achieved in `v008`.
- **Falsification criterion**: The workload OOMs at `bs=8` (indicating we need further memory optimizations like Tokamax CE or Activation Checkpointing tuning), or the MFU plateaus/regresses.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 8
- **Seq Length**: 2048
- **Remat Policy**: `checkpoint_dots_with_no_batch_dims`
- **Attention**: Splash Attention
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v008-splash-tp2-bs4`
- **Flags**: `--tp_parallelism=2 --batch_size=8 --use_splash=True`

## Setup
Launched directly via `gke-cluster-runner`.

## Results
- **Status**: Crashed
- **Error**: `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 35.21G of 31.25G hbm. Exceeded hbm capacity by 3.96G.`

## Verdict
**Refuted**.
While TP=2 + Splash Attention cleared massive chunks of memory at `bs=4`, it is not enough to clear the final jump to `bs=8`. HBM allocation scaled linearly with batch size and peaked at 35.21GB (exceeding the 31.25GB limit by ~4GB).

## Next Steps
To unlock `bs=8`, we need to find another 4GB of memory per chip.
Since we failed to integrate Tokamax CE previously (it ballooned memory to 41.77GB under pure FSDP), we could try combining Tokamax CE *with* Tensor Parallelism (TP=2) and Splash Attention. Tokamax natively supports TP and could provide the 4-5GB savings needed.
Alternatively, we could adjust the activation checkpointing/rematerialization policy from `checkpoint_dots_with_no_batch_dims` to a more aggressive rematerialization policy (e.g. `checkpoint_dots`) to save activation memory at the cost of recompute.
