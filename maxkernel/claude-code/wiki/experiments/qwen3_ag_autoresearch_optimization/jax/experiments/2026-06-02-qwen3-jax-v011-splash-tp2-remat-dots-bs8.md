# Experiment: v011 Splash + TP=2 + Remat Dots (bs=8)

**Status:** Completed
**Lane:** jax
**Date:** 2026-06-02

## Hypothesis under test
- **Hypothesis**: The OOM at batch size 8 (even with Splash, TP=2, and Tokamax CE) is caused by the fundamental volume of activation intermediates saved for the backward pass across 36 blocks. By switching the JAX rematerialization policy from `checkpoint_dots_with_no_batch_dims` to the far more aggressive `checkpoint_dots`, we instruct the compiler to drop almost all matrix multiplication outputs and recompute them dynamically during the backward pass. This trades compute efficiency for a drastic reduction in HBM footprint, theoretically providing the ~3-4GB of headroom needed to compile `bs=8`.
- **Mechanism**: Use the `v008` Splash Attention + TP=2 image. Edit `train.py` to change `_ckpt_policies.checkpoint_dots_with_no_batch_dims` to `_ckpt_policies.checkpoint_dots`.
- **Predicted signal**: The workload will compile and run successfully at `bs=8` without OOM. The MFU may regress slightly compared to `bs=4` due to the overhead of recomputing the dots, but the total throughput (tok/s) should scale higher.
- **Falsification criterion**: The workload still OOMs, or the massive recompute penalty degrades throughput worse than the `bs=4` baseline.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 8
- **Seq Length**: 2048
- **Remat Policy**: `checkpoint_dots`
- **Attention**: Splash Attention
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v011-splash-tp2-remat-dots-bs8`
- **Flags**: `--tp_parallelism=2 --batch_size=8 --use_splash=True`

## Setup
- Branched from `v008`.
- Modified `train.py` to use `jax.checkpoint_policies.checkpoint_dots` instead of `checkpoint_dots_with_no_batch_dims`.
- Built and pushed image as `v011-splash-tp2-remat-dots-bs8`.
- Launched via `gke-cluster-runner`.

## Results
- **Status**: Crashed
- **Error**: `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 35.21G of 31.25G hbm. Exceeded hbm capacity by 3.96G.`

## Verdict
**Refuted**.
The compile OOMed at exactly the same memory usage as `v009` (35.21G). The change to `checkpoint_dots` did not meaningfully reduce the peak memory footprint during compilation. This implies either the policy didn't apply to the bulk of the activations (perhaps due to how Flax NNX layers are wrapped), or the `checkpoint_dots_with_no_batch_dims` was already rematerializing almost everything that could be rematerialized, and the remaining 35.21G is the absolute bare minimum required for `bs=8` at `TP=2`.

## Next Steps
Since we cannot fit `bs=8` into 31.25G of HBM at TP=2, and activation checkpointing tweaking yielded zero gains, we must scale structural parallelism further. We should increase Tensor Parallelism to `TP=4` (leaving `FSDP=2` on the 8-chip slice). This will divide the massive sequence/KV cache footprints by 4 instead of 2.
