# Experiment: v008 Splash + TP=2 (bs=4)

**Status:** Completed
**Lane:** jax
**Date:** 2026-06-02

## Hypothesis under test
- **Hypothesis**: The combination of Splash Attention (which eliminates sequence-dimension activation materialization) and Tensor Parallelism TP=2 (which halves the per-chip KV cache footprint) will synergize to yield enough HBM headroom to double the batch size again to `bs=4`.
- **Mechanism**: Use the `v004` Splash Attention image and set `--tp_parallelism=2` and `--batch_size=4` in the launch command.
- **Predicted signal**: The workload will compile and run successfully at `bs=4` without OOM, pushing MFU beyond the 25.3% frontier.
- **Falsification criterion**: The workload OOMs at compile or run time, or the TP=2 communication overhead scales poorly with batch size and degrades MFU below 25.3%.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 4
- **Seq Length**: 2048
- **Remat Policy**: `checkpoint_dots_with_no_batch_dims`
- **Attention**: Splash Attention
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v008-splash-tp2-bs4`
- **Flags**: `--tp_parallelism=2 --batch_size=4 --use_splash=True`

## Setup
- Rebuilt image from `v004` base to fix a `set_splash_mesh` import bug in `train.py`.
- Final image pushed as `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v008-splash-tp2-bs4`.
- Launched via `gke-cluster-runner` using `v008d` job name.

## Results
- **Status**: Completed (Exit Code 0)
- **Throughput**: 44509 tok/s (5564/chip)
- **Approx MFU**: 28.6%

## Profile Analysis
- `convolution fusion` (main matmuls) accounts for 34.0% of the step.
- `all-reduce-scatter fusion`, `all-gather`, and `all-to-all` accounts for a combined ~29.3% of the step. This is the communication overhead introduced by Tensor Parallelism (TP=2).
- `custom-call` (Splash attention kernel) takes 9.8%.
- HBM Peak: 28.5 GB / 31.25 GB (91.2%). This confirms `bs=4` is tightly packing the memory.

## Verdict
**Supported**.
Stacking Splash Attention (to remove `[seq, seq]` materialization) with Tensor Parallelism (to shard `[batch, num_heads]` KV caches) successfully bridged the memory gap, allowing the per-replica batch size to scale up to `bs=4`. MFU improved from 25.3% (at `bs=2` pure Splash) to 28.6%. The TP=2 communication overhead (~29.3%) is completely outweighed by the improved compute utilization of larger batch dimensions, with zero device idle starvation. Memory is getting tight (91.2%).

## Next Steps
We have successfully broken the `bs=2` frontier! We are currently running `v009` to attempt to push batch size to `bs=8` (using the same Splash+TP=2 stack) to see if we can hit our >35% MFU target, although the 91.2% memory utilization at `bs=4` suggests it may OOM without further intervention.
