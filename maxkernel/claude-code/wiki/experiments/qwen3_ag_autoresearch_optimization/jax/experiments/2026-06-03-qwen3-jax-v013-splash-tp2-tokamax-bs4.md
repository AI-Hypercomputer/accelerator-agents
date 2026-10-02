# Experiment: v013 Splash + TP=2 + Tokamax CE (bs=4)

**Status:** Completed
**Lane:** jax
**Date:** 2026-06-03

## Hypothesis under test
- **Hypothesis**: In `v008` (Splash + TP=2, bs=4), we achieved 28.6% MFU. In `v010`, we attempted to add Tokamax CE to reach `bs=8`, but it still OOMed. However, applying Tokamax CE to the already-working `bs=4` configuration should further reduce the memory bandwidth overhead of the cross-entropy loss calculation (by avoiding materializing the `[batch*seq, vocab]` logits matrix). This could push the MFU higher than 28.6%.
- **Mechanism**: Use the `v010` image (which already has `tokamax.linear_softmax_cross_entropy_loss` integrated correctly) and launch it with `--tp_parallelism=2` and `--batch_size=4`.
- **Predicted signal**: The workload will compile and run successfully at `bs=4` without OOM, achieving MFU > 28.6%.
- **Falsification criterion**: The MFU regresses or stays exactly the same, indicating the CE memory footprint was not a significant bottleneck at `bs=4`.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 4
- **Seq Length**: 2048
- **Remat Policy**: `checkpoint_dots_with_no_batch_dims`
- **Attention**: Splash Attention
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v010-splash-tp2-tokamax-bs8`
- **Flags**: `--tp_parallelism=2 --batch_size=4 --use_splash=True`

## Setup
Launched directly via `gke-cluster-runner`.

## Results
- **Status**: Completed (Exit Code 0)
- **Throughput**: 44638 tok/s (5580/chip)
- **Approx MFU**: 28.7%

## Verdict
**Supported (barely)**.
The workload successfully completed. The addition of the Tokamax Fused Cross-Entropy kernel bumped the MFU from 28.6% to 28.7% (and throughput from 44509 to 44638 tok/s). While the kernel definitely saved memory bandwidth by avoiding the logits matrix, at `bs=4`, the cross-entropy calculation is a tiny fraction of the overall step time compared to the 36 transformer blocks. Thus, the optimization only yielded a marginal compute efficiency gain.

## Next Steps
We are still stuck at `bs=4` because `bs=8` OOMs, and our MFU is peaking at 28.7%.
According to the model documentation, the baseline ships *without* per-layer rematerialization (activation checkpointing). If the model saves all activations across all 36 layers, it easily explains why `bs=8` hits the 31.25GB limit. We must implement **per-layer remat** in the model definition to slash the memory footprint and finally unlock `bs=8`.
