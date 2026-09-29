# Experiment: v005 Splash Tuned (BTHD Layout + Tokamax Overlap)

**Status:** In Progress
**Lane:** jax
**Date:** 2026-06-02

## Hypothesis under test
- **Hypothesis**: Changing QKV inputs to natively emit the `(B, T, H, D)` layout and enabling the Tokamax experimental scheduler will eliminate the layout tax and hide the ring collective overhead, increasing MFU.
- **Mechanism**: Modified `Qwen3Attention` to emit `(B, T, H, D)` layout natively instead of transposing to `(B, H, T, D)`, set `q_sharding = P("fsdp", None, "tp", None)`. Set `USE_TOKAMAX_SPLASH=1` and `TOKAMAX_USE_EXP_SCHED=1`.
- **Predicted signal**: `data formatting` bucket drops to ~0%. `collective-permute` buckets shrink or are better hidden under compute.
- **Falsification criterion**: Step time > 390ms or MFU ~25% means layout tax was not the primary bottleneck or scheduler failed to overlap.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 2
- **Seq Length**: 2048
- **Remat Policy**: `checkpoint_dots_with_no_batch_dims`
- **Attention**: Splash Attention Tuned
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v005-splash-tuned`

## Results
- **Step time**: 446.0ms
- **Throughput**: 35044 tok/s (4380/chip)
- **MFU**: 22.5% (dropped from 25.3% in v004)

### Profile Analysis (Mode B)
- **Hardware Envelope**: TPU v6e has `peak_flop_rate` ~946.7 TFLOP/s and `peak_hbm_bw` ~1525 GB/s.
- **Convolution fusion (matmuls)**: Highly efficient! Consumed 29.5% of step time while executing 4468.6 TF. Achieved rate is ~644 TFLOP/s (~68% of hardware peak).
- **Collective-permute (Ring Attention Sync)**: `TOKAMAX_USE_EXP_SCHED=1` **failed** to hide the sync overhead. `collective-permute-done` takes 18.8% of step time, and `collective-permute-start` takes 8.4%. Total = 27.2% (virtually unchanged from 28.5% in v004).
- **Loop fusion (Memory-bound)**: A memory-bound kernel consumed 17.0% of step time, pushing 4572 GiB of data at ~1146 GiB/s (~75% of HBM peak bandwidth).
- **Data formatting**: 6.3% of step time (reduced but not eliminated, as we had to revert the strict `(B, T, H, D)` layout requirement due to kernel compilation constraints in Splash Attention requiring `[..., seq_len, head_dim]`).

## Verdict
**Falsified**.
The Tokamax experimental scheduler failed to overlap the P2P communication for this shape. The step time remains highly diluted by unhidden network syncs (27.2%) and a massive memory-bound loop fusion (17.0%).

## Next Steps
We have an open question:
1. What is the massive `loop fusion` (17% step time) actually computing, and can it be fused or optimized?
We should investigate if this loop fusion is the RMSNorm or SwiGLU operations and see if we can optimize them.
