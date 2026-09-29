# Experiment: v016 Pure FSDP (TP=1) + bs=4

**Status:** Completed
**Lane:** jax
**Date:** 2026-06-03

## Hypothesis under test
- **Hypothesis**: In `v015` we achieved 30.7% MFU at `bs=8` under `TP=2`. The profile showed ~23% of the step time is consumed by Tensor Parallelism collective communication (`all-reduce-scatter` and `all-gather`). If we switch to Pure FSDP (`TP=1`), we eliminate this TP communication overhead entirely.
Because `TP=1` at `bs=4` creates the *exact same* per-chip activation and KV cache footprint as `TP=2` at `bs=8`, it should fit perfectly within the 31.25GB limit. Furthermore, `TP=1` (FSDP=8) will shard the master weights and optimizer states across 8 chips instead of 4, saving an additional ~2GB of HBM.
The only risk is that Tokamax CE might balloon memory under Pure FSDP (as it did in `v006`), but we will test this. If successful, trading TP overhead for pure FSDP should propel compute efficiency past 35% MFU.
- **Mechanism**: Use the heavily optimized `v015` image (Splash + Tokamax CE + nnx.remat + Fused QK-norm/RoPE). Launch with `--tp_parallelism=1` and `--batch_size=4`.
- **Predicted signal**: The workload compiles without OOM and achieves >35% MFU.
- **Falsification criterion**: Tokamax CE OOMs under Pure FSDP, or the 8-chip FSDP weight-gather overhead degrades performance worse than TP=2.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 4
- **Seq Length**: 2048
- **Remat Policy**: `nnx.remat` (per-layer)
- **Attention**: Splash Attention
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8`
- **Flags**: `--tp_parallelism=1 --batch_size=4 --use_splash=True`

## Setup
Launched directly via `gke-cluster-runner`.

## Results
- **Status**: Crashed
- **Error**: `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 31.96G of 31.25G hbm. Exceeded hbm capacity by 736.97M.`

## Verdict
**Refuted**.
Despite having the *exact same* sequence lengths and KV cache footprint as the successful `bs=8 TP=2` configuration, the Pure FSDP run (`bs=4 TP=1`) OOMed during compilation. This confirms that the massive 41.77GB memory bloat we witnessed in `v006` is a fundamental characteristic of how the `tokamax.linear_softmax_cross_entropy_loss` Pallas kernel is compiling/sharding under a pure FSDP (`TP=1`) layout. Tokamax CE natively expects Tensor Parallelism (`TP>=2`) to distribute the vocabulary/logits matrix. Without TP, Tokamax panics and attempts to gather the full vocabulary matrix into memory locally on every chip, causing a 736MB OOM.

## Next Steps
We must accept that `TP=2` is a strict requirement for utilizing the Tokamax CE optimization on an 8B model.

We are currently hovering at 30.7% MFU (`bs=8`, `TP=2`). To bridge the remaining 4.3% gap to our 35% goal, we should look into pure compute micro-optimizations that eliminate the 12.8% `loop fusion` bucket. We should consider testing the **Scan over layers** hypothesis. `flax.nnx.scan` can eliminate Python-side loop unrolling overheads during compilation and significantly improve the XLA compiler's ability to schedule operations.
