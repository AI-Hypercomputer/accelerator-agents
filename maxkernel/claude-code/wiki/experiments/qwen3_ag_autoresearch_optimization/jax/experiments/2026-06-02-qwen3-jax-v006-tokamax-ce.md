# Experiment: v006 Tokamax Fused Cross-Entropy

**Status:** Planned
**Lane:** jax
**Date:** 2026-06-02

## Hypothesis under test
- **Hypothesis**: The ~300ms `broadcast_select_fusion` is not an attention mask, but rather the `take_along_axis` operation inside the default cross-entropy loss materializing a massive broadcast select; porting `tokamax_ce` (fused cross-entropy) will eliminate this 300ms overhead and shrink HBM usage, unblocking bs=2.
- **Mechanism**: Replace `optax.softmax_cross_entropy_with_integer_labels` / `take_along_axis` with `tokamax._src.ops.experimental.tpu.loss.pallas_mosaic_tpu_kernel.linear_softmax_cross_entropy_loss` wrapped in `shard_map`.
- **Predicted signal**: `broadcast_select_fusion` drops from ~300ms to ~0. HBM footprint is reduced sufficiently to allow `bs=2` to compile and run.
- **Falsification criterion**: Step MFU does not exceed current frontier + 1.5 pp over median of steps 10-50, or bs=2 fails to fit.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 2
- **Seq Length**: 2048
- **Remat Policy**: `checkpoint_dots_with_no_batch_dims`
- **Attention**: Splash Attention
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v006-tokamax-ce-b`

## Setup
- Created fork directory: `.repo/2026-06-02-v006-tokamax-ce/`
- Extracted `train.py` from base image.
- Modified `loss_fn` in `train.py` to use `tokamax.linear_softmax_cross_entropy_loss` wrapped in `jax.experimental.shard_map.shard_map` with `psum('fsdp')` reduction, bypassing the model's `lm_head` projection to apply the fused CE kernel on the pre-projection hidden states directly.
- Wrapped the `tokamax` import in `train.py` with `sys.argv[:1]` to prevent `absl.flags.FLAGS` initialization from crashing due to unrecognized command line flags before `fire.Fire` runs.
- Built and pushed image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v006-tokamax-ce-b`
## Results
(To be populated)
