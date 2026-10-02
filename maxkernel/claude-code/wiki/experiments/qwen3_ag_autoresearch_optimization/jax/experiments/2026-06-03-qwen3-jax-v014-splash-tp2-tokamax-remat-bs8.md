# Experiment: v014 Splash + TP=2 + Tokamax CE + Per-Layer Remat (bs=8)

**Status:** Planned
**Lane:** jax
**Date:** 2026-06-03

## Hypothesis under test
- **Hypothesis**: In `v010` (Splash + TP=2 + Tokamax CE at bs=8) we OOMed at ~33.9GB. We proved in `v011` that the `checkpoint_dots` policy alone wasn't enough. The root issue is the `Qwen3Model` loop saves every activation intermediate across all 36 decoder blocks. By applying per-layer rematerialization (`nnx.remat`) around the `Qwen3DecoderLayer` application in the forward loop, we will trade recompute efficiency to drastically slash the activation memory footprint, finally unlocking `bs=8` at `TP=2`.
- **Mechanism**: Use the `v010` Tokamax CE image. Edit `model/modeling_qwen3.py` to wrap the layer application in `nnx.remat(layer)(hidden_states, ...)`. Launch with `--tp_parallelism=2` and `--batch_size=8`.
- **Predicted signal**: The workload will compile and run at `bs=8` without OOM, pushing MFU significantly higher.
- **Falsification criterion**: The workload still OOMs, or the massive recompute penalty degrades throughput worse than the `bs=4` baseline.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 8
- **Seq Length**: 2048
- **Remat Policy**: `nnx.remat` (per-layer)
- **Attention**: Splash Attention
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v014-splash-tp2-tokamax-remat-bs8`
- **Flags**: `--tp_parallelism=2 --batch_size=8 --use_splash=True`

## Setup
Derived from `v010` (Splash + TP=2 + Tokamax CE).
1. In `model/modeling_qwen3.py`, wrapped the `Qwen3DecoderLayer` application inside the `Qwen3Model.__call__` loop with `nnx.remat`: `hidden_states = nnx.remat(layer)(hidden_states, (cos, sin), attention_mask=None)`.
2. Built new docker image `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v014-splash-tp2-tokamax-remat-bs8` containing the modified model code.

## Results
(To be populated)
