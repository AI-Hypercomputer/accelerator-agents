---
title: "qwen3-jax-v057-j-tokamax-layer-ckpt-bs4"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "Block rematerialization caused a 60-minute compiler hang during Pallas autotuning for the Tokamax CE loss kernel (`v057-i`). This implies that the shape of the inputs to `tokamax.linear_softmax_cross_entropy_loss` when passed through an `nnx.remat` block boundary causes catastrophic Pallas trace blowup. To bypass the hang and prove out Tokamax CE + Splash + per-layer checkpointing, we will drop the batch size from 8 to 4 so that the 15 GB boundary memory requirement for layer-wise checkpointing fits comfortably within the 31.25 GB HBM limit."
status: filed
created: 2026-06-15
origin: 2026-06-15-qwen3-jax-v057-i-tokamax-block-remat-fsdp-fix
---

# qwen3-jax-v057-j-tokamax-layer-ckpt-bs4

## Hypothesis under test

**Hypothesis**: In `v057-i`, the compiler hung for an hour while autotuning `PallasMosaicTpuLinearSoftmaxCrossEntropyLoss`. This indicates that `nnx.remat` block boundaries generate an unpredictable tensor lineage for the custom CE loss function, triggering an infinite loop in the Pallas autotuner. Reverting to per-layer `jax.checkpoint` (from `v057-g`) successfully compiled the Pallas CE kernel in <2 mins previously. Because per-layer checkpointing uses 15 GB of HBM for boundary storage at `bs=8` (causing an OOM), dropping the batch size to `bs=4` will halve the boundary storage to 7.5 GB, allowing the model to fit well within the 31.25 GB HBM budget.

**Mechanism**:
1. Revert to `v057-g`'s per-layer checkpointing (`Qwen3DecoderLayer._ckpt_call = jax.checkpoint(self._call_impl)`).
2. Set `batch_size=4` in `train.py`.
3. Keep Tokamax CE and Splash Attention enabled.
4. Keep `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536` and Splash block variables `SPLASH_BQ=2048 SPLASH_BKV=2048 SPLASH_BQ_DKV=2048 SPLASH_BKV_DKV=2048`.

**Predicted signal**: The model will compile quickly (<5 mins) and train successfully at `batch_size=4`, `seqlen=8192` with Tokamax CE and Splash Attention.

**Falsification criterion**: HBM OOM, VMEM OOM, or compiler hang.

## Setup
Image: `<your-registry>/torchtitan-images/v6e8-qwen3-8b-jax-20260615-exp057-d-tokamax-layer-ckpt:latest` (with dynamic archive layering over it for `v057_c/trainer`).

## Results
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
