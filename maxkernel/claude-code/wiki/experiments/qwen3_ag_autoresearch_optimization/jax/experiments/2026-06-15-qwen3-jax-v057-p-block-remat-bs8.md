---
title: "qwen3-jax-v057-p-block-remat-bs8"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "Block Rematerialization (wrapping 4 layers into a `Qwen3Block` and checkpointing the block boundary) is required to fit `batch_size=8` into 31.25GB HBM because layer-wise checkpointing uses 15 GB just for boundary activations. In `v057-i`, block remat hung the Pallas compiler for 60 minutes because we removed the `jax.checkpoint` wrapper from the outer `loss_fn`. Re-introducing `jax.checkpoint` on `loss_fn` (with the dots policy, or no policy) will isolate the Tokamax Cross Entropy kernel into its own backward block, preventing the compiler fusion/tracing issues that caused the hang. This will allow the model to compile and run successfully with `batch_size=8`."
status: filed
created: 2026-06-15
origin: 2026-06-15-qwen3-jax-v057-n-tokamax-layer-dots-bs8
---

# qwen3-jax-v057-p-block-remat-bs8

## Hypothesis under test

**Hypothesis**: Block Rematerialization (`K=4` layers per block) successfully reduces boundary memory from 15 GB to 3.75 GB. The 60-minute Pallas compiler hang in `v057-i` was caused by removing the `jax.checkpoint` wrapper from the outer `loss_fn` (which we did during `v057-i`), causing the Tokamax CE backward pass to fuse/trace unpredictably with the final block's backward pass. Restoring `jax.checkpoint(loss_fn)` will fix the compiler hang.

**Mechanism**:
1. Use `Qwen3Block` with `K=4` layers. Checkpoint the block boundary `self._call_impl`.
2. Do NOT checkpoint individual layers.
3. Keep `jax.checkpoint(loss_fn, policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)` in `train.py`.
4. Fix the sharding plan to match `model.blocks.*.layers.*.`.
5. Set `batch_size=8`.
6. Use Tokamax CE and Splash Attention.

**Predicted signal**: The model will compile quickly (<5 mins) and train successfully at `batch_size=8`, `seqlen=8192`.

**Falsification criterion**: HBM OOM, VMEM OOM, or compiler hang.

## Setup
Image: `<your-registry>/torchtitan-images/v6e8-qwen3-8b-jax-20260615-exp057-d-tokamax-layer-ckpt:latest` (with dynamic archive layering over it for `v057_c/trainer`).
Cluster: `charles-v6e`

## Results
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
