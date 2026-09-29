---
title: "qwen3-jax-v057-r-nested-ckpt-bs8"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "The 60-minute Pallas compiler hang in block rematerialization is caused by XLA blowing up on a giant 4-layer backward pass graph. By nesting `jax.checkpoint` (putting one on `Qwen3Block` AND one on `Qwen3DecoderLayer`), the outer block checkpoint will discard layer boundaries during the main forward pass (saving boundary memory), and the inner layer checkpoints will trigger during the backward recomputation to isolate the layers (preventing XLA hang). Combining this with a policy-less `jax.checkpoint(loss_fn)` will fit `bs=8` at 8k seqlen in 31.25GB HBM."
status: filed
created: 2026-06-15
origin: 2026-06-15-qwen3-jax-v057-q-block-remat-bs8-no-policy
---

# qwen3-jax-v057-r-nested-ckpt-bs8

## Hypothesis under test

**Hypothesis**: Nested checkpointing solves both the HBM OOM and the Pallas compiler hang. The outer `jax.checkpoint(Qwen3Block)` saves 3.75 GB of boundary memory. The inner `jax.checkpoint(Qwen3DecoderLayer)` prevents the XLA compiler from trying to fuse 4 layers into a single giant backward pass graph (which takes 60+ minutes to compile).

**Mechanism**:
1. Keep `Qwen3Block` with `K=4` layers. Checkpoint the block boundary.
2. Restore `jax.checkpoint(layer._call_impl)` inside `Qwen3DecoderLayer`.
3. Keep `jax.checkpoint(loss_fn)` in `train.py` WITHOUT any policy.
4. Set `batch_size=8`.
5. Use Tokamax CE and Splash Attention.

**Predicted signal**: The model will compile quickly (<5 mins) and train successfully at `batch_size=8`, `seqlen=8192`.

**Falsification criterion**: HBM OOM, VMEM OOM, or compiler hang.

## Setup
Image: `<your-registry>/torchtitan-images/v6e8-qwen3-8b-jax-20260615-exp057-d-tokamax-layer-ckpt:latest` (with dynamic archive layering over it for `v057_c/trainer`).
Cluster: `charles-v6e`

## Results
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
