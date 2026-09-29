---
title: "qwen3-jax-v057-s-nested-ckpt-nothing-saveable-bs8"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "In `v057-r`, we added an inner `jax.checkpoint` on `Qwen3DecoderLayer` without any policy. JAX defaults to `save_anything_that_can_be_saved`, causing the inner layer checkpoints to eagerly save their boundaries during the main forward pass and completely negating the memory savings of the outer `Qwen3Block` checkpoint (resulting in the exact same 35.83GB HBM OOM as layer-wise checkpointing). By explicitly setting `policy=jax.checkpoint_policies.nothing_saveable` on the inner layer checkpoint, the inner checkpoint will NOT save any boundaries during the main forward pass, but it WILL still trigger a `remat2` primitive during the backward pass recomputation. This creates a compiler barrier separating the layers in the backward pass (preventing the XLA compiler hang) while preserving the 11.25 GB memory savings of Block Rematerialization."
status: filed
created: 2026-06-15
origin: 2026-06-15-qwen3-jax-v057-r-nested-ckpt-bs8
---

# qwen3-jax-v057-s-nested-ckpt-nothing-saveable-bs8

## Hypothesis under test

**Hypothesis**: Nested checkpointing using `policy=nothing_saveable` on the inner layers fixes both the compiler hang and the HBM OOM.
1. Outer checkpoint: `jax.checkpoint(Qwen3Block)` (saves block inputs, 3.75GB).
2. Inner checkpoint: `jax.checkpoint(Qwen3DecoderLayer, policy=jax.checkpoint_policies.nothing_saveable)` (acts ONLY as a compiler barrier against backward fusion, saves NO memory during main forward pass).
3. Loss checkpoint: `jax.checkpoint(loss_fn)` (prevents Tokamax CE fusion, saves `hidden_states` input, recomputes 39GB logits).

**Mechanism**:
1. Keep `Qwen3Block` with `K=4` layers. Checkpoint the block boundary.
2. Ensure `jax.checkpoint(layer._call_impl, policy=jax.checkpoint_policies.nothing_saveable)` inside `Qwen3DecoderLayer`.
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
