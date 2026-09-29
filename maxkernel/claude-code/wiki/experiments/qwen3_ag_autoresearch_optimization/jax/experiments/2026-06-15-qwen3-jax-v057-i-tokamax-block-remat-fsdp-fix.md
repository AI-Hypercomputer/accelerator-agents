---
title: "qwen3-jax-v057-i-tokamax-block-remat-fsdp-fix"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "Fixing the broken FSDP parameter sharding for block rematerialization allows the model to compile and fit within HBM. In `v057-h`, the `Qwen3Block` refactoring broke the `sharding.py` regexes, causing all 8B parameters to be fully replicated across all chips rather than sharded. This caused an insurmountable 32+ GB HBM parameter+gradient footprint. By fixing `sharding.py` to match `blocks.*.layers.*`, we restore FSDP parameter sharding, which combined with block remat should easily fit within the 31.25 GB v6e limit."
status: filed
created: 2026-06-15
origin: 2026-06-15-qwen3-jax-v057-h-tokamax-block-remat
---

# qwen3-jax-v057-i-tokamax-block-remat-fsdp-fix

## Hypothesis under test

**Hypothesis**: In `v057-h`, the `Qwen3Block` block rematerialization structure changed the attribute paths of the parameters (from `model.layers.0...` to `model.blocks.0.layers.0...`). The regex rules in `sharding.py` failed to match these new paths, falling back to a fully `replicated` sharding plan for all 396 attention and MLP matrices! This forced each TPU chip to allocate a full 16 GB for weights and 16 GB for gradients, causing the `35.83 GB` HBM OOM.
By updating the `SHARDING_PLAN` dictionary to correctly map `model.blocks.*.layers.*`, FSDP parameter sharding is restored. Combined with the block-wise `@nnx.remat` (K=4), activation checkpointing bounds will be minimal (3.75 GB) and parameter/gradient boundaries will be heavily sharded (4 GB), allowing the run to easily fit inside the 31.25 GB HBM capacity.

**Mechanism**:
1. Updated `modeling_qwen3.py` with `nnx.data()` around the blocks correctly fixed the pytree node issues in Flax v0.12.0.
2. Updated `sharding.py` to include `model.blocks.*.layers.*` rules to properly shard FSDP models.
3. Smoke tested to confirm `[sharding] matched=399 replicated=0`.

**Predicted signal**: The model will compile and train successfully at `batch_size=8`, `seqlen=8192` with Tokamax CE and Splash Attention.

**Falsification criterion**: HBM OOM due to excessive activation materialization during the block's backward pass.

## Setup
Image: `<your-registry>/torchtitan-images/v6e8-qwen3-8b-jax-20260615-exp057-d-tokamax-layer-ckpt:latest` (with dynamic archive layering over it for `v057_c/trainer` containing the sharding patch).

## Results
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
