---
title: "qwen3-jax-v054-fused-qknorm-rope-e"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "Fusing RMSNorm and RoPE into a single Pallas kernel reduces HBM traffic and improves MFU."
status: completed
created: 2026-06-15
origin: 2026-06-15-qwen3-jax-v054-fused-qknorm-rope
---

# qwen3-jax-v054-fused-qknorm-rope-e

## Hypothesis under test

**Hypothesis**: The Qwen3 architecture uniquely applies RMSNorm to Q and K immediately before applying RoPE. In the XLA implementation, this requires materializing the full `[B, S, H, D]` tensors between the norm and the RoPE operation, leading to significant HBM bandwidth pressure and HLO fragmentation. By fusing RMSNorm and RoPE into a single Pallas kernel, we can keep the intermediate normalized values in SRAM/VMEM and only write the final RoPE'd vectors to HBM, saving two memory traversals and ~2-3pp MFU.

**Mechanism**: Write a custom Pallas kernel `fused_qknorm_rope` that takes un-normalized `q` and `k`, the norm weights, and the RoPE frequencies. Inside the block, it computes the RMSNorm denominator, applies the scale, and then immediately applies the RoPE rotation. Substitute `jnp.einsum`/`rmsnorm` followed by `apply_rotary_emb` with this kernel call in `model/modeling_qwen3.py`. Wrap the `pallas_call` in `shard_map` to prevent JAX compilation partitioning errors. Call `set_splash_mesh()` unconditionally in `train.py` so the mesh is available for the `shard_map`.

**Predicted signal**: expect `custom_call(target=fused_qknorm_rope)` to appear in train_step module, and HBM memory traffic bucket to drop due to avoiding materialization of intermediate Q and K after RMSNorm.

**Falsification criterion**: If the step time does not improve by at least 2% MFU (or the compiled memory footprint is not reduced), then the fusion is bottlenecked by compute or Pallas launch overhead rather than memory bandwidth, and the hypothesis is refuted.

## Setup
Rerun of `v054-d` with `train.py` updated to unconditionally call `set_splash_mesh()`.

## Results

## Profile

missing

## HLO Dump

missing

## Verdict
**invalid**

The workload crashed with an Out Of Memory (OOM) error during the first `jitted_step`: `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Allocation (size=34359738368) would exceed memory (size=33822867456) :: #allocation6476 [shape = 'f32[4,32,8192,8192]...`. The size of the allocation (~32GB) and shape (`4x32x8192x8192`) indicates that it is trying to materialize the full attention matrix in HBM. This suggests that inserting the `shard_map` `custom_call` right before XLA SDPA somehow breaks XLA's ability to fuse the SDPA kernel or triggers a memory-intensive rewrite.
