---
title: "qwen3-jax-v054-fused-qknorm-rope-f"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "Fusing RMSNorm and RoPE into a single Pallas kernel reduces HBM traffic and improves MFU."
status: filed
created: 2026-06-15
origin: 2026-06-15-qwen3-jax-v054-fused-qknorm-rope
---

# qwen3-jax-v054-fused-qknorm-rope-f

## Hypothesis under test

**Hypothesis**: The Qwen3 architecture uniquely applies RMSNorm to Q and K immediately before applying RoPE. In the XLA implementation, this requires materializing the full `[B, S, H, D]` tensors between the norm and the RoPE operation, leading to significant HBM bandwidth pressure and HLO fragmentation. By fusing RMSNorm and RoPE into a single Pallas kernel, we can keep the intermediate normalized values in SRAM/VMEM and only write the final RoPE'd vectors to HBM, saving two memory traversals and ~2-3pp MFU.

**Mechanism**: Write a custom Pallas kernel `fused_qknorm_rope` that takes un-normalized `q` and `k`, the norm weights, and the RoPE frequencies. Inside the block, it computes the RMSNorm denominator, applies the scale, and then immediately applies the RoPE rotation. Substitute `jnp.einsum`/`rmsnorm` followed by `apply_rotary_emb` with this kernel call in `model/modeling_qwen3.py`.

**Predicted signal**: expect `custom_call(target=fused_qknorm_rope)` to appear in train_step module, and HBM memory traffic bucket to drop due to avoiding materialization of intermediate Q and K after RMSNorm.

**Falsification criterion**: If the step time does not improve by at least 2% MFU (or the compiled memory footprint is not reduced), then the fusion is bottlenecked by compute or Pallas launch overhead rather than memory bandwidth, and the hypothesis is refuted.

## Setup
Rerun of v054-e with fixed import of `_SPLASH_MESH` inside `fused_rope.py`.

## Results
The workload was rejected by the cluster webhook with `follower pod node selector for topology domain not found`.

## Profile
missing

## HLO Dump
missing

## Verdict
**invalid**

Crashed due to a cluster admission error, not related to the code.
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
