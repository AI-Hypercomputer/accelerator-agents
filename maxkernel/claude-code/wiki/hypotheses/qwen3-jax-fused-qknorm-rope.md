---
title: "Qwen3 jax — Fused QK-norm + RoPE Pallas kernel"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: open
expected_gain: "+2-3pp MFU"
confidence: medium
effort: L
origin: 2026-06-03-qwen3-jax-retrospective
hlo_prefilter: unchecked
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 jax — Fused QK-norm + RoPE Pallas kernel

*Hypothesis*: The Qwen3 architecture uniquely applies RMSNorm to Q and K immediately before applying RoPE. In the XLA implementation, this requires materializing the full `[B, S, H, D]` tensors between the norm and the RoPE operation, leading to significant HBM bandwidth pressure and HLO fragmentation. By fusing RMSNorm and RoPE into a single Pallas kernel, we can keep the intermediate normalized values in SRAM/VMEM and only write the final RoPE'd vectors to HBM, saving two memory traversals and ~2-3pp MFU.

*Mechanism*: Write a custom Pallas kernel `fused_qknorm_rope` that takes un-normalized `q` and `k`, the norm weights, and the RoPE frequencies. Inside the block, it computes the RMSNorm denominator, applies the scale, and then immediately applies the RoPE rotation. Substitute `jnp.einsum`/`rmsnorm` followed by `apply_rotary_emb` with this kernel call in `model/modeling_qwen3.py`.

*Falsification criterion*: If the step time does not improve by at least 2% MFU (or the compiled memory footprint is not reduced), then the fusion is bottlenecked by compute or Pallas launch overhead rather than memory bandwidth, and the hypothesis is refuted.
