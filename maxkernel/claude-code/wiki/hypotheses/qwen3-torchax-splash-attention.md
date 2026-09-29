---
title: "Qwen3 torchax — splash attention"
type: hypothesis
model: qwen3-cc-torchax
variants: ["8B/v6e-8"]
status: open
expected_gain: "15-40% on attention; unlocks seq 8192"
confidence: high
effort: M
origin: 2026-06-02-qwen3-torchax-v6e8-baseline
hlo_prefilter: pending
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 torchax — splash attention

*Hypothesis*: Replacing the XLA-through-HF `scaled_dot_product_attention` with the
TPU **splash attention** Pallas kernel (GQA-native, 32q/8kv) eliminates the
`[B, n_heads, S, S]` score materialization, cutting attention HBM traffic and
removing the OOM that blocks seq 8192 on the minimal trainer.

*Mechanism*: Override `F.scaled_dot_product_attention` with the splash kernel
(canonical torchax pattern, bridged via `torchax.interop.call_jax`), mesh-sharded
`P('fsdp','tp',_,_)`. GQA is native — do not `repeat` K/V. Qwen3 applies QK-norm
before RoPE; the kernel sees post-norm post-RoPE Q/K, so no kernel change needed.

*Predicted signal*: attention's share of the step drops; the 66% TC idle in the
[baseline](../experiments/qwen3_cc_autoresearch_optimization/torchax/experiments/2026-06-02-qwen3-torchax-v6e8-baseline.md)
shrinks; seq 4096/8192 become runnable without OOM.

*Falsification criterion*: no measurable step-time improvement at seq 2048 **and**
still OOMs at seq 8192 → refuted.
