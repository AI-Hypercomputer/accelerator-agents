---
title: "Qwen3 jax — target sequence length 8192"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: supported
expected_gain: "reaches the program target seq (8192); MFU at target shape TBD"
confidence: medium
effort: S
origin: 2026-06-12-v002-splash
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — target sequence length 8192

*Hypothesis*: With splash attention (landed,
[v002](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v002-splash.md))
and a working streamed CE ([bf16-inputs variant](qwen3-jax-tokamax-ce-bf16-inputs.md)),
seq 8192 at bs1 fits in 31 GB/chip and trains at competitive MFU — the
model-level program's target sequence length.

*Mechanism*: The two seq-scaling memory walls are the S² attention scores
(eliminated by splash) and the `[B,L,V]` logits (~7.5 GB at seq 8192 in
f32+bf16; eliminated by streamed CE). Remaining activation growth is linear.

*Falsification criterion*: OOM at seq 8192 bs1 with both kernels on →
blocked; needs remat or sharding change.

*Dependencies*: [qwen3-jax-tokamax-ce-bf16-inputs](qwen3-jax-tokamax-ce-bf16-inputs.md)
(or another streamed-CE variant) must land first.

## See also

- [v002-splash](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v002-splash.md)

**Supported** by [v015](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v015-save-attn-remat.md)
(2026-06-12): first seq-8192 run — 1475.7 ms / 44,361 TPS / **31.8% MFU**
(splash + chunked CE f32-x + save_attn remat + vmem flag).
