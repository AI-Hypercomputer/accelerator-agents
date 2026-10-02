---
title: "Qwen3 jax — splash attention"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: supported
expected_gain: "15-40% on attention; unlocks seq 8192"
confidence: high
effort: M
origin: 2026-06-02-qwen3-jax-v6e8-baseline
hlo_prefilter: pending
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 jax — splash attention

*Hypothesis*: Replacing the XLA-SDPA (`jnp.einsum`) attention in
`model/modeling_qwen3.py` with the TPU **splash attention** Pallas kernel
(GQA-native) eliminates the `[B,H,S,S]` score materialization, cutting attention
HBM traffic and removing the OOM that blocks seq 8192.

*Mechanism*: Add an env-gated `_attn_splash` path (mirror the llama3 jax lane's
`splash_attn.py` + `set_splash_mesh`), shard `P('fsdp','tp',_,_)`. QK-norm is
applied before the kernel, so no kernel change.

*Falsification criterion*: no step-time improvement at seq 2048 AND still OOMs at
seq 8192 → refuted.
