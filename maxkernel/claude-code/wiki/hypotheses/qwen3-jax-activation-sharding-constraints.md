---
title: "Qwen3-jax: activation sharding constraints (MaxText with_logical_constraint)"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "REFUTED — v041 parity (+0.15%); no-op in pure FSDP (XLA already infers the layout)"
confidence: medium
effort: M
origin: human
hlo_prefilter: n/a
tags: [qwen3-cc, jax, sharding, activation-sharding, with-logical-constraint, maxtext, seq8192]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3-jax: activation sharding constraints

## Statement

Our model applies **zero** activation sharding constraints; MaxText pins every layer-boundary activation via
`nn.with_logical_constraint(("activation_batch","activation_length","activation_embed"))` (and the MLP
intermediate via `activation_mlp`). Adding the same constraints — pinning activations to the FSDP
data-parallel layout `P("fsdp", ...)` at the layer input, both post-norms, the attention output, and the MLP
intermediate — stops XLA's SPMD partitioner from inserting relayout/copy ops and resharding the residual
stream, lifting the seq8192 frontier above v035 (6,030).

## Rationale

This is the **obvious code-level gap** (the config search was exhausted; this is structural). The
[v035 profile](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v035-maxtext-ce-s8k-bs3.md)
shows our **loop-fusion (norms/acts) at 17.4%** vs MaxText's **7.2%** (2.4×) and **data-formatting 3.8%** vs
MaxText 1.8% — exactly the signature of spurious layout conversions / resharding that explicit activation
constraints prevent. MaxText's [`layers/qwen3.py`](../../?) wraps `inputs`, `lnx`, `attention_output`,
`mlp_input` in `with_logical_constraint`, and [`layers/linears.py`] wraps the MLP intermediate. We never
ported any of this — the residual stream and intermediates are left to XLA's inference.

## Proposed experiment

- v041: v035 config + `--shard_acts` (image v041-shard-acts; `_sac()` pins activations to `P("fsdp", ...)` at
  the MaxText constraint points). Support: > 6,030 (loop-fusion/data-formatting drop). Refute: ≤ 6,030 (XLA
  already infers the optimal layout in pure FSDP; the constraints are no-ops / the inflated buckets are not
  layout-conversion).

## See also

- [Qwen3 8B — jax](../models/qwen3-cc-jax.md) · [v035 profile](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v035-maxtext-ce-s8k-bs3.md) · [retrospective #5](../analyses/2026-06-02-qwen3_cc-jax-retrospective-5-maxtext-delta.md)
