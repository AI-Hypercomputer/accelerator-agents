---
title: "Qwen3-jax: replicate MaxText's exact offload recipe (decoder_layer_input + qkvo, recompute mlp) + offload-pipelining flags"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "REFUTED — v039 (exact MaxText recipe) −6.6% vs no-offload; host-offload doesn't pipeline on our stack"
confidence: medium
effort: M
origin: 2026-06-02-qwen3_cc-jax-retrospective-5-maxtext-delta
hlo_prefilter: n/a
tags: [qwen3-cc, jax, remat, offload, host-offload, maxtext, scan, seq8192]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-jax: MaxText's exact offload recipe

## Statement

v036 refuted host-offload, but **mis-tested it**: it offloaded `q/k/v/o proj + mlpwi` and missed
`decoder_layer_input`, whereas MaxText offloads `decoder_layer_input + q/k/v/o proj` and **recomputes**
`mlpwi`. Replicating MaxText's exact set — offload the attention path (decoder input + qkvo), recompute the
MLP — **plus** the MaxText offload-pipelining flags (`loop_variant_parameter_in_chain`,
`loop_invariant_op_in_chain`, `aggressive_opt_barrier_removal`, `lhs_prioritize_async_depth_over_stall`)
that make the host transfer overlap across the `lax.scan` iterations, lifts the seq8192 frontier above v035
(6,030) toward MaxText (6,942).

## Rationale

[Retrospective #5](../analyses/2026-06-02-qwen3_cc-jax-retrospective-5-maxtext-delta.md): MaxText's
`remat_policy: custom` offloads exactly `decoder_layer_input, out_proj, query_proj, key_proj, value_proj`
(mlpwi stays `remat`). v036 did the opposite on the two key tensors — it shipped the giant `[B,L,12288]`
MLP activations to host (non-overlapped DMA) and never offloaded the residual-stream input. The MLP is
cheap to recompute relative to its host-traffic cost; the attention proj outputs + decoder input are small
and form clean checkpoint boundaries. The `*_in_chain` flags exist specifically for host offload inside
loops — the mechanism the v036 verdict speculated we lacked.

## Proposed experiment

- v039: image with (a) a `decoder_layer_input` `checkpoint_name` tag on the per-layer residual input,
  (b) offload-names = {decoder_layer_input, query_proj, key_proj, value_proj, out_proj} (DROP mlpwi),
  (c) the full MaxText HOST_OFFLOAD flag set incl the `*_in_chain` / opt-barrier flags. bs3 seq8192 vs v035
  (6,030). Support: > 6,030 toward 6,942. Refute: ≤ 6,030 (even the correct recipe doesn't pipeline on our
  build → the host-offload path genuinely differs at the libtpu level).

## See also

- [Retrospective #5](../analyses/2026-06-02-qwen3_cc-jax-retrospective-5-maxtext-delta.md) · [v036 (mis-tested offload)](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v036-maxtext-ce-offload-s8k-bs3.md) · [Qwen3 8B — jax](../models/qwen3-cc-jax.md)
