---
title: "Qwen3 jax — gradient rematerialization"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: supported
expected_gain: "unlock bs>1 / seq>2048 (cut ~43G HLO-temp to ~1-layer footprint)"
confidence: high
effort: S
origin: 2026-06-02-v001-batch-bs2
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 jax — gradient rematerialization

*Hypothesis*: Wrapping each decoder layer in `jax.checkpoint`
(policy `nothing_saveable`) cuts the activation HLO-temp from ~43 G (all layers'
saved activations + attention scores) to roughly one layer's footprint,
unlocking bs≥2 and longer seq that currently OOM at compile.

*Mechanism*: The minimal jax trainer has **no remat** — `v001` (bs=2) and `v002`
(bs=4) both OOM'd at compile with 100% HLO-temp, dominated by many
`bf16[B,32,2048,2048]` attention-score copies across layers. Per-layer checkpoint
keeps only one layer's activations live during backward recompute.

*Predicted signal*: compile HBM drops well under 31.25 G at bs=2/seq2048; the run
reaches steps; modest step-time cost from recompute (~1.3×) traded for the memory
headroom. Pairs with [splash](qwen3-jax-splash-attention.md) (which removes the
score tensor entirely).

*Falsification criterion*: still OOMs at bs=2/seq2048 after remat (→ the score
tensor alone exceeds HBM; splash is then the only lever), or step-time regression
outweighs the unlock with no net throughput at a feasible batch.
