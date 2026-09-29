---
title: "Qwen3 JAX v217 scan custom-VJP tiled CE (LOCAL INVALID, CPU equivalence drift)"
type: experiment
tags: [experiment, qwen3-cc, jax, v6e-8, ce, logit-memory, custom-vjp, scan, invalid, local-only]
hypothesis: qwen3-jax-scan-custom-vjp-vocab-tiled-ce
model: qwen3-cc-jax
created: 2026-06-05
updated: 2026-06-05
status: invalid
---

# v217 scan custom-VJP tiled CE local invalid

## Hypothesis Under Test

[v216](2026-06-05-v216-no-scan-custom-vjp-tiled-ce-bs4.md) showed that an
unrolled Python custom VJP removes saved predicate masks but still fails TPU
compile at **38.12G / 31.25G HBM** because the backward retains many
`f32[32768,4096]` chunk-dot temps. This local candidate tried to express the
same tiled-CE forward/backward as `lax.scan`, expecting XLA to keep the vocab
chunk body loop-shaped rather than unrolled.

Expected outcome:

- Supported for TPU launch only if local CPU equivalence matches the full
  lm-head CE within the existing toy-test tolerances.
- Invalid if the scan formulation changes numerics enough to fail the local
  equivalence gate.

## Setup

- **Fork**:
  `wiki/experiments/qwen3_cx_autoresearch_optimization/jax/.repo/v217-scan-custom-vjp-tiled-ce`
- **Base image**:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v216-custom-vjp-tiled-ce`
- **Intended runtime shape**: v210 stack, seq8192, per-chip batch 4/global
  batch 32, tiled CE chunk size 4096.
- **TPU workload**: not launched.

## Code Change

The local fork keeps v216's custom-VJP residual contract but replaces the
Python chunk loop with `lax.scan` over vocab chunks. Two variants were checked:

- padded/reshaped lm-head weight chunks;
- dynamic-slice chunks with tail overlap.

The dynamic-slice version was also checked on a divisible toy vocab size to
remove tail handling as a possible cause.

## Result

Invalid before TPU launch. `py_compile` passed, but CPU toy equivalence against
full CE failed on the loss value:

| Variant | Toy shape | Loss delta | Hidden grad max delta | Lm-head grad max delta |
|---|---|---:|---:|---:|
| padded scan | V=17, chunk=4 | 0.0066075325 | 0.001953125 | 0.0009765625 |
| dynamic-slice scan | V=17, chunk=4 | 0.0066075325 | 0.001953125 | 0.0009765625 |
| dynamic-slice scan | V=16, chunk=4 | 0.0066699982 | 0.001953125 | 0.0009765625 |

Changing `lax.Precision` among default, high, and highest did not remove the
loss drift. Because the same drift appears with divisible vocab/chunk sizes,
the cause is the scan/dot reduction structure rather than padding or tail
masking.

## Verdict

Invalid local candidate. Do not launch v217 on TPU and do not continue the
scan-custom-VJP path without first restoring strict CE/loss equivalence. The CE
branch has now refuted naive autodiff, remat-only, smaller chunks, unrolled
custom VJP, and the cheap scan custom-VJP rewrite. The next CE/logit-memory
attempt should be either a more faithful streaming/Pallas kernel or a different
structural approach with an explicit local equivalence gate before cluster use.
