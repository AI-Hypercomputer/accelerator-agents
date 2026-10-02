---
title: "Qwen3 jax — AMP fp32-master / bf16-compute"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: open
expected_gain: "2-5% MFU"
confidence: medium
effort: M
origin: 2026-06-02-qwen3_cc-jax-retrospective
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 jax — AMP fp32-master / bf16-compute

The trainer currently casts weights to a single `weights_dtype` (bf16). True mixed precision
keeps an fp32 master copy for the optimizer state/update while casting to bf16 only for the
matmul compute path. This is a standard MFU lever and is unexplored on this lane (retrospective
gap: dtype/AMP).

## Statement

Separating a bf16 compute dtype from an fp32 master-weight/optimizer dtype (true AMP), stacked
on the v008 frontier, raises MFU ≥ +1 pp over 32.4% without changing the loss trajectory beyond noise.

## Rationale

- Today optimizer and compute share bf16; an fp32 master copy is the conventional AMP recipe and
  often improves both numerics and effective MFU (cleaner matmul shapes, fewer rescales).
- Orthogonal to splash/remat/CE — composes with the frontier.
- Listed as rec #3 in the [retrospective](../analyses/2026-06-02-qwen3_cc-jax-retrospective.md).

## Proposed experiment

Code change (via `/edit-model-code`): add a `--compute_dtype` flag distinct from `--weights_dtype`;
cast params to compute_dtype inside the forward, keep fp32 master in the optax state. Build a tagged
variant image, smoke-test `--help`, dispatch on the v008 frontier shape. Measure MFU + loss parity.

## Risks

- If it changes the loss trajectory materially → semantic concern; verify parity over profiled steps.
- May not help if XLA already runs matmuls in bf16 regardless (check HLO dtype on the dots).

## Dependencies

Trainer edit + image build.
