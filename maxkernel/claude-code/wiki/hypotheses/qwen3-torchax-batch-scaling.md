---
title: "Qwen3 torchax — per-chip batch scaling"
type: hypothesis
model: qwen3-cc-torchax
variants: ["8B/v6e-8"]
status: open
expected_gain: "1.5-2.5x throughput / MFU"
confidence: high
effort: S
origin: 2026-06-02-qwen3-torchax-v6e8-baseline
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 torchax — per-chip batch scaling

*Hypothesis*: Raising `--batch_size` (per-chip batch > 1) fills the **66% TC idle**
seen in the baseline, lifting MFU well above 19.2% with no code change — until
HBM (Adam state + activations + `[B,L,V]` logits) becomes the binding constraint.

*Mechanism*: The baseline ran per-chip batch 1 (global 8 on fsdp=8). The device
is idle ⅔ of the step (FSDP-collective + dispatch bound). More work per step
amortizes the fixed per-step overhead and raises MXU occupancy.

*Predicted signal*: MFU and tok/s/chip rise roughly with batch until an HBM wall;
`tc_idle` fraction falls.

*Falsification criterion*: throughput/chip does **not** improve as batch rises
(overhead is not the bottleneck), or OOM appears before any meaningful gain →
refuted (and re-rank splash / CE first to win memory).

*Dependencies*: at seq ≥ 4096 the `[B,L,V]` logits + N² scores cap batch — pairs
with [tokamax CE](qwen3-torchax-tokamax-ce.md) and [splash](qwen3-torchax-splash-attention.md).
