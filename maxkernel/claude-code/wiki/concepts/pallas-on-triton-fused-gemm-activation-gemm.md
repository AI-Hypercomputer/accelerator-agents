---
title: "Pallas-on-Triton fused GEMM + activation + GEMM"
type: concept
tags: [pallas, triton, gpu, gated-linear-unit, swiglu, fused-activation, alphafold3, stub]
created: 2026-04-23
updated: 2026-04-23
---

Pattern: two matmuls sharing one activation load, with the activation function (SiLU / GELU / ...) applied between them — all in a single Pallas-on-Triton kernel with optional epilogue and `dst` output aliasing. First-party reference: AlphaFold3 `PallasGatedLinearUnit` @ v3.0.1. *Stub — expand when more sources are available.*

## Definition

For gated linear units (SwiGLU / GEGLU / REGLU / GLU): `out = activation(x @ W_gate) * (x @ W_up)`. Naive implementation does four passes (load x, W_gate, matmul, apply activation, load x, W_up, matmul, multiply) — eight HBM transactions for x + weights + intermediates. The fused kernel does one pass: load x + W_gate + W_up once, compute both matmuls, apply activation between, multiply into output.

## Why it matters for TPU perf

Saves two HBM round-trips on the gate / up intermediates. On memory-bound activation layers, roughly proportional to HBM-bandwidth reduction. **GPU-side the pattern is well-established** (AlphaFold3 ships it in production). **TPU-side** is still open: tokamax's fused GLU falls back to XLA on TPU, and exp 33 suggests XLA may already fuse enough that a Pallas version regresses (same lesson as RMSNorm).

## Mechanism

Inside the kernel:
1. Load `x_tile`, `w_gate_tile`, `w_up_tile` in one pass.
2. `gate_preact = x_tile @ w_gate_tile`
3. `gate = activation(gate_preact)` (SiLU / GELU / ReLU)
4. `up = x_tile @ w_up_tile`
5. `out_tile = gate * up` (optional epilogue applied before write)
6. Write `out_tile` (with optional `dst` aliasing to avoid extra HBM buffer).

Autotuned over `block_m ∈ [32, 128]`, `block_n ∈ [32, 256]`, `block_k = 32`, with split-K fallback when `num_blocks < core_count`.

## When it applies / when it doesn't

- **Applies** to GPU Pallas / Triton for any gated-linear-unit MLP block.
- **Does not apply on TPU yet** — tokamax GLU falls back to XLA; Gemma 4 exp 33's RMSNorm result suggests XLA fusion may already be sufficient. Needs HLO-level validation before porting to Mosaic-TPU.

## Known results

| Reference | Repo | Notes |
|---|---|---|
| `PallasGatedLinearUnit` / `_gated_linear_unit_kernel` | [alphafold3 @ v3.0.1](../codebases/alphafold3.md) | Only public production-grade Pallas fused GLU; GPU via Triton |
| `gated_linear_unit.gated_linear_unit()` public API | [alphafold3 @ v3.0.1](../codebases/alphafold3.md) | Dispatcher; tries triton, falls back to XLA on exception |

## Connections

- [gated-linear-unit](gated-linear-unit.md)
- [xla-fusion](xla-fusion.md)
- [pallas-kernel](pallas-kernel.md)
- [grouped-program-ids-for-l2](grouped-program-ids-for-l2.md) — GPU-side co-pattern.

## Sources

- [alphafold3 codebase](../codebases/alphafold3.md) "Performance-relevant surfaces §1".
- [Pallas kernel directory §4.5](../analyses/pallas-kernel-directory/04-research-labs.md#45-google-deepmindalphafold3-pinned-to-tag-v301).
