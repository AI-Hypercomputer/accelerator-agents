---
title: "in-kernel dropout from prng_key + block indices"
type: concept
tags: [pallas, dropout, prng, attention, axlearn, stub]
created: 2026-04-23
updated: 2026-04-23
---

Generating dropout masks inside a Pallas attention kernel directly from the caller's prng_key and the kernel's block indices, rather than host-materializing the mask tensor and passing it in. *Stub — expand when more sources are available.*

## Definition

Instead of producing a `[B, H, Q, K]` dropout-mask tensor on the host and feeding it to the kernel, the kernel derives the per-position Bernoulli mask on-the-fly from `(prng_key, q_block_idx, kv_block_idx)`. The key + block indices are a deterministic hash input; each position gets a reproducible pseudo-random bit.

## Why it matters for TPU perf

A host-materialized `[B, H, Q, K]` mask at typical training shapes is a large HBM write (e.g., `[8, 32, 8192, 8192] × 1 byte = 16 GiB`). Generating the mask in-kernel means zero HBM traffic for the mask — one less round-trip in the memory-bound attention loop.

## Mechanism

1. Accept `prng_key` as a Pallas kernel input.
2. Inside the kernel, hash `(prng_key, program_id(0), program_id(1), ...)` to derive block-local randomness.
3. Apply the Bernoulli mask to the attention scores before softmax.

Known Pallas gotcha: **`key<pl>` lowering has historically been buggy**; current workaround is to **prefetch the prng_key** as a kernel input rather than relying on in-kernel key-splitting.

## When it applies / when it doesn't

- **Applies** to any attention training path with dropout > 0 where the mask would otherwise materialize.
- **Does not apply** at inference (dropout = 0) nor at bf16/fp32 boundaries where the mask-gen arithmetic is cheaper on host.

## Known results

| Reference | Repo | Notes |
|---|---|---|
| `axlearn/common/flash_attention/tpu_splash_attention.py` | [axlearn](../codebases/axlearn.md) | First-party splash extension; documents the `key<pl>` workaround |

## Connections

- [flash-attention](flash-attention.md)
- [splash-attention](splash-attention.md)
- [pallas-kernel](pallas-kernel.md)

## Sources

- [axlearn codebase](../codebases/axlearn.md) "Performance-relevant surfaces §4".
- [Pallas kernel directory §4.1](../analyses/pallas-kernel-directory/04-research-labs.md#41-appleaxlearn).
