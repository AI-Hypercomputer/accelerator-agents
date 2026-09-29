---
title: "jaxite (google/jaxite)"
type: codebase
tags: [jaxite, fhe, cggi, bootstrap, pallas, mosaic-tpu, non-ml, integer-arithmetic, niche]
repo: google/jaxite
commit: e4a3351
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [jaxite overview](jaxite/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

Google's JAX-based **Fully Homomorphic Encryption** (FHE) library. Ingested for one data-point: `jaxite_lib/polymul_kernel.py` — the only non-ML Pallas TPU kernel in this wiki. CGGI boolean-gate bootstrap via **negacyclic polymul** implemented as **four byte-split bf16 matmuls reassembled to u32**. Concrete reference for integer-heavy Pallas patterns. Per-kernel detail in [§5 of the Pallas kernel directory](../analyses/pallas-kernel-directory/05-frameworks-quant.md).

## Architecture

- `jaxite/jaxite_lib/polymul_kernel.py` — the Pallas kernel (TPU).
- Rest of `jaxite/` — CGGI primitives + bootstrapping orchestration (out of scope for the wiki).

## Key abstractions

| Kernel | File | Backend | Notes |
|---|---|---|---|
| Negacyclic polymul (CGGI bootstrap core) | `jaxite/jaxite_lib/polymul_kernel.py` | `mosaic_tpu` | **Novel** non-ML Pallas: 4 bytes-split bf16 matmuls reassembled to u32; `bat_matmul` batched variant |

## Performance-relevant surfaces

### 1. Integer-heavy Pallas via bf16 byte-split + reassembly
The only public precedent in this wiki for handling **integer arithmetic inside Pallas TPU** despite Pallas being bf16/fp-centric. Four bf16 matmuls over byte slices + integer reassembly to u32 preserves integer correctness while using the MXU. Template if any future hypothesis needs integer / int4 / mixed-precision matmul on TPU.

### 2. Confirmation Pallas isn't ML-only
Non-ML domains (FHE, crypto, scientific computing) can use the same Pallas DSL — useful grounding point for the wiki's scope.

## Connections

- [Pallas kernel directory §5](../analyses/pallas-kernel-directory/05-frameworks-quant.md)
- [pallas-kernel](../concepts/pallas-kernel.md)
- [int8-quantization](../concepts/int8-quantization.md) — conceptually-related integer-arithmetic patterns.

## Sources

- `raw/code/jaxite/jaxite/jaxite_lib/polymul_kernel.py`
- Upstream: <https://github.com/google/jaxite>
