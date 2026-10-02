---
title: "manual MLIR-dialect Pallas (below `pallas_call`)"
type: concept
tags: [pallas, mosaic-tpu, mlir, dialect, sparsecore, tcgen05, maxtext, tokamax, stub]
created: 2026-04-23
updated: 2026-04-23
---

When `pl.pallas_call` doesn't expose the hardware feature you need (SparseCore primitives, Blackwell TCGEN05 / TMEM, specific register layouts), drop below it to `jax.experimental.mosaic` + `jaxlib.mlir.dialects.{arith, func, memref, scf, vector}`. First-party references: MaxText `sc_gather_reduce.py` (SparseCore), tokamax SM100 attention (TMEM). *Stub — expand when more sources are available.*

## Definition

Pallas is a Python DSL that lowers through the Mosaic MLIR dialect to hardware code. `pl.pallas_call` wraps this lowering with a convenient kernel-body Python function. But some hardware features (SparseCore primitives, Blackwell TMEM operations) don't have a Python API exposed — you have to **construct the MLIR directly** using `jax.experimental.mosaic` dialect registration and `jaxlib.mlir.dialects` constructors.

## Why it matters for TPU perf

Used only when `pl.pallas_call` literally cannot express the operation. SparseCore kernels on v5p/v7x and Blackwell TMEM kernels on sm100 are the two confirmed cases. The resulting kernel is less maintainable (raw MLIR, no Python syntactic sugar) but gives access to hardware features otherwise inaccessible.

## Mechanism

1. Write a Python function that constructs MLIR using `jaxlib.mlir.dialects.{arith, func, memref, scf, vector}` builders.
2. Register the function with `jax.experimental.mosaic` as a custom dialect pattern.
3. Invoke as a JAX `custom_call` — not a `pallas_call`.

Autotune surface is typically larger (more raw knobs exposed) than via `pallas_call`.

## When it applies / when it doesn't

- **Applies** to SparseCore operations (`sc_gather_reduce` family), Blackwell TMEM-resident matmul/attention (tokamax `mosaic_gpu_sm100`), and any other feature without a `pl.pallas_call` wrapper.
- **Does not apply** when the kernel can be expressed in standard Pallas — the extra complexity isn't justified.

## Known results

| Reference | Repo | Notes |
|---|---|---|
| `gather_reduce_sc.py` | [maxtext](../codebases/maxtext.md) | SparseCore MoE unroute/reduce; heavy autotune surface (`col_chunk_size`, `row_chunk_size`, `loop_unroll_factor_{1,2,3}`, etc.) |
| SM100 Blackwell attention | [tokamax](../codebases/tokamax.md) | Uses `mosaic_gpu_sm100` + raw dialects for TCGEN05/TMEM |

## Connections

- [pallas-kernel](pallas-kernel.md)
- [mosaic-kernel](mosaic-kernel.md)
- [custom-call](custom-call.md)
- [sparsecore](sparsecore.md)

## Sources

- [maxtext codebase](../codebases/maxtext.md) "Performance-relevant surfaces §4".
- [Pallas kernel directory §2.1](../analyses/pallas-kernel-directory/02-ai-hypercomputer.md#21-ai-hypercomputermaxtext).
