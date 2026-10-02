---
title: "dynamism — dynamic shapes and shape refinement"
type: concept
tags: [stablehlo, dynamism, dynamic-shapes, shape-refinement, reference]
created: 2026-07-04
updated: 2026-07-04
---

**Dynamism** — how StableHLO carries unknown tensor dimensions and refines them back to static. Under the reference lens: `tensor<?x…>` types and `dynamic_*` ops (`dynamic_broadcast_in_dim`, `dynamic_reshape`, `dynamic_slice`, `get_dimension_size`) in a dump signal a not-yet-specialized program. Since TPU wants static shapes for efficient compilation, this page explains what kind of dynamism you're looking at and which passes collapse it.

## Kinds of dynamism

From [dynamism-terminology-support-overview](src:docs/dynamism.md#dynamism-terminology-support-overview):

- **[Dynamic dimensions](src:docs/dynamism.md#dynamic-dimensions)** — unknown size, written `?` (`tensor<16x?xf32>`).
- **[Bounded dynamism](src:docs/dynamism.md#bounded-dynamism)** — dynamic dim with a known upper bound, encoded `#stablehlo.bounds<16, ?>`; the runtime pads to the bound. Limited framework support (TF, some PyTorch/XLA).
- **[Unbounded dynamism](src:docs/dynamism.md#unbounded-dynamism)** — no bound (`tensor<?x?xf32>`); common, broad JAX/PyTorch-XLA/TF support; used for dynamic batch/seq-len export.
- **[Shape polymorphism](src:docs/dynamism.md#shape-polymorphism)** (from JAX) — all dynamism traces to input args and is shape-only (not data-dependent), so known input shapes ⇒ fully static program.
- **[Data-dependent dynamism](src:docs/dynamism.md#data-dependent-dynamism)** — dim sizes depend on tensor *contents* (`nonzeros`); modeled via bounded dynamism + padding.

## Refinement: dynamic → static

The remove-dynamism pipeline ([remove-dynamism-pass-pipeline](src:docs/dynamism.md#remove-dynamism-pass-pipeline), [compiler-passes-for-refining-dynamic-programs](src:docs/dynamism.md#compiler-passes-for-refining-dynamic-programs)) runs:

1. [`stablehlo-refine-arguments`](src:docs/dynamism.md#individual-passes-for-refining-dynamism) — replace `?` input types with concrete shapes.
2. `stablehlo-refine-shapes` — propagate those shapes through the program.
3. `stablehlo-canonicalize-dynamism` — replace `dynamic_*` ops with static counterparts.

The [add_one example](src:docs/dynamism.md#refining-the-dynamic-model) walks `tensor<?xf32>` (`get_dimension_size` + `dynamic_broadcast_in_dim`) all the way back to the [static form](src:docs/dynamism.md#static-add_one-model) `tensor<16xf32>`, and finally the aggressive-simplification pass folds the constant broadcast away.

> [!inferred]
> Reading a dump: a lingering `dynamic_broadcast_in_dim` / `dynamic_reshape` (rather than the static `broadcast_in_dim` / `reshape`) means shape refinement didn't fully fire — the program is still shape-polymorphic. That usually blocks the best static-shape TPU codegen; forcing concrete input shapes and re-running refinement is the fix. `get_dimension_size` marks where a runtime dimension is read.

## See also

- [optimization-passes](optimization-passes.md) (the refinement + canonicalization passes), [type-inference](type-inference.md), [elementwise-and-fusion](elementwise-and-fusion.md)
- Sources: [dynamism.md](../sources/dynamism.md), [generated-stablehlo_passes.md](../sources/generated-stablehlo_passes.md)

## Sources

- `raw/code/stablehlo/docs/dynamism.md`
