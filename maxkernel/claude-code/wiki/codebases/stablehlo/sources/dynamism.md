---
title: "dynamism.md — dynamism in StableHLO"
type: source
tags: [stablehlo, docs, dynamism, dynamic-shapes, shape-refinement, reference]
created: 2026-07-04
updated: 2026-07-04
---

How StableHLO models **dynamic shapes** and refines them back to static. Under the reference lens: when a dump shows `tensor<?x…>` types or `dynamic_broadcast_in_dim` / `get_dimension_size` / `dynamic_reshape` ops, this doc explains what kind of dynamism produced them and which passes collapse them to static ops before TPU lowering (dynamic shapes generally can't be compiled efficiently — the target wants static).

## Key points

- **[Terminology](src:docs/dynamism.md#dynamism-terminology-support-overview)**: [dynamic dimensions](src:docs/dynamism.md#dynamic-dimensions) (`?`), [bounded dynamism](src:docs/dynamism.md#bounded-dynamism) (`#stablehlo.bounds<16, ?>` — padded to bound at runtime), [unbounded dynamism](src:docs/dynamism.md#unbounded-dynamism) (elide bounds; common for dynamic batch/seq len).
- **[Shape polymorphism](src:docs/dynamism.md#shape-polymorphism)** (from JAX): all dynamism traces to input args and pertains to shapes only (not data) — so once input shapes are known the whole program refines to static.
- **[Data-dependent dynamism](src:docs/dynamism.md#data-dependent-dynamism)**: dim sizes that depend on tensor *contents* (e.g. `nonzeros`); modeled with bounded dynamism + padding.
- **[Refinement pipeline](src:docs/dynamism.md#compiler-passes-for-refining-dynamic-programs)**: [`stablehlo-refine-arguments`](src:docs/dynamism.md#individual-passes-for-refining-dynamism) → `stablehlo-refine-shapes` → [`stablehlo-canonicalize-dynamism`](src:docs/dynamism.md#refining-the-dynamic-model) (bundled as [`createStablehloRemoveDynamismPipeline`](src:docs/dynamism.md#remove-dynamism-pass-pipeline)). The [add_one example](src:docs/dynamism.md#example-how-is-dynamism-useful-and-how-can-i-use-it) shows a `tensor<?xf32>` program refined all the way back to the [static form](src:docs/dynamism.md#static-add_one-model).

## Feeds these topics

- [dynamism](../topics/dynamism.md)
- [optimization-passes](../topics/optimization-passes.md)
- [type-inference](../topics/type-inference.md)

## Sources

- `raw/code/stablehlo/docs/dynamism.md`
