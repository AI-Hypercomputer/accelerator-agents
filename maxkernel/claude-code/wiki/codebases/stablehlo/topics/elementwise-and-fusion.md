---
title: "elementwise & fusion vocabulary"
type: concept
tags: [stablehlo, elementwise, fusion, canonicalization, optimization_barrier, reference]
created: 2026-07-04
updated: 2026-07-04
---

The **elementwise / shape-plumbing ops** — `add`, `multiply`, `broadcast_in_dim`, `reshape`, `transpose`, `select`, `convert`, `iota`, `concatenate`, `slice` — are the connective tissue of a StableHLO program. Individually cheap, they matter for perf as **fusion candidates**: XLA fuses chains of these (and the reductions/dots around them) into single kernels, so the vocabulary here explains why a dozen ops in your program collapse to one `fusion` in the HLO dump. This page also covers `optimization_barrier`, the op that *blocks* that fusion.

## Elementwise & shape ops

Elementwise ops (`add`, `subtract`, `multiply`, `maximum`, `logistic`, `tanh`, `exponential`, `convert`, `compare`, `select`, …) map 1:1 over tensor elements. Shape ops (`broadcast_in_dim`, `reshape`, `transpose`, `concatenate`, `slice`, `pad`, `iota`) rearrange without arithmetic. Together they form the elementwise regions XLA fuses onto the boundaries of a `dot_general` or `reduce`.

## Canonicalization decides what survives to the dump

The [`-stablehlo-aggressive-simplification`](src:docs/generated/stablehlo_optimization_passes.md#-stablehlo-aggressive-simplification) pass rewrites these ops before target lowering. High-value patterns to recognize:

- **Algebraic identities**: `add(X, 0) -> X`, `multiply(X, 0i) -> 0i`, `multiply(X, 1i) -> X`, `subtract(X, X) -> 0`, `select(not(p), t, f) -> select(p, f, t)`.
- **Shape-op collapse**: `broadcast_in_dim(broadcast_in_dim(X)) -> broadcast_in_dim(X)`, `reshape(reshape(X)) -> reshape(X)`, `transpose(X, [iota]) -> X`, `transpose(X, [no_mem_layout_change]) -> reshape(X)`, `broadcast_in_dim(X, [sorted]) -> reshape(X)`.
- **Slice/concat**: `slice(concat(X,Y,Z),…) -> concat(slice(X),…)`, `concatenate(concatenate(X,Y),Z) -> concatenate(X,Y,Z)`.

Full list in [generated-stablehlo_optimization_passes](../sources/generated-stablehlo_optimization_passes.md) (pulled from code comments — high coverage, not exhaustive). `-optimize-float` additionally allows FP-perturbing rewrites like `log(sqrt(x)) -> 0.5*log(x)` ([-stablehlo-aggressive-folder](src:docs/generated/stablehlo_optimization_passes.md#-stablehlo-aggressive-folder)).

> [!inferred]
> This is why a program you wrote with explicit transposes/reshapes shows *fewer* ops in the HLO dump — the canonicalizer ate the no-ops. Conversely, a transpose that *does* change memory layout survives and costs a real copy. When counting ops in a dump, remember you're seeing the post-canonicalization form.

## optimization_barrier — the anti-fusion op

`optimization_barrier` is an identity (`result = operand`) that **forces producers to run before consumers and blocks compiler code motion across it** ([optimization_barrier](src:docs/spec.md#optimization_barrier)).

> [!inferred]
> This is the op frameworks insert to control **rematerialization** and scheduling — e.g. pinning an activation so it isn't recomputed, or preventing a fusion that would blow up a live buffer. Seeing `optimization-barrier` in a dump marks a deliberate scheduling/memory boundary, not dead code.

## See also

- [optimization-passes](optimization-passes.md) (the pass that applies these rewrites), [reduce-and-windowing](reduce-and-windowing.md), [dot_general](dot_general.md)
- CHLO's implicitly-broadcasting elementwise ops: [generated-chlo](../sources/generated-chlo.md)
- Source: [spec.md](../sources/spec.md), [generated-stablehlo_optimization_passes.md](../sources/generated-stablehlo_optimization_passes.md)

## Sources

- `raw/code/stablehlo/docs/spec.md`
- `raw/code/stablehlo/docs/generated/stablehlo_optimization_passes.md`
