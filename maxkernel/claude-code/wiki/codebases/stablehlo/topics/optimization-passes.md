---
title: "optimization passes — the compiler-pass vocabulary"
type: concept
tags: [stablehlo, passes, canonicalization, folding, legalization, reference]
created: 2026-07-04
updated: 2026-07-04
---

The StableHLO **pass vocabulary** — the `-stablehlo-…` / `-vhlo-…` / `-chlo-…` names you see in compiler logs, pass pipelines, and `stablehlo-opt` invocations. Under the reference lens: these passes are what transform the IR between what a framework emits and what reaches the TPU backend, so recognizing a pass name tells you *which rewrite stage* rewrote an op you're tracking. There are two doc sources: target-independent optimization ([generated-stablehlo_optimization_passes](../sources/generated-stablehlo_optimization_passes.md)) and the broader transformation set ([generated-stablehlo_passes](../sources/generated-stablehlo_passes.md)).

## Target-independent optimization (the everyday ones)

- **[`-stablehlo-target-independent-optimization`](src:docs/generated/stablehlo_optimization_passes.md#-stablehlo-target-independent-optimization)** — the umbrella pass; runs simplification + folding in one pattern set. Doc says **prefer this** over the two below.
- **[`-stablehlo-aggressive-simplification`](src:docs/generated/stablehlo_optimization_passes.md#-stablehlo-aggressive-simplification)** — canonicalization (algebraic identities, shape-op collapse, dynamic→static). See [elementwise-and-fusion](elementwise-and-fusion.md) for the pattern catalog.
- **[`-stablehlo-aggressive-folder`](src:docs/generated/stablehlo_optimization_passes.md#-stablehlo-aggressive-folder)** — constant folding; `-fold-op-element-limit` caps memory blowup, `-optimize-float` allows FP-perturbing rewrites.

## Legalization / lowering passes

Convert between dialects on the way in/out of StableHLO ([generated-stablehlo_passes](../sources/generated-stablehlo_passes.md)):

- [`-chlo-legalize-to-stablehlo`](src:docs/generated/stablehlo_passes.md#-chlo-legalize-to-stablehlo) — lower the higher-level [CHLO](../sources/generated-chlo.md) ops (implicit-broadcast elementwise, transcendentals) into StableHLO.
- [`-shape-legalize-to-stablehlo`](src:docs/generated/stablehlo_passes.md#-shape-legalize-to-stablehlo), [`-stablehlo-convert-to-signless`](src:docs/generated/stablehlo_passes.md#-stablehlo-convert-to-signless), [`-stablehlo-legalize-deprecated-ops`](src:docs/generated/stablehlo_passes.md#-stablehlo-legalize-deprecated-ops).
- Backend legalizations to Linalg / TOSA live in [generated-stablehlo_linalg_passes](../sources/generated-stablehlo_linalg_passes.md) and [generated-stablehlo_tosa_passes](../sources/generated-stablehlo_tosa_passes.md).

## Dynamism refinement

[`-stablehlo-refine-arguments`](src:docs/generated/stablehlo_passes.md#-stablehlo-refine-arguments) → [`-stablehlo-refine-shapes`](src:docs/generated/stablehlo_passes.md#-stablehlo-refine-shapes) → [`-stablehlo-canonicalize-dynamism`](src:docs/generated/stablehlo_passes.md#-stablehlo-canonicalize-dynamism) → [`-stablehlo-check-shape-assertions`](src:docs/generated/stablehlo_passes.md#-stablehlo-check-shape-assertions) turn a dynamic program static. Full walkthrough in [dynamism](dynamism.md).

## Quantization & composites

- Quant lowering: [`-stablehlo-legalize-qdq-to-quantized-op`](src:docs/generated/stablehlo_passes.md#-stablehlo-legalize-qdq-to-quantized-op), [`-stablehlo-legalize-quantized-op-to-qdq`](src:docs/generated/stablehlo_passes.md#-stablehlo-legalize-quantized-op-to-qdq), [`-stablehlo-legalize-quant-to-math`](src:docs/generated/stablehlo_passes.md#-stablehlo-legalize-quant-to-math) — see [quantization](quantization.md).
- Composites: [`-stablehlo-legalize-composite-to-call`](src:docs/generated/stablehlo_passes.md#-stablehlo-legalize-composite-to-call), [`-stablehlo-wrap-in-composite`](src:docs/generated/stablehlo_passes.md#-stablehlo-wrap-in-composite) (wrap ops in a named `stablehlo.composite` with a decomposition — the mechanism behind marking a subgraph as a known kernel).
- Compatibility: [`-stablehlo-legalize-to-vhlo`](src:docs/generated/stablehlo_passes.md#-stablehlo-legalize-to-vhlo) / [`-vhlo-to-version`](src:docs/generated/stablehlo_passes.md#-vhlo-to-version) / [`-stablehlo-compatibility-expander`](src:docs/generated/stablehlo_passes.md#-stablehlo-compatibility-expander) — see [compatibility-and-versioning](compatibility-and-versioning.md).

> [!inferred]
> These StableHLO passes are the *portability-layer* rewrites; XLA's own TPU passes (fusion, layout assignment, latency-hiding scheduler) run downstream and are where most TPU-specific optimization actually happens. Still, a `composite` in a dump often marks a framework-designated kernel boundary, and a lingering `dynamic_*` op means shape refinement didn't fully fire.

## See also

- [elementwise-and-fusion](elementwise-and-fusion.md), [dynamism](dynamism.md), [quantization](quantization.md), [compatibility-and-versioning](compatibility-and-versioning.md)
- Sources: [generated-stablehlo_optimization_passes.md](../sources/generated-stablehlo_optimization_passes.md), [generated-stablehlo_passes.md](../sources/generated-stablehlo_passes.md)

## Sources

- `raw/code/stablehlo/docs/generated/stablehlo_optimization_passes.md`
- `raw/code/stablehlo/docs/generated/stablehlo_passes.md`
