---
title: "StableHLO"
type: codebase
tags: [compiler, mlir, ir, hlo, reference]
commit: 9793611e490531cc40a7a4e2a4e92caa7ea89f0f
created: 2026-04-22
updated: 2026-07-04
---

StableHLO is the MLIR dialect and operation set that acts as a portability layer between ML frameworks (JAX, PyTorch, TensorFlow) and ML compilers (XLA, IREE). For this wiki it is a **reference** for interpreting HLO op names, semantics, and compiler-pass vocabulary when reading XLA dumps and xprof traces — **not an optimization target**. We do not modify StableHLO.

> [!note] Thin companion page
> The grounded, docs-cited internals wiki now lives at **[stablehlo/overview.md](stablehlo/overview.md)** (ingested via wikify docs-mode from `docs/**/*.md`). Start there for op semantics and pass vocabulary. This page is the on-demand perf-lens companion: *when* and *why* you'd reach for StableHLO from the optimization loop.

## When to reach for StableHLO (perf lens)

You consult StableHLO only to interpret what XLA already emitted — there are no StableHLO-side knobs in this wiki's hypotheses or experiments (optimization knobs live in XLA and the frameworks that emit StableHLO). Use it when an `xla_dump` HLO module or an xprof op name needs its semantics or attributes confirmed:

- **An op's meaning / attributes** — contraction & batch dims of a matmul, scatter update semantics, a collective's replica-group layout, precision config. → [dot_general](stablehlo/topics/dot_general.md), [scatter / gather](stablehlo/topics/scatter-gather.md), [collectives](stablehlo/topics/collectives.md), [convolution](stablehlo/topics/convolution.md), [reduce & windowing](stablehlo/topics/reduce-and-windowing.md), [elementwise & fusion](stablehlo/topics/elementwise-and-fusion.md).
- **A compiler-pass name** seen in a dump stage — canonicalization, folding, simplification, dynamism refinement. → [optimization passes](stablehlo/topics/optimization-passes.md).
- **Precision / quantization semantics** (relevant to the loop's mandatory semantics check — the spec permits backends to differ numerically). → [quantization](stablehlo/topics/quantization.md), [type inference](stablehlo/topics/type-inference.md).
- **Dynamic shapes, versioning/serialization** — why a `dynamic_*` op lingers, VHLO/bytecode compatibility. → [dynamism](stablehlo/topics/dynamism.md), [compatibility & versioning](stablehlo/topics/compatibility-and-versioning.md).

The authoritative op-by-op semantics are in [docs/spec.md](../../raw/code/stablehlo/docs/spec.md); the pass catalogs are [stablehlo_passes.md](../../raw/code/stablehlo/docs/generated/stablehlo_passes.md) and [stablehlo_optimization_passes.md](../../raw/code/stablehlo/docs/generated/stablehlo_optimization_passes.md) — all now summarized and cross-linked under the grounded overview.

## Connections

- Serves as a reference for any HLO-dump observation or `concepts/` page that cites StableHLO op names (fusion, layout, collectives, precision).
- Upstream producers: JAX, PyTorch/XLA, TensorFlow. Downstream consumers: XLA (the main TPU compiler), IREE.

## See also

- [stablehlo — grounded overview](stablehlo/overview.md) — the docs-cited internals wiki (op semantics + pass vocabulary).

## Sources

- [raw/code/stablehlo/docs/spec.md](../../raw/code/stablehlo/docs/spec.md)
- [raw/code/stablehlo/docs/generated/stablehlo_passes.md](../../raw/code/stablehlo/docs/generated/stablehlo_passes.md)
- [raw/code/stablehlo/docs/generated/stablehlo_optimization_passes.md](../../raw/code/stablehlo/docs/generated/stablehlo_optimization_passes.md)
