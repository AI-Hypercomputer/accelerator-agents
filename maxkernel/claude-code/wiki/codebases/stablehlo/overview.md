---
title: "stablehlo — overview"
type: overview
tags: [stablehlo, docs, reference, hlo, xla, overview]
created: 2026-07-04
updated: 2026-07-04
---

**StableHLO** is the OpenXLA operation-set that sits between ML frameworks (JAX, PyTorch/XLA, TF) and compilers (XLA, IREE). For this wiki it is **not an optimization target** — it is a **reference for interpreting HLO op names, op semantics, and compiler-pass vocabulary** when reading `xla_dump` HLO modules and xprof traces. XLA's HLO is StableHLO's downstream twin, so the op mnemonics (`dot`, `all-reduce`, `gather`, `reduce-window`, `convolution`) and attribute names (`dimension_numbers`, `replica_groups`, `precision_config`) you see in a TPU dump map almost verbatim onto the spec here. Every topic page answers: *when I see this op / this pass, what does it mean and what are its perf-relevant attributes?*

## How the docs relate

Two doc families do the work, plus supporting material:

- **The op-set spec** ([spec.md](sources/spec.md)) — one section per op: Semantics / Inputs / Outputs / Constraints. This is the semantic ground truth; the [reference interpreter](sources/reference.md) is its executable form and [type_inference](sources/type_inference.md) is what enforces the constraint tables.
- **The generated pass docs** ([stablehlo_optimization_passes](sources/generated-stablehlo_optimization_passes.md), [stablehlo_passes](sources/generated-stablehlo_passes.md), plus [tosa](sources/generated-stablehlo_tosa_passes.md) / [linalg](sources/generated-stablehlo_linalg_passes.md) / [interpreter](sources/generated-interpreter_passes.md)) — the `-stablehlo-…` pass names and rewrite patterns you see in compiler logs. These transform the IR between what a framework emits and what reaches the TPU backend.
- **Cross-cutting concerns** — [quantization](sources/quantization.md), [dynamism](sources/dynamism.md), and compatibility ([compatibility](sources/compatibility.md) / [vhlo](sources/vhlo.md) / [bytecode](sources/bytecode.md)) each get a doc and a topic.
- **CHLO** ([chlo](sources/generated-chlo.md)) — higher-level client ops that legalize *down* to StableHLO.

## Which topic answers which question

| When you see… in a dump / trace / log | Read this topic |
|---|---|
| `dot` / `dot_general`, `precision_config`, `DotAlgorithm`, bf16/tf32 emulation | [dot_general](topics/dot_general.md) |
| `all-reduce`, `all-gather`, `reduce-scatter`, `all-to-all`, `collective-permute`, `replica_groups`, `channel_id` | [collectives](topics/collectives.md) |
| `gather`, `scatter`, `dynamic-slice`, embedding lookups, MoE dispatch/combine | [scatter / gather](topics/scatter-gather.md) |
| `convolution`, `feature_group_count`, depthwise/grouped conv | [convolution](topics/convolution.md) |
| `reduce`, `reduce-window`, softmax/norm reductions, pooling | [reduce / reduce_window](topics/reduce-and-windowing.md) |
| chains of `add`/`multiply`/`broadcast`/`transpose`/`reshape`, `optimization-barrier`, fusion questions | [elementwise & fusion](topics/elementwise-and-fusion.md) |
| `-stablehlo-…` pass names, canonicalization/folding/legalization | [optimization passes](topics/optimization-passes.md) |
| `!quant.uniform<…>`, `uniform_quantize`/`dequantize`, int8/fp8 | [quantization](topics/quantization.md) |
| `tensor<?x…>`, `dynamic_broadcast_in_dim`, `get_dimension_size`, shape refinement | [dynamism](topics/dynamism.md) |
| why the spec's shape/dtype constraints are trustworthy | [type inference](topics/type-inference.md) |
| `#.#.#` versions, `.mlir.bc` artifacts, `vhlo.*` ops, op downgrade/expansion | [compatibility & versioning](topics/compatibility-and-versioning.md) |

## Topic map (how they connect)

- **[dot_general](topics/dot_general.md)** is the hub — [convolution](topics/convolution.md) is defined on top of it, [quantization](topics/quantization.md) wraps it, and sharded dots emit [collectives](topics/collectives.md).
- **[collectives](topics/collectives.md)** share the process-group model; [all_to_all](topics/collectives.md) underlies MoE, distinct from the local [scatter/gather](topics/scatter-gather.md).
- **[reduce/windowing](topics/reduce-and-windowing.md)** shares window geometry with [convolution](topics/convolution.md) and fuses with [elementwise](topics/elementwise-and-fusion.md) ops.
- **[optimization passes](topics/optimization-passes.md)** is the verb layer: it applies the [elementwise/fusion](topics/elementwise-and-fusion.md) rewrites, drives [dynamism](topics/dynamism.md) refinement, and runs the [quantization](topics/quantization.md) and [compatibility](topics/compatibility-and-versioning.md) legalizations.

## Scope caveat

StableHLO makes **no numerical-accuracy guarantees** across consumers or versions ([compatibility](topics/compatibility-and-versioning.md), [quantization](topics/quantization.md)). Under this project's rules that means any optimization touching precision/quantization must be validated by the loop's own semantics check — the spec permits backends to differ numerically.

## See also

- Hand-curated perf-lens companion: [stablehlo.md](../stablehlo.md) (*when* and *why* to reach for StableHLO from the optimization loop — the complement to this grounded structural layer)

## Sources

- `raw/code/stablehlo/docs/` (27 doc packets; see `sources/`)
