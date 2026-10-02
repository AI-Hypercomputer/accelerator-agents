---
title: "spec.md — StableHLO Specification"
type: source
tags: [stablehlo, docs, spec, opset, reference]
created: 2026-07-04
updated: 2026-07-04
---

The **op-set specification** — the single source of truth for what every StableHLO op *means*. Under the reference lens this is the decoder ring: when an op name appears in an `xla_dump` HLO or an xprof trace, this doc tells you its semantics and its perf-relevant attributes (contraction dims, replica groups, precision configs, dynamism, quantization). StableHLO is a portability opset between ML frameworks (JAX, PyTorch, TF) and compilers (XLA, IREE); XLA's HLO is its downstream twin, so the op names and attribute vocabulary transfer almost verbatim into what you read in dumps.

## What it covers

The spec is organized in three layers ([stablehlo-specification](src:docs/spec.md#stablehlo-specification)):

- **[Programs](src:docs/spec.md#programs)** — structure: modules → functions → ops. Values are tensors, quantized tensors, tokens, tuples (legacy HLO-ABI), buffers, futures. Element types include the full float zoo relevant to TPU precision work: `bf16`, `f16`, `f32`, `tf32`, and the FP8/MX microscaling formats (`f8E4M3FN`, `f8E5M2`, `f4E2M1FN`, …) — see [types](src:docs/spec.md#types).
- **[Ops](src:docs/spec.md#ops)** — one section per op with Semantics / Inputs / Outputs / Constraints. This is where the perf-relevant attributes live.
- **[Execution](src:docs/spec.md#execution)** — how ops run together, the process grid, and the [collective-ops](src:docs/spec.md#collective-ops) sharding-communication model ([replica-groups](src:docs/spec.md#replica-groups), [cross_replica](src:docs/spec.md#cross_replica), [cross_partition](src:docs/spec.md#cross_partition), [flattened_ids](src:docs/spec.md#flattened_ids)).

## Perf-relevant op families (feed the topic pages)

| Op family | Spec anchor | Why it matters when reading dumps |
|---|---|---|
| Matmul / contraction | [dot_general](src:docs/spec.md#dot_general) | The workhorse behind every attention & FFN matmul; carries `precision_config` + `DotAlgorithm`. |
| Collectives | [all_gather](src:docs/spec.md#all_gather), [all_reduce](src:docs/spec.md#all_reduce), [reduce_scatter](src:docs/spec.md#reduce_scatter), [all_to_all](src:docs/spec.md#all_to_all), [collective_permute](src:docs/spec.md#collective_permute), [collective_broadcast](src:docs/spec.md#collective_broadcast) | Sharding communication; replica/partition groups set the cost. |
| Dynamic indexing | [gather](src:docs/spec.md#gather), [scatter](src:docs/spec.md#scatter), [dynamic_slice](src:docs/spec.md#dynamic_slice) | Embedding lookups, MoE dispatch/combine. |
| Windowed / reductions | [reduce](src:docs/spec.md#reduce), [reduce_window](src:docs/spec.md#reduce_window), [convolution](src:docs/spec.md#convolution), [select_and_scatter](src:docs/spec.md#select_and_scatter) | Norms, softmax reductions, pooling, conv. |
| Scheduling / barrier | [optimization_barrier](src:docs/spec.md#optimization_barrier) | Blocks compiler code motion; shows up in rematerialization control. |

## Feeds these topics

- [dot_general](../topics/dot_general.md)
- [collectives](../topics/collectives.md)
- [scatter-gather](../topics/scatter-gather.md)
- [convolution](../topics/convolution.md)
- [reduce-and-windowing](../topics/reduce-and-windowing.md)
- [elementwise-and-fusion](../topics/elementwise-and-fusion.md)
- [quantization](../topics/quantization.md)
- [dynamism](../topics/dynamism.md)

## Gaps & caveats

- Numerical accuracy is **not** specified — `precision_config` enum values are underspecified and accuracy is implementation-defined across consumers/versions.
- Some attributes seen in the wild (`layout`, `mhlo.sharding`, `mhlo.frontend_attributes`) are not yet part of the spec.

## Sources

- `raw/code/stablehlo/docs/spec.md`
