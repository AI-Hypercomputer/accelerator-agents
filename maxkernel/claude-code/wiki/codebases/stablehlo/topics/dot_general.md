---
title: "dot_general — matmul / contraction semantics"
type: concept
tags: [stablehlo, dot_general, matmul, contraction, precision, reference]
created: 2026-07-04
updated: 2026-07-04
---

`dot_general` is StableHLO's **general contraction op** — the single primitive behind every matmul in a transformer: attention QKᵀ / softmax·V, and every FFN/projection GEMM. When you read an `xla_dump` HLO or an xprof op profile, `dot`/`dot_general` (and the fused kernels built around it) is almost always the dominant FLOP contributor. This page decodes its attributes so you can read *what a given dot is contracting and at what precision*.

## What it computes

Computes dot products between slices of `lhs` and `rhs` ([dot_general](src:docs/spec.md#dot_general)). The op is parameterized by four dimension-number lists rather than assuming 2-D matmul:

| Attribute | Role |
|---|---|
| `lhs_batching_dimensions`, `rhs_batching_dimensions` | Dims iterated in lockstep, not contracted (e.g. the batch and head axes of attention). |
| `lhs_contracting_dimensions`, `rhs_contracting_dimensions` | Dims summed over (the shared `K` of an `M×K · K×N`). |

The remaining dims are "result" dims. Output shape = `batching_dims ++ lhs_result_dims ++ rhs_result_dims` (constraint C12). Constraints require batching sizes to match (C9) and contracting sizes to match (C10) — [dot_general](src:docs/spec.md#dot_general).

> [!inferred]
> Reading a dump: the `dimension_numbers={...}` attribute on an HLO `dot` tells you directly which axes are batch vs contracting. A batch dim that *should* be contracting (or vice-versa) is a common source of an unexpectedly shaped/large dot. For attention, expect the head dim in batching and the head_dim (feature) in contracting.

## Precision — the perf/accuracy knob

Two mechanisms control the speed/accuracy tradeoff, both visible on the op:

1. **`precision_config`** — a per-operand enum `DEFAULT` / `HIGH` / `HIGHEST` (fastest→slowest, least→most accurate). Semantics are underspecified in the spec ([dot_general](src:docs/spec.md#dot_general)).
2. **`DotAlgorithm`** — a fully explicit description: `lhs_precision_type` / `rhs_precision_type` (what LHS/RHS are rounded to, independent of storage type), `accumulation_type`, and the component-decomposition fields (`lhs_component_count`, `rhs_component_count`, `num_primitive_operations`) used to emulate higher precision by splitting inputs into multiple components — e.g. `bf16_6x` (3 bf16 components each, 6 primitive dots, f32 accumulation) or `tf32_3x`. `allow_imprecise_accumulation` permits lower-precision accumulation on some steps. If a `DotAlgorithm` is set, `precision_config` must be `DEFAULT` ([dot_general](src:docs/spec.md#dot_general)).

> [!inferred]
> This is the vocabulary behind TPU precision experiments. When a run's matmuls are configured `bf16` inputs / `f32` accumulate, that is a `DotAlgorithm`; the `_Nx` component counts are exactly the "bf16 3-pass / 6-pass" emulation tricks. If a dump shows `tf32` precision types, that's the `precision_config=HIGH`-style path. An algorithm not supported on the target raises an error rather than silently falling back.

## Quantized dot

For quantized/hybrid types, `dot_general` is defined as `dequantize_op_quantize` (resp. `hybrid_dequantize_then_op`) wrapping the float dot ([dot_general](src:docs/spec.md#dot_general)). Constraints pin `zero_points(rhs)=0` (C15) and forbid the RHS per-axis quantization dim from being a contracting dim (C16). See [quantization](quantization.md).

## Related ops that lower to / from dot

- **[convolution](convolution.md)** is *defined in terms of* `dot_general` — each output spatial window is a `dot_general` of the windowed LHS against the kernel ([convolution](src:docs/spec.md#convolution)).
- The TOSA-prep pass simplifies `dot_general`→`dot` for easier lowering ([generated-stablehlo_tosa_passes](../sources/generated-stablehlo_tosa_passes.md)).

## See also

- [convolution](convolution.md), [quantization](quantization.md), [collectives](collectives.md) (sharded matmuls emit collectives around the dot)
- Source: [spec.md](../sources/spec.md)

## Sources

- `raw/code/stablehlo/docs/spec.md`
