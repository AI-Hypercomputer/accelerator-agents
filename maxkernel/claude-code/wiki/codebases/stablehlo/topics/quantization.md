---
title: "quantization — quantized types and lowering"
type: concept
tags: [stablehlo, quantization, precision, int8, fp8, reference]
created: 2026-07-04
updated: 2026-07-04
---

**Quantization** in StableHLO — how low-precision integer (and the pass machinery around it) is represented and lowered. Under the reference lens: when a dump shows `!quant.uniform<…>` element types or `uniform_quantize` / `uniform_dequantize` ops, this page decodes the scale/zero-point encoding and tells you which pass converts the quantized op into the integer or float math the target actually runs.

## The type: uniform affine quantization

StableHLO uses a uniform affine scheme (LiteRT spec, MLIR Quant dialect): `real_value = scale * (quantized_value - zero_point)` ([quantization-types-in-stablehlo](src:docs/quantization.md#quantization-types-in-stablehlo)).

- **[Per-tensor](src:docs/quantization.md#per-tensor-quantization)**: one `scale:zero_point` for the whole tensor — `!quant.uniform<i8:f32, 0.01:50>`.
- **[Per-axis](src:docs/quantization.md#per-axis-quantization)**: a `scale:zero_point` per slice along `quantized_dimension` — `tensor<4x3x2x!quant.uniform<i8:f32:1, {0.2:20, 0.1:10, 0.3:30}>>`. Used for per-channel weight quantization. (Sub-channel quantization is in development.)

The spec's [types](../sources/spec.md) section carries the grammar and constraints (C1–C13) for these quantized element types.

## The lowering passes

Three passes move between representations ([quantization-passes-in-stablehlo](src:docs/quantization.md#quantization-passes-in-stablehlo)):

| Pass | Effect |
|---|---|
| [`stablehlo-legalize-qdq-to-quantized-op`](src:docs/quantization.md#stablehlo-legalize-qdq-to-quantized-op) | **Fuse** dequantize → float-op → quantize into one quantized op (does not delete the originals). |
| [`stablehlo-legalize-quantized-op-to-qdq`](src:docs/quantization.md#stablehlo-legalize-quantized-op-to-qdq) | **Decompose** a quantized op back into the dequant/float/quant triple. |
| [`stablehlo-legalize-quant-to-math`](src:docs/quantization.md#stablehlo-legalize-quant-to-math) | Rewrite quantized ops to **integer arithmetic** (convert→multiply-by-scale→round→clamp→convert) for targets without native quant. Introduces CHLO broadcast ops. |

TOSA rescale bridges ([stablehlo-quant-legalize-to-tosa-rescale](src:docs/quantization.md#stablehlo-quant-legalize-to-tosa-rescale), [tosa-rescale-legalize-to-stablehlo](src:docs/quantization.md#tosa-rescale-legalize-to-stablehlo)) handle the `tosa.rescale` path. The [interpreter](../sources/reference.md) evaluates quantized programs by lowering to integer math through a fixed pass sequence ([evaluating-quantized-programs](src:docs/quantization.md#evaluating-quantized-programs)).

## Quantized dot / conv

The compute ops define quantized behavior as `dequantize_op_quantize` wrapping the float op — see [dot_general](dot_general.md) (constraints require `zero_points(rhs)=0`, RHS quant-dim not a contracting dim) and [convolution](convolution.md).

> [!inferred]
> Reading a dump: `uniform_quantize`/`uniform_dequantize` bracketing a `dot` is the QDQ pattern *before* `legalize-qdq-to-quantized-op` fuses it. If you see integer `convert`/`multiply`/`round_nearest_even`/`clamp` chains where a quantized op used to be, `legalize-quant-to-math` already ran. StableHLO makes **no numerical-accuracy guarantees** ([compatibility](compatibility-and-versioning.md)), so quantized-vs-float parity is a semantics check the loop must do itself.

## See also

- [dot_general](dot_general.md), [convolution](convolution.md), [optimization-passes](optimization-passes.md)
- Sources: [quantization.md](../sources/quantization.md), [spec.md](../sources/spec.md), [generated-stablehlo_passes.md](../sources/generated-stablehlo_passes.md)

## Sources

- `raw/code/stablehlo/docs/quantization.md`
- `raw/code/stablehlo/docs/spec.md`
