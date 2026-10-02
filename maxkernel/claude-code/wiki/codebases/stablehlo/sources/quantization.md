---
title: "quantization.md — StableHLO quantization"
type: source
tags: [stablehlo, docs, quantization, precision, reference]
created: 2026-07-04
updated: 2026-07-04
---

How StableHLO represents and lowers **quantized** computation. Under the reference lens: when a dump shows `!quant.uniform<…>` element types or `uniform_quantize`/`uniform_dequantize` ops, this doc decodes the scale/zero-point encoding and the passes that convert quantized ops into integer math on the target.

## Key points

- **[Quantization types](src:docs/quantization.md#quantization-types-in-stablehlo)** — uniform affine scheme inherited from MLIR's Quant dialect, following the LiteRT spec. Real value = `scale * (quantized_value - zero_point)`.
- **[Per-tensor](src:docs/quantization.md#per-tensor-quantization)** — one `scale`/`zero_point` for the whole tensor: `!quant.uniform<i8:f32, 0.01:50>`.
- **[Per-axis](src:docs/quantization.md#per-axis-quantization)** — separate scale/zero-point per slice along `quantized_dimension`: `tensor<4x3x2x!quant.uniform<i8:f32:1, {0.2:20, 0.1:10, 0.3:30}>>`. (Sub-channel quantization is coming.)
- **[Quantization passes](src:docs/quantization.md#quantization-passes-in-stablehlo)** — [`stablehlo-legalize-qdq-to-quantized-op`](src:docs/quantization.md#stablehlo-legalize-qdq-to-quantized-op) fuses dequantize→float-op→quantize into one quantized op; [`stablehlo-legalize-quantized-op-to-qdq`](src:docs/quantization.md#stablehlo-legalize-quantized-op-to-qdq) is the inverse; [`stablehlo-legalize-quant-to-math`](src:docs/quantization.md#stablehlo-legalize-quant-to-math) rewrites to integer arithmetic for targets without native quant support.
- **TOSA rescale bridges** — [`stablehlo-quant-legalize-to-tosa-rescale`](src:docs/quantization.md#stablehlo-quant-legalize-to-tosa-rescale) and [`tosa-rescale-legalize-to-stablehlo`](src:docs/quantization.md#tosa-rescale-legalize-to-stablehlo).
- **[Evaluating quantized programs](src:docs/quantization.md#evaluating-quantized-programs)** / **[test cases](src:docs/quantization.md#quantized-test-cases)** — the interpreter lowers via the full pass sequence (`stablehlo-legalize-quant-to-math` → `chlo-legalize-to-stablehlo` → `canonicalize` → `shape-legalize-to-stablehlo` → `stablehlo-canonicalize-dynamism`).

## Feeds these topics

- [quantization](../topics/quantization.md)
- [optimization-passes](../topics/optimization-passes.md)
- [dot_general](../topics/dot_general.md) — quantized/hybrid dot semantics

## Sources

- `raw/code/stablehlo/docs/quantization.md`
