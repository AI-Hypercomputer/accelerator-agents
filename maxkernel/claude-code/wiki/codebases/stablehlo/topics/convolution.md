---
title: "convolution — windowed contraction"
type: concept
tags: [stablehlo, convolution, conv, dot_general, reference]
created: 2026-07-04
updated: 2026-07-04
---

`convolution` is StableHLO's **windowed contraction** op — spatial conv, grouped/depthwise conv, and (via 1×1) another face of matmul. Less central than `dot_general` for transformer workloads, but it appears in vision stems, conv-augmented models, and anywhere a windowed dot is emitted. This page maps its attributes so a `conv` in a dump is legible.

## What it computes

For each output spatial index, `result = dot_general(reversed_lhs_window, rhs, …)` — i.e. convolution is **defined in terms of [dot_general](dot_general.md)**: it slices a window of the (padded, dilated) `lhs` and contracts it against the kernel `rhs` ([convolution](src:docs/spec.md#convolution)). Everything `dot_general` knows about precision applies here too (it carries the same `precision_config`).

## The attribute groups

- **Window geometry**: `window_strides`, `padding`, `lhs_dilation` (input/transposed-conv dilation), `rhs_dilation` (atrous/dilated kernel), `window_reversal` (unused, slated for removal).
- **Dimension layout**: `input_batch_dimension` / `input_feature_dimension` / `input_spatial_dimensions`, the kernel's `kernel_input_feature_dimension` / `kernel_output_feature_dimension` / `kernel_spatial_dimensions`, and the matching `output_*` dims. These encode the data layout (NCHW vs NHWC etc.) directly.
- **Grouping**: `feature_group_count` (grouped / depthwise conv — splits input feature & kernel output feature into groups, C11/C14) and `batch_group_count` (splits the batch dim, used for some backward-conv formulations) ([convolution](src:docs/spec.md#convolution)).

For quantized/hybrid types, convolution is `dequantize_op_quantize` around the float conv, like `dot_general`.

> [!inferred]
> Reading a dump: `feature_group_count == input_feature_dim` is a depthwise conv; `feature_group_count == 1` is dense. The `dimension_numbers` on an HLO `convolution` spell out the layout — a mismatched layout (forcing a transpose before/after) is a common conv perf tax. Because conv reduces to `dot_general`, the same precision/accumulation reasoning transfers, and 1×1 convs often canonicalize toward plain dots.

## See also

- [dot_general](dot_general.md) — convolution's underlying contraction
- [reduce-and-windowing](reduce-and-windowing.md) — `reduce_window` shares the window-geometry vocabulary (strides/padding/dilation)
- Source: [spec.md](../sources/spec.md)

## Sources

- `raw/code/stablehlo/docs/spec.md`
