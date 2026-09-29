---
title: "reduce / reduce_window — reductions and pooling"
type: concept
tags: [stablehlo, reduce, reduce_window, softmax, normalization, pooling, reference]
created: 2026-07-04
updated: 2026-07-04
---

`reduce` and `reduce_window` are StableHLO's **reduction** primitives. `reduce` is behind every softmax normalization, LayerNorm/RMSNorm mean/variance, and loss reduction; `reduce_window` is behind pooling and windowed statistics. In a dump these show up as `reduce`/`reduce-window` fused into the surrounding elementwise math — a norm or softmax is a `reduce` sandwiched between broadcasts and elementwise ops.

## reduce — full-axis reduction

Applies a reduction `body` to `inputs`/`init_values` along `dimensions` ([reduce](src:docs/spec.md#reduce)). Key points:

- The `body` + `init_values` must form a **monoid** for determinism; the spec notes this fails for FP addition (non-associative), so reduction order is implementation-defined and FP-reductions are not bit-reproducible across impls.
- Variadic: multiple inputs reduced together (e.g. simultaneous max+argmax, or mean via sum+count).
- `dimensions` names the reduced axes; result drops those axes (C7).

> [!inferred]
> A softmax's denominator is a `reduce(add)` over the last axis; the max-subtraction trick is a `reduce(max)` first. RMSNorm is `reduce(add)` of squares. Seeing two reduces feeding a `broadcast_in_dim` then elementwise `divide`/`multiply` is the canonical normalization shape in a trace. Because FP reduce order is impl-defined, small numeric drift between backends here is expected, not a bug.

## reduce_window — windowed reduction

`results[result_index] = reduce(windows, init_values, …, body)` over sliding windows of the (padded, dilated) input ([reduce_window](src:docs/spec.md#reduce_window)). It shares the window-geometry vocabulary with [convolution](convolution.md): `window_dimensions`, `window_strides`, `base_dilations`, `window_dilations`, `padding`. Max-pool = `reduce_window(max)`; avg-pool = `reduce_window(add)` + divide. Its gradient counterpart is [select_and_scatter](src:docs/spec.md#select_and_scatter).

## Fusion note

The aggressive-simplification pass folds degenerate reductions: `reduce(X…, dims=[], add) -> X…`, unused-input pruning, and empty-reduce → broadcast ([generated-stablehlo_optimization_passes](../sources/generated-stablehlo_optimization_passes.md)). See [elementwise-and-fusion](elementwise-and-fusion.md).

## See also

- [convolution](convolution.md) (shared window geometry), [elementwise-and-fusion](elementwise-and-fusion.md), [dot_general](dot_general.md)
- Source: [spec.md](../sources/spec.md)

## Sources

- `raw/code/stablehlo/docs/spec.md`
