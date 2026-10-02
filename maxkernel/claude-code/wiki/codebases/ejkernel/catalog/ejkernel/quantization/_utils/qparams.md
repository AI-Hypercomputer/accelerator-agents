---
title: 'Module: ejkernel/quantization/_utils/qparams.py'
type: catalog
provenance: extracted
module: ejkernel/quantization/_utils/qparams.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.quantization._utils.qparams`/
symbols:
  QuantizationAxis: QuantizationAxis.
  QuantizationMode: QuantizationMode.
  resolve_qparams: resolve_qparams().
  select_qmm_kernel_family: select_qmm_kernel_family().
  GemvMode: GemvMode.
  RevSplitKMode: RevSplitKMode.
  validate_packed_quantized_matmul_layout: validate_packed_quantized_matmul_layout().
  resolve_runtime_axis_and_transpose: resolve_runtime_axis_and_transpose().
  normalize_axis: normalize_axis().
  normalize_gemv_mode: normalize_gemv_mode().
  normalize_revsplitk_mode: normalize_revsplitk_mode().
  to_backend_mode: to_backend_mode().
  resolve_prepack_axis: resolve_prepack_axis().
  normalize_revsplitk_parts: normalize_revsplitk_parts().
  BackendQuantizationMode: BackendQuantizationMode.
  normalize_mode_and_bits: normalize_mode_and_bits().
  is_effective_4bit_mode: is_effective_4bit_mode().
  AFFINE_NF4_GROUP_SIZES: AFFINE_NF4_GROUP_SIZES.
  _ceil_div: _ceil_div().
  KernelFamily: KernelFamily.
---
# Module: [`ejkernel/quantization/_utils/qparams.py`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py)

## Functions
- `_ceil_div(a: int, b: int)` — [`L388`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L388)
- `is_effective_4bit_mode(mode: QuantizationMode, bits: int)` — [`L236`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L236) — Check whether the effective runtime quantization is 4-bit. — documented in [ejkernel-quantization-_utils-qparams](../../../../concepts/ejkernel-quantization-_utils-qparams.md)
- `normalize_axis(axis: str | None, *, default: QuantizationAxis = "row")` — [`L36`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L36) — Normalize and validate the quantization axis name. — documented in [ejkernel-quantization-_utils-qparams](../../../../concepts/ejkernel-quantization-_utils-qparams.md)
- `normalize_gemv_mode(mode: str | None)` — [`L173`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L173) — Normalize and validate the GEMV dispatch override mode. — documented in [ejkernel-quantization-_utils-qparams](../../../../concepts/ejkernel-quantization-_utils-qparams.md)
- `normalize_mode_and_bits(mode: str, bits: int | None)` — [`L60`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L60) — Normalize and validate explicit quantization mode and optional bit-width.
- `normalize_revsplitk_mode(mode: str | None)` — [`L193`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L193) — Normalize and validate the reverse split-K dispatch override mode.
- `normalize_revsplitk_parts(parts: int | None)` — [`L213`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L213) — Normalize and validate the optional reverse split-K partition count.
- `resolve_prepack_axis(*, axis: str | None, transpose: bool)` — [`L365`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L365) — Resolve the quantization axis for the prepack API.
- `resolve_qparams(mode: str, group_size: int | None, bits: int | None)` — [`L94`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L94) — Resolve and validate the full set of quantization parameters. — documented in [ejkernel-quantization-_utils-qparams](../../../../concepts/ejkernel-quantization-_utils-qparams.md)
- `resolve_runtime_axis_and_transpose(*, axis: str | None, transpose: bool)` — [`L331`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L331) — Resolve runtime quantization axis and ensure transpose consistency.
- `select_qmm_kernel_family(*, m: int, mode: QuantizationMode, bits: int, gemv_mode: GemvMode, revsplit_k: RevSplitKMode, revsplit_k_parts: int | None)` — [`L255`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L255) — Select the quantized matmul kernel family using a GemLite-style policy. — documented in [ejkernel-quantization-_utils-qparams](../../../../concepts/ejkernel-quantization-_utils-qparams.md)
- `to_backend_mode(mode: QuantizationMode, bits: int)` — [`L155`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L155) — Map a user-facing quantization mode to the backend mode key.
- `validate_packed_quantized_matmul_layout(x, w_q, scales, zeros, *, mode: QuantizationMode, group_size: int, bits: int, axis: QuantizationAxis, transpose: bool)` — [`L394`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L394) — Validate packed QMM weight/metadata layout against the canonical contract.

## Module values
- `AFFINE_NF4_GROUP_SIZES` — [`L33`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L33) — documented in [ejkernel-quantization-_utils-qparams](../../../../concepts/ejkernel-quantization-_utils-qparams.md)
- `BackendQuantizationMode` — [`L29`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L29) — documented in [ejkernel-quantization-_utils-qparams](../../../../concepts/ejkernel-quantization-_utils-qparams.md)
- `GemvMode` — [`L30`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L30) — documented in [ejkernel-quantization-_utils-qparams](../../../../concepts/ejkernel-quantization-_utils-qparams.md)
- `KernelFamily` — [`L32`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L32) — documented in [ejkernel-quantization-_utils-qparams](../../../../concepts/ejkernel-quantization-_utils-qparams.md)
- `QuantizationAxis` — [`L28`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L28) — documented in [ejkernel-quantization-_utils-qparams](../../../../concepts/ejkernel-quantization-_utils-qparams.md)
- `QuantizationMode` — [`L27`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L27) — documented in [ejkernel-quantization-_utils-qparams](../../../../concepts/ejkernel-quantization-_utils-qparams.md)
- `RevSplitKMode` — [`L31`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/qparams.py#L31) — documented in [ejkernel-quantization-_utils-qparams](../../../../concepts/ejkernel-quantization-_utils-qparams.md)

