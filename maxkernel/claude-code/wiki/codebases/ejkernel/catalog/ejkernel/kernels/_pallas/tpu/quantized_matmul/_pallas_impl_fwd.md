---
title: 'Module: ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.quantized_matmul._pallas_impl_fwd`/_
symbols:
  _pallas_qmm_transpose_false_packed: pallas_qmm_transpose_false_packed().
  _pallas_qmm_transpose_false_packed._kernel_no_bias: pallas_qmm_transpose_false_packed()._kernel_no_bias().
  _pallas_qmm_transpose_false_packed._kernel_with_bias: pallas_qmm_transpose_false_packed()._kernel_with_bias().
  _pallas_qmm_transpose_false: pallas_qmm_transpose_false().
  _PACKED_SUPPORTED_MODES: PACKED_SUPPORTED_MODES.
  _pallas_qmm_transpose_false_packed._kernel_no_bias._zero_acc: pallas_qmm_transpose_false_packed()._kernel_no_bias()._zero_acc().
  _pallas_qmm_transpose_false_packed._kernel_no_bias._store: pallas_qmm_transpose_false_packed()._kernel_no_bias()._store().
  _pallas_qmm_transpose_false_packed._kernel_with_bias._zero_acc: pallas_qmm_transpose_false_packed()._kernel_with_bias()._zero_acc().
  _pallas_qmm_transpose_false_packed._kernel_with_bias._store: pallas_qmm_transpose_false_packed()._kernel_with_bias()._store().
  __all__: _all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_fwd.py)

## Functions
- `_kernel_no_bias(x_ref, w_ref, s_ref, out_ref, acc_ref)` — [`L146`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_fwd.py#L146)
- `_kernel_with_bias(x_ref, w_ref, s_ref, b_ref, out_ref, acc_ref)` — [`L173`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_fwd.py#L173)
- `_pallas_qmm_transpose_false(x: jax.Array, w_q: jax.Array, scales: jax.Array, biases: jax.Array | None, *, group_size: int, bits: int, mode: str, block_m: int, block_n: int, block_k: int, use_bf16: bool, path: str, packed_legal: bool)` — [`L269`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_fwd.py#L269) — Dispatch forward TPU QMM for packed-only execution.
- `_pallas_qmm_transpose_false_packed(x: jax.Array, w_q: jax.Array, scales: jax.Array, biases: jax.Array | None, *, group_size: int, bits: int, mode: str, block_m: int, block_n: int, block_k: int, use_bf16: bool)` — [`L59`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_fwd.py#L59) — Packed fused TPU Pallas path for forward Y = X @ dequant(W).
- `_store()` — [`L170`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_fwd.py#L170)
- `_store()` — [`L203`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_fwd.py#L203)
- `_zero_acc()` — [`L150`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_fwd.py#L150)
- `_zero_acc()` — [`L177`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_fwd.py#L177)

## Module values
- `_PACKED_SUPPORTED_MODES` — [`L56`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_fwd.py#L56)
- `__all__` — [`L329`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_fwd.py#L329)

