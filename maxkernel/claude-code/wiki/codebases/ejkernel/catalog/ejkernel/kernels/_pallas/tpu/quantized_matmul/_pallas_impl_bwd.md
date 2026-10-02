---
title: 'Module: ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_bwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_bwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.quantized_matmul._pallas_impl_bwd`/
symbols:
  _pallas_qmm_input_grad_transpose_false_packed: _pallas_qmm_input_grad_transpose_false_packed().
  quantized_matmul_input_grad: quantized_matmul_input_grad().
  _pallas_qmm_input_grad_transpose_false_packed._kernel_no_bias: _pallas_qmm_input_grad_transpose_false_packed()._kernel_no_bias().
  _pallas_qmm_input_grad_transpose_false_packed._kernel_with_bias: _pallas_qmm_input_grad_transpose_false_packed()._kernel_with_bias().
  _quantized_matmul_input_grad_packed: _quantized_matmul_input_grad_packed().
  _PACKED_SUPPORTED_MODES: _PACKED_SUPPORTED_MODES.
  _pallas_qmm_input_grad_transpose_false_packed._kernel_no_bias._zero_acc: _pallas_qmm_input_grad_transpose_false_packed()._kernel_no_bias()._zero_acc().
  _pallas_qmm_input_grad_transpose_false_packed._kernel_no_bias._store: _pallas_qmm_input_grad_transpose_false_packed()._kernel_no_bias()._store().
  _pallas_qmm_input_grad_transpose_false_packed._kernel_with_bias._zero_acc: _pallas_qmm_input_grad_transpose_false_packed()._kernel_with_bias()._zero_acc().
  _pallas_qmm_input_grad_transpose_false_packed._kernel_with_bias._store: _pallas_qmm_input_grad_transpose_false_packed()._kernel_with_bias()._store().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_bwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_bwd.py)

## Functions
- `_kernel_no_bias(dy_ref, w_ref, s_ref, out_ref, acc_ref)` — [`L163`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_bwd.py#L163)
- `_kernel_with_bias(dy_ref, w_ref, s_ref, b_ref, out_ref, acc_ref)` — [`L190`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_bwd.py#L190)
- `_pallas_qmm_input_grad_transpose_false_packed(dy: jax.Array, w_q: jax.Array, scales: jax.Array, biases: jax.Array | None, *, group_size: int, bits: int, mode: str, block_m: int, block_n: int, block_k: int, use_bf16: bool)` — [`L72`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_bwd.py#L72) — Packed fused TPU Pallas path for dX when forward transpose=False.
- `_quantized_matmul_input_grad_packed(dy: jax.Array, w_q: jax.Array, scales: jax.Array, biases: jax.Array | None, *, group_size: int, bits: int, mode: str, block_m: int, block_n: int, block_k: int, use_bf16: bool, path: str, packed_legal: bool)` — [`L288`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_bwd.py#L288) — Dispatch dX computation for packed-only execution.
- `_store()` — [`L187`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_bwd.py#L187)
- `_store()` — [`L220`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_bwd.py#L220)
- `_zero_acc()` — [`L167`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_bwd.py#L167)
- `_zero_acc()` — [`L194`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_bwd.py#L194)
- `quantized_matmul_input_grad(dy: jax.Array, w_q: jax.Array, scales: jax.Array, biases: jax.Array | None, *, transpose: bool, group_size: int, bits: int, mode: str, block_m: int, block_n: int, block_k: int, use_bf16: bool, allow_dense_fallback: bool, path: str, packed_legal: bool)` — [`L360`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_bwd.py#L360) — Gradient w.r.t. input for TPU Pallas quantized matmul.

## Module values
- `_PACKED_SUPPORTED_MODES` — [`L69`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_bwd.py#L69)
- `__all__` — [`L470`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_bwd.py#L470)

