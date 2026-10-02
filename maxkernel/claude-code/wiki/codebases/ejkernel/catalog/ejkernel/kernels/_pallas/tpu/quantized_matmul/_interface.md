---
title: 'Module: ejkernel/kernels/_pallas/tpu/quantized_matmul/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/quantized_matmul/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.quantized_matmul._interface`/
symbols:
  quantized_matmul: quantized_matmul().
  _operate_impl: _operate_impl().
  _operate_bwd: _operate_bwd().
  _recover_packed_legal_blocks: _recover_packed_legal_blocks().
  _operate: _operate().
  _operate_fwd: _operate_fwd().
  _is_packed_tpu_legal: _is_packed_tpu_legal().
  _should_force_xla_wide_packed_v4: _should_force_xla_wide_packed_v4().
  _normalize_tpu_path: _normalize_tpu_path().
  _biases_to_zeros: _biases_to_zeros().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/quantized_matmul/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_interface.py)

## Functions
- `_biases_to_zeros(scales: jax.Array, biases: jax.Array | None)` — [`L75`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_interface.py#L75) — Convert internal affine additive biases back to canonical affine zeros.
- `_is_packed_tpu_legal(*, is_input_grad: bool, x_or_dy: jax.Array, w_q: jax.Array, scales: jax.Array, group_size: int, bits: int, block_m: int, block_n: int, block_k: int)` — [`L96`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_interface.py#L96) — Strict legality gate for packed TPU Pallas QMM BlockSpecs.
- `_normalize_tpu_path(path: str | None)` — [`L154`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_interface.py#L154) — Normalize TPU path selection to packed-only execution.
- `_operate(x: jax.Array, w: jax.Array, scales: jax.Array, biases: jax.Array | None, transpose: bool, group_size: int, bits: int, mode: str, tpu_path: str, allow_dense_fallback: bool, block_m: int, block_n: int, block_k: int, use_bf16: bool, gemv_mode: GemvMode, revsplit_k: RevSplitKMode, revsplit_k_parts: int | None)` — [`L389`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_interface.py#L389) — Forward pass of the custom-VJP quantized matmul primitive.
- `_operate_bwd(transpose: bool, group_size: int, bits: int, mode: str, tpu_path: str, allow_dense_fallback: bool, block_m: int, block_n: int, block_k: int, use_bf16: bool, gemv_mode: GemvMode, revsplit_k: RevSplitKMode, revsplit_k_parts: int | None, residual: tuple[jax.Array, jax.Array, jax.Array | None], grad_out: jax.Array)` — [`L510`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_interface.py#L510) — Backward rule for the custom VJP.
- `_operate_fwd(x: jax.Array, w: jax.Array, scales: jax.Array, biases: jax.Array | None, transpose: bool, group_size: int, bits: int, mode: str, tpu_path: str, allow_dense_fallback: bool, block_m: int, block_n: int, block_k: int, use_bf16: bool, gemv_mode: GemvMode, revsplit_k: RevSplitKMode, revsplit_k_parts: int | None)` — [`L458`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_interface.py#L458) — Forward rule for the custom VJP.
- `_operate_impl(x: jax.Array, w: jax.Array, scales: jax.Array, biases: jax.Array | None, *, transpose: bool, group_size: int, bits: int, mode: str, tpu_path: str, allow_dense_fallback: bool, block_m: int, block_n: int, block_k: int, use_bf16: bool, gemv_mode: GemvMode, revsplit_k: RevSplitKMode, revsplit_k_parts: int | None)` — [`L237`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_interface.py#L237) — Core forward implementation for TPU Pallas quantized matmul.
- `_recover_packed_legal_blocks(x: jax.Array, w_q: jax.Array, scales: jax.Array, *, group_size: int, bits: int, block_m: int, block_n: int, block_k: int)` — [`L176`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_interface.py#L176) — Try to find packed-legal block sizes when the caller's choice is illegal.
- `_should_force_xla_wide_packed_v4(*, bits: int, block_n: int)` — [`L160`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_interface.py#L160) — Return True when TPU v4 packed fused tiles are known to exceed VMEM.
- `quantized_matmul(x: Float[Array, "m k"], w: Array, scales: Array, zeros: Array | None = None, transpose: bool = False, group_size: int | None = None, bits: int | None = None, mode: str = "affine", axis: QuantizationAxis | None = None, gemv_mode: GemvMode = "auto", revsplit_k: RevSplitKMode = "auto", revsplit_k_parts: int | None = None, *, tpu_path: str | None = None, allow_dense_fallback: bool = True, block_m: int = 128, block_n: int = 128, block_k: int = 64, use_bf16: bool = True, num_warps: int | None = None, num_stages: int | None = None, split_k: int | None = None)` — [`L598`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_interface.py#L598) — Quantized matmul on TPU via Pallas with custom backward support.

## Module values
- `__all__` — [`L734`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_interface.py#L734)

