---
title: 'Module: tpu_inference/kernels/quantized_matmul/util.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/quantized_matmul/util.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.quantized_matmul.util`/
symbols:
  get_kernel_name: get_kernel_name().
  quantize_tensor: quantize_tensor().
  unfold_args: unfold_args().
  xla_quantized_batched_matmul: xla_quantized_batched_matmul().
  quantize_block: quantize_block().
  get_max_min: get_max_min().
  next_multiple: next_multiple().
  quantize_array: quantize_array().
  get_vmem_limit: get_vmem_limit().
  validate_inputs: validate_inputs().
---
# Module: [`tpu_inference/kernels/quantized_matmul/util.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/util.py)

## Functions
- `get_kernel_name(tuned_value: TunedValue)` — [`L60`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/util.py#L60)
- `get_max_min(target_dtype)` — [`L245`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/util.py#L245)
- `get_vmem_limit(n_batch: int, n_out: int, n_in: int, batch_block_size: int, out_block_size: int, in_block_size: int, x_dtype: jnp.dtype, x_q_dtype: jnp.dtype, w_q_dtype: jnp.dtype, scale_dtype: jnp.dtype, out_dtype: jnp.dtype, acc_dtype: jnp.dtype, save_acc: bool, save_x_q: bool, upper_limit_bytes: int)` — [`L154`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/util.py#L154) — Calculate VMEM limit for the kernel.
- `next_multiple(x, multiple)` — [`L56`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/util.py#L56)
- `quantize_array(x: jax.Array, x_abs_max: jax.Array, quant_dtype: jnp.dtype)` — [`L139`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/util.py#L139)
- `quantize_block(data, axis, target_dtype)` — [`L253`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/util.py#L253) — Calculates scale and quantizes a block of data.
- `quantize_tensor(x: jax.Array, dtype: jnp.dtype, dim: int = -1, block_size: int | None = None)` — [`L35`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/util.py#L35)
- `unfold_args(conditions: tuple[jax.Array | bool, ...], fn_conditions: tuple[bool, ...], fn: Callable[..., Any])` — [`L12`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/util.py#L12) — Minimize run-time branching of fn by converting jnp.bool to python bool.
- `validate_inputs(x: jax.Array, w_q: jax.Array, w_scale: jax.Array, x_abs_max: jax.Array, batch_block_size: int, out_block_size: int, in_block_size: int)` — [`L213`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/util.py#L213) — Verify inputs invoking the kernel.
- `xla_quantized_batched_matmul(x: jax.Array, w_q: jax.Array, w_scale: jax.Array, dimension_numbers: tuple, quantize_activation: bool = True)` — [`L69`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/quantized_matmul/util.py#L69) — Quantized matmul with batch dimensions via dot_general.

