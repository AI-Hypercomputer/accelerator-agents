---
title: 'Module: python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/util.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/util.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels.util`/
symbols:
  get_kernel_name: get_kernel_name().
  unfold_args: unfold_args().
  _dtype_bits: _dtype_bits().
  next_multiple: next_multiple().
  quantize_tensor: quantize_tensor().
  quantize_block: quantize_block().
  get_vmem_limit: get_vmem_limit().
  xla_quantized_matmul: xla_quantized_matmul().
  xla_quantized_batched_matmul: xla_quantized_batched_matmul().
  validate_inputs: validate_inputs().
  quantize_array: quantize_array().
  get_max_min: get_max_min().
---
# Module: [`python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/util.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/util.py)

## Functions
- `_dtype_bits(dtype: jnp.dtype)` — [`L14`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/util.py#L14)
- `get_kernel_name(tuned_value: TunedValue)` — [`L61`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/util.py#L61)
- `get_max_min(target_dtype)` — [`L288`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/util.py#L288)
- `get_vmem_limit(n_batch: int, n_out: int, n_in: int, batch_block_size: int, out_block_size: int, in_block_size: int, x_dtype: jnp.dtype, x_q_dtype: jnp.dtype, w_q_dtype: jnp.dtype, scale_dtype: jnp.dtype, out_dtype: jnp.dtype, acc_dtype: jnp.dtype, save_acc: bool, save_x_q: bool, upper_limit_bytes: int)` — [`L194`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/util.py#L194) — Calculate VMEM limit for the kernel.
- `next_multiple(x, multiple)` — [`L57`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/util.py#L57)
- `quantize_array(x: jax.Array, x_abs_max: jax.Array, quant_dtype: jnp.dtype)` — [`L178`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/util.py#L178)
- `quantize_block(data, axis, target_dtype)` — [`L297`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/util.py#L297) — Calculates scale and quantizes a block of data.
- `quantize_tensor(x: jax.Array, dtype: jnp.dtype, dim: int = -1, block_size: int | None = None)` — [`L39`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/util.py#L39)
- `unfold_args(conditions: tuple[jax.Array | bool, ...], fn_conditions: tuple[bool, ...], fn: Callable[..., Any])` — [`L18`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/util.py#L18) — Minimize run-time branching of fn by converting jnp.bool to python bool.
- `validate_inputs(x: jax.Array, w_q: jax.Array, w_scale: jax.Array, x_abs_max: jax.Array, x_q_dtype: jnp.dtype, batch_block_size: int, out_block_size: int, in_block_size: int)` — [`L249`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/util.py#L249) — Verify inputs invoking the kernel.
- `xla_quantized_batched_matmul(x: jax.Array, w_q: jax.Array, w_scale: jax.Array, dimension_numbers: tuple, quantize_activation: bool = True)` — [`L111`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/util.py#L111) — Quantized matmul with batch dimensions via dot_general.
- `xla_quantized_matmul(x: jax.Array, w_q: jax.Array, w_scale: jax.Array, quantize_activation=True)` — [`L68`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/util.py#L68) — Reference (pure JAX) implementation of the quantized matmul kernel below.

