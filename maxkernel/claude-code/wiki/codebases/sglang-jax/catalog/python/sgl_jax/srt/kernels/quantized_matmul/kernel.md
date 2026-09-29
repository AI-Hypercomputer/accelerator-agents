---
title: 'Module: python/sgl_jax/srt/kernels/quantized_matmul/kernel.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/quantized_matmul/kernel.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.quantized_matmul.kernel`/xla_quantized_matmul_local().
symbols:
  xla_quantized_matmul_local: ''
---
# Module: [`python/sgl_jax/srt/kernels/quantized_matmul/kernel.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/kernel.py)

## Functions
- `xla_quantized_matmul_local(x: jax.Array, w_q: jax.Array, w_scale: jax.Array, quantize_activation: bool = True, reduce_axis: str | None = None, compute_dtype: jnp.dtype | None = None, weight_block_size: tuple[int, int] | None = None, activation_quant_dtype: jnp.dtype | None = None, allow_narrow_n_blockwise: bool = False, output_scatter_dimension: int | None = None)` — [`L16`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/quantized_matmul/kernel.py#L16) — Local quantized matmul for use inside shard_map.

