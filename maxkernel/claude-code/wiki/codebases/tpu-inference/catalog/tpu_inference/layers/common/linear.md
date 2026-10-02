---
title: 'Module: tpu_inference/layers/common/linear.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/linear.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.linear`/
symbols:
  sharded_quantized_matmul: sharded_quantized_matmul().
  sharded_quantized_batched_matmul: sharded_quantized_batched_matmul().
  xla_quantized_matmul: xla_quantized_matmul().
  sharded_quantized_matmul.wrapper: sharded_quantized_matmul().wrapper().
  logger: logger.
  sharded_quantized_batched_matmul.wrapper: sharded_quantized_batched_matmul().wrapper().
  _get_x_q_dtype: _get_x_q_dtype().
  _parse_einsum_dims: _parse_einsum_dims().
---
# Module: [`tpu_inference/layers/common/linear.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/linear.py)

## Functions
- `_get_x_q_dtype(w_q_dtype: jnp.dtype)` — [`L93`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/linear.py#L93) — Return 8-bit float or integer dtype depending on w_q_dtype.
- `_parse_einsum_dims(einsum_str: str)` — [`L193`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/linear.py#L193) — Parse an einsum string to extract dimension classifications.
- `sharded_quantized_batched_matmul(x: jax.Array, w_q: jax.Array, w_s: jax.Array, einsum_str: str, weight_sharding: P | NamedSharding, *, mesh: Mesh | None = None)` — [`L234`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/linear.py#L234) — Sharded quantized matmul with batch dimensions.
- `sharded_quantized_matmul(x: jax.Array, w_q: jax.Array, w_s: jax.Array, weight_sharding: P | NamedSharding, *, mesh: Mesh | None = None, x_q_dtype: jnp.dtype | None = None)` — [`L106`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/linear.py#L106) — Wrapper around the quantized matmul kernel.
- `wrapper(x, w_q, w_s)` — [`L165`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/linear.py#L165)
- `wrapper(x, w_q, w_s)` — [`L329`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/linear.py#L329)
- `xla_quantized_matmul(x: jax.Array, w_q: jax.Array, w_scale: jax.Array, quantize_activation=True)` — [`L29`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/linear.py#L29) — Reference (pure JAX) implementation of the quantized matmul kernel below.

## Module values
- `logger` — [`L26`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/linear.py#L26)

