---
title: 'Module: jax/experimental/pallas/ops/tpu/paged_attention/quantization_utils.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/tpu/paged_attention/quantization_utils.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.tpu.paged_attention.quantization_utils`/
symbols:
  unquantize_from_int8: unquantize_from_int8().
  quantize_to_int8: quantize_to_int8().
  from_int8: from_int8().
  to_int8: to_int8().
  QuantizedTensor: QuantizedTensor#
  get_quantization_scales: get_quantization_scales().
  QuantizedTensor.weight: QuantizedTensor#weight.
  QuantizedTensor.scales: QuantizedTensor#scales.
  P: P.
  MAX_INT8: MAX_INT8.
---
# Module: [`jax/experimental/pallas/ops/tpu/paged_attention/quantization_utils.py`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/quantization_utils.py)

## Classes
### `QuantizedTensor`  ·  implements/extends NamedTuple
- def: [`jax/experimental/pallas/ops/tpu/paged_attention/quantization_utils.py:23`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/quantization_utils.py#L23)
- doc: A tensor which has been quantized to int8 and its scales.
- signature: `class QuantizedTensor(NamedTuple):`
- members:
  - `scales` — [`L32`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/quantization_utils.py#L32)
  - `weight` — [`L31`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/quantization_utils.py#L31)
- uses (calls/refs, reference-scoped): [`ndarray`](../../../../../numpy/__init__.pyi.md#ndarray)
- used by: [`paged_attention`](paged_attention_kernel.md#paged_attention), [`grouped_query_attention_reference`](util.md#grouped_query_attention_reference), [`unquantize_from_int8`](quantization_utils.md#unquantize_from_int8), [`quantize_to_int8`](quantization_utils.md#quantize_to_int8)

## Functions
- `from_int8(x: jnp.ndarray, h: jnp.ndarray, dtype: jnp.dtype = jnp.bfloat16)` — [`L48`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/quantization_utils.py#L48) — Converts an int8 array to a float array with a scale.
- `get_quantization_scales(x: jnp.ndarray)` — [`L64`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/quantization_utils.py#L64) — Computes the quantization scales for a float array.
- `quantize_to_int8(x: jnp.ndarray)` — [`L79`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/quantization_utils.py#L79) — Quantizes a float array to an int8 QuantizedTensor.
- `to_int8(x: jnp.ndarray, h: jnp.ndarray)` — [`L35`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/quantization_utils.py#L35) — Converts a float array to an int8 array with a scale.
- `unquantize_from_int8(x: QuantizedTensor, dtype: jnp.dtype = jnp.bfloat16)` — [`L94`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/quantization_utils.py#L94) — Unquantizes an int8 QuantizedTensor to a float array.

## Module values
- `MAX_INT8` — [`L20`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/quantization_utils.py#L20)
- `P` — [`L19`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/quantization_utils.py#L19)

