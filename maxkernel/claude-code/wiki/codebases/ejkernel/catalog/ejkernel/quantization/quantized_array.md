---
title: 'Module: ejkernel/quantization/quantized_array.py'
type: catalog
provenance: extracted
module: ejkernel/quantization/quantized_array.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.quantization.quantized_array`/
symbols:
  quantize_array: quantize_array().
  prepack_quantized_array: prepack_quantized_array().
  QuantizedArray.__repr__: QuantizedArray#__repr__().
  QuantizedArray.matmul: QuantizedArray#matmul().
  QuantizedArray.dequantize: QuantizedArray#dequantize().
  QuantizedArray.tree_flatten: QuantizedArray#tree_flatten().
  QuantizedArray.from_quantized: QuantizedArray#from_quantized().
  QuantizedArray.scales: QuantizedArray#scales.
  QuantizedArray.mode: QuantizedArray#mode.
  QuantizedArray.data: QuantizedArray#data.
  QuantizedArray.zeros: QuantizedArray#zeros.
  QuantizedArray.as_tuple: QuantizedArray#as_tuple().
  QuantizedArray.storage_bits: QuantizedArray#storage_bits().
  QuantizedArray.group_size: QuantizedArray#group_size.
  QuantizedArray.axis: QuantizedArray#axis.
  QuantizedArray._storage_bits: QuantizedArray#_storage_bits().
  QuantizedArray.logical_num_values: QuantizedArray#logical_num_values().
  QuantizedArray.runtime_config: QuantizedArray#runtime_config.
  QuantizedArray.metadata_bits_per_value: QuantizedArray#metadata_bits_per_value().
  QuantizedArray: QuantizedArray#
  QuantizedArray.bits: QuantizedArray#bits.
  QuantizedArray.data_storage_bits: QuantizedArray#data_storage_bits().
  QuantizedArray.scales_storage_bits: QuantizedArray#scales_storage_bits().
  QuantizedArray.zeros_storage_bits: QuantizedArray#zeros_storage_bits().
  QuantizedArray.actual_bits_per_value: QuantizedArray#actual_bits_per_value().
  QuantizedArray.storage_bytes: QuantizedArray#storage_bytes().
  QuantizedArray.payload_bits_per_value: QuantizedArray#payload_bits_per_value().
  QuantizedArray.storage_kib: QuantizedArray#storage_kib().
  QuantizedArray._shape_or_none: QuantizedArray#_shape_or_none().
  QuantizedArray.storage_mib: QuantizedArray#storage_mib().
  QuantizedArray.effective_bits_per_value: QuantizedArray#effective_bits_per_value().
  QuantizedArray.__str__: QuantizedArray#__str__().
  QuantizedArray._numel: QuantizedArray#_numel().
  QuantizedArray.tree_unflatten: QuantizedArray#tree_unflatten().
  __all__: __all__.
---
# Module: [`ejkernel/quantization/quantized_array.py`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py)

## Classes
### `QuantizedArray`
- def: [`ejkernel/quantization/quantized_array.py:50`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L50)
- doc: Packed quantized tensor with all metadata required for runtime use.
- signature: `class QuantizedArray:`
- members:
  - `_numel(x: jax.Array | None)` — [`L147`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L147) — Return the total element count of *x*, or 0 if *x* is ``None``.
  - `_shape_or_none(x: jax.Array | None)` — [`L140`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L140) — Return the shape tuple of *x*, or ``None`` if *x* is ``None``.
  - `_storage_bits(x: jax.Array | None)` — [`L156`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L156) — Return storage size of *x* in bits, or 0 if *x* is ``None``.
  - `actual_bits_per_value(self)` — [`L230`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L230) — Effective bits/value including packed payload and metadata overhead.
  - `as_tuple(self)` — [`L320`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L320) — Return the payload as a legacy tuple compatible with low-level APIs.
  - `data_storage_bits(self)` — [`L172`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L172) — Storage size of the packed weight data array in bits.
  - `dequantize(self, *, runtime_config: QuantRuntimeConfig | None = None)` — [`L334`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L334) — Dequantize this container into floating-point weights.
  - `effective_bits_per_value(self)` — [`L238`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L238) — Alias for ``actual_bits_per_value``.
  - `from_quantized(cls, data: jax.Array, scales: jax.Array, zeros: jax.Array | None = None, *, group_size: int | None = None, bits: int | None = None, mode: QuantizationMode = "affine", axis: QuantizationAxis = "row", runtime_config: QuantRuntimeConfig | None = None)` — [`L263`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L263) — Construct a container from already-quantized buffers.
  - `logical_num_values(self)` — [`L163`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L163) — Number of dequantized scalar values represented by this payload.
  - `matmul(self, x: jax.Array, *, fuse: bool = True, strict_fuse: bool | None = None, tpu_path: Literal["packed"] | None = None, allow_dense_fallback: bool | None = None, transpose: bool | None = None, axis: QuantizationAxis | None = None, platform: Literal["triton", "pallas", "cuda", "cute", "xla", "auto"] | None = None)` — [`L364`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L364) — Run quantized matmul against activation tensor *x*. — documented in [ejkernel-quantization-quantized_array](../../../concepts/ejkernel-quantization-quantized_array.md)
  - `metadata_bits_per_value(self)` — [`L219`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L219) — Bits per dequantized value contributed by scales and zeros overhead.
  - `payload_bits_per_value(self)` — [`L207`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L207) — Bits per dequantized value contributed by the packed payload only.
  - `scales_storage_bits(self)` — [`L177`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L177) — Storage size of the scales metadata array in bits.
  - `storage_bits(self)` — [`L187`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L187) — Total storage in bits: packed data + scales + zeros (if present).
  - `storage_bytes(self)` — [`L192`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L192) — Total storage in bytes (``storage_bits // 8``).
  - `storage_kib(self)` — [`L197`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L197) — Total storage in kibibytes (KiB = 1024 bytes).
  - `storage_mib(self)` — [`L202`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L202) — Total storage in mebibytes (MiB = 1024 KiB).
  - `tree_flatten(self)` — [`L100`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L100) — Flatten for JAX pytree traversal.
  - `tree_unflatten(cls, aux, children)` — [`L114`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L114) — Reconstruct from flattened pytree representation.
  - `zeros_storage_bits(self)` — [`L182`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L182) — Storage size of the zeros metadata array in bits (0 for non-affine modes).
  - `axis` — [`L97`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L97)
  - `bits` — [`L96`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L96)
  - `data` — [`L91`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L91) — documented in [ejkernel-quantization-quantized_array](../../../concepts/ejkernel-quantization-quantized_array.md)
  - `group_size` — [`L95`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L95)
  - `mode` — [`L94`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L94) — documented in [ejkernel-quantization-quantized_array](../../../concepts/ejkernel-quantization-quantized_array.md)
  - `runtime_config` — [`L98`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L98)
  - `scales` — [`L92`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L92) — documented in [ejkernel-quantization-quantized_array](../../../concepts/ejkernel-quantization-quantized_array.md)
  - `zeros` — [`L93`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L93) — documented in [ejkernel-quantization-quantized_array](../../../concepts/ejkernel-quantization-quantized_array.md)
- protocol/private: `__repr__`[`L242`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L242), `__str__`[`L259`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L259)
- uses (calls/refs, reference-scoped): [`quantized_matmul`](../modules/operations/quantized_matmul.md#quantized_matmul), [`QuantRuntimeConfig`](runtime.md#QuantRuntimeConfig), [`dequantize`](_quants/quantizations.md#dequantize), [`QuantizationAxis`](_utils/qparams.md#QuantizationAxis), [`QuantizationMode`](_utils/qparams.md#QuantizationMode), [`quantized_matmul`](_quants/quantizations.md#quantized_matmul), [`resolve_qparams`](_utils/qparams.md#resolve_qparams), [`normalize_axis`](_utils/qparams.md#normalize_axis)
- used by: [`prepack_quantized_array`](quantized_array.md#prepack_quantized_array), [`quantize_array`](quantized_array.md#quantize_array)

## Functions
- `prepack_quantized_array(w: jax.Array,, group_size: int | None = None, bits: int | None = None, mode: QuantizationMode = "affine", transpose: bool = True, axis: QuantizationAxis | None = None, runtime_config: QuantRuntimeConfig | None = None)` — [`L531`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L531) — Quantize and prepack *w* into a :class:`QuantizedArray` for fused kernels. — documented in [ejkernel-quantization-quantized_array](../../../concepts/ejkernel-quantization-quantized_array.md)
- `quantize_array(w: jax.Array,, group_size: int | None = None, bits: int | None = None, mode: QuantizationMode = "affine", axis: QuantizationAxis | None = None, runtime_config: QuantRuntimeConfig | None = None)` — [`L445`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L445) — Quantize *w* and return a :class:`QuantizedArray` container. — documented in [ejkernel-quantization-quantized_array](../../../concepts/ejkernel-quantization-quantized_array.md)

## Module values
- `__all__` — [`L599`](../../../../../../raw/code/ejkernel/ejkernel/quantization/quantized_array.py#L599)

