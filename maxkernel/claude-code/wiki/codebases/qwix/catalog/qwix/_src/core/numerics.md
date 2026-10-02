---
title: 'Module: qwix/_src/core/numerics.py'
type: catalog
provenance: extracted
module: qwix/_src/core/numerics.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.core.numerics`/
symbols:
  convert_to: convert_to().
  should_quantize: should_quantize().
  get_symmetric_bound: get_symmetric_bound().
  convert_from: convert_from().
  fp_to_nf4: fp_to_nf4().
  nf4_to_fp: nf4_to_fp().
  NoiseFn: NoiseFn.
  get_asymmetric_bound: get_asymmetric_bound().
  get_nf4_buckets: get_nf4_buckets().
  _QUANTIZE_DTYPES: _QUANTIZE_DTYPES.
  can_dequant_on_output: can_dequant_on_output().
  fp_to_nf4.bucketize: fp_to_nf4().bucketize().
  nf4_to_fp.reverse_bucketize: nf4_to_fp().reverse_bucketize().
---
# Module: [`qwix/_src/core/numerics.py`](../../../../../../../raw/code/qwix/qwix/_src/core/numerics.py)

## Functions
- `bucketize(x)` — [`L255`](../../../../../../../raw/code/qwix/qwix/_src/core/numerics.py#L255)
- `can_dequant_on_output(qtype: jax.typing.DTypeLike)` — [`L44`](../../../../../../../raw/code/qwix/qwix/_src/core/numerics.py#L44) — Checks if the qtype supports dequantizing after quantized computation.
- `convert_from(x: jax.Array, qtype: jax.typing.DTypeLike)` — [`L194`](../../../../../../../raw/code/qwix/qwix/_src/core/numerics.py#L194) — Converts a non-uniform quantized array back to floating-point values. — documented in [qwix-_src-core-numerics](../../../../concepts/qwix-_src-core-numerics.md)
- `convert_to(x: jax.Array, qtype: jax.typing.DTypeLike, noise_fn: NoiseFn | None = None)` — [`L134`](../../../../../../../raw/code/qwix/qwix/_src/core/numerics.py#L134) — Converts a high-precision array to the quantized representation. — documented in [qwix-_src-core-numerics](../../../../concepts/qwix-_src-core-numerics.md)
- `fp_to_nf4(array: jax.Array)` — [`L251`](../../../../../../../raw/code/qwix/qwix/_src/core/numerics.py#L251) — Quantizes an array to a 4-bit NormalFloat representation. — documented in [qwix-_src-core-numerics](../../../../concepts/qwix-_src-core-numerics.md)
- `get_asymmetric_bound(qtype: jax.typing.DTypeLike)` — [`L64`](../../../../../../../raw/code/qwix/qwix/_src/core/numerics.py#L64) — Returns the continuous range of target values before rounding.
- `get_nf4_buckets()` — [`L225`](../../../../../../../raw/code/qwix/qwix/_src/core/numerics.py#L225) — Returns the NF4 buckets. — documented in [qwix-_src-core-numerics](../../../../concepts/qwix-_src-core-numerics.md)
- `get_symmetric_bound(qtype: jax.typing.DTypeLike)` — [`L93`](../../../../../../../raw/code/qwix/qwix/_src/core/numerics.py#L93) — Returns the maximum magnitude of continuous target values before rounding. — documented in [qwix-_src-core-numerics](../../../../concepts/qwix-_src-core-numerics.md)
- `nf4_to_fp(array: jax.Array)` — [`L263`](../../../../../../../raw/code/qwix/qwix/_src/core/numerics.py#L263) — Dequantizes a NF4 array to original dtype. — documented in [qwix-_src-core-numerics](../../../../concepts/qwix-_src-core-numerics.md)
- `reverse_bucketize(x)` — [`L267`](../../../../../../../raw/code/qwix/qwix/_src/core/numerics.py#L267)
- `should_quantize(dtype: jax.typing.DTypeLike)` — [`L29`](../../../../../../../raw/code/qwix/qwix/_src/core/numerics.py#L29) — Checks if a given dtype is a floating-point type eligible for quantization. — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)

## Module values
- `NoiseFn` — [`L25`](../../../../../../../raw/code/qwix/qwix/_src/core/numerics.py#L25) — documented in [qwix-_src-core-numerics](../../../../concepts/qwix-_src-core-numerics.md)
- `_QUANTIZE_DTYPES` — [`L26`](../../../../../../../raw/code/qwix/qwix/_src/core/numerics.py#L26)

