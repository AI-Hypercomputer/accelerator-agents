---
title: 'Module: aqt/jax/v2/pallas/dot_general.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/pallas/dot_general.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.pallas.dot_general`/
symbols:
  dot_general: dot_general().
  QTensor: QTensor.
  DequantMode: DequantMode.
  _dtype_to_bits: _dtype_to_bits().
  load_qtensor: load_qtensor().
---
# Module: [`aqt/jax/v2/pallas/dot_general.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/dot_general.py)

## Functions
- `_dtype_to_bits(dtype)` — [`L41`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/dot_general.py#L41)
- `dot_general(lhs: QTensor | jax.Array, rhs: QTensor | jax.Array, dimension_numbers: jax.lax.DotDimensionNumbers, precision=None, preferred_element_type=None, lhs_dequant_mode: DequantMode = DequantMode.OUTPUT, rhs_dequant_mode: DequantMode = DequantMode.OUTPUT)` — [`L57`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/dot_general.py#L57) — lax.dot_general replacement for pallas which always returns dequantized output. — documented in [aqt-jax-aqt_dot_general](../../../../../concepts/aqt-jax-aqt_dot_general.md)
- `load_qtensor(qt: QTensor)` — [`L28`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/dot_general.py#L28) — Materialize QTensor of MemoryRef of pallas into QTensor of jax.Array.

## Module values
- `DequantMode` — [`L23`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/dot_general.py#L23)
- `QTensor` — [`L22`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/dot_general.py#L22) — documented in [aqt-jax-v2-aqt_tensor](../../../../../concepts/aqt-jax-v2-aqt_tensor.md)

