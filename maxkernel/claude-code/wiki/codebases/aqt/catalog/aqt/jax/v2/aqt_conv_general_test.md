---
title: 'Module: aqt/jax/v2/aqt_conv_general_test.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/aqt_conv_general_test.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.aqt_conv_general_test`/
symbols:
  rand_unif: rand_unif().
  _apply_po2_scale: _apply_po2_scale().
  AqtConvGeneralTest: AqtConvGeneralTest#
  AqtConvGeneralTest.test_conv_general_dilated: AqtConvGeneralTest#test_conv_general_dilated().
  AqtConvGeneralTest.test_conv_general_dilated_quantized: AqtConvGeneralTest#test_conv_general_dilated_quantized().
---
# Module: [`aqt/jax/v2/aqt_conv_general_test.py`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_conv_general_test.py)

## Classes
### `AqtConvGeneralTest`
- def: [`aqt/jax/v2/aqt_conv_general_test.py:50`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_conv_general_test.py#L50)
- signature: `class AqtConvGeneralTest(parameterized.TestCase):`
- members:
  - `test_conv_general_dilated(self, lhs_bits, rhs_bits, lhs_maxval=10.0, rhs_maxval=20.0, seed=0)` — [`L62`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_conv_general_test.py#L62)
  - `test_conv_general_dilated_quantized(self, lhs_bits, rhs_bits, lhs_maxval=10.0, rhs_maxval=20.0, seed=0)` — [`L111`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_conv_general_test.py#L111) — Check that passing quantized lhs/rhs to aqt_conv_fn works.

## Functions
- `_apply_po2_scale(quantizer)` — [`L33`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_conv_general_test.py#L33)
- `rand_unif(shape, maxval, seed, dtype=jnp.float32)` — [`L26`](../../../../../../../raw/code/aqt/aqt/jax/v2/aqt_conv_general_test.py#L26)

