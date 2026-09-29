---
title: 'Module: simply/utils/position_encoding_test.py'
type: catalog
provenance: extracted
module: simply/utils/position_encoding_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.position_encoding_test`/RoPETest#
symbols:
  RoPETest.test_preserves_shape: test_preserves_shape().
  RoPETest.test_preserves_dtype: test_preserves_dtype().
  RoPETest.test_position_dependent_output: test_position_dependent_output().
  RoPETest.test_custom_segment_positions: test_custom_segment_positions().
  RoPETest.test_different_max_timescales: test_different_max_timescales().
  RoPETest.test_different_scale_factors: test_different_scale_factors().
  RoPETest: ''
---
# Module: [`simply/utils/position_encoding_test.py`](../../../../../../raw/code/simply/simply/utils/position_encoding_test.py)

## Classes
### `RoPETest`  ·  implements/extends TestCase
- def: [`simply/utils/position_encoding_test.py:21`](../../../../../../raw/code/simply/simply/utils/position_encoding_test.py#L21)
- signature: `class RoPETest(parameterized.TestCase):`
- members:
  - `test_custom_segment_positions(self)` — [`L46`](../../../../../../raw/code/simply/simply/utils/position_encoding_test.py#L46) — Custom segment_positions should affect output.
  - `test_different_max_timescales(self)` — [`L61`](../../../../../../raw/code/simply/simply/utils/position_encoding_test.py#L61) — Different max_timescale should produce different outputs.
  - `test_different_scale_factors(self)` — [`L68`](../../../../../../raw/code/simply/simply/utils/position_encoding_test.py#L68) — Different scale_factor should produce different outputs.
  - `test_position_dependent_output(self)` — [`L38`](../../../../../../raw/code/simply/simply/utils/position_encoding_test.py#L38) — Same embedding at different positions should produce different outputs.
  - `test_preserves_dtype(self, dtype)` — [`L31`](../../../../../../raw/code/simply/simply/utils/position_encoding_test.py#L31) — RoPE should preserve dtype.
  - `test_preserves_shape(self)` — [`L23`](../../../../../../raw/code/simply/simply/utils/position_encoding_test.py#L23) — RoPE.apply() should return same shape as input.
- uses (calls/refs, reference-scoped): [`RoPE`](position_encoding.md#RoPE)

