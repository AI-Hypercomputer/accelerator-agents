---
title: 'Module: tests/_src/core/conv_general_tpu_test.py'
type: catalog
provenance: extracted
module: tests/_src/core/conv_general_tpu_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.core.conv_general_tpu_test`/
symbols:
  ConvGeneralTest.f: ConvGeneralTest#f().
  ConvGeneralTest.quantize: ConvGeneralTest#quantize().
  ConvGeneralTest.test_numerics: ConvGeneralTest#test_numerics().
  mae: mae().
  ConvGeneralTest._make_array: ConvGeneralTest#_make_array().
  ConvGeneralTest: ConvGeneralTest#
---
# Module: [`tests/_src/core/conv_general_tpu_test.py`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_tpu_test.py)

## Classes
### `ConvGeneralTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/conv_general_tpu_test.py:30`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_tpu_test.py#L30)
- signature: `class ConvGeneralTest(parameterized.TestCase):`
- members:
  - `f(lhs, rhs)` — [`L137`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_tpu_test.py#L137)
  - `quantize(array, for_lhs)` — [`L126`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_tpu_test.py#L126)
  - `test_numerics(self, *, lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...], qtype: jax.typing.DTypeLike, dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers, lhs_asymmetric: bool = False, conv_kwargs: dict[str, Any] | None = None, expected_mae: float = 0.01)` — [`L101`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_tpu_test.py#L101)
- protocol/private: `_make_array`[`L32`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_tpu_test.py#L32)
- uses (calls/refs, reference-scoped): [`quantize`](../../../qwix/_src/core/qarray.md#quantize), [`_fast_conv_general_dilated`](../../../qwix/_src/core/conv_general.md#_fast_conv_general_dilated), [`get_how_to_quantize`](../../../qwix/_src/core/conv_general.md#get_how_to_quantize), [`_slow_conv_general_dilated`](../../../qwix/_src/core/conv_general.md#_slow_conv_general_dilated)  (1 test-only)

## Functions
- `mae(lhs: jax.Array, rhs: jax.Array)` — [`L25`](../../../../../../../raw/code/qwix/tests/_src/core/conv_general_tpu_test.py#L25)

