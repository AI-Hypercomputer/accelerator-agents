---
title: 'Module: tests/_src/core/numerics_test.py'
type: catalog
provenance: extracted
module: tests/_src/core/numerics_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.core.numerics_test`/NumericsTest#
symbols:
  NumericsTest._assert_equal: _assert_equal().
  NumericsTest.test_mxfp: test_mxfp().
  NumericsTest.test_arbitrary_integer_dtype: test_arbitrary_integer_dtype().
  NumericsTest.test_uint: test_uint().
  NumericsTest.test_nf4: test_nf4().
  NumericsTest.test_convert_to: test_convert_to().
  NumericsTest.test_inf: test_inf().
  NumericsTest.test_stochastic_rounding: test_stochastic_rounding().
  NumericsTest.test_invalid_cases: test_invalid_cases().
  NumericsTest: ''
---
# Module: [`tests/_src/core/numerics_test.py`](../../../../../../../raw/code/qwix/tests/_src/core/numerics_test.py)

## Classes
### `NumericsTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/numerics_test.py:23`](../../../../../../../raw/code/qwix/tests/_src/core/numerics_test.py#L23)
- signature: `class NumericsTest(absltest.TestCase):`
- members:
  - `test_arbitrary_integer_dtype(self)` — [`L52`](../../../../../../../raw/code/qwix/tests/_src/core/numerics_test.py#L52)
  - `test_convert_to(self)` — [`L34`](../../../../../../../raw/code/qwix/tests/_src/core/numerics_test.py#L34)
  - `test_inf(self)` — [`L46`](../../../../../../../raw/code/qwix/tests/_src/core/numerics_test.py#L46)
  - `test_invalid_cases(self)` — [`L28`](../../../../../../../raw/code/qwix/tests/_src/core/numerics_test.py#L28)
  - `test_mxfp(self)` — [`L120`](../../../../../../../raw/code/qwix/tests/_src/core/numerics_test.py#L120)
  - `test_nf4(self)` — [`L101`](../../../../../../../raw/code/qwix/tests/_src/core/numerics_test.py#L101)
  - `test_stochastic_rounding(self)` — [`L83`](../../../../../../../raw/code/qwix/tests/_src/core/numerics_test.py#L83)
  - `test_uint(self)` — [`L76`](../../../../../../../raw/code/qwix/tests/_src/core/numerics_test.py#L76)
- protocol/private: `_assert_equal`[`L25`](../../../../../../../raw/code/qwix/tests/_src/core/numerics_test.py#L25)
- uses (calls/refs, reference-scoped): [`convert_to`](../../../qwix/_src/core/numerics.md#convert_to), [`get_symmetric_bound`](../../../qwix/_src/core/numerics.md#get_symmetric_bound), [`convert_from`](../../../qwix/_src/core/numerics.md#convert_from), [`get_asymmetric_bound`](../../../qwix/_src/core/numerics.md#get_asymmetric_bound), [`uniform_noise`](../../../qwix/_src/core/stochastic_rounding.md#uniform_noise)

