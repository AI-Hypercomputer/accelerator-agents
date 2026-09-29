---
title: 'Module: tests/_src/core/stochastic_rounding_test.py'
type: catalog
provenance: extracted
module: tests/_src/core/stochastic_rounding_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.core.stochastic_rounding_test`/StochasticRoundingTest#
symbols:
  StochasticRoundingTest.test_noise_fn_jit_compatible: test_noise_fn_jit_compatible().
  StochasticRoundingTest.test_uniform_noise: test_uniform_noise().
  StochasticRoundingTest.test_low_bit_uniform_noise: test_low_bit_uniform_noise().
  StochasticRoundingTest.apply_noise: apply_noise().
  StochasticRoundingTest: ''
---
# Module: [`tests/_src/core/stochastic_rounding_test.py`](../../../../../../../raw/code/qwix/tests/_src/core/stochastic_rounding_test.py)

## Classes
### `StochasticRoundingTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/stochastic_rounding_test.py:22`](../../../../../../../raw/code/qwix/tests/_src/core/stochastic_rounding_test.py#L22)
- signature: `class StochasticRoundingTest(parameterized.TestCase):`
- members:
  - `apply_noise(fn, s)` — [`L62`](../../../../../../../raw/code/qwix/tests/_src/core/stochastic_rounding_test.py#L62)
  - `test_low_bit_uniform_noise(self)` — [`L36`](../../../../../../../raw/code/qwix/tests/_src/core/stochastic_rounding_test.py#L36)
  - `test_noise_fn_jit_compatible(self)` — [`L52`](../../../../../../../raw/code/qwix/tests/_src/core/stochastic_rounding_test.py#L52)
  - `test_uniform_noise(self)` — [`L24`](../../../../../../../raw/code/qwix/tests/_src/core/stochastic_rounding_test.py#L24)
- uses (calls/refs, reference-scoped): [`get_noise_fn`](../../../qwix/_src/core/stochastic_rounding.md#get_noise_fn)

