---
title: 'Module: aqt/jax/v2/numerics/fp8_numerics_test.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/numerics/fp8_numerics_test.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.numerics.fp8_numerics_test`/
symbols:
  averaged_stochastic_rounding: averaged_stochastic_rounding().
  MyTest: MyTest#
  MyTest.test_2bit_mantissa: MyTest#test_2bit_mantissa().
  MyTest.test_3bit_mantissa: MyTest#test_3bit_mantissa().
  MyTest.test_retains_dtype: MyTest#test_retains_dtype().
  MyTest.test_mnist_training: MyTest#test_mnist_training().
  MyTest.test_fp8_stochastic_rounding: MyTest#test_fp8_stochastic_rounding().
  MyTest.test_fp8_realization_with_jit: MyTest#test_fp8_realization_with_jit().
  illustrate_bf16: illustrate_bf16().
  illustrate_bf16_2: illustrate_bf16_2().
  plot_sr_error: plot_sr_error().
---
# Module: [`aqt/jax/v2/numerics/fp8_numerics_test.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics_test.py)

## Classes
### `MyTest`
- def: [`aqt/jax/v2/numerics/fp8_numerics_test.py:48`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics_test.py#L48)
- signature: `class MyTest(parameterized.TestCase):`
- members:
  - `test_2bit_mantissa(self)` — [`L50`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics_test.py#L50)
  - `test_3bit_mantissa(self)` — [`L81`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics_test.py#L81)
  - `test_fp8_realization_with_jit(self, dtype)` — [`L221`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics_test.py#L221) — Test that the fp8 quantization is not simplified away by jit.
  - `test_fp8_stochastic_rounding(self, key=0, x_count=1024, x_min=1.0, x_max=4.0, sr_count=10000, dtype=jnp.float8_e5m2, exponent_bits=5, mantissa_bits=2)` — [`L184`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics_test.py#L184)
  - `test_mnist_training(self, fwd_bits: str)` — [`L127`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics_test.py#L127)
  - `test_retains_dtype(self)` — [`L112`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics_test.py#L112)

## Functions
- `averaged_stochastic_rounding(numerics, key, x_count, x_min, x_max, sr_count)` — [`L29`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics_test.py#L29)
- `illustrate_bf16()` — [`L245`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics_test.py#L245)
- `illustrate_bf16_2()` — [`L259`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics_test.py#L259)
- `plot_sr_error(x_min=1.0, x_max=2.0, sr_count=64 * 1024, x_count=1024)` — [`L299`](../../../../../../../../raw/code/aqt/aqt/jax/v2/numerics/fp8_numerics_test.py#L299)

