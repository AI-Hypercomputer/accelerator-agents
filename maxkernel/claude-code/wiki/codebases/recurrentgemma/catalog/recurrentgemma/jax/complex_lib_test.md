---
title: 'Module: recurrentgemma/jax/complex_lib_test.py'
type: catalog
provenance: extracted
module: recurrentgemma/jax/complex_lib_test.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.jax.complex_lib_test`/
symbols:
  _TEST_ARRAY._TEST_ARRAY: _TEST_ARRAY._TEST_ARRAY.
  TestForwardCase: TestForwardCase#
  TestForwardCase.module: TestForwardCase#module.
  TestForwardCase.op_name: TestForwardCase#op_name.
  TestForwardCase.args: TestForwardCase#args.
  TestForwardCase.kwargs: TestForwardCase#kwargs.
  _FORWARD_OPS_UNDER_TEST._FORWARD_OPS_UNDER_TEST: _FORWARD_OPS_UNDER_TEST._FORWARD_OPS_UNDER_TEST.
  ComplexTest.test_change_dtype: ComplexTest#test_change_dtype().
  ComplexTest.test_complex_wrapper_with_complex_wrapper_ops: ComplexTest#test_complex_wrapper_with_complex_wrapper_ops().
  ComplexTest.test_complex_wrapper_with_jax_array_ops: ComplexTest#test_complex_wrapper_with_jax_array_ops().
  ComplexTest.test_complex_wrapper_with_jax_array_ops_commutes: ComplexTest#test_complex_wrapper_with_jax_array_ops_commutes().
  ComplexTest.test_complex_wrapper_ops_with_jax_complex_exception: ComplexTest#test_complex_wrapper_ops_with_jax_complex_exception().
  ComplexTest.test_complex_wrapper_ops_with_different_dtype_exception: ComplexTest#test_complex_wrapper_ops_with_different_dtype_exception().
  ComplexTest.to_complex_wrapper: ComplexTest#to_complex_wrapper().
  _get_testing_arrays: _get_testing_arrays().
  ComplexTest.test_treat_methods: ComplexTest#test_treat_methods().
  ComplexTest.test_einsum: ComplexTest#test_einsum().
  _BINARY_OPS._BINARY_OPS: _BINARY_OPS._BINARY_OPS.
  ComplexTest.convert_to_complex_wrapper: ComplexTest#convert_to_complex_wrapper().
  _REAL_PART: _REAL_PART.
  _IMAG_PART: _IMAG_PART.
  _BINARY_OPS_REVERSED._BINARY_OPS_REVERSED: _BINARY_OPS_REVERSED._BINARY_OPS_REVERSED.
  ComplexTest: ComplexTest#
---
# Module: [`recurrentgemma/jax/complex_lib_test.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py)

## Classes
### `ComplexTest`  ·  implements/extends TestCase
- def: [`recurrentgemma/jax/complex_lib_test.py:177`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L177)
- signature: `class ComplexTest(parameterized.TestCase):`
- members:
  - `convert_to_complex_wrapper(x: Any)` — [`L280`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L280)
  - `test_change_dtype(self)` — [`L223`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L223)
  - `test_complex_wrapper_ops_with_different_dtype_exception(self, op_name: str)` — [`L250`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L250)
  - `test_complex_wrapper_ops_with_jax_complex_exception(self, op_name: str)` — [`L237`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L237)
  - `test_complex_wrapper_with_complex_wrapper_ops(self, op_name: str)` — [`L180`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L180)
  - `test_complex_wrapper_with_jax_array_ops(self, op_name: str)` — [`L195`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L195)
  - `test_complex_wrapper_with_jax_array_ops_commutes(self, op_name: str)` — [`L209`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L209) — Tests to make sure that the reverse operations are implemented.
  - `test_einsum(self)` — [`L291`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L291)
  - `test_treat_methods(self, module: types.ModuleType, op_name: str, args: list[Any], kwargs: dict[str, Any])` — [`L270`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L270)
  - `to_complex_wrapper(x: Any)` — [`L296`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L296)
- uses (calls/refs, reference-scoped): [`Complex`](complex_lib.md#Complex), [`real`](complex_lib.md#Complex.real), [`imag`](complex_lib.md#Complex.imag), [`_TEST_ARRAY`](complex_lib_test.md#_TEST_ARRAY._TEST_ARRAY), [`_FORWARD_OPS_UNDER_TEST`](complex_lib_test.md#_FORWARD_OPS_UNDER_TEST._FORWARD_OPS_UNDER_TEST), [`dtype`](complex_lib.md#Complex.dtype), [`_get_testing_arrays`](complex_lib_test.md#_get_testing_arrays), [`_BINARY_OPS`](complex_lib_test.md#_BINARY_OPS._BINARY_OPS), [`_BINARY_OPS_REVERSED`](complex_lib_test.md#_BINARY_OPS_REVERSED._BINARY_OPS_REVERSED)

### `TestForwardCase`  ·  implements/extends NamedTuple
- def: [`recurrentgemma/jax/complex_lib_test.py:44`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L44)
- signature: `class TestForwardCase(NamedTuple):`
- members:
  - `args` — [`L47`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L47)
  - `kwargs` — [`L48`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L48)
  - `module` — [`L45`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L45)
  - `op_name` — [`L46`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L46)
- used by: [`_FORWARD_OPS_UNDER_TEST`](complex_lib_test.md#_FORWARD_OPS_UNDER_TEST._FORWARD_OPS_UNDER_TEST)

## Functions
- `_get_testing_arrays(count: int, shape: tuple[int, ...] = (8, 8))` — [`L169`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L169)

## Module values
- `_BINARY_OPS` — [`L31`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L31)
- `_BINARY_OPS_REVERSED` — [`L32`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L32)
- `_FORWARD_OPS_UNDER_TEST` — [`L51`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L51)
- `_IMAG_PART` — [`L37`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L37)
- `_REAL_PART` — [`L34`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L34)
- `_TEST_ARRAY` — [`L41`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib_test.py#L41)

