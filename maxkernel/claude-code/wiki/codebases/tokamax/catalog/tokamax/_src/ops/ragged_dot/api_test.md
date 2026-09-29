---
title: 'Module: tokamax/_src/ops/ragged_dot/api_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/api_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.api_test`/
symbols:
  RaggedDotTest.test_basic_api: RaggedDotTest#test_basic_api().
  RaggedDotImplementationTest: RaggedDotImplementationTest#
  RaggedDotImplementationTest.__init__: RaggedDotImplementationTest#__init__().
  RaggedDotImplementationTest.setUp: RaggedDotImplementationTest#setUp().
  RaggedDotTest.test_manual_axis_type: RaggedDotTest#test_manual_axis_type().
  RaggedDotMosaicTest.__init__: RaggedDotMosaicTest#__init__().
  _get_input_data: _get_input_data().
  RaggedDotMosaicTest._dot_fn: RaggedDotMosaicTest#_dot_fn.
  RaggedDotMosaicTest: RaggedDotMosaicTest#
  RaggedDotMosaicTest.setUp: RaggedDotMosaicTest#setUp().
  RaggedDotTritonTest: RaggedDotTritonTest#
  RaggedDotTritonTest.__init__: RaggedDotTritonTest#__init__().
  RaggedDotTritonTest.setUp: RaggedDotTritonTest#setUp().
  RaggedDotXlaTest: RaggedDotXlaTest#
  RaggedDotXlaTest.__init__: RaggedDotXlaTest#__init__().
  RaggedDotTest.f: RaggedDotTest#f().
  relu: relu().
  RaggedDotTest.f_gt: RaggedDotTest#f_gt().
  RaggedDotMosaicTest.fn: RaggedDotMosaicTest#fn().
  RaggedDotTest: RaggedDotTest#
  RaggedDotTritonTest._test_bench: RaggedDotTritonTest#_test_bench().
---
# Module: [`tokamax/_src/ops/ragged_dot/api_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py)

## Classes
### `RaggedDotImplementationTest`  ·  implements/extends RaggedDotTestBase
- def: [`tokamax/_src/ops/ragged_dot/api_test.py:151`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L151)
- signature: `class RaggedDotImplementationTest(test_base.RaggedDotTestBase):`
- members:
  - `setUp(self)` — [`L160`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L160)
- protocol/private: `__init__`[`L153`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L153)
- uses (calls/refs, reference-scoped): [`ragged_dot`](api.md#ragged_dot), [`RaggedDotMosaicTest`](api_test.md#RaggedDotMosaicTest), [`RaggedDotTritonTest`](api_test.md#RaggedDotTritonTest), [`RaggedDotXlaTest`](api_test.md#RaggedDotXlaTest), [`setUp`](api_test.md#RaggedDotMosaicTest.setUp), [`setUp`](api_test.md#RaggedDotTritonTest.setUp)  (2 test-only)
- used by: [`__init__`](api_test.md#RaggedDotMosaicTest.__init__), [`RaggedDotMosaicTest`](api_test.md#RaggedDotMosaicTest), [`RaggedDotTritonTest`](api_test.md#RaggedDotTritonTest), [`RaggedDotXlaTest`](api_test.md#RaggedDotXlaTest), [`__init__`](api_test.md#RaggedDotTritonTest.__init__), [`__init__`](api_test.md#RaggedDotXlaTest.__init__), [`setUp`](api_test.md#RaggedDotMosaicTest.setUp), [`setUp`](api_test.md#RaggedDotTritonTest.setUp)  (1 test-only)

### `RaggedDotMosaicTest`  ·  implements/extends RaggedDotImplementationTest
- def: [`tokamax/_src/ops/ragged_dot/api_test.py:174`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L174)
- signature: `class RaggedDotMosaicTest(RaggedDotImplementationTest):`
- members:
  - `fn(lhs, rhs, **kwargs)` — [`L182`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L182)
  - `setUp(self)` — [`L207`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L207)
- protocol/private: `__init__`[`L176`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L176), `_dot_fn`[`L205`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L205)
- uses (calls/refs, reference-scoped): [`RaggedDotImplementationTest`](api_test.md#RaggedDotImplementationTest), [`__init__`](api_test.md#RaggedDotImplementationTest.__init__), [`setUp`](api_test.md#RaggedDotImplementationTest.setUp)
- used by: [`RaggedDotImplementationTest`](api_test.md#RaggedDotImplementationTest), [`setUp`](api_test.md#RaggedDotImplementationTest.setUp)

### `RaggedDotTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/ragged_dot/api_test.py:43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L43)
- signature: `class RaggedDotTest(parameterized.TestCase):`
- members:
  - `f(lhs, rhs)` — [`L80`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L80)
  - `f_gt(lhs, rhs)` — [`L92`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L92)
  - `test_basic_api(self, implementation, activation)` — [`L49`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L49)
  - `test_manual_axis_type(self)` — [`L135`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L135)
- uses (calls/refs, reference-scoped): [`op`](../op.md#BoundArguments.op), [`ragged_dot`](api.md#ragged_dot), [`get_opspecs`](../../hlo_utils.md#get_opspecs), [`IMPLEMENTATIONS`](api.md#IMPLEMENTATIONS.IMPLEMENTATIONS), [`_get_input_data`](api_test.md#_get_input_data), [`relu`](api_test.md#relu)

### `RaggedDotTritonTest`  ·  implements/extends RaggedDotImplementationTest
- def: [`tokamax/_src/ops/ragged_dot/api_test.py:213`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L213)
- signature: `class RaggedDotTritonTest(RaggedDotImplementationTest):`
- members:
  - `setUp(self)` — [`L218`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L218)
- protocol/private: `__init__`[`L215`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L215), `_test_bench`[`L224`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L224)
- uses (calls/refs, reference-scoped): [`RaggedDotImplementationTest`](api_test.md#RaggedDotImplementationTest), [`__init__`](api_test.md#RaggedDotImplementationTest.__init__), [`setUp`](api_test.md#RaggedDotImplementationTest.setUp)
- used by: [`RaggedDotImplementationTest`](api_test.md#RaggedDotImplementationTest), [`setUp`](api_test.md#RaggedDotImplementationTest.setUp)  (1 test-only)

### `RaggedDotXlaTest`  ·  implements/extends RaggedDotImplementationTest
- def: [`tokamax/_src/ops/ragged_dot/api_test.py:232`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L232)
- signature: `class RaggedDotXlaTest(RaggedDotImplementationTest):`
- protocol/private: `__init__`[`L234`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L234)
- uses (calls/refs, reference-scoped): [`RaggedDotImplementationTest`](api_test.md#RaggedDotImplementationTest), [`__init__`](api_test.md#RaggedDotImplementationTest.__init__)
- used by: [`RaggedDotImplementationTest`](api_test.md#RaggedDotImplementationTest)

## Functions
- `_get_input_data(num_experts, m, k, n, dtype=jnp.bfloat16)` — [`L30`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L30)
- `relu(x)` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/api_test.py#L40)

