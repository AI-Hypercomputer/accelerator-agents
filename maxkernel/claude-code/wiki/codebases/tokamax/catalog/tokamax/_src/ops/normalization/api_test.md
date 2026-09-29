---
title: 'Module: tokamax/_src/ops/normalization/api_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/normalization/api_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.normalization.api_test`/
symbols:
  LayerNormTest.test_basic_api: LayerNormTest#test_basic_api().
  LayerNormTritonTest.__init__: LayerNormTritonTest#__init__().
  LayerNormXlATest.__init__: LayerNormXlATest#__init__().
  LayerNormTest.norm_fn: LayerNormTest#norm_fn().
  _IMPLEMENTATIONS_VJP._IMPLEMENTATIONS_VJP: _IMPLEMENTATIONS_VJP._IMPLEMENTATIONS_VJP.
  FLAGS: FLAGS.
  LayerNormTest.norm_fn_ref: LayerNormTest#norm_fn_ref().
  _IMPLEMENTATIONS._IMPLEMENTATIONS: _IMPLEMENTATIONS._IMPLEMENTATIONS.
  LayerNormTritonTest: LayerNormTritonTest#
  LayerNormXlATest: LayerNormXlATest#
  _get_input_data: _get_input_data().
  LayerNormTritonTest.IMPL: LayerNormTritonTest#IMPL.
  LayerNormXlATest.IMPL: LayerNormXlATest#IMPL.
  LayerNormTest: LayerNormTest#
---
# Module: [`tokamax/_src/ops/normalization/api_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/api_test.py)

## Classes
### `LayerNormTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/normalization/api_test.py:55`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/api_test.py#L55)
- signature: `class LayerNormTest(parameterized.TestCase):`
- members:
  - `norm_fn(x, scale, offset)` — [`L66`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/api_test.py#L66)
  - `norm_fn_ref(x, scale, offset)` — [`L72`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/api_test.py#L72)
  - `test_basic_api(self, implementation)` — [`L58`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/api_test.py#L58)
- uses (calls/refs, reference-scoped): [`op`](../op.md#BoundArguments.op), [`get_opspecs`](../../hlo_utils.md#get_opspecs), [`layer_norm`](api.md#layer_norm), [`IMPLEMENTATIONS`](api.md#IMPLEMENTATIONS.IMPLEMENTATIONS), [`_IMPLEMENTATIONS_VJP`](api_test.md#_IMPLEMENTATIONS_VJP._IMPLEMENTATIONS_VJP), [`_IMPLEMENTATIONS`](api_test.md#_IMPLEMENTATIONS._IMPLEMENTATIONS), [`DISABLE_JAX_EXPORT_CHECKS`](../../hlo_utils.md#DISABLE_JAX_EXPORT_CHECKS.DISABLE_JAX_EXPORT_CHECKS), [`_get_input_data`](api_test.md#_get_input_data)

### `LayerNormTritonTest`  ·  implements/extends NormalizationTestBase
- def: [`tokamax/_src/ops/normalization/api_test.py:125`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/api_test.py#L125)
- signature: `class LayerNormTritonTest(test_base.NormalizationTestBase):`
- members:
  - `IMPL` — [`L126`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/api_test.py#L126)
- protocol/private: `__init__`[`L128`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/api_test.py#L128)
- uses (calls/refs, reference-scoped): [`layer_norm`](api.md#layer_norm)  (2 test-only)
- used by: (1 test-only callers)

### `LayerNormXlATest`  ·  implements/extends NormalizationTestBase
- def: [`tokamax/_src/ops/normalization/api_test.py:137`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/api_test.py#L137)
- signature: `class LayerNormXlATest(test_base.NormalizationTestBase):`
- members:
  - `IMPL` — [`L138`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/api_test.py#L138)
- protocol/private: `__init__`[`L140`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/api_test.py#L140)
- uses (calls/refs, reference-scoped): [`layer_norm`](api.md#layer_norm)  (2 test-only)
- used by: (1 test-only callers)

## Functions
- `_get_input_data(shape, dtype=jnp.bfloat16)` — [`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/api_test.py#L47)

## Module values
- `FLAGS` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/api_test.py#L43)
- `_IMPLEMENTATIONS` — [`L32`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/api_test.py#L32)
- `_IMPLEMENTATIONS_VJP` — [`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/api_test.py#L36)

