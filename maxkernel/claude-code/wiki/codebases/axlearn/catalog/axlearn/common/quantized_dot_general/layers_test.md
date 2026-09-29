---
title: 'Module: axlearn/common/quantized_dot_general/layers_test.py'
type: catalog
provenance: extracted
module: axlearn/common/quantized_dot_general/layers_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.quantized_dot_general.layers_test`/TestQuantizedDotGeneral#
symbols:
  TestQuantizedDotGeneral: ''
  TestQuantizedDotGeneral.test_einsum_maybe_quantized: test_einsum_maybe_quantized().
  TestQuantizedDotGeneral.test_set_quantized_dot_general_recursively: test_set_quantized_dot_general_recursively().
  TestQuantizedDotGeneral.test_einsum_maybe_quantized_fp8: test_einsum_maybe_quantized_fp8().
---
# Module: [`axlearn/common/quantized_dot_general/layers_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers_test.py)

## Classes
### `TestQuantizedDotGeneral`
- def: [`axlearn/common/quantized_dot_general/layers_test.py:22`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers_test.py#L22)
- doc: Tests QuantizedDotGeneral layer.
- signature: `class TestQuantizedDotGeneral(TestCase):`
- members:
  - `test_einsum_maybe_quantized(self, b, d, h, quantization_type_and_assert_output)` — [`L38`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers_test.py#L38)
  - `test_einsum_maybe_quantized_fp8(self, b, d, h)` — [`L103`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers_test.py#L103)
  - `test_set_quantized_dot_general_recursively(self)` — [`L76`](../../../../../../../raw/code/axlearn/axlearn/common/quantized_dot_general/layers_test.py#L76)

