---
title: 'Module: tokamax/_src/ops/flex_attention/test_base.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/flex_attention/test_base.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.flex_attention.test_base`/FlexAttentionTestBase#
symbols:
  FlexAttentionTestBase.test_bias_upscale: test_bias_upscale().
  FlexAttentionTestBase: ''
  FlexAttentionTestBase.test_bias_relative_positional: test_bias_relative_positional().
  FlexAttentionTestBase.score_mod: score_mod().
  FlexAttentionTestBase.__init__: __init__().
  FlexAttentionTestBase._flex_attn: _flex_attn.
  FlexAttentionTestBase.upscale_bias: upscale_bias().
---
# Module: [`tokamax/_src/ops/flex_attention/test_base.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/test_base.py)

## Classes
### `FlexAttentionTestBase`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/flex_attention/test_base.py:25`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/test_base.py#L25)
- doc: Base class for FlexAttention tests.
- signature: `class FlexAttentionTestBase(absltest.TestCase):`
- members:
  - `score_mod(scores)` — [`L50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/test_base.py#L50)
  - `test_bias_relative_positional(self)` — [`L56`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/test_base.py#L56)
  - `test_bias_upscale(self)` — [`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/test_base.py#L36)
  - `upscale_bias(x)` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/test_base.py#L43)
- protocol/private: `__init__`[`L32`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/test_base.py#L32), `_flex_attn`[`L34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/flex_attention/test_base.py#L34)
- uses (calls/refs, reference-scoped): [`FlexAttentionTest`](base_test.md#FlexAttentionTest), [`PallasTritonFlexAttentionTest`](pallas_triton_test.md#PallasTritonFlexAttentionTest)
- used by: [`__init__`](base_test.md#FlexAttentionTest.__init__), [`__init__`](pallas_triton_test.md#PallasTritonFlexAttentionTest.__init__), [`FlexAttentionTest`](base_test.md#FlexAttentionTest), [`PallasTritonFlexAttentionTest`](pallas_triton_test.md#PallasTritonFlexAttentionTest)

