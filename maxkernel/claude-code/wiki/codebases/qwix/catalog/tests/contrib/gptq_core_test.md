---
title: 'Module: tests/contrib/gptq_core_test.py'
type: catalog
provenance: extracted
module: tests/contrib/gptq_core_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests.contrib.gptq_core_test`/
symbols:
  GptqCoreTest.test_quantize_weight: GptqCoreTest#test_quantize_weight().
  GptqCoreTest.test_quantize_weight_defaults_to_per_channel: GptqCoreTest#test_quantize_weight_defaults_to_per_channel().
  rel_rmse: rel_rmse().
  GptqCoreTest: GptqCoreTest#
---
# Module: [`tests/contrib/gptq_core_test.py`](../../../../../../raw/code/qwix/tests/contrib/gptq_core_test.py)

## Classes
### `GptqCoreTest`  ·  implements/extends TestCase
- def: [`tests/contrib/gptq_core_test.py:30`](../../../../../../raw/code/qwix/tests/contrib/gptq_core_test.py#L30)
- signature: `class GptqCoreTest(parameterized.TestCase):`
- members:
  - `test_quantize_weight(self, groupsize, blocksize)` — [`L49`](../../../../../../raw/code/qwix/tests/contrib/gptq_core_test.py#L49)
  - `test_quantize_weight_defaults_to_per_channel(self)` — [`L86`](../../../../../../raw/code/qwix/tests/contrib/gptq_core_test.py#L86) — Test GPTQ produces valid QArray when columns isn't divisible by rows.
- uses (calls/refs, reference-scoped): [`HowToQuantize`](../../qwix/_src/core/qarray.md#HowToQuantize), [`qtype`](../../qwix/_src/core/qarray.md#HowToQuantize.qtype), [`quantize`](../../qwix/_src/core/qarray.md#quantize), [`qvalue`](../../qwix/_src/core/qarray.md#QArray.qvalue), [`scale`](../../qwix/_src/core/qarray.md#QArray.scale), [`dequantize`](../../qwix/_src/core/qarray.md#dequantize), [`tiled_axes`](../../qwix/_src/core/qarray.md#HowToQuantize.tiled_axes), [`channelwise_axes`](../../qwix/_src/core/qarray.md#HowToQuantize.channelwise_axes), [`quantize_weight`](../../qwix/contrib/gptq_core.md#quantize_weight), [`shape`](../../qwix/_src/core/qarray.md#QArray.shape), [`compute_hessian`](../../qwix/contrib/gptq_core.md#compute_hessian)  (1 test-only)

## Functions
- `rel_rmse(x: jax.Array, y: jax.Array)` — [`L26`](../../../../../../raw/code/qwix/tests/contrib/gptq_core_test.py#L26)

