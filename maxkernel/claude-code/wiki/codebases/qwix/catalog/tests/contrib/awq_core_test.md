---
title: 'Module: tests/contrib/awq_core_test.py'
type: catalog
provenance: extracted
module: tests/contrib/awq_core_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests.contrib.awq_core_test`/
symbols:
  AwqCoreTest.test_quantize_weight: AwqCoreTest#test_quantize_weight().
  rel_rmse: rel_rmse().
  AwqCoreTest: AwqCoreTest#
---
# Module: [`tests/contrib/awq_core_test.py`](../../../../../../raw/code/qwix/tests/contrib/awq_core_test.py)

## Classes
### `AwqCoreTest`  ·  implements/extends TestCase
- def: [`tests/contrib/awq_core_test.py:30`](../../../../../../raw/code/qwix/tests/contrib/awq_core_test.py#L30)
- signature: `class AwqCoreTest(parameterized.TestCase):`
- members:
  - `test_quantize_weight(self, groupsize)` — [`L42`](../../../../../../raw/code/qwix/tests/contrib/awq_core_test.py#L42)
- uses (calls/refs, reference-scoped): [`HowToQuantize`](../../qwix/_src/core/qarray.md#HowToQuantize), [`qtype`](../../qwix/_src/core/qarray.md#HowToQuantize.qtype), [`quantize`](../../qwix/_src/core/qarray.md#quantize), [`dequantize`](../../qwix/_src/core/qarray.md#dequantize), [`tiled_axes`](../../qwix/_src/core/qarray.md#HowToQuantize.tiled_axes), [`channelwise_axes`](../../qwix/_src/core/qarray.md#HowToQuantize.channelwise_axes), [`quantize_weight`](../../qwix/contrib/awq_core.md#quantize_weight), [`compute_act_scale`](../../qwix/contrib/awq_core.md#compute_act_scale)  (1 test-only)

## Functions
- `rel_rmse(x: jax.Array, y: jax.Array)` — [`L26`](../../../../../../raw/code/qwix/tests/contrib/awq_core_test.py#L26)

