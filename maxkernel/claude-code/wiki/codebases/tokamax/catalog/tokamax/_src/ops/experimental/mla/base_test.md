---
title: 'Module: tokamax/_src/ops/experimental/mla/base_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/experimental/mla/base_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.experimental.mla.base_test`/BaselineMlaTest#
symbols:
  BaselineMlaTest.test_reference_running_correctly: test_reference_running_correctly().
  BaselineMlaTest.test_shape_mismatch_validation: test_shape_mismatch_validation().
  BaselineMlaTest.test_page_indices_divisibility_validation: test_page_indices_divisibility_validation().
  BaselineMlaTest.test_cu_q_lens_shape_validation: test_cu_q_lens_shape_validation().
  BaselineMlaTest: ''
---
# Module: [`tokamax/_src/ops/experimental/mla/base_test.py`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/base_test.py)

## Classes
### `BaselineMlaTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/experimental/mla/base_test.py:29`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/base_test.py#L29)
- signature: `class BaselineMlaTest(parameterized.TestCase):`
- members:
  - `test_cu_q_lens_shape_validation(self)` — [`L188`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/base_test.py#L188)
  - `test_page_indices_divisibility_validation(self)` — [`L160`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/base_test.py#L160)
  - `test_reference_running_correctly(self, seq_lens_list, num_heads, lkv_dim, r_dim, page_size, q_dtype, kv_dtype, output_lkv_dim)` — [`L55`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/base_test.py#L55)
  - `test_shape_mismatch_validation(self)` — [`L129`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/base_test.py#L129)
- uses (calls/refs, reference-scoped): [`MultiHeadLatentAttention`](base.md#MultiHeadLatentAttention), [`generate_mla_inputs`](utils.md#generate_mla_inputs), [`bind`](base.md#MultiHeadLatentAttention.bind), [`mla_attention`](reference.md#mla_attention), [`cdiv`](utils.md#cdiv)

