---
title: 'Module: axlearn/common/flash_attention/utils_test.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/utils_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.utils_test`/
symbols:
  setUpModule: setUpModule().
  _get_inputs: _get_inputs().
  _get_paged_inputs: _get_paged_inputs().
  TestFlashAttention: TestFlashAttention#
  TestFlashAttention._TEST_CONFIGS: TestFlashAttention#_TEST_CONFIGS.
  TestFlashAttention.test_forward: TestFlashAttention#test_forward().
  TestFlashAttention.test_decoding: TestFlashAttention#test_decoding().
  TestFlashAttention.test_incremental_prefill: TestFlashAttention#test_incremental_prefill().
---
# Module: [`axlearn/common/flash_attention/utils_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/utils_test.py)

## Classes
### `TestFlashAttention`
- def: [`axlearn/common/flash_attention/utils_test.py:103`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/utils_test.py#L103)
- doc: Tests FlashAttention layer.
- signature: `class TestFlashAttention(TestCase):`
- members:
  - `test_decoding(self, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names, backend, bias_type, input_dtype, step_len, page_size)` — [`L215`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/utils_test.py#L215)
  - `test_forward(self, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names, backend, bias_type, input_dtype)` — [`L136`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/utils_test.py#L136)
  - `test_incremental_prefill(self, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names, backend, bias_type, input_dtype, prefix_len, suffix_len)` — [`L361`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/utils_test.py#L361) — Tests that incremental prefill (asymmetric Q/K with offset) matches forward pass.
- protocol/private: `_TEST_CONFIGS`[`L106`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/utils_test.py#L106)

## Functions
- `_get_inputs(*, batch: int, seq_len: int, num_heads: int, num_kv_heads: int, per_head_dim: int, input_dtype: jnp.dtype=jnp.bfloat16)` — [`L42`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/utils_test.py#L42)
- `_get_paged_inputs(*, batch: int, seq_len: int, num_heads: int, num_kv_heads: int, per_head_dim: int, page_size: int, input_dtype: jnp.dtype=jnp.bfloat16)` — [`L69`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/utils_test.py#L69)
- `setUpModule()` — [`L34`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/utils_test.py#L34)

