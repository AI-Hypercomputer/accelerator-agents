---
title: 'Module: simply/utils/ragged_paged_attention_test.py'
type: catalog
provenance: extracted
module: simply/utils/ragged_paged_attention_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.ragged_paged_attention_test`/
symbols:
  SamplingStateTest.test_continue_decode: SamplingStateTest#test_continue_decode().
  DecodeStateTest.test_update_decode_state_and_compute_attn: DecodeStateTest#test_update_decode_state_and_compute_attn().
  SamplingStateTest.test_ragged_issue_tokens: SamplingStateTest#test_ragged_issue_tokens().
  DecodeStateTest.test_release_for_window: DecodeStateTest#test_release_for_window().
  SamplingStateTest.test_push_and_release: SamplingStateTest#test_push_and_release().
  DecodeStateTest.test_allocate: DecodeStateTest#test_allocate().
  DecodeStateTest.test_insert: DecodeStateTest#test_insert().
  SamplingStateTest.forward_fn: SamplingStateTest#forward_fn().
  RaggedArray: RaggedArray.
  DecodeStateTest.test_autotune_block_sizes: DecodeStateTest#test_autotune_block_sizes().
  qkv_attn: qkv_attn().
  DecodeStateTest: DecodeStateTest#
  SamplingStateTest: SamplingStateTest#
---
# Module: [`simply/utils/ragged_paged_attention_test.py`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention_test.py)

## Classes
### `DecodeStateTest`  ·  implements/extends TestCase
- def: [`simply/utils/ragged_paged_attention_test.py:51`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention_test.py#L51)
- signature: `class DecodeStateTest(parameterized.TestCase):`
- members:
  - `test_allocate(self)` — [`L53`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention_test.py#L53)
  - `test_autotune_block_sizes(self)` — [`L269`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention_test.py#L269)
  - `test_insert(self)` — [`L98`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention_test.py#L98)
  - `test_release_for_window(self)` — [`L294`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention_test.py#L294)
  - `test_update_decode_state_and_compute_attn(self, use_partition: bool)` — [`L178`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention_test.py#L178)
- uses (calls/refs, reference-scoped): [`init`](ragged_paged_attention.md#DecodeStateConfig.init), [`update_decode_state_and_compute_attn`](ragged_paged_attention.md#DecodeState.update_decode_state_and_compute_attn), [`data`](common.md#RaggedArray.data), [`lens`](common.md#RaggedArray.lens), [`insert`](ragged_paged_attention.md#DecodeState.insert), [`release_for_window`](ragged_paged_attention.md#DecodeState.release_for_window), [`allocate`](ragged_paged_attention.md#DecodeState.allocate), [`concat`](common.md#RaggedArray.concat), [`kv_lens`](ragged_paged_attention.md#DecodeState.kv_lens), [`get_dtype_packing`](../kernels/ragged_paged_attention.md#get_dtype_packing), [`max_num_pages_per_seq_per_shard`](ragged_paged_attention.md#DecodeState.max_num_pages_per_seq_per_shard), [`page_indices`](ragged_paged_attention.md#DecodeState.page_indices), [`local_num_pages`](ragged_paged_attention.md#DecodeState.local_num_pages), [`pages`](ragged_paged_attention.md#DecodeState.pages), [`max_available_kv_lens`](ragged_paged_attention.md#DecodeState.max_available_kv_lens), [`set_mesh`](sharding.md#set_mesh), [`autotune_block_sizes`](ragged_paged_attention.md#autotune_block_sizes), [`to_numpy_list`](common.md#RaggedArray.to_numpy_list), [`available_page_indices`](ragged_paged_attention.md#DecodeState.available_page_indices), [`available_page_indices_np`](ragged_paged_attention.md#DecodeState.available_page_indices_np), [`num_available_pages`](ragged_paged_attention.md#DecodeState.num_available_pages), [`RaggedArray`](ragged_paged_attention_test.md#RaggedArray), [`DecodeState`](ragged_paged_attention.md#DecodeState), [`batch_size`](ragged_paged_attention.md#DecodeStateConfig.batch_size), [`dtype`](ragged_paged_attention.md#DecodeStateConfig.dtype), [`max_seq_len`](ragged_paged_attention.md#DecodeStateConfig.max_seq_len), [`page_size`](ragged_paged_attention.md#DecodeStateConfig.page_size), [`row`](common.md#RaggedArray.row), [`total_num_pages`](ragged_paged_attention.md#DecodeStateConfig.total_num_pages), [`n_kv_heads`](ragged_paged_attention.md#DecodeStateConfig.n_kv_heads), [`per_head_dim`](ragged_paged_attention.md#DecodeStateConfig.per_head_dim), [`DecodeStateConfig`](ragged_paged_attention.md#DecodeStateConfig), [`page_indices_nplist`](ragged_paged_attention.md#DecodeState.page_indices_nplist), [`seq_partition`](ragged_paged_attention.md#DecodeStateConfig.seq_partition), [`kv_nplist`](ragged_paged_attention.md#DecodeState.kv_nplist), [`head_partition`](ragged_paged_attention.md#DecodeStateConfig.head_partition), [`max_seq_len`](ragged_paged_attention.md#DecodeState.max_seq_len), [`window_size`](ragged_paged_attention.md#DecodeStateConfig.window_size), [`get_tpu_version`](../kernels/ragged_paged_attention.md#get_tpu_version), [`qkv_attn`](ragged_paged_attention_test.md#qkv_attn)

### `SamplingStateTest`  ·  implements/extends TestCase
- def: [`simply/utils/ragged_paged_attention_test.py:321`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention_test.py#L321)
- signature: `class SamplingStateTest(parameterized.TestCase):`
- members:
  - `forward_fn(params: common.PyTree, tokens: jax.Array, segment_ids: jax.Array, segment_positions: jax.Array, extra_inputs: common.PyTree = None, decode_state: common.PyTree = None, ragged: bool = True)` — [`L507`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention_test.py#L507)
  - `test_continue_decode(self, use_partition: bool)` — [`L448`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention_test.py#L448)
  - `test_push_and_release(self)` — [`L323`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention_test.py#L323)
  - `test_ragged_issue_tokens(self)` — [`L374`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention_test.py#L374)
- uses (calls/refs, reference-scoped): [`PyTree`](common.md#PyTree.PyTree), [`init`](ragged_paged_attention.md#DecodeStateConfig.init), [`data`](common.md#RaggedArray.data), [`lens`](common.md#RaggedArray.lens), [`update_with_ragged_output`](ragged_paged_attention.md#SamplingState.update_with_ragged_output), [`push`](ragged_paged_attention.md#SamplingState.push), [`has_ended`](ragged_paged_attention.md#SamplingState.has_ended), [`issue_lens`](ragged_paged_attention.md#SamplingState.issue_lens), [`ragged_issue_tokens`](ragged_paged_attention.md#SamplingState.ragged_issue_tokens), [`get`](ragged_paged_attention.md#SamplingState.get), [`SamplingState`](ragged_paged_attention.md#SamplingState), [`continue_decode`](ragged_paged_attention.md#SamplingState.continue_decode), [`compute_log_likelihood`](sampling_lib.md#compute_log_likelihood), [`tokens`](ragged_paged_attention.md#SamplingState.tokens), [`position`](ragged_paged_attention.md#SamplingState.position), [`create`](ragged_paged_attention.md#SamplingState.create), [`from_numpy_list`](common.md#RaggedArray.from_numpy_list), [`is_pad_seq`](ragged_paged_attention.md#SamplingState.is_pad_seq), [`set_mesh`](sharding.md#set_mesh), [`release`](ragged_paged_attention.md#SamplingState.release), [`to_numpy_list`](common.md#RaggedArray.to_numpy_list), [`max_seq_len`](ragged_paged_attention.md#SamplingState.max_seq_len), [`input_lens`](ragged_paged_attention.md#SamplingState.input_lens), [`to_padded_dense`](common.md#RaggedArray.to_padded_dense), [`RaggedArray`](ragged_paged_attention_test.md#RaggedArray), [`batch_size`](ragged_paged_attention.md#DecodeStateConfig.batch_size), [`dtype`](ragged_paged_attention.md#DecodeStateConfig.dtype), [`lens`](ragged_paged_attention.md#SamplingState.lens), [`max_seq_len`](ragged_paged_attention.md#DecodeStateConfig.max_seq_len), [`page_size`](ragged_paged_attention.md#DecodeStateConfig.page_size), [`row`](common.md#RaggedArray.row), [`total_num_pages`](ragged_paged_attention.md#DecodeStateConfig.total_num_pages), [`n_kv_heads`](ragged_paged_attention.md#DecodeStateConfig.n_kv_heads), [`per_head_dim`](ragged_paged_attention.md#DecodeStateConfig.per_head_dim), [`DecodeStateConfig`](ragged_paged_attention.md#DecodeStateConfig), [`rank`](ragged_paged_attention.md#SamplingState.rank), [`seq_partition`](ragged_paged_attention.md#DecodeStateConfig.seq_partition), [`head_partition`](ragged_paged_attention.md#DecodeStateConfig.head_partition), [`get_tpu_version`](../kernels/ragged_paged_attention.md#get_tpu_version), [`qkv_attn`](ragged_paged_attention_test.md#qkv_attn)

## Functions
- `qkv_attn(q: jax.Array, k: jax.Array, v: jax.Array)` — [`L33`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention_test.py#L33) — Computes qkv attention (reference implementation).

## Module values
- `RaggedArray` — [`L30`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention_test.py#L30)

