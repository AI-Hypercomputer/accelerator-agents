---
title: 'Module: simply/utils/sampling_lib_test.py'
type: catalog
provenance: extracted
module: simply/utils/sampling_lib_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.sampling_lib_test`/SamplingLibTest#
symbols:
  SamplingLibTest.test_sampling_params: test_sampling_params().
  SamplingLibTest.test_decoding_schedule: test_decoding_schedule().
  SamplingLibTest.test_chunk_dump_and_load: test_chunk_dump_and_load().
  SamplingLibTest.test_sample_from_logits: test_sample_from_logits().
  SamplingLibTest.test_processed_input_batching: test_processed_input_batching().
  SamplingLibTest.test_compute_log_likelihood: test_compute_log_likelihood().
  SamplingLibTest.test_sample_fused_top_k_top_p_properties: test_sample_fused_top_k_top_p_properties().
  SamplingLibTest.test_log_likelihood_matches_sample_logprobs: test_log_likelihood_matches_sample_logprobs().
  SamplingLibTest.test_top_k_mask: test_top_k_mask().
  SamplingLibTest.test_top_p_mask: test_top_p_mask().
  SamplingLibTest.test_top_k_mask_matches_argsort: test_top_k_mask_matches_argsort().
  SamplingLibTest.test_top_k_mask_k_equals_vocab_size: test_top_k_mask_k_equals_vocab_size().
  SamplingLibTest.test_top_k_mask_k_exceeds_vocab_size: test_top_k_mask_k_exceeds_vocab_size().
  SamplingLibTest.test_sample_from_logits_all_modes: test_sample_from_logits_all_modes().
  SamplingLibTest.test_greedy_returns_argmax: test_greedy_returns_argmax().
  SamplingLibTest.test_top_k_1_returns_argmax: test_top_k_1_returns_argmax().
  SamplingLibTest.test_top_k_mask_large_vocab: test_top_k_mask_large_vocab().
  SamplingLibTest.test_same_seed_same_result: test_same_seed_same_result().
  SamplingLibTest: ''
---
# Module: [`simply/utils/sampling_lib_test.py`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py)

## Classes
### `SamplingLibTest`  ·  implements/extends TestCase
- def: [`simply/utils/sampling_lib_test.py:24`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L24)
- signature: `class SamplingLibTest(parameterized.TestCase):`
- members:
  - `test_chunk_dump_and_load(self)` — [`L89`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L89)
  - `test_compute_log_likelihood(self)` — [`L154`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L154)
  - `test_decoding_schedule(self)` — [`L26`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L26)
  - `test_greedy_returns_argmax(self)` — [`L257`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L257)
  - `test_log_likelihood_matches_sample_logprobs(self, temperature, top_k, top_p)` — [`L243`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L243)
  - `test_processed_input_batching(self)` — [`L52`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L52)
  - `test_same_seed_same_result(self)` — [`L276`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L276)
  - `test_sample_from_logits(self)` — [`L138`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L138)
  - `test_sample_from_logits_all_modes(self, temperature, top_k, top_p)` — [`L223`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L223)
  - `test_sample_fused_top_k_top_p_properties(self, top_k, top_p)` — [`L199`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L199) — Sampled tokens from fused top-k+top-p are within the top-k set.
  - `test_sampling_params(self)` — [`L38`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L38)
  - `test_top_k_1_returns_argmax(self)` — [`L264`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L264)
  - `test_top_k_mask(self, top_k: int)` — [`L102`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L102)
  - `test_top_k_mask_k_equals_vocab_size(self)` — [`L182`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L182)
  - `test_top_k_mask_k_exceeds_vocab_size(self)` — [`L188`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L188)
  - `test_top_k_mask_large_vocab(self)` — [`L271`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L271)
  - `test_top_k_mask_matches_argsort(self, top_k: int)` — [`L167`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L167) — Verify top_k_mask matches a reference argsort-based implementation.
  - `test_top_p_mask(self, top_p: float)` — [`L123`](../../../../../../raw/code/simply/simply/utils/sampling_lib_test.py#L123)
- uses (calls/refs, reference-scoped): [`get_decoding_schedule`](sampling_lib.md#SamplingParams.get_decoding_schedule), [`Chunk`](sampling_lib.md#Chunk), [`max_decode_steps`](sampling_lib.md#SamplingParams.max_decode_steps), [`ProcessedInput`](sampling_lib.md#ProcessedInput), [`compute_log_likelihood`](sampling_lib.md#compute_log_likelihood), [`get_next_length`](sampling_lib.md#DecodingSchedule.get_next_length), [`sample_from_logits`](sampling_lib.md#sample_from_logits), [`top_k_mask`](sampling_lib.md#top_k_mask), [`load_pytree_from`](pytree.md#load_pytree_from), [`save_pytree_to`](pytree.md#save_pytree_to), [`prefill_size`](sampling_lib.md#DecodingSchedule.prefill_size), [`ProcessedInputBatch`](sampling_lib.md#ProcessedInputBatch), [`intermediate_decode_steps`](sampling_lib.md#SamplingParams.intermediate_decode_steps), [`begin_position`](sampling_lib.md#DecodingSchedule.begin_position), [`chunk_size`](sampling_lib.md#DecodingSchedule.chunk_size), [`end_position`](sampling_lib.md#DecodingSchedule.end_position), [`top_p_mask`](sampling_lib.md#top_p_mask), [`DecodingSchedule`](sampling_lib.md#DecodingSchedule), [`SamplingParams`](sampling_lib.md#SamplingParams)

