---
title: 'Module: simply/model_lib_test.py'
type: catalog
provenance: extracted
module: simply/model_lib_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.model_lib_test`/
symbols:
  ModelLibTest.test_batch_sampling_with_max_seq_len: ModelLibTest#test_batch_sampling_with_max_seq_len().
  ModelLibTest.tfm_lm: ModelLibTest#tfm_lm.
  ModelLibTest.test_sampling_token_scores: ModelLibTest#test_sampling_token_scores().
  ModelLibTest.test_lm_interface_generate_with_local_state: ModelLibTest#test_lm_interface_generate_with_local_state().
  ModelLibTest.test_batch_sampling: ModelLibTest#test_batch_sampling().
  ModelLibTest.test_lm_interface_generate_without_scoring: ModelLibTest#test_lm_interface_generate_without_scoring().
  ModelLibTest.test_continue_decoding: ModelLibTest#test_continue_decoding().
  ModelLibTest.test_sampling_output_logprobs: ModelLibTest#test_sampling_output_logprobs().
  ModelLibTest.test_sampling_max_decode_steps_equals_prefill_size: ModelLibTest#test_sampling_max_decode_steps_equals_prefill_size().
  ModelLibTest.test_lm_interface_generate: ModelLibTest#test_lm_interface_generate().
  ModelLibTest.test_lm_interface_batch: ModelLibTest#test_lm_interface_batch().
  ModelLibTest.test_lm_interface_score: ModelLibTest#test_lm_interface_score().
  ModelLibTest.test_lm_interface_score_tokens: ModelLibTest#test_lm_interface_score_tokens().
  MoETest.test_pipelined_moe_feed_forward_equivalence: MoETest#test_pipelined_moe_feed_forward_equivalence().
  lm_test: lm_test().
  MoETest.test_moe_feed_forward_equivalence: MoETest#test_moe_feed_forward_equivalence().
  ModelLibTest.config: ModelLibTest#config.
  ModelLibTest.test_grad_accumulation: ModelLibTest#test_grad_accumulation().
  ModelLibTest.test_backward_pass: ModelLibTest#test_backward_pass().
  ModelLibTest.test_quantization: ModelLibTest#test_quantization().
  ModelLibTest.test_expand_dim_in_ffn: ModelLibTest#test_expand_dim_in_ffn().
  TestNpArrayQuantizer.encode: TestNpArrayQuantizer#encode().
  simple_moe: simple_moe().
  ModelLibTest.test_dim_annotation: ModelLibTest#test_dim_annotation().
  ModelLibTest.test_mixed_precision: ModelLibTest#test_mixed_precision().
  ModelLibTest.test_chunked_local_attn: ModelLibTest#test_chunked_local_attn().
  ModelLibTest.test_logits_with_kv_cache: ModelLibTest#test_logits_with_kv_cache().
  ModelLibTest.test_dump_load: ModelLibTest#test_dump_load().
  ModelLibTest.test_n_blocks: ModelLibTest#test_n_blocks().
  TestNpArrayQuantizer.__init__: TestNpArrayQuantizer#__init__().
  TestNpArrayQuantizer.unk_id: TestNpArrayQuantizer#unk_id.
  ModelLibTest.test_forward_pass: ModelLibTest#test_forward_pass().
  ModelLibTest.test_identical_inputs_run_to_run_consistency: ModelLibTest#test_identical_inputs_run_to_run_consistency().
  ModelLibTest.test_identitical_inputs_batch_consistency: ModelLibTest#test_identitical_inputs_batch_consistency().
  ModelLibTest.test_segment_info: ModelLibTest#test_segment_info().
  ModelLibTest.test_get_scaling_info: ModelLibTest#test_get_scaling_info().
  TestNpArrayQuantizer.decode: TestNpArrayQuantizer#decode().
  ModelLibTest.train_one_step_fn: ModelLibTest#train_one_step_fn().
  ModelLibTest.compute_logits_decoding_from_k: ModelLibTest#compute_logits_decoding_from_k().
  ModelLibTest.train_one_step_fn_grad_accum: ModelLibTest#train_one_step_fn_grad_accum().
  TestNpArrayQuantizer._rev_vocab_dict: TestNpArrayQuantizer#_rev_vocab_dict.
  MoETest.check_error: MoETest#check_error().
  TestNpArrayQuantizer.feature_dim: TestNpArrayQuantizer#feature_dim.
  ModelLibTest.test_replace_embeddings: ModelLibTest#test_replace_embeddings().
  ModelLibTest.test_create_mask: ModelLibTest#test_create_mask().
  ModelLibTest.test_create_mask_with_window_size: ModelLibTest#test_create_mask_with_window_size().
  ModelLibTest.test_create_mask_decode: ModelLibTest#test_create_mask_decode().
  ModelLibTest.test_tree_norm: ModelLibTest#test_tree_norm().
  ModelLibTest.test_flatten_dict: ModelLibTest#test_flatten_dict().
  TestNpArrayQuantizer: TestNpArrayQuantizer#
  TestNpArrayQuantizer.vocab_size: TestNpArrayQuantizer#vocab_size.
  TestNpArrayQuantizer._vocab_dict: TestNpArrayQuantizer#_vocab_dict.
  MoETest.vector_norm_error: MoETest#vector_norm_error().
  MoETest.loss1: MoETest#loss1().
  MoETest.loss2: MoETest#loss2().
  ModelLibTest: ModelLibTest#
  ModelLibTest.setUp: ModelLibTest#setUp().
  MoETest: MoETest#
---
# Module: [`simply/model_lib_test.py`](../../../../../raw/code/simply/simply/model_lib_test.py)

## Classes
### `MoETest`  ·  implements/extends TestCase
- def: [`simply/model_lib_test.py:1287`](../../../../../raw/code/simply/simply/model_lib_test.py#L1287)
- signature: `class MoETest(parameterized.TestCase):`
- members:
  - `check_error(g1, g2)` — [`L1511`](../../../../../raw/code/simply/simply/model_lib_test.py#L1511)
  - `loss1(params, inputs, inputs_mask)` — [`L1401`](../../../../../raw/code/simply/simply/model_lib_test.py#L1401)
  - `loss2(params, inputs, inputs_mask)` — [`L1405`](../../../../../raw/code/simply/simply/model_lib_test.py#L1405)
  - `test_moe_feed_forward_equivalence(self, use_gated_activation_in_ffn, num_experts, ep_capacity_factor, num_experts_per_token=2, activation_dtype='bfloat16', ep_method='ra2a')` — [`L1354`](../../../../../raw/code/simply/simply/model_lib_test.py#L1354)
  - `test_pipelined_moe_feed_forward_equivalence(self, ep_method, ep_pipeline_fine_grained_ra2a, ep_capacity_factor, use_gated_activation_in_ffn=False, num_experts=8, num_experts_per_token=2, activation_dtype='bfloat16')` — [`L1448`](../../../../../raw/code/simply/simply/model_lib_test.py#L1448)
  - `vector_norm_error(self, a, b, axis=-1, reduce='max')` — [`L1288`](../../../../../raw/code/simply/simply/model_lib_test.py#L1288)
- uses (calls/refs, reference-scoped): [`moe_sharding`](config_lib.md#moe_sharding), [`MoEFeedForward`](model_lib.md#MoEFeedForward), [`simple_moe`](model_lib_test.md#simple_moe), [`set_default_mesh_shape`](utils/sharding.md#set_default_mesh_shape)

### `ModelLibTest`  ·  implements/extends TestCase
- def: [`simply/model_lib_test.py:76`](../../../../../raw/code/simply/simply/model_lib_test.py#L76)
- signature: `class ModelLibTest(parameterized.TestCase):`
- members:
  - `compute_logits_decoding_from_k(model, params, k)` — [`L569`](../../../../../raw/code/simply/simply/model_lib_test.py#L569)
  - `setUp(self)` — [`L78`](../../../../../raw/code/simply/simply/model_lib_test.py#L78)
  - `test_backward_pass(self)` — [`L181`](../../../../../raw/code/simply/simply/model_lib_test.py#L181)
  - `test_batch_sampling(self)` — [`L1000`](../../../../../raw/code/simply/simply/model_lib_test.py#L1000)
  - `test_batch_sampling_with_max_seq_len(self)` — [`L1056`](../../../../../raw/code/simply/simply/model_lib_test.py#L1056)
  - `test_chunked_local_attn(self)` — [`L531`](../../../../../raw/code/simply/simply/model_lib_test.py#L531)
  - `test_continue_decoding(self)` — [`L761`](../../../../../raw/code/simply/simply/model_lib_test.py#L761)
  - `test_create_mask(self)` — [`L437`](../../../../../raw/code/simply/simply/model_lib_test.py#L437)
  - `test_create_mask_decode(self)` — [`L518`](../../../../../raw/code/simply/simply/model_lib_test.py#L518)
  - `test_create_mask_with_window_size(self)` — [`L461`](../../../../../raw/code/simply/simply/model_lib_test.py#L461)
  - `test_dim_annotation(self)` — [`L330`](../../../../../raw/code/simply/simply/model_lib_test.py#L330)
  - `test_dump_load(self)` — [`L1129`](../../../../../raw/code/simply/simply/model_lib_test.py#L1129)
  - `test_expand_dim_in_ffn(self, ffn_expand_dim: int)` — [`L1163`](../../../../../raw/code/simply/simply/model_lib_test.py#L1163)
  - `test_flatten_dict(self)` — [`L610`](../../../../../raw/code/simply/simply/model_lib_test.py#L610)
  - `test_forward_pass(self)` — [`L83`](../../../../../raw/code/simply/simply/model_lib_test.py#L83)
  - `test_get_scaling_info(self)` — [`L1172`](../../../../../raw/code/simply/simply/model_lib_test.py#L1172)
  - `test_grad_accumulation(self)` — [`L244`](../../../../../raw/code/simply/simply/model_lib_test.py#L244)
  - `test_identical_inputs_run_to_run_consistency(self)` — [`L130`](../../../../../raw/code/simply/simply/model_lib_test.py#L130) — Tests for deterministic outputs across multiple runs with identical inputs.
  - `test_identitical_inputs_batch_consistency(self)` — [`L153`](../../../../../raw/code/simply/simply/model_lib_test.py#L153) — Tests that identical inputs yield identical outputs regardless of batch position.
  - `test_lm_interface_batch(self)` — [`L636`](../../../../../raw/code/simply/simply/model_lib_test.py#L636)
  - `test_lm_interface_generate(self)` — [`L616`](../../../../../raw/code/simply/simply/model_lib_test.py#L616)
  - `test_lm_interface_generate_with_local_state(self, use_scan, prefill_size)` — [`L697`](../../../../../raw/code/simply/simply/model_lib_test.py#L697)
  - `test_lm_interface_generate_without_scoring(self)` — [`L666`](../../../../../raw/code/simply/simply/model_lib_test.py#L666)
  - `test_lm_interface_score(self)` — [`L732`](../../../../../raw/code/simply/simply/model_lib_test.py#L732)
  - `test_lm_interface_score_tokens(self)` — [`L746`](../../../../../raw/code/simply/simply/model_lib_test.py#L746)
  - `test_logits_with_kv_cache(self, use_scan)` — [`L565`](../../../../../raw/code/simply/simply/model_lib_test.py#L565)
  - `test_mixed_precision(self, activation_dtype_name)` — [`L396`](../../../../../raw/code/simply/simply/model_lib_test.py#L396)
  - `test_n_blocks(self)` — [`L1179`](../../../../../raw/code/simply/simply/model_lib_test.py#L1179)
  - `test_quantization(self, activation_dtype_name)` — [`L420`](../../../../../raw/code/simply/simply/model_lib_test.py#L420)
  - `test_replace_embeddings(self)` — [`L307`](../../../../../raw/code/simply/simply/model_lib_test.py#L307)
  - `test_sampling_max_decode_steps_equals_prefill_size(self)` — [`L950`](../../../../../raw/code/simply/simply/model_lib_test.py#L950)
  - `test_sampling_output_logprobs(self, temperature: float, top_k: int, top_p: float)` — [`L807`](../../../../../raw/code/simply/simply/model_lib_test.py#L807)
  - `test_sampling_token_scores(self, temperature: float, top_k: int, top_p: float)` — [`L872`](../../../../../raw/code/simply/simply/model_lib_test.py#L872)
  - `test_segment_info(self)` — [`L1135`](../../../../../raw/code/simply/simply/model_lib_test.py#L1135)
  - `test_tree_norm(self)` — [`L605`](../../../../../raw/code/simply/simply/model_lib_test.py#L605)
  - `train_one_step_fn(state, batch, lr, add_log_info=False)` — [`L198`](../../../../../raw/code/simply/simply/model_lib_test.py#L198)
  - `train_one_step_fn_grad_accum(state, batch, lr, add_log_info=False)` — [`L275`](../../../../../raw/code/simply/simply/model_lib_test.py#L275)
  - `config` — [`L80`](../../../../../raw/code/simply/simply/model_lib_test.py#L80)
  - `tfm_lm` — [`L81`](../../../../../raw/code/simply/simply/model_lib_test.py#L81)
- uses (calls/refs, reference-scoped): [`generate`](model_lib.md#LMInterface.generate), [`score`](model_lib.md#LMInterface.score), [`score_tokens`](model_lib.md#LMInterface.score_tokens), [`TransformerLM`](model_lib.md#TransformerLM), [`input_processor`](model_lib.md#LMInterface.input_processor), [`SamplingParams`](model_lib.md#SamplingParams), [`SamplingOutput`](model_lib.md#SamplingOutput), [`max_decode_steps`](utils/sampling_lib.md#SamplingParams.max_decode_steps), [`train_one_step`](model_lib.md#train_one_step), [`LMInterface`](model_lib.md#LMInterface), [`compute_log_likelihood`](utils/sampling_lib.md#compute_log_likelihood), [`gspmd_sharding`](config_lib.md#gspmd_sharding), [`lm_test`](model_lib_test.md#lm_test), [`temperature`](utils/sampling_lib.md#SamplingParams.temperature), [`Adam`](utils/optimizers.md#Adam), [`TestVocab`](utils/tokenization.md#TestVocab), [`encode`](utils/sampling_lib.md#InputProcessorInterface.encode), [`AnnotatedArray`](utils/common.md#AnnotatedArray), [`top_k`](utils/sampling_lib.md#SamplingParams.top_k), [`top_p`](utils/sampling_lib.md#SamplingParams.top_p), [`max_seq_len`](utils/sampling_lib.md#SamplingParams.max_seq_len), [`ScoringParams`](model_lib.md#ScoringParams), [`num_samples`](utils/sampling_lib.md#SamplingParams.num_samples), [`pad_decode_state_to`](model_lib.md#pad_decode_state_to), [`quantize_tfm_params`](model_lib.md#quantize_tfm_params), [`attn`](model_lib.md#attn), [`dump_dataclasses`](utils/pytree.md#dump_dataclasses), [`create_mask`](model_lib.md#create_mask), [`input_as_chunks`](utils/sampling_lib.md#InputProcessorInterface.input_as_chunks), [`SimplyConfig`](model_lib.md#SimplyConfig), [`chunked_local_attn`](model_lib.md#chunked_local_attn), [`encode`](utils/tokenization.md#TestVocab.encode), [`intermediate_decode_steps`](utils/sampling_lib.md#SamplingParams.intermediate_decode_steps), [`load_dataclasses`](utils/pytree.md#load_dataclasses), [`temperature`](model_lib.md#ScoringParams.temperature), [`top_k`](model_lib.md#ScoringParams.top_k), [`top_p`](model_lib.md#ScoringParams.top_p), [`get_scaling_info`](model_lib.md#get_scaling_info), [`sort_by`](utils/sampling_lib.md#SamplingParams.sort_by), [`bos_id`](utils/tokenization.md#SimplyVocab.bos_id)  (+7 more)

### `TestNpArrayQuantizer`  ·  implements/extends SimplyVocab
- def: [`simply/model_lib_test.py:1193`](../../../../../raw/code/simply/simply/model_lib_test.py#L1193)
- signature: `class TestNpArrayQuantizer(tokenization.SimplyVocab[np.ndarray]):`
- members:
  - `decode(self, token_ids: list[int])` — [`L1229`](../../../../../raw/code/simply/simply/model_lib_test.py#L1229)
  - `encode(self, raw_sequence: np.ndarray)` — [`L1220`](../../../../../raw/code/simply/simply/model_lib_test.py#L1220)
  - `feature_dim` — [`L1196`](../../../../../raw/code/simply/simply/model_lib_test.py#L1196)
  - `unk_id` — [`L1211`](../../../../../raw/code/simply/simply/model_lib_test.py#L1211)
  - `vocab_size` — [`L1195`](../../../../../raw/code/simply/simply/model_lib_test.py#L1195)
- protocol/private: `__init__`[`L1198`](../../../../../raw/code/simply/simply/model_lib_test.py#L1198), `_rev_vocab_dict`[`L1218`](../../../../../raw/code/simply/simply/model_lib_test.py#L1218), `_vocab_dict`[`L1215`](../../../../../raw/code/simply/simply/model_lib_test.py#L1215)
- uses (calls/refs, reference-scoped): [`SimplyVocab`](utils/tokenization.md#SimplyVocab), [`bos_id`](utils/tokenization.md#SimplyVocab.bos_id), [`eos_id`](utils/tokenization.md#SimplyVocab.eos_id), [`pad_id`](utils/tokenization.md#SimplyVocab.pad_id)
- used by: [`SimplyVocab`](utils/tokenization.md#SimplyVocab), [`decode`](utils/tokenization.md#SimplyVocab.decode), [`encode`](utils/tokenization.md#SimplyVocab.encode)

## Functions
- `lm_test()` — [`L40`](../../../../../raw/code/simply/simply/model_lib_test.py#L40) — Returns a test config for TransformerLM.
- `simple_moe(params, inputs, inputs_mask, num_experts_per_token, num_experts, ffn_activation, use_gated_activation_in_ffn, activation_dtype)` — [`L1238`](../../../../../raw/code/simply/simply/model_lib_test.py#L1238)

