---
title: 'Module: tpu_inference/runner/compilation_manager.py'
type: catalog
provenance: extracted
module: tpu_inference/runner/compilation_manager.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.runner.compilation_manager`/
symbols:
  CompilationManager.runner: CompilationManager#runner.
  CompilationManager._precompile_continue_decode: CompilationManager#_precompile_continue_decode().
  CompilationManager._create_dummy_tensor: CompilationManager#_create_dummy_tensor().
  CompilationManager.capture_model: CompilationManager#capture_model().
  CompilationManager._precompile_mtp_helpers: CompilationManager#_precompile_mtp_helpers().
  CompilationManager._precompile_eagle3_helpers: CompilationManager#_precompile_eagle3_helpers().
  CompilationManager._precompile_process_and_extend_logits: CompilationManager#_precompile_process_and_extend_logits().
  CompilationManager._precompile_backbone_helper: CompilationManager#_precompile_backbone_helper().
  CompilationManager._precompile_sampling: CompilationManager#_precompile_sampling().
  CompilationManager._precompile_extract_last_sampled_tokens: CompilationManager#_precompile_extract_last_sampled_tokens().
  CompilationManager._precompile_rejection_sampler: CompilationManager#_precompile_rejection_sampler().
  CompilationManager._run_compilation: CompilationManager#_run_compilation().
  CompilationManager._precompile_compute_logits: CompilationManager#_precompile_compute_logits().
  CompilationManager._precompile_gather_logprobs: CompilationManager#_precompile_gather_logprobs().
  CompilationManager._precompile_backbone_with_inputs_embeds: CompilationManager#_precompile_backbone_with_inputs_embeds().
  CompilationManager._precompile_backbone_text_only: CompilationManager#_precompile_backbone_text_only().
  CompilationManager._precompile_structured_decoding: CompilationManager#_precompile_structured_decoding().
  logger: logger.
  CompilationManager._precompile_select_from_array: CompilationManager#_precompile_select_from_array().
  CompilationManager._precompile_extract_draft_token_ids: CompilationManager#_precompile_extract_draft_token_ids().
  CompilationManager._precompile_extend_logits_simple: CompilationManager#_precompile_extend_logits_simple().
  CompilationManager._flush_compilations: CompilationManager#_flush_compilations().
  CompilationManager._precompile_speculative_decoding: CompilationManager#_precompile_speculative_decoding().
  CompilationManager._precompile_input_embeddings_merger: CompilationManager#_precompile_input_embeddings_merger().
  CompilationManager._precompile_substitute_placeholder_token: CompilationManager#_precompile_substitute_placeholder_token().
  CompilationManager._precompile_subtract_num_rejected_tokens: CompilationManager#_precompile_subtract_num_rejected_tokens().
  CompilationManager._precompile_concat_last_sampled_tokens_and_draft_tokens: CompilationManager#_precompile_concat_last_sampled_tokens_and_draft_tokens().
  CompilationManager._precompile_disagg_utils: CompilationManager#_precompile_disagg_utils().
  CompilationManager.build_attn: CompilationManager#build_attn().
  CompilationManager._precompile_select_from_array_helper: CompilationManager#_precompile_select_from_array_helper().
  CompilationManager.build_block_table: CompilationManager#build_block_table().
  CompilationManager._compile_one: CompilationManager#_compile_one().
  CompilationManager._compile_executor: CompilationManager#_compile_executor.
  CompilationManager.model_fn_warmup: CompilationManager#model_fn_warmup().
  CompilationManager.drafter_propose_warmup: CompilationManager#drafter_propose_warmup().
  CompilationManager.continue_decode_wrapper: CompilationManager#continue_decode_wrapper().
  CompilationManager._gather_logprobs_precompiled: CompilationManager#_gather_logprobs_precompiled.
  CompilationManager._finalize_compilation: CompilationManager#_finalize_compilation().
  CompilationManager.continue_decode_warmup: CompilationManager#continue_decode_warmup().
  CompilationManager._prev_stack_size: CompilationManager#_prev_stack_size.
  CompilationManager._compile: CompilationManager#_compile().
  CompilationManager: CompilationManager#
  CompilationManager._precompile_compute_pooling: CompilationManager#_precompile_compute_pooling().
  CompilationManager._compile_futures: CompilationManager#_compile_futures.
  CompilationManager._warmup_tasks: CompilationManager#_warmup_tasks.
  CompilationManager._skip_self_arg_warmup_handler: CompilationManager#_skip_self_arg_warmup_handler().
  CompilationManager.__init__: CompilationManager#__init__().
  CompilationManager._sampling_precompiled: CompilationManager#_sampling_precompiled.
  CompilationManager._should_skip_padding_combination: CompilationManager#_should_skip_padding_combination().
  BLOCK_BUCKETS: BLOCK_BUCKETS.
---
# Module: [`tpu_inference/runner/compilation_manager.py`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py)

## Classes
### `CompilationManager`
- def: [`tpu_inference/runner/compilation_manager.py:56`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L56)
- signature: `class CompilationManager:`
- members:
  - `_create_dummy_tensor(self, shape: Tuple[int, ...], dtype: Any, sharding: Optional[NamedSharding] = None)` — [`L96`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L96) — Helper to create dummy tensors for precompilation. — documented in [root](../../../concepts/root.md)
  - `_finalize_compilation(self)` — [`L273`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L273) — Shut down the precompile pool and restore the thread stack default
  - `_flush_compilations(self)` — [`L178`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L178) — Wait for all currently-pending background compilations and run their
  - `_precompile_select_from_array_helper(self, name: str, source_paddings: List[int], indices_paddings: List[int], hidden_dim: int, input_sharding: Optional[NamedSharding] = None, indices_sharding: Optional[NamedSharding] = None, only_equal_paddings: bool = False, check_should_skip_padding: bool = True)` — [`L711`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L711) — Precompile select_from_array operations with various input shape combinations.
  - `_should_skip_padding_combination(self, outer_val: int, inner_val: int, only_equal: bool)` — [`L117`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L117) — Helper to determine if we should skip this padding combination.
  - `_skip_self_arg_warmup_handler(self, fn, args, call_kwargs)` — [`L767`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L767) — Warmup handler for methods compiled with an explicit `self` as the
  - `build_attn(block_tables: jax.Array | None)` — [`L399`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L399)
  - `build_block_table(kv_cache_gid: int)` — [`L388`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L388)
  - `capture_model(self)` — [`L212`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L212)
  - `continue_decode_warmup(_fn, _args, _call_kwargs)` — [`L1746`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L1746)
  - `continue_decode_wrapper(state, model_fn, compute_logits_fn, sample_fn, mesh, sampling_metadata, init_state, kv_caches, max_decode_steps, static_max_decode_steps, eos_token_id, padding_token_id, rng, layer_name_to_kvcache_index, lora_metadata, is_first_rank, is_last_rank, dp_size, collect_expert_indices)` — [`L1701`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L1701)
  - `drafter_propose_warmup(_fn, _args, _call_kwargs)` — [`L1360`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L1360)
  - `model_fn_warmup(_fn, _args, _call_kwargs)` — [`L425`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L425)
  - `runner` — [`L59`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L59)
- protocol/private: `__init__`[`L58`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L58), `_compile`[`L162`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L162), `_compile_executor`[`L78`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L78), `_compile_futures`[`L93`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L93), `_compile_one`[`L471`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L471), `_gather_logprobs_precompiled`[`L61`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L61), `_precompile_backbone_helper`[`L343`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L343), `_precompile_backbone_text_only`[`L589`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L589), `_precompile_backbone_with_inputs_embeds`[`L638`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L638), `_precompile_compute_logits`[`L821`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L821), `_precompile_compute_pooling`[`L849`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L849), `_precompile_concat_last_sampled_tokens_and_draft_tokens`[`L561`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L561), `_precompile_continue_decode`[`L1596`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L1596), `_precompile_disagg_utils`[`L921`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L921), `_precompile_eagle3_helpers`[`L1295`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L1295), `_precompile_extend_logits_simple`[`L1096`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L1096), `_precompile_extract_draft_token_ids`[`L1143`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L1143), `_precompile_extract_last_sampled_tokens`[`L1178`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L1178), `_precompile_gather_logprobs`[`L947`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L947), `_precompile_input_embeddings_merger`[`L287`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L287), `_precompile_mtp_helpers`[`L1430`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L1430), `_precompile_process_and_extend_logits`[`L1029`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L1029), `_precompile_rejection_sampler`[`L1231`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L1231), `_precompile_sampling`[`L858`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L858), `_precompile_select_from_array`[`L774`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L774), `_precompile_speculative_decoding`[`L1130`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L1130), `_precompile_structured_decoding`[`L1563`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L1563), `_precompile_substitute_placeholder_token`[`L465`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L465), `_precompile_subtract_num_rejected_tokens`[`L520`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L520), `_prev_stack_size`[`L76`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L76), `_run_compilation`[`L124`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L124), `_sampling_precompiled`[`L60`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L60), `_warmup_tasks`[`L94`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L94)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../layers/common/sharding.md#ShardingAxisName), `mesh`, `input_batch`, [`AttentionMetadata`](../layers/common/attention_metadata.md#AttentionMetadata), `max_num_reqs`, `vllm_config`, `model_config`, `dp_size`, [`JaxIntermediateTensors`](../models/jax/jax_intermediate_tensor.md#JaxIntermediateTensors), `speculative_config`, [`continue_decode`](decode_loop.md#continue_decode), [`input_positions`](../layers/common/attention_metadata.md#AttentionMetadata.input_positions), `kv_caches`, [`logger`](compilation_manager.md#logger), `is_last_rank`, `num_tokens_paddings`, [`sample`](../layers/jax/sample/sampling.md#sample), `drafter`, `num_logits_paddings`, `rng_params_for_sampling`, [`SpecDecodeMetadata`](../utils.md#SpecDecodeMetadata), [`TPUSupportedSamplingMetadata`](../layers/jax/sample/sampling_metadata.md#TPUSupportedSamplingMetadata), `kv_cache_config`, [`process_and_extend_logits`](../spec_decode/jax/utils.md#process_and_extend_logits), `state_leaves`, `kv_cache_manager`, [`seq_lens`](../layers/common/attention_metadata.md#AttentionMetadata.seq_lens), [`block_tables`](../layers/common/attention_metadata.md#AttentionMetadata.block_tables), [`query_start_loc`](../layers/common/attention_metadata.md#AttentionMetadata.query_start_loc), [`request_distribution`](../layers/common/attention_metadata.md#AttentionMetadata.request_distribution), [`prepare_inputs`](../spec_decode/jax/eagle3.md#Eagle3Proposer.prepare_inputs), `num_reqs_paddings`, `rank`, [`propose`](../spec_decode/jax/eagle3.md#Eagle3Proposer.propose), [`tensors`](../models/jax/jax_intermediate_tensor.md#JaxIntermediateTensors.tensors), [`do_sampling`](../layers/jax/sample/sampling_metadata.md#TPUSupportedSamplingMetadata.do_sampling), `_select_from_array_fn`, `lora_utils`, [`extract_last_sampled_tokens`](../spec_decode/jax/utils.md#extract_last_sampled_tokens), `vocab_size`  (+59 more)
- used by: `compilation_manager`, `capture_model`

## Module values
- `BLOCK_BUCKETS` — [`L53`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L53)
- `logger` — [`L50`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/compilation_manager.py#L50)

