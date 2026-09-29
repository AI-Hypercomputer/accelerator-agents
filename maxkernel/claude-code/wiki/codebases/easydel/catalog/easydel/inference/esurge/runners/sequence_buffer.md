---
title: 'Module: easydel/inference/esurge/runners/sequence_buffer.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/runners/sequence_buffer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.runners.sequence_buffer`/
symbols:
  SequenceBuffer.remove_request: SequenceBuffer#remove_request().
  SequenceBuffer.clear: SequenceBuffer#clear().
  SequenceBuffer._process_sampling_params: SequenceBuffer#_process_sampling_params().
  SequenceBuffer.swap_states: SequenceBuffer#swap_states().
  SequenceBuffer.add_request: SequenceBuffer#add_request().
  SequenceBuffer._move_request: SequenceBuffer#_move_request().
  SequenceBuffer._process_optional_params: SequenceBuffer#_process_optional_params().
  SequenceBuffer.get_active_sampling_params: SequenceBuffer#get_active_sampling_params().
  SequenceBuffer._req_ids: SequenceBuffer#_req_ids.
  SequenceBuffer.num_computed_tokens: SequenceBuffer#num_computed_tokens.
  ModelRunnerSamplingMetadata.from_sequence_buffer: ModelRunnerSamplingMetadata#from_sequence_buffer().
  SequenceBuffer.token_ids: SequenceBuffer#token_ids.
  SequenceBuffer.top_k: SequenceBuffer#top_k.
  SequenceBuffer.allowed_token_ids_mask: SequenceBuffer#allowed_token_ids_mask.
  SequenceBuffer.temperature: SequenceBuffer#temperature.
  SequenceBuffer.top_p: SequenceBuffer#top_p.
  SequenceBuffer.min_p: SequenceBuffer#min_p.
  SequenceBuffer.req_id_to_index: SequenceBuffer#req_id_to_index.
  swap_rows: swap_rows().
  SequenceBuffer.req_output_token_ids: SequenceBuffer#req_output_token_ids.
  SequenceBuffer.logit_bias: SequenceBuffer#logit_bias.
  SequenceBuffer.frequency_penalties: SequenceBuffer#frequency_penalties.
  SequenceBuffer.presence_penalties: SequenceBuffer#presence_penalties.
  SequenceBuffer.repetition_penalties: SequenceBuffer#repetition_penalties.
  SequenceBuffer.num_tokens: SequenceBuffer#num_tokens.
  move_row: move_row().
  SequenceBuffer.page_table: SequenceBuffer#page_table.
  SequenceBuffer.condense: SequenceBuffer#condense().
  SequenceBuffer._move_sparse_data: SequenceBuffer#_move_sparse_data().
  SequenceBuffer._make_prompt_token_ids_tensor: SequenceBuffer#_make_prompt_token_ids_tensor().
  SequenceBuffer.num_tokens_no_spec: SequenceBuffer#num_tokens_no_spec.
  SequenceBuffer._update_request_distribution: SequenceBuffer#_update_request_distribution().
  SequenceBuffer.num_reqs: SequenceBuffer#num_reqs().
  SequenceBuffer.max_num_reqs: SequenceBuffer#max_num_reqs.
  SequenceBuffer.num_prompt_tokens: SequenceBuffer#num_prompt_tokens.
  SequenceBuffer._set_allowed_token_ids: SequenceBuffer#_set_allowed_token_ids().
  SequenceBuffer.req_ids: SequenceBuffer#req_ids().
  SequenceBuffer.get_request_indices_with_penalty: SequenceBuffer#get_request_indices_with_penalty().
  SequenceBuffer.compact_holes_in_range: SequenceBuffer#compact_holes_in_range().
  SequenceBuffer._allocate_index: SequenceBuffer#_allocate_index().
  SequenceBuffer.no_penalties: SequenceBuffer#no_penalties().
  SequenceBuffer: SequenceBuffer#
  SequenceBuffer.min_tokens: SequenceBuffer#min_tokens.
  SequenceBuffer.generator_seeds: SequenceBuffer#generator_seeds.
  SequenceBuffer.bad_words_token_ids: SequenceBuffer#bad_words_token_ids.
  SequenceBuffer.frequency_penalties_reqs: SequenceBuffer#frequency_penalties_reqs.
  SequenceBuffer.presence_penalties_reqs: SequenceBuffer#presence_penalties_reqs.
  SequenceBuffer.repetition_penalties_reqs: SequenceBuffer#repetition_penalties_reqs.
  SequenceBuffer._layout_version: SequenceBuffer#_layout_version.
  fill_slice: fill_slice().
  SequenceBuffer.num_slots: SequenceBuffer#num_slots().
  SequenceBuffer.max_model_len: SequenceBuffer#max_model_len.
  SequenceBuffer.vocab_size: SequenceBuffer#vocab_size.
  SequenceBuffer.min_p_reqs: SequenceBuffer#min_p_reqs.
  SequenceBuffer.num_logprobs: SequenceBuffer#num_logprobs.
  build_sampling_arrays: build_sampling_arrays().
  SequenceBuffer.in_progress_prompt_logprobs_cpu: SequenceBuffer#in_progress_prompt_logprobs_cpu.
  SequenceBuffer.layout_version: SequenceBuffer#layout_version().
  build_sampling_arrays.fill: build_sampling_arrays().fill().
  SequenceBuffer.greedy_reqs: SequenceBuffer#greedy_reqs.
  SequenceBuffer.random_reqs: SequenceBuffer#random_reqs.
  SequenceBuffer.top_p_reqs: SequenceBuffer#top_p_reqs.
  SequenceBuffer.top_k_reqs: SequenceBuffer#top_k_reqs.
  SequenceBuffer.has_allowed_token_ids: SequenceBuffer#has_allowed_token_ids.
  SequenceBuffer.num_prompt_logprobs: SequenceBuffer#num_prompt_logprobs.
  SequenceBuffer.all_greedy: SequenceBuffer#all_greedy().
  SequenceBuffer._ensure_logit_bias_capacity: SequenceBuffer#_ensure_logit_bias_capacity().
  pack_prompts: pack_prompts().
  swap_rows_pytree: swap_rows_pytree().
  SequenceBuffer.all_random: SequenceBuffer#all_random().
  SequenceBuffer.no_top_p: SequenceBuffer#no_top_p().
  SequenceBuffer.no_top_k: SequenceBuffer#no_top_k().
  SequenceBuffer.no_min_p: SequenceBuffer#no_min_p().
  SequenceBuffer.max_num_logprobs: SequenceBuffer#max_num_logprobs().
  SequenceBuffer.no_prompt_logprob: SequenceBuffer#no_prompt_logprob().
  SequenceBuffer.no_allowed_token_ids: SequenceBuffer#no_allowed_token_ids().
  logger: logger.
  build_allowed_mask: build_allowed_mask().
  SequenceBuffer.request_distribution: SequenceBuffer#request_distribution.
  SequenceBuffer.__init__: SequenceBuffer#__init__().
  SequenceBuffer.max_num_batched_tokens: SequenceBuffer#max_num_batched_tokens.
  ModelRunnerSamplingMetadata: ModelRunnerSamplingMetadata#
  ModelRunnerSamplingMetadata.temperature: ModelRunnerSamplingMetadata#temperature.
  ModelRunnerSamplingMetadata.min_p: ModelRunnerSamplingMetadata#min_p.
  ModelRunnerSamplingMetadata.top_k: ModelRunnerSamplingMetadata#top_k.
  ModelRunnerSamplingMetadata.top_p: ModelRunnerSamplingMetadata#top_p.
  ModelRunnerSamplingMetadata.all_greedy: ModelRunnerSamplingMetadata#all_greedy.
  ModelRunnerSamplingMetadata.logprobs: ModelRunnerSamplingMetadata#logprobs.
  ModelRunnerSamplingMetadata.no_penalties: ModelRunnerSamplingMetadata#no_penalties.
  ModelRunnerSamplingMetadata.prompt_token_ids: ModelRunnerSamplingMetadata#prompt_token_ids.
  ModelRunnerSamplingMetadata.frequency_penalties: ModelRunnerSamplingMetadata#frequency_penalties.
  ModelRunnerSamplingMetadata.presence_penalties: ModelRunnerSamplingMetadata#presence_penalties.
  ModelRunnerSamplingMetadata.repetition_penalties: ModelRunnerSamplingMetadata#repetition_penalties.
  ModelRunnerSamplingMetadata.output_token_ids: ModelRunnerSamplingMetadata#output_token_ids.
  ModelRunnerSamplingMetadata.min_tokens: ModelRunnerSamplingMetadata#min_tokens.
  ModelRunnerSamplingMetadata.logit_bias: ModelRunnerSamplingMetadata#logit_bias.
  ModelRunnerSamplingMetadata.allowed_token_ids_mask: ModelRunnerSamplingMetadata#allowed_token_ids_mask.
  ModelRunnerSamplingMetadata.bad_words_token_ids: ModelRunnerSamplingMetadata#bad_words_token_ids.
---
# Module: [`easydel/inference/esurge/runners/sequence_buffer.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py)

## Classes
### `ModelRunnerSamplingMetadata`
- def: [`easydel/inference/esurge/runners/sequence_buffer.py:1093`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1093)
- doc: Metadata for sampling operations during model execution.
- signature: `class ModelRunnerSamplingMetadata:`
- members:
  - `from_sequence_buffer(cls, sequence_buffer: SequenceBuffer, padded_num_reqs: int, generate_params_if_all_greedy: bool = False)` — [`L1139`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1139) — Create sampling metadata from a sequence buffer.
  - `all_greedy` — [`L1123`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1123)
  - `allowed_token_ids_mask` — [`L1135`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1135)
  - `bad_words_token_ids` — [`L1136`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1136)
  - `frequency_penalties` — [`L1128`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1128)
  - `logit_bias` — [`L1134`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1134)
  - `logprobs` — [`L1124`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1124)
  - `min_p` — [`L1119`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1119)
  - `min_tokens` — [`L1133`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1133)
  - `no_penalties` — [`L1125`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1125)
  - `output_token_ids` — [`L1132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1132)
  - `presence_penalties` — [`L1129`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1129)
  - `prompt_token_ids` — [`L1127`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1127)
  - `repetition_penalties` — [`L1130`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1130)
  - `temperature` — [`L1118`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1118)
  - `top_k` — [`L1120`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1120)
  - `top_p` — [`L1121`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1121)
- uses (calls/refs, reference-scoped): [`top_k`](sequence_buffer.md#SequenceBuffer.top_k), [`min_p`](sequence_buffer.md#SequenceBuffer.min_p), [`temperature`](sequence_buffer.md#SequenceBuffer.temperature), [`top_p`](sequence_buffer.md#SequenceBuffer.top_p), [`num_reqs`](sequence_buffer.md#SequenceBuffer.num_reqs), [`SequenceBuffer`](sequence_buffer.md#SequenceBuffer), [`fill_slice`](sequence_buffer.md#fill_slice), [`all_greedy`](sequence_buffer.md#SequenceBuffer.all_greedy)

### `SequenceBuffer`
- def: [`easydel/inference/esurge/runners/sequence_buffer.py:251`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L251)
- doc: Buffer for managing token sequences during generation.
- signature: `class SequenceBuffer:`
- members:
  - `__init__(self, max_num_reqs: int, max_model_len: int, max_num_batched_tokens: int, vocab_size: int, page_sizes: list[int], sharding: jax.sharding.Sharding | None = None)` — [`L264`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L264) — Initialize a SequenceBuffer with all arrays and page table.
  - `_allocate_index(self, req_index: int | None)` — [`L917`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L917) — Allocate an index for a new request.
  - `_ensure_logit_bias_capacity(self, upto_idx: int)` — [`L412`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L412) — Ensure logit_bias list has sufficient capacity.
  - `_make_prompt_token_ids_tensor(self)` — [`L955`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L955) — Create a padded tensor of prompt token IDs.
  - `_move_request(self, from_idx: int, to_idx: int)` — [`L683`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L683) — Move a request from one index to another.
  - `_move_sparse_data(self, from_idx: int, to_idx: int)` — [`L747`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L747) — Move sparse and optional data between indices.
  - `_process_optional_params(self, request: EngineRequest, sampling_params: SamplingParams, req_id: str, req_index: int)` — [`L837`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L837) — Process optional and sparse sampling parameters.
  - `_process_sampling_params(self, sampling_params: SamplingParams, req_id: str, req_index: int)` — [`L780`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L780) — Process and store core sampling parameters. — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `_set_allowed_token_ids(self, req_id: str, req_index: int, allowed_token_ids: list[int])` — [`L884`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L884) — Set the allowed token IDs for a request.
  - `_update_request_distribution(self)` — [`L338`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L338) — Update the request distribution triple [decode_only, chunked_prefill, total]. — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `add_request(self, request: EngineRequest, req_index: int | None = None)` — [`L424`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L424) — Add a new request to the buffer. — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `all_greedy(self)` — [`L364`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L364) — Whether all active requests use greedy (deterministic) sampling.
  - `all_random(self)` — [`L369`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L369) — Whether all active requests use random (stochastic) sampling.
  - `clear(self)` — [`L1033`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1033) — Clear all data in the buffer. — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `compact_holes_in_range(self, lo: int, hi: int)` — [`L575`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L575) — Move all None (hole) slots to the end within [lo, hi).
  - `condense(self, empty_req_indices: list[int])` — [`L654`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L654) — Condense the buffer by removing gaps.
  - `get_active_sampling_params(self, req_index: int)` — [`L999`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L999) — Get active sampling parameters for a request.
  - `get_request_indices_with_penalty(self)` — [`L981`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L981) — Get indices of requests with penalties.
  - `layout_version(self)` — [`L359`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L359) — Monotonic version for row-layout or membership changes.
  - `max_num_logprobs(self)` — [`L398`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L398) — Maximum number of log probabilities requested across all active requests, or None if none.
  - `no_allowed_token_ids(self)` — [`L408`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L408) — Whether no active requests have constrained token ID sets.
  - `no_min_p(self)` — [`L384`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L384) — Whether no active requests use min-p sampling.
  - `no_penalties(self)` — [`L389`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L389) — Whether no active requests use any penalty (presence, frequency, or repetition).
  - `no_prompt_logprob(self)` — [`L403`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L403) — Whether no active requests need prompt log probabilities.
  - `no_top_k(self)` — [`L379`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L379) — Whether no active requests use top-k sampling.
  - `no_top_p(self)` — [`L374`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L374) — Whether no active requests use top-p (nucleus) sampling.
  - `num_reqs(self)` — [`L349`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L349) — Number of active requests currently in the buffer.
  - `num_slots(self)` — [`L354`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L354) — Number of materialized row slots (includes empty holes).
  - `remove_request(self, req_id: str)` — [`L494`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L494) — Remove a request from the buffer. — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `req_ids(self)` — [`L344`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L344) — List of request IDs indexed by row slot, with None for empty slots.
  - `swap_states(self, i1: int, i2: int)` — [`L596`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L596) — Swap the states of two requests at given indices. — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `allowed_token_ids_mask` — [`L335`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L335) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `bad_words_token_ids` — [`L334`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L334) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `frequency_penalties` — [`L300`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L300) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `frequency_penalties_reqs` — [`L323`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L323)
  - `generator_seeds` — [`L329`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L329) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `greedy_reqs` — [`L318`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L318)
  - `has_allowed_token_ids` — [`L326`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L326)
  - `in_progress_prompt_logprobs_cpu` — [`L332`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L332)
  - `logit_bias` — [`L333`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L333) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `max_model_len` — [`L285`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L285)
  - `max_num_batched_tokens` — [`L286`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L286)
  - `max_num_reqs` — [`L284`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L284)
  - `min_p` — [`L299`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L299) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `min_p_reqs` — [`L322`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L322)
  - `min_tokens` — [`L328`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L328) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `num_computed_tokens` — [`L294`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L294) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `num_logprobs` — [`L330`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L330) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `num_prompt_logprobs` — [`L331`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L331)
  - `num_prompt_tokens` — [`L293`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L293) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `num_tokens` — [`L291`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L291) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `num_tokens_no_spec` — [`L292`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L292) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `page_table` — [`L305`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L305) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `presence_penalties` — [`L301`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L301) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `presence_penalties_reqs` — [`L324`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L324)
  - `random_reqs` — [`L319`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L319)
  - `repetition_penalties` — [`L302`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L302) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `repetition_penalties_reqs` — [`L325`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L325)
  - `req_id_to_index` — [`L315`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L315) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `req_output_token_ids` — [`L316`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L316) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `request_distribution` — [`L317`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L317)
  - `temperature` — [`L296`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L296) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `token_ids` — [`L290`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L290) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `top_k` — [`L298`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L298) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `top_k_reqs` — [`L321`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L321)
  - `top_p` — [`L297`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L297) — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
  - `top_p_reqs` — [`L320`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L320)
  - `vocab_size` — [`L287`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L287)
- protocol/private: `_layout_version`[`L336`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L336), `_req_ids`[`L314`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L314)
- uses (calls/refs, reference-scoped): [`SamplingParams`](../../sampling_params.md#SamplingParams), [`EngineRequest`](../request.md#EngineRequest), [`num_computed_tokens`](../request.md#EngineRequest.num_computed_tokens), [`swap_rows`](sequence_buffer.md#swap_rows), [`temperature`](../../sampling_params.md#SamplingParams.temperature), [`move_row`](sequence_buffer.md#move_row), [`sampling_type`](../../sampling_params.md#SamplingParams.sampling_type), [`num_tokens`](../request.md#EngineRequest.num_tokens), [`output_token_ids`](../request.md#EngineRequest.output_token_ids), [`swap_dict_values`](../outputs.md#swap_dict_values), [`top_p`](../../sampling_params.md#SamplingParams.top_p), [`LogprobsTensors`](../outputs.md#LogprobsTensors), [`add_row`](../page_table.md#MultiGroupPageTable.add_row), [`clear_row`](../page_table.md#MultiGroupPageTable.clear_row), [`presence_penalty`](../../sampling_params.md#SamplingParams.presence_penalty), [`repetition_penalty`](../../sampling_params.md#SamplingParams.repetition_penalty), [`SamplingType`](../../sampling_params.md#SamplingType), [`clear`](../page_table.md#MultiGroupPageTable.clear), [`move_row`](../page_table.md#MultiGroupPageTable.move_row), [`swap_row`](../page_table.md#MultiGroupPageTable.swap_row), [`frequency_penalty`](../../sampling_params.md#SamplingParams.frequency_penalty), [`logit_bias`](../../sampling_params.md#SamplingParams.logit_bias), [`min_tokens`](../../sampling_params.md#SamplingParams.min_tokens), [`prompt_logprobs`](../../sampling_params.md#SamplingParams.prompt_logprobs), [`prompt_token_ids`](../request.md#EngineRequest.prompt_token_ids), [`sampling_params`](../request.md#EngineRequest.sampling_params), [`top_k`](../../sampling_params.md#SamplingParams.top_k), [`all_stop_token_ids`](../../sampling_params.md#SamplingParams.all_stop_token_ids), [`bad_words_token_ids`](../../sampling_params.md#SamplingParams.bad_words_token_ids), [`allowed_token_ids`](../../sampling_params.md#SamplingParams.allowed_token_ids), [`logprobs`](../../sampling_params.md#SamplingParams.logprobs), [`min_p`](../../sampling_params.md#SamplingParams.min_p), [`pack_prompts`](sequence_buffer.md#pack_prompts), [`MultiGroupPageTable`](../page_table.md#MultiGroupPageTable), [`GREEDY`](../../sampling_params.md#SamplingType.GREEDY), [`logger`](sequence_buffer.md#logger)
- used by: [`_execute_model_impl`](model_runner.md#eSurgeRunner._execute_model_impl), [`_update_states`](model_runner.md#eSurgeRunner._update_states), [`sequence_buffer`](model_runner.md#eSurgeRunner.sequence_buffer), [`get_compile_configurations`](execution_manager.md#ExecutionManager.get_compile_configurations), [`_get_window_state_views`](model_runner.md#eSurgeRunner._get_window_state_views), [`_modify_prev_results`](model_runner.md#eSurgeRunner._modify_prev_results), [`_reorder_decode_first_per_shard`](model_runner.md#eSurgeRunner._reorder_decode_first_per_shard), [`from_sequence_buffer`](sequence_buffer.md#ModelRunnerSamplingMetadata.from_sequence_buffer), [`_reorder_decode_first`](model_runner.md#eSurgeRunner._reorder_decode_first), [`_update_placeholder`](model_runner.md#eSurgeRunner._update_placeholder), [`_window_row_indices_cpu`](model_runner.md#eSurgeRunner._window_row_indices_cpu), [`_finalize_sync_runner_state`](model_runner.md#eSurgeRunner._finalize_sync_runner_state), [`reset_state`](model_runner.md#eSurgeRunner.reset_state), [`_collect_schedulable_window_rows`](model_runner.md#eSurgeRunner._collect_schedulable_window_rows), [`_find_reuse_index_in_shard`](model_runner.md#eSurgeRunner._find_reuse_index_in_shard), [`_window_frequency_penalties_cpu`](model_runner.md#eSurgeRunner._window_frequency_penalties_cpu), [`_window_min_p_cpu`](model_runner.md#eSurgeRunner._window_min_p_cpu), [`_window_presence_penalties_cpu`](model_runner.md#eSurgeRunner._window_presence_penalties_cpu), [`_window_repetition_penalties_cpu`](model_runner.md#eSurgeRunner._window_repetition_penalties_cpu), [`_window_temperature_cpu`](model_runner.md#eSurgeRunner._window_temperature_cpu), [`_window_top_k_cpu`](model_runner.md#eSurgeRunner._window_top_k_cpu), [`_window_top_p_cpu`](model_runner.md#eSurgeRunner._window_top_p_cpu)

## Functions
- `build_allowed_mask(allowed_ids_padded, allowed_lens, vocab_size, max_allowed)` — [`L213`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L213) — Build a mask for allowed token IDs.
- `build_sampling_arrays(temperature, min_p, top_p, top_k, num_reqs, padded_num_reqs)` — [`L89`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L89) — Build padded sampling parameter arrays.
- `fill(arr, fill_val)` — [`L114`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L114) — Create a padded array filled with a default value and active entries copied in.
- `fill_slice(arr, fill_val, num_reqs, padded_num_reqs)` — [`L1179`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L1179) — Fill array slice with padding value.
- `move_row(arr, from_idx, to_idx)` — [`L185`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L185) — Move a row from one index to another.
- `pack_prompts(token_ids, num_prompt_tokens, padded_num_reqs, padded_prompt_len, pad_id)` — [`L58`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L58) — Pack prompt tokens into a padded tensor.
- `swap_rows(arr, i1, i2)` — [`L135`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L135) — Swap two rows in an array. — documented in [easydel-inference-esurge-runners-sequence_buffer](../../../../../concepts/easydel-inference-esurge-runners-sequence_buffer.md)
- `swap_rows_pytree(arrs, i1, i2)` — [`L171`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L171) — Swap rows across all arrays in a pytree.

## Module values
- `logger` — [`L54`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/runners/sequence_buffer.py#L54)

