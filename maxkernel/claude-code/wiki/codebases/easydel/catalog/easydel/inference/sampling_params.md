---
title: 'Module: easydel/inference/sampling_params.py'
type: catalog
provenance: extracted
module: easydel/inference/sampling_params.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.sampling_params`/
symbols:
  SamplingParams: SamplingParams#
  SamplingParams.__post_init__: SamplingParams#__post_init__().
  JitableSamplingParams.from_host_params: JitableSamplingParams#from_host_params().
  JitableSamplingParams.insert: JitableSamplingParams#insert().
  JitableSamplingParams.view_1d: JitableSamplingParams#view_1d().
  JitableSamplingParams.view_2d: JitableSamplingParams#view_2d().
  JitableSamplingParams.get_logits_warper: JitableSamplingParams#get_logits_warper().
  SamplingParams._verify_args: SamplingParams#_verify_args().
  SamplingParams.max_tokens: SamplingParams#max_tokens.
  JitableSamplingParams.get_logits_processor: JitableSamplingParams#get_logits_processor().
  SamplingParams.temperature: SamplingParams#temperature.
  GuidedDecodingParams.__post_init__: GuidedDecodingParams#__post_init__().
  JitableSamplingParams: JitableSamplingParams#
  SamplingParams.sampling_type: SamplingParams#sampling_type().
  SamplingParams.update_with_generation_config: SamplingParams#update_with_generation_config().
  JitableSamplingParams.update_idx1d: JitableSamplingParams#update_idx1d().
  SamplingParams.n: SamplingParams#n.
  SamplingParams.make_jitable: SamplingParams#make_jitable().
  SamplingParams.top_p: SamplingParams#top_p.
  SamplingParams.stop: SamplingParams#stop.
  SamplingParams.update_with_tokenizer: SamplingParams#update_with_tokenizer().
  SamplingParams.presence_penalty: SamplingParams#presence_penalty.
  SamplingParams.repetition_penalty: SamplingParams#repetition_penalty.
  SamplingParams.stop_token_ids: SamplingParams#stop_token_ids.
  SamplingParams._bad_words_token_ids: SamplingParams#_bad_words_token_ids.
  SamplingType: SamplingType#
  JitableSamplingParams.temperature: JitableSamplingParams#temperature.
  JitableSamplingParams.top_k: JitableSamplingParams#top_k.
  JitableSamplingParams.top_p: JitableSamplingParams#top_p.
  JitableSamplingParams.min_p: JitableSamplingParams#min_p.
  JitableSamplingParams.repetition_penalty: JitableSamplingParams#repetition_penalty.
  JitableSamplingParams.frequency_penalty: JitableSamplingParams#frequency_penalty.
  JitableSamplingParams.presence_penalty: JitableSamplingParams#presence_penalty.
  JitableSamplingParams.allowed_token_ids: JitableSamplingParams#allowed_token_ids.
  SamplingParams.frequency_penalty: SamplingParams#frequency_penalty.
  SamplingParams.top_k: SamplingParams#top_k.
  SamplingParams.min_tokens: SamplingParams#min_tokens.
  SamplingParams.prompt_logprobs: SamplingParams#prompt_logprobs.
  SamplingParams.output_kind: SamplingParams#output_kind.
  SamplingParams.logit_bias: SamplingParams#logit_bias.
  SamplingParams._all_stop_token_ids: SamplingParams#_all_stop_token_ids.
  SamplingParams.all_stop_token_ids: SamplingParams#all_stop_token_ids().
  SamplingParams.bad_words_token_ids: SamplingParams#bad_words_token_ids().
  SamplingParams.clone: SamplingParams#clone().
  JitableSamplingParams.random_sampling: JitableSamplingParams#random_sampling.
  JitableSamplingParams.max_tokens: JitableSamplingParams#max_tokens.
  JitableSamplingParams.min_tokens: JitableSamplingParams#min_tokens.
  SamplingParams.best_of: SamplingParams#best_of.
  SamplingParams.min_p: SamplingParams#min_p.
  SamplingParams.logprobs: SamplingParams#logprobs.
  SamplingParams.allowed_token_ids: SamplingParams#allowed_token_ids.
  SamplingParams._verify_greedy_sampling: SamplingParams#_verify_greedy_sampling().
  JitableSamplingParams.all_stop_token_ids: JitableSamplingParams#all_stop_token_ids.
  JitableSamplingParams.bad_words_token_ids: JitableSamplingParams#bad_words_token_ids.
  JitableSamplingParams.bad_words_lengths: JitableSamplingParams#bad_words_lengths.
  SamplingParams.bad_words: SamplingParams#bad_words.
  SamplingParams.ignore_eos: SamplingParams#ignore_eos.
  JitableSamplingParams.logits_processor: JitableSamplingParams#logits_processor().
  JitableSamplingParams.logits_warper: JitableSamplingParams#logits_warper().
  SamplingParams.guided_decoding: SamplingParams#guided_decoding.
  logger: logger.
  SamplingType.GREEDY: SamplingType#GREEDY.
  SamplingType.RANDOM: SamplingType#RANDOM.
  RequestOutputKind: RequestOutputKind#
  SamplingParams.seed: SamplingParams#seed.
  SamplingParams.skip_special_tokens: SamplingParams#skip_special_tokens.
  SamplingParams.extra_args: SamplingParams#extra_args.
  SamplingParams._real_n: SamplingParams#_real_n.
  RequestOutputKind.CUMULATIVE: RequestOutputKind#CUMULATIVE.
  GuidedDecodingParams: GuidedDecodingParams#
  GuidedDecodingParams.json: GuidedDecodingParams#json.
  GuidedDecodingParams.regex: GuidedDecodingParams#regex.
  GuidedDecodingParams.choice: GuidedDecodingParams#choice.
  GuidedDecodingParams.grammar: GuidedDecodingParams#grammar.
  GuidedDecodingParams.json_object: GuidedDecodingParams#json_object.
  SamplingParams.detokenize: SamplingParams#detokenize.
  SamplingParams.include_stop_str_in_output: SamplingParams#include_stop_str_in_output.
  SamplingParams.ignore_stop_strings_in_reasoning: SamplingParams#ignore_stop_strings_in_reasoning.
  SamplingParams._output_text_buffer_length: SamplingParams#_output_text_buffer_length.
  RequestOutputKind.DELTA: RequestOutputKind#DELTA.
  RequestOutputKind.FINAL_ONLY: RequestOutputKind#FINAL_ONLY.
  GuidedDecodingParams.backend: GuidedDecodingParams#backend.
  GuidedDecodingParams.backend_was_auto: GuidedDecodingParams#backend_was_auto.
  GuidedDecodingParams.disable_fallback: GuidedDecodingParams#disable_fallback.
  GuidedDecodingParams.disable_any_whitespace: GuidedDecodingParams#disable_any_whitespace.
  GuidedDecodingParams.disable_additional_properties: GuidedDecodingParams#disable_additional_properties.
  GuidedDecodingParams.whitespace_pattern: GuidedDecodingParams#whitespace_pattern.
  GuidedDecodingParams.structural_tag: GuidedDecodingParams#structural_tag.
  JitableSamplingParams.init_empty: JitableSamplingParams#init_empty().
  JitableSamplingParams.make_jitable: JitableSamplingParams#make_jitable().
  SamplingParams.stop_pattern: SamplingParams#stop_pattern.
  SamplingParams.spaces_between_special_tokens: SamplingParams#spaces_between_special_tokens.
  SamplingParams.truncate_prompt_tokens: SamplingParams#truncate_prompt_tokens.
  BeamSearchParams: BeamSearchParams#
  BeamSearchParams.beam_width: BeamSearchParams#beam_width.
  BeamSearchParams.max_tokens: BeamSearchParams#max_tokens.
  BeamSearchParams.ignore_eos: BeamSearchParams#ignore_eos.
  BeamSearchParams.temperature: BeamSearchParams#temperature.
  BeamSearchParams.length_penalty: BeamSearchParams#length_penalty.
  BeamSearchParams.include_stop_str_in_output: BeamSearchParams#include_stop_str_in_output.
---
# Module: [`easydel/inference/sampling_params.py`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py)

## Classes
### `BeamSearchParams`
- def: [`easydel/inference/sampling_params.py:727`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L727)
- doc: Beam search parameters for text generation.
- signature: `class BeamSearchParams:`
- members:
  - `beam_width` — [`L755`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L755)
  - `ignore_eos` — [`L757`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L757)
  - `include_stop_str_in_output` — [`L760`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L760)
  - `length_penalty` — [`L759`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L759)
  - `max_tokens` — [`L756`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L756)
  - `temperature` — [`L758`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L758)

### `GuidedDecodingParams`
- def: [`easydel/inference/sampling_params.py:106`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L106)
- doc: Parameters for guided (constrained) decoding.
- signature: `class GuidedDecodingParams:`
- members:
  - `__post_init__(self)` — [`L148`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L148) — Validate that only one guided decoding mode is specified.
  - `backend` — [`L140`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L140)
  - `backend_was_auto` — [`L141`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L141)
  - `choice` — [`L137`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L137)
  - `disable_additional_properties` — [`L144`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L144)
  - `disable_any_whitespace` — [`L143`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L143)
  - `disable_fallback` — [`L142`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L142)
  - `grammar` — [`L138`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L138)
  - `json` — [`L135`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L135)
  - `json_object` — [`L139`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L139)
  - `regex` — [`L136`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L136)
  - `structural_tag` — [`L146`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L146)
  - `whitespace_pattern` — [`L145`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L145)
- used by: [`guided_decoding`](sampling_params.md#SamplingParams.guided_decoding)

### `JitableSamplingParams`
- def: [`easydel/inference/sampling_params.py:170`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L170)
- doc: A JAX-native, device-ready version of sampling parameters.
- signature: `class JitableSamplingParams:`
- members:
  - `from_host_params(cls, params: SamplingParams)` — [`L321`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L321) — Convert host-side SamplingParams to a JIT-compatible version.
  - `get_logits_processor(self)` — [`L378`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L378) — Construct a LogitsProcessorList containing configured logits processors.
  - `get_logits_warper(self)` — [`L360`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L360) — Construct a LogitsProcessorList containing configured logits warpers.
  - `init_empty(cls, batch_size: int)` — [`L295`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L295) — Create an empty JitableSamplingParams with default values.
  - `insert(self, second_sample: JitableSamplingParams, slot: int)` — [`L212`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L212) — Insert another sample's parameters at the specified slot index.
  - `logits_processor(self)` — [`L395`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L395) — Cached logits processor list.
  - `logits_warper(self)` — [`L404`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L404) — Cached logits warper list.
  - `make_jitable(self)` — [`L412`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L412) — Return self since this is already a JIT-compatible class.
  - `update_idx1d(x, y)` — [`L226`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L226) — Update array x at the given slot with values from y, preserving sharding.
  - `view_1d(self)` — [`L248`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L248) — Reshape all parameter arrays to 1D (batch_size,).
  - `view_2d(self)` — [`L271`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L271) — Reshape all parameter arrays to 2D (batch_size, 1).
  - `all_stop_token_ids` — [`L207`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L207)
  - `allowed_token_ids` — [`L210`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L210)
  - `bad_words_lengths` — [`L209`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L209)
  - `bad_words_token_ids` — [`L208`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L208)
  - `frequency_penalty` — [`L201`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L201)
  - `max_tokens` — [`L204`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L204)
  - `min_p` — [`L199`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L199)
  - `min_tokens` — [`L205`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L205)
  - `presence_penalty` — [`L202`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L202)
  - `random_sampling` — [`L194`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L194)
  - `repetition_penalty` — [`L200`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L200)
  - `temperature` — [`L196`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L196)
  - `top_k` — [`L197`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L197)
  - `top_p` — [`L198`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L198)
- uses (calls/refs, reference-scoped): [`SamplingParams`](sampling_params.md#SamplingParams), [`LogitsProcessorList`](logits_process.md#LogitsProcessorList), [`max_tokens`](sampling_params.md#SamplingParams.max_tokens), [`temperature`](sampling_params.md#SamplingParams.temperature), [`sampling_type`](sampling_params.md#SamplingParams.sampling_type), [`top_p`](sampling_params.md#SamplingParams.top_p), [`_bad_words_token_ids`](sampling_params.md#SamplingParams._bad_words_token_ids), [`presence_penalty`](sampling_params.md#SamplingParams.presence_penalty), [`repetition_penalty`](sampling_params.md#SamplingParams.repetition_penalty), [`FrequencyPenaltyLogitsProcessor`](logits_process.md#FrequencyPenaltyLogitsProcessor), [`PresencePenaltyLogitsProcessor`](logits_process.md#PresencePenaltyLogitsProcessor), [`RepetitionPenaltyLogitsProcessor`](logits_process.md#RepetitionPenaltyLogitsProcessor), [`TemperatureLogitsWarper`](logits_process.md#TemperatureLogitsWarper), [`TopKLogitsWarper`](logits_process.md#TopKLogitsWarper), [`TopPLogitsWarper`](logits_process.md#TopPLogitsWarper), [`SamplingType`](sampling_params.md#SamplingType), [`frequency_penalty`](sampling_params.md#SamplingParams.frequency_penalty), [`min_tokens`](sampling_params.md#SamplingParams.min_tokens), [`top_k`](sampling_params.md#SamplingParams.top_k), [`MinPLogitsWarper`](logits_process.md#MinPLogitsWarper), [`all_stop_token_ids`](sampling_params.md#SamplingParams.all_stop_token_ids), [`allowed_token_ids`](sampling_params.md#SamplingParams.allowed_token_ids), [`min_p`](sampling_params.md#SamplingParams.min_p), [`RANDOM`](sampling_params.md#SamplingType.RANDOM)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`make_jitable`](sampling_params.md#SamplingParams.make_jitable)

### `RequestOutputKind`  ·  implements/extends Enum
- def: [`easydel/inference/sampling_params.py:88`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L88)
- doc: Defines the kind of output format for a generation request.
- signature: `class RequestOutputKind(Enum):`
- members:
  - `CUMULATIVE` — [`L100`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L100)
  - `DELTA` — [`L101`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L101)
  - `FINAL_ONLY` — [`L102`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L102)
- used by: [`output_kind`](sampling_params.md#SamplingParams.output_kind)

### `SamplingParams`
- def: [`easydel/inference/sampling_params.py:422`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L422)
- members:
  - `__post_init__(self)` — [`L518`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L518) — Initialize and validate parameters after construction.
  - `_verify_args(self)` — [`L569`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L569) — Perform detailed validation of parameter values.
  - `_verify_greedy_sampling(self)` — [`L592`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L592) — Validate parameters are compatible with greedy sampling.
  - `all_stop_token_ids(self)` — [`L684`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L684) — Get all stop token IDs including EOS.
  - `bad_words_token_ids(self)` — [`L693`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L693) — Get the tokenized versions of bad_words.
  - `clone(self)` — [`L717`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L717) — Create a deep copy of this instance.
  - `make_jitable(self)` — [`L701`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L701) — Convert to a JAX-jittable JitableSamplingParams object.
  - `sampling_type(self)` — [`L673`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L673) — Determine the sampling type based on temperature.
  - `update_with_generation_config(self, generation_config: dict[str, Any], model_eos_token_id: int | None = None)` — [`L601`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L601) — Update sampling parameters with model's generation config.
  - `update_with_tokenizer(self, tokenizer: AutoTokenizer)` — [`L634`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L634) — Update sampling parameters with tokenized bad words.
  - `allowed_token_ids` — [`L508`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L508)
  - `bad_words` — [`L489`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L489)
  - `best_of` — [`L475`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L475)
  - `detokenize` — [`L497`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L497)
  - `extra_args` — [`L509`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L509)
  - `frequency_penalty` — [`L477`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L477)
  - `guided_decoding` — [`L506`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L506)
  - `ignore_eos` — [`L490`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L490)
  - `ignore_stop_strings_in_reasoning` — [`L502`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L502)
  - `include_stop_str_in_output` — [`L500`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L500)
  - `logit_bias` — [`L507`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L507)
  - `logprobs` — [`L495`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L495)
  - `max_tokens` — [`L491`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L491)
  - `min_p` — [`L481`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L481)
  - `min_tokens` — [`L492`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L492)
  - `n` — [`L474`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L474)
  - `output_kind` — [`L501`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L501)
  - `presence_penalty` — [`L476`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L476)
  - `prompt_logprobs` — [`L496`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L496)
  - `repetition_penalty` — [`L478`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L478)
  - `seed` — [`L483`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L483)
  - `skip_special_tokens` — [`L498`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L498)
  - `spaces_between_special_tokens` — [`L499`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L499)
  - `stop` — [`L486`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L486)
  - `stop_pattern` — [`L488`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L488)
  - `stop_token_ids` — [`L487`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L487)
  - `temperature` — [`L479`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L479)
  - `top_k` — [`L482`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L482)
  - `top_p` — [`L480`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L480)
  - `truncate_prompt_tokens` — [`L505`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L505)
- protocol/private: `_all_stop_token_ids`[`L515`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L515), `_bad_words_token_ids`[`L516`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L516), `_output_text_buffer_length`[`L514`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L514), `_real_n`[`L513`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L513)
- uses (calls/refs, reference-scoped): [`from_host_params`](sampling_params.md#JitableSamplingParams.from_host_params), [`JitableSamplingParams`](sampling_params.md#JitableSamplingParams), [`SamplingType`](sampling_params.md#SamplingType), [`RequestOutputKind`](sampling_params.md#RequestOutputKind), [`GREEDY`](sampling_params.md#SamplingType.GREEDY), [`RANDOM`](sampling_params.md#SamplingType.RANDOM), [`logger`](sampling_params.md#logger), [`GuidedDecodingParams`](sampling_params.md#GuidedDecodingParams), [`CUMULATIVE`](sampling_params.md#RequestOutputKind.CUMULATIVE)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`generate_unified`](../trainers/base_trainer.md#BaseTrainer.generate_unified), [`stream`](esurge/mixins/io.md#EngineIOMixin.stream), [`_add_request`](esurge/mixins/requests.md#EngineRequestsMixin._add_request), [`_generate`](evaluations/esurge_eval.md#eSurgeLMEvalAdapter._generate), [`_process_sampling_params`](esurge/runners/sequence_buffer.md#SequenceBuffer._process_sampling_params), [`_process_optional_params`](esurge/runners/sequence_buffer.md#SequenceBuffer._process_optional_params), [`from_host_params`](sampling_params.md#JitableSamplingParams.from_host_params), [`check_stop`](esurge/scheduler/utils.md#check_stop), [`get_computed_pages`](esurge/core/manager.md#CacheManager.get_computed_pages), [`_create_sampling_params`](esurge/server/api_server.md#eSurgeApiServer._create_sampling_params), [`_create_sampling_params_from_responses`](inference_engine_interface.md#BaseInferenceApiServer._create_sampling_params_from_responses), [`EngineIOMixin`](esurge/mixins/io.md#EngineIOMixin), [`_prepare_sampling_params`](esurge/server/api_server.md#eSurgeApiServer._prepare_sampling_params), [`_chat_multimodal`](esurge/mixins/io.md#EngineIOMixin._chat_multimodal), [`iter_responses_stream`](esurge/mixins/io.md#EngineIOMixin.iter_responses_stream), [`_finalize`](esurge/mixins/utils.md#EngineUtilsMixin._finalize), [`generate`](esurge/mixins/io.md#EngineIOMixin.generate), [`iter_chat_completion_stream`](esurge/mixins/io.md#EngineIOMixin.iter_chat_completion_stream), [`_prepare_chat_sampling_params`](esurge/mixins/utils.md#EngineUtilsMixin._prepare_chat_sampling_params), [`_apply_extra_stops_to_sampling_params`](esurge/server/api_server.md#eSurgeApiServer._apply_extra_stops_to_sampling_params), [`chat`](esurge/mixins/io.md#EngineIOMixin.chat), [`esurge_generate`](../infra/mixins/generation.md#EasyGenerationMixin.esurge_generate), [`__init__`](evaluations/esurge_eval.md#eSurgeLMEvalAdapter.__init__), [`_create_sampling_params`](inference_engine_interface.md#BaseInferenceApiServer._create_sampling_params), [`_prepare_sampling_params_for_request`](esurge/mixins/utils.md#EngineUtilsMixin._prepare_sampling_params_for_request), [`generate`](esurge/server/api_server.md#eSurgeAdapter.generate), [`RefineSamplingParamsFn`](esurge/server/api_server.md#RefineSamplingParamsFn), [`_apply_extra_stops_to_sampling_params`](esurge/mixins/utils.md#EngineUtilsMixin._apply_extra_stops_to_sampling_params), [`_apply_generation_config_to_sampling_params`](esurge/mixins/utils.md#EngineUtilsMixin._apply_generation_config_to_sampling_params), [`max_tokens`](esurge/request.md#EngineRequest.max_tokens), [`_call_esurge_engine`](../infra/mixins/generation.md#EasyGenerationMixin._call_esurge_engine), [`_clone_sampling_params`](esurge/mixins/utils.md#EngineUtilsMixin._clone_sampling_params), [`sampling_params`](esurge/scheduler/output.md#NewRequestData.sampling_params), [`_coerce_sampling_params_template`](evaluations/esurge_eval.md#_coerce_sampling_params_template), [`use_structured_output`](esurge/request.md#EngineRequest.use_structured_output), [`__init__`](esurge/request.md#EngineRequest.__init__), [`generate`](inference_engine_interface.md#InferenceEngineAdapter.generate), [`set_sampling_params_callback`](esurge/esurge_engine.md#eSurge.set_sampling_params_callback), [`SamplingCallable`](esurge/esurge_engine.md#SamplingCallable)  (+4 more)

### `SamplingType`  ·  implements/extends IntEnum
- def: [`easydel/inference/sampling_params.py:74`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L74)
- doc: Defines the sampling strategy for token selection.
- signature: `class SamplingType(IntEnum):`
- members:
  - `GREEDY` — [`L84`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L84)
  - `RANDOM` — [`L85`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L85)
- used by: [`_process_sampling_params`](esurge/runners/sequence_buffer.md#SequenceBuffer._process_sampling_params), [`from_host_params`](sampling_params.md#JitableSamplingParams.from_host_params), [`sampling_type`](sampling_params.md#SamplingParams.sampling_type)

## Module values
- `logger` — [`L71`](../../../../../../raw/code/EasyDeL/easydel/inference/sampling_params.py#L71)

