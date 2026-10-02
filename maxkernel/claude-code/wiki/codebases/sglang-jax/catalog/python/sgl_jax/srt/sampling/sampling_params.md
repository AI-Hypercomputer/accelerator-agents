---
title: 'Module: python/sgl_jax/srt/sampling/sampling_params.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/sampling/sampling_params.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.sampling.sampling_params`/
symbols:
  SamplingParams.verify: SamplingParams#verify().
  SamplingParams.max_new_tokens: SamplingParams#max_new_tokens.
  DEFAULT_SAMPLING_SEED: DEFAULT_SAMPLING_SEED.
  SamplingParams.stop_token_ids: SamplingParams#stop_token_ids.
  SamplingParams.sampling_seed: SamplingParams#sampling_seed.
  SamplingParams.top_k: SamplingParams#top_k.
  SamplingParams.stop_strs: SamplingParams#stop_strs.
  SamplingParams.stop_str_max_len: SamplingParams#stop_str_max_len.
  TOP_K_ALL: TOP_K_ALL.
  SamplingParams.logit_bias: SamplingParams#logit_bias.
  SamplingParams.temperature: SamplingParams#temperature.
  SamplingParams.json_schema: SamplingParams#json_schema.
  SamplingParams.regex: SamplingParams#regex.
  SamplingParams.ebnf: SamplingParams#ebnf.
  SamplingParams.top_p: SamplingParams#top_p.
  SamplingParams.min_p: SamplingParams#min_p.
  SamplingParams: SamplingParams#
  SamplingParams.min_new_tokens: SamplingParams#min_new_tokens.
  SamplingParams.ignore_eos: SamplingParams#ignore_eos.
  SamplingParams.structural_tag: SamplingParams#structural_tag.
  SamplingParams.normalize: SamplingParams#normalize().
  SamplingParams.frequency_penalty: SamplingParams#frequency_penalty.
  SamplingParams.presence_penalty: SamplingParams#presence_penalty.
  SamplingParams.repetition_penalty: SamplingParams#repetition_penalty.
  SamplingParams.stream_interval: SamplingParams#stream_interval.
  SamplingParams.skip_special_tokens: SamplingParams#skip_special_tokens.
  SamplingParams.spaces_between_special_tokens: SamplingParams#spaces_between_special_tokens.
  SamplingParams.no_stop_trim: SamplingParams#no_stop_trim.
  _SAMPLING_EPS: _SAMPLING_EPS.
  SamplingParams.__init__: SamplingParams#__init__().
  SamplingParams.n: SamplingParams#n.
  SamplingParams.convert_to_dict: SamplingParams#convert_to_dict().
---
# Module: [`python/sgl_jax/srt/sampling/sampling_params.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py)

## Classes
### `SamplingParams`
- def: [`python/sgl_jax/srt/sampling/sampling_params.py:10`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L10)
- doc: The sampling parameters.
- signature: `class SamplingParams:`
- members:
  - `convert_to_dict(self)` — [`L145`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L145)
  - `normalize(self, tokenizer)` — [`L127`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L127)
  - `verify(self, vocab_size)` — [`L84`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L84)
  - `ebnf` — [`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L62)
  - `frequency_penalty` — [`L55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L55)
  - `ignore_eos` — [`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L64)
  - `json_schema` — [`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L61)
  - `logit_bias` — [`L69`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L69)
  - `max_new_tokens` — [`L45`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L45)
  - `min_new_tokens` — [`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L58)
  - `min_p` — [`L54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L54)
  - `n` — [`L60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L60)
  - `no_stop_trim` — [`L67`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L67)
  - `presence_penalty` — [`L56`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L56)
  - `regex` — [`L59`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L59)
  - `repetition_penalty` — [`L57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L57)
  - `sampling_seed` — [`L74`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L74)
  - `skip_special_tokens` — [`L65`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L65)
  - `spaces_between_special_tokens` — [`L66`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L66)
  - `stop_str_max_len` — [`L131`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L131)
  - `stop_strs` — [`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L46)
  - `stop_token_ids` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L48)
  - `stream_interval` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L68)
  - `structural_tag` — [`L63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L63)
  - `temperature` — [`L51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L51)
  - `top_k` — [`L53`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L53)
  - `top_p` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L52)
- protocol/private: `__init__`[`L19`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L19)
- uses (calls/refs, reference-scoped): [`get_bool_env_var`](../utils/common_utils.md#get_bool_env_var), [`DEFAULT_SAMPLING_SEED`](sampling_params.md#DEFAULT_SAMPLING_SEED), [`TOP_K_ALL`](sampling_params.md#TOP_K_ALL), [`_SAMPLING_EPS`](sampling_params.md#_SAMPLING_EPS)
- used by: [`handle_generate_request`](../managers/scheduler.md#Scheduler.handle_generate_request), [`stream_output_generation`](../managers/scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.stream_output_generation), [`add_one_req`](../managers/schedule_policy.md#PrefillAdder.add_one_req), [`add_one_req_ignore_eos`](../managers/schedule_policy.md#PrefillAdder.add_one_req_ignore_eos), `init_new`, `retract_decode`, [`add_chunked_req`](../managers/schedule_policy.md#PrefillAdder.add_chunked_req), `check_finished`, [`from_schedule_batch`](sampling_batch_info.md#SamplingBatchInfo.from_schedule_batch), `_check_token_based_finish`, [`log_input_tokens`](../managers/schedule_policy.md#PrefillAdder.log_input_tokens), `_check_str_based_finish`, `adjust_max_prefix_ids`, `_check_vocab_boundary_finish`, [`_sort_by_longest_output`](../managers/schedule_policy.md#SchedulePolicy._sort_by_longest_output), `__init__`

## Module values
- `DEFAULT_SAMPLING_SEED` — [`L7`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L7)
- `TOP_K_ALL` — [`L6`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L6)
- `_SAMPLING_EPS` — [`L5`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/sampling/sampling_params.py#L5)

