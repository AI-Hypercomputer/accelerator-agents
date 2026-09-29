---
title: 'Module: easydel/inference/logits_process.py'
type: catalog
provenance: extracted
module: easydel/inference/logits_process.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.logits_process`/
symbols:
  LogitsProcessor: LogitsProcessor#
  LogitsProcessorList: LogitsProcessorList#
  WhisperTimeStampLogitsProcessor.__call__: WhisperTimeStampLogitsProcessor#__call__().
  LogitsWarper: LogitsWarper#
  TopKLogitsWarper.__call__: TopKLogitsWarper#__call__().
  NoRepeatNGramLogitsProcessor.ngram_size: NoRepeatNGramLogitsProcessor#ngram_size.
  WhisperTimeStampLogitsProcessor.handle_pairs: WhisperTimeStampLogitsProcessor#handle_pairs().
  NoRepeatNGramLogitsProcessor.true_fn: NoRepeatNGramLogitsProcessor#true_fn().
  TopPLogitsWarper.__call__: TopPLogitsWarper#__call__().
  WhisperTimeStampLogitsProcessor: WhisperTimeStampLogitsProcessor#
  WhisperTimeStampLogitsProcessor.max_initial_timestamp_index: WhisperTimeStampLogitsProcessor#max_initial_timestamp_index.
  WhisperTimeStampLogitsProcessor.timestamp_begin: WhisperTimeStampLogitsProcessor#timestamp_begin.
  TemperatureLogitsWarper: TemperatureLogitsWarper#
  TopPLogitsWarper: TopPLogitsWarper#
  TopKLogitsWarper: TopKLogitsWarper#
  ForceTokensLogitsProcessor: ForceTokensLogitsProcessor#
  WhisperTimeStampLogitsProcessor.begin_index: WhisperTimeStampLogitsProcessor#begin_index.
  PresencePenaltyLogitsProcessor: PresencePenaltyLogitsProcessor#
  FrequencyPenaltyLogitsProcessor: FrequencyPenaltyLogitsProcessor#
  RepetitionPenaltyLogitsProcessor: RepetitionPenaltyLogitsProcessor#
  MinPLogitsWarper._apply: MinPLogitsWarper#_apply().
  NoRepeatNGramLogitsProcessor.forward: NoRepeatNGramLogitsProcessor#forward().
  NoRepeatNGramLogitsProcessor.get_previous_ngrams: NoRepeatNGramLogitsProcessor#get_previous_ngrams().
  add_start_docstrings: add_start_docstrings().
  LogitsProcessor.__call__: LogitsProcessor#__call__().
  LogitsWarper.__call__: LogitsWarper#__call__().
  LogitsProcessorList.__call__: LogitsProcessorList#__call__().
  ForcedEOSTokenLogitsProcessor.__call__: ForcedEOSTokenLogitsProcessor#__call__().
  MinLengthLogitsProcessor.__post_init__: MinLengthLogitsProcessor#__post_init__().
  MinLengthLogitsProcessor.__call__: MinLengthLogitsProcessor#__call__().
  SuppressTokensAtBeginLogitsProcessor.__call__: SuppressTokensAtBeginLogitsProcessor#__call__().
  ForceTokensLogitsProcessor.__call__: ForceTokensLogitsProcessor#__call__().
  PresencePenaltyLogitsProcessor.__call__: PresencePenaltyLogitsProcessor#__call__().
  FrequencyPenaltyLogitsProcessor.__call__: FrequencyPenaltyLogitsProcessor#__call__().
  RepetitionPenaltyLogitsProcessor.__call__: RepetitionPenaltyLogitsProcessor#__call__().
  MinPLogitsWarper.__call__: MinPLogitsWarper#__call__().
  NoRepeatNGramLogitsProcessor.__call__: NoRepeatNGramLogitsProcessor#__call__().
  ForcedBOSTokenLogitsProcessor: ForcedBOSTokenLogitsProcessor#
  ForcedEOSTokenLogitsProcessor: ForcedEOSTokenLogitsProcessor#
  MinLengthLogitsProcessor: MinLengthLogitsProcessor#
  SuppressTokensAtBeginLogitsProcessor: SuppressTokensAtBeginLogitsProcessor#
  SuppressTokensLogitsProcessor: SuppressTokensLogitsProcessor#
  MinPLogitsWarper: MinPLogitsWarper#
  NoRepeatNGramLogitsProcessor: NoRepeatNGramLogitsProcessor#
  MinLengthLogitsProcessor.min_length: MinLengthLogitsProcessor#min_length.
  MinLengthLogitsProcessor.eos_token_id: MinLengthLogitsProcessor#eos_token_id.
  TopPLogitsWarper._apply: TopPLogitsWarper#_apply().
  TopKLogitsWarper._filter_scores: TopKLogitsWarper#_filter_scores().
  ForceTokensLogitsProcessor._force_token: ForceTokensLogitsProcessor#_force_token().
  WhisperTimeStampLogitsProcessor.handle_cumulative_probs: WhisperTimeStampLogitsProcessor#handle_cumulative_probs().
  NoRepeatNGramLogitsProcessor.body_fun: NoRepeatNGramLogitsProcessor#body_fun().
  NoRepeatNGramLogitsProcessor.get_banned_tokens_mask: NoRepeatNGramLogitsProcessor#get_banned_tokens_mask().
  LOGITS_PROCESSOR_INPUTS_DOCSTRING: LOGITS_PROCESSOR_INPUTS_DOCSTRING.
  ForceTokensLogitsProcessor.force_token_array: ForceTokensLogitsProcessor#force_token_array.
  LogitsProcessor.__hash__: LogitsProcessor#__hash__.
  LogitsWarper.__hash__: LogitsWarper#__hash__.
  EmptyProcessor: EmptyProcessor#
  LogitsProcessorList.__hash__: LogitsProcessorList#__hash__.
  TemperatureLogitsWarper.__call__: TemperatureLogitsWarper#__call__().
  ForcedBOSTokenLogitsProcessor.__call__: ForcedBOSTokenLogitsProcessor#__call__().
  SuppressTokensLogitsProcessor.__call__: SuppressTokensLogitsProcessor#__call__().
  TemperatureLogitsWarper.temperature: TemperatureLogitsWarper#temperature.
  SuppressTokensLogitsProcessor.suppress_tokens: SuppressTokensLogitsProcessor#suppress_tokens.
  PresencePenaltyLogitsProcessor.presence_penalty: PresencePenaltyLogitsProcessor#presence_penalty.
  FrequencyPenaltyLogitsProcessor.frequency_penalty: FrequencyPenaltyLogitsProcessor#frequency_penalty.
  RepetitionPenaltyLogitsProcessor.repetition_penalty: RepetitionPenaltyLogitsProcessor#repetition_penalty.
  add_start_docstrings.docstring_decorator: add_start_docstrings().docstring_decorator().
  TopPLogitsWarper.top_p: TopPLogitsWarper#top_p.
  TopPLogitsWarper.filter_value: TopPLogitsWarper#filter_value.
  TopPLogitsWarper.min_tokens_to_keep: TopPLogitsWarper#min_tokens_to_keep.
  TopKLogitsWarper.top_k: TopKLogitsWarper#top_k.
  TopKLogitsWarper.filter_value: TopKLogitsWarper#filter_value.
  TopKLogitsWarper.min_tokens_to_keep: TopKLogitsWarper#min_tokens_to_keep.
  TopKLogitsWarper._identity: TopKLogitsWarper#_identity().
  ForcedBOSTokenLogitsProcessor.bos_token_id: ForcedBOSTokenLogitsProcessor#bos_token_id.
  ForcedEOSTokenLogitsProcessor.max_length: ForcedEOSTokenLogitsProcessor#max_length.
  ForcedEOSTokenLogitsProcessor.eos_token_id: ForcedEOSTokenLogitsProcessor#eos_token_id.
  SuppressTokensAtBeginLogitsProcessor.begin_suppress_tokens: SuppressTokensAtBeginLogitsProcessor#begin_suppress_tokens.
  SuppressTokensAtBeginLogitsProcessor.begin_index: SuppressTokensAtBeginLogitsProcessor#begin_index.
  WhisperTimeStampLogitsProcessor.eos_token_id: WhisperTimeStampLogitsProcessor#eos_token_id.
  WhisperTimeStampLogitsProcessor.no_timestamps_token_id: WhisperTimeStampLogitsProcessor#no_timestamps_token_id.
  PresencePenaltyLogitsProcessor._apply: PresencePenaltyLogitsProcessor#_apply().
  FrequencyPenaltyLogitsProcessor._apply: FrequencyPenaltyLogitsProcessor#_apply().
  RepetitionPenaltyLogitsProcessor._apply: RepetitionPenaltyLogitsProcessor#_apply().
  MinPLogitsWarper.min_p: MinPLogitsWarper#min_p.
  MinPLogitsWarper.filter_value: MinPLogitsWarper#filter_value.
  MinPLogitsWarper.min_tokens_to_keep: MinPLogitsWarper#min_tokens_to_keep.
  NoRepeatNGramLogitsProcessor.inner_fn: NoRepeatNGramLogitsProcessor#inner_fn().
  logger: logger.
  EmptyProcessor.__call__: EmptyProcessor#__call__().
  ForceTokensLogitsProcessor.__init__: ForceTokensLogitsProcessor#__init__().
  WhisperTimeStampLogitsProcessor.__init__: WhisperTimeStampLogitsProcessor#__init__().
---
# Module: [`easydel/inference/logits_process.py`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py)

## Classes
### `EmptyProcessor`  ·  implements/extends LogitsProcessor
- def: [`easydel/inference/logits_process.py:169`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L169)
- doc: A placeholder `LogitsProcessor` that performs no operation.
- signature: `class EmptyProcessor(LogitsProcessor):`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L178`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L178) — Return scores unchanged (no-op processor).
- uses (calls/refs, reference-scoped): [`LogitsProcessor`](logits_process.md#LogitsProcessor)
- used by: [`LogitsProcessor`](logits_process.md#LogitsProcessor)

### `ForceTokensLogitsProcessor`  ·  implements/extends LogitsProcessor
- def: [`easydel/inference/logits_process.py:575`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L575)
- doc: \[`LogitsProcessor`\] that forces specific tokens to be generated at predefined
- signature: `class ForceTokensLogitsProcessor(LogitsProcessor):`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L611`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L611) — Force a specific token at predefined generation steps.
  - `__init__(self, force_token_map)` — [`L593`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L593) — Initialize with a mapping from generation indices to forced token IDs.
  - `force_token_array` — [`L609`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L609)
- protocol/private: `_force_token`[`L623`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L623)
- uses (calls/refs, reference-scoped): [`LogitsProcessor`](logits_process.md#LogitsProcessor)
- used by: [`__all__`](../modules/whisper/modeling_whisper.md#__all__), [`_get_logits_processor`](../infra/mixins/generation.md#EasyGenerationMixin._get_logits_processor), [`LogitsProcessor`](logits_process.md#LogitsProcessor), [`_force_generate`](../modules/whisper/modeling_whisper.md#WhisperForConditionalGeneration._force_generate)

### `ForcedBOSTokenLogitsProcessor`  ·  implements/extends LogitsProcessor
- def: [`easydel/inference/logits_process.py:399`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L399)
- doc: \[`LogitsProcessor`\] that ensures the beginning-of-sequence (BOS) token is
- signature: `class ForcedBOSTokenLogitsProcessor(LogitsProcessor):`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L414`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L414) — Force the BOS token at the first generation step.
  - `bos_token_id` — [`L412`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L412)
- uses (calls/refs, reference-scoped): [`LogitsProcessor`](logits_process.md#LogitsProcessor)
- used by: [`_get_logits_processor`](../infra/mixins/generation.md#EasyGenerationMixin._get_logits_processor), [`LogitsProcessor`](logits_process.md#LogitsProcessor)

### `ForcedEOSTokenLogitsProcessor`  ·  implements/extends LogitsProcessor
- def: [`easydel/inference/logits_process.py:432`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L432)
- doc: \[`LogitsProcessor`\] that forces the end-of-sequence (EOS) token to be generated
- signature: `class ForcedEOSTokenLogitsProcessor(LogitsProcessor):`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L449`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L449) — Force the EOS token when generation reaches max_length.
  - `eos_token_id` — [`L447`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L447)
  - `max_length` — [`L446`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L446)
- uses (calls/refs, reference-scoped): [`LogitsProcessor`](logits_process.md#LogitsProcessor)
- used by: [`_get_logits_processor`](../infra/mixins/generation.md#EasyGenerationMixin._get_logits_processor), [`LogitsProcessor`](logits_process.md#LogitsProcessor)

### `FrequencyPenaltyLogitsProcessor`  ·  implements/extends LogitsProcessor
- def: [`easydel/inference/logits_process.py:828`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L828)
- doc: \[`LogitsProcessor`\] that penalizes tokens based on their frequency (number of
- signature: `class FrequencyPenaltyLogitsProcessor(LogitsProcessor):`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L844`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L844) — Subtract a count-proportional penalty from logits of repeated tokens.
  - `frequency_penalty` — [`L842`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L842)
- protocol/private: `_apply`[`L856`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L856)
- uses (calls/refs, reference-scoped): [`LogitsProcessor`](logits_process.md#LogitsProcessor)
- used by: [`_get_logits_processor`](../infra/mixins/generation.md#EasyGenerationMixin._get_logits_processor), [`LogitsProcessor`](logits_process.md#LogitsProcessor), [`get_logits_processor`](sampling_params.md#JitableSamplingParams.get_logits_processor)

### `LogitsProcessor`
- def: [`easydel/inference/logits_process.py:98`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L98)
- doc: Abstract base class for all logit processors.
- signature: `class LogitsProcessor:`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L112`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L112) — Applies the processor to the logits.
- protocol/private: `__hash__`[`L130`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L130)
- uses (calls/refs, reference-scoped): [`hash_fn`](../utils/compiling_utils.md#hash_fn), [`WhisperTimeStampLogitsProcessor`](logits_process.md#WhisperTimeStampLogitsProcessor), [`ForceTokensLogitsProcessor`](logits_process.md#ForceTokensLogitsProcessor), [`FrequencyPenaltyLogitsProcessor`](logits_process.md#FrequencyPenaltyLogitsProcessor), [`PresencePenaltyLogitsProcessor`](logits_process.md#PresencePenaltyLogitsProcessor), [`RepetitionPenaltyLogitsProcessor`](logits_process.md#RepetitionPenaltyLogitsProcessor), [`add_start_docstrings`](logits_process.md#add_start_docstrings), [`ForcedBOSTokenLogitsProcessor`](logits_process.md#ForcedBOSTokenLogitsProcessor), [`ForcedEOSTokenLogitsProcessor`](logits_process.md#ForcedEOSTokenLogitsProcessor), [`MinLengthLogitsProcessor`](logits_process.md#MinLengthLogitsProcessor), [`NoRepeatNGramLogitsProcessor`](logits_process.md#NoRepeatNGramLogitsProcessor), [`SuppressTokensAtBeginLogitsProcessor`](logits_process.md#SuppressTokensAtBeginLogitsProcessor), [`SuppressTokensLogitsProcessor`](logits_process.md#SuppressTokensLogitsProcessor), [`LOGITS_PROCESSOR_INPUTS_DOCSTRING`](logits_process.md#LOGITS_PROCESSOR_INPUTS_DOCSTRING), [`EmptyProcessor`](logits_process.md#EmptyProcessor)
- used by: [`WhisperTimeStampLogitsProcessor`](logits_process.md#WhisperTimeStampLogitsProcessor), [`ForceTokensLogitsProcessor`](logits_process.md#ForceTokensLogitsProcessor), [`FrequencyPenaltyLogitsProcessor`](logits_process.md#FrequencyPenaltyLogitsProcessor), [`PresencePenaltyLogitsProcessor`](logits_process.md#PresencePenaltyLogitsProcessor), [`RepetitionPenaltyLogitsProcessor`](logits_process.md#RepetitionPenaltyLogitsProcessor), [`ForcedBOSTokenLogitsProcessor`](logits_process.md#ForcedBOSTokenLogitsProcessor), [`ForcedEOSTokenLogitsProcessor`](logits_process.md#ForcedEOSTokenLogitsProcessor), [`MinLengthLogitsProcessor`](logits_process.md#MinLengthLogitsProcessor), [`NoRepeatNGramLogitsProcessor`](logits_process.md#NoRepeatNGramLogitsProcessor), [`SuppressTokensAtBeginLogitsProcessor`](logits_process.md#SuppressTokensAtBeginLogitsProcessor), [`SuppressTokensLogitsProcessor`](logits_process.md#SuppressTokensLogitsProcessor), [`EmptyProcessor`](logits_process.md#EmptyProcessor)

### `LogitsProcessorList`  ·  implements/extends list
- def: [`easydel/inference/logits_process.py:192`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L192)
- doc: A container class, inheriting from `list`, designed to hold a sequence of
- signature: `class LogitsProcessorList(list):`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int, **kwargs)` — [`L207`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L207) — Applies all contained processors and warpers sequentially to the logits.
- protocol/private: `__hash__`[`L238`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L238)
- uses (calls/refs, reference-scoped): [`hash_fn`](../utils/compiling_utils.md#hash_fn), [`add_start_docstrings`](logits_process.md#add_start_docstrings), [`LOGITS_PROCESSOR_INPUTS_DOCSTRING`](logits_process.md#LOGITS_PROCESSOR_INPUTS_DOCSTRING)
- used by: [`generation_config`](../infra/mixins/generation.md#EasyGenerationMixin.generation_config), [`__all__`](../modules/whisper/modeling_whisper.md#__all__), [`_beam_search`](../infra/mixins/generation.md#EasyGenerationMixin._beam_search), [`_get_logits_processor`](../infra/mixins/generation.md#EasyGenerationMixin._get_logits_processor), [`generate`](../infra/mixins/generation.md#EasyGenerationMixin.generate), [`_greedy_search`](../infra/mixins/generation.md#EasyGenerationMixin._greedy_search), [`_sample`](../infra/mixins/generation.md#EasyGenerationMixin._sample), [`get_logits_warper`](sampling_params.md#JitableSamplingParams.get_logits_warper), [`_get_logits_warper`](../infra/mixins/generation.md#EasyGenerationMixin._get_logits_warper), [`get_logits_processor`](sampling_params.md#JitableSamplingParams.get_logits_processor), [`_force_generate`](../modules/whisper/modeling_whisper.md#WhisperForConditionalGeneration._force_generate), [`_merge_criteria_processor_list`](../infra/mixins/generation.md#EasyGenerationMixin._merge_criteria_processor_list)

### `LogitsWarper`
- def: [`easydel/inference/logits_process.py:133`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L133)
- doc: Abstract base class for all logit warpers.
- signature: `class LogitsWarper:`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L147`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L147) — Applies the warper to the logits.
- protocol/private: `__hash__`[`L165`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L165)
- uses (calls/refs, reference-scoped): [`hash_fn`](../utils/compiling_utils.md#hash_fn), [`TemperatureLogitsWarper`](logits_process.md#TemperatureLogitsWarper), [`TopKLogitsWarper`](logits_process.md#TopKLogitsWarper), [`TopPLogitsWarper`](logits_process.md#TopPLogitsWarper), [`add_start_docstrings`](logits_process.md#add_start_docstrings), [`MinPLogitsWarper`](logits_process.md#MinPLogitsWarper), [`LOGITS_PROCESSOR_INPUTS_DOCSTRING`](logits_process.md#LOGITS_PROCESSOR_INPUTS_DOCSTRING)
- used by: [`TemperatureLogitsWarper`](logits_process.md#TemperatureLogitsWarper), [`TopKLogitsWarper`](logits_process.md#TopKLogitsWarper), [`TopPLogitsWarper`](logits_process.md#TopPLogitsWarper), [`MinPLogitsWarper`](logits_process.md#MinPLogitsWarper)

### `MinLengthLogitsProcessor`  ·  implements/extends LogitsProcessor
- def: [`easydel/inference/logits_process.py:467`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L467)
- doc: \[`LogitsProcessor`\] that prevents the generation of the end-of-sequence (EOS)
- signature: `class MinLengthLogitsProcessor(LogitsProcessor):`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L491`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L491) — Suppress the EOS token until min_length tokens have been generated.
  - `eos_token_id` — [`L482`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L482)
  - `min_length` — [`L481`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L481)
- protocol/private: `__post_init__`[`L484`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L484)
- uses (calls/refs, reference-scoped): [`LogitsProcessor`](logits_process.md#LogitsProcessor)
- used by: [`_get_logits_processor`](../infra/mixins/generation.md#EasyGenerationMixin._get_logits_processor), [`LogitsProcessor`](logits_process.md#LogitsProcessor)

### `MinPLogitsWarper`  ·  implements/extends LogitsWarper
- def: [`easydel/inference/logits_process.py:938`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L938)
- doc: \[`LogitsWarper`\] implementing min-p sampling.
- signature: `class MinPLogitsWarper(LogitsWarper):`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L961`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L961) — Apply min-p filtering, removing tokens below min_p times the peak probability.
  - `filter_value` — [`L958`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L958)
  - `min_p` — [`L957`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L957)
  - `min_tokens_to_keep` — [`L959`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L959)
- protocol/private: `_apply`[`L975`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L975)
- uses (calls/refs, reference-scoped): [`LogitsWarper`](logits_process.md#LogitsWarper)
- used by: [`get_logits_warper`](sampling_params.md#JitableSamplingParams.get_logits_warper), [`LogitsWarper`](logits_process.md#LogitsWarper)

### `NoRepeatNGramLogitsProcessor`  ·  implements/extends LogitsProcessor
- def: [`easydel/inference/logits_process.py:995`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L995)
- doc: \[`LogitsProcessor`\] that prevents the generation of n-grams that have already
- signature: `class NoRepeatNGramLogitsProcessor(LogitsProcessor):`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L1098`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L1098) — Apply no-repeat n-gram filtering if ngram_size is non-zero.
  - `body_fun(i, val)` — [`L1037`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L1037) — Record the n-gram at position i in the sparse index tensor.
  - `forward(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L1013`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L1013) — Apply no-repeat n-gram filtering to scores.
  - `get_banned_tokens_mask(latest_tokens: jnp.ndarray, previous_ngrams)` — [`L1060`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L1060) — Determines which tokens must be banned given latest tokens and the previously seen
  - `get_previous_ngrams(input_ids: jnp.ndarray, vocab_size: int, cur_len: int)` — [`L1031`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L1031) — Build a sparse tensor indexing all previously seen n-grams.
  - `inner_fn(latest_tokens, previous_ngrams)` — [`L1071`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L1071) — Look up whether the latest n-1 tokens form a banned n-gram prefix.
  - `true_fn()` — [`L1028`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L1028) — Compute and apply n-gram ban when sequence is long enough.
  - `ngram_size` — [`L1011`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L1011)
- uses (calls/refs, reference-scoped): [`LogitsProcessor`](logits_process.md#LogitsProcessor)
- used by: [`_get_logits_processor`](../infra/mixins/generation.md#EasyGenerationMixin._get_logits_processor), [`LogitsProcessor`](logits_process.md#LogitsProcessor)

### `PresencePenaltyLogitsProcessor`  ·  implements/extends LogitsProcessor
- def: [`easydel/inference/logits_process.py:780`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L780)
- doc: \[`LogitsProcessor`\] that penalizes tokens based on their presence in the sequence
- signature: `class PresencePenaltyLogitsProcessor(LogitsProcessor):`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L796`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L796) — Subtract a fixed penalty from logits of tokens present in input_ids.
  - `presence_penalty` — [`L794`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L794)
- protocol/private: `_apply`[`L808`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L808)
- uses (calls/refs, reference-scoped): [`LogitsProcessor`](logits_process.md#LogitsProcessor)
- used by: [`_get_logits_processor`](../infra/mixins/generation.md#EasyGenerationMixin._get_logits_processor), [`LogitsProcessor`](logits_process.md#LogitsProcessor), [`get_logits_processor`](sampling_params.md#JitableSamplingParams.get_logits_processor)

### `RepetitionPenaltyLogitsProcessor`  ·  implements/extends LogitsProcessor
- def: [`easydel/inference/logits_process.py:876`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L876)
- doc: \[`LogitsProcessor`\] that applies a multiplicative penalty to the logits of tokens
- signature: `class RepetitionPenaltyLogitsProcessor(LogitsProcessor):`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L900`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L900) — Apply multiplicative repetition penalty to previously seen tokens.
  - `repetition_penalty` — [`L898`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L898)
- protocol/private: `_apply`[`L915`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L915)
- uses (calls/refs, reference-scoped): [`LogitsProcessor`](logits_process.md#LogitsProcessor)
- used by: [`_get_logits_processor`](../infra/mixins/generation.md#EasyGenerationMixin._get_logits_processor), [`LogitsProcessor`](logits_process.md#LogitsProcessor), [`get_logits_processor`](sampling_params.md#JitableSamplingParams.get_logits_processor)

### `SuppressTokensAtBeginLogitsProcessor`  ·  implements/extends LogitsProcessor
- def: [`easydel/inference/logits_process.py:508`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L508)
- doc: \[`LogitsProcessor`\] that suppresses a specified list of tokens only at a specific
- signature: `class SuppressTokensAtBeginLogitsProcessor(LogitsProcessor):`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L528`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L528) — Suppress specified tokens at the begin_index generation step.
  - `begin_index` — [`L526`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L526)
  - `begin_suppress_tokens` — [`L525`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L525)
- uses (calls/refs, reference-scoped): [`LogitsProcessor`](logits_process.md#LogitsProcessor)
- used by: [`_get_logits_processor`](../infra/mixins/generation.md#EasyGenerationMixin._get_logits_processor), [`LogitsProcessor`](logits_process.md#LogitsProcessor)

### `SuppressTokensLogitsProcessor`  ·  implements/extends LogitsProcessor
- def: [`easydel/inference/logits_process.py:545`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L545)
- doc: \[`LogitsProcessor`\] that suppresses a specified list of tokens throughout the
- signature: `class SuppressTokensLogitsProcessor(LogitsProcessor):`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L559`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L559) — Suppress specified tokens at every generation step.
  - `suppress_tokens` — [`L557`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L557)
- uses (calls/refs, reference-scoped): [`LogitsProcessor`](logits_process.md#LogitsProcessor)
- used by: [`_get_logits_processor`](../infra/mixins/generation.md#EasyGenerationMixin._get_logits_processor), [`LogitsProcessor`](logits_process.md#LogitsProcessor)

### `TemperatureLogitsWarper`  ·  implements/extends LogitsWarper
- def: [`easydel/inference/logits_process.py:242`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L242)
- doc: \[`LogitsWarper`\] that applies temperature scaling to the logits distribution.
- signature: `class TemperatureLogitsWarper(LogitsWarper):`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L257`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L257) — Scale logits by dividing by the temperature value.
  - `temperature` — [`L255`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L255)
- uses (calls/refs, reference-scoped): [`LogitsWarper`](logits_process.md#LogitsWarper)
- used by: [`get_logits_warper`](sampling_params.md#JitableSamplingParams.get_logits_warper), [`_get_logits_warper`](../infra/mixins/generation.md#EasyGenerationMixin._get_logits_warper), [`LogitsWarper`](logits_process.md#LogitsWarper)

### `TopKLogitsWarper`  ·  implements/extends LogitsWarper
- def: [`easydel/inference/logits_process.py:335`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L335)
- doc: \[`LogitsWarper`\] that implements top-k sampling.
- signature: `class TopKLogitsWarper(LogitsWarper):`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L357`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L357) — Apply top-k filtering to scores.
  - `_filter_scores(s: jnp.ndarray)` — [`L376`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L376) — Applies the dynamic filtering logic.
  - `_identity(s: jnp.ndarray)` — [`L386`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L386) — Returns scores unchanged.
  - `filter_value` — [`L354`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L354)
  - `min_tokens_to_keep` — [`L355`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L355)
  - `top_k` — [`L353`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L353)
- uses (calls/refs, reference-scoped): [`LogitsWarper`](logits_process.md#LogitsWarper)
- used by: [`get_logits_warper`](sampling_params.md#JitableSamplingParams.get_logits_warper), [`_get_logits_warper`](../infra/mixins/generation.md#EasyGenerationMixin._get_logits_warper), [`LogitsWarper`](logits_process.md#LogitsWarper)

### `TopPLogitsWarper`  ·  implements/extends LogitsWarper
- def: [`easydel/inference/logits_process.py:278`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L278)
- doc: \[`LogitsWarper`\] that implements top-p (nucleus) sampling.
- signature: `class TopPLogitsWarper(LogitsWarper):`
- members:
  - `__call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int)` — [`L302`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L302) — Apply top-p (nucleus) filtering to scores.
  - `filter_value` — [`L299`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L299)
  - `min_tokens_to_keep` — [`L300`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L300)
  - `top_p` — [`L298`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L298)
- protocol/private: `_apply`[`L319`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L319)
- uses (calls/refs, reference-scoped): [`LogitsWarper`](logits_process.md#LogitsWarper)
- used by: [`get_logits_warper`](sampling_params.md#JitableSamplingParams.get_logits_warper), [`_get_logits_warper`](../infra/mixins/generation.md#EasyGenerationMixin._get_logits_warper), [`LogitsWarper`](logits_process.md#LogitsWarper)

### `WhisperTimeStampLogitsProcessor`  ·  implements/extends LogitsProcessor
- def: [`easydel/inference/logits_process.py:644`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L644)
- doc: A specialized \[`LogitsProcessor`\] tailored for handling timestamp tokens during
- signature: `class WhisperTimeStampLogitsProcessor(LogitsProcessor):`
- members:
  - `__call__(self, input_ids, scores, cur_len)` — [`L701`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L701) — Apply Whisper timestamp constraints to scores.
  - `__init__(self, generate_config, model_config, decoder_input_length)` — [`L677`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L677) — Initialize Whisper timestamp processor from generation and model configs.
  - `handle_cumulative_probs(logprobs_k, scores_k)` — [`L764`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L764) — Force timestamp sampling when timestamp probability exceeds text token probability.
  - `handle_pairs(input_ids_k, scores_k)` — [`L718`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L718) — Apply timestamp pairing constraints for a single sequence.
  - `begin_index` — [`L690`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L690)
  - `eos_token_id` — [`L686`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L686)
  - `max_initial_timestamp_index` — [`L695`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L695)
  - `no_timestamps_token_id` — [`L687`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L687)
  - `timestamp_begin` — [`L688`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L688)
- uses (calls/refs, reference-scoped): [`LogitsProcessor`](logits_process.md#LogitsProcessor)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`__all__`](../modules/whisper/modeling_whisper.md#__all__), [`LogitsProcessor`](logits_process.md#LogitsProcessor), [`_force_generate`](../modules/whisper/modeling_whisper.md#WhisperForConditionalGeneration._force_generate), [`generate`](../modules/whisper/modeling_whisper.md#WhisperForConditionalGeneration.generate)

## Functions
- `add_start_docstrings(*docstr)` — [`L51`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L51) — A decorator that prepends a given docstring section to the decorated function's docstring.
- `docstring_decorator(fn)` — [`L66`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L66) — Prepend the given docstring fragments to the function's existing docstring.

## Module values
- `LOGITS_PROCESSOR_INPUTS_DOCSTRING` — [`L77`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L77)
- `logger` — [`L74`](../../../../../../raw/code/EasyDeL/easydel/inference/logits_process.py#L74)

