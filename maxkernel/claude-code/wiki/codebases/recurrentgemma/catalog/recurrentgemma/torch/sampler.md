---
title: 'Module: recurrentgemma/torch/sampler.py'
type: catalog
provenance: extracted
module: recurrentgemma/torch/sampler.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.torch.sampler`/
symbols:
  Sampler._prompt_processing_fn: Sampler#_prompt_processing_fn().
  Sampler._sample_step: Sampler#_sample_step().
  Sampler.__call__: Sampler#__call__().
  Sampler.apply_model: Sampler#apply_model().
  Sampler._sample_fn: Sampler#_sample_fn().
  Sampler.tokenize: Sampler#tokenize().
  Sampler._get_padded_tokens: Sampler#_get_padded_tokens().
  SamplingState: SamplingState#
  Sampler.device: Sampler#device().
  Sampler.vocab_size: Sampler#vocab_size().
  Sampler._sample_from_logits: Sampler#_sample_from_logits().
  Cache: Cache.
  Sampler.vocab: Sampler#vocab.
  Sampler._eos_token: Sampler#_eos_token.
  SamplingState.cache: SamplingState#cache.
  SamplingState.tokens_buffer: SamplingState#tokens_buffer.
  SamplingState.step: SamplingState#step.
  SamplingState.total_steps: SamplingState#total_steps.
  SamplingState.positions: SamplingState#positions.
  SamplingState.done: SamplingState#done.
  SamplingState.logits_buffer: SamplingState#logits_buffer.
  Sampler.model: Sampler#model.
  Sampler.dtype: Sampler#dtype().
  Sampler.__init__: Sampler#__init__().
  SamplerOutput: SamplerOutput#
  SamplerOutput.text: SamplerOutput#text.
  SamplerOutput.logits: SamplerOutput#logits.
  SamplerOutput.tokens: SamplerOutput#tokens.
  Sampler: Sampler#
  Sampler.greedy_sampling: Sampler#greedy_sampling.
  Sampler._is_it_model: Sampler#_is_it_model.
---
# Module: [`recurrentgemma/torch/sampler.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py)

## Classes
### `Sampler`
- def: [`recurrentgemma/torch/sampler.py:70`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L70)
- doc: Sampler for a Griffin model.
- signature: `class Sampler:`
- members:
  - `__call__(self, input_strings: Sequence[str], total_generation_steps: int, echo: bool = False, return_logits: bool = False, end_sampling_at_eos_token: bool = True)` — [`L364`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L364) — Samples a completion of the input string. — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `__init__(self, model: griffin_lib.Griffin, vocab: spm.SentencePieceProcessor, greedy_sampling: bool = True, is_it_model: bool = False)` — [`L73`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L73) — Initializes a sampler for a Griffin model.
  - `_get_padded_tokens(self, tokens: Sequence[torch.Tensor])` — [`L339`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L339) — Returns an array of padded tokens. — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `_prompt_processing_fn(self, tokens: at.Tokens, input_lengths: at.NumTokens, total_generation_steps: int, return_logits: bool, echo: bool)` — [`L222`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L222) — Pre-processes the prompt. — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `_sample_fn(self, sampler_state: SamplingState, end_sampling_at_eos_token: bool = True)` — [`L204`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L204) — Internal sampling function (to be jitted). — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `_sample_from_logits(self, logits: jt.Float[torch.Tensor, "*b v"])` — [`L125`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L125) — Samples from the logits categorical distribution. — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `_sample_step(self, sampler_state: SamplingState, end_sampling_at_eos_token: bool = True)` — [`L136`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L136) — Performs a single sampling step. — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `apply_model(self, tokens: at.Tokens, segment_pos: at.SegmentPos, cache: Cache | None = None, return_logits: bool = True, return_cache: bool = True)` — [`L108`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L108) — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `device(self)` — [`L100`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L100) — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `dtype(self)` — [`L96`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L96) — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `tokenize(self, input_string: str)` — [`L190`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L190) — Tokenizes the input string. — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `vocab_size(self)` — [`L104`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L104)
  - `greedy_sampling` — [`L91`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L91) — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `model` — [`L89`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L89)
  - `vocab` — [`L90`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L90)
- protocol/private: `_eos_token`[`L92`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L92), `_is_it_model`[`L93`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L93)
- uses (calls/refs, reference-scoped): [`Griffin`](griffin.md#Griffin), [`typed`](array_typing.md#typed), [`SegmentPos`](array_typing.md#SegmentPos), [`vocab_size`](../common.md#GriffinConfig.vocab_size), [`config`](griffin.md#Griffin.config), [`SamplingState`](sampler.md#SamplingState), [`Tokens`](array_typing.md#Tokens), [`Cache`](sampler.md#Cache), [`cache`](sampler.md#SamplingState.cache), [`TokenLogits`](array_typing.md#TokenLogits), [`tokens_buffer`](sampler.md#SamplingState.tokens_buffer), [`done`](sampler.md#SamplingState.done), [`logits_buffer`](sampler.md#SamplingState.logits_buffer), [`positions`](sampler.md#SamplingState.positions), [`step`](sampler.md#SamplingState.step), [`total_steps`](sampler.md#SamplingState.total_steps), [`NumTokens`](array_typing.md#NumTokens), [`SamplerOutput`](sampler.md#SamplerOutput), [`apply_it_formatter`](../common.md#apply_it_formatter), [`logits`](sampler.md#SamplerOutput.logits), [`text`](sampler.md#SamplerOutput.text), [`tokens`](sampler.md#SamplerOutput.tokens)
- used by: [`Sampler`](__init__.md#Sampler)

### `SamplerOutput`  ·  implements/extends NamedTuple
- def: [`recurrentgemma/torch/sampler.py:56`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L56) — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
- doc: Output of the sampler.
- signature: `class SamplerOutput(NamedTuple):`
- members:
  - `logits` — [`L66`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L66) — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `text` — [`L65`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L65) — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `tokens` — [`L67`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L67) — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
- used by: [`__call__`](sampler.md#Sampler.__call__)

### `SamplingState`  ·  implements/extends Generic
- def: [`recurrentgemma/torch/sampler.py:35`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L35) — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
- doc: Internal sampling state.
- signature: `class SamplingState(Generic[Cache]):`
- members:
  - `cache` — [`L51`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L51) — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `done` — [`L52`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L52) — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `logits_buffer` — [`L53`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L53) — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `positions` — [`L50`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L50) — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `step` — [`L48`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L48) — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `tokens_buffer` — [`L47`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L47) — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
  - `total_steps` — [`L49`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L49) — documented in [recurrentgemma-torch-sampler](../../../concepts/recurrentgemma-torch-sampler.md)
- uses (calls/refs, reference-scoped): [`Cache`](sampler.md#Cache)
- used by: [`_prompt_processing_fn`](sampler.md#Sampler._prompt_processing_fn), [`_sample_step`](sampler.md#Sampler._sample_step), [`__call__`](sampler.md#Sampler.__call__), [`_sample_fn`](sampler.md#Sampler._sample_fn)

## Module values
- `Cache` — [`L30`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/sampler.py#L30)

