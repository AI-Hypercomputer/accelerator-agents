---
title: 'Module: recurrentgemma/jax/sampler.py'
type: catalog
provenance: extracted
module: recurrentgemma/jax/sampler.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.jax.sampler`/
symbols:
  Sampler._prompt_processing_fn: Sampler#_prompt_processing_fn().
  Sampler._sample_step: Sampler#_sample_step().
  Sampler.__call__: Sampler#__call__().
  Sampler.apply_model: Sampler#apply_model().
  SamplingState: SamplingState#
  Sampler._sample_fn: Sampler#_sample_fn().
  Sampler.prompt_processing_fn: Sampler#prompt_processing_fn().
  Sampler.sample_fn: Sampler#sample_fn().
  Cache: Cache.
  Sampler.tokenize: Sampler#tokenize().
  Sampler.cond_fn: Sampler#cond_fn().
  Sampler._get_padded_tokens: Sampler#_get_padded_tokens().
  Sampler._sample_from_logits: Sampler#_sample_from_logits().
  Sampler.vocab: Sampler#vocab.
  Sampler.sample_with_params: Sampler#sample_with_params().
  SamplingState.cache: SamplingState#cache.
  SamplingState.tokens_buffer: SamplingState#tokens_buffer.
  SamplingState.logits_buffer: SamplingState#logits_buffer.
  SamplerOutput.logits: SamplerOutput#logits.
  SamplingState.step: SamplingState#step.
  SamplingState.total_steps: SamplingState#total_steps.
  SamplingState.positions: SamplingState#positions.
  SamplingState.done: SamplingState#done.
  Sampler: Sampler#
  Sampler._compiled_prompt_processing_fn: Sampler#_compiled_prompt_processing_fn.
  Sampler._compiled_sample_fn: Sampler#_compiled_sample_fn.
  Sampler.dtype: Sampler#dtype().
  Sampler.vocab_size: Sampler#vocab_size().
  SamplingState.rng: SamplingState#rng.
  SamplerOutput.tokens: SamplerOutput#tokens.
  Sampler.params: Sampler#params.
  Sampler.__init__: Sampler#__init__().
  SamplerOutput: SamplerOutput#
  SamplerOutput.text: SamplerOutput#text.
  Sampler.model: Sampler#model.
  Sampler.deterministic_sampling: Sampler#deterministic_sampling.
  Sampler.jit_compile: Sampler#jit_compile.
  Sampler._is_it_model: Sampler#_is_it_model.
---
# Module: [`recurrentgemma/jax/sampler.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py)

## Classes
### `Sampler`  ·  implements/extends Generic
- def: [`recurrentgemma/jax/sampler.py:75`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L75)
- doc: Sampler for a Griffin model.
- signature: `class Sampler(Generic[Cache]):`
- members:
  - `__call__(self, input_strings: Sequence[str], total_generation_steps: int, rng: jt.PRNGKeyArray | None = None, echo: bool = False, return_logits: bool = False, end_sampling_at_eos_token: bool = True)` — [`L403`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L403) — Samples a completion of the input string. — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `__init__(self, model: nn.Module, vocab: spm.SentencePieceProcessor, params: at.Params, jit_compile: bool = True, deterministic_sampling: bool = True, is_it_model: bool = False)` — [`L78`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L78) — Initializes a sampler for a Griffin model.
  - `_get_padded_tokens(self, tokens: Sequence[jax.Array])` — [`L385`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L385) — Returns an array of padded tokens. — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `_prompt_processing_fn(self, params: at.Params, tokens: at.Tokens, rng: jt.PRNGKeyArray | None, input_lengths: at.NumTokens, total_generation_steps: int, return_logits: bool, echo: bool)` — [`L263`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L263) — Pre-processes the prompt. — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `_sample_fn(self, params: at.Params, initial_sampling_state: SamplingState, end_sampling_at_eos_token: bool = True)` — [`L241`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L241) — Internal sampling function (to be jitted). — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `_sample_from_logits(self, rng: jt.PRNGKeyArray | None, logits: jt.Float[jt.Array, "*b v"])` — [`L158`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L158) — Samples from the logits categorical distribution. — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `_sample_step(self, params: at.Params, sampler_state: SamplingState[Cache], end_sampling_at_eos_token: bool = True)` — [`L172`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L172) — Performs a single sampling step. — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `apply_model(self, params: at.Params, tokens: at.Tokens, segment_pos: at.SegmentPos, cache: Cache | None, return_logits: bool, return_cache: bool)` — [`L139`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L139) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `cond_fn(sampler_state: SamplingState)` — [`L252`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L252)
  - `dtype(self)` — [`L117`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L117) — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `prompt_processing_fn(self)` — [`L125`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L125) — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `sample_fn(self)` — [`L132`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L132) — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `sample_with_params(sampler_state: SamplingState)` — [`L249`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L249)
  - `tokenize(self, input_string: str)` — [`L232`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L232) — Tokenizes the input string. — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `vocab_size(self)` — [`L121`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L121)
  - `deterministic_sampling` — [`L102`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L102)
  - `jit_compile` — [`L103`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L103)
  - `model` — [`L99`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L99)
  - `params` — [`L101`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L101)
  - `vocab` — [`L100`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L100)
- protocol/private: `_compiled_prompt_processing_fn`[`L104`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L104), `_compiled_sample_fn`[`L109`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L109), `_is_it_model`[`L114`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L114)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`SegmentPos`](array_typing.md#SegmentPos), [`SamplingState`](sampler.md#SamplingState), [`Cache`](sampler.md#Cache), [`Tokens`](array_typing.md#Tokens), [`Params`](array_typing.md#Params), [`cache`](sampler.md#SamplingState.cache), [`TokenLogits`](array_typing.md#TokenLogits), [`logits`](sampler.md#SamplerOutput.logits), [`logits_buffer`](sampler.md#SamplingState.logits_buffer), [`tokens_buffer`](sampler.md#SamplingState.tokens_buffer), [`done`](sampler.md#SamplingState.done), [`positions`](sampler.md#SamplingState.positions), [`step`](sampler.md#SamplingState.step), [`total_steps`](sampler.md#SamplingState.total_steps), [`NumTokens`](array_typing.md#NumTokens), [`rng`](sampler.md#SamplingState.rng), [`tokens`](sampler.md#SamplerOutput.tokens), [`SamplerOutput`](sampler.md#SamplerOutput), [`apply_it_formatter`](../common.md#apply_it_formatter), [`text`](sampler.md#SamplerOutput.text)
- used by: [`Sampler`](__init__.md#Sampler)

### `SamplerOutput`
- def: [`recurrentgemma/jax/sampler.py:61`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L61) — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
- doc: Output of the sampler.
- signature: `class SamplerOutput:`
- members:
  - `logits` — [`L72`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L72) — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `text` — [`L70`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L70) — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `tokens` — [`L71`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L71) — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
- used by: [`test_forward_equivalence`](sampler_test.md#SamplerTest.test_forward_equivalence), [`test_output_shapes`](sampler_test.md#SamplerTest.test_output_shapes), [`__call__`](sampler.md#Sampler.__call__)  (1 test-only)

### `SamplingState`  ·  implements/extends Generic
- def: [`recurrentgemma/jax/sampler.py:36`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L36) — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
- doc: Internal sampling state.
- signature: `class SamplingState(Generic[Cache]):`
- members:
  - `cache` — [`L55`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L55) — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `done` — [`L56`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L56) — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `logits_buffer` — [`L57`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L57) — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `positions` — [`L54`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L54) — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `rng` — [`L51`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L51) — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `step` — [`L52`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L52) — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `tokens_buffer` — [`L50`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L50) — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
  - `total_steps` — [`L53`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L53) — documented in [recurrentgemma-jax-sampler](../../../concepts/recurrentgemma-jax-sampler.md)
- uses (calls/refs, reference-scoped): [`Cache`](sampler.md#Cache)
- used by: [`_prompt_processing_fn`](sampler.md#Sampler._prompt_processing_fn), [`_sample_step`](sampler.md#Sampler._sample_step), [`__call__`](sampler.md#Sampler.__call__), [`_sample_fn`](sampler.md#Sampler._sample_fn), [`prompt_processing_fn`](sampler.md#Sampler.prompt_processing_fn), [`sample_fn`](sampler.md#Sampler.sample_fn), [`cond_fn`](sampler.md#Sampler.cond_fn), [`sample_with_params`](sampler.md#Sampler.sample_with_params)

## Module values
- `Cache` — [`L31`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/sampler.py#L31)

