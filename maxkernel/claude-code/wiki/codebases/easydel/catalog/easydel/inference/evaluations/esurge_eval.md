---
title: 'Module: easydel/inference/evaluations/esurge_eval.py'
type: catalog
provenance: extracted
module: easydel/inference/evaluations/esurge_eval.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.evaluations.esurge_eval`/
symbols:
  eSurgeLMEvalAdapter._generate: eSurgeLMEvalAdapter#_generate().
  eSurgeLMEvalAdapter.tokenizer: eSurgeLMEvalAdapter#tokenizer.
  eSurgeLMEvalAdapter.apply_chat_template: eSurgeLMEvalAdapter#apply_chat_template().
  eSurgeLMEvalAdapter.loglikelihood: eSurgeLMEvalAdapter#loglikelihood().
  eSurgeLMEvalAdapter.loglikelihood_rolling: eSurgeLMEvalAdapter#loglikelihood_rolling().
  eSurgeLMEvalAdapter._loglikelihood_token_ids: eSurgeLMEvalAdapter#_loglikelihood_token_ids().
  eSurgeLMEvalAdapter.max_length: eSurgeLMEvalAdapter#max_length().
  eSurgeLMEvalAdapter._setup: eSurgeLMEvalAdapter#_setup().
  eSurgeLMEvalAdapter._maybe_normalize_math: eSurgeLMEvalAdapter#_maybe_normalize_math().
  _extract_math_answer: _extract_math_answer().
  eSurgeLMEvalAdapter.max_gen_toks: eSurgeLMEvalAdapter#max_gen_toks().
  eSurgeLMEvalAdapter.__init__: eSurgeLMEvalAdapter#__init__().
  eSurgeLMEvalAdapter.greedy_until: eSurgeLMEvalAdapter#greedy_until().
  eSurgeLMEvalAdapter.tok_decode: eSurgeLMEvalAdapter#tok_decode().
  eSurgeLMEvalAdapter.model: eSurgeLMEvalAdapter#model.
  eSurgeLMEvalAdapter._batch_size: eSurgeLMEvalAdapter#_batch_size.
  eSurgeLMEvalAdapter._get_scoring_logits_fn: eSurgeLMEvalAdapter#_get_scoring_logits_fn().
  eSurgeLMEvalAdapter.tok_encode: eSurgeLMEvalAdapter#tok_encode().
  eSurgeLMEvalAdapter.prefix_token_id: eSurgeLMEvalAdapter#prefix_token_id().
  _freeze_generation_value: _freeze_generation_value().
  eSurgeLMEvalAdapter.generate_until: eSurgeLMEvalAdapter#generate_until().
  eSurgeLMEvalAdapter._get_scoring_model: eSurgeLMEvalAdapter#_get_scoring_model().
  eSurgeLMEvalAdapter: eSurgeLMEvalAdapter#
  err: err.
  _postprocess_generation_text: _postprocess_generation_text().
  _coerce_sampling_params_template: _coerce_sampling_params_template().
  _normalize_math_generation: _normalize_math_generation().
  eSurgeLMEvalAdapter.surge: eSurgeLMEvalAdapter#surge.
  eSurgeLMEvalAdapter.stop: eSurgeLMEvalAdapter#stop().
  eSurgeLMEvalAdapter._build_scoring_model: eSurgeLMEvalAdapter#_build_scoring_model().
  _normalize_number_candidate: _normalize_number_candidate().
  eSurgeLMEvalAdapter.setup_complete: eSurgeLMEvalAdapter#setup_complete.
  eSurgeLMEvalAdapter.eot_token_id: eSurgeLMEvalAdapter#eot_token_id().
  logger: logger.
  _normalize_stop_list: _normalize_stop_list().
  eSurgeLMEvalAdapter.tokenizer_name: eSurgeLMEvalAdapter#tokenizer_name().
  eSurgeLMEvalAdapter._encode_text: eSurgeLMEvalAdapter#_encode_text().
  eSurgeLMEvalAdapter.enable_thinking: eSurgeLMEvalAdapter#enable_thinking.
  eSurgeLMEvalAdapter._sampling_params_template_fields: eSurgeLMEvalAdapter#_sampling_params_template_fields.
  eSurgeLMEvalAdapter.temperature: eSurgeLMEvalAdapter#temperature.
  eSurgeLMEvalAdapter.top_p: eSurgeLMEvalAdapter#top_p.
  _merge_stop_lists: _merge_stop_lists().
  eSurgeLMEvalAdapter.normalize_math_answers: eSurgeLMEvalAdapter#normalize_math_answers.
  eSurgeLMEvalAdapter._encode_pair: eSurgeLMEvalAdapter#_encode_pair().
  eSurgeLMEvalAdapter.ignore_benchmark_eos_flags: eSurgeLMEvalAdapter#ignore_benchmark_eos_flags.
  eSurgeLMEvalAdapter._scoring_model: eSurgeLMEvalAdapter#_scoring_model.
  eSurgeLMEvalAdapter._scoring_logits_fn: eSurgeLMEvalAdapter#_scoring_logits_fn.
  eSurgeLMEvalAdapter._decode_supports_skip_special: eSurgeLMEvalAdapter#_decode_supports_skip_special.
  eSurgeLMEvalAdapter._chat_template_signature: eSurgeLMEvalAdapter#_chat_template_signature.
  eSurgeLMEvalAdapter.batch_size: eSurgeLMEvalAdapter#batch_size().
  _STRICT_MATH_ANSWER_RE: _STRICT_MATH_ANSWER_RE.
  _NUMBER_RE: _NUMBER_RE.
  eSurgeLMEvalAdapter.hard_max_new_tokens: eSurgeLMEvalAdapter#hard_max_new_tokens.
  eSurgeLMEvalAdapter.chat_template_args: eSurgeLMEvalAdapter#chat_template_args.
  eSurgeLMEvalAdapter.think_end_token: eSurgeLMEvalAdapter#think_end_token.
  eSurgeLMEvalAdapter.sampling_params: eSurgeLMEvalAdapter#sampling_params.
  eSurgeLMEvalAdapter._decode_supports_spaces_between: eSurgeLMEvalAdapter#_decode_supports_spaces_between.
  eSurgeLMEvalAdapter._chat_template_sig_resolved: eSurgeLMEvalAdapter#_chat_template_sig_resolved.
  eSurgeLMEvalAdapter._parse_instances: eSurgeLMEvalAdapter#_parse_instances().
  _DEFAULT_REQUEST_ID_PREFIX: _DEFAULT_REQUEST_ID_PREFIX.
  _DEFAULT_MATH_ANSWER_TASK_HINTS: _DEFAULT_MATH_ANSWER_TASK_HINTS.
  _BOXED_ANSWER_RE: _BOXED_ANSWER_RE.
  _chunked: _chunked().
  _trim_stop_sequences: _trim_stop_sequences().
  _strip_empty_reasoning_scaffold: _strip_empty_reasoning_scaffold().
  _coerce_optional_bool: _coerce_optional_bool().
  _is_math_answer_task: _is_math_answer_task().
  _get_rolling_token_windows: _get_rolling_token_windows().
  _make_disjoint_window: _make_disjoint_window().
  eSurgeLMEvalAdapter.max_new_tokens: eSurgeLMEvalAdapter#max_new_tokens.
  eSurgeLMEvalAdapter.think_start_token: eSurgeLMEvalAdapter#think_start_token.
  eSurgeLMEvalAdapter.truncation_side: eSurgeLMEvalAdapter#truncation_side.
  eSurgeLMEvalAdapter.math_answer_task_hints: eSurgeLMEvalAdapter#math_answer_task_hints.
  eSurgeLMEvalAdapter._forward: eSurgeLMEvalAdapter#_forward().
  eSurgeLMEvalAdapter._max_length: eSurgeLMEvalAdapter#_max_length.
  eSurgeLMEvalAdapter._extract_choice_from_generation: eSurgeLMEvalAdapter#_extract_choice_from_generation().
  eSurgeLMEvalAdapter.device: eSurgeLMEvalAdapter#device().
  eSurgeLMEvalAdapter._model_call: eSurgeLMEvalAdapter#_model_call().
  eSurgeLMEvalAdapter._model_generate: eSurgeLMEvalAdapter#_model_generate().
---
# Module: [`easydel/inference/evaluations/esurge_eval.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py)

## Classes
### `eSurgeLMEvalAdapter`
- def: [`easydel/inference/evaluations/esurge_eval.py:449`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L449)
- doc: Adapter for EasyDeL models to be compatible with lm-evaluation-harness.
- signature: `class eSurgeLMEvalAdapter(LM):`
- members:
  - `__init__(self, surge: eSurge, processor: ProcessingClassType, max_length: int = 8192, max_new_tokens: int = 2048, hard_max_new_tokens: bool = False, enable_thinking: bool = False, chat_template_args: dict[str, Any] | None = None, think_start_token: str | None = None, think_end_token: str | None = None, ignore_benchmark_eos_flags: bool = False, top_p: float = 0.95, temperature: float = 0, batch_size: int | None = None, sampling_params: SamplingParams | collections.abc.Mapping[str, Any] | None = None, normalize_math_answers: bool = True, math_answer_task_hints: tuple[str, ...] | list[str] | None = None)` — [`L481`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L481) — Initialize the eSurgeLMEvalAdapter.
  - `_build_scoring_model(self)` — [`L620`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L620) — Build a scoring model that supports direct teacher-forced forward passes.
  - `_encode_pair(self, context: str, continuation: str)` — [`L1278`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1278) — Encode a context/continuation pair with TemplateLM causal semantics.
  - `_encode_text(self, text: str)` — [`L1235`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1235) — Encode text to token IDs using the tokenizer.
  - `_extract_choice_from_generation(self, generation: str)` — [`L945`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L945) — Extract a multiple-choice answer (A, B, C, D) from generated text.
  - `_forward(input_ids, attention_mask)` — [`L677`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L677) — Compute logits from the scoring model for the given inputs.
  - `_generate(self, prompts: list[str], max_tokens: int | None = None, temperature: float | None = None, top_p: float | None = None, stop_sequences: list[list[str]] | None = None, generation_kwargs: list[dict[str, Any]] | None = None, allow_per_request_sampling_overrides: bool = True)` — [`L692`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L692) — Generate responses for prompts with optional per-request overrides.
  - `_get_scoring_logits_fn(self)` — [`L672`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L672) — Return a cached jitted forward that outputs logits only.
  - `_get_scoring_model(self)` — [`L658`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L658) — Return the cached scoring model, building it on first access.
  - `_loglikelihood_token_ids(self, context_token_ids: list[list[int]], continuation_token_ids: list[list[int]])` — [`L1298`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1298) — Compute log-likelihood of continuations given contexts.
  - `_maybe_normalize_math(self, generations: list[str], instances)` — [`L1027`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1027) — Apply math answer normalization for GSM-like tasks when enabled.
  - `_model_call(self, inps)` — [`L1458`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1458) — Raw model call interface (not implemented).
  - `_model_generate(self, context, max_length, eos_token_id)` — [`L1474`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1474) — Raw model generation interface (not implemented).
  - `_parse_instances(instances)` — [`L983`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L983) — Extract prompts and generation kwargs from lm-eval Instance objects.
  - `_setup(self)` — [`L594`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L594) — Set up the eSurge engine.
  - `apply_chat_template(self, messages, add_generation_prompt: bool)` — [`L1127`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1127) — Apply chat template to messages.
  - `batch_size(self)` — [`L1094`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1094) — Get the batch size.
  - `device(self)` — [`L1103`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1103) — Get the device (CPU/GPU).
  - `eot_token_id(self)` — [`L1058`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1058) — Get the end-of-text token ID.
  - `generate_until(self, instances)` — [`L1039`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1039) — Generate text until a specified set of stop sequences is reached.
  - `greedy_until(self, requests)` — [`L1608`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1608) — Generate completions using greedy decoding until stop sequences.
  - `loglikelihood(self, instances, disable_tqdm: bool = False)` — [`L1492`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1492) — Compute log-likelihood of completions given contexts.
  - `loglikelihood_rolling(self, instances)` — [`L1543`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1543) — Calculate log-likelihood of token sequences in a rolling fashion.
  - `max_gen_toks(self)` — [`L1085`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1085) — Get the maximum number of tokens to generate.
  - `max_length(self)` — [`L1067`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1067) — Get the maximum context length.
  - `prefix_token_id(self)` — [`L1260`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1260) — Return the prefix token used for empty-context scoring.
  - `stop(self)` — [`L611`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L611) — Stop the eSurge engine.
  - `tok_decode(self, tokens, *, skip_special_tokens: bool = False, spaces_between_special_tokens: bool | None = None)` — [`L1202`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1202) — Decode token IDs into a string.
  - `tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs)` — [`L1183`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1183) — Encode a string into token IDs.
  - `tokenizer_name(self)` — [`L1112`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1112) — Get the tokenizer name for chat template support.
  - `chat_template_args` — [`L557`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L557)
  - `enable_thinking` — [`L558`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L558)
  - `hard_max_new_tokens` — [`L556`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L556)
  - `ignore_benchmark_eos_flags` — [`L565`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L565)
  - `math_answer_task_hints` — [`L578`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L578)
  - `max_new_tokens` — [`L555`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L555)
  - `model` — [`L584`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L584)
  - `normalize_math_answers` — [`L569`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L569)
  - `sampling_params` — [`L567`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L567)
  - `setup_complete` — [`L591`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L591)
  - `surge` — [`L582`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L582)
  - `temperature` — [`L554`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L554)
  - `think_end_token` — [`L562`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L562)
  - `think_start_token` — [`L559`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L559)
  - `tokenizer` — [`L548`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L548)
  - `top_p` — [`L566`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L566)
  - `truncation_side` — [`L568`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L568)
- protocol/private: `_batch_size`[`L583`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L583), `_chat_template_sig_resolved`[`L590`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L590), `_chat_template_signature`[`L589`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L589), `_decode_supports_skip_special`[`L587`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L587), `_decode_supports_spaces_between`[`L588`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L588), `_max_length`[`L1082`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L1082), `_sampling_params_template_fields`[`L567`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L567), `_scoring_logits_fn`[`L586`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L586), `_scoring_model`[`L585`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L585)
- uses (calls/refs, reference-scoped): [`SamplingParams`](../sampling_params.md#SamplingParams), [`ProcessingClassType`](../../infra/utils.md#ProcessingClassType), [`eSurge`](../esurge/esurge_engine.md#eSurge), [`max_tokens`](../sampling_params.md#SamplingParams.max_tokens), [`terminate`](../esurge/mixins/lifecycle.md#EngineLifecycleMixin.terminate), [`generate`](../esurge/mixins/io.md#EngineIOMixin.generate), [`get_text`](../esurge/esurge_engine.md#RequestOutput.get_text), [`_freeze_generation_value`](esurge_eval.md#_freeze_generation_value), [`initiate`](../esurge/mixins/lifecycle.md#EngineLifecycleMixin.initiate), [`_coerce_sampling_params_template`](esurge_eval.md#_coerce_sampling_params_template), [`_normalize_math_generation`](esurge_eval.md#_normalize_math_generation), [`_postprocess_generation_text`](esurge_eval.md#_postprocess_generation_text), [`request_id`](../esurge/esurge_engine.md#RequestOutput.request_id), [`_normalize_stop_list`](esurge_eval.md#_normalize_stop_list), [`logger`](esurge_eval.md#logger), [`clone`](../sampling_params.md#SamplingParams.clone), [`max_num_seqs`](../esurge/esurge_engine.md#eSurge.max_num_seqs), [`_merge_stop_lists`](esurge_eval.md#_merge_stop_lists), [`_scheduler_running`](../esurge/esurge_engine.md#eSurge._scheduler_running), `name_or_path`, [`_DEFAULT_MATH_ANSWER_TASK_HINTS`](esurge_eval.md#_DEFAULT_MATH_ANSWER_TASK_HINTS), [`_DEFAULT_REQUEST_ID_PREFIX`](esurge_eval.md#_DEFAULT_REQUEST_ID_PREFIX), [`_chunked`](esurge_eval.md#_chunked), [`_coerce_optional_bool`](esurge_eval.md#_coerce_optional_bool), [`_get_rolling_token_windows`](esurge_eval.md#_get_rolling_token_windows), [`_is_math_answer_task`](esurge_eval.md#_is_math_answer_task), [`_make_disjoint_window`](esurge_eval.md#_make_disjoint_window), [`_strip_empty_reasoning_scaffold`](esurge_eval.md#_strip_empty_reasoning_scaffold)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`run_lm_eval_with_esurge`](../../infra/elarge/benchmarking.md#run_lm_eval_with_esurge)

## Functions
- `_chunked(seq: list[Any], size: int)` — [`L97`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L97) — Split a sequence into chunks of a specified size.
- `_coerce_optional_bool(value: Any)` — [`L212`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L212) — Normalize optional boolean-like values from lm-eval configs.
- `_coerce_sampling_params_template(sampling_params: SamplingParams | collections.abc.Mapping[str, Any] | None)` — [`L245`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L245) — Normalize a benchmark sampling_params template into SamplingParams.
- `_extract_math_answer(generation: str)` — [`L307`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L307) — Extract a numeric answer from a model generation string.
- `_freeze_generation_value(value: Any)` — [`L199`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L199) — Convert nested generation kwargs into a hashable grouping key.
- `_get_rolling_token_windows(token_list: list[int], *, prefix_token: int, max_seq_len: int, context_len: int)` — [`L366`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L366) — Generate rolling token windows for perplexity computation.
- `_is_math_answer_task(task_name: str | None, task_hints: tuple[str, ...])` — [`L264`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L264) — Determine whether a task name indicates a math-answer evaluation task.
- `_make_disjoint_window(context: list[int], continuation: list[int])` — [`L425`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L425) — Make context and continuation disjoint for rolling evaluation.
- `_merge_stop_lists(existing_stops: Any, requested_stops: Any)` — [`L236`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L236) — Merge stop strings while preserving order and avoiding duplicates.
- `_normalize_math_generation(generation: str)` — [`L337`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L337) — Ensure a model generation ends with a ``#### <answer>`` marker.
- `_normalize_number_candidate(candidate: str)` — [`L282`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L282) — Normalize a raw string into a clean numeric representation.
- `_normalize_stop_list(stop_value: Any)` — [`L225`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L225) — Coerce lm-eval stop configuration into a flat string list.
- `_postprocess_generation_text(generation: str, stop: list[str] | str | None, think_end_token: str | None)` — [`L149`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L149) — Strip optional reasoning prefix and trim any trailing stop text.
- `_strip_empty_reasoning_scaffold(rendered_prompt: str, think_start_token: str | None, think_end_token: str | None)` — [`L169`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L169) — Remove tokenizer-injected empty reasoning blocks from rendered prompts.
- `_trim_stop_sequences(text: str, stop: list[str])` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L120) — Remove everything after the first occurrence of any stop sequence.

## Module values
- `_BOXED_ANSWER_RE` — [`L93`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L93)
- `_DEFAULT_MATH_ANSWER_TASK_HINTS` — [`L85`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L85)
- `_DEFAULT_REQUEST_ID_PREFIX` — [`L84`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L84)
- `_NUMBER_RE` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L94)
- `_STRICT_MATH_ANSWER_RE` — [`L92`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L92)
- `err` — [`L81`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L81)
- `logger` — [`L75`](../../../../../../../raw/code/EasyDeL/easydel/inference/evaluations/esurge_eval.py#L75)

