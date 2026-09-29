---
title: 'Module: easydel/inference/esurge/mixins/utils.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/mixins/utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.mixins.utils`/
symbols:
  EngineUtilsMixin._format_chat_prompt: EngineUtilsMixin#_format_chat_prompt().
  EngineUtilsMixin._idle_monitor_thread: EngineUtilsMixin#_idle_monitor_thread.
  EngineUtilsMixin._finalize: EngineUtilsMixin#_finalize().
  EngineUtilsMixin: EngineUtilsMixin#
  EngineUtilsMixin._prepare_chat_sampling_params: EngineUtilsMixin#_prepare_chat_sampling_params().
  EngineUtilsMixin._prepare_sampling_params_for_request: EngineUtilsMixin#_prepare_sampling_params_for_request().
  EngineUtilsMixin._normalize_chat_template_messages: EngineUtilsMixin#_normalize_chat_template_messages().
  EngineUtilsMixin._collapse_system_messages: EngineUtilsMixin#_collapse_system_messages().
  EngineUtilsMixin._apply_extra_stops_to_sampling_params: EngineUtilsMixin#_apply_extra_stops_to_sampling_params().
  EngineUtilsMixin._apply_generation_config_to_sampling_params: EngineUtilsMixin#_apply_generation_config_to_sampling_params().
  EngineUtilsMixin._log_cache_event: EngineUtilsMixin#_log_cache_event().
  EngineUtilsMixin._drain_pipeline_workers: EngineUtilsMixin#_drain_pipeline_workers().
  EngineUtilsMixin._clone_sampling_params: EngineUtilsMixin#_clone_sampling_params().
  EngineUtilsMixin._merge_system_content: EngineUtilsMixin#_merge_system_content().
  EngineUtilsMixin._normalize_wrapped_chat_template_tools: EngineUtilsMixin#_normalize_wrapped_chat_template_tools().
  EngineUtilsMixin._record_cache_event: EngineUtilsMixin#_record_cache_event().
  EngineUtilsMixin._idle_loop: EngineUtilsMixin#_idle_loop().
  EngineUtilsMixin._decode_with_pipeline: EngineUtilsMixin#_decode_with_pipeline().
  EngineUtilsMixin._stop_idle_monitor: EngineUtilsMixin#_stop_idle_monitor().
  EngineUtilsMixin._sanitize_metrics_payload: EngineUtilsMixin#_sanitize_metrics_payload().
  EngineUtilsMixin._kv_cache_metadata: EngineUtilsMixin#_kv_cache_metadata().
  EngineUtilsMixin._coerce_mapping_like: EngineUtilsMixin#_coerce_mapping_like().
  EngineUtilsMixin._normalize_stop_sequences: EngineUtilsMixin#_normalize_stop_sequences().
  EngineUtilsMixin._idle_reset_last_activity: EngineUtilsMixin#_idle_reset_last_activity.
  EngineUtilsMixin._tokenize_prompt_segments: EngineUtilsMixin#_tokenize_prompt_segments().
  EngineUtilsMixin._compute_snapshot_delta_text: EngineUtilsMixin#_compute_snapshot_delta_text().
  EngineUtilsMixin._start_idle_monitor: EngineUtilsMixin#_start_idle_monitor().
  EngineUtilsMixin._content_to_text_parts: EngineUtilsMixin#_content_to_text_parts().
  EngineUtilsMixin._normalize_chat_template_tools: EngineUtilsMixin#_normalize_chat_template_tools().
  EngineUtilsMixin._is_recoverable_chat_template_tool_error: EngineUtilsMixin#_is_recoverable_chat_template_tool_error().
  EngineUtilsMixin._to_python_scalar: EngineUtilsMixin#_to_python_scalar().
  WORKER_DRAIN_MAX_RETRIES: WORKER_DRAIN_MAX_RETRIES.
  WORKER_DRAIN_INITIAL_DELAY: WORKER_DRAIN_INITIAL_DELAY.
  EngineUtilsMixin._to_structured_text_messages: EngineUtilsMixin#_to_structured_text_messages().
  EngineUtilsMixin._prepare_prompt_segments: EngineUtilsMixin#_prepare_prompt_segments().
  EngineUtilsMixin._filter_eos_tokens: EngineUtilsMixin#_filter_eos_tokens().
  EngineUtilsMixin._TOOL_TOKEN_PATTERNS: EngineUtilsMixin#_TOOL_TOKEN_PATTERNS.
  EngineUtilsMixin._tokenize_prompt: EngineUtilsMixin#_tokenize_prompt().
  EngineUtilsMixin._touch_activity: EngineUtilsMixin#_touch_activity().
---
# Module: [`easydel/inference/esurge/mixins/utils.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py)

## Classes
### `EngineUtilsMixin`
- def: [`easydel/inference/esurge/mixins/utils.py:35`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L35)
- doc: Mixin providing utility helpers for the eSurge engine.
- signature: `class EngineUtilsMixin:`
- members:
  - `_apply_extra_stops_to_sampling_params(self, sampling_params: SamplingParams)` — [`L338`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L338) — Merge engine-level stop strings into request sampling parameters.
  - `_apply_generation_config_to_sampling_params(self, sampling_params: SamplingParams)` — [`L355`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L355) — Merge model generation-config EOS IDs into sampling stop-token policy.
  - `_clone_sampling_params(self, sampling_params: SamplingParams)` — [`L800`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L800) — Create a deep copy of sampling parameters.
  - `_coerce_mapping_like(value: Any)` — [`L48`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L48) — Coerce JSON-string payloads into mapping-like objects when possible.
  - `_collapse_system_messages(messages: list[dict[str, Any]])` — [`L137`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L137) — Fold all system turns into one leading system message for strict templates.
  - `_compute_snapshot_delta_text(current_text: str, previous_text: str, fallback_delta: str)` — [`L575`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L575) — Compute a safe streaming delta from accumulated text snapshots.
  - `_content_to_text_parts(content: Any)` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L106) — Convert arbitrary message content into text-part arrays.
  - `_decode_with_pipeline(self, request_id: str, generated_tokens: list[int], *, finished: bool, skip_special_tokens: bool = False, spaces_between_special_tokens: bool = True, prompt_context: list[int] | None = None)` — [`L538`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L538) — Decode tokens using the detokenizer worker pipeline.
  - `_drain_pipeline_workers(self, reason: str)` — [`L657`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L657) — Drain tokenizer/detokenizer workers with retry logic.
  - `_filter_eos_tokens(self, tokens: list[int])` — [`L495`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L495) — Remove EOS tokens from a token list before decoding.
  - `_format_chat_prompt(self, messages: list[dict[str, str]], add_generation_prompt: bool = True, chat_template: str | None = None, tools: list[dict] | None = None, chat_template_kwargs: dict[str, Any] | None = None)` — [`L373`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L373) — Format chat messages into a prompt string using the tokenizer's chat template.
  - `_is_recoverable_chat_template_tool_error(exc: Exception)` — [`L261`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L261) — Return True for template/tool shape mismatches we can retry around.
  - `_kv_cache_metadata(self)` — [`L612`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L612) — Get current KV cache configuration metadata.
  - `_log_cache_event(self, event: str, extra: dict[str, Any] | None = None)` — [`L643`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L643) — Log a KV cache event with metadata.
  - `_merge_system_content(existing: Any, new_content: Any)` — [`L122`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L122) — Merge multiple system-message contents into a single leading turn.
  - `_normalize_chat_template_messages(messages: list[dict[str, Any]])` — [`L60`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L60) — Normalize message payloads for HF/Jinja chat template compatibility.
  - `_normalize_chat_template_tools(tools: list[dict[str, Any]] | None)` — [`L176`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L176) — Normalize tool definitions for HF chat templates.
  - `_normalize_stop_sequences(stop: typing.Any)` — [`L314`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L314) — Normalize stop input into a de-duplicated list of non-empty strings.
  - `_normalize_wrapped_chat_template_tools(tools: list[dict[str, Any]] | None)` — [`L238`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L238) — Normalize tools into OpenAI-style ``{"type":"function","function":...}`` wrappers.
  - `_prepare_chat_sampling_params(self, sampling_params: SamplingParams | None, *, tools: list[dict[str, Any]] | None = None, tool_choice: str | None = None)` — [`L754`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L754) — Prepare sampling params with tool-token awareness.
  - `_prepare_prompt_segments(self, prompt: typing.Any)` — [`L482`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L482) — Convert a prompt to a list of string segments.
  - `_prepare_sampling_params_for_request(self, template: SamplingParams, *, request_id: str, prompt: str)` — [`L815`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L815) — Prepare sampling parameters for a specific request.
  - `_record_cache_event(self, event: str, payload: dict[str, Any])` — [`L632`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L632) — Record a cache event to the metrics collector.
  - `_sanitize_metrics_payload(self, payload: dict[str, Any])` — [`L601`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L601) — Sanitize a metrics payload by converting arrays to scalars.
  - `_start_idle_monitor(self)` — [`L700`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L700) — Start the idle-reset monitor thread if enabled.
  - `_stop_idle_monitor(self)` — [`L738`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L738) — Stop the idle-reset monitor thread if running.
  - `_to_python_scalar(value: Any)` — [`L585`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L585) — Convert a value to a Python scalar if possible.
  - `_to_structured_text_messages(messages: list[dict[str, Any]])` — [`L281`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L281) — Convert message content into structured text-part arrays.
  - `_tokenize_prompt(self, request_id: str, prompt: str)` — [`L470`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L470) — Tokenize a prompt string using the worker pipeline.
  - `_tokenize_prompt_segments(self, prompt: typing.Any)` — [`L512`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L512) — Tokenize prompt segments individually.
  - `_touch_activity(self)` — [`L694`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L694) — Update the last-activity timestamp for idle reset tracking.
- protocol/private: `_TOOL_TOKEN_PATTERNS`[`L752`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L752), `_finalize`[`L837`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L837), `_idle_loop`[`L710`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L710), `_idle_monitor_thread`[`L735`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L735), `_idle_reset_last_activity`[`L698`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L698)
- uses (calls/refs, reference-scoped): [`SamplingParams`](../../sampling_params.md#SamplingParams), [`logger`](../logger.md#logger), [`eSurge`](../esurge_engine.md#eSurge), [`get_metrics_collector`](../metrics.md#get_metrics_collector), [`update_with_generation_config`](../../sampling_params.md#SamplingParams.update_with_generation_config), [`compute_stream_delta_text`](../../stream_protocol.md#compute_stream_delta_text), [`stop`](../../sampling_params.md#SamplingParams.stop), [`record_cache_event`](../metrics.md#MetricsCollector.record_cache_event), [`DetokenizerResult`](../../../workers/esurge/pipeline/zmq_workers.md#DetokenizerResult), [`logit_bias`](../../sampling_params.md#SamplingParams.logit_bias), [`skip_special_tokens`](../../sampling_params.md#SamplingParams.skip_special_tokens), [`WORKER_DRAIN_INITIAL_DELAY`](utils.md#WORKER_DRAIN_INITIAL_DELAY), [`WORKER_DRAIN_MAX_RETRIES`](utils.md#WORKER_DRAIN_MAX_RETRIES), [`ignore_stop_strings_in_reasoning`](../../sampling_params.md#SamplingParams.ignore_stop_strings_in_reasoning)
- used by: [`eSurge`](../esurge_engine.md#eSurge), [`extra_stops`](../esurge_engine.md#eSurge.extra_stops)

## Module values
- `WORKER_DRAIN_INITIAL_DELAY` — [`L32`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L32)
- `WORKER_DRAIN_MAX_RETRIES` — [`L31`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/utils.py#L31)

