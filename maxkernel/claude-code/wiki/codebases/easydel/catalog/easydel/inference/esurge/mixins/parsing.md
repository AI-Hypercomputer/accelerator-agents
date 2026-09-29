---
title: 'Module: easydel/inference/esurge/mixins/parsing.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/mixins/parsing.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.mixins.parsing`/EngineParsingMixin#
symbols:
  EngineParsingMixin._process_engine_outputs: _process_engine_outputs().
  EngineParsingMixin._finalize_request: _finalize_request().
  EngineParsingMixin._parse_with_stop_string_policy: _parse_with_stop_string_policy().
  EngineParsingMixin._finish_request_from_scheduler_signal: _finish_request_from_scheduler_signal().
  EngineParsingMixin: ''
  EngineParsingMixin._apply_stop_string_policy: _apply_stop_string_policy().
  EngineParsingMixin._decode_and_parse: _decode_and_parse().
  EngineParsingMixin._update_outputs: _update_outputs().
  EngineParsingMixin._run_output_parsers: _run_output_parsers().
  EngineParsingMixin._resolve_public_finish_reason: _resolve_public_finish_reason().
  EngineParsingMixin._find_first_stop_string: _find_first_stop_string().
  EngineParsingMixin._stop_strings_ignore_reasoning: _stop_strings_ignore_reasoning().
  EngineParsingMixin._update_metrics: _update_metrics().
---
# Module: [`easydel/inference/esurge/mixins/parsing.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/parsing.py)

## Classes
### `EngineParsingMixin`
- def: [`easydel/inference/esurge/mixins/parsing.py:26`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/parsing.py#L26)
- doc: Mixin for output parsing, stop-string detection, and token processing.
- signature: `class EngineParsingMixin:`
- members:
  - `_apply_stop_string_policy(self, rd: dict, *, accumulated_text: str, fallback_delta: str)` — [`L60`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/parsing.py#L60) — Apply stop-string trimming policy to decoded text.
  - `_decode_and_parse(self, request_id: str, rd: dict, decodable_tokens: list[int], now: float, finished: bool)` — [`L275`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/parsing.py#L275) — Decode tokens and run parser pipeline.
  - `_finalize_request(self, request_id: str, rd: dict, ro, comp, sample_index: int, parent_request_id: str, engine_output, force_finished: bool, stop_string_finishes: dict[str, str], metrics_collector, now: float)` — [`L407`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/parsing.py#L407) — Handle request completion: final decode, finish_reason, cleanup.
  - `_find_first_stop_string(text: str, stop_sequences: list[str])` — [`L43`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/parsing.py#L43) — Return the earliest stop-string match in text, if any.
  - `_finish_request_from_scheduler_signal(self, request_id: str, *, metrics_collector)` — [`L207`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/parsing.py#L207) — Finalize a request when the scheduler reports completion without token output.
  - `_parse_with_stop_string_policy(self, rd: dict, *, accumulated_text: str, delta_text: str, token_ids: list[int], finished: bool)` — [`L108`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/parsing.py#L108) — Parse decoded text and apply stop-string policy in the correct domain.
  - `_process_engine_outputs(self, engine_outputs: dict[int, EngineCoreOutputs])` — [`L500`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/parsing.py#L500) — Process engine outputs and update request outputs (thread-safe).
  - `_resolve_public_finish_reason(outputs)` — [`L190`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/parsing.py#L190) — Collapse per-sample completion reasons into a single public reason.
  - `_run_output_parsers(self, rd: dict, accumulated_text: str, delta_text: str, token_ids: list[int], finished: bool)` — [`L151`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/parsing.py#L151) — Run reasoning and tool parsers on decoded text via DelegatingParser.
  - `_stop_strings_ignore_reasoning(rd: dict)` — [`L98`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/parsing.py#L98) — Return whether stop strings should only match parsed visible content.
  - `_update_metrics(rd: dict, ro, now: float, num_generated: int)` — [`L388`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/parsing.py#L388) — Update TTFT, tokens/sec, and timing metrics.
  - `_update_outputs(comp, ro, sample_index: int, parsed: dict, raw_accumulated: str, raw_delta: str, visible_delta: str = "")` — [`L339`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/parsing.py#L339) — Apply parsed results to CompletionOutput and RequestOutput.
- uses (calls/refs, reference-scoped): [`logger`](../logger.md#logger), [`eSurge`](../esurge_engine.md#eSurge), [`EngineRequestStatus`](../request.md#EngineRequestStatus), [`get_metrics_collector`](../metrics.md#get_metrics_collector), [`add_generated_tokens`](../metrics.md#MetricsCollector.add_generated_tokens), [`record_first_token`](../metrics.md#MetricsCollector.record_first_token), [`compute_stream_delta_text`](../../stream_protocol.md#compute_stream_delta_text), [`EngineCoreOutputs`](../engine_types.md#EngineCoreOutputs), [`finished`](../engine_types.md#EngineCoreOutput.finished), [`outputs`](../engine_types.md#EngineCoreOutputs.outputs), [`FINISHED_STOPPED`](../request.md#EngineRequestStatus.FINISHED_STOPPED), [`finished_requests`](../engine_types.md#EngineCoreOutputs.finished_requests), [`new_token_ids`](../engine_types.md#EngineCoreOutput.new_token_ids), [`request_id`](../engine_types.md#EngineCoreOutput.request_id)
- used by: [`eSurge`](../esurge_engine.md#eSurge)

