---
title: 'Module: easydel/inference/esurge/mixins/io.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/mixins/io.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.mixins.io`/EngineIOMixin#
symbols:
  EngineIOMixin.stream: stream().
  EngineIOMixin._stream_multimodal_request: _stream_multimodal_request().
  EngineIOMixin: ''
  EngineIOMixin.iter_responses_stream: iter_responses_stream().
  EngineIOMixin._chat_multimodal: _chat_multimodal().
  EngineIOMixin.generate: generate().
  EngineIOMixin.iter_chat_completion_stream: iter_chat_completion_stream().
  EngineIOMixin.chat: chat().
  EngineIOMixin._wait_for_request: _wait_for_request().
  EngineIOMixin._recover_orphaned_request: _recover_orphaned_request().
  EngineIOMixin._build_tool_parser_request: _build_tool_parser_request().
  EngineIOMixin._ensure_scheduler_running: _ensure_scheduler_running().
  EngineIOMixin._messages_have_multimodal_content: _messages_have_multimodal_content().
---
# Module: [`easydel/inference/esurge/mixins/io.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/io.py)

## Classes
### `EngineIOMixin`
- def: [`easydel/inference/esurge/mixins/io.py:37`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/io.py#L37)
- doc: Mixin providing input/output operations for the eSurge engine.
- signature: `class EngineIOMixin:`
- members:
  - `_build_tool_parser_request(*, prompt: str, tools: list[dict] | None = None, tool_choice: str | dict[str, Any] | None = None)` — [`L117`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/io.py#L117) — Build a minimal chat request for parser-side schema access.
  - `_chat_multimodal(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str | dict[str, Any] | None = None, sampling_params: SamplingParams | None = None, request_id: str | None = None, stream: bool = False, chat_template: str | None = None, chat_template_kwargs: dict[str, Any] | None = None)` — [`L734`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/io.py#L734) — Handle multimodal chat with images/videos.
  - `_ensure_scheduler_running(self, *, context: str)` — [`L51`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/io.py#L51) — Raise a clear error when scheduler is unavailable.
  - `_messages_have_multimodal_content(self, messages: list[dict])` — [`L713`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/io.py#L713) — Check if messages contain multimodal content (images/videos).
  - `_recover_orphaned_request(self, request_id: str)` — [`L62`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/io.py#L62) — Abort a request when unresolved samples have no live scheduler/request state.
  - `_stream_multimodal_request(self, request_id: str)` — [`L838`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/io.py#L838) — Stream output for a multimodal request.
  - `_wait_for_request(self, request_id: str)` — [`L957`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/io.py#L957) — Wait for a request to complete and return the output.
  - `chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str | dict[str, Any] | None = None, sampling_params: SamplingParams | None = None, request_id: str | None = None, *, stream: Literal[False] = ..., chat_template: str | None = None, chat_template_kwargs: dict[str, Any] | None = None)` — [`L486`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/io.py#L486)
  - `generate(self, prompts: str | list[str], sampling_params: SamplingParams | None = None, request_id: str | list[str] | None = None, use_tqdm: bool = True, tool_parser_request: Any | None = None)` — [`L167`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/io.py#L167) — Generate completions for one or more prompts (blocking).
  - `iter_chat_completion_stream(self, *, model: str, messages: list[dict], tools: list[dict] | None = None, tool_choice: str | dict[str, Any] | None = None, sampling_params: SamplingParams | None = None, request_id: str | None = None, chat_template: str | None = None, chat_template_kwargs: dict[str, Any] | None = None)` — [`L647`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/io.py#L647) — Yield OpenAI chat stream chunks assembled from engine snapshots.
  - `iter_responses_stream(self, *, response_id: str, model: str, messages: list[dict], tools: list[dict] | None = None, tool_choice: str | dict[str, Any] | None = None, sampling_params: SamplingParams | None = None, request_id: str | None = None, include_reasoning_summary: bool = False, final_response_overrides: ResponsesFinalizationOptions | None = None, created_at: int | None = None, chat_template: str | None = None, chat_template_kwargs: dict[str, Any] | None = None)` — [`L676`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/io.py#L676) — Yield Responses API stream frames assembled from engine snapshots.
  - `stream(self, prompts: str | list[str], sampling_params: SamplingParams | None = None, request_id: str | None = None, tool_parser_request: Any | None = None)` — [`L299`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/io.py#L299) — Stream generation output as tokens are produced.
- uses (calls/refs, reference-scoped): [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`iter_chat_completion_stream_responses`](../../stream_protocol.md#iter_chat_completion_stream_responses), [`SamplingParams`](../../sampling_params.md#SamplingParams), [`logger`](../logger.md#logger), [`eSurge`](../esurge_engine.md#eSurge), [`RequestOutput`](../esurge_engine.md#RequestOutput), [`iter_responses_stream_frames`](../../stream_protocol.md#iter_responses_stream_frames), [`outputs`](../esurge_engine.md#RequestOutput.outputs), [`StreamEventFrame`](../../typed_models.md#StreamEventFrame), [`ChatCompletionStreamResponse`](../../openai_api_modules.md#ChatCompletionStreamResponse), [`ResponsesFinalizationOptions`](../../typed_models.md#ResponsesFinalizationOptions), [`num_generated_tokens`](../esurge_engine.md#RequestOutput.num_generated_tokens), [`RequestOutputLike`](../../stream_protocol.md#RequestOutputLike), [`accumulated_text`](../esurge_engine.md#RequestOutput.accumulated_text), [`CompletionOutput`](../esurge_engine.md#CompletionOutput), [`finished`](../esurge_engine.md#RequestOutput.finished), [`tokens_per_second`](../esurge_engine.md#RequestOutput.tokens_per_second), [`first_token_time`](../esurge_engine.md#RequestOutput.first_token_time), [`processing_time`](../esurge_engine.md#RequestOutput.processing_time), [`prompt_token_ids`](../esurge_engine.md#RequestOutput.prompt_token_ids), [`request_id`](../esurge_engine.md#RequestOutput.request_id), [`delta_text`](../esurge_engine.md#RequestOutput.delta_text), [`finish_reason`](../esurge_engine.md#CompletionOutput.finish_reason), [`reasoning_content`](../esurge_engine.md#CompletionOutput.reasoning_content), [`text`](../esurge_engine.md#CompletionOutput.text), [`tool_calls`](../esurge_engine.md#CompletionOutput.tool_calls), [`tool_calls`](../esurge_engine.md#RequestOutput.tool_calls), [`delta_tool_calls`](../esurge_engine.md#RequestOutput.delta_tool_calls), [`reasoning_content`](../esurge_engine.md#RequestOutput.reasoning_content), [`time_spent_generating`](../esurge_engine.md#RequestOutput.time_spent_generating), [`token_ids`](../esurge_engine.md#CompletionOutput.token_ids), [`delta_reasoning_content`](../esurge_engine.md#RequestOutput.delta_reasoning_content), [`delta_seq`](../esurge_engine.md#RequestOutput.delta_seq), [`index`](../esurge_engine.md#CompletionOutput.index), [`prompt`](../esurge_engine.md#RequestOutput.prompt), [`raw_accumulated_text`](../esurge_engine.md#RequestOutput.raw_accumulated_text), [`raw_delta_text`](../esurge_engine.md#RequestOutput.raw_delta_text), [`update_seq`](../esurge_engine.md#RequestOutput.update_seq), [`cumulative_logprob`](../esurge_engine.md#CompletionOutput.cumulative_logprob), [`logprobs`](../esurge_engine.md#CompletionOutput.logprobs)  (+2 more)
- used by: [`generate_stream`](../server/api_server.md#eSurgeApiServer.generate_stream), [`_generate`](../../evaluations/esurge_eval.md#eSurgeLMEvalAdapter._generate), [`eSurge`](../esurge_engine.md#eSurge), [`_run_chat`](../server/api_server.md#eSurgeApiServer._run_chat), [`generate`](../server/api_server.md#eSurgeAdapter.generate), [`_run_generate`](../server/api_server.md#eSurgeApiServer._run_generate)

