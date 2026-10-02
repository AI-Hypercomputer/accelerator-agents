---
title: 'Module: easydel/inference/esurge/server/api_server.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/server/api_server.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.server.api_server`/
symbols:
  eSurgeApiServer.responses: eSurgeApiServer#responses().
  eSurgeApiServer.generate_stream: eSurgeApiServer#generate_stream().
  eSurgeApiServer._build_chat_completion_response: eSurgeApiServer#_build_chat_completion_response().
  eSurgeApiServer._handle_completion_response: eSurgeApiServer#_handle_completion_response().
  eSurgeApiServer._build_stream_debug_context: eSurgeApiServer#_build_stream_debug_context().
  eSurgeApiServer._create_standard_response: eSurgeApiServer#_create_standard_response().
  eSurgeApiServer._endpoints: eSurgeApiServer#_endpoints().
  eSurgeApiServer.chat_completions: eSurgeApiServer#chat_completions().
  eSurgeApiServer.get_metrics: eSurgeApiServer#get_metrics().
  eSurgeApiServer._authorize_request: eSurgeApiServer#_authorize_request().
  eSurgeApiServer.completions: eSurgeApiServer#completions().
  eSurgeApiServer._handle_chat_completion: eSurgeApiServer#_handle_chat_completion().
  eSurgeApiServer._create_sampling_params: eSurgeApiServer#_create_sampling_params().
  logger: logger.
  eSurgeApiServer.health_check: eSurgeApiServer#health_check().
  eSurgeApiServer._prepare_chat_input: eSurgeApiServer#_prepare_chat_input().
  eSurgeApiServer.list_models: eSurgeApiServer#list_models().
  eSurgeApiServer._prepare_sampling_params: eSurgeApiServer#_prepare_sampling_params().
  eSurgeApiServer.adapters: eSurgeApiServer#adapters.
  eSurgeApiServer._stream_debug_len: eSurgeApiServer#_stream_debug_len().
  eSurgeApiServer.get_model: eSurgeApiServer#get_model().
  eSurgeApiServer.auth_manager: eSurgeApiServer#auth_manager.
  eSurgeApiServer._ensure_request_max_tokens: eSurgeApiServer#_ensure_request_max_tokens().
  eSurgeApiServer._prepare_messages_for_engine: eSurgeApiServer#_prepare_messages_for_engine().
  eSurgeApiServer._run_chat: eSurgeApiServer#_run_chat().
  eSurgeApiServer._handle_chat_streaming: eSurgeApiServer#_handle_chat_streaming().
  eSurgeAdapter.esurge: eSurgeAdapter#esurge.
  eSurgeAdapter.get_model_info: eSurgeAdapter#get_model_info().
  eSurgeApiServer._create_tools_response: eSurgeApiServer#_create_tools_response().
  eSurgeApiServer._infer_and_assign: eSurgeApiServer#_infer_and_assign().
  eSurgeApiServer._apply_extra_stops_to_sampling_params: eSurgeApiServer#_apply_extra_stops_to_sampling_params().
  eSurgeApiServer._auth_system_enabled: eSurgeApiServer#_auth_system_enabled().
  eSurgeApiServer._record_api_key_usage: eSurgeApiServer#_record_api_key_usage().
  eSurgeApiServer._infer_sequence_length_from_engine: eSurgeApiServer#_infer_sequence_length_from_engine().
  eSurgeApiServer._handle_completion_streaming: eSurgeApiServer#_handle_completion_streaming().
  eSurgeApiServer.status: eSurgeApiServer#status.
  eSurgeApiServer._abort_non_stream_request_on_disconnect: eSurgeApiServer#_abort_non_stream_request_on_disconnect().
  eSurgeApiServer._abort_request_after_handler_cancel: eSurgeApiServer#_abort_request_after_handler_cancel().
  eSurgeAdapter.generate: eSurgeAdapter#generate().
  eSurgeApiServer.__init__: eSurgeApiServer#__init__().
  eSurgeApiServer._get_adapter: eSurgeApiServer#_get_adapter().
  eSurgeApiServer.generate_api_key: eSurgeApiServer#generate_api_key().
  eSurgeApiServer._prepare_chat_input_async: eSurgeApiServer#_prepare_chat_input_async().
  eSurgeApiServer._handle_chat_completion_multimodal: eSurgeApiServer#_handle_chat_completion_multimodal().
  eSurgeApiServer: eSurgeApiServer#
  RefineSamplingParamsFn: RefineSamplingParamsFn.
  eSurgeApiServer._count_tokens: eSurgeApiServer#_count_tokens().
  eSurgeApiServer._handle_chat_streaming_multimodal: eSurgeApiServer#_handle_chat_streaming_multimodal().
  eSurgeAdapter.count_tokens: eSurgeAdapter#count_tokens().
  eSurgeAdapter.processor: eSurgeAdapter#processor().
  eSurgeApiServer._extra_stops: eSurgeApiServer#_extra_stops.
  eSurgeApiServer._prompt_token_count_from_output: eSurgeApiServer#_prompt_token_count_from_output().
  eSurgeAdapter: eSurgeAdapter#
  eSurgeApiServer._run_generate: eSurgeApiServer#_run_generate().
  eSurgeApiServer._extract_payload_api_keys: eSurgeApiServer#_extract_payload_api_keys().
  eSurgeApiServer.on_startup: eSurgeApiServer#on_startup().
  eSurgeApiServer.add_candidate: eSurgeApiServer#add_candidate().
  eSurgeApiServer.list_tools: eSurgeApiServer#list_tools().
  eSurgeApiServer.execute_tool: eSurgeApiServer#execute_tool().
  eSurgeApiServer._stream_debug_preview: eSurgeApiServer#_stream_debug_preview().
  RefineChatRequestFn: RefineChatRequestFn.
  eSurgeApiServer._get_engine_tool_parser: eSurgeApiServer#_get_engine_tool_parser().
  eSurgeApiServer._mark_auto: eSurgeApiServer#_mark_auto().
  eSurgeApiServer._mark_manual: eSurgeApiServer#_mark_manual().
  eSurgeApiServer._resolve_esurge_runtime_request_cap: eSurgeApiServer#_resolve_esurge_runtime_request_cap().
  _STREAM_ERROR: _STREAM_ERROR.
  _STREAM_END: _STREAM_END.
  eSurgeAdapter.__init__: eSurgeAdapter#__init__().
  eSurgeAdapter.model_name: eSurgeAdapter#model_name().
  eSurgeApiServer.on_shutdown: eSurgeApiServer#on_shutdown().
  eSurgeAdapter._model_name: eSurgeAdapter#_model_name.
  eSurgeApiServer.oai_like_processor: eSurgeApiServer#oai_like_processor.
  eSurgeApiServer._refine_sampling_params_callback: eSurgeApiServer#_refine_sampling_params_callback.
  eSurgeApiServer._refine_chat_request_callback: eSurgeApiServer#_refine_chat_request_callback.
  eSurgeApiServer._require_api_key: eSurgeApiServer#_require_api_key.
  eSurgeApiServer._normalize_stop_sequences: eSurgeApiServer#_normalize_stop_sequences().
  eSurgeApiServer._coerce_optional_bool: eSurgeApiServer#_coerce_optional_bool().
  eSurgeApiServer._extract_api_key: eSurgeApiServer#_extract_api_key().
  eSurgeApiServer._example_tool_definitions: eSurgeApiServer#_example_tool_definitions().
  eSurgeApiServer._create_tool_execution_placeholder_response: eSurgeApiServer#_create_tool_execution_placeholder_response().
  eSurgeApiServer._messages_have_multimodal_content: eSurgeApiServer#_messages_have_multimodal_content().
  eSurgeApiServer._prepare_text_messages_for_chat: eSurgeApiServer#_prepare_text_messages_for_chat().
  __all__: __all__.
  _STREAM_DATA: _STREAM_DATA.
  eSurgeApiServer.esurge_map: eSurgeApiServer#esurge_map.
  eSurgeApiServer._resolved_generation_slot_cap: eSurgeApiServer#_resolved_generation_slot_cap.
  eSurgeApiServer._looks_like_tool_protocol_text: eSurgeApiServer#_looks_like_tool_protocol_text().
---
# Module: [`easydel/inference/esurge/server/api_server.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py)

## Classes
### `eSurgeAdapter`  ·  implements/extends InferenceEngineAdapter
- def: [`easydel/inference/esurge/server/api_server.py:153`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L153)
- doc: Adapter for eSurge inference engine.
- signature: `class eSurgeAdapter(InferenceEngineAdapter):`
- members:
  - `__init__(self, esurge_instance: eSurge, model_name: str)` — [`L161`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L161) — Initialize the adapter with an eSurge engine and model name.
  - `count_tokens(self, content: str)` — [`L196`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L196) — Count tokens using eSurge tokenizer.
  - `generate(self, prompts: str | list[str], sampling_params: SamplingParams, stream: bool = False)` — [`L171`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L171) — Generate text using eSurge engine.
  - `get_model_info(self)` — [`L207`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L207) — Get eSurge model information.
  - `model_name(self)` — [`L222`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L222) — Return the model name.
  - `processor(self)` — [`L231`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L231) — Return the tokenizer/processor associated with the eSurge instance.
  - `esurge` — [`L168`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L168)
- protocol/private: `_model_name`[`L169`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L169)
- uses (calls/refs, reference-scoped): [`SamplingParams`](../../sampling_params.md#SamplingParams), [`eSurge`](../esurge_engine.md#eSurge), [`RequestOutput`](../esurge_engine.md#RequestOutput), [`generate`](../mixins/io.md#EngineIOMixin.generate), [`tokenizer`](../esurge_engine.md#eSurge.tokenizer), [`max_model_len`](../esurge_engine.md#eSurge.max_model_len), [`InferenceEngineAdapter`](../../inference_engine_interface.md#InferenceEngineAdapter), [`max_num_seqs`](../esurge_engine.md#eSurge.max_num_seqs)
- used by: [`responses`](api_server.md#eSurgeApiServer.responses), [`chat_completions`](api_server.md#eSurgeApiServer.chat_completions), [`completions`](api_server.md#eSurgeApiServer.completions), [`health_check`](api_server.md#eSurgeApiServer.health_check), [`list_models`](api_server.md#eSurgeApiServer.list_models), [`adapters`](api_server.md#eSurgeApiServer.adapters), [`get_model`](api_server.md#eSurgeApiServer.get_model), [`_create_tools_response`](api_server.md#eSurgeApiServer._create_tools_response), [`_infer_sequence_length_from_engine`](api_server.md#eSurgeApiServer._infer_sequence_length_from_engine), [`_get_adapter`](api_server.md#eSurgeApiServer._get_adapter), [`_count_tokens`](api_server.md#eSurgeApiServer._count_tokens), [`generate`](../../inference_engine_interface.md#InferenceEngineAdapter.generate), [`InferenceEngineAdapter`](../../inference_engine_interface.md#InferenceEngineAdapter), [`count_tokens`](../../inference_engine_interface.md#InferenceEngineAdapter.count_tokens), [`get_model_info`](../../inference_engine_interface.md#InferenceEngineAdapter.get_model_info), [`model_name`](../../inference_engine_interface.md#InferenceEngineAdapter.model_name), [`processor`](../../inference_engine_interface.md#InferenceEngineAdapter.processor)

### `eSurgeApiServer`  ·  implements/extends AuthEndpointsMixin, BaseInferenceApiServer
- def: [`easydel/inference/esurge/server/api_server.py:241`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L241)
- doc: eSurge-specific API server implementation with OpenAI compatibility.
- signature: `class eSurgeApiServer(BaseInferenceApiServer, AuthEndpointsMixin):`
- members:
  - `__init__(self, esurge_map: dict[str, eSurge] | eSurge, oai_like_processor: bool = True, enable_function_calling: bool = True, require_api_key: bool = False, admin_key: str | None = None, enable_audit_logging: bool = True, max_audit_entries: int = 10000, storage_dir: str | None = None, enable_persistence: bool = True, auto_save_interval: float = 60, auth_worker_client: tp.Any | None = None, response_store_worker_client: tp.Any | None = None, max_concurrent_generations: int | None = None, overload_message: str = "Server is busy, please try again later", extra_stops: str | list[str] | None = None, refine_sampling_params: RefineSamplingParamsFn | None = None, refine_chat_request: RefineChatRequestFn | None = None, enable_response_store: bool = True, default_store_responses: bool = True, max_stored_responses: int = 10000, max_stored_conversations: int = 1000, **kwargs)` — [`L258`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L258) — Initialize the eSurge API server.
  - `_abort_non_stream_request_on_disconnect(self, *, raw_request: Request, esurge: eSurge, request_id: str, endpoint: str, model: str | None, done_event: asyncio.Event, poll_interval_s: float = 0.1)` — [`L943`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L943) — Abort non-stream engine work when the HTTP client disconnects.
  - `_abort_request_after_handler_cancel(*, esurge: eSurge, request_id: str, endpoint: str, model: str | None)` — [`L995`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L995) — Abort engine work when the HTTP handler itself is cancelled.
  - `_apply_extra_stops_to_sampling_params(self, sampling_params: SamplingParams)` — [`L1201`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1201) — Merge server-level stop strings into request sampling parameters.
  - `_auth_system_enabled(self)` — [`L550`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L550) — Determine whether authentication enforcement is active.
  - `_authorize_request(self, raw_request: Request, payload_api_keys: str | Iterable[str | None] | None = None, endpoint: str | None = None, model: str | None = None, requested_tokens: int = 0)` — [`L587`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L587) — Authorize request using enhanced auth system with RBAC and rate limiting.
  - `_build_chat_completion_response(self, request: ChatCompletionRequest, esurge: eSurge, output: RequestOutput, raw_request: Request)` — [`L1929`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1929) — Build a ChatCompletionResponse from a finalized RequestOutput snapshot.
  - `_build_stream_debug_context(cls, *, endpoint: str, request_id: str | None, model: str | None, queue_kind: str | None = None, disconnected: bool | None = None, output: RequestOutput | None = None, last_output: RequestOutput | None = None, previous_text: tp.Any = None, current_text: tp.Any = None, delta_text: tp.Any = None, previous_token_ids: tp.Any = None, current_token_ids: tp.Any = None, delta_token_ids: tp.Any = None, raw_delta_message: tp.Any = None, delta_message: tp.Any = None, delta_tool_calls_raw: tp.Any = None, saw_tool_call_delta: bool | None = None, saw_function_call_delta: bool | None = None, stream_error: Exception | None = None, tools: tp.Any = None, messages: tp.Any = None)` — [`L1016`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1016) — Build a bounded debug payload for streaming failures.
  - `_count_tokens(self, content: str, model_name: str | None = None)` — [`L434`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L434) — Count tokens for the given content.
  - `_create_sampling_params(self, request: ChatCompletionRequest | CompletionRequest)` — [`L1217`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1217) — Create sampling parameters from request.
  - `_create_standard_response(self, request: ChatCompletionRequest, output: tp.Any, prompt_tokens: int, start_time: float)` — [`L2640`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L2640) — Create a standard chat completion response without function calling.
  - `_create_tool_execution_placeholder_response()` — [`L789`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L789) — Return the placeholder response for the unimplemented tool executor.
  - `_create_tools_response(self)` — [`L773`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L773) — Describe tool-calling capabilities as reported by each eSurge engine.
  - `_endpoints(self)` — [`L2705`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L2705) — Define all API endpoints including admin authentication endpoints.
  - `_ensure_request_max_tokens(self, request: ChatCompletionRequest | CompletionRequest, esurge: eSurge | None = None)` — [`L802`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L802) — Ensure the request has a valid max_tokens value.
  - `_example_tool_definitions()` — [`L746`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L746) — Return placeholder tool definitions for discovery endpoints.
  - `_extract_api_key(self, raw_request: Request)` — [`L472`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L472) — Extract API key from various request locations.
  - `_extract_payload_api_keys(payload: tp.Any)` — [`L506`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L506) — Extract API key candidates from the request JSON payload.
  - `_get_adapter(self, model_name: str)` — [`L416`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L416) — Get adapter by model name.
  - `_get_engine_tool_parser(esurge: eSurge)` — [`L736`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L736) — Return the parser name configured on the eSurge engine, if any.
  - `_handle_chat_completion(self, request: ChatCompletionRequest, esurge: eSurge, messages: list[dict[str, tp.Any]], request_id: str, raw_request: Request)` — [`L1990`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1990) — Handle non-streaming chat completion via eSurge.chat().
  - `_handle_chat_completion_multimodal(self, request: ChatCompletionRequest, esurge: eSurge, messages: list[dict[str, tp.Any]], request_id: str, raw_request: Request)` — [`L1523`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1523) — Backward-compatible wrapper around the unified chat completion path.
  - `_handle_chat_streaming(self, request: ChatCompletionRequest, esurge: eSurge, messages: list[dict[str, tp.Any]], request_id: str, raw_request: Request)` — [`L2055`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L2055) — Handle streaming chat completion via eSurge.chat().
  - `_handle_chat_streaming_multimodal(self, request: ChatCompletionRequest, esurge: eSurge, messages: list[dict[str, tp.Any]], request_id: str, raw_request: Request)` — [`L1534`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1534) — Backward-compatible wrapper around the unified chat streaming path.
  - `_handle_completion_response(self, request: CompletionRequest, esurge: eSurge, prompt: str, request_id: str, raw_request: Request)` — [`L2211`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L2211) — Handle non-streaming completion.
  - `_handle_completion_streaming(self, request: CompletionRequest, esurge: eSurge, prompt: str, request_id: str, raw_request: Request)` — [`L2311`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L2311) — Handle streaming completion with delta chunks.
  - `_infer_sequence_length_from_engine(self, esurge: eSurge | None)` — [`L702`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L702) — Infer the maximum sequence length from an eSurge engine.
  - `_looks_like_tool_protocol_text(text: str | None)` — [`L895`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L895) — Return True when text appears to be tool protocol/control markup.
  - `_messages_have_multimodal_content(messages: list[dict[str, tp.Any]])` — [`L1437`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1437) — Check if the message list contains multimodal content (images or videos).
  - `_normalize_stop_sequences(stop: tp.Any)` — [`L1144`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1144) — Normalize stop input into a de-duplicated list of non-empty strings.
  - `_prepare_chat_input(self, request: ChatCompletionRequest, esurge: eSurge)` — [`L1288`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1288) — Prepare chat input for model.
  - `_prepare_chat_input_async(self, request: ChatCompletionRequest, esurge: eSurge)` — [`L1365`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1365) — Prepare chat input asynchronously.
  - `_prepare_messages_for_engine(self, request: ChatCompletionRequest, esurge: eSurge)` — [`L1469`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1469) — Convert Pydantic request messages into dictionaries for the eSurge engine.
  - `_prepare_sampling_params(self, request: ChatCompletionRequest | CompletionRequest, esurge: eSurge)` — [`L868`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L868) — Create and optionally refine sampling parameters for generation.
  - `_prepare_text_messages_for_chat(messages: list[dict[str, tp.Any]])` — [`L1495`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1495) — Normalize text-only messages into plain string content for chat templates.
  - `_prompt_token_count_from_output(output: RequestOutput)` — [`L1546`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1546) — Extract the prompt token count from a RequestOutput.
  - `_record_api_key_usage(self, raw_request: Request | None, prompt_tokens: int, completion_tokens: int)` — [`L670`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L670) — Record token usage statistics for the authenticated API key.
  - `_resolve_esurge_runtime_request_cap(esurge: eSurge)` — [`L1169`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1169) — Infer the effective request cap for one engine.
  - `_stream_debug_len(value: tp.Any)` — [`L934`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L934) — Return length for strings/sequences/mappings when available.
  - `_stream_debug_preview(text: tp.Any, *, max_chars: int = 160)` — [`L923`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L923) — Return a compact escaped preview for debug logging.
  - `add_candidate(value: tp.Any)` — [`L528`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L528) — Append a non-empty, stripped string to the candidates list if unique.
  - `chat_completions(self, request: ChatCompletionRequest, raw_request: Request)` — [`L1380`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1380) — Handle chat completion requests.
  - `completions(self, request: CompletionRequest, raw_request: Request)` — [`L2161`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L2161) — Handle completion requests.
  - `execute_tool(self, raw_request: Request)` — [`L2685`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L2685) — Execute a tool/function call.
  - `generate_api_key(self, name: str, role: tp.Any = None, **kwargs)` — [`L450`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L450) — Create and register a new random API key with enhanced features.
  - `generate_stream()` — [`L1805`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1805) — Async generator yielding SSE events for Responses API streaming.
  - `get_metrics(self, raw_request: Request)` — [`L2536`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L2536) — Get server performance metrics.
  - `get_model(self, model_id: str, raw_request: Request)` — [`L2599`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L2599) — Get model details.
  - `health_check(self, raw_request: Request)` — [`L2469`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L2469) — Health check endpoint.
  - `list_models(self, raw_request: Request)` — [`L2564`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L2564) — List available models.
  - `list_tools(self, raw_request: Request)` — [`L2629`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L2629) — List available tools/functions for each model.
  - `on_shutdown(self)` — [`L407`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L407) — Custom shutdown logic for eSurge.
  - `on_startup(self)` — [`L396`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L396) — Custom startup logic for eSurge.
  - `responses(self, request: ResponsesRequest, raw_request: Request)` — [`L1568`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1568) — Handle OpenAI Responses API requests.
  - `adapters` — [`L317`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L317)
  - `auth_manager` — [`L362`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L362)
  - `esurge_map` — [`L316`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L316)
  - `oai_like_processor` — [`L330`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L330)
  - `status` — [`L413`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L413)
- protocol/private: `_coerce_optional_bool`[`L1258`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1258), `_extra_stops`[`L333`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L333), `_infer_and_assign`[`L843`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L843), `_mark_auto`[`L830`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L830), `_mark_manual`[`L837`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L837), `_refine_chat_request_callback`[`L332`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L332), `_refine_sampling_params_callback`[`L331`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L331), `_require_api_key`[`L358`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L358), `_resolved_generation_slot_cap`[`L354`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L354), `_run_chat`[`L1714`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L1714), `_run_generate`[`L2255`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L2255)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`create_api_key_endpoint`](auth_endpoints.md#AuthEndpointsMixin.create_api_key_endpoint), [`update_api_key_endpoint`](auth_endpoints.md#AuthEndpointsMixin.update_api_key_endpoint), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`stream`](../mixins/io.md#EngineIOMixin.stream), [`SamplingParams`](../../sampling_params.md#SamplingParams), [`eSurge`](../esurge_engine.md#eSurge), [`ChatMessage`](../../openai_api_modules.md#ChatMessage), [`_endpoints`](../../inference_engine_interface.md#BaseInferenceApiServer._endpoints), [`as_update_dict`](../../typed_models.md#ResponsesFinalizationOptions.as_update_dict), [`RequestOutput`](../esurge_engine.md#RequestOutput), [`authorize_request`](../../../workers/esurge/auth/auth_manager.md#EnhancedApiKeyManager.authorize_request), [`get_statistics`](../../../workers/esurge/auth/auth_manager.md#EnhancedApiKeyManager.get_statistics), [`metrics`](../../inference_engine_interface.md#BaseInferenceApiServer.metrics), [`ApiKeyRole`](../../../workers/esurge/auth/auth_models.md#ApiKeyRole), [`_create_sampling_params_from_responses`](../../inference_engine_interface.md#BaseInferenceApiServer._create_sampling_params_from_responses), [`logger`](api_server.md#logger), [`ResponsesOutputItem`](../../typed_models.md#ResponsesOutputItem), [`record_usage`](../../../workers/esurge/auth/auth_manager.md#EnhancedApiKeyManager.record_usage), [`reasoning_content`](../../openai_api_modules.md#DeltaMessage.reasoning_content), [`abort_request`](../mixins/requests.md#EngineRequestsMixin.abort_request), [`EndpointConfig`](../../inference_engine_interface.md#EndpointConfig), [`handler`](../../inference_engine_interface.md#EndpointConfig.handler), [`methods`](../../inference_engine_interface.md#EndpointConfig.methods), [`path`](../../inference_engine_interface.md#EndpointConfig.path), [`summary`](../../inference_engine_interface.md#EndpointConfig.summary), [`tags`](../../inference_engine_interface.md#EndpointConfig.tags), [`_responses_payload_to_messages`](../../inference_engine_interface.md#BaseInferenceApiServer._responses_payload_to_messages), [`content`](../../openai_api_modules.md#ChatMessage.content), [`generate_api_key`](../../../workers/esurge/auth/auth_manager.md#EnhancedApiKeyManager.generate_api_key), [`CompletionRequest`](../../openai_api_modules.md#CompletionRequest), [`outputs`](../esurge_engine.md#RequestOutput.outputs), [`StreamEventFrame`](../../typed_models.md#StreamEventFrame), [`_response_store_put_conversation`](../../inference_engine_interface.md#BaseInferenceApiServer._response_store_put_conversation), [`_response_store_put_response`](../../inference_engine_interface.md#BaseInferenceApiServer._response_store_put_response), [`role`](../../openai_api_modules.md#ChatMessage.role), [`convert_to_openai_format`](../../../trainers/prompt_utils.md#convert_to_openai_format), [`create_error_response`](../../inference_engine_interface.md#create_error_response)  (+174 more)
- used by: [`_import_structure`](../../../__init__.md#_import_structure), [`_endpoints`](../../inference_engine_interface.md#BaseInferenceApiServer._endpoints), [`_create_sampling_params`](../../inference_engine_interface.md#BaseInferenceApiServer._create_sampling_params), [`chat_completions`](../../inference_engine_interface.md#BaseInferenceApiServer.chat_completions), [`completions`](../../inference_engine_interface.md#BaseInferenceApiServer.completions), [`responses`](../../inference_engine_interface.md#BaseInferenceApiServer.responses), [`AuthEndpointsMixin`](auth_endpoints.md#AuthEndpointsMixin), [`_infer_sequence_length_from_engine`](../../inference_engine_interface.md#BaseInferenceApiServer._infer_sequence_length_from_engine), [`BaseInferenceApiServer`](../../inference_engine_interface.md#BaseInferenceApiServer), [`_count_tokens`](../../inference_engine_interface.md#BaseInferenceApiServer._count_tokens), [`execute_tool`](../../inference_engine_interface.md#BaseInferenceApiServer.execute_tool), [`get_metrics`](../../inference_engine_interface.md#BaseInferenceApiServer.get_metrics), [`get_model`](../../inference_engine_interface.md#BaseInferenceApiServer.get_model), [`health_check`](../../inference_engine_interface.md#BaseInferenceApiServer.health_check), [`list_models`](../../inference_engine_interface.md#BaseInferenceApiServer.list_models), [`list_tools`](../../inference_engine_interface.md#BaseInferenceApiServer.list_tools), [`on_shutdown`](../../inference_engine_interface.md#BaseInferenceApiServer.on_shutdown), [`on_startup`](../../inference_engine_interface.md#BaseInferenceApiServer.on_startup)

## Module values
- `RefineChatRequestFn` — [`L145`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L145)
- `RefineSamplingParamsFn` — [`L134`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L134)
- `_STREAM_DATA` — [`L129`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L129)
- `_STREAM_END` — [`L131`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L131)
- `_STREAM_ERROR` — [`L130`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L130)
- `__all__` — [`L120`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L120)
- `logger` — [`L118`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/api_server.py#L118)

