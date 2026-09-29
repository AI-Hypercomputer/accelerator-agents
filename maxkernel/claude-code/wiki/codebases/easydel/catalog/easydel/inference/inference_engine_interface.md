---
title: 'Module: easydel/inference/inference_engine_interface.py'
type: catalog
provenance: extracted
module: easydel/inference/inference_engine_interface.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.inference_engine_interface`/
symbols:
  BaseInferenceApiServer.app: BaseInferenceApiServer#app.
  BaseInferenceApiServer._endpoints: BaseInferenceApiServer#_endpoints().
  BaseInferenceApiServer.metrics: BaseInferenceApiServer#metrics.
  BaseInferenceApiServer._create_sampling_params_from_responses: BaseInferenceApiServer#_create_sampling_params_from_responses().
  BaseInferenceApiServer._lifespan: BaseInferenceApiServer#_lifespan().
  BaseInferenceApiServer._add_function_calling_endpoints: BaseInferenceApiServer#_add_function_calling_endpoints().
  EndpointConfig: EndpointConfig#
  EndpointConfig.path: EndpointConfig#path.
  EndpointConfig.handler: EndpointConfig#handler.
  EndpointConfig.methods: EndpointConfig#methods.
  EndpointConfig.summary: EndpointConfig#summary.
  EndpointConfig.tags: EndpointConfig#tags.
  BaseInferenceApiServer._register_endpoints: BaseInferenceApiServer#_register_endpoints().
  BaseInferenceApiServer._responses_payload_to_messages: BaseInferenceApiServer#_responses_payload_to_messages().
  BaseInferenceApiServer.track_metrics: BaseInferenceApiServer#track_metrics().
  BaseInferenceApiServer._response_store_put_response: BaseInferenceApiServer#_response_store_put_response().
  BaseInferenceApiServer._response_store_put_conversation: BaseInferenceApiServer#_response_store_put_conversation().
  create_error_response: create_error_response().
  BaseInferenceApiServer.custom_openapi: BaseInferenceApiServer#custom_openapi().
  BaseInferenceApiServer._acquire_generation_slot: BaseInferenceApiServer#_acquire_generation_slot().
  BaseInferenceApiServer._extract_responses_tools: BaseInferenceApiServer#_extract_responses_tools().
  BaseInferenceApiServer._parse_responses_max_tokens: BaseInferenceApiServer#_parse_responses_max_tokens().
  BaseInferenceApiServer._normalize_conversation_id: BaseInferenceApiServer#_normalize_conversation_id().
  BaseInferenceApiServer._responses_reasoning_summary_requested: BaseInferenceApiServer#_responses_reasoning_summary_requested().
  BaseInferenceApiServer._response_store_get_response: BaseInferenceApiServer#_response_store_get_response().
  BaseInferenceApiServer._response_store_get_conversation: BaseInferenceApiServer#_response_store_get_conversation().
  BaseInferenceApiServer.extract_tools: BaseInferenceApiServer#extract_tools().
  BaseInferenceApiServer._mark_stream_failure: BaseInferenceApiServer#_mark_stream_failure().
  logger: logger.
  BaseInferenceApiServer._create_sampling_params: BaseInferenceApiServer#_create_sampling_params().
  BaseInferenceApiServer._responses_assistant_message_from_output_items: BaseInferenceApiServer#_responses_assistant_message_from_output_items().
  BaseInferenceApiServer._normalize_chat_message: BaseInferenceApiServer#_normalize_chat_message().
  BaseInferenceApiServer._graceful_shutdown: BaseInferenceApiServer#_graceful_shutdown().
  BaseInferenceApiServer._build_responses_object: BaseInferenceApiServer#_build_responses_object().
  BaseInferenceApiServer.responses: BaseInferenceApiServer#responses().
  BaseInferenceApiServer.chat_completions: BaseInferenceApiServer#chat_completions().
  BaseInferenceApiServer.completions: BaseInferenceApiServer#completions().
  BaseInferenceApiServer.run: BaseInferenceApiServer#run().
  BaseInferenceApiServer._enable_response_store: BaseInferenceApiServer#_enable_response_store.
  BaseInferenceApiServer._active_requests: BaseInferenceApiServer#_active_requests.
  BaseInferenceApiServer._generation_slots: BaseInferenceApiServer#_generation_slots.
  BaseInferenceApiServer._response_store_client: BaseInferenceApiServer#_response_store_client.
  BaseInferenceApiServer._start_stream_task: BaseInferenceApiServer#_start_stream_task().
  BaseInferenceApiServer.add_request_id: BaseInferenceApiServer#add_request_id().
  BaseInferenceApiServer.thread_pool: BaseInferenceApiServer#thread_pool.
  ServerStatus: ServerStatus#
  ServerMetrics.total_tokens_generated: ServerMetrics#total_tokens_generated.
  BaseInferenceApiServer.status: BaseInferenceApiServer#status.
  BaseInferenceApiServer.__init__: BaseInferenceApiServer#__init__().
  BaseInferenceApiServer._configure_openapi_security: BaseInferenceApiServer#_configure_openapi_security().
  BaseInferenceApiServer._build_responses_output_items: BaseInferenceApiServer#_build_responses_output_items().
  BaseInferenceApiServer._infer_sequence_length_from_engine: BaseInferenceApiServer#_infer_sequence_length_from_engine().
  ServerMetrics.average_tokens_per_second: ServerMetrics#average_tokens_per_second.
  ServerMetrics.uptime_seconds: ServerMetrics#uptime_seconds.
  ServerMetrics.start_time: ServerMetrics#start_time.
  BaseInferenceApiServer._flatten_messages_to_text: BaseInferenceApiServer#_flatten_messages_to_text().
  BaseInferenceApiServer._build_responses_reasoning_item: BaseInferenceApiServer#_build_responses_reasoning_item().
  BaseInferenceApiServer._build_responses_function_call_items: BaseInferenceApiServer#_build_responses_function_call_items().
  BaseInferenceApiServer._build_responses_message_item: BaseInferenceApiServer#_build_responses_message_item().
  BaseInferenceApiServer._coerce_stream_delta_message: BaseInferenceApiServer#_coerce_stream_delta_message().
  BaseInferenceApiServer._count_tokens_async: BaseInferenceApiServer#_count_tokens_async().
  InferenceEngineAdapter.generate: InferenceEngineAdapter#generate().
  BaseInferenceApiServer: BaseInferenceApiServer#
  BaseInferenceApiServer._conversation_from_messages: BaseInferenceApiServer#_conversation_from_messages().
  InferenceEngineAdapter: InferenceEngineAdapter#
  ServerMetrics.successful_requests: ServerMetrics#successful_requests.
  ServerMetrics.failed_requests: ServerMetrics#failed_requests.
  BaseInferenceApiServer._response_store_lock: BaseInferenceApiServer#_response_store_lock.
  BaseInferenceApiServer.on_startup: BaseInferenceApiServer#on_startup().
  BaseInferenceApiServer.on_shutdown: BaseInferenceApiServer#on_shutdown().
  BaseInferenceApiServer._setup_cors: BaseInferenceApiServer#_setup_cors().
  BaseInferenceApiServer._compute_delta_text: BaseInferenceApiServer#_compute_delta_text().
  BaseInferenceApiServer._jsonify_tool_calls: BaseInferenceApiServer#_jsonify_tool_calls().
  BaseInferenceApiServer.health_check: BaseInferenceApiServer#health_check().
  BaseInferenceApiServer.get_metrics: BaseInferenceApiServer#get_metrics().
  BaseInferenceApiServer.list_models: BaseInferenceApiServer#list_models().
  BaseInferenceApiServer.get_model: BaseInferenceApiServer#get_model().
  BaseInferenceApiServer.list_tools: BaseInferenceApiServer#list_tools().
  BaseInferenceApiServer.execute_tool: BaseInferenceApiServer#execute_tool().
  BaseInferenceApiServer._count_tokens: BaseInferenceApiServer#_count_tokens().
  BaseInferenceApiServer.enable_function_calling: BaseInferenceApiServer#enable_function_calling.
  BaseInferenceApiServer._max_generation_slots: BaseInferenceApiServer#_max_generation_slots.
  BaseInferenceApiServer._should_emit_responses_message_item: BaseInferenceApiServer#_should_emit_responses_message_item().
  BaseInferenceApiServer.fire: BaseInferenceApiServer#fire.
  InferenceEngineAdapter.count_tokens: InferenceEngineAdapter#count_tokens().
  InferenceEngineAdapter.get_model_info: InferenceEngineAdapter#get_model_info().
  InferenceEngineAdapter.model_name: InferenceEngineAdapter#model_name().
  InferenceEngineAdapter.processor: InferenceEngineAdapter#processor().
  ServerStatus.READY: ServerStatus#READY.
  ServerStatus.SHUTTING_DOWN: ServerStatus#SHUTTING_DOWN.
  ServerMetrics.total_requests: ServerMetrics#total_requests.
  BaseInferenceApiServer._request_lock: BaseInferenceApiServer#_request_lock.
  BaseInferenceApiServer._overload_message: BaseInferenceApiServer#_overload_message.
  BaseInferenceApiServer._stored_responses: BaseInferenceApiServer#_stored_responses.
  BaseInferenceApiServer._stored_conversations: BaseInferenceApiServer#_stored_conversations.
  BaseInferenceApiServer._lru_set: BaseInferenceApiServer#_lru_set().
  BaseInferenceApiServer._sse_event: BaseInferenceApiServer#_sse_event().
  BaseInferenceApiServer._default_store_responses: BaseInferenceApiServer#_default_store_responses.
  TIMEOUT_KEEP_ALIVE: TIMEOUT_KEEP_ALIVE.
  ServerStatus.STARTING: ServerStatus#STARTING.
  ServerMetrics: ServerMetrics#
  EndpointConfig.response_model: EndpointConfig#response_model.
  ErrorResponse: ErrorResponse#
  ErrorResponse.error: ErrorResponse#error.
  ErrorResponse.request_id: ErrorResponse#request_id.
  BaseInferenceApiServer._max_stored_responses: BaseInferenceApiServer#_max_stored_responses.
  BaseInferenceApiServer._max_stored_conversations: BaseInferenceApiServer#_max_stored_conversations.
  BaseInferenceApiServer._setup_middleware: BaseInferenceApiServer#_setup_middleware().
  BaseInferenceApiServer._producer: BaseInferenceApiServer#_producer().
  ServerStatus.BUSY: ServerStatus#BUSY.
  ServerStatus.ERROR: ServerStatus#ERROR.
  ErrorResponse.timestamp: ErrorResponse#timestamp.
  BaseInferenceApiServer.max_request_size: BaseInferenceApiServer#max_request_size.
  BaseInferenceApiServer.request_timeout: BaseInferenceApiServer#request_timeout.
  BaseInferenceApiServer.default_function_format: BaseInferenceApiServer#default_function_format.
  BaseInferenceApiServer._determine_finish_reason: BaseInferenceApiServer#_determine_finish_reason().
---
# Module: [`easydel/inference/inference_engine_interface.py`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py)

## Classes
### `BaseInferenceApiServer`  ·  implements/extends ABC
- def: [`easydel/inference/inference_engine_interface.py:191`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L191)
- doc: Abstract base class for inference API servers.
- signature: `class BaseInferenceApiServer(ABC):`
- members:
  - `__init__(self, max_workers: int | None = None, enable_cors: bool = True, cors_origins: list[str] | None = None, max_request_size: int = 10 * 1024 * 1024, request_timeout: float = 300, enable_function_calling: bool = True, default_function_format: FunctionCallFormat = FunctionCallFormat.OPENAI, server_name: str = "EasyDeL Inference API Server", server_description: str = "High-performance inference server with OpenAI API compatibility", server_version: str = "2.0.0", enable_auth_ui: bool = True, max_concurrent_generations: int | None = None, overload_message: str = "Server is busy, please try again later", enable_response_store: bool = True, default_store_responses: bool = True, max_stored_responses: int = 10000, max_stored_conversations: int = 1000, response_store_client: tp.Any | None = None)` — [`L199`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L199) — Initialize the base inference API server.
  - `_acquire_generation_slot(self, *, endpoint: str | None = None, request_id: str | None = None, model: str | None = None, raw_request: Request | None = None, stream: bool | None = None)` — [`L703`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L703) — Acquire a generation slot or raise HTTP 503 when the server is saturated.
  - `_add_function_calling_endpoints(self)` — [`L574`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L574) — Add function calling specific endpoints.
  - `_coerce_stream_delta_message(cls, delta_message: tp.Any, *, fallback_text: str = "", default_role: str | None = None)` — [`L1181`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1181) — Normalize parser/engine streaming deltas into a safe DeltaMessage.
  - `_compute_delta_text(current_text: str, previous_text: str, fallback_delta: str)` — [`L693`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L693) — Compute delta text by comparing accumulated text.
  - `_configure_openapi_security(self)` — [`L338`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L338) — Configure OpenAPI security schemes for API key authentication.
  - `_conversation_from_messages(messages: list[ChatMessage], assistant_turn: str | ChatMessage)` — [`L825`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L825) — Create conversation items (excluding ``instructions``) for storage.
  - `_count_tokens(self, content: str, model_name: str | None = None)` — [`L1418`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1418) — Count tokens for the given content.
  - `_count_tokens_async(self, content: str, model_name: str | None = None)` — [`L1403`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1403) — Count tokens asynchronously.
  - `_create_sampling_params(self, request: ChatCompletionRequest | CompletionRequest)` — [`L1375`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1375) — Create sampling parameters from request.
  - `_create_sampling_params_from_responses(request: ResponsesRequest, max_tokens: int | None)` — [`L1067`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1067) — Translate a Responses API payload into SamplingParams.
  - `_determine_finish_reason(self, tokens_generated: int, max_tokens: float, text: str)` — [`L1387`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1387) — Determine the finish reason for a generation.
  - `_endpoints(self)` — [`L450`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L450) — Define all API endpoints.
  - `_extract_responses_tools(request: ResponsesRequest)` — [`L1010`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1010) — Return (raw_tools, tools_for_chat_template) from a Responses payload.
  - `_flatten_messages_to_text(messages: list[ChatMessage])` — [`L989`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L989) — Collapse content arrays into plain text for tool parsing and templating.
  - `_graceful_shutdown(self)` — [`L614`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L614) — Perform graceful shutdown.
  - `_infer_sequence_length_from_engine(self, engine: tp.Any | None = None)` — [`L1034`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1034) — Infer maximum sequence length from the engine or fall back to 128 tokens.
  - `_lifespan(self, app: FastAPI)` — [`L309`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L309) — Manage server lifecycle.
  - `_mark_stream_failure(self)` — [`L685`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L685) — Adjust metrics when a streaming response fails after headers are sent.
  - `_normalize_chat_message(message: ChatMessage)` — [`L872`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L872) — Canonicalize multimodal/text content parts in a typed chat message.
  - `_normalize_conversation_id(value: tp.Any)` — [`L799`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L799) — Extract a conversation ID from request payload.
  - `_parse_responses_max_tokens(self, request: ResponsesRequest, engine: tp.Any | None)` — [`L1044`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1044) — Return (requested_tokens_for_auth, max_tokens_for_sampling).
  - `_register_endpoints(self)` — [`L602`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L602) — Register all API endpoints.
  - `_responses_payload_to_messages(cls, request: ResponsesRequest, *, include_instructions: bool = False)` — [`L904`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L904) — Convert OpenAI Responses API payload into OpenAI-style chat messages.
  - `_responses_reasoning_summary_requested(request: ResponsesRequest)` — [`L839`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L839) — Return True when reasoning summaries should be emitted in output items.
  - `_setup_cors(self, origins: list[str] | None)` — [`L388`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L388) — Setup CORS middleware.
  - `_setup_middleware(self)` — [`L405`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L405) — Setup request middleware.
  - `_should_emit_responses_message_item(output_text: str, tool_calls: list[tp.Any] | None = None)` — [`L1121`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1121) — Return whether a Responses output should include a message item.
  - `_start_stream_task(self, stream_fn: tp.Callable[[], Iterator[tp.Any]])` — [`L769`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L769) — Run blocking ``stream_fn`` in a worker thread and push results to an asyncio queue.
  - `add_request_id(request: Request, call_next)` — [`L415`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L415) — Add unique request ID to each request.
  - `chat_completions(self, request: ChatCompletionRequest, raw_request: Request)` — [`L1258`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1258) — Handle chat completion requests.
  - `completions(self, request: CompletionRequest, raw_request: Request)` — [`L1276`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1276) — Handle completion requests.
  - `custom_openapi()` — [`L346`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L346) — Generate a custom OpenAPI schema with Bearer auth security scheme.
  - `execute_tool(self, request: Request)` — [`L1360`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1360) — Execute a tool/function call.
  - `extract_tools(self, request: ChatCompletionRequest)` — [`L629`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L629) — Extract tool/function definitions from a chat completion request.
  - `get_metrics(self, raw_request: Request)` — [`L1307`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1307) — Get server performance metrics.
  - `get_model(self, model_id: str, raw_request: Request)` — [`L1333`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1333) — Get detailed information about a specific model.
  - `health_check(self, raw_request: Request)` — [`L1294`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1294) — Perform comprehensive health check.
  - `list_models(self, raw_request: Request)` — [`L1320`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1320) — List available models.
  - `list_tools(self, raw_request: Request)` — [`L1347`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1347) — List available tools/functions.
  - `on_shutdown(self)` — [`L329`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L329) — Hook for server shutdown.
  - `on_startup(self)` — [`L320`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L320) — Hook for server startup.
  - `responses(self, request: ResponsesRequest, raw_request: Request)` — [`L1250`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1250) — Handle OpenAI Responses API requests (default: not implemented).
  - `run(self, host: str = "0.0.0.0", port: int = 11556, workers: int = 1, log_level: str = "info", ssl_keyfile: str | None = None, ssl_certfile: str | None = None, reload: bool = False)` — [`L1431`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1431) — Start the server with enhanced configuration.
  - `track_metrics(request: Request, call_next)` — [`L432`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L432) — Track request metrics.
  - `app` — [`L284`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L284)
  - `default_function_format` — [`L251`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L251)
  - `enable_function_calling` — [`L250`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L250)
  - `fire` — [`L1478`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1478)
  - `max_request_size` — [`L244`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L244)
  - `metrics` — [`L247`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L247)
  - `request_timeout` — [`L245`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L245)
  - `status` — [`L246`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L246)
  - `thread_pool` — [`L243`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L243)
- protocol/private: `_active_requests`[`L248`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L248), `_build_responses_function_call_items`[`L1113`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1113), `_build_responses_message_item`[`L1117`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1117), `_build_responses_object`[`L1151`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1151), `_build_responses_output_items`[`L1129`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1129), `_build_responses_reasoning_item`[`L1109`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1109), `_default_store_responses`[`L268`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L268), `_enable_response_store`[`L267`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L267), `_generation_slots`[`L261`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L261), `_jsonify_tool_calls`[`L1177`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1177), `_lru_set`[`L815`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L815), `_max_generation_slots`[`L260`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L260), `_max_stored_conversations`[`L270`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L270), `_max_stored_responses`[`L269`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L269), `_overload_message`[`L252`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L252), `_producer`[`L778`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L778), `_request_lock`[`L249`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L249), `_response_store_client`[`L274`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L274), `_response_store_get_conversation`[`L1225`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1225), `_response_store_get_response`[`L1204`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1204), `_response_store_lock`[`L273`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L273), `_response_store_put_conversation`[`L1235`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1235), `_response_store_put_response`[`L1214`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1214), `_responses_assistant_message_from_output_items`[`L1145`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1145), `_sse_event`[`L1196`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1196), `_stored_conversations`[`L272`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L272), `_stored_responses`[`L271`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L271)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](openai_api_modules.md#DeltaMessage), [`responses`](esurge/server/api_server.md#eSurgeApiServer.responses), [`ChatCompletionRequest`](openai_api_modules.md#ChatCompletionRequest), [`SamplingParams`](sampling_params.md#SamplingParams), [`ChatMessage`](openai_api_modules.md#ChatMessage), [`_endpoints`](esurge/server/api_server.md#eSurgeApiServer._endpoints), [`chat_completions`](esurge/server/api_server.md#eSurgeApiServer.chat_completions), [`build_responses_object`](stream_protocol.md#build_responses_object), [`get_metrics`](esurge/server/api_server.md#eSurgeApiServer.get_metrics), [`completions`](esurge/server/api_server.md#eSurgeApiServer.completions), [`tools`](openai_api_modules.md#ChatCompletionRequest.tools), [`_create_sampling_params`](esurge/server/api_server.md#eSurgeApiServer._create_sampling_params), [`health_check`](esurge/server/api_server.md#eSurgeApiServer.health_check), [`build_responses_function_call_items`](stream_protocol.md#build_responses_function_call_items), [`ResponsesOutputItem`](typed_models.md#ResponsesOutputItem), [`list_models`](esurge/server/api_server.md#eSurgeApiServer.list_models), [`EndpointConfig`](inference_engine_interface.md#EndpointConfig), [`handler`](inference_engine_interface.md#EndpointConfig.handler), [`methods`](inference_engine_interface.md#EndpointConfig.methods), [`path`](inference_engine_interface.md#EndpointConfig.path), [`summary`](inference_engine_interface.md#EndpointConfig.summary), [`tags`](inference_engine_interface.md#EndpointConfig.tags), [`ToolDefinition`](openai_api_modules.md#ToolDefinition), [`content`](openai_api_modules.md#ChatMessage.content), [`CompletionRequest`](openai_api_modules.md#CompletionRequest), [`get_model`](esurge/server/api_server.md#eSurgeApiServer.get_model), [`role`](openai_api_modules.md#ChatMessage.role), [`create_error_response`](inference_engine_interface.md#create_error_response), [`build_responses_message_item`](stream_protocol.md#build_responses_message_item), [`build_responses_output_items`](stream_protocol.md#build_responses_output_items), [`coerce_stream_delta_message`](stream_protocol.md#coerce_stream_delta_message), [`ResponseFunctionCallItem`](typed_models.md#ResponseFunctionCallItem), [`build_responses_reasoning_item`](stream_protocol.md#build_responses_reasoning_item), [`ResponseMessageItem`](typed_models.md#ResponseMessageItem), [`ResponsesRequest`](openai_api_modules.md#ResponsesRequest), [`ResponsesResponse`](typed_models.md#ResponsesResponse), [`ChatCompletionResponse`](openai_api_modules.md#ChatCompletionResponse), [`FunctionDefinition`](openai_api_modules.md#FunctionDefinition), [`ResponseReasoningItem`](typed_models.md#ResponseReasoningItem), [`_infer_sequence_length_from_engine`](esurge/server/api_server.md#eSurgeApiServer._infer_sequence_length_from_engine)  (+55 more)
- used by: [`responses`](esurge/server/api_server.md#eSurgeApiServer.responses), [`generate_stream`](esurge/server/api_server.md#eSurgeApiServer.generate_stream), [`_build_chat_completion_response`](esurge/server/api_server.md#eSurgeApiServer._build_chat_completion_response), [`_handle_completion_response`](esurge/server/api_server.md#eSurgeApiServer._handle_completion_response), [`_endpoints`](esurge/server/api_server.md#eSurgeApiServer._endpoints), [`get_metrics`](esurge/server/api_server.md#eSurgeApiServer.get_metrics), [`_handle_chat_completion`](esurge/server/api_server.md#eSurgeApiServer._handle_chat_completion), [`health_check`](esurge/server/api_server.md#eSurgeApiServer.health_check), [`_prepare_chat_input`](esurge/server/api_server.md#eSurgeApiServer._prepare_chat_input), [`list_models`](esurge/server/api_server.md#eSurgeApiServer.list_models), [`get_model`](esurge/server/api_server.md#eSurgeApiServer.get_model), [`auth_manager`](esurge/server/api_server.md#eSurgeApiServer.auth_manager), [`_handle_chat_streaming`](esurge/server/api_server.md#eSurgeApiServer._handle_chat_streaming), [`_run_chat`](esurge/server/api_server.md#eSurgeApiServer._run_chat), [`_create_tools_response`](esurge/server/api_server.md#eSurgeApiServer._create_tools_response), [`_prepare_chat_input_async`](esurge/server/api_server.md#eSurgeApiServer._prepare_chat_input_async), [`eSurgeApiServer`](esurge/server/api_server.md#eSurgeApiServer)

### `EndpointConfig`  ·  implements/extends BaseModel
- def: [`easydel/inference/inference_engine_interface.py:138`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L138)
- doc: Configuration for a FastAPI endpoint.
- signature: `class EndpointConfig(BaseModel):`
- members:
  - `handler` — [`L153`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L153)
  - `methods` — [`L154`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L154)
  - `path` — [`L152`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L152)
  - `response_model` — [`L157`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L157)
  - `summary` — [`L155`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L155)
  - `tags` — [`L156`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L156)
- used by: [`_endpoints`](esurge/server/api_server.md#eSurgeApiServer._endpoints), [`_endpoints`](inference_engine_interface.md#BaseInferenceApiServer._endpoints), [`_add_function_calling_endpoints`](inference_engine_interface.md#BaseInferenceApiServer._add_function_calling_endpoints), [`_register_endpoints`](inference_engine_interface.md#BaseInferenceApiServer._register_endpoints)

### `ErrorResponse`  ·  implements/extends BaseModel
- def: [`easydel/inference/inference_engine_interface.py:160`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L160)
- doc: Standard error response model.
- signature: `class ErrorResponse(BaseModel):`
- members:
  - `error` — [`L171`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L171)
  - `request_id` — [`L172`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L172)
  - `timestamp` — [`L173`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L173)
- used by: [`create_error_response`](inference_engine_interface.md#create_error_response)

### `InferenceEngineAdapter`  ·  implements/extends ABC
- def: [`easydel/inference/inference_engine_interface.py:1481`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1481)
- doc: Abstract adapter interface for different inference engines.
- signature: `class InferenceEngineAdapter(ABC):`
- members:
  - `count_tokens(self, content: str)` — [`L1510`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1510) — Count tokens in the given content.
  - `generate(self, prompts: str | list[str], sampling_params: SamplingParams, stream: bool = False)` — [`L1490`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1490) — Generate text from prompts.
  - `get_model_info(self)` — [`L1523`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1523) — Get information about the loaded model.
  - `model_name(self)` — [`L1534`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1534) — Get the name of the model.
  - `processor(self)` — [`L1540`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L1540) — Get the processor/tokenizer for the model.
- uses (calls/refs, reference-scoped): [`SamplingParams`](sampling_params.md#SamplingParams), [`get_model_info`](esurge/server/api_server.md#eSurgeAdapter.get_model_info), [`generate`](esurge/server/api_server.md#eSurgeAdapter.generate), [`count_tokens`](esurge/server/api_server.md#eSurgeAdapter.count_tokens), [`processor`](esurge/server/api_server.md#eSurgeAdapter.processor), [`eSurgeAdapter`](esurge/server/api_server.md#eSurgeAdapter), [`model_name`](esurge/server/api_server.md#eSurgeAdapter.model_name)
- used by: [`eSurgeAdapter`](esurge/server/api_server.md#eSurgeAdapter)

### `ServerMetrics`
- def: [`easydel/inference/inference_engine_interface.py:114`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L114)
- doc: Server performance metrics.
- signature: `class ServerMetrics:`
- members:
  - `average_tokens_per_second` — [`L133`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L133)
  - `failed_requests` — [`L131`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L131)
  - `start_time` — [`L135`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L135)
  - `successful_requests` — [`L130`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L130)
  - `total_requests` — [`L129`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L129)
  - `total_tokens_generated` — [`L132`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L132)
  - `uptime_seconds` — [`L134`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L134)
- used by: [`responses`](esurge/server/api_server.md#eSurgeApiServer.responses), [`generate_stream`](esurge/server/api_server.md#eSurgeApiServer.generate_stream), [`_build_chat_completion_response`](esurge/server/api_server.md#eSurgeApiServer._build_chat_completion_response), [`_handle_completion_response`](esurge/server/api_server.md#eSurgeApiServer._handle_completion_response), [`get_metrics`](esurge/server/api_server.md#eSurgeApiServer.get_metrics), [`metrics`](inference_engine_interface.md#BaseInferenceApiServer.metrics), [`health_check`](esurge/server/api_server.md#eSurgeApiServer.health_check), [`list_models`](esurge/server/api_server.md#eSurgeApiServer.list_models), [`get_model`](esurge/server/api_server.md#eSurgeApiServer.get_model), [`track_metrics`](inference_engine_interface.md#BaseInferenceApiServer.track_metrics), [`_mark_stream_failure`](inference_engine_interface.md#BaseInferenceApiServer._mark_stream_failure)

### `ServerStatus`  ·  implements/extends StrEnum
- def: [`easydel/inference/inference_engine_interface.py:93`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L93)
- doc: Server status enumeration.
- signature: `class ServerStatus(StrEnum):`
- members:
  - `BUSY` — [`L108`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L108)
  - `ERROR` — [`L109`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L109)
  - `READY` — [`L107`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L107)
  - `SHUTTING_DOWN` — [`L110`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L110)
  - `STARTING` — [`L106`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L106)
- used by: [`health_check`](esurge/server/api_server.md#eSurgeApiServer.health_check), [`_lifespan`](inference_engine_interface.md#BaseInferenceApiServer._lifespan), [`status`](esurge/server/api_server.md#eSurgeApiServer.status), [`status`](inference_engine_interface.md#BaseInferenceApiServer.status)

## Functions
- `create_error_response(status_code: HTTPStatus, message: str, request_id: str | None = None)` — [`L176`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L176) — Creates a standardized JSON error response.

## Module values
- `TIMEOUT_KEEP_ALIVE` — [`L89`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L89)
- `logger` — [`L90`](../../../../../../raw/code/EasyDeL/easydel/inference/inference_engine_interface.py#L90)

