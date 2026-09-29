---
title: 'Module: easydel/inference/oai_proxies.py'
type: catalog
provenance: extracted
module: easydel/inference/oai_proxies.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.oai_proxies`/
symbols:
  InferenceApiRouter.build_oai_params_from_chat_request: InferenceApiRouter#build_oai_params_from_chat_request().
  InferenceApiRouter._endpoints: InferenceApiRouter#_endpoints().
  InferenceApiRouter.metrics: InferenceApiRouter#metrics.
  InferenceApiRouter.build_oai_params_from_request: InferenceApiRouter#build_oai_params_from_request().
  InferenceApiRouter.chat_completions: InferenceApiRouter#chat_completions().
  InferenceApiRouter.completions: InferenceApiRouter#completions().
  InferenceApiRouter.health_check: InferenceApiRouter#health_check().
  InferenceApiRouter.get_metrics: InferenceApiRouter#get_metrics().
  InferenceApiRouter._add_function_calling_endpoints: InferenceApiRouter#_add_function_calling_endpoints().
  InferenceApiRouter.responses: InferenceApiRouter#responses().
  InferenceApiRouter._register_endpoints: InferenceApiRouter#_register_endpoints().
  create_error_response: create_error_response().
  InferenceApiRouter._stream_chat_completion: InferenceApiRouter#_stream_chat_completion().
  InferenceApiRouter._stream_completion: InferenceApiRouter#_stream_completion().
  InferenceApiRouter.stream_events: InferenceApiRouter#stream_events().
  InferenceApiRouter.app: InferenceApiRouter#app.
  InferenceApiRouter.get_model: InferenceApiRouter#get_model().
  EndpointConfig.path: EndpointConfig#path.
  EndpointConfig.handler: EndpointConfig#handler.
  EndpointConfig.methods: EndpointConfig#methods.
  EndpointConfig.summary: EndpointConfig#summary.
  EndpointConfig.tags: EndpointConfig#tags.
  ServerMetrics.failed_requests: ServerMetrics#failed_requests.
  EndpointConfig: EndpointConfig#
  InferenceApiRouter.list_models: InferenceApiRouter#list_models().
  InferenceApiRouter.client: InferenceApiRouter#client.
  InferenceApiRouter.status: InferenceApiRouter#status.
  InferenceApiRouter.base_url: InferenceApiRouter#base_url.
  InferenceApiRouter.execute_tool: InferenceApiRouter#execute_tool().
  ServerMetrics.successful_requests: ServerMetrics#successful_requests.
  InferenceApiRouter.process_request_params: InferenceApiRouter#process_request_params().
  InferenceApiRouter.list_tools: InferenceApiRouter#list_tools().
  InferenceApiRouter.run: InferenceApiRouter#run().
  ServerMetrics.total_requests: ServerMetrics#total_requests.
  InferenceApiRouter: InferenceApiRouter#
  ServerStatus: ServerStatus#
  InferenceApiRouter.openai_module: InferenceApiRouter#openai_module.
  InferenceApiRouter.enable_function_calling: InferenceApiRouter#enable_function_calling.
  InferenceApiRouter.readiness: InferenceApiRouter#readiness().
  ServerStatus.READY: ServerStatus#READY.
  ServerMetrics.uptime_seconds: ServerMetrics#uptime_seconds.
  ServerMetrics.start_time: ServerMetrics#start_time.
  InferenceApiRouter._format_metadata_sse: InferenceApiRouter#_format_metadata_sse().
  TIMEOUT_KEEP_ALIVE: TIMEOUT_KEEP_ALIVE.
  ServerStatus.STARTING: ServerStatus#STARTING.
  ServerStatus.ERROR: ServerStatus#ERROR.
  ServerMetrics: ServerMetrics#
  ServerMetrics.total_tokens_generated: ServerMetrics#total_tokens_generated.
  ServerMetrics.average_tokens_per_second: ServerMetrics#average_tokens_per_second.
  EndpointConfig.response_model: EndpointConfig#response_model.
  ErrorResponse: ErrorResponse#
  ErrorResponse.error: ErrorResponse#error.
  ErrorResponse.request_id: ErrorResponse#request_id.
  InferenceApiRouter.liveness: InferenceApiRouter#liveness().
  ServerStatus.BUSY: ServerStatus#BUSY.
  ServerStatus.SHUTTING_DOWN: ServerStatus#SHUTTING_DOWN.
  ErrorResponse.timestamp: ErrorResponse#timestamp.
  InferenceApiRouter.__init__: InferenceApiRouter#__init__().
  InferenceApiRouter.fire: InferenceApiRouter#fire.
---
# Module: [`easydel/inference/oai_proxies.py`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py)

## Classes
### `EndpointConfig`  ·  implements/extends BaseModel
- def: [`easydel/inference/oai_proxies.py:115`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L115)
- doc: Configuration for a FastAPI endpoint.
- signature: `class EndpointConfig(BaseModel):`
- members:
  - `handler` — [`L130`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L130)
  - `methods` — [`L131`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L131)
  - `path` — [`L129`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L129)
  - `response_model` — [`L134`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L134)
  - `summary` — [`L132`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L132)
  - `tags` — [`L133`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L133)
- used by: [`_endpoints`](oai_proxies.md#InferenceApiRouter._endpoints), [`_add_function_calling_endpoints`](oai_proxies.md#InferenceApiRouter._add_function_calling_endpoints), [`_register_endpoints`](oai_proxies.md#InferenceApiRouter._register_endpoints)

### `ErrorResponse`  ·  implements/extends BaseModel
- def: [`easydel/inference/oai_proxies.py:137`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L137)
- doc: Standard error response model.
- signature: `class ErrorResponse(BaseModel):`
- members:
  - `error` — [`L148`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L148)
  - `request_id` — [`L149`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L149)
  - `timestamp` — [`L150`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L150)
- used by: [`create_error_response`](oai_proxies.md#create_error_response)

### `InferenceApiRouter`
- def: [`easydel/inference/oai_proxies.py:175`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L175)
- doc: Enhanced FastAPI server acting as an OpenAI API proxy.
- signature: `class InferenceApiRouter:`
- members:
  - `__init__(self, api_key: str | None = None, base_url: str | None = None, organization: str | None = None, enable_function_calling: bool = True, **kwargs)` — [`L201`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L201) — Initialize the Inference API Router with EasyDeL compatibility.
  - `_add_function_calling_endpoints(self)` — [`L326`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L326) — Add function calling specific endpoints to the router.
  - `_endpoints(self)` — [`L254`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L254) — Define all API endpoints matching EasyDeL API servers.
  - `_format_metadata_sse(metadata: BaseModel | dict[str, tp.Any] | tp.Any)` — [`L467`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L467) — Format metadata as a standard SSE event payload.
  - `_register_endpoints(self)` — [`L358`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L358) — Register all API endpoints with the FastAPI application.
  - `_stream_chat_completion(self, params: dict, metadata: dict | None, request_id: str | None = None)` — [`L599`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L599) — Handle streaming chat completion responses.
  - `_stream_completion(self, params: dict, metadata: dict | None, request_id: str | None = None)` — [`L685`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L685) — Handle streaming completion responses.
  - `build_oai_params_from_chat_request(self, request: ChatCompletionRequest)` — [`L403`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L403) — Build OpenAI parameters from a chat completion request.
  - `build_oai_params_from_request(self, request: CompletionRequest)` — [`L374`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L374) — Build OpenAI parameters from a completion request.
  - `chat_completions(self, request: ChatCompletionRequest)` — [`L546`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L546) — Handle chat completion requests with function calling support.
  - `completions(self, request: CompletionRequest)` — [`L632`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L632) — Handle text completion requests.
  - `execute_tool(self, request: Request)` — [`L970`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L970) — Execute a tool/function call.
  - `get_metrics(self)` — [`L789`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L789) — Get server performance metrics.
  - `get_model(self, model_id: str)` — [`L883`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L883) — Get detailed information about a specific model.
  - `health_check(self)` — [`L718`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L718) — Perform a comprehensive health check.
  - `list_models(self)` — [`L834`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L834) — List available models with metadata.
  - `list_tools(self)` — [`L927`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L927) — List available tools/functions for each model.
  - `liveness(self)` — [`L761`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L761) — Perform a liveness check.
  - `process_request_params(self, openai_params: dict, request: ChatCompletionRequest | CompletionRequest)` — [`L445`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L445) — Process request parameters before sending to OpenAI.
  - `readiness(self)` — [`L772`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L772) — Perform a readiness check.
  - `responses(self, request: ResponsesRequest, raw_request: Request)` — [`L477`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L477) — Handle OpenAI Responses API requests.
  - `run(self, host: str = "0.0.0.0", port: int = 8084, log_level: str = "info", ssl_keyfile: str | None = None, ssl_certfile: str | None = None, workers: int = 1, reload: bool = False)` — [`L1008`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L1008) — Start the server with enhanced configuration.
  - `stream_events()` — [`L506`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L506) — Stream Server-Sent Events from the backend.
  - `app` — [`L240`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L240)
  - `base_url` — [`L237`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L237)
  - `client` — [`L227`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L227)
  - `enable_function_calling` — [`L234`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L234)
  - `fire` — [`L1061`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L1061)
  - `metrics` — [`L236`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L236)
  - `openai_module` — [`L233`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L233) — ---
  - `status` — [`L235`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L235)
- uses (calls/refs, reference-scoped): [`ChatCompletionRequest`](openai_api_modules.md#ChatCompletionRequest), [`tools`](openai_api_modules.md#ChatCompletionRequest.tools), [`create_error_response`](oai_proxies.md#create_error_response), [`CompletionRequest`](openai_api_modules.md#CompletionRequest), [`ResponsesRequest`](openai_api_modules.md#ResponsesRequest), [`tool_choice`](openai_api_modules.md#ChatCompletionRequest.tool_choice), [`handler`](oai_proxies.md#EndpointConfig.handler), [`methods`](oai_proxies.md#EndpointConfig.methods), [`path`](oai_proxies.md#EndpointConfig.path), [`summary`](oai_proxies.md#EndpointConfig.summary), [`tags`](oai_proxies.md#EndpointConfig.tags), [`EndpointConfig`](oai_proxies.md#EndpointConfig), [`failed_requests`](oai_proxies.md#ServerMetrics.failed_requests), [`model`](openai_api_modules.md#ChatCompletionRequest.model), [`model`](openai_api_modules.md#CompletionRequest.model), [`successful_requests`](oai_proxies.md#ServerMetrics.successful_requests), [`messages`](openai_api_modules.md#ChatCompletionRequest.messages), [`functions`](openai_api_modules.md#ChatCompletionRequest.functions), `body`, [`total_requests`](oai_proxies.md#ServerMetrics.total_requests), [`function_call`](openai_api_modules.md#ChatCompletionRequest.function_call), [`ServerStatus`](oai_proxies.md#ServerStatus), [`stream`](openai_api_modules.md#ChatCompletionRequest.stream), [`stream`](openai_api_modules.md#CompletionRequest.stream), [`READY`](oai_proxies.md#ServerStatus.READY), [`frequency_penalty`](openai_api_modules.md#ChatCompletionRequest.frequency_penalty), [`max_tokens`](openai_api_modules.md#ChatCompletionRequest.max_tokens), [`n`](openai_api_modules.md#ChatCompletionRequest.n), [`presence_penalty`](openai_api_modules.md#ChatCompletionRequest.presence_penalty), [`prompt`](openai_api_modules.md#CompletionRequest.prompt), [`start_time`](oai_proxies.md#ServerMetrics.start_time), [`stop`](openai_api_modules.md#ChatCompletionRequest.stop), [`temperature`](openai_api_modules.md#ChatCompletionRequest.temperature), [`top_p`](openai_api_modules.md#ChatCompletionRequest.top_p), [`uptime_seconds`](oai_proxies.md#ServerMetrics.uptime_seconds), [`ServerMetrics`](oai_proxies.md#ServerMetrics), [`ERROR`](oai_proxies.md#ServerStatus.ERROR), [`STARTING`](oai_proxies.md#ServerStatus.STARTING), [`TIMEOUT_KEEP_ALIVE`](oai_proxies.md#TIMEOUT_KEEP_ALIVE), [`average_tokens_per_second`](oai_proxies.md#ServerMetrics.average_tokens_per_second)  (+11 more)
- used by: [`_import_structure`](../__init__.md#_import_structure)

### `ServerMetrics`
- def: [`easydel/inference/oai_proxies.py:91`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L91)
- doc: Server performance metrics.
- signature: `class ServerMetrics:`
- members:
  - `average_tokens_per_second` — [`L110`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L110)
  - `failed_requests` — [`L108`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L108)
  - `start_time` — [`L112`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L112)
  - `successful_requests` — [`L107`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L107)
  - `total_requests` — [`L106`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L106)
  - `total_tokens_generated` — [`L109`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L109)
  - `uptime_seconds` — [`L111`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L111)
- used by: [`metrics`](oai_proxies.md#InferenceApiRouter.metrics), [`chat_completions`](oai_proxies.md#InferenceApiRouter.chat_completions), [`completions`](oai_proxies.md#InferenceApiRouter.completions), [`health_check`](oai_proxies.md#InferenceApiRouter.health_check), [`get_metrics`](oai_proxies.md#InferenceApiRouter.get_metrics), [`responses`](oai_proxies.md#InferenceApiRouter.responses), [`_stream_chat_completion`](oai_proxies.md#InferenceApiRouter._stream_chat_completion), [`_stream_completion`](oai_proxies.md#InferenceApiRouter._stream_completion), [`stream_events`](oai_proxies.md#InferenceApiRouter.stream_events)

### `ServerStatus`  ·  implements/extends StrEnum
- def: [`easydel/inference/oai_proxies.py:70`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L70)
- doc: Server status enumeration.
- signature: `class ServerStatus(StrEnum):`
- members:
  - `BUSY` — [`L85`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L85)
  - `ERROR` — [`L86`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L86)
  - `READY` — [`L84`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L84)
  - `SHUTTING_DOWN` — [`L87`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L87)
  - `STARTING` — [`L83`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L83)
- used by: [`health_check`](oai_proxies.md#InferenceApiRouter.health_check), [`app`](oai_proxies.md#InferenceApiRouter.app), [`status`](oai_proxies.md#InferenceApiRouter.status)

## Functions
- `create_error_response(status_code: HTTPStatus, message: str, request_id: str | None = None)` — [`L153`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L153) — Create a standardized JSON error response.

## Module values
- `TIMEOUT_KEEP_ALIVE` — [`L67`](../../../../../../raw/code/EasyDeL/easydel/inference/oai_proxies.py#L67)

