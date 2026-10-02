---
title: 'Module: easydel/inference/openai_api_modules.py'
type: catalog
provenance: extracted
module: easydel/inference/openai_api_modules.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.openai_api_modules`/
symbols:
  DeltaMessage: DeltaMessage#
  OpenAIBaseModel: OpenAIBaseModel#
  ExtractedToolCallInformation: ExtractedToolCallInformation#
  ChatCompletionRequest: ChatCompletionRequest#
  DeltaToolCall: DeltaToolCall#
  DeltaMessage.content: DeltaMessage#content.
  DeltaFunctionCall: DeltaFunctionCall#
  ExtractedToolCallInformation.tool_calls: ExtractedToolCallInformation#tool_calls.
  ExtractedToolCallInformation.content: ExtractedToolCallInformation#content.
  ExtractedToolCallInformation.tools_called: ExtractedToolCallInformation#tools_called.
  ToolCall: ToolCall#
  DeltaMessage.tool_calls: DeltaMessage#tool_calls.
  DeltaToolCall.index: DeltaToolCall#index.
  DeltaFunctionCall.arguments: DeltaFunctionCall#arguments.
  FunctionCall: FunctionCall#
  FunctionCall.arguments: FunctionCall#arguments.
  ChatMessage: ChatMessage#
  FunctionCall.name: FunctionCall#name.
  DeltaToolCall.id: DeltaToolCall#id.
  DeltaFunctionCall.name: DeltaFunctionCall#name.
  ChatCompletionRequest.tools: ChatCompletionRequest#tools.
  DeltaMessage.reasoning_content: DeltaMessage#reasoning_content.
  ToolDefinition: ToolDefinition#
  ChatMessage.content: ChatMessage#content.
  CompletionRequest: CompletionRequest#
  ChatMessage.role: ChatMessage#role.
  UsageInfo: UsageInfo#
  ToolCall.id: ToolCall#id.
  ChatCompletionStreamResponse: ChatCompletionStreamResponse#
  ChatCompletionRequest.tool_choice: ChatCompletionRequest#tool_choice.
  ResponsesRequest: ResponsesRequest#
  ChatCompletionRequest.model: ChatCompletionRequest#model.
  FunctionDefinition: FunctionDefinition#
  ChatCompletionResponse: ChatCompletionResponse#
  CompletionRequest.model: CompletionRequest#model.
  ToolChoiceOption: ToolChoiceOption#
  ChatCompletionStreamResponseChoice: ChatCompletionStreamResponseChoice#
  ChatCompletionStreamResponse.choices: ChatCompletionStreamResponse#choices.
  UsageInfo.prompt_tokens: UsageInfo#prompt_tokens.
  UsageInfo.completion_tokens: UsageInfo#completion_tokens.
  FunctionDefinition.name: FunctionDefinition#name.
  FunctionDefinition.parameters: FunctionDefinition#parameters.
  ChatCompletionRequest.messages: ChatCompletionRequest#messages.
  ChatCompletionResponseChoice: ChatCompletionResponseChoice#
  ChatCompletionStreamResponseChoice.delta: ChatCompletionStreamResponseChoice#delta.
  ChatCompletionStreamResponse.usage: ChatCompletionStreamResponse#usage.
  CompletionResponse: CompletionResponse#
  UsageInfo.total_tokens: UsageInfo#total_tokens.
  UsageInfo.tokens_per_second: UsageInfo#tokens_per_second.
  UsageInfo.processing_time: UsageInfo#processing_time.
  ChatCompletionRequest.functions: ChatCompletionRequest#functions.
  DeltaMessage.role: DeltaMessage#role.
  ChatCompletionRequest.chat_template_kwargs: ChatCompletionRequest#chat_template_kwargs.
  ConversationReference: ConversationReference#
  CompletionResponseChoice: CompletionResponseChoice#
  Tool: Tool#
  ChatCompletionRequest.function_call: ChatCompletionRequest#function_call.
  ChatCompletionResponseChoice.message: ChatCompletionResponseChoice#message.
  ChatCompletionResponse.choices: ChatCompletionResponse#choices.
  ChatCompletionResponse.usage: ChatCompletionResponse#usage.
  ResponsesRequest.messages: ResponsesRequest#messages.
  CompletionLogprobs: CompletionLogprobs#
  ChatCompletionStreamResponseChoice.index: ChatCompletionStreamResponseChoice#index.
  ChatCompletionStreamResponseChoice.finish_reason: ChatCompletionStreamResponseChoice#finish_reason.
  ChatCompletionStreamResponse.model: ChatCompletionStreamResponse#model.
  ResponsesRequest.temperature: ResponsesRequest#temperature.
  ResponsesRequest.top_p: ResponsesRequest#top_p.
  Function: Function#
  FunctionCallSelection: FunctionCallSelection#
  ToolChoiceFunction: ToolChoiceFunction#
  ResponseReasoningConfig: ResponseReasoningConfig#
  ResponsesRequest.conversation: ResponsesRequest#conversation.
  ResponsesRequest.tools: ResponsesRequest#tools.
  ResponsesRequest.functions: ResponsesRequest#functions.
  ResponsesRequest.tool_choice: ResponsesRequest#tool_choice.
  ResponsesRequest.reasoning: ResponsesRequest#reasoning.
  CompletionResponse.choices: CompletionResponse#choices.
  CompletionResponse.usage: CompletionResponse#usage.
  CompletionStreamResponseChoice: CompletionStreamResponseChoice#
  ChatMessage.tool_calls: ChatMessage#tool_calls.
  ChatCompletionRequest.stream: ChatCompletionRequest#stream.
  CompletionRequest.stream: CompletionRequest#stream.
  FunctionCallFormat: FunctionCallFormat#
  OpenAIBaseModel.__log_extra_fields__: OpenAIBaseModel#__log_extra_fields__().
  CountTokenRequest: CountTokenRequest#
  CountTokenRequest.conversation: CountTokenRequest#conversation.
  CompletionResponseChoice.logprobs: CompletionResponseChoice#logprobs.
  CompletionStreamResponseChoice.logprobs: CompletionStreamResponseChoice#logprobs.
  CompletionStreamResponse: CompletionStreamResponse#
  CompletionStreamResponse.choices: CompletionStreamResponse#choices.
  CompletionStreamResponse.usage: CompletionStreamResponse#usage.
  OpenAIBaseModel.field_names: OpenAIBaseModel#field_names.
  ChatMessage.reasoning_content: ChatMessage#reasoning_content.
  ChatCompletionRequest.max_tokens: ChatCompletionRequest#max_tokens.
  ChatCompletionRequest.presence_penalty: ChatCompletionRequest#presence_penalty.
  ChatCompletionRequest.frequency_penalty: ChatCompletionRequest#frequency_penalty.
  ChatCompletionRequest.temperature: ChatCompletionRequest#temperature.
  ChatCompletionRequest.top_p: ChatCompletionRequest#top_p.
  ChatCompletionRequest.n: ChatCompletionRequest#n.
  ChatCompletionRequest.stop: ChatCompletionRequest#stop.
  ChatCompletionResponseChoice.index: ChatCompletionResponseChoice#index.
  ChatCompletionResponseChoice.finish_reason: ChatCompletionResponseChoice#finish_reason.
  ChatCompletionResponse.model: ChatCompletionResponse#model.
  CompletionRequest.prompt: CompletionRequest#prompt.
  ResponsesRequest.instructions: ResponsesRequest#instructions.
  ResponsesRequest.max_output_tokens: ResponsesRequest#max_output_tokens.
  ResponsesRequest.n: ResponsesRequest#n.
  ResponsesRequest.parallel_tool_calls: ResponsesRequest#parallel_tool_calls.
  ResponsesRequest.metadata: ResponsesRequest#metadata.
  ConversationReference.id: ConversationReference#id.
  ConversationReference.conversation_id: ConversationReference#conversation_id.
  ConversationReference.conversation: ConversationReference#conversation.
  ResponseReasoningConfig.summary: ResponseReasoningConfig#summary.
  ChatCompletionRequest.repetition_penalty: ChatCompletionRequest#repetition_penalty.
  CompletionRequest.max_tokens: CompletionRequest#max_tokens.
  CompletionRequest.presence_penalty: CompletionRequest#presence_penalty.
  CompletionRequest.frequency_penalty: CompletionRequest#frequency_penalty.
  CompletionRequest.repetition_penalty: CompletionRequest#repetition_penalty.
  CompletionRequest.temperature: CompletionRequest#temperature.
  CompletionRequest.top_p: CompletionRequest#top_p.
  CompletionRequest.n: CompletionRequest#n.
  CompletionRequest.stop: CompletionRequest#stop.
  ResponsesRequest.model: ResponsesRequest#model.
  ResponsesRequest.input: ResponsesRequest#input.
  ResponsesRequest.store: ResponsesRequest#store.
  ResponsesRequest.previous_response_id: ResponsesRequest#previous_response_id.
  ResponsesRequest.max_tokens: ResponsesRequest#max_tokens.
  ResponsesRequest.max_completion_tokens: ResponsesRequest#max_completion_tokens.
  ResponsesRequest.top_k: ResponsesRequest#top_k.
  ResponsesRequest.min_p: ResponsesRequest#min_p.
  ResponsesRequest.presence_penalty: ResponsesRequest#presence_penalty.
  ResponsesRequest.frequency_penalty: ResponsesRequest#frequency_penalty.
  ResponsesRequest.repetition_penalty: ResponsesRequest#repetition_penalty.
  ResponsesRequest.stop: ResponsesRequest#stop.
  ResponsesRequest.stream: ResponsesRequest#stream.
  ResponsesRequest.truncation: ResponsesRequest#truncation.
  ResponsesRequest.include: ResponsesRequest#include.
  CompletionResponseChoice.text: CompletionResponseChoice#text.
  CompletionResponseChoice.index: CompletionResponseChoice#index.
  CompletionResponseChoice.finish_reason: CompletionResponseChoice#finish_reason.
  CompletionResponse.model: CompletionResponse#model.
  FunctionCallFormat.OPENAI: FunctionCallFormat#OPENAI.
  OpenAIBaseModel.model_config: OpenAIBaseModel#model_config.
  ChatMessage.name: ChatMessage#name.
  ChatMessage.function_call: ChatMessage#function_call.
  DeltaMessage.function_call: DeltaMessage#function_call.
  Function.name: Function#name.
  Function.description: Function#description.
  Function.parameters: Function#parameters.
  FunctionDefinition.description: FunctionDefinition#description.
  FunctionDefinition.required: FunctionDefinition#required.
  FunctionCallSelection.name: FunctionCallSelection#name.
  ToolChoiceFunction.name: ToolChoiceFunction#name.
  ChatCompletionRequest.top_k: ChatCompletionRequest#top_k.
  ChatCompletionRequest.min_p: ChatCompletionRequest#min_p.
  ChatCompletionRequest.suppress_tokens: ChatCompletionRequest#suppress_tokens.
  ChatCompletionRequest.logit_bias: ChatCompletionRequest#logit_bias.
  ChatCompletionRequest.user: ChatCompletionRequest#user.
  ChatCompletionResponse.id: ChatCompletionResponse#id.
  ChatCompletionResponse.created: ChatCompletionResponse#created.
  ChatCompletionStreamResponse.id: ChatCompletionStreamResponse#id.
  ChatCompletionStreamResponse.created: ChatCompletionStreamResponse#created.
  CountTokenRequest.model: CountTokenRequest#model.
  CompletionRequest.top_k: CompletionRequest#top_k.
  CompletionRequest.min_p: CompletionRequest#min_p.
  CompletionRequest.suppress_tokens: CompletionRequest#suppress_tokens.
  CompletionRequest.logit_bias: CompletionRequest#logit_bias.
  CompletionRequest.user: CompletionRequest#user.
  CompletionLogprobs.tokens: CompletionLogprobs#tokens.
  CompletionLogprobs.token_logprobs: CompletionLogprobs#token_logprobs.
  CompletionLogprobs.top_logprobs: CompletionLogprobs#top_logprobs.
  CompletionLogprobs.text_offset: CompletionLogprobs#text_offset.
  CompletionResponse.id: CompletionResponse#id.
  CompletionResponse.created: CompletionResponse#created.
  CompletionStreamResponseChoice.index: CompletionStreamResponseChoice#index.
  CompletionStreamResponseChoice.text: CompletionStreamResponseChoice#text.
  CompletionStreamResponseChoice.finish_reason: CompletionStreamResponseChoice#finish_reason.
  CompletionStreamResponse.id: CompletionStreamResponse#id.
  CompletionStreamResponse.created: CompletionStreamResponse#created.
  CompletionStreamResponse.model: CompletionStreamResponse#model.
  FunctionCallFormat.JSON_SCHEMA: FunctionCallFormat#JSON_SCHEMA.
  FunctionCallFormat.HERMES: FunctionCallFormat#HERMES.
  FunctionCallFormat.GORILLA: FunctionCallFormat#GORILLA.
  FunctionCallFormat.QWEN: FunctionCallFormat#QWEN.
  FunctionCallFormat.NOUS: FunctionCallFormat#NOUS.
---
# Module: [`easydel/inference/openai_api_modules.py`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py)

## Classes
### `ChatCompletionRequest`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:276`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L276) — documented in [easydel-inference-openai_api_modules](../../../concepts/easydel-inference-openai_api_modules.md)
- doc: Represents a request to the chat completion endpoint.
- signature: `class ChatCompletionRequest(OpenAIBaseModel):`
- members:
  - `chat_template_kwargs` — [`L326`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L326)
  - `frequency_penalty` — [`L310`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L310)
  - `function_call` — [`L318`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L318)
  - `functions` — [`L317`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L317)
  - `logit_bias` — [`L324`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L324)
  - `max_tokens` — [`L308`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L308)
  - `messages` — [`L307`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L307)
  - `min_p` — [`L315`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L315)
  - `model` — [`L306`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L306)
  - `n` — [`L321`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L321)
  - `presence_penalty` — [`L309`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L309)
  - `repetition_penalty` — [`L311`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L311)
  - `stop` — [`L323`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L323)
  - `stream` — [`L322`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L322)
  - `suppress_tokens` — [`L316`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L316)
  - `temperature` — [`L312`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L312)
  - `tool_choice` — [`L320`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L320)
  - `tools` — [`L319`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L319)
  - `top_k` — [`L314`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L314)
  - `top_p` — [`L313`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L313)
  - `user` — [`L325`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L325)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`ChatMessage`](openai_api_modules.md#ChatMessage), [`ToolDefinition`](openai_api_modules.md#ToolDefinition), [`FunctionDefinition`](openai_api_modules.md#FunctionDefinition), [`ToolChoiceOption`](openai_api_modules.md#ToolChoiceOption), [`FunctionCallSelection`](openai_api_modules.md#FunctionCallSelection)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`generate_stream`](esurge/server/api_server.md#eSurgeApiServer.generate_stream), [`extract_tool_calls_streaming`](tools/parsers/minimax_m2_tool_parser.md#MinimaxM2ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/qwen3coder_tool_parser.md#Qwen3CoderToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/seed_oss_tool_parser.md#SeedOssToolParser.extract_tool_calls_streaming), [`_build_chat_completion_response`](esurge/server/api_server.md#eSurgeApiServer._build_chat_completion_response), [`extract_tool_calls`](tools/abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](tools/abstract_tool.md#ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/ernie45_tool_parser.md#Ernie45ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/deepseekv31_tool_parser.md#DeepSeekV31ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/deepseekv3_tool_parser.md#DeepSeekV3ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/kimi_k2_tool_parser.md#KimiK2ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/minimax_tool_parser.md#MinimaxToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/hermes_tool_parser.md#HermesToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/qwen3xml_tool_parser.md#Qwen3XMLToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/glm4_moe_tool_parser.md#Glm4MoeModelToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/step3_tool_parser.md#Step3ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/granite_tool_parser.md#GraniteToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/functiongemma_tool_parser.md#FunctionGemmaToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/granite_20b_fc_tool_parser.md#Granite20bFCToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/mistral_tool_parser.md#MistralToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/openai_tool_parser.md#OpenAIToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/xlam_tool_parser.md#xLAMToolParser.extract_tool_calls_streaming), [`_create_standard_response`](esurge/server/api_server.md#eSurgeApiServer._create_standard_response), [`extract_tool_calls_streaming`](tools/parsers/gigachat3_tool_parser.md#GigaChat3ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/jamba_tool_parser.md#JambaToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/internlm2_tool_parser.md#Internlm2ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/llama_tool_parser.md#Llama3JsonToolParser.extract_tool_calls_streaming), [`extract_tool_calls`](tools/parsers/glm4_moe_tool_parser.md#Glm4MoeModelToolParser.extract_tool_calls), [`chat_completions`](esurge/server/api_server.md#eSurgeApiServer.chat_completions), [`build_oai_params_from_chat_request`](oai_proxies.md#InferenceApiRouter.build_oai_params_from_chat_request), [`extract_tool_calls`](tools/parsers/seed_oss_tool_parser.md#SeedOssToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/step3_tool_parser.md#Step3ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/minimax_m2_tool_parser.md#MinimaxM2ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/qwen3coder_tool_parser.md#Qwen3CoderToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](tools/parsers/olmo3_tool_parser.md#Olmo3PythonicToolParser.extract_tool_calls_streaming), [`_extract_delta_tool_calls`](tools/parsers/deepseekv32_tool_parser.md#DeepSeekV32ToolParser._extract_delta_tool_calls), [`extract_tool_calls`](tools/parsers/deepseekv32_tool_parser.md#DeepSeekV32ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](tools/parsers/llama4_pythonic_tool_parser.md#Llama4PythonicToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/pythonic_tool_parser.md#PythonicToolParser.extract_tool_calls_streaming)  (+64 more)

### `ChatCompletionResponse`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:343`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L343)
- doc: Represents a non-streaming response from the chat completion endpoint.
- signature: `class ChatCompletionResponse(OpenAIBaseModel):`
- members:
  - `choices` — [`L359`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L359)
  - `created` — [`L357`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L357)
  - `id` — [`L355`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L355)
  - `model` — [`L358`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L358)
  - `usage` — [`L360`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L360)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`UsageInfo`](openai_api_modules.md#UsageInfo), [`ChatCompletionResponseChoice`](openai_api_modules.md#ChatCompletionResponseChoice)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`_build_chat_completion_response`](esurge/server/api_server.md#eSurgeApiServer._build_chat_completion_response), [`_create_standard_response`](esurge/server/api_server.md#eSurgeApiServer._create_standard_response), [`_handle_chat_completion`](esurge/server/api_server.md#eSurgeApiServer._handle_chat_completion), [`_handle_chat_completion_multimodal`](esurge/server/api_server.md#eSurgeApiServer._handle_chat_completion_multimodal), [`chat_completions`](inference_engine_interface.md#BaseInferenceApiServer.chat_completions)

### `ChatCompletionResponseChoice`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:329`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L329)
- doc: Represents a single choice in a non-streaming chat completion response.
- signature: `class ChatCompletionResponseChoice(OpenAIBaseModel):`
- members:
  - `finish_reason` — [`L340`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L340)
  - `index` — [`L338`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L338)
  - `message` — [`L339`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L339)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`ChatMessage`](openai_api_modules.md#ChatMessage)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`_build_chat_completion_response`](esurge/server/api_server.md#eSurgeApiServer._build_chat_completion_response), [`_create_standard_response`](esurge/server/api_server.md#eSurgeApiServer._create_standard_response), [`choices`](openai_api_modules.md#ChatCompletionResponse.choices)

### `ChatCompletionStreamResponse`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:377`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L377)
- doc: Represents a single chunk in a streaming response from chat completions.
- signature: `class ChatCompletionStreamResponse(OpenAIBaseModel):`
- members:
  - `choices` — [`L393`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L393)
  - `created` — [`L391`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L391)
  - `id` — [`L389`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L389)
  - `model` — [`L392`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L392)
  - `usage` — [`L394`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L394)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`UsageInfo`](openai_api_modules.md#UsageInfo), [`ChatCompletionStreamResponseChoice`](openai_api_modules.md#ChatCompletionStreamResponseChoice)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`generate_stream`](esurge/server/api_server.md#eSurgeApiServer.generate_stream), [`iter_chat_completion_stream_responses`](stream_protocol.md#iter_chat_completion_stream_responses), [`iter_chat_completion_stream`](esurge/mixins/io.md#EngineIOMixin.iter_chat_completion_stream)

### `ChatCompletionStreamResponseChoice`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:363`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L363)
- doc: Represents a single choice in a streaming chat completion response chunk.
- signature: `class ChatCompletionStreamResponseChoice(OpenAIBaseModel):`
- members:
  - `delta` — [`L373`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L373)
  - `finish_reason` — [`L374`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L374)
  - `index` — [`L372`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L372)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](openai_api_modules.md#DeltaMessage), [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`generate_stream`](esurge/server/api_server.md#eSurgeApiServer.generate_stream), [`iter_chat_completion_stream_responses`](stream_protocol.md#iter_chat_completion_stream_responses), [`choices`](openai_api_modules.md#ChatCompletionStreamResponse.choices)

### `ChatMessage`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:101`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L101) — documented in [easydel-inference-openai_api_modules](../../../concepts/easydel-inference-openai_api_modules.md)
- doc: Represents a single message in a chat conversation.
- signature: `class ChatMessage(OpenAIBaseModel):`
- members:
  - `content` — [`L114`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L114)
  - `function_call` — [`L116`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L116)
  - `name` — [`L115`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L115)
  - `reasoning_content` — [`L118`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L118)
  - `role` — [`L113`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L113)
  - `tool_calls` — [`L117`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L117)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)
- used by: [`responses`](esurge/server/api_server.md#eSurgeApiServer.responses), [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`_build_chat_completion_response`](esurge/server/api_server.md#eSurgeApiServer._build_chat_completion_response), [`_create_standard_response`](esurge/server/api_server.md#eSurgeApiServer._create_standard_response), [`assistant_message_from_output_items`](typed_models.md#assistant_message_from_output_items), [`extract_tool_calls_batch`](tools/tool_calling_mixin.md#ToolCallingMixin.extract_tool_calls_batch), [`_responses_payload_to_messages`](inference_engine_interface.md#BaseInferenceApiServer._responses_payload_to_messages), [`_get_tool_request`](parsing/delegating_parser.md#DelegatingParser._get_tool_request), [`_responses_assistant_message_from_output_items`](inference_engine_interface.md#BaseInferenceApiServer._responses_assistant_message_from_output_items), [`responses_assistant_message_from_output_items`](stream_protocol.md#responses_assistant_message_from_output_items), [`_normalize_chat_message`](inference_engine_interface.md#BaseInferenceApiServer._normalize_chat_message), [`messages`](openai_api_modules.md#ChatCompletionRequest.messages), [`_flatten_messages_to_text`](inference_engine_interface.md#BaseInferenceApiServer._flatten_messages_to_text), [`_conversation_from_messages`](inference_engine_interface.md#BaseInferenceApiServer._conversation_from_messages), [`message`](openai_api_modules.md#ChatCompletionResponseChoice.message), [`messages`](openai_api_modules.md#ResponsesRequest.messages), [`conversation`](openai_api_modules.md#CountTokenRequest.conversation)

### `CompletionLogprobs`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:525`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L525)
- doc: Log probabilities for token generation.
- signature: `class CompletionLogprobs(OpenAIBaseModel):`
- members:
  - `text_offset` — [`L538`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L538)
  - `token_logprobs` — [`L536`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L536)
  - `tokens` — [`L535`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L535)
  - `top_logprobs` — [`L537`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L537)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`logprobs`](openai_api_modules.md#CompletionResponseChoice.logprobs), [`logprobs`](openai_api_modules.md#CompletionStreamResponseChoice.logprobs)

### `CompletionRequest`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:409`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L409)
- doc: Represents a request to the completions endpoint.
- signature: `class CompletionRequest(OpenAIBaseModel):`
- members:
  - `frequency_penalty` — [`L438`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L438)
  - `logit_bias` — [`L448`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L448)
  - `max_tokens` — [`L436`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L436)
  - `min_p` — [`L443`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L443)
  - `model` — [`L434`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L434)
  - `n` — [`L445`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L445)
  - `presence_penalty` — [`L437`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L437)
  - `prompt` — [`L435`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L435)
  - `repetition_penalty` — [`L439`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L439)
  - `stop` — [`L447`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L447)
  - `stream` — [`L446`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L446)
  - `suppress_tokens` — [`L444`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L444)
  - `temperature` — [`L440`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L440)
  - `top_k` — [`L442`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L442)
  - `top_p` — [`L441`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L441)
  - `user` — [`L449`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L449)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`generate_stream`](esurge/server/api_server.md#eSurgeApiServer.generate_stream), [`_handle_completion_response`](esurge/server/api_server.md#eSurgeApiServer._handle_completion_response), [`completions`](esurge/server/api_server.md#eSurgeApiServer.completions), [`build_oai_params_from_request`](oai_proxies.md#InferenceApiRouter.build_oai_params_from_request), [`completions`](oai_proxies.md#InferenceApiRouter.completions), [`_create_sampling_params`](esurge/server/api_server.md#eSurgeApiServer._create_sampling_params), [`_prepare_sampling_params`](esurge/server/api_server.md#eSurgeApiServer._prepare_sampling_params), [`_ensure_request_max_tokens`](esurge/server/api_server.md#eSurgeApiServer._ensure_request_max_tokens), [`_handle_completion_streaming`](esurge/server/api_server.md#eSurgeApiServer._handle_completion_streaming), [`_create_sampling_params`](inference_engine_interface.md#BaseInferenceApiServer._create_sampling_params), [`RefineSamplingParamsFn`](esurge/server/api_server.md#RefineSamplingParamsFn), [`completions`](inference_engine_interface.md#BaseInferenceApiServer.completions), [`process_request_params`](oai_proxies.md#InferenceApiRouter.process_request_params)

### `CompletionResponse`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:557`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L557)
- doc: Represents a response from the completions endpoint.
- signature: `class CompletionResponse(OpenAIBaseModel):`
- members:
  - `choices` — [`L573`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L573)
  - `created` — [`L571`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L571)
  - `id` — [`L569`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L569)
  - `model` — [`L572`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L572)
  - `usage` — [`L574`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L574)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`UsageInfo`](openai_api_modules.md#UsageInfo), [`CompletionResponseChoice`](openai_api_modules.md#CompletionResponseChoice)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`_handle_completion_response`](esurge/server/api_server.md#eSurgeApiServer._handle_completion_response), [`completions`](inference_engine_interface.md#BaseInferenceApiServer.completions)

### `CompletionResponseChoice`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:541`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L541)
- doc: Represents a single choice within a completion response.
- signature: `class CompletionResponseChoice(OpenAIBaseModel):`
- members:
  - `finish_reason` — [`L554`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L554)
  - `index` — [`L552`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L552)
  - `logprobs` — [`L553`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L553)
  - `text` — [`L551`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L551)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`CompletionLogprobs`](openai_api_modules.md#CompletionLogprobs)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`_handle_completion_response`](esurge/server/api_server.md#eSurgeApiServer._handle_completion_response), [`choices`](openai_api_modules.md#CompletionResponse.choices)

### `CompletionStreamResponse`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:593`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L593)
- doc: Represents a streaming response from the completions endpoint.
- signature: `class CompletionStreamResponse(OpenAIBaseModel):`
- members:
  - `choices` — [`L609`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L609)
  - `created` — [`L607`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L607)
  - `id` — [`L605`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L605)
  - `model` — [`L608`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L608)
  - `usage` — [`L610`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L610)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`UsageInfo`](openai_api_modules.md#UsageInfo), [`CompletionStreamResponseChoice`](openai_api_modules.md#CompletionStreamResponseChoice)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)

### `CompletionStreamResponseChoice`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:577`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L577)
- doc: Represents a single choice within a streaming completion response chunk.
- signature: `class CompletionStreamResponseChoice(OpenAIBaseModel):`
- members:
  - `finish_reason` — [`L590`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L590)
  - `index` — [`L587`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L587)
  - `logprobs` — [`L589`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L589)
  - `text` — [`L588`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L588)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`CompletionLogprobs`](openai_api_modules.md#CompletionLogprobs)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`choices`](openai_api_modules.md#CompletionStreamResponse.choices)

### `ConversationReference`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:262`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L262)
- doc: Typed conversation reference for Responses API continuation.
- signature: `class ConversationReference(OpenAIBaseModel):`
- members:
  - `conversation` — [`L267`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L267)
  - `conversation_id` — [`L266`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L266)
  - `id` — [`L265`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L265)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`_normalize_conversation_id`](inference_engine_interface.md#BaseInferenceApiServer._normalize_conversation_id), [`conversation`](openai_api_modules.md#ResponsesRequest.conversation)

### `CountTokenRequest`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:397`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L397)
- doc: Represents a request to the token counting endpoint.
- signature: `class CountTokenRequest(OpenAIBaseModel):`
- members:
  - `conversation` — [`L406`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L406)
  - `model` — [`L405`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L405)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`ChatMessage`](openai_api_modules.md#ChatMessage)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)

### `DeltaFunctionCall`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:167`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L167) — documented in [easydel-inference-openai_api_modules](../../../concepts/easydel-inference-openai_api_modules.md)
- doc: Incremental function call update for streaming.
- signature: `class DeltaFunctionCall(OpenAIBaseModel):`
- members:
  - `arguments` — [`L176`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L176)
  - `name` — [`L175`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L175)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`add_output`](stream_protocol.md#ResponsesStreamAccumulator.add_output), [`extract_tool_calls_streaming`](tools/parsers/minimax_m2_tool_parser.md#MinimaxM2ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/qwen3coder_tool_parser.md#Qwen3CoderToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/seed_oss_tool_parser.md#SeedOssToolParser.extract_tool_calls_streaming), [`index`](openai_api_modules.md#DeltaToolCall.index), [`extract_tool_calls_streaming`](tools/parsers/ernie45_tool_parser.md#Ernie45ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/deepseekv31_tool_parser.md#DeepSeekV31ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/deepseekv3_tool_parser.md#DeepSeekV3ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/kimi_k2_tool_parser.md#KimiK2ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/hermes_tool_parser.md#HermesToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/qwen3xml_tool_parser.md#Qwen3XMLToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/glm4_moe_tool_parser.md#Glm4MoeModelToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/step3_tool_parser.md#Step3ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/granite_tool_parser.md#GraniteToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/functiongemma_tool_parser.md#FunctionGemmaToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/granite_20b_fc_tool_parser.md#Granite20bFCToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/mistral_tool_parser.md#MistralToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/openai_tool_parser.md#OpenAIToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/xlam_tool_parser.md#xLAMToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/gigachat3_tool_parser.md#GigaChat3ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/jamba_tool_parser.md#JambaToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/internlm2_tool_parser.md#Internlm2ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/llama_tool_parser.md#Llama3JsonToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/olmo3_tool_parser.md#Olmo3PythonicToolParser.extract_tool_calls_streaming), [`_extract_delta_tool_calls`](tools/parsers/deepseekv32_tool_parser.md#DeepSeekV32ToolParser._extract_delta_tool_calls), [`extract_tool_calls_streaming`](tools/parsers/llama4_pythonic_tool_parser.md#Llama4PythonicToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/pythonic_tool_parser.md#PythonicToolParser.extract_tool_calls_streaming), [`_build_missing_final_tool_deltas`](parsing/delegating_parser.md#DelegatingParser._build_missing_final_tool_deltas), [`_handle_tool_args_streaming`](tools/parsers/minimax_tool_parser.md#MinimaxToolParser._handle_tool_args_streaming), [`_handle_tool_call_middle`](tools/parsers/gemma4_tool_parser.md#Gemma4ToolParser._handle_tool_call_middle), [`_merge_streamed_tool_call_state`](parsing/delegating_parser.md#DelegatingParser._merge_streamed_tool_call_state), [`_handle_test_compatibility`](tools/parsers/hunyuan_a13b_tool_parser.md#HunyuanA13BToolParser._handle_test_compatibility), [`_handle_tool_name_streaming`](tools/parsers/minimax_tool_parser.md#MinimaxToolParser._handle_tool_name_streaming), [`_handle_tool_args_streaming`](tools/parsers/hunyuan_a13b_tool_parser.md#HunyuanA13BToolParser._handle_tool_args_streaming), [`_handle_tool_call_end`](tools/parsers/gemma4_tool_parser.md#Gemma4ToolParser._handle_tool_call_end), [`_handle_tool_name_streaming`](tools/parsers/hunyuan_a13b_tool_parser.md#HunyuanA13BToolParser._handle_tool_name_streaming), [`_compute_tool_delta`](tools/parsers/llama4_pythonic_tool_parser.md#_compute_tool_delta), [`_compute_tool_delta`](tools/parsers/olmo3_tool_parser.md#_compute_tool_delta), [`_compute_tool_delta`](tools/parsers/pythonic_tool_parser.md#_compute_tool_delta)  (+1 more)

### `DeltaMessage`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:121`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L121) — documented in [easydel-inference-openai_api_modules](../../../concepts/easydel-inference-openai_api_modules.md)
- doc: Represents a change (delta) in a chat message.
- signature: `class DeltaMessage(OpenAIBaseModel):`
- members:
  - `content` — [`L135`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L135)
  - `function_call` — [`L136`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L136)
  - `reasoning_content` — [`L138`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L138)
  - `role` — [`L134`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L134)
  - `tool_calls` — [`L137`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L137)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`generate_stream`](esurge/server/api_server.md#eSurgeApiServer.generate_stream), [`extract_tool_calls_streaming`](tools/parsers/minimax_m2_tool_parser.md#MinimaxM2ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/qwen3coder_tool_parser.md#Qwen3CoderToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/seed_oss_tool_parser.md#SeedOssToolParser.extract_tool_calls_streaming), [`iter_chat_completion_stream_responses`](stream_protocol.md#iter_chat_completion_stream_responses), [`extract_tool_calls_streaming`](tools/abstract_tool.md#ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/ernie45_tool_parser.md#Ernie45ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/deepseekv31_tool_parser.md#DeepSeekV31ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/deepseekv3_tool_parser.md#DeepSeekV3ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/kimi_k2_tool_parser.md#KimiK2ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/minimax_tool_parser.md#MinimaxToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/hermes_tool_parser.md#HermesToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/qwen3xml_tool_parser.md#Qwen3XMLToolParser.extract_tool_calls_streaming), [`_process_tool_delta`](parsing/delegating_parser.md#DelegatingParser._process_tool_delta), [`extract_tool_calls_streaming`](tools/parsers/glm4_moe_tool_parser.md#Glm4MoeModelToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/step3_tool_parser.md#Step3ToolParser.extract_tool_calls_streaming), [`_process_reasoning_delta`](parsing/delegating_parser.md#DelegatingParser._process_reasoning_delta), [`extract_tool_calls_streaming`](tools/parsers/granite_tool_parser.md#GraniteToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/functiongemma_tool_parser.md#FunctionGemmaToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/granite_20b_fc_tool_parser.md#Granite20bFCToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/mistral_tool_parser.md#MistralToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/openai_tool_parser.md#OpenAIToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/xlam_tool_parser.md#xLAMToolParser.extract_tool_calls_streaming), [`_build_stream_debug_context`](esurge/server/api_server.md#eSurgeApiServer._build_stream_debug_context), [`extract_tool_calls_streaming`](tools/parsers/gigachat3_tool_parser.md#GigaChat3ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/jamba_tool_parser.md#JambaToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/internlm2_tool_parser.md#Internlm2ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/llama_tool_parser.md#Llama3JsonToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/olmo3_tool_parser.md#Olmo3PythonicToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/llama4_pythonic_tool_parser.md#Llama4PythonicToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/pythonic_tool_parser.md#PythonicToolParser.extract_tool_calls_streaming), [`_handle_tool_args_streaming`](tools/parsers/minimax_tool_parser.md#MinimaxToolParser._handle_tool_args_streaming), [`_handle_tool_call_middle`](tools/parsers/gemma4_tool_parser.md#Gemma4ToolParser._handle_tool_call_middle), [`_handle_test_compatibility`](tools/parsers/hunyuan_a13b_tool_parser.md#HunyuanA13BToolParser._handle_test_compatibility), [`_handle_tool_name_streaming`](tools/parsers/minimax_tool_parser.md#MinimaxToolParser._handle_tool_name_streaming), [`extract_tool_calls_streaming`](tools/parsers/hunyuan_a13b_tool_parser.md#HunyuanA13BToolParser.extract_tool_calls_streaming), [`extract_reasoning_streaming`](reasoning/basic_parsers.md#BaseThinkingReasoningParser.extract_reasoning_streaming), [`_handle_tool_args_streaming`](tools/parsers/hunyuan_a13b_tool_parser.md#HunyuanA13BToolParser._handle_tool_args_streaming), [`_handle_tool_call_end`](tools/parsers/gemma4_tool_parser.md#Gemma4ToolParser._handle_tool_call_end)  (+20 more)

### `DeltaToolCall`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:179`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L179) — documented in [easydel-inference-openai_api_modules](../../../concepts/easydel-inference-openai_api_modules.md)
- doc: Incremental tool call update for streaming.
- signature: `class DeltaToolCall(OpenAIBaseModel):`
- members:
  - `id` — [`L189`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L189)
  - `index` — [`L191`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L191)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`DeltaFunctionCall`](openai_api_modules.md#DeltaFunctionCall)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`add_output`](stream_protocol.md#ResponsesStreamAccumulator.add_output), [`extract_tool_calls_streaming`](tools/parsers/minimax_m2_tool_parser.md#MinimaxM2ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/qwen3coder_tool_parser.md#Qwen3CoderToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/seed_oss_tool_parser.md#SeedOssToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/ernie45_tool_parser.md#Ernie45ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/deepseekv31_tool_parser.md#DeepSeekV31ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/deepseekv3_tool_parser.md#DeepSeekV3ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/kimi_k2_tool_parser.md#KimiK2ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/hermes_tool_parser.md#HermesToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/qwen3xml_tool_parser.md#Qwen3XMLToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/glm4_moe_tool_parser.md#Glm4MoeModelToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/step3_tool_parser.md#Step3ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/granite_tool_parser.md#GraniteToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/functiongemma_tool_parser.md#FunctionGemmaToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/granite_20b_fc_tool_parser.md#Granite20bFCToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/mistral_tool_parser.md#MistralToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/openai_tool_parser.md#OpenAIToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/xlam_tool_parser.md#xLAMToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/gigachat3_tool_parser.md#GigaChat3ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/jamba_tool_parser.md#JambaToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/internlm2_tool_parser.md#Internlm2ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/llama_tool_parser.md#Llama3JsonToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/olmo3_tool_parser.md#Olmo3PythonicToolParser.extract_tool_calls_streaming), [`_extract_delta_tool_calls`](tools/parsers/deepseekv32_tool_parser.md#DeepSeekV32ToolParser._extract_delta_tool_calls), [`_build_missing_final_tool_deltas`](parsing/delegating_parser.md#DelegatingParser._build_missing_final_tool_deltas), [`_handle_tool_args_streaming`](tools/parsers/minimax_tool_parser.md#MinimaxToolParser._handle_tool_args_streaming), [`_handle_tool_call_middle`](tools/parsers/gemma4_tool_parser.md#Gemma4ToolParser._handle_tool_call_middle), [`_merge_streamed_tool_call_state`](parsing/delegating_parser.md#DelegatingParser._merge_streamed_tool_call_state), [`_handle_test_compatibility`](tools/parsers/hunyuan_a13b_tool_parser.md#HunyuanA13BToolParser._handle_test_compatibility), [`_handle_tool_name_streaming`](tools/parsers/minimax_tool_parser.md#MinimaxToolParser._handle_tool_name_streaming), [`_handle_tool_args_streaming`](tools/parsers/hunyuan_a13b_tool_parser.md#HunyuanA13BToolParser._handle_tool_args_streaming), [`_handle_tool_call_end`](tools/parsers/gemma4_tool_parser.md#Gemma4ToolParser._handle_tool_call_end), [`_handle_tool_name_streaming`](tools/parsers/hunyuan_a13b_tool_parser.md#HunyuanA13BToolParser._handle_tool_name_streaming), [`_compute_tool_delta`](tools/parsers/llama4_pythonic_tool_parser.md#_compute_tool_delta), [`_compute_tool_delta`](tools/parsers/olmo3_tool_parser.md#_compute_tool_delta), [`_compute_tool_delta`](tools/parsers/pythonic_tool_parser.md#_compute_tool_delta), [`_emit_argument_diff`](tools/parsers/gemma4_tool_parser.md#Gemma4ToolParser._emit_argument_diff), [`normalize_delta_tool_calls`](stream_protocol.md#normalize_delta_tool_calls)

### `ExtractedToolCallInformation`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:663`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L663) — documented in [easydel-inference-openai_api_modules](../../../concepts/easydel-inference-openai_api_modules.md)
- doc: Information extracted from model output about tool calls.
- signature: `class ExtractedToolCallInformation(OpenAIBaseModel):`
- members:
  - `content` — [`L674`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L674)
  - `tool_calls` — [`L673`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L673)
  - `tools_called` — [`L672`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L672)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`ToolCall`](openai_api_modules.md#ToolCall)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`extract_tool_calls`](tools/abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](tools/parsers/ernie45_tool_parser.md#Ernie45ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/hermes_tool_parser.md#HermesToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/glm4_moe_tool_parser.md#Glm4MoeModelToolParser.extract_tool_calls_streaming), [`_process_tool_final`](parsing/delegating_parser.md#DelegatingParser._process_tool_final), [`extract_tool_calls`](tools/parsers/glm4_moe_tool_parser.md#Glm4MoeModelToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/seed_oss_tool_parser.md#SeedOssToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/step3_tool_parser.md#Step3ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/minimax_m2_tool_parser.md#MinimaxM2ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/qwen3coder_tool_parser.md#Qwen3CoderToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/deepseekv32_tool_parser.md#DeepSeekV32ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/functiongemma_tool_parser.md#FunctionGemmaToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/gemma4_tool_parser.md#Gemma4ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/granite_20b_fc_tool_parser.md#Granite20bFCToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/kimi_k2_tool_parser.md#KimiK2ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/minimax_tool_parser.md#MinimaxToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/mistral_tool_parser.md#MistralToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/xlam_tool_parser.md#xLAMToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/ernie45_tool_parser.md#Ernie45ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/deepseekv31_tool_parser.md#DeepSeekV31ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/deepseekv3_tool_parser.md#DeepSeekV3ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/granite_tool_parser.md#GraniteToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/hunyuan_a13b_tool_parser.md#HunyuanA13BToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/jamba_tool_parser.md#JambaToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/qwen3xml_tool_parser.md#Qwen3XMLToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/hermes_tool_parser.md#HermesToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/internlm2_tool_parser.md#Internlm2ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/llama_tool_parser.md#Llama3JsonToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/openai_tool_parser.md#OpenAIToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/phi4mini_tool_parser.md#Phi4MiniJsonToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/gigachat3_tool_parser.md#GigaChat3ToolParser.extract_tool_calls), [`extract_tool_calls_batch`](tools/tool_calling_mixin.md#ToolCallingMixin.extract_tool_calls_batch), [`extract_tool_calls`](tools/parsers/llama4_pythonic_tool_parser.md#Llama4PythonicToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/olmo3_tool_parser.md#Olmo3PythonicToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/pythonic_tool_parser.md#PythonicToolParser.extract_tool_calls)

### `Function`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:141`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L141)
- doc: Function definition for OpenAI-compatible function calling.
- signature: `class Function(OpenAIBaseModel):`
- members:
  - `description` — [`L151`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L151)
  - `name` — [`L150`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L150)
  - `parameters` — [`L152`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L152)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`Tool`](openai_api_modules.md#Tool)

### `FunctionCall`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:614`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L614) — documented in [easydel-inference-openai_api_modules](../../../concepts/easydel-inference-openai_api_modules.md)
- doc: Represents a function call in the OpenAI format.
- signature: `class FunctionCall(OpenAIBaseModel):`
- members:
  - `arguments` — [`L623`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L623)
  - `name` — [`L622`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L622)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)
- used by: [`finalize`](stream_protocol.md#ResponsesStreamAccumulator.finalize), [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`extract_tool_calls_streaming`](tools/parsers/qwen3coder_tool_parser.md#Qwen3CoderToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/seed_oss_tool_parser.md#SeedOssToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/ernie45_tool_parser.md#Ernie45ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/hermes_tool_parser.md#HermesToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/qwen3xml_tool_parser.md#Qwen3XMLToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/glm4_moe_tool_parser.md#Glm4MoeModelToolParser.extract_tool_calls_streaming), [`assistant_message_from_output_items`](typed_models.md#assistant_message_from_output_items), [`extract_tool_calls`](tools/parsers/glm4_moe_tool_parser.md#Glm4MoeModelToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/seed_oss_tool_parser.md#SeedOssToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/step3_tool_parser.md#Step3ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/minimax_m2_tool_parser.md#MinimaxM2ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/qwen3coder_tool_parser.md#Qwen3CoderToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/deepseekv32_tool_parser.md#DeepSeekV32ToolParser.extract_tool_calls), [`_build_missing_final_tool_deltas`](parsing/delegating_parser.md#DelegatingParser._build_missing_final_tool_deltas), [`extract_tool_calls`](tools/parsers/functiongemma_tool_parser.md#FunctionGemmaToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/gemma4_tool_parser.md#Gemma4ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/granite_20b_fc_tool_parser.md#Granite20bFCToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/kimi_k2_tool_parser.md#KimiK2ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/minimax_tool_parser.md#MinimaxToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/mistral_tool_parser.md#MistralToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/xlam_tool_parser.md#xLAMToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/ernie45_tool_parser.md#Ernie45ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/deepseekv31_tool_parser.md#DeepSeekV31ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/deepseekv3_tool_parser.md#DeepSeekV3ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/granite_tool_parser.md#GraniteToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/hunyuan_a13b_tool_parser.md#HunyuanA13BToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/jamba_tool_parser.md#JambaToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/qwen3xml_tool_parser.md#Qwen3XMLToolParser.extract_tool_calls), [`_parse_xml_function_call`](tools/parsers/qwen3xml_tool_parser.md#Qwen3XMLToolParser._parse_xml_function_call), [`extract_tool_calls`](tools/parsers/hermes_tool_parser.md#HermesToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/internlm2_tool_parser.md#Internlm2ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/llama_tool_parser.md#Llama3JsonToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/openai_tool_parser.md#OpenAIToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/phi4mini_tool_parser.md#Phi4MiniJsonToolParser.extract_tool_calls), [`_compute_tool_delta`](tools/parsers/llama4_pythonic_tool_parser.md#_compute_tool_delta), [`_compute_tool_delta`](tools/parsers/olmo3_tool_parser.md#_compute_tool_delta), [`_compute_tool_delta`](tools/parsers/pythonic_tool_parser.md#_compute_tool_delta), [`build_responses_function_call_items`](stream_protocol.md#build_responses_function_call_items)  (+8 more)

### `FunctionCallFormat`  ·  implements/extends StrEnum
- def: [`easydel/inference/openai_api_modules.py:640`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L640)
- doc: Supported function call formats.
- signature: `class FunctionCallFormat(StrEnum):`
- members:
  - `GORILLA` — [`L658`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L658)
  - `HERMES` — [`L657`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L657)
  - `JSON_SCHEMA` — [`L656`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L656)
  - `NOUS` — [`L660`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L660)
  - `OPENAI` — [`L655`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L655)
  - `QWEN` — [`L659`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L659)
- used by: [`__init__`](inference_engine_interface.md#BaseInferenceApiServer.__init__)

### `FunctionCallSelection`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:231`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L231)
- doc: Typed legacy function-call selector used by request models.
- signature: `class FunctionCallSelection(OpenAIBaseModel):`
- members:
  - `name` — [`L234`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L234)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`function_call`](openai_api_modules.md#ChatCompletionRequest.function_call)

### `FunctionDefinition`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:215`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L215)
- doc: Defines a function that can be called by the model.
- signature: `class FunctionDefinition(OpenAIBaseModel):`
- members:
  - `description` — [`L226`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L226)
  - `name` — [`L225`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L225)
  - `parameters` — [`L227`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L227)
  - `required` — [`L228`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L228)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`extract_tool_calls_streaming`](tools/parsers/minimax_m2_tool_parser.md#MinimaxM2ToolParser.extract_tool_calls_streaming), [`extract_tool_calls`](tools/parsers/internlm2_tool_parser.md#Internlm2ToolParser.extract_tool_calls), [`ToolDefinition`](openai_api_modules.md#ToolDefinition), [`_convert_params_with_schema`](tools/parsers/deepseekv32_tool_parser.md#DeepSeekV32ToolParser._convert_params_with_schema), [`_extract_responses_tools`](inference_engine_interface.md#BaseInferenceApiServer._extract_responses_tools), [`_get_arguments_config`](tools/parsers/qwen3coder_tool_parser.md#Qwen3CoderToolParser._get_arguments_config), [`_cast_arguments`](tools/parsers/step3_tool_parser.md#Step3ToolParser._cast_arguments), [`_is_string_type`](tools/parsers/glm4_moe_tool_parser.md#Glm4MoeModelToolParser._is_string_type), [`get_arguments_config`](tools/parsers/seed_oss_tool_parser.md#SeedOssToolParser.get_arguments_config), [`tools`](typed_models.md#ResponsesFinalizationOptions.tools), [`functions`](openai_api_modules.md#ChatCompletionRequest.functions), [`tools`](typed_models.md#ResponsesResponse.tools), [`functions`](openai_api_modules.md#ResponsesRequest.functions)

### `OpenAIBaseModel`  ·  implements/extends BaseModel
- def: [`easydel/inference/openai_api_modules.py:60`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L60) — documented in [easydel-inference-openai_api_modules](../../../concepts/easydel-inference-openai_api_modules.md)
- doc: Base model for OpenAI API compatibility.
- signature: `class OpenAIBaseModel(BaseModel):`
- members:
  - `__log_extra_fields__(cls, data, handler)` — [`L76`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L76) — Validate and track field names for the model.
  - `field_names` — [`L72`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L72)
  - `model_config` — [`L71`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L71)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](openai_api_modules.md#ChatCompletionRequest), [`DeltaToolCall`](openai_api_modules.md#DeltaToolCall), [`DeltaFunctionCall`](openai_api_modules.md#DeltaFunctionCall), [`ToolCall`](openai_api_modules.md#ToolCall), [`FunctionCall`](openai_api_modules.md#FunctionCall), [`ChatMessage`](openai_api_modules.md#ChatMessage), [`ToolDefinition`](openai_api_modules.md#ToolDefinition), [`CompletionRequest`](openai_api_modules.md#CompletionRequest), [`UsageInfo`](openai_api_modules.md#UsageInfo), [`ChatCompletionStreamResponse`](openai_api_modules.md#ChatCompletionStreamResponse), [`ResponseFunctionCallItem`](typed_models.md#ResponseFunctionCallItem), [`ResponseMessageItem`](typed_models.md#ResponseMessageItem), [`ResponsesRequest`](openai_api_modules.md#ResponsesRequest), [`ResponsesResponse`](typed_models.md#ResponsesResponse), [`ResponsesFinalizationOptions`](typed_models.md#ResponsesFinalizationOptions), [`ChatCompletionResponse`](openai_api_modules.md#ChatCompletionResponse), [`FunctionDefinition`](openai_api_modules.md#FunctionDefinition), [`ResponseReasoningItem`](typed_models.md#ResponseReasoningItem), [`ToolChoiceOption`](openai_api_modules.md#ToolChoiceOption), [`ChatCompletionStreamResponseChoice`](openai_api_modules.md#ChatCompletionStreamResponseChoice), [`ResponseOutputTextPart`](typed_models.md#ResponseOutputTextPart), [`ResponsesTextConfig`](typed_models.md#ResponsesTextConfig), [`ChatCompletionResponseChoice`](openai_api_modules.md#ChatCompletionResponseChoice), [`CompletionResponse`](openai_api_modules.md#CompletionResponse), [`ResponseCompletedEvent`](typed_models.md#ResponseCompletedEvent), [`ResponseOutputItemAddedEvent`](typed_models.md#ResponseOutputItemAddedEvent), [`ResponseOutputItemDoneEvent`](typed_models.md#ResponseOutputItemDoneEvent), [`ResponseFunctionCallArgumentsDeltaEvent`](typed_models.md#ResponseFunctionCallArgumentsDeltaEvent), [`CompletionResponseChoice`](openai_api_modules.md#CompletionResponseChoice), [`ConversationReference`](openai_api_modules.md#ConversationReference), [`ResponseContentPartAddedEvent`](typed_models.md#ResponseContentPartAddedEvent), [`ResponseCreatedEvent`](typed_models.md#ResponseCreatedEvent), [`ResponseFunctionCallArgumentsDoneEvent`](typed_models.md#ResponseFunctionCallArgumentsDoneEvent), [`ResponseOutputTextDeltaEvent`](typed_models.md#ResponseOutputTextDeltaEvent), [`ResponseOutputTextDoneEvent`](typed_models.md#ResponseOutputTextDoneEvent), [`ResponseReasoningSummaryTextDeltaEvent`](typed_models.md#ResponseReasoningSummaryTextDeltaEvent), [`ResponseReasoningSummaryTextDoneEvent`](typed_models.md#ResponseReasoningSummaryTextDoneEvent), [`ResponseSummaryText`](typed_models.md#ResponseSummaryText)  (+11 more)
- used by: [`DeltaMessage`](openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](openai_api_modules.md#ChatCompletionRequest), [`DeltaToolCall`](openai_api_modules.md#DeltaToolCall), [`DeltaFunctionCall`](openai_api_modules.md#DeltaFunctionCall), [`ToolCall`](openai_api_modules.md#ToolCall), [`FunctionCall`](openai_api_modules.md#FunctionCall), [`ChatMessage`](openai_api_modules.md#ChatMessage), [`ToolDefinition`](openai_api_modules.md#ToolDefinition), [`CompletionRequest`](openai_api_modules.md#CompletionRequest), [`UsageInfo`](openai_api_modules.md#UsageInfo), [`ChatCompletionStreamResponse`](openai_api_modules.md#ChatCompletionStreamResponse), [`ResponseFunctionCallItem`](typed_models.md#ResponseFunctionCallItem), [`ResponseMessageItem`](typed_models.md#ResponseMessageItem), [`ResponsesRequest`](openai_api_modules.md#ResponsesRequest), [`ResponsesResponse`](typed_models.md#ResponsesResponse), [`ResponsesFinalizationOptions`](typed_models.md#ResponsesFinalizationOptions), [`ChatCompletionResponse`](openai_api_modules.md#ChatCompletionResponse), [`FunctionDefinition`](openai_api_modules.md#FunctionDefinition), [`ResponseReasoningItem`](typed_models.md#ResponseReasoningItem), [`ToolChoiceOption`](openai_api_modules.md#ToolChoiceOption), [`ChatCompletionStreamResponseChoice`](openai_api_modules.md#ChatCompletionStreamResponseChoice), [`ResponseOutputTextPart`](typed_models.md#ResponseOutputTextPart), [`ResponsesTextConfig`](typed_models.md#ResponsesTextConfig), [`ChatCompletionResponseChoice`](openai_api_modules.md#ChatCompletionResponseChoice), [`CompletionResponse`](openai_api_modules.md#CompletionResponse), [`ResponseCompletedEvent`](typed_models.md#ResponseCompletedEvent), [`ResponseOutputItemAddedEvent`](typed_models.md#ResponseOutputItemAddedEvent), [`ResponseOutputItemDoneEvent`](typed_models.md#ResponseOutputItemDoneEvent), [`ResponseFunctionCallArgumentsDeltaEvent`](typed_models.md#ResponseFunctionCallArgumentsDeltaEvent), [`CompletionResponseChoice`](openai_api_modules.md#CompletionResponseChoice), [`ConversationReference`](openai_api_modules.md#ConversationReference), [`ResponseContentPartAddedEvent`](typed_models.md#ResponseContentPartAddedEvent), [`ResponseCreatedEvent`](typed_models.md#ResponseCreatedEvent), [`ResponseFunctionCallArgumentsDoneEvent`](typed_models.md#ResponseFunctionCallArgumentsDoneEvent), [`ResponseOutputTextDeltaEvent`](typed_models.md#ResponseOutputTextDeltaEvent), [`ResponseOutputTextDoneEvent`](typed_models.md#ResponseOutputTextDoneEvent), [`ResponseReasoningSummaryTextDeltaEvent`](typed_models.md#ResponseReasoningSummaryTextDeltaEvent), [`ResponseReasoningSummaryTextDoneEvent`](typed_models.md#ResponseReasoningSummaryTextDoneEvent), [`ResponseSummaryText`](typed_models.md#ResponseSummaryText)  (+11 more)

### `ResponseReasoningConfig`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:270`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L270)
- doc: Typed reasoning config for Responses API requests.
- signature: `class ResponseReasoningConfig(OpenAIBaseModel):`
- members:
  - `summary` — [`L273`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L273)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`_responses_reasoning_summary_requested`](inference_engine_interface.md#BaseInferenceApiServer._responses_reasoning_summary_requested), [`reasoning`](openai_api_modules.md#ResponsesRequest.reasoning)

### `ResponsesRequest`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:452`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L452)
- doc: Represents a request to the OpenAI Responses API (POST /v1/responses).
- signature: `class ResponsesRequest(OpenAIBaseModel):`
- members:
  - `conversation` — [`L495`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L495)
  - `frequency_penalty` — [`L506`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L506)
  - `functions` — [`L513`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L513)
  - `include` — [`L521`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L521)
  - `input` — [`L488`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L488)
  - `instructions` — [`L490`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L490)
  - `max_completion_tokens` — [`L500`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L500)
  - `max_output_tokens` — [`L498`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L498)
  - `max_tokens` — [`L499`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L499)
  - `messages` — [`L489`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L489)
  - `metadata` — [`L522`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L522)
  - `min_p` — [`L504`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L504)
  - `model` — [`L487`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L487)
  - `n` — [`L509`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L509)
  - `parallel_tool_calls` — [`L515`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L515)
  - `presence_penalty` — [`L505`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L505)
  - `previous_response_id` — [`L494`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L494)
  - `reasoning` — [`L516`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L516)
  - `repetition_penalty` — [`L507`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L507)
  - `stop` — [`L508`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L508)
  - `store` — [`L493`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L493)
  - `stream` — [`L519`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L519)
  - `temperature` — [`L501`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L501)
  - `tool_choice` — [`L514`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L514)
  - `tools` — [`L512`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L512)
  - `top_k` — [`L503`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L503)
  - `top_p` — [`L502`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L502)
  - `truncation` — [`L520`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L520)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`ChatMessage`](openai_api_modules.md#ChatMessage), [`ToolDefinition`](openai_api_modules.md#ToolDefinition), [`FunctionDefinition`](openai_api_modules.md#FunctionDefinition), [`ToolChoiceOption`](openai_api_modules.md#ToolChoiceOption), [`ConversationReference`](openai_api_modules.md#ConversationReference), [`ResponseReasoningConfig`](openai_api_modules.md#ResponseReasoningConfig)
- used by: [`responses`](esurge/server/api_server.md#eSurgeApiServer.responses), [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`_create_sampling_params_from_responses`](inference_engine_interface.md#BaseInferenceApiServer._create_sampling_params_from_responses), [`responses`](oai_proxies.md#InferenceApiRouter.responses), [`_responses_payload_to_messages`](inference_engine_interface.md#BaseInferenceApiServer._responses_payload_to_messages), [`_extract_responses_tools`](inference_engine_interface.md#BaseInferenceApiServer._extract_responses_tools), [`_parse_responses_max_tokens`](inference_engine_interface.md#BaseInferenceApiServer._parse_responses_max_tokens), [`_responses_reasoning_summary_requested`](inference_engine_interface.md#BaseInferenceApiServer._responses_reasoning_summary_requested), [`responses`](inference_engine_interface.md#BaseInferenceApiServer.responses)

### `Tool`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:155`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L155)
- doc: Tool definition supporting function calling.
- signature: `class Tool(OpenAIBaseModel):`
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`Function`](openai_api_modules.md#Function)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)

### `ToolCall`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:626`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L626) — documented in [easydel-inference-openai_api_modules](../../../concepts/easydel-inference-openai_api_modules.md)
- doc: Represents a tool call in responses.
- signature: `class ToolCall(OpenAIBaseModel):`
- members:
  - `id` — [`L635`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L635)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`FunctionCall`](openai_api_modules.md#FunctionCall), [`MistralToolCall`](tools/parsers/mistral_tool_parser.md#MistralToolCall)
- used by: [`finalize`](stream_protocol.md#ResponsesStreamAccumulator.finalize), [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`tool_calls`](openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`extract_tool_calls_streaming`](tools/parsers/ernie45_tool_parser.md#Ernie45ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](tools/parsers/glm4_moe_tool_parser.md#Glm4MoeModelToolParser.extract_tool_calls_streaming), [`assistant_message_from_output_items`](typed_models.md#assistant_message_from_output_items), [`extract_tool_calls`](tools/parsers/glm4_moe_tool_parser.md#Glm4MoeModelToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/step3_tool_parser.md#Step3ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/minimax_m2_tool_parser.md#MinimaxM2ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/deepseekv32_tool_parser.md#DeepSeekV32ToolParser.extract_tool_calls), [`_build_missing_final_tool_deltas`](parsing/delegating_parser.md#DelegatingParser._build_missing_final_tool_deltas), [`extract_tool_calls`](tools/parsers/functiongemma_tool_parser.md#FunctionGemmaToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/gemma4_tool_parser.md#Gemma4ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/granite_20b_fc_tool_parser.md#Granite20bFCToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/kimi_k2_tool_parser.md#KimiK2ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/minimax_tool_parser.md#MinimaxToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/xlam_tool_parser.md#xLAMToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/ernie45_tool_parser.md#Ernie45ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/deepseekv31_tool_parser.md#DeepSeekV31ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/deepseekv3_tool_parser.md#DeepSeekV3ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/granite_tool_parser.md#GraniteToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/hunyuan_a13b_tool_parser.md#HunyuanA13BToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/jamba_tool_parser.md#JambaToolParser.extract_tool_calls), [`_parse_xml_function_call`](tools/parsers/qwen3xml_tool_parser.md#Qwen3XMLToolParser._parse_xml_function_call), [`extract_tool_calls`](tools/parsers/hermes_tool_parser.md#HermesToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/internlm2_tool_parser.md#Internlm2ToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/llama_tool_parser.md#Llama3JsonToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/openai_tool_parser.md#OpenAIToolParser.extract_tool_calls), [`extract_tool_calls`](tools/parsers/phi4mini_tool_parser.md#Phi4MiniJsonToolParser.extract_tool_calls), [`_compute_tool_delta`](tools/parsers/llama4_pythonic_tool_parser.md#_compute_tool_delta), [`_compute_tool_delta`](tools/parsers/olmo3_tool_parser.md#_compute_tool_delta), [`_compute_tool_delta`](tools/parsers/pythonic_tool_parser.md#_compute_tool_delta), [`build_responses_function_call_items`](stream_protocol.md#build_responses_function_call_items), [`extract_tool_calls`](tools/parsers/gigachat3_tool_parser.md#GigaChat3ToolParser.extract_tool_calls), [`_parse_xml_function_call`](tools/parsers/qwen3coder_tool_parser.md#Qwen3CoderToolParser._parse_xml_function_call), [`_parse_xml_function_call`](tools/parsers/seed_oss_tool_parser.md#SeedOssToolParser._parse_xml_function_call), [`_parse_single_invoke`](tools/parsers/minimax_m2_tool_parser.md#MinimaxM2ToolParser._parse_single_invoke), [`_handle_single_tool`](tools/parsers/llama4_pythonic_tool_parser.md#_handle_single_tool), [`_handle_single_tool`](tools/parsers/olmo3_tool_parser.md#_handle_single_tool), [`_handle_single_tool`](tools/parsers/pythonic_tool_parser.md#_handle_single_tool)  (+4 more)

### `ToolChoiceFunction`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:249`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L249)
- doc: Typed tool-choice selector for OpenAI-compatible requests.
- signature: `class ToolChoiceFunction(OpenAIBaseModel):`
- members:
  - `name` — [`L252`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L252)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`ToolChoiceOption`](openai_api_modules.md#ToolChoiceOption)

### `ToolChoiceOption`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:255`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L255)
- doc: Typed tool-choice object preserving OpenAI wire compatibility.
- signature: `class ToolChoiceOption(OpenAIBaseModel):`
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`ToolChoiceFunction`](openai_api_modules.md#ToolChoiceFunction)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`tool_choice`](openai_api_modules.md#ChatCompletionRequest.tool_choice), [`tool_choice`](typed_models.md#ResponsesFinalizationOptions.tool_choice), [`tool_choice`](openai_api_modules.md#ResponsesRequest.tool_choice), [`tool_choice`](typed_models.md#ResponsesResponse.tool_choice)

### `ToolDefinition`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:237`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L237)
- doc: Defines a tool that can be called by the model.
- signature: `class ToolDefinition(OpenAIBaseModel):`
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`FunctionDefinition`](openai_api_modules.md#FunctionDefinition)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`tools`](openai_api_modules.md#ChatCompletionRequest.tools), [`_parse_xml_function_call`](tools/parsers/qwen3coder_tool_parser.md#Qwen3CoderToolParser._parse_xml_function_call), [`_parse_xml_function_call`](tools/parsers/seed_oss_tool_parser.md#SeedOssToolParser._parse_xml_function_call), [`_extract_responses_tools`](inference_engine_interface.md#BaseInferenceApiServer._extract_responses_tools), [`_get_arguments_config`](tools/parsers/qwen3coder_tool_parser.md#Qwen3CoderToolParser._get_arguments_config), [`_is_string_type`](tools/parsers/glm4_moe_tool_parser.md#Glm4MoeModelToolParser._is_string_type), [`tools`](typed_models.md#ResponsesFinalizationOptions.tools), [`tools`](typed_models.md#ResponsesResponse.tools), [`tools`](openai_api_modules.md#ResponsesRequest.tools)

### `UsageInfo`  ·  implements/extends OpenAIBaseModel
- def: [`easydel/inference/openai_api_modules.py:195`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L195)
- doc: Token usage and performance metrics.
- signature: `class UsageInfo(OpenAIBaseModel):`
- members:
  - `completion_tokens` — [`L209`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L209)
  - `processing_time` — [`L212`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L212)
  - `prompt_tokens` — [`L208`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L208)
  - `tokens_per_second` — [`L211`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L211)
  - `total_tokens` — [`L210`](../../../../../../raw/code/EasyDeL/easydel/inference/openai_api_modules.py#L210)
- uses (calls/refs, reference-scoped): [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel)
- used by: [`OpenAIBaseModel`](openai_api_modules.md#OpenAIBaseModel), [`generate_stream`](esurge/server/api_server.md#eSurgeApiServer.generate_stream), [`_build_chat_completion_response`](esurge/server/api_server.md#eSurgeApiServer._build_chat_completion_response), [`iter_chat_completion_stream_responses`](stream_protocol.md#iter_chat_completion_stream_responses), [`_handle_completion_response`](esurge/server/api_server.md#eSurgeApiServer._handle_completion_response), [`_create_standard_response`](esurge/server/api_server.md#eSurgeApiServer._create_standard_response), [`usage`](openai_api_modules.md#ChatCompletionStreamResponse.usage), [`usage`](openai_api_modules.md#ChatCompletionResponse.usage), [`usage`](openai_api_modules.md#CompletionResponse.usage), [`usage`](openai_api_modules.md#CompletionStreamResponse.usage)

