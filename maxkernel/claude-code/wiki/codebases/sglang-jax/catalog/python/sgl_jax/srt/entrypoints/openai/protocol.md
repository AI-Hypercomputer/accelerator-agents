---
title: 'Module: python/sgl_jax/srt/entrypoints/openai/protocol.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/entrypoints/openai/protocol.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.entrypoints.openai.protocol`/
symbols:
  Tool: Tool#
  OpenAIServingRequest: OpenAIServingRequest.
  ToolChoice: ToolChoice#
  ChatCompletionMessageContentPart: ChatCompletionMessageContentPart.
  Function.name: Function#name.
  Function.parameters: Function#parameters.
  UsageInfo: UsageInfo#
  ChatCompletionMessageParam: ChatCompletionMessageParam.
  ChatCompletionRequest.response_format: ChatCompletionRequest#response_format.
  ChatCompletionResponseChoice.logprobs: ChatCompletionResponseChoice#logprobs.
  ChatCompletionResponseStreamChoice.logprobs: ChatCompletionResponseStreamChoice#logprobs.
  ChoiceLogprobs: ChoiceLogprobs#
  LogProbs: LogProbs#
  StructuresResponseFormat: StructuresResponseFormat#
  StructuralTagResponseFormat: StructuralTagResponseFormat#
  ChatCompletionMessageContentVideoPart: ChatCompletionMessageContentVideoPart#
  ChatCompletionMessageContentAudioPart: ChatCompletionMessageContentAudioPart#
  ChatCompletionMessageUserParam: ChatCompletionMessageUserParam#
  ResponseFormat: ResponseFormat#
  StructuralTagResponseFormat.structures: StructuralTagResponseFormat#structures.
  EmbeddingInput: EmbeddingInput.
  ToolCall: ToolCall#
  ModelList.data: ModelList#data.
  ChatCompletionTokenLogprob.top_logprobs: ChatCompletionTokenLogprob#top_logprobs.
  CompletionRequest.stream_options: CompletionRequest#stream_options.
  CompletionResponseChoice.logprobs: CompletionResponseChoice#logprobs.
  CompletionResponse.choices: CompletionResponse#choices.
  CompletionResponse.usage: CompletionResponse#usage.
  CompletionResponseStreamChoice.logprobs: CompletionResponseStreamChoice#logprobs.
  CompletionStreamResponse.choices: CompletionStreamResponse#choices.
  CompletionStreamResponse.usage: CompletionStreamResponse#usage.
  ChatCompletionMessageContentImagePart.image_url: ChatCompletionMessageContentImagePart#image_url.
  ToolCall.index: ToolCall#index.
  ChatCompletionMessageGenericParam.content: ChatCompletionMessageGenericParam#content.
  ChatCompletionMessageGenericParam.tool_calls: ChatCompletionMessageGenericParam#tool_calls.
  ChatCompletionRequest.messages: ChatCompletionRequest#messages.
  ChatCompletionRequest.stream_options: ChatCompletionRequest#stream_options.
  ChatCompletionRequest.tools: ChatCompletionRequest#tools.
  ChatCompletionRequest.tool_choice: ChatCompletionRequest#tool_choice.
  ChatMessage.tool_calls: ChatMessage#tool_calls.
  ChatCompletionResponseChoice.message: ChatCompletionResponseChoice#message.
  ChatCompletionResponse.choices: ChatCompletionResponse#choices.
  ChatCompletionResponse.usage: ChatCompletionResponse#usage.
  DeltaMessage.tool_calls: DeltaMessage#tool_calls.
  ChatCompletionResponseStreamChoice.delta: ChatCompletionResponseStreamChoice#delta.
  ChatCompletionStreamResponse.choices: ChatCompletionStreamResponse#choices.
  ChatCompletionStreamResponse.usage: ChatCompletionStreamResponse#usage.
  EmbeddingRequest.input: EmbeddingRequest#input.
  EmbeddingResponse.data: EmbeddingResponse#data.
  EmbeddingResponse.usage: EmbeddingResponse#usage.
  ScoringResponse.usage: ScoringResponse#usage.
  StreamOptions: StreamOptions#
  ChatCompletionMessageContentTextPart: ChatCompletionMessageContentTextPart#
  Function.strict: Function#strict.
  ToolChoiceFuncName.name: ToolChoiceFuncName#name.
  ModelCard: ModelCard#
  TopLogprob: TopLogprob#
  ChatCompletionTokenLogprob: ChatCompletionTokenLogprob#
  JsonSchemaResponseFormat: JsonSchemaResponseFormat#
  CompletionRequest: CompletionRequest#
  CompletionResponseChoice: CompletionResponseChoice#
  CompletionResponseStreamChoice: CompletionResponseStreamChoice#
  ChatCompletionMessageContentImageURL: ChatCompletionMessageContentImageURL#
  ChatCompletionMessageContentVideoURL: ChatCompletionMessageContentVideoURL#
  ChatCompletionMessageContentAudioURL: ChatCompletionMessageContentAudioURL#
  ChatCompletionMessageContentImagePart: ChatCompletionMessageContentImagePart#
  FunctionResponse: FunctionResponse#
  ChatCompletionMessageGenericParam: ChatCompletionMessageGenericParam#
  StructuresResponseFormat.begin: StructuresResponseFormat#begin.
  StructuresResponseFormat.end: StructuresResponseFormat#end.
  StructuralTagResponseFormat.triggers: StructuralTagResponseFormat#triggers.
  Function: Function#
  ToolChoiceFuncName: ToolChoiceFuncName#
  ChatCompletionRequest: ChatCompletionRequest#
  ChatMessage: ChatMessage#
  ChatCompletionResponseChoice: ChatCompletionResponseChoice#
  DeltaMessage: DeltaMessage#
  ChatCompletionResponseStreamChoice: ChatCompletionResponseStreamChoice#
  MultimodalEmbeddingInput: MultimodalEmbeddingInput#
  EmbeddingRequest: EmbeddingRequest#
  EmbeddingObject: EmbeddingObject#
  ScoringRequest: ScoringRequest#
  V1RerankReqInput: V1RerankReqInput#
  ModelCard.id: ModelCard#id.
  ModelCard.created: ModelCard#created.
  ModelCard.owned_by: ModelCard#owned_by.
  ModelCard.root: ModelCard#root.
  ModelCard.max_model_len: ModelCard#max_model_len.
  ModelList: ModelList#
  ErrorResponse: ErrorResponse#
  ErrorResponse.message: ErrorResponse#message.
  ErrorResponse.param: ErrorResponse#param.
  ErrorResponse.code: ErrorResponse#code.
  LogProbs.text_offset: LogProbs#text_offset.
  LogProbs.token_logprobs: LogProbs#token_logprobs.
  LogProbs.tokens: LogProbs#tokens.
  LogProbs.top_logprobs: LogProbs#top_logprobs.
  TopLogprob.token: TopLogprob#token.
  TopLogprob.logprob: TopLogprob#logprob.
  ChatCompletionTokenLogprob.token: ChatCompletionTokenLogprob#token.
  ChatCompletionTokenLogprob.logprob: ChatCompletionTokenLogprob#logprob.
  ChoiceLogprobs.content: ChoiceLogprobs#content.
  UsageInfo.prompt_tokens: UsageInfo#prompt_tokens.
  UsageInfo.total_tokens: UsageInfo#total_tokens.
  UsageInfo.completion_tokens: UsageInfo#completion_tokens.
  UsageInfo.prompt_tokens_details: UsageInfo#prompt_tokens_details.
  StreamOptions.include_usage: StreamOptions#include_usage.
  JsonSchemaResponseFormat.name: JsonSchemaResponseFormat#name.
  JsonSchemaResponseFormat.description: JsonSchemaResponseFormat#description.
  JsonSchemaResponseFormat.schema_: JsonSchemaResponseFormat#schema_.
  JsonSchemaResponseFormat.strict: JsonSchemaResponseFormat#strict.
  FileRequest: FileRequest#
  FileRequest.file: FileRequest#file.
  FileRequest.purpose: FileRequest#purpose.
  FileResponse: FileResponse#
  FileResponse.id: FileResponse#id.
  FileResponse.created_at: FileResponse#created_at.
  FileResponse.filename: FileResponse#filename.
  FileResponse.purpose: FileResponse#purpose.
  FileDeleteResponse: FileDeleteResponse#
  FileDeleteResponse.id: FileDeleteResponse#id.
  FileDeleteResponse.deleted: FileDeleteResponse#deleted.
  BatchRequest: BatchRequest#
  BatchRequest.input_file_id: BatchRequest#input_file_id.
  BatchRequest.endpoint: BatchRequest#endpoint.
  BatchRequest.completion_window: BatchRequest#completion_window.
  BatchRequest.metadata: BatchRequest#metadata.
  BatchResponse: BatchResponse#
  BatchResponse.id: BatchResponse#id.
  BatchResponse.endpoint: BatchResponse#endpoint.
  BatchResponse.errors: BatchResponse#errors.
  BatchResponse.input_file_id: BatchResponse#input_file_id.
  BatchResponse.completion_window: BatchResponse#completion_window.
  BatchResponse.status: BatchResponse#status.
  BatchResponse.output_file_id: BatchResponse#output_file_id.
  BatchResponse.error_file_id: BatchResponse#error_file_id.
  BatchResponse.created_at: BatchResponse#created_at.
  BatchResponse.in_progress_at: BatchResponse#in_progress_at.
  BatchResponse.expires_at: BatchResponse#expires_at.
  BatchResponse.finalizing_at: BatchResponse#finalizing_at.
  BatchResponse.completed_at: BatchResponse#completed_at.
  BatchResponse.failed_at: BatchResponse#failed_at.
  BatchResponse.expired_at: BatchResponse#expired_at.
  BatchResponse.cancelling_at: BatchResponse#cancelling_at.
  BatchResponse.cancelled_at: BatchResponse#cancelled_at.
  BatchResponse.request_counts: BatchResponse#request_counts.
  BatchResponse.metadata: BatchResponse#metadata.
  CompletionRequest.model: CompletionRequest#model.
  CompletionRequest.prompt: CompletionRequest#prompt.
  CompletionRequest.best_of: CompletionRequest#best_of.
  CompletionRequest.echo: CompletionRequest#echo.
  CompletionRequest.frequency_penalty: CompletionRequest#frequency_penalty.
  CompletionRequest.logit_bias: CompletionRequest#logit_bias.
  CompletionRequest.logprobs: CompletionRequest#logprobs.
  CompletionRequest.max_tokens: CompletionRequest#max_tokens.
  CompletionRequest.n: CompletionRequest#n.
  CompletionRequest.presence_penalty: CompletionRequest#presence_penalty.
  CompletionRequest.seed: CompletionRequest#seed.
  CompletionRequest.stop: CompletionRequest#stop.
  CompletionRequest.stream: CompletionRequest#stream.
  CompletionRequest.suffix: CompletionRequest#suffix.
  CompletionRequest.temperature: CompletionRequest#temperature.
  CompletionRequest.top_p: CompletionRequest#top_p.
  CompletionRequest.user: CompletionRequest#user.
  CompletionRequest.return_hidden_states: CompletionRequest#return_hidden_states.
  CompletionRequest.top_k: CompletionRequest#top_k.
  CompletionRequest.min_p: CompletionRequest#min_p.
  CompletionRequest.min_tokens: CompletionRequest#min_tokens.
  CompletionRequest.json_schema: CompletionRequest#json_schema.
  CompletionRequest.regex: CompletionRequest#regex.
  CompletionRequest.ebnf: CompletionRequest#ebnf.
  CompletionRequest.repetition_penalty: CompletionRequest#repetition_penalty.
  CompletionRequest.stop_token_ids: CompletionRequest#stop_token_ids.
  CompletionRequest.no_stop_trim: CompletionRequest#no_stop_trim.
  CompletionRequest.ignore_eos: CompletionRequest#ignore_eos.
  CompletionRequest.skip_special_tokens: CompletionRequest#skip_special_tokens.
  CompletionRequest.lora_path: CompletionRequest#lora_path.
  CompletionRequest.extra_key: CompletionRequest#extra_key.
  CompletionRequest.session_params: CompletionRequest#session_params.
  CompletionRequest.bootstrap_host: CompletionRequest#bootstrap_host.
  CompletionRequest.bootstrap_port: CompletionRequest#bootstrap_port.
  CompletionRequest.bootstrap_room: CompletionRequest#bootstrap_room.
  CompletionRequest.disagg_transfer_id: CompletionRequest#disagg_transfer_id.
  CompletionRequest.rid: CompletionRequest#rid.
  CompletionRequest.return_routed_experts: CompletionRequest#return_routed_experts.
  CompletionRequest.validate_max_tokens_positive: CompletionRequest#validate_max_tokens_positive().
  CompletionResponseChoice.index: CompletionResponseChoice#index.
  CompletionResponseChoice.text: CompletionResponseChoice#text.
  CompletionResponseChoice.finish_reason: CompletionResponseChoice#finish_reason.
  CompletionResponseChoice.matched_stop: CompletionResponseChoice#matched_stop.
  CompletionResponseChoice.hidden_states: CompletionResponseChoice#hidden_states.
  CompletionResponseChoice.routed_experts: CompletionResponseChoice#routed_experts.
  CompletionResponse: CompletionResponse#
  CompletionResponse.id: CompletionResponse#id.
  CompletionResponse.created: CompletionResponse#created.
  CompletionResponse.model: CompletionResponse#model.
  CompletionResponseStreamChoice.index: CompletionResponseStreamChoice#index.
  CompletionResponseStreamChoice.text: CompletionResponseStreamChoice#text.
  CompletionResponseStreamChoice.finish_reason: CompletionResponseStreamChoice#finish_reason.
  CompletionResponseStreamChoice.matched_stop: CompletionResponseStreamChoice#matched_stop.
  CompletionResponseStreamChoice.hidden_states: CompletionResponseStreamChoice#hidden_states.
  CompletionStreamResponse: CompletionStreamResponse#
  CompletionStreamResponse.id: CompletionStreamResponse#id.
  CompletionStreamResponse.created: CompletionStreamResponse#created.
  CompletionStreamResponse.model: CompletionStreamResponse#model.
  ChatCompletionMessageContentTextPart.text: ChatCompletionMessageContentTextPart#text.
  ChatCompletionMessageContentImageURL.url: ChatCompletionMessageContentImageURL#url.
  ChatCompletionMessageContentImageURL.detail: ChatCompletionMessageContentImageURL#detail.
  ChatCompletionMessageContentVideoURL.url: ChatCompletionMessageContentVideoURL#url.
  ChatCompletionMessageContentAudioURL.url: ChatCompletionMessageContentAudioURL#url.
  ChatCompletionMessageContentImagePart.modalities: ChatCompletionMessageContentImagePart#modalities.
  ChatCompletionMessageContentVideoPart.video_url: ChatCompletionMessageContentVideoPart#video_url.
  ChatCompletionMessageContentAudioPart.audio_url: ChatCompletionMessageContentAudioPart#audio_url.
  FunctionResponse.name: FunctionResponse#name.
  FunctionResponse.arguments: FunctionResponse#arguments.
  ToolCall.id: ToolCall#id.
  ChatCompletionMessageGenericParam.role: ChatCompletionMessageGenericParam#role.
  ChatCompletionMessageGenericParam.tool_call_id: ChatCompletionMessageGenericParam#tool_call_id.
  ChatCompletionMessageGenericParam.name: ChatCompletionMessageGenericParam#name.
  ChatCompletionMessageGenericParam.reasoning_content: ChatCompletionMessageGenericParam#reasoning_content.
  ChatCompletionMessageGenericParam._normalize_role: ChatCompletionMessageGenericParam#_normalize_role().
  ChatCompletionMessageUserParam.role: ChatCompletionMessageUserParam#role.
  ChatCompletionMessageUserParam.content: ChatCompletionMessageUserParam#content.
  ResponseFormat.json_schema: ResponseFormat#json_schema.
  StructuresResponseFormat.schema_: StructuresResponseFormat#schema_.
  Function.description: Function#description.
  ChatCompletionRequest.model: ChatCompletionRequest#model.
  ChatCompletionRequest.frequency_penalty: ChatCompletionRequest#frequency_penalty.
  ChatCompletionRequest.logit_bias: ChatCompletionRequest#logit_bias.
  ChatCompletionRequest.logprobs: ChatCompletionRequest#logprobs.
  ChatCompletionRequest.top_logprobs: ChatCompletionRequest#top_logprobs.
  ChatCompletionRequest.max_tokens: ChatCompletionRequest#max_tokens.
  ChatCompletionRequest.max_completion_tokens: ChatCompletionRequest#max_completion_tokens.
  ChatCompletionRequest.n: ChatCompletionRequest#n.
  ChatCompletionRequest.presence_penalty: ChatCompletionRequest#presence_penalty.
  ChatCompletionRequest.seed: ChatCompletionRequest#seed.
  ChatCompletionRequest.stop: ChatCompletionRequest#stop.
  ChatCompletionRequest.stream: ChatCompletionRequest#stream.
  ChatCompletionRequest.temperature: ChatCompletionRequest#temperature.
  ChatCompletionRequest.top_p: ChatCompletionRequest#top_p.
  ChatCompletionRequest.user: ChatCompletionRequest#user.
  ChatCompletionRequest.return_hidden_states: ChatCompletionRequest#return_hidden_states.
  ChatCompletionRequest.set_tool_choice_default: ChatCompletionRequest#set_tool_choice_default().
  ChatCompletionRequest.top_k: ChatCompletionRequest#top_k.
  ChatCompletionRequest.min_p: ChatCompletionRequest#min_p.
  ChatCompletionRequest.min_tokens: ChatCompletionRequest#min_tokens.
  ChatCompletionRequest.regex: ChatCompletionRequest#regex.
  ChatCompletionRequest.ebnf: ChatCompletionRequest#ebnf.
  ChatCompletionRequest.repetition_penalty: ChatCompletionRequest#repetition_penalty.
  ChatCompletionRequest.stop_token_ids: ChatCompletionRequest#stop_token_ids.
  ChatCompletionRequest.no_stop_trim: ChatCompletionRequest#no_stop_trim.
  ChatCompletionRequest.ignore_eos: ChatCompletionRequest#ignore_eos.
  ChatCompletionRequest.continue_final_message: ChatCompletionRequest#continue_final_message.
  ChatCompletionRequest.skip_special_tokens: ChatCompletionRequest#skip_special_tokens.
  ChatCompletionRequest.lora_path: ChatCompletionRequest#lora_path.
  ChatCompletionRequest.extra_key: ChatCompletionRequest#extra_key.
  ChatCompletionRequest.session_params: ChatCompletionRequest#session_params.
  ChatCompletionRequest.separate_reasoning: ChatCompletionRequest#separate_reasoning.
  ChatCompletionRequest.stream_reasoning: ChatCompletionRequest#stream_reasoning.
  ChatCompletionRequest.chat_template_kwargs: ChatCompletionRequest#chat_template_kwargs.
  ChatCompletionRequest.rid: ChatCompletionRequest#rid.
  ChatCompletionRequest.bootstrap_host: ChatCompletionRequest#bootstrap_host.
  ChatCompletionRequest.bootstrap_port: ChatCompletionRequest#bootstrap_port.
  ChatCompletionRequest.bootstrap_room: ChatCompletionRequest#bootstrap_room.
  ChatCompletionRequest.disagg_transfer_id: ChatCompletionRequest#disagg_transfer_id.
  ChatMessage.role: ChatMessage#role.
  ChatMessage.content: ChatMessage#content.
  ChatMessage.reasoning_content: ChatMessage#reasoning_content.
  ChatCompletionResponseChoice.index: ChatCompletionResponseChoice#index.
  ChatCompletionResponseChoice.finish_reason: ChatCompletionResponseChoice#finish_reason.
  ChatCompletionResponseChoice.matched_stop: ChatCompletionResponseChoice#matched_stop.
  ChatCompletionResponseChoice.hidden_states: ChatCompletionResponseChoice#hidden_states.
  ChatCompletionResponse: ChatCompletionResponse#
  ChatCompletionResponse.id: ChatCompletionResponse#id.
  ChatCompletionResponse.created: ChatCompletionResponse#created.
  ChatCompletionResponse.model: ChatCompletionResponse#model.
  DeltaMessage.role: DeltaMessage#role.
  DeltaMessage.content: DeltaMessage#content.
  DeltaMessage.reasoning_content: DeltaMessage#reasoning_content.
  DeltaMessage.hidden_states: DeltaMessage#hidden_states.
  ChatCompletionResponseStreamChoice.index: ChatCompletionResponseStreamChoice#index.
  ChatCompletionResponseStreamChoice.finish_reason: ChatCompletionResponseStreamChoice#finish_reason.
  ChatCompletionResponseStreamChoice.matched_stop: ChatCompletionResponseStreamChoice#matched_stop.
  ChatCompletionStreamResponse: ChatCompletionStreamResponse#
  ChatCompletionStreamResponse.id: ChatCompletionStreamResponse#id.
  ChatCompletionStreamResponse.created: ChatCompletionStreamResponse#created.
  ChatCompletionStreamResponse.model: ChatCompletionStreamResponse#model.
  MultimodalEmbeddingInput.text: MultimodalEmbeddingInput#text.
  MultimodalEmbeddingInput.image: MultimodalEmbeddingInput#image.
  EmbeddingRequest.model: EmbeddingRequest#model.
  EmbeddingRequest.encoding_format: EmbeddingRequest#encoding_format.
  EmbeddingRequest.dimensions: EmbeddingRequest#dimensions.
  EmbeddingRequest.user: EmbeddingRequest#user.
  EmbeddingRequest.rid: EmbeddingRequest#rid.
  EmbeddingObject.embedding: EmbeddingObject#embedding.
  EmbeddingObject.index: EmbeddingObject#index.
  EmbeddingResponse: EmbeddingResponse#
  EmbeddingResponse.model: EmbeddingResponse#model.
  ScoringRequest.query: ScoringRequest#query.
  ScoringRequest.items: ScoringRequest#items.
  ScoringRequest.label_token_ids: ScoringRequest#label_token_ids.
  ScoringRequest.apply_softmax: ScoringRequest#apply_softmax.
  ScoringRequest.item_first: ScoringRequest#item_first.
  ScoringRequest.model: ScoringRequest#model.
  ScoringResponse: ScoringResponse#
  ScoringResponse.scores: ScoringResponse#scores.
  ScoringResponse.model: ScoringResponse#model.
  V1RerankReqInput.query: V1RerankReqInput#query.
  V1RerankReqInput.documents: V1RerankReqInput#documents.
  RerankResponse: RerankResponse#
  RerankResponse.score: RerankResponse#score.
  RerankResponse.document: RerankResponse#document.
  RerankResponse.index: RerankResponse#index.
  RerankResponse.meta_info: RerankResponse#meta_info.
  MessageProcessingResult: MessageProcessingResult#
  MessageProcessingResult.prompt: MessageProcessingResult#prompt.
  MessageProcessingResult.prompt_ids: MessageProcessingResult#prompt_ids.
  MessageProcessingResult.image_data: MessageProcessingResult#image_data.
  MessageProcessingResult.audio_data: MessageProcessingResult#audio_data.
  MessageProcessingResult.video_data: MessageProcessingResult#video_data.
  MessageProcessingResult.modalities: MessageProcessingResult#modalities.
  MessageProcessingResult.stop: MessageProcessingResult#stop.
  MessageProcessingResult.tool_call_constraint: MessageProcessingResult#tool_call_constraint.
---
# Module: [`python/sgl_jax/srt/entrypoints/openai/protocol.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py)

## Classes
### `BatchRequest`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:102`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L102)
- signature: `class BatchRequest(BaseModel):`
- members:
  - `completion_window` — [`L105`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L105)
  - `endpoint` — [`L104`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L104)
  - `input_file_id` — [`L103`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L103)
  - `metadata` — [`L106`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L106)

### `BatchResponse`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:109`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L109)
- signature: `class BatchResponse(BaseModel):`
- members:
  - `cancelled_at` — [`L127`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L127)
  - `cancelling_at` — [`L126`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L126)
  - `completed_at` — [`L123`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L123)
  - `completion_window` — [`L115`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L115)
  - `created_at` — [`L119`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L119)
  - `endpoint` — [`L112`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L112)
  - `error_file_id` — [`L118`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L118)
  - `errors` — [`L113`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L113)
  - `expired_at` — [`L125`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L125)
  - `expires_at` — [`L121`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L121)
  - `failed_at` — [`L124`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L124)
  - `finalizing_at` — [`L122`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L122)
  - `id` — [`L110`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L110)
  - `in_progress_at` — [`L120`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L120)
  - `input_file_id` — [`L114`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L114)
  - `metadata` — [`L129`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L129)
  - `output_file_id` — [`L117`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L117)
  - `request_counts` — [`L128`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L128)
  - `status` — [`L116`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L116)

### `ChatCompletionMessageContentAudioPart`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:270`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L270)
- signature: `class ChatCompletionMessageContentAudioPart(BaseModel):`
- members:
  - `audio_url` — [`L272`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L272)
- uses (calls/refs, reference-scoped): [`ChatCompletionMessageContentAudioURL`](protocol.md#ChatCompletionMessageContentAudioURL)
- used by: [`ChatCompletionMessageContentPart`](protocol.md#ChatCompletionMessageContentPart)

### `ChatCompletionMessageContentAudioURL`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:255`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L255)
- signature: `class ChatCompletionMessageContentAudioURL(BaseModel):`
- members:
  - `url` — [`L256`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L256)
- used by: [`ChatCompletionMessageContentAudioPart`](protocol.md#ChatCompletionMessageContentAudioPart)

### `ChatCompletionMessageContentImagePart`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:259`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L259)
- signature: `class ChatCompletionMessageContentImagePart(BaseModel):`
- members:
  - `image_url` — [`L261`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L261)
  - `modalities` — [`L262`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L262)
- uses (calls/refs, reference-scoped): [`ChatCompletionMessageContentImageURL`](protocol.md#ChatCompletionMessageContentImageURL)
- used by: [`ChatCompletionMessageContentPart`](protocol.md#ChatCompletionMessageContentPart)

### `ChatCompletionMessageContentImageURL`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:246`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L246)
- signature: `class ChatCompletionMessageContentImageURL(BaseModel):`
- members:
  - `detail` — [`L248`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L248)
  - `url` — [`L247`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L247)
- used by: [`image_url`](protocol.md#ChatCompletionMessageContentImagePart.image_url)

### `ChatCompletionMessageContentTextPart`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:241`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L241)
- signature: `class ChatCompletionMessageContentTextPart(BaseModel):`
- members:
  - `text` — [`L243`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L243)
- used by: [`ChatCompletionMessageContentPart`](protocol.md#ChatCompletionMessageContentPart), [`content`](protocol.md#ChatCompletionMessageGenericParam.content)

### `ChatCompletionMessageContentVideoPart`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:265`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L265)
- signature: `class ChatCompletionMessageContentVideoPart(BaseModel):`
- members:
  - `video_url` — [`L267`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L267)
- uses (calls/refs, reference-scoped): [`ChatCompletionMessageContentVideoURL`](protocol.md#ChatCompletionMessageContentVideoURL)
- used by: [`ChatCompletionMessageContentPart`](protocol.md#ChatCompletionMessageContentPart)

### `ChatCompletionMessageContentVideoURL`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:251`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L251)
- signature: `class ChatCompletionMessageContentVideoURL(BaseModel):`
- members:
  - `url` — [`L252`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L252)
- used by: [`ChatCompletionMessageContentVideoPart`](protocol.md#ChatCompletionMessageContentVideoPart)

### `ChatCompletionMessageGenericParam`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:299`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L299)
- signature: `class ChatCompletionMessageGenericParam(BaseModel):`
- members:
  - `content` — [`L301`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L301)
  - `name` — [`L303`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L303)
  - `reasoning_content` — [`L304`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L304)
  - `role` — [`L300`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L300)
  - `tool_call_id` — [`L302`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L302)
  - `tool_calls` — [`L305`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L305)
- protocol/private: `_normalize_role`[`L309`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L309)
- uses (calls/refs, reference-scoped): [`ToolCall`](protocol.md#ToolCall), [`ChatCompletionMessageContentTextPart`](protocol.md#ChatCompletionMessageContentTextPart)
- used by: [`ChatCompletionMessageParam`](protocol.md#ChatCompletionMessageParam)

### `ChatCompletionMessageUserParam`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:320`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L320)
- signature: `class ChatCompletionMessageUserParam(BaseModel):`
- members:
  - `content` — [`L322`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L322)
  - `role` — [`L321`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L321)
- uses (calls/refs, reference-scoped): [`ChatCompletionMessageContentPart`](protocol.md#ChatCompletionMessageContentPart)
- used by: [`ChatCompletionMessageParam`](protocol.md#ChatCompletionMessageParam)

### `ChatCompletionRequest`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:374`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L374)
- signature: `class ChatCompletionRequest(BaseModel):`
- members:
  - `set_tool_choice_default(cls, values)` — [`L411`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L411)
  - `bootstrap_host` — [`L442`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L442)
  - `bootstrap_port` — [`L443`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L443)
  - `bootstrap_room` — [`L444`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L444)
  - `chat_template_kwargs` — [`L436`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L436)
  - `continue_final_message` — [`L429`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L429)
  - `disagg_transfer_id` — [`L445`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L445)
  - `ebnf` — [`L424`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L424)
  - `extra_key` — [`L432`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L432)
  - `frequency_penalty` — [`L379`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L379)
  - `ignore_eos` — [`L428`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L428)
  - `logit_bias` — [`L380`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L380)
  - `logprobs` — [`L381`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L381)
  - `lora_path` — [`L431`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L431)
  - `max_completion_tokens` — [`L388`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L388)
  - `max_tokens` — [`L383`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L383)
  - `messages` — [`L377`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L377)
  - `min_p` — [`L421`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L421)
  - `min_tokens` — [`L422`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L422)
  - `model` — [`L378`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L378)
  - `n` — [`L393`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L393)
  - `no_stop_trim` — [`L427`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L427)
  - `presence_penalty` — [`L394`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L394)
  - `regex` — [`L423`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L423)
  - `repetition_penalty` — [`L425`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L425)
  - `response_format` — [`L395`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L395)
  - `return_hidden_states` — [`L407`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L407)
  - `rid` — [`L439`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L439)
  - `seed` — [`L396`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L396)
  - `separate_reasoning` — [`L434`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L434)
  - `session_params` — [`L433`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L433)
  - `skip_special_tokens` — [`L430`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L430)
  - `stop` — [`L397`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L397)
  - `stop_token_ids` — [`L426`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L426)
  - `stream` — [`L398`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L398)
  - `stream_options` — [`L399`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L399)
  - `stream_reasoning` — [`L435`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L435)
  - `temperature` — [`L400`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L400)
  - `tool_choice` — [`L404`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L404)
  - `tools` — [`L403`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L403)
  - `top_k` — [`L420`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L420)
  - `top_logprobs` — [`L382`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L382)
  - `top_p` — [`L401`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L401)
  - `user` — [`L402`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L402)
- uses (calls/refs, reference-scoped): [`Tool`](protocol.md#Tool), [`ToolChoice`](protocol.md#ToolChoice), [`ChatCompletionMessageParam`](protocol.md#ChatCompletionMessageParam), [`StructuralTagResponseFormat`](protocol.md#StructuralTagResponseFormat), [`ResponseFormat`](protocol.md#ResponseFormat), [`StreamOptions`](protocol.md#StreamOptions)
- used by: [`OpenAIServingRequest`](protocol.md#OpenAIServingRequest)

### `ChatCompletionResponse`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:473`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L473)
- signature: `class ChatCompletionResponse(BaseModel):`
- members:
  - `choices` — [`L478`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L478)
  - `created` — [`L476`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L476)
  - `id` — [`L474`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L474)
  - `model` — [`L477`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L477)
  - `usage` — [`L479`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L479)
- uses (calls/refs, reference-scoped): [`UsageInfo`](protocol.md#UsageInfo), [`ChatCompletionResponseChoice`](protocol.md#ChatCompletionResponseChoice)

### `ChatCompletionResponseChoice`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:455`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L455)
- signature: `class ChatCompletionResponseChoice(BaseModel):`
- members:
  - `finish_reason` — [`L459`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L459)
  - `hidden_states` — [`L463`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L463)
  - `index` — [`L456`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L456)
  - `logprobs` — [`L458`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L458)
  - `matched_stop` — [`L462`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L462)
  - `message` — [`L457`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L457)
- uses (calls/refs, reference-scoped): [`ChoiceLogprobs`](protocol.md#ChoiceLogprobs), [`LogProbs`](protocol.md#LogProbs), [`ChatMessage`](protocol.md#ChatMessage)
- used by: [`choices`](protocol.md#ChatCompletionResponse.choices)

### `ChatCompletionResponseStreamChoice`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:497`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L497)
- signature: `class ChatCompletionResponseStreamChoice(BaseModel):`
- members:
  - `delta` — [`L499`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L499)
  - `finish_reason` — [`L501`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L501)
  - `index` — [`L498`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L498)
  - `logprobs` — [`L500`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L500)
  - `matched_stop` — [`L504`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L504)
- uses (calls/refs, reference-scoped): [`ChoiceLogprobs`](protocol.md#ChoiceLogprobs), [`LogProbs`](protocol.md#LogProbs), [`DeltaMessage`](protocol.md#DeltaMessage)
- used by: [`choices`](protocol.md#ChatCompletionStreamResponse.choices)

### `ChatCompletionStreamResponse`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:507`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L507)
- signature: `class ChatCompletionStreamResponse(BaseModel):`
- members:
  - `choices` — [`L512`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L512)
  - `created` — [`L510`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L510)
  - `id` — [`L508`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L508)
  - `model` — [`L511`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L511)
  - `usage` — [`L513`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L513)
- uses (calls/refs, reference-scoped): [`UsageInfo`](protocol.md#UsageInfo), [`ChatCompletionResponseStreamChoice`](protocol.md#ChatCompletionResponseStreamChoice)

### `ChatCompletionTokenLogprob`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:49`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L49)
- signature: `class ChatCompletionTokenLogprob(BaseModel):`
- members:
  - `logprob` — [`L52`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L52)
  - `token` — [`L50`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L50)
  - `top_logprobs` — [`L53`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L53)
- uses (calls/refs, reference-scoped): [`TopLogprob`](protocol.md#TopLogprob)
- used by: [`ChoiceLogprobs`](protocol.md#ChoiceLogprobs)

### `ChatMessage`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:448`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L448)
- signature: `class ChatMessage(BaseModel):`
- members:
  - `content` — [`L450`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L450)
  - `reasoning_content` — [`L451`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L451)
  - `role` — [`L449`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L449)
  - `tool_calls` — [`L452`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L452)
- uses (calls/refs, reference-scoped): [`ToolCall`](protocol.md#ToolCall)
- used by: [`message`](protocol.md#ChatCompletionResponseChoice.message)

### `ChoiceLogprobs`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:56`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L56)
- signature: `class ChoiceLogprobs(BaseModel):`
- members:
  - `content` — [`L58`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L58)
- uses (calls/refs, reference-scoped): [`ChatCompletionTokenLogprob`](protocol.md#ChatCompletionTokenLogprob)
- used by: [`logprobs`](protocol.md#ChatCompletionResponseChoice.logprobs), [`logprobs`](protocol.md#ChatCompletionResponseStreamChoice.logprobs)

### `CompletionRequest`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:132`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L132)
- signature: `class CompletionRequest(BaseModel):`
- members:
  - `validate_max_tokens_positive(cls, v)` — [`L184`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L184)
  - `best_of` — [`L137`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L137)
  - `bootstrap_host` — [`L172`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L172)
  - `bootstrap_port` — [`L173`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L173)
  - `bootstrap_room` — [`L174`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L174)
  - `disagg_transfer_id` — [`L175`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L175)
  - `ebnf` — [`L161`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L161)
  - `echo` — [`L138`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L138)
  - `extra_key` — [`L168`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L168)
  - `frequency_penalty` — [`L139`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L139)
  - `ignore_eos` — [`L165`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L165)
  - `json_schema` — [`L159`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L159)
  - `logit_bias` — [`L140`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L140)
  - `logprobs` — [`L141`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L141)
  - `lora_path` — [`L167`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L167)
  - `max_tokens` — [`L142`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L142)
  - `min_p` — [`L157`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L157)
  - `min_tokens` — [`L158`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L158)
  - `model` — [`L135`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L135)
  - `n` — [`L143`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L143)
  - `no_stop_trim` — [`L164`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L164)
  - `presence_penalty` — [`L144`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L144)
  - `prompt` — [`L136`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L136)
  - `regex` — [`L160`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L160)
  - `repetition_penalty` — [`L162`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L162)
  - `return_hidden_states` — [`L153`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L153)
  - `return_routed_experts` — [`L180`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L180)
  - `rid` — [`L178`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L178)
  - `seed` — [`L145`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L145)
  - `session_params` — [`L169`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L169)
  - `skip_special_tokens` — [`L166`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L166)
  - `stop` — [`L146`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L146)
  - `stop_token_ids` — [`L163`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L163)
  - `stream` — [`L147`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L147)
  - `stream_options` — [`L148`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L148)
  - `suffix` — [`L149`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L149)
  - `temperature` — [`L150`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L150)
  - `top_k` — [`L156`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L156)
  - `top_p` — [`L151`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L151)
  - `user` — [`L152`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L152)
- uses (calls/refs, reference-scoped): [`StreamOptions`](protocol.md#StreamOptions)
- used by: [`OpenAIServingRequest`](protocol.md#OpenAIServingRequest)

### `CompletionResponse`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:207`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L207)
- signature: `class CompletionResponse(BaseModel):`
- members:
  - `choices` — [`L212`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L212)
  - `created` — [`L210`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L210)
  - `id` — [`L208`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L208)
  - `model` — [`L211`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L211)
  - `usage` — [`L213`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L213)
- uses (calls/refs, reference-scoped): [`UsageInfo`](protocol.md#UsageInfo), [`CompletionResponseChoice`](protocol.md#CompletionResponseChoice)

### `CompletionResponseChoice`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:190`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L190)
- signature: `class CompletionResponseChoice(BaseModel):`
- members:
  - `finish_reason` — [`L194`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L194)
  - `hidden_states` — [`L196`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L196)
  - `index` — [`L191`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L191)
  - `logprobs` — [`L193`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L193)
  - `matched_stop` — [`L195`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L195)
  - `routed_experts` — [`L197`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L197)
  - `text` — [`L192`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L192)
- uses (calls/refs, reference-scoped): [`LogProbs`](protocol.md#LogProbs)
- used by: [`choices`](protocol.md#CompletionResponse.choices)

### `CompletionResponseStreamChoice`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:216`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L216)
- signature: `class CompletionResponseStreamChoice(BaseModel):`
- members:
  - `finish_reason` — [`L220`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L220)
  - `hidden_states` — [`L222`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L222)
  - `index` — [`L217`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L217)
  - `logprobs` — [`L219`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L219)
  - `matched_stop` — [`L221`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L221)
  - `text` — [`L218`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L218)
- uses (calls/refs, reference-scoped): [`LogProbs`](protocol.md#LogProbs)
- used by: [`choices`](protocol.md#CompletionStreamResponse.choices)

### `CompletionStreamResponse`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:232`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L232)
- signature: `class CompletionStreamResponse(BaseModel):`
- members:
  - `choices` — [`L237`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L237)
  - `created` — [`L235`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L235)
  - `id` — [`L233`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L233)
  - `model` — [`L236`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L236)
  - `usage` — [`L238`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L238)
- uses (calls/refs, reference-scoped): [`UsageInfo`](protocol.md#UsageInfo), [`CompletionResponseStreamChoice`](protocol.md#CompletionResponseStreamChoice)

### `DeltaMessage`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:482`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L482)
- signature: `class DeltaMessage(BaseModel):`
- members:
  - `content` — [`L484`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L484)
  - `hidden_states` — [`L487`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L487)
  - `reasoning_content` — [`L485`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L485)
  - `role` — [`L483`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L483)
  - `tool_calls` — [`L486`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L486)
- uses (calls/refs, reference-scoped): [`ToolCall`](protocol.md#ToolCall)
- used by: [`delta`](protocol.md#ChatCompletionResponseStreamChoice.delta)

### `EmbeddingObject`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:537`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L537)
- signature: `class EmbeddingObject(BaseModel):`
- members:
  - `embedding` — [`L538`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L538)
  - `index` — [`L539`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L539)
- used by: [`data`](protocol.md#EmbeddingResponse.data)

### `EmbeddingRequest`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:524`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L524)
- signature: `class EmbeddingRequest(BaseModel):`
- members:
  - `dimensions` — [`L530`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L530)
  - `encoding_format` — [`L529`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L529)
  - `input` — [`L527`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L527)
  - `model` — [`L528`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L528)
  - `rid` — [`L534`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L534)
  - `user` — [`L531`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L531)
- uses (calls/refs, reference-scoped): [`EmbeddingInput`](protocol.md#EmbeddingInput)
- used by: [`OpenAIServingRequest`](protocol.md#OpenAIServingRequest)

### `EmbeddingResponse`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:543`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L543)
- signature: `class EmbeddingResponse(BaseModel):`
- members:
  - `data` — [`L544`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L544)
  - `model` — [`L545`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L545)
  - `usage` — [`L547`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L547)
- uses (calls/refs, reference-scoped): [`UsageInfo`](protocol.md#UsageInfo), [`EmbeddingObject`](protocol.md#EmbeddingObject)

### `ErrorResponse`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:28`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L28)
- signature: `class ErrorResponse(BaseModel):`
- members:
  - `code` — [`L33`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L33)
  - `message` — [`L30`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L30)
  - `param` — [`L32`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L32)

### `FileDeleteResponse`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:96`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L96)
- signature: `class FileDeleteResponse(BaseModel):`
- members:
  - `deleted` — [`L99`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L99)
  - `id` — [`L97`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L97)

### `FileRequest`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:81`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L81)
- signature: `class FileRequest(BaseModel):`
- members:
  - `file` — [`L83`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L83)
  - `purpose` — [`L84`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L84)

### `FileResponse`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:87`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L87)
- signature: `class FileResponse(BaseModel):`
- members:
  - `created_at` — [`L91`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L91)
  - `filename` — [`L92`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L92)
  - `id` — [`L88`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L88)
  - `purpose` — [`L93`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L93)

### `Function`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:345`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L345)
- doc: Function descriptions.
- signature: `class Function(BaseModel):`
- members:
  - `description` — [`L348`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L348)
  - `name` — [`L349`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L349)
  - `parameters` — [`L350`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L350)
  - `strict` — [`L351`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L351)
- used by: [`Tool`](protocol.md#Tool), [`get_structure_tag`](../../function_call/function_call_parser.md#FunctionCallParser.get_structure_tag), [`get_ebnf`](../../function_call/function_call_parser.md#FunctionCallParser.get_ebnf), [`get_structure_constraint`](../../function_call/function_call_parser.md#FunctionCallParser.get_structure_constraint), [`get_json_schema_constraint`](../../function_call/utils.md#get_json_schema_constraint), [`get_argument_type`](../../function_call/glm4_moe_detector.md#get_argument_type), [`_get_tool_indices`](../../function_call/base_format_detector.md#BaseFormatDetector._get_tool_indices), [`_get_tool_schema`](../../function_call/utils.md#_get_tool_schema), [`get_argument_type`](../../function_call/glm47_moe_detector.md#get_argument_type), [`_get_param_type`](../../function_call/mimo_detector.md#_get_param_type), [`_get_tool_schema_defs`](../../function_call/utils.md#_get_tool_schema_defs)

### `FunctionResponse`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:283`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L283)
- doc: Function response.
- signature: `class FunctionResponse(BaseModel):`
- members:
  - `arguments` — [`L287`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L287)
  - `name` — [`L286`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L286)
- used by: [`index`](protocol.md#ToolCall.index)

### `JsonSchemaResponseFormat`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:73`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L73)
- signature: `class JsonSchemaResponseFormat(BaseModel):`
- members:
  - `description` — [`L75`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L75)
  - `name` — [`L74`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L74)
  - `schema_` — [`L77`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L77)
  - `strict` — [`L78`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L78)
- used by: [`ResponseFormat`](protocol.md#ResponseFormat)

### `LogProbs`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:36`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L36)
- signature: `class LogProbs(BaseModel):`
- members:
  - `text_offset` — [`L37`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L37)
  - `token_logprobs` — [`L38`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L38)
  - `tokens` — [`L39`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L39)
  - `top_logprobs` — [`L40`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L40)
- used by: [`logprobs`](protocol.md#ChatCompletionResponseChoice.logprobs), [`logprobs`](protocol.md#ChatCompletionResponseStreamChoice.logprobs), [`logprobs`](protocol.md#CompletionResponseChoice.logprobs), [`logprobs`](protocol.md#CompletionResponseStreamChoice.logprobs)

### `MessageProcessingResult`
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:588`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L588)
- doc: Result of processing chat messages and applying templates.
- signature: `class MessageProcessingResult:`
- members:
  - `audio_data` — [`L608`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L608)
  - `image_data` — [`L607`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L607)
  - `modalities` — [`L610`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L610)
  - `prompt` — [`L605`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L605)
  - `prompt_ids` — [`L606`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L606)
  - `stop` — [`L611`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L611)
  - `tool_call_constraint` — [`L612`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L612)
  - `video_data` — [`L609`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L609)

### `ModelCard`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:10`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L10)
- doc: Model cards.
- signature: `class ModelCard(BaseModel):`
- members:
  - `created` — [`L15`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L15)
  - `id` — [`L13`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L13)
  - `max_model_len` — [`L18`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L18)
  - `owned_by` — [`L16`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L16)
  - `root` — [`L17`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L17)
- used by: [`data`](protocol.md#ModelList.data)

### `ModelList`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:21`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L21)
- doc: Model list consists of model cards.
- signature: `class ModelList(BaseModel):`
- members:
  - `data` — [`L25`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L25)
- uses (calls/refs, reference-scoped): [`ModelCard`](protocol.md#ModelCard)

### `MultimodalEmbeddingInput`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:516`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L516)
- signature: `class MultimodalEmbeddingInput(BaseModel):`
- members:
  - `image` — [`L518`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L518)
  - `text` — [`L517`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L517)
- used by: [`EmbeddingInput`](protocol.md#EmbeddingInput)

### `RerankResponse`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:575`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L575)
- signature: `class RerankResponse(BaseModel):`
- members:
  - `document` — [`L577`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L577)
  - `index` — [`L578`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L578)
  - `meta_info` — [`L579`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L579)
  - `score` — [`L576`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L576)

### `ResponseFormat`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:328`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L328)
- signature: `class ResponseFormat(BaseModel):`
- members:
  - `json_schema` — [`L330`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L330)
- uses (calls/refs, reference-scoped): [`JsonSchemaResponseFormat`](protocol.md#JsonSchemaResponseFormat)
- used by: [`response_format`](protocol.md#ChatCompletionRequest.response_format)

### `ScoringRequest`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:550`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L550)
- signature: `class ScoringRequest(BaseModel):`
- members:
  - `apply_softmax` — [`L556`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L556)
  - `item_first` — [`L557`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L557)
  - `items` — [`L552`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L552)
  - `label_token_ids` — [`L555`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L555)
  - `model` — [`L558`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L558)
  - `query` — [`L551`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L551)
- used by: [`OpenAIServingRequest`](protocol.md#OpenAIServingRequest)

### `ScoringResponse`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:561`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L561)
- signature: `class ScoringResponse(BaseModel):`
- members:
  - `model` — [`L565`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L565)
  - `scores` — [`L562`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L562)
  - `usage` — [`L566`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L566)
- uses (calls/refs, reference-scoped): [`UsageInfo`](protocol.md#UsageInfo)

### `StreamOptions`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:69`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L69)
- signature: `class StreamOptions(BaseModel):`
- members:
  - `include_usage` — [`L70`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L70)
- used by: [`stream_options`](protocol.md#ChatCompletionRequest.stream_options), [`stream_options`](protocol.md#CompletionRequest.stream_options)

### `StructuralTagResponseFormat`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:339`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L339)
- signature: `class StructuralTagResponseFormat(BaseModel):`
- members:
  - `structures` — [`L341`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L341)
  - `triggers` — [`L342`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L342)
- uses (calls/refs, reference-scoped): [`StructuresResponseFormat`](protocol.md#StructuresResponseFormat)
- used by: [`get_structure_tag`](../../function_call/function_call_parser.md#FunctionCallParser.get_structure_tag), [`response_format`](protocol.md#ChatCompletionRequest.response_format)

### `StructuresResponseFormat`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:333`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L333)
- signature: `class StructuresResponseFormat(BaseModel):`
- members:
  - `begin` — [`L334`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L334)
  - `end` — [`L336`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L336)
  - `schema_` — [`L335`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L335)
- used by: [`get_structure_tag`](../../function_call/function_call_parser.md#FunctionCallParser.get_structure_tag), [`structures`](protocol.md#StructuralTagResponseFormat.structures)

### `Tool`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:354`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L354)
- doc: Function wrapper.
- signature: `class Tool(BaseModel):`
- uses (calls/refs, reference-scoped): [`Function`](protocol.md#Function)
- used by: [`_process_xml_to_json_streaming`](../../function_call/glm47_moe_detector.md#Glm47MoeDetector._process_xml_to_json_streaming), [`_process_xml_to_json_streaming`](../../function_call/glm4_moe_detector.md#Glm4MoeDetector._process_xml_to_json_streaming), [`parse_streaming_increment`](../../function_call/base_format_detector.md#BaseFormatDetector.parse_streaming_increment), [`_finalize_tool_call`](../../function_call/glm47_moe_detector.md#Glm47MoeDetector._finalize_tool_call), [`detect_and_parse`](../../function_call/glm47_moe_detector.md#Glm47MoeDetector.detect_and_parse), [`detect_and_parse`](../../function_call/glm4_moe_detector.md#Glm4MoeDetector.detect_and_parse), [`_process_arguments_streaming`](../../function_call/glm47_moe_detector.md#Glm47MoeDetector._process_arguments_streaming), [`detect_and_parse`](../../function_call/mimo_detector.md#MiMoDetector.detect_and_parse), [`detect_and_parse`](../../function_call/base_format_detector.md#BaseFormatDetector.detect_and_parse), [`parse_base_json`](../../function_call/base_format_detector.md#BaseFormatDetector.parse_base_json), [`detect_and_parse`](../../function_call/qwen25_detector.md#Qwen25Detector.detect_and_parse), [`_parse_block`](../../function_call/qwen3_coder_detector.md#Qwen3CoderDetector._parse_block), [`parse_streaming_increment`](../../function_call/glm47_moe_detector.md#Glm47MoeDetector.parse_streaming_increment), [`parse_streaming_increment`](../../function_call/qwen25_detector.md#Qwen25Detector.parse_streaming_increment), [`get_json_schema_constraint`](../../function_call/utils.md#get_json_schema_constraint), [`build_ebnf`](../../function_call/base_format_detector.md#BaseFormatDetector.build_ebnf), [`parse_streaming_increment`](../../function_call/mimo_detector.md#MiMoDetector.parse_streaming_increment), [`_extract`](../../function_call/qwen3_coder_detector.md#Qwen3CoderDetector._extract), [`build_ebnf`](../../function_call/qwen25_detector.md#Qwen25Detector.build_ebnf), [`build_ebnf`](../../function_call/qwen3_coder_detector.md#Qwen3CoderDetector.build_ebnf), [`detect_and_parse`](../../function_call/qwen3_coder_detector.md#Qwen3CoderDetector.detect_and_parse), [`get_argument_type`](../../function_call/glm4_moe_detector.md#get_argument_type), [`_get_tool_indices`](../../function_call/base_format_detector.md#BaseFormatDetector._get_tool_indices), [`_parse_tool_call`](../../function_call/mimo_detector.md#MiMoDetector._parse_tool_call), [`parse_streaming_increment`](../../function_call/glm4_moe_detector.md#Glm4MoeDetector.parse_streaming_increment), [`parse_streaming_increment`](../../function_call/qwen3_coder_detector.md#Qwen3CoderDetector.parse_streaming_increment), [`_get_tool_schema`](../../function_call/utils.md#_get_tool_schema), [`_parse_argument_pairs`](../../function_call/glm47_moe_detector.md#Glm47MoeDetector._parse_argument_pairs), [`_parse_argument_pairs`](../../function_call/glm4_moe_detector.md#Glm4MoeDetector._parse_argument_pairs), [`get_argument_type`](../../function_call/glm47_moe_detector.md#get_argument_type), [`_convert_param_value`](../../function_call/mimo_detector.md#_convert_param_value), [`_get_param_type`](../../function_call/mimo_detector.md#_get_param_type), [`_get_value_type`](../../function_call/glm47_moe_detector.md#Glm47MoeDetector._get_value_type), [`_get_value_type`](../../function_call/glm4_moe_detector.md#Glm4MoeDetector._get_value_type), [`__init__`](../../function_call/function_call_parser.md#FunctionCallParser.__init__), [`_get_tool_schema_defs`](../../function_call/utils.md#_get_tool_schema_defs), [`build_ebnf`](../../function_call/glm47_moe_detector.md#Glm47MoeDetector.build_ebnf), [`build_ebnf`](../../function_call/glm4_moe_detector.md#Glm4MoeDetector.build_ebnf), [`build_ebnf`](../../function_call/mimo_detector.md#MiMoDetector.build_ebnf), [`tools`](protocol.md#ChatCompletionRequest.tools)

### `ToolCall`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:290`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L290)
- doc: Tool call response.
- signature: `class ToolCall(BaseModel):`
- members:
  - `id` — [`L293`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L293)
  - `index` — [`L294`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L294)
- uses (calls/refs, reference-scoped): [`FunctionResponse`](protocol.md#FunctionResponse)
- used by: [`tool_calls`](protocol.md#ChatCompletionMessageGenericParam.tool_calls), [`tool_calls`](protocol.md#ChatMessage.tool_calls), [`tool_calls`](protocol.md#DeltaMessage.tool_calls)

### `ToolChoice`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:367`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L367)
- doc: The tool choice definition.
- signature: `class ToolChoice(BaseModel):`
- uses (calls/refs, reference-scoped): [`ToolChoiceFuncName`](protocol.md#ToolChoiceFuncName)
- used by: [`get_ebnf`](../../function_call/function_call_parser.md#FunctionCallParser.get_ebnf), [`get_structure_constraint`](../../function_call/function_call_parser.md#FunctionCallParser.get_structure_constraint), [`get_json_schema_constraint`](../../function_call/utils.md#get_json_schema_constraint), [`tool_choice`](protocol.md#ChatCompletionRequest.tool_choice)

### `ToolChoiceFuncName`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:361`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L361)
- doc: The name of tool choice function.
- signature: `class ToolChoiceFuncName(BaseModel):`
- members:
  - `name` — [`L364`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L364)
- used by: [`get_ebnf`](../../function_call/function_call_parser.md#FunctionCallParser.get_ebnf), [`get_json_schema_constraint`](../../function_call/utils.md#get_json_schema_constraint), [`ToolChoice`](protocol.md#ToolChoice)

### `TopLogprob`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:43`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L43)
- signature: `class TopLogprob(BaseModel):`
- members:
  - `logprob` — [`L46`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L46)
  - `token` — [`L44`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L44)
- used by: [`top_logprobs`](protocol.md#ChatCompletionTokenLogprob.top_logprobs)

### `UsageInfo`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:61`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L61)
- signature: `class UsageInfo(BaseModel):`
- members:
  - `completion_tokens` — [`L64`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L64)
  - `prompt_tokens` — [`L62`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L62)
  - `prompt_tokens_details` — [`L66`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L66)
  - `total_tokens` — [`L63`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L63)
- used by: [`usage`](protocol.md#ChatCompletionResponse.usage), [`usage`](protocol.md#ChatCompletionStreamResponse.usage), [`usage`](protocol.md#CompletionResponse.usage), [`usage`](protocol.md#CompletionStreamResponse.usage), [`usage`](protocol.md#EmbeddingResponse.usage), [`usage`](protocol.md#ScoringResponse.usage)

### `V1RerankReqInput`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/entrypoints/openai/protocol.py:570`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L570)
- signature: `class V1RerankReqInput(BaseModel):`
- members:
  - `documents` — [`L572`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L572)
  - `query` — [`L571`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L571)
- used by: [`OpenAIServingRequest`](protocol.md#OpenAIServingRequest)

## Module values
- `ChatCompletionMessageContentPart` — [`L275`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L275)
- `ChatCompletionMessageParam` — [`L325`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L325)
- `EmbeddingInput` — [`L521`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L521)
- `OpenAIServingRequest` — [`L582`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/entrypoints/openai/protocol.py#L582)

