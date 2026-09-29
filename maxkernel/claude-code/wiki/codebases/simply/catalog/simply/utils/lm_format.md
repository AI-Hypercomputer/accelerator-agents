---
title: 'Module: simply/utils/lm_format.py'
type: catalog
provenance: extracted
module: simply/utils/lm_format.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.lm_format`/
symbols:
  LMFormatRegistry: LMFormatRegistry#
  LMFormat.format: LMFormat#format().
  QwenV2Chat.format: QwenV2Chat#format().
  SimplyV1Chat.format: SimplyV1Chat#format().
  GemmaV2Chat.format: GemmaV2Chat#format().
  SimplyV1Chat: SimplyV1Chat#
  QwenV2Chat: QwenV2Chat#
  LMFormat: LMFormat#
  Pretrain: Pretrain#
  GemmaV2Chat: GemmaV2Chat#
  DeepSeekQwenR1DistillChat: DeepSeekQwenR1DistillChat#
  QwQChat: QwQChat#
  DeepSeekQwenR1DistillChat.format_tokens: DeepSeekQwenR1DistillChat#format_tokens().
  DeepSeekQwenR1DistillChat.format: DeepSeekQwenR1DistillChat#format().
  DeepSeekQwenR1DistillChat.assistant_marker: DeepSeekQwenR1DistillChat#assistant_marker.
  LMFormat.format_tokens: LMFormat#format_tokens().
  LMFormat.bos_id: LMFormat#bos_id.
  LMFormat.pad_id: LMFormat#pad_id.
  LMFormat.extra_eos_tokens: LMFormat#extra_eos_tokens.
  SimplyV1Chat.assistant_marker: SimplyV1Chat#assistant_marker.
  GemmaV2Chat.assistant_marker: GemmaV2Chat#assistant_marker.
  DeepSeekQwenR1DistillChat.user_marker: DeepSeekQwenR1DistillChat#user_marker.
  DeepSeekQwenR1DistillChat.end_of_message_marker: DeepSeekQwenR1DistillChat#end_of_message_marker.
  QwenV2Chat.assistant_marker: QwenV2Chat#assistant_marker.
  SimplyV1Chat.user_marker: SimplyV1Chat#user_marker.
  SimplyV1Chat.system_marker: SimplyV1Chat#system_marker.
  SimplyV1Chat.end_of_message_marker: SimplyV1Chat#end_of_message_marker.
  GemmaV2Chat.system_marker: GemmaV2Chat#system_marker.
  GemmaV2Chat.user_marker: GemmaV2Chat#user_marker.
  GemmaV2Chat.end_of_message_marker: GemmaV2Chat#end_of_message_marker.
  QwenV2Chat.user_marker: QwenV2Chat#user_marker.
  QwenV2Chat.system_marker: QwenV2Chat#system_marker.
  QwenV2Chat.end_of_message_marker: QwenV2Chat#end_of_message_marker.
  QwenV2Chat.add_think_marker: QwenV2Chat#add_think_marker.
  QwenV2Chat.begin_of_thought_marker: QwenV2Chat#begin_of_thought_marker.
  LMFormatRegistry.namespace: LMFormatRegistry#namespace.
  LMFormat.begin_of_thought_marker: LMFormat#begin_of_thought_marker.
  LMFormat.end_of_thought_marker: LMFormat#end_of_thought_marker.
  Pretrain.format: Pretrain#format().
  SimplyV1Chat.extra_eos_tokens: SimplyV1Chat#extra_eos_tokens.
  SimplyV1Chat.bos_id: SimplyV1Chat#bos_id.
  SimplyV1Chat.pad_id: SimplyV1Chat#pad_id.
  GemmaV2Chat.extra_eos_tokens: GemmaV2Chat#extra_eos_tokens.
  DeepSeekQwenR1DistillChat.extra_eos_tokens: DeepSeekQwenR1DistillChat#extra_eos_tokens.
  QwenV2Chat.extra_eos_tokens: QwenV2Chat#extra_eos_tokens.
  QwenV2Chat.end_of_thought_marker: QwenV2Chat#end_of_thought_marker.
  QwQChat.add_think_marker: QwQChat#add_think_marker.
---
# Module: [`simply/utils/lm_format.py`](../../../../../../raw/code/simply/simply/utils/lm_format.py)

## Classes
### `DeepSeekQwenR1DistillChat`
- def: [`simply/utils/lm_format.py:171`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L171) — documented in [simply-utils-lm_format](../../../concepts/simply-utils-lm_format.md)
- doc: LM format for Qwen R1 distill.
- signature: `class DeepSeekQwenR1DistillChat(LMFormat):`
- members:
  - `format(self, messages: Sequence[Mapping[str, Any]])` — [`L241`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L241) — documented in [simply-utils-lm_format](../../../concepts/simply-utils-lm_format.md)
  - `format_tokens(self, messages: Sequence[Mapping[str, Any]], tokenizer, trainable_roles: tuple[str, ...] | None = None)` — [`L184`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L184) — Formats and tokenizes with DeepSeek-specific handling.
  - `assistant_marker` — [`L178`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L178)
  - `end_of_message_marker` — [`L180`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L180)
  - `extra_eos_tokens` — [`L181`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L181)
  - `user_marker` — [`L177`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L177)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`LMFormatRegistry`](lm_format.md#LMFormatRegistry), [`LMFormat`](lm_format.md#LMFormat)
- used by: [`format`](lm_format.md#LMFormat.format), [`format_tokens`](lm_format.md#LMFormat.format_tokens), [`test_deep_seek_qwen_r1_distill_chat`](lm_format_test.md#LmFormatTest.test_deep_seek_qwen_r1_distill_chat)

### `GemmaV2Chat`
- def: [`simply/utils/lm_format.py:144`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L144) — documented in [simply-utils-lm_format](../../../concepts/simply-utils-lm_format.md)
- doc: LM format for Gemma V2.
- signature: `class GemmaV2Chat(LMFormat):`
- members:
  - `format(self, messages: Sequence[Mapping[str, Any]])` — [`L152`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L152)
  - `assistant_marker` — [`L148`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L148)
  - `end_of_message_marker` — [`L149`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L149)
  - `extra_eos_tokens` — [`L150`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L150)
  - `system_marker` — [`L146`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L146)
  - `user_marker` — [`L147`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L147)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`LMFormatRegistry`](lm_format.md#LMFormatRegistry), [`LMFormat`](lm_format.md#LMFormat)
- used by: [`format`](lm_format.md#LMFormat.format), [`test_gemma_v2_chat`](lm_format_test.md#LmFormatTest.test_gemma_v2_chat)

### `LMFormat`  ·  implements/extends ABC
- def: [`simply/utils/lm_format.py:29`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L29)
- doc: Base class for Language Model formatting.
- signature: `class LMFormat(abc.ABC):`
- members:
  - `format(self, messages: Sequence[Mapping[str, Any]])` — [`L42`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L42) — Formats the messages into string string (for inference). — documented in [simply-serving-page_batcher](../../../concepts/simply-serving-page_batcher.md)
  - `format_tokens(self, messages: Sequence[Mapping[str, Any]], tokenizer, trainable_roles: tuple[str, ...] | None = None)` — [`L45`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L45) — Formats and tokenizes conversation with per-token loss mask.
  - `begin_of_thought_marker` — [`L38`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L38)
  - `bos_id` — [`L35`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L35)
  - `end_of_thought_marker` — [`L39`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L39)
  - `extra_eos_tokens` — [`L37`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L37)
  - `pad_id` — [`L36`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L36)
- uses (calls/refs, reference-scoped): [`format`](lm_format.md#QwenV2Chat.format), [`format`](lm_format.md#GemmaV2Chat.format), [`format`](lm_format.md#SimplyV1Chat.format), [`format`](lm_format.md#DeepSeekQwenR1DistillChat.format), [`format_tokens`](lm_format.md#DeepSeekQwenR1DistillChat.format_tokens), [`format`](lm_format.md#Pretrain.format)
- used by: [`sample_with_tool`](../tool_lib.md#ToolExecutor.sample_with_tool), [`loop`](../serving/vanilla_server.md#Batcher.loop), [`_try_get_request`](../serving/page_batcher.md#Batcher._try_get_request), [`input_processor`](../serving/page_batcher.md#Batcher.input_processor), [`input_processor`](../serving/vanilla_server.md#Batcher.input_processor), [`get_sampling_input`](evaluation_lib.md#Evaluation.get_sampling_input), [`lm_format`](../serving/page_batcher.md#Batcher.lm_format), [`QwenV2Chat`](lm_format.md#QwenV2Chat), [`SimplyV1Chat`](lm_format.md#SimplyV1Chat), [`DeepSeekQwenR1DistillChat`](lm_format.md#DeepSeekQwenR1DistillChat), [`GemmaV2Chat`](lm_format.md#GemmaV2Chat), [`Pretrain`](lm_format.md#Pretrain), [`lm_format`](../serving/vanilla_server.md#Batcher.lm_format)

### `LMFormatRegistry`  ·  implements/extends RootRegistry
- def: [`simply/utils/lm_format.py:23`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L23) — documented in [simply-utils-lm_format](../../../concepts/simply-utils-lm_format.md)
- doc: Evaluation registry.
- signature: `class LMFormatRegistry(registry.RootRegistry):`
- members:
  - `namespace` — [`L25`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L25)
- uses (calls/refs, reference-scoped): [`RootRegistry`](registry.md#RootRegistry)
- used by: [`run_experiment`](../rl_lib.md#run_experiment), [`main`](../eval/page_decode_eval.md#main), [`main`](../eval/decode_eval.md#main), [`main`](../serving/page_server.md#main), [`RootRegistry`](registry.md#RootRegistry), [`main`](../serving/vanilla_server.md#main), [`QwenV2Chat`](lm_format.md#QwenV2Chat), [`SimplyV1Chat`](lm_format.md#SimplyV1Chat), [`DeepSeekQwenR1DistillChat`](lm_format.md#DeepSeekQwenR1DistillChat), [`GemmaV2Chat`](lm_format.md#GemmaV2Chat), [`Pretrain`](lm_format.md#Pretrain), [`QwQChat`](lm_format.md#QwQChat), [`_MOCK_VOCAB_NAME`](../rl_lib_test.md#_MOCK_VOCAB_NAME), [`_get_lm_format`](../data_lib.md#_get_lm_format)

### `Pretrain`
- def: [`simply/utils/lm_format.py:99`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L99) — documented in [simply-utils-lm_format](../../../concepts/simply-utils-lm_format.md)
- doc: Pre-training model format.
- signature: `class Pretrain(LMFormat):`
- members:
  - `format(self, messages: Sequence[Mapping[str, Any]])` — [`L102`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L102) — documented in [simply-utils-lm_format](../../../concepts/simply-utils-lm_format.md)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`LMFormatRegistry`](lm_format.md#LMFormatRegistry), [`LMFormat`](lm_format.md#LMFormat)
- used by: [`format`](lm_format.md#LMFormat.format), [`test_pretrain`](lm_format_test.md#LmFormatTest.test_pretrain)

### `QwQChat`
- def: [`simply/utils/lm_format.py:291`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L291) — documented in [simply-utils-lm_format](../../../concepts/simply-utils-lm_format.md)
- signature: `class QwQChat(QwenV2Chat):`
- members:
  - `add_think_marker` — [`L292`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L292)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`LMFormatRegistry`](lm_format.md#LMFormatRegistry), [`QwenV2Chat`](lm_format.md#QwenV2Chat)
- used by: [`test_qwq_chat`](lm_format_test.md#LmFormatTest.test_qwq_chat)

### `QwenV2Chat`
- def: [`simply/utils/lm_format.py:259`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L259) — documented in [simply-utils-lm_format](../../../concepts/simply-utils-lm_format.md)
- doc: LM format for Qwen V2.
- signature: `class QwenV2Chat(LMFormat):`
- members:
  - `format(self, messages: Sequence[Mapping[str, Any]])` — [`L270`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L270) — documented in [simply-utils-lm_format](../../../concepts/simply-utils-lm_format.md)
  - `add_think_marker` — [`L266`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L266)
  - `assistant_marker` — [`L262`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L262)
  - `begin_of_thought_marker` — [`L267`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L267)
  - `end_of_message_marker` — [`L264`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L264)
  - `end_of_thought_marker` — [`L268`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L268)
  - `extra_eos_tokens` — [`L265`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L265)
  - `system_marker` — [`L263`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L263)
  - `user_marker` — [`L261`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L261)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`LMFormatRegistry`](lm_format.md#LMFormatRegistry), [`LMFormat`](lm_format.md#LMFormat)
- used by: [`format`](lm_format.md#LMFormat.format), [`QwQChat`](lm_format.md#QwQChat), [`test_qwen_v2_chat`](lm_format_test.md#LmFormatTest.test_qwen_v2_chat)

### `SimplyV1Chat`
- def: [`simply/utils/lm_format.py:112`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L112) — documented in [simply-utils-lm_format](../../../concepts/simply-utils-lm_format.md)
- doc: LM format for Simply V1.
- signature: `class SimplyV1Chat(LMFormat):`
- members:
  - `format(self, messages: Sequence[Mapping[str, Any]])` — [`L125`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L125)
  - `assistant_marker` — [`L115`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L115)
  - `bos_id` — [`L122`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L122)
  - `end_of_message_marker` — [`L117`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L117) — documented in [simply-utils-lm_format](../../../concepts/simply-utils-lm_format.md)
  - `extra_eos_tokens` — [`L118`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L118)
  - `pad_id` — [`L123`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L123)
  - `system_marker` — [`L116`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L116)
  - `user_marker` — [`L114`](../../../../../../raw/code/simply/simply/utils/lm_format.py#L114)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`LMFormatRegistry`](lm_format.md#LMFormatRegistry), [`LMFormat`](lm_format.md#LMFormat)
- used by: [`format`](lm_format.md#LMFormat.format), [`MockSimplyV1Chat`](../rl_lib_test.md#MockSimplyV1Chat), [`test_simply_v1_chat`](lm_format_test.md#LmFormatTest.test_simply_v1_chat)

