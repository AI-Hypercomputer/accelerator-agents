---
title: 'Module: easydel/data/transforms/chat_template.py'
type: catalog
provenance: extracted
module: easydel/data/transforms/chat_template.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.transforms.chat_template`/
symbols:
  ChatTemplateTransform.__call__: ChatTemplateTransform#__call__().
  ConvertInputOutputToChatML.__call__: ConvertInputOutputToChatML#__call__().
  ConvertToChatML.__call__: ConvertToChatML#__call__().
  ConvertToChatML._role_mapping: ConvertToChatML#_role_mapping.
  MaybeApplyChatTemplate.__call__: MaybeApplyChatTemplate#__call__().
  ChatTemplateTransform: ChatTemplateTransform#
  ChatTemplateTransform.__repr__: ChatTemplateTransform#__repr__().
  ConvertInputOutputToChatML.__repr__: ConvertInputOutputToChatML#__repr__().
  ConvertToChatML.__repr__: ConvertToChatML#__repr__().
  MaybeApplyChatTemplate: MaybeApplyChatTemplate#
  ConvertInputOutputToChatML: ConvertInputOutputToChatML#
  ConvertToChatML: ConvertToChatML#
  MaybeApplyChatTemplate._chat_transform: MaybeApplyChatTemplate#_chat_transform.
  ConvertInputOutputToChatML._output_field: ConvertInputOutputToChatML#_output_field.
  ConvertToChatML._output_field: ConvertToChatML#_output_field.
  ChatTemplateTransform._messages_field: ChatTemplateTransform#_messages_field.
  ChatTemplateTransform._output_field: ChatTemplateTransform#_output_field.
  ConvertInputOutputToChatML._input_field: ConvertInputOutputToChatML#_input_field.
  ConvertToChatML._input_field: ConvertToChatML#_input_field.
  DEFAULT_ROLE_MAPPING.DEFAULT_ROLE_MAPPING: DEFAULT_ROLE_MAPPING.DEFAULT_ROLE_MAPPING.
  is_conversational: is_conversational().
  convert_to_chatml: convert_to_chatml().
  ChatTemplateTransform._tokenizer: ChatTemplateTransform#_tokenizer.
  ChatTemplateTransform._tools: ChatTemplateTransform#_tools.
  ChatTemplateTransform._convert_from_value: ChatTemplateTransform#_convert_from_value.
  ChatTemplateTransform._drop_messages: ChatTemplateTransform#_drop_messages.
  ChatTemplateTransform._template_kwargs: ChatTemplateTransform#_template_kwargs.
  ChatTemplateTransform._simple_format: ChatTemplateTransform#_simple_format().
  ConvertInputOutputToChatML._user_role: ConvertInputOutputToChatML#_user_role.
  ConvertInputOutputToChatML._assistant_role: ConvertInputOutputToChatML#_assistant_role.
  ChatTemplateTransform.__init__: ChatTemplateTransform#__init__().
  MaybeApplyChatTemplate.__init__: MaybeApplyChatTemplate#__init__().
  MaybeApplyChatTemplate._tokenizer: MaybeApplyChatTemplate#_tokenizer.
  MaybeApplyChatTemplate.__repr__: MaybeApplyChatTemplate#__repr__().
  ConvertInputOutputToChatML.__init__: ConvertInputOutputToChatML#__init__().
  ConvertToChatML.__init__: ConvertToChatML#__init__().
---
# Module: [`easydel/data/transforms/chat_template.py`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py)

## Classes
### `ChatTemplateTransform`  ·  implements/extends Transform
- def: [`easydel/data/transforms/chat_template.py:82`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L82)
- doc: Apply chat template to convert messages to formatted text.
- signature: `class ChatTemplateTransform(Transform):`
- members:
  - `__call__(self, example: Example)` — [`L133`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L133) — Apply chat template to convert messages to formatted text.
  - `__init__(self, tokenizer: tp.Any, messages_field: str = "messages", output_field: str = "text", tools: list[dict | tp.Callable] | None = None, convert_from_value_format: bool = True, drop_messages: bool = True, **template_kwargs)` — [`L103`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L103) — Initialize ChatTemplateTransform.
  - `_simple_format(self, messages: list[dict])` — [`L188`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L188) — Simple fallback formatting if chat template fails.
- protocol/private: `__repr__`[`L197`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L197), `_convert_from_value`[`L129`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L129), `_drop_messages`[`L130`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L130), `_messages_field`[`L126`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L126), `_output_field`[`L127`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L127), `_template_kwargs`[`L131`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L131), `_tokenizer`[`L125`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L125), `_tools`[`L128`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L128)
- uses (calls/refs, reference-scoped): [`Transform`](base.md#Transform), [`Example`](base.md#Example), [`convert_to_chatml`](chat_template.md#convert_to_chatml)
- used by: [`Transform`](base.md#Transform), [`_chat_transform`](chat_template.md#MaybeApplyChatTemplate._chat_transform)

### `ConvertInputOutputToChatML`  ·  implements/extends Transform
- def: [`easydel/data/transforms/chat_template.py:241`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L241)
- doc: Convert input/output conversation format to ChatML messages format.
- signature: `class ConvertInputOutputToChatML(Transform):`
- members:
  - `__call__(self, example: Example)` — [`L286`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L286) — Convert input/output conversation pairs to ChatML messages format.
  - `__init__(self, input_field: str = "conversation", output_field: str = "messages", user_role: str = "user", assistant_role: str = "assistant")` — [`L266`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L266) — Initialize ConvertInputOutputToChatML.
- protocol/private: `__repr__`[`L324`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L324), `_assistant_role`[`L284`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L284), `_input_field`[`L281`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L281), `_output_field`[`L282`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L282), `_user_role`[`L283`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L283)
- uses (calls/refs, reference-scoped): [`Transform`](base.md#Transform), [`Example`](base.md#Example)
- used by: [`Transform`](base.md#Transform)

### `ConvertToChatML`  ·  implements/extends Transform
- def: [`easydel/data/transforms/chat_template.py:343`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L343)
- doc: Convert from/value format to ChatML messages format.
- signature: `class ConvertToChatML(Transform):`
- members:
  - `__call__(self, example: Example)` — [`L402`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L402) — Convert from/value format messages to standard ChatML format.
  - `__init__(self, input_field: str = "conversations", output_field: str = "messages", role_mapping: dict[str, str] | None = None, use_default_mapping: bool = True)` — [`L374`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L374) — Initialize ConvertToChatML.
- protocol/private: `__repr__`[`L462`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L462), `_input_field`[`L391`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L391), `_output_field`[`L392`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L392), `_role_mapping`[`L396`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L396)
- uses (calls/refs, reference-scoped): [`Transform`](base.md#Transform), [`Example`](base.md#Example), [`DEFAULT_ROLE_MAPPING`](chat_template.md#DEFAULT_ROLE_MAPPING.DEFAULT_ROLE_MAPPING)
- used by: [`Transform`](base.md#Transform)

### `MaybeApplyChatTemplate`  ·  implements/extends Transform
- def: [`easydel/data/transforms/chat_template.py:201`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L201)
- doc: Conditionally apply chat template only if example is conversational.
- signature: `class MaybeApplyChatTemplate(Transform):`
- members:
  - `__call__(self, example: Example)` — [`L224`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L224) — Apply chat template only if the example is in conversational format.
  - `__init__(self, tokenizer: tp.Any, **chat_template_kwargs)` — [`L214`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L214) — Initialize MaybeApplyChatTemplate.
- protocol/private: `__repr__`[`L237`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L237), `_chat_transform`[`L222`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L222), `_tokenizer`[`L221`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L221)
- uses (calls/refs, reference-scoped): [`Transform`](base.md#Transform), [`Example`](base.md#Example), [`ChatTemplateTransform`](chat_template.md#ChatTemplateTransform), [`is_conversational`](chat_template.md#is_conversational)
- used by: [`Transform`](base.md#Transform)

## Functions
- `convert_to_chatml(messages: list[dict])` — [`L58`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L58) — Convert from/value format to role/content (ChatML) format.
- `is_conversational(example: dict)` — [`L30`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L30) — Check if an example is in conversational format.

## Module values
- `DEFAULT_ROLE_MAPPING` — [`L329`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/chat_template.py#L329)

