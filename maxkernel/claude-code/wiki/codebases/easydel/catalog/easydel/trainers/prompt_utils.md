---
title: 'Module: easydel/trainers/prompt_utils.py'
type: catalog
provenance: extracted
module: easydel/trainers/prompt_utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.prompt_utils`/
symbols:
  apply_chat_template: apply_chat_template().
  convert_to_openai_format: convert_to_openai_format().
  maybe_apply_chat_template: maybe_apply_chat_template().
  maybe_convert_to_chatml: maybe_convert_to_chatml().
  resolve_example_tools: resolve_example_tools().
  normalize_message_payload: normalize_message_payload().
  _SegmentTree.tree: _SegmentTree#tree.
  _pack_bfd: _pack_bfd().
  reverse_openai_format: reverse_openai_format().
  normalize_message_payload._normalize_single: normalize_message_payload()._normalize_single().
  keep_array_and_primitives: keep_array_and_primitives().
  unpair_preference_dataset: unpair_preference_dataset().
  _SegmentTree.add: _SegmentTree#add().
  normalize_tool_payload: normalize_tool_payload().
  maybe_extract_prompt: maybe_extract_prompt().
  InputType: InputType.
  is_conversational: is_conversational().
  _SegmentTree.remove: _SegmentTree#remove().
  _SegmentTree.search: _SegmentTree#search().
  DatasetType: DatasetType.
  render_prompt_with_suffix: render_prompt_with_suffix().
  pack_dataset: pack_dataset().
  remove_none_values: remove_none_values().
  maybe_unpair_preference_dataset: maybe_unpair_preference_dataset().
  OpenAIMessageList: OpenAIMessageList.
  OutputType: OutputType.
  _convert_single_dict: _convert_single_dict().
  _SegmentTree.maxval: _SegmentTree#maxval.
  OpenAIMessage: OpenAIMessage.
  InputDict: InputDict.
  truncate_dataset: truncate_dataset().
  pad_and_truncate_dataset: pad_and_truncate_dataset().
  OutputListDict: OutputListDict.
  TListOrMapping: TListOrMapping.
  _SegmentTree.tree_size: _SegmentTree#tree_size.
  InputListDict: InputListDict.
  InputListListDict: InputListListDict.
  _normalize_chat_suffix: _normalize_chat_suffix().
  pad_and_truncate_dataset.process_batch: pad_and_truncate_dataset().process_batch().
  OutputDict: OutputDict.
  truncate_dataset.truncate: truncate_dataset().truncate().
  DatasetLike: DatasetLike.
  _maybe_json_load: _maybe_json_load().
  keep_array_and_primitives.is_valid_type: keep_array_and_primitives().is_valid_type().
  OpenAIMessageContentPart: OpenAIMessageContentPart.
  _CHATML_ROLE_MAPPING: _CHATML_ROLE_MAPPING.
  _is_valid_openai_message_list: _is_valid_openai_message_list().
  _unpair_row: _unpair_row().
  extract_prompt: extract_prompt().
  _SegmentTree: _SegmentTree#
  _pack_wrapped: _pack_wrapped().
  pad_and_truncate_dataset.get_padding_value: pad_and_truncate_dataset().get_padding_value().
  keep_arrays_map: keep_arrays_map().
  _SegmentTree.__init__: _SegmentTree#__init__().
  is_conversational_from_value: is_conversational_from_value().
---
# Module: [`easydel/trainers/prompt_utils.py`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py)

## Classes
### `_SegmentTree`
- def: [`easydel/trainers/prompt_utils.py:1149`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1149)
- doc: A segment tree data structure that, when initialized as `_SegmentTree(maxval)`, efficiently finds the next larger
- signature: `class _SegmentTree:`
- members:
  - `add(self, val)` — [`L1161`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1161)
  - `remove(self, val)` — [`L1172`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1172)
  - `search(self, val)` — [`L1183`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1183)
  - `maxval` — [`L1156`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1156)
  - `tree` — [`L1159`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1159)
  - `tree_size` — [`L1158`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1158)
- protocol/private: `__init__`[`L1155`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1155)
- used by: [`_pack_bfd`](prompt_utils.md#_pack_bfd)

## Functions
- `_convert_single_dict(source_dict: InputDict)` — [`L214`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L214) — Convert a single dictionary to OpenAI message format.
- `_is_valid_openai_message_list(data: tp.Any)` — [`L175`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L175) — Check if data conforms to OpenAI message list format.
- `_maybe_json_load(value: str)` — [`L68`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L68)
- `_normalize_chat_suffix(value: list[dict[str, tp.Any]] | str, field_name: str)` — [`L456`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L456) — Normalize a prompt or completion suffix value into a conversational message list.
- `_normalize_single(item: tp.Any)` — [`L85`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L85)
- `_pack_bfd(examples: pa.Table, seq_length: int)` — [`L1195`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1195) — Pack sequences in a pyarrow Table using Best Fit Decreasing strategy.
- `_pack_wrapped(examples: pa.Table, seq_length: int)` — [`L1260`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1260) — Pack sequences in a pyarrow Table using a wrapped strategy.
- `_unpair_row(examples: dict[str, list[tp.Any]])` — [`L908`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L908) — Convert a batch of paired preference rows into unpaired rows.
- `apply_chat_template(example: dict[str, tp.Any], tokenizer: ProcessingClassType, tools: list[dict | tp.Callable] | None = None, **template_kwargs)` — [`L561`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L561) — Apply chat template to conversational examples.
- `convert_to_openai_format(input_data: InputType)` — [`L367`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L367) — Converts various input formats (list[list[dict]], list[dict], dict)
- `extract_prompt(example: dict[str, Sequence])` — [`L1043`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1043) — Extracts the shared prompt from a preference data example, where the prompt is implicit within both the chosen and
- `get_padding_value(column_name: str)` — [`L1431`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1431)
- `is_conversational(example: dict[str, tp.Any])` — [`L431`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L431) — Check if an example is in conversational format.
- `is_conversational_from_value(example: dict[str, tp.Any])` — [`L1474`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1474) — Check if the example is in a conversational format (from/value). Note that this format isn't recommended. Prefer
- `is_valid_type(value)` — [`L850`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L850) — Check if value is numpy/jax array or primitive type.
- `keep_array_and_primitives(example: TListOrMapping)` — [`L822`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L822) — Recursively keeps only numpy/jax arrays, ints, floats, and bools from a nested structure.
- `keep_arrays_map(example: dict[str, tp.Any], array_fields: list[str] | None = None, drop_fields: list[str] | None = None)` — [`L880`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L880) — Keep only array fields and convert them to numpy arrays for HF datasets compatibility.
- `maybe_apply_chat_template(example: dict[str, tp.Any], tokenizer: ProcessingClassType, tools: list[dict | tp.Callable] | None = None)` — [`L713`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L713) — Conditionally apply chat template to conversational examples.
- `maybe_convert_to_chatml(example: dict[str, list])` — [`L741`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L741) — Convert a conversational dataset with fields `from` and `value` to ChatML format.
- `maybe_extract_prompt(example: dict[str, list])` — [`L1063`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1063) — Extracts the shared prompt from a preference data example, where the prompt is implicit within both the chosen and
- `maybe_unpair_preference_dataset(dataset: DatasetType, num_proc: int | None = None, desc: str | None = None)` — [`L991`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L991) — Unpair a preference dataset if it is paired.
- `normalize_message_payload(payload: tp.Any, *, allow_plain_text: bool = False)` — [`L78`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L78) — Normalize chat payloads into ``[{role, content, ...}, ...]`` form.
- `normalize_tool_payload(payload: tp.Any)` — [`L134`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L134) — Normalize stringified JSON tool payloads into dict/list form.
- `pack_dataset(dataset: DatasetType, seq_length: int, strategy: str = "bfd", map_kwargs: dict[str, tp.Any] | None = None)` — [`L1278`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1278) — Pack sequences in a dataset into chunks of size `seq_length`.
- `pad_and_truncate_dataset(dataset: DatasetLike, max_length: int, padding_token_id: int | None = None, padding_values: dict[str, tp.Any] | None = None, truncate: bool = True, padding: bool = True, side: tp.Literal["left", "right"] = "left", map_kwargs: dict[str, tp.Any] | None = None, make_it_1d: bool = True)` — [`L1396`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1396) — Pad and/or truncate sequences in a dataset to a specified `max_length`.
- `process_batch(batch: dict[str, list[tp.Any]])` — [`L1451`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1451)
- `remove_none_values(example: TListOrMapping)` — [`L790`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L790) — Recursively removes entries with `None` values from a nested structure (list or dictionary).
- `render_prompt_with_suffix(prompt_messages: list[dict[str, str]], suffix: list[dict[str, str]] | str, tokenizer: ProcessingClassType, *, field_name: str, tools: list[dict | tp.Callable] | None = None, **template_kwargs)` — [`L491`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L491) — Render a conversational prompt using a chat template and derive the rendered suffix.
- `resolve_example_tools(example: dict[str, tp.Any], fallback_tools: list | None = None)` — [`L161`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L161) — Return per-example tool schemas when available, otherwise ``fallback_tools``.
- `reverse_openai_format(openai_messages: OpenAIMessageList, content_key_name: str = "content")` — [`L281`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L281) — Converts a list of OpenAI Chat Completion messages back into simpler formats.
- `truncate(examples)` — [`L1371`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1371)
- `truncate_dataset(dataset: DatasetType, max_length: int, map_kwargs: dict[str, tp.Any] | None = None)` — [`L1337`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L1337) — Truncate sequences in a dataset to a specified `max_length`.
- `unpair_preference_dataset(dataset: DatasetType, num_proc: int | None = None, desc: str | None = None)` — [`L921`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L921) — Unpair a preference dataset.

## Module values
- `DatasetLike` — [`L57`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L57)
- `DatasetType` — [`L44`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L44)
- `InputDict` — [`L46`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L46)
- `InputListDict` — [`L47`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L47)
- `InputListListDict` — [`L48`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L48)
- `InputType` — [`L49`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L49)
- `OpenAIMessage` — [`L51`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L51)
- `OpenAIMessageContentPart` — [`L50`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L50)
- `OpenAIMessageList` — [`L55`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L55)
- `OutputDict` — [`L52`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L52)
- `OutputListDict` — [`L53`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L53)
- `OutputType` — [`L54`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L54)
- `TListOrMapping` — [`L56`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L56)
- `_CHATML_ROLE_MAPPING` — [`L59`](../../../../../../raw/code/EasyDeL/easydel/trainers/prompt_utils.py#L59)

