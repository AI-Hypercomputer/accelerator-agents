---
title: 'Module: easydel/inference/tools/parsers/longcat_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/longcat_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.longcat_tool_parser`/LongcatFlashToolParser#
symbols:
  LongcatFlashToolParser.__init__: __init__().
  LongcatFlashToolParser: ''
  LongcatFlashToolParser.tool_call_start_token_ids: tool_call_start_token_ids.
  LongcatFlashToolParser.tool_call_end_token_ids: tool_call_end_token_ids.
  LongcatFlashToolParser.tool_call_start_token_array: tool_call_start_token_array.
  LongcatFlashToolParser.tool_call_regex: tool_call_regex.
  LongcatFlashToolParser.tool_call_end_token_array: tool_call_end_token_array.
---
# Module: [`easydel/inference/tools/parsers/longcat_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/longcat_tool_parser.py)

## Classes
### `LongcatFlashToolParser`  ·  implements/extends HermesToolParser
- def: [`easydel/inference/tools/parsers/longcat_tool_parser.py:43`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/longcat_tool_parser.py#L43)
- doc: Tool parser for Longcat Flash models.
- signature: `class LongcatFlashToolParser(HermesToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L74`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/longcat_tool_parser.py#L74) — Initialize the LongcatFlashToolParser.
  - `tool_call_end_token_array` — [`L103`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/longcat_tool_parser.py#L103)
  - `tool_call_end_token_ids` — [`L98`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/longcat_tool_parser.py#L98)
  - `tool_call_regex` — [`L89`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/longcat_tool_parser.py#L89)
  - `tool_call_start_token_array` — [`L100`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/longcat_tool_parser.py#L100)
  - `tool_call_start_token_ids` — [`L94`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/longcat_tool_parser.py#L94)
- uses (calls/refs, reference-scoped): [`ToolParserManager`](../abstract_tool.md#ToolParserManager), [`register_module`](../abstract_tool.md#ToolParserManager.register_module), [`model_tokenizer`](../abstract_tool.md#ToolParser.model_tokenizer), [`HermesToolParser`](hermes_tool_parser.md#HermesToolParser), [`tool_call_start_token`](hermes_tool_parser.md#HermesToolParser.tool_call_start_token), [`tool_call_end_token`](hermes_tool_parser.md#HermesToolParser.tool_call_end_token), [`__init__`](hermes_tool_parser.md#HermesToolParser.__init__)
- used by: [`HermesToolParser`](hermes_tool_parser.md#HermesToolParser)

