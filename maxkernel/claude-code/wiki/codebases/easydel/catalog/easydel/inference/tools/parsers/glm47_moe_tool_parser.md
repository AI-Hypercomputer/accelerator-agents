---
title: 'Module: easydel/inference/tools/parsers/glm47_moe_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/glm47_moe_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.glm47_moe_tool_parser`/Glm47MoeModelToolParser#
symbols:
  Glm47MoeModelToolParser: ''
  Glm47MoeModelToolParser.__init__: __init__().
  Glm47MoeModelToolParser.func_detail_regex: func_detail_regex.
  Glm47MoeModelToolParser.func_arg_regex: func_arg_regex.
---
# Module: [`easydel/inference/tools/parsers/glm47_moe_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm47_moe_tool_parser.py)

## Classes
### `Glm47MoeModelToolParser`  ·  implements/extends Glm4MoeModelToolParser
- def: [`easydel/inference/tools/parsers/glm47_moe_tool_parser.py:39`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm47_moe_tool_parser.py#L39)
- doc: Tool parser for GLM-4.7 MoE (Mixture of Experts) models.
- signature: `class Glm47MoeModelToolParser(Glm4MoeModelToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L70`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm47_moe_tool_parser.py#L70) — Initialize the GLM-4.7 MoE tool parser.
  - `func_arg_regex` — [`L85`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm47_moe_tool_parser.py#L85)
  - `func_detail_regex` — [`L84`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm47_moe_tool_parser.py#L84)
- uses (calls/refs, reference-scoped): [`ToolParserManager`](../abstract_tool.md#ToolParserManager), [`register_module`](../abstract_tool.md#ToolParserManager.register_module), [`Glm4MoeModelToolParser`](glm4_moe_tool_parser.md#Glm4MoeModelToolParser), [`__init__`](glm4_moe_tool_parser.md#Glm4MoeModelToolParser.__init__)
- used by: [`Glm4MoeModelToolParser`](glm4_moe_tool_parser.md#Glm4MoeModelToolParser)

