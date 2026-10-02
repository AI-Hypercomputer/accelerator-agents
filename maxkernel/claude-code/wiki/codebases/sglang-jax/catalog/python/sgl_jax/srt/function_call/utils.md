---
title: 'Module: python/sgl_jax/srt/function_call/utils.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/function_call/utils.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.function_call.utils`/
symbols:
  get_json_schema_constraint: get_json_schema_constraint().
  _get_tool_schema: _get_tool_schema().
  infer_type_from_json_schema: infer_type_from_json_schema().
  _get_tool_schema_defs: _get_tool_schema_defs().
  _find_common_prefix: _find_common_prefix().
  _partial_json_loads: _partial_json_loads().
  _is_complete_json: _is_complete_json().
---
# Module: [`python/sgl_jax/srt/function_call/utils.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/utils.py)

## Functions
- `_find_common_prefix(s1: str, s2: str)` — [`L12`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/utils.py#L12)
- `_get_tool_schema(tool: Tool)` — [`L90`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/utils.py#L90)
- `_get_tool_schema_defs(tools: list[Tool])` — [`L60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/utils.py#L60) — Get consolidated $defs from all tools, validating for conflicts.
- `_is_complete_json(input_str: str)` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/utils.py#L52)
- `_partial_json_loads(input_str: str, flags: Allow)` — [`L23`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/utils.py#L23) — Parse incomplete or partial JSON strings commonly encountered during streaming.
- `get_json_schema_constraint(tools: list[Tool], tool_choice: ToolChoice | Literal["required"])` — [`L105`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/utils.py#L105) — Get the JSON schema constraint for the specified tool choice.
- `infer_type_from_json_schema(schema: dict[str, Any])` — [`L147`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/utils.py#L147) — Infer the primary type of a parameter from JSON Schema.

