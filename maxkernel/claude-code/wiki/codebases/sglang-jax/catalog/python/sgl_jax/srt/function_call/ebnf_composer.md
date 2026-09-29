---
title: 'Module: python/sgl_jax/srt/function_call/ebnf_composer.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/function_call/ebnf_composer.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.function_call.ebnf_composer`/EBNFComposer#
symbols:
  EBNFComposer.build_ebnf: build_ebnf().
  EBNFComposer: ''
  EBNFComposer.get_value_rule: get_value_rule().
  EBNFComposer.get_type_mapping: get_type_mapping().
  EBNFComposer._handle_type: _handle_type().
  EBNFComposer.json_grammar_ebnf_str: json_grammar_ebnf_str.
  EBNFComposer.pythonic_grammar_ebnf_str: pythonic_grammar_ebnf_str.
  EBNFComposer.xml_grammar_ebnf_str: xml_grammar_ebnf_str.
  EBNFComposer._handle_enum: _handle_enum().
  EBNFComposer.BASE_PRIMITIVE_GRAMMAR: BASE_PRIMITIVE_GRAMMAR.
  EBNFComposer.CALL_RULE_MAP: CALL_RULE_MAP.
  EBNFComposer.ARGUMENTS_RULE_MAP: ARGUMENTS_RULE_MAP.
  EBNFComposer.KEY_VALUE_RULE_MAP: KEY_VALUE_RULE_MAP.
  EBNFComposer.BASE_TYPE_MAPPING: BASE_TYPE_MAPPING.
  EBNFComposer.FORMAT_TYPE_OVERRIDES: FORMAT_TYPE_OVERRIDES.
  EBNFComposer.format_enum_val: format_enum_val().
---
# Module: [`python/sgl_jax/srt/function_call/ebnf_composer.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py)

## Classes
### `EBNFComposer`
- def: [`python/sgl_jax/srt/function_call/ebnf_composer.py:4`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py#L4)
- signature: `class EBNFComposer:`
- members:
  - `_handle_enum(prop: dict, function_format: str)` — [`L104`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py#L104) — Handle enum properties by formatting each value according to type and format.
  - `_handle_type(prop: dict, function_format: str)` — [`L140`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py#L140) — Handle type properties using the appropriate type mapping.
  - `build_ebnf(tools, function_format: Literal["pythonic", "json", "xml"] = "json", sequence_start_token: str | None = None, sequence_end_token: str | None = None, individual_call_start_token: str | None = None, individual_call_end_token: str | None = None, tool_call_separator: str | None = None, call_rule_fmt: str | None = None, key_value_rule_fmt: str | None = None, key_value_separator: str = 'ws "," ws')` — [`L154`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py#L154) — Generalized EBNF builder for all detectors.
  - `format_enum_val(v: Any)` — [`L109`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py#L109)
  - `get_type_mapping(function_format: str)` — [`L132`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py#L132) — Get the complete type mapping for a given format.
  - `get_value_rule(prop: dict, function_format: Literal["pythonic", "json", "xml"] = "json")` — [`L92`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py#L92)
  - `ARGUMENTS_RULE_MAP` — [`L57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py#L57)
  - `BASE_PRIMITIVE_GRAMMAR` — [`L7`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py#L7)
  - `BASE_TYPE_MAPPING` — [`L70`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py#L70)
  - `CALL_RULE_MAP` — [`L51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py#L51)
  - `FORMAT_TYPE_OVERRIDES` — [`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py#L81)
  - `KEY_VALUE_RULE_MAP` — [`L63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py#L63)
  - `json_grammar_ebnf_str` — [`L19`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py#L19)
  - `pythonic_grammar_ebnf_str` — [`L29`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py#L29)
  - `xml_grammar_ebnf_str` — [`L39`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/ebnf_composer.py#L39)
- used by: [`build_ebnf`](qwen25_detector.md#Qwen25Detector.build_ebnf), [`build_ebnf`](qwen3_coder_detector.md#Qwen3CoderDetector.build_ebnf)

