---
title: 'Module: easydel/inference/reasoning/auto_detect.py'
type: catalog
provenance: extracted
module: easydel/inference/reasoning/auto_detect.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.reasoning.auto_detect`/
symbols:
  detect_reasoning_parser: detect_reasoning_parser().
  make_reasoning_stripper: make_reasoning_stripper().
  get_reasoning_tags: get_reasoning_tags().
  MODEL_TYPE_TO_REASONING_PARSER.MODEL_TYPE_TO_REASONING_PARSER: MODEL_TYPE_TO_REASONING_PARSER.MODEL_TYPE_TO_REASONING_PARSER.
  _TEMPLATE_HINTS._TEMPLATE_HINTS: _TEMPLATE_HINTS._TEMPLATE_HINTS.
  _VOCAB_HINTS._VOCAB_HINTS: _VOCAB_HINTS._VOCAB_HINTS.
  _DEFAULT_PARSER._DEFAULT_PARSER: _DEFAULT_PARSER._DEFAULT_PARSER.
  make_reasoning_stripper.strip_reasoning: make_reasoning_stripper().strip_reasoning().
---
# Module: [`easydel/inference/reasoning/auto_detect.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/auto_detect.py)

## Functions
- `detect_reasoning_parser(*, parser_name: str | None = None, model_type: str | None = None, tokenizer: tp.Any | None = None)` — [`L149`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/auto_detect.py#L149) — Auto-detect the appropriate reasoning parser.
- `get_reasoning_tags(*, parser_name: str | None = None, model_type: str | None = None, tokenizer: tp.Any | None = None)` — [`L195`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/auto_detect.py#L195) — Resolve the start/end reasoning tags for a model.
- `make_reasoning_stripper(*, parser_name: str | None = None, model_type: str | None = None, tokenizer: tp.Any | None = None)` — [`L229`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/auto_detect.py#L229) — Build a function that strips reasoning blocks from model output.
- `strip_reasoning(text: str)` — [`L281`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/auto_detect.py#L281)

## Module values
- `MODEL_TYPE_TO_REASONING_PARSER` — [`L48`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/auto_detect.py#L48)
- `_DEFAULT_PARSER` — [`L144`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/auto_detect.py#L144)
- `_TEMPLATE_HINTS` — [`L122`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/auto_detect.py#L122)
- `_VOCAB_HINTS` — [`L136`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/auto_detect.py#L136)

