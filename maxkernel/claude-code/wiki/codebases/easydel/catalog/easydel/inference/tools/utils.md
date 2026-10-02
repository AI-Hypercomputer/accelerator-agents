---
title: 'Module: easydel/inference/tools/utils.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.utils`/
symbols:
  extract_intermediate_diff: extract_intermediate_diff().
  consume_space: consume_space().
  find_common_prefix: find_common_prefix().
  partial_json_loads: partial_json_loads().
  is_complete_json: is_complete_json().
  find_common_suffix: find_common_suffix().
  find_all_indices: find_all_indices().
---
# Module: [`easydel/inference/tools/utils.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/utils.py)

## Functions
- `consume_space(i: int, s: str)` — [`L282`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/utils.py#L282) — Skip whitespace characters starting from a given index.
- `extract_intermediate_diff(curr: str, old: str)` — [`L127`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/utils.py#L127) — Extract the difference between two strings accounting for common prefix and suffix.
- `find_all_indices(string: str, substring: str)` — [`L174`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/utils.py#L174) — Find all starting indices of a substring within a string.
- `find_common_prefix(s1: str, s2: str)` — [`L51`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/utils.py#L51) — Find the common prefix shared between two strings.
- `find_common_suffix(s1: str, s2: str)` — [`L88`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/utils.py#L88) — Find the common suffix shared between two strings.
- `is_complete_json(input_str: str)` — [`L250`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/utils.py#L250) — Check if a string contains complete, valid JSON.
- `partial_json_loads(input_str: str, flags: Allow)` — [`L207`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/utils.py#L207) — Load and parse potentially incomplete JSON with fallback handling.

