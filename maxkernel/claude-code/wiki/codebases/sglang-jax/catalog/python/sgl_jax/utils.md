---
title: 'Module: python/sgl_jax/utils.py'
type: catalog
provenance: extracted
module: python/sgl_jax/utils.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.utils`/
symbols:
  get_exception_traceback: get_exception_traceback().
  traverse_and_update: traverse_and_update().
  TypeBasedDispatcher.__call__: TypeBasedDispatcher#__call__().
  TypeBasedDispatcher: TypeBasedDispatcher#
  TypeBasedDispatcher._mapping: TypeBasedDispatcher#_mapping.
  _create_dummy_buffer: _create_dummy_buffer().
  logger: logger.
  TypeBasedDispatcher.__init__: TypeBasedDispatcher#__init__().
  find_printable_text: find_printable_text().
  convert_json_schema_to_str: convert_json_schema_to_str().
---
# Module: [`python/sgl_jax/utils.py`](../../../../../../raw/code/sglang-jax/python/sgl_jax/utils.py)

## Classes
### `TypeBasedDispatcher`
- def: [`python/sgl_jax/utils.py:14`](../../../../../../raw/code/sglang-jax/python/sgl_jax/utils.py#L14)
- signature: `class TypeBasedDispatcher:`
- protocol/private: `__call__`[`L18`](../../../../../../raw/code/sglang-jax/python/sgl_jax/utils.py#L18), `__init__`[`L15`](../../../../../../raw/code/sglang-jax/python/sgl_jax/utils.py#L15), `_mapping`[`L16`](../../../../../../raw/code/sglang-jax/python/sgl_jax/utils.py#L16)
- used by: [`_request_dispatcher`](srt/managers/scheduler.md#Scheduler._request_dispatcher)

## Functions
- `_create_dummy_buffer(buffer)` — [`L74`](../../../../../../raw/code/sglang-jax/python/sgl_jax/utils.py#L74) — Create dummy buffer with sequential values, preserving type and sharding.
- `convert_json_schema_to_str(json_schema: dict | str | type[BaseModel])` — [`L44`](../../../../../../raw/code/sglang-jax/python/sgl_jax/utils.py#L44) — Convert a JSON schema to a string.
- `find_printable_text(text: str)` — [`L25`](../../../../../../raw/code/sglang-jax/python/sgl_jax/utils.py#L25) — Find printable text by removing invalid UTF-8 sequences.
- `get_exception_traceback()` — [`L39`](../../../../../../raw/code/sglang-jax/python/sgl_jax/utils.py#L39) — Get the current exception traceback as a string.
- `traverse_and_update(state_obj, target_modules)` — [`L97`](../../../../../../raw/code/sglang-jax/python/sgl_jax/utils.py#L97) — Recursively traverse state structure and update A_buffer/B_buffer in target modules.

## Module values
- `logger` — [`L11`](../../../../../../raw/code/sglang-jax/python/sgl_jax/utils.py#L11)

