---
title: 'Module: graphcast/checkpoint.py'
type: catalog
provenance: extracted
module: graphcast/checkpoint.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.checkpoint`/
symbols:
  _convert_types: _convert_types().
  load: load().
  _flatten: _flatten().
  _T: _T.
  dump: dump().
  _unflatten: _unflatten().
  _SEP: _SEP.
---
# Module: [`graphcast/checkpoint.py`](../../../../../raw/code/graphcast/graphcast/checkpoint.py)

## Functions
- `_convert_types(typ: type[_T], value: Any)` — [`L98`](../../../../../raw/code/graphcast/graphcast/checkpoint.py#L98) — Convert some structure into the given type. The structures must match.
- `_flatten(tree: Any)` — [`L60`](../../../../../raw/code/graphcast/graphcast/checkpoint.py#L60) — Flatten a tree of dicts/dataclasses/lists/tuples to a single dict.
- `_unflatten(flat: dict[str, Any])` — [`L84`](../../../../../raw/code/graphcast/graphcast/checkpoint.py#L84) — Unflatten a dict to a tree of dicts.
- `dump(dest: BinaryIO, value: Any)` — [`L26`](../../../../../raw/code/graphcast/graphcast/checkpoint.py#L26) — Dump a tree of dicts/dataclasses to a file object.
- `load(source: BinaryIO, typ: type[_T])` — [`L42`](../../../../../raw/code/graphcast/graphcast/checkpoint.py#L42) — Load from a file object and convert it to the specified type.

## Module values
- `_SEP` — [`L57`](../../../../../raw/code/graphcast/graphcast/checkpoint.py#L57)
- `_T` — [`L23`](../../../../../raw/code/graphcast/graphcast/checkpoint.py#L23)

