---
title: 'Module: easydel/inference/esurge/utils.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.utils`/
symbols:
  T: T.
  ConstantList: ConstantList#
  model_uses_mrope: model_uses_mrope().
  cdiv: cdiv().
  ConstantList._x: ConstantList#_x.
  ConstantList.index: ConstantList#index().
  ConstantList.__getitem__: ConstantList#__getitem__().
  ConstantList.__setitem__: ConstantList#__setitem__().
  get_dtype_size: get_dtype_size().
  truncate_tokens: truncate_tokens().
  ConstantList.__init__: ConstantList#__init__().
  ConstantList.__iter__: ConstantList#__iter__().
  ConstantList.__contains__: ConstantList#__contains__().
  ConstantList.__len__: ConstantList#__len__().
  ConstantList.__repr__: ConstantList#__repr__().
  is_list_of: is_list_of().
  chunk_list: chunk_list().
  _get_text_config: _get_text_config().
  _rope_scaling_uses_mrope: _rope_scaling_uses_mrope().
  ConstantList.append: ConstantList#append().
  ConstantList.extend: ConstantList#extend().
  ConstantList.insert: ConstantList#insert().
  ConstantList.pop: ConstantList#pop().
  ConstantList.remove: ConstantList#remove().
  ConstantList.clear: ConstantList#clear().
  ConstantList.__delitem__: ConstantList#__delitem__().
  next_power_of_2: next_power_of_2().
  prev_power_of_2: prev_power_of_2().
  round_up: round_up().
  round_down: round_down().
---
# Module: [`easydel/inference/esurge/utils.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py)

## Classes
### `ConstantList`  ·  implements/extends Generic, Sequence
- def: [`easydel/inference/esurge/utils.py:54`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L54)
- doc: Immutable list wrapper that prevents modifications.
- signature: `class ConstantList(Generic[T], Sequence):`
- members:
  - `__contains__(self, item)` — [`L216`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L216) — Check if an item is in the list.
  - `__delitem__(self, item)` — [`L197`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L197) — Prevent deleting items from the list.
  - `__init__(self, x: list[T])` — [`L74`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L74) — Initialize with a list to make immutable.
  - `__iter__(self)` — [`L208`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L208) — Return an iterator over the list.
  - `__len__(self)` — [`L227`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L227) — Return the length of the list.
  - `__repr__(self)` — [`L235`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L235) — Return a string representation of the ConstantList.
  - `append(self, item)` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L82) — Prevent appending to the list.
  - `clear(self)` — [`L138`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L138) — Prevent clearing the list.
  - `extend(self, item)` — [`L93`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L93) — Prevent extending the list.
  - `index(self, item: T, start: int = 0, stop: int | None = None)` — [`L146`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L146) — Find the index of an item in the list.
  - `insert(self, index, item)` — [`L104`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L104) — Prevent inserting into the list.
  - `pop(self, index=-1)` — [`L116`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L116) — Prevent popping from the list.
  - `remove(self, item)` — [`L127`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L127) — Prevent removing from the list.
- protocol/private: `__getitem__`[`L163`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L163), `__setitem__`[`L180`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L180), `_x`[`L80`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L80)
- uses (calls/refs, reference-scoped): [`T`](utils.md#T)
- used by: [`output_token_ids`](request.md#EngineRequest.output_token_ids), [`all_token_ids`](request.md#EngineRequest.all_token_ids)

## Functions
- `_get_text_config(config: Any)` — [`L485`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L485) — Best-effort resolver for text configs on composite models.
- `_rope_scaling_uses_mrope(rope_scaling: Any)` — [`L521`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L521) — Check if a rope_scaling config indicates multi-dimensional RoPE.
- `cdiv(a: int, b: int)` — [`L306`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L306) — Compute ceiling division.
- `chunk_list(lst: list[T], chunk_size: int)` — [`L283`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L283) — Yield successive chunks of a specified size from a list.
- `get_dtype_size(dtype: jnp.dtype)` — [`L422`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L422) — Get the size of a JAX/NumPy data type in bytes.
- `is_list_of(value: object, typ: type[T] | tuple[type[T], ...], *, check: Literal["first", "all"] = "first")` — [`L244`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L244) — Type guard for checking if a value is a list of a specific type.
- `model_uses_mrope(model: Any)` — [`L567`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L567) — Infer whether a model uses multi-dimensional RoPE (mRoPE).
- `next_power_of_2(n)` — [`L330`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L330) — Find the next power of 2 greater than or equal to n.
- `prev_power_of_2(n: int)` — [`L355`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L355) — Find the previous power of 2 less than or equal to n.
- `round_down(x: int, y: int)` — [`L401`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L401) — Round x down to the nearest multiple of y.
- `round_up(x: int, y: int)` — [`L380`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L380) — Round x up to the nearest multiple of y.
- `truncate_tokens(tokens, target_len: int, mode: str = "left")` — [`L440`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L440) — Truncate a token list to a target length.

## Module values
- `T` — [`L51`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/utils.py#L51)

