---
title: 'Module: ejkernel/callib/_utils.py'
type: catalog
provenance: extracted
module: ejkernel/callib/_utils.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.callib._utils`/
symbols:
  ShapeDtype: ShapeDtype#
  check_bool_flag: check_bool_flag().
  cdiv: cdiv().
  get_cache_dir: get_cache_dir().
  quiet: quiet().
  DummyStream: DummyStream#
  strides_from_shape: strides_from_shape().
  next_power_of_2: next_power_of_2().
  ShapeDtype.shape: ShapeDtype#shape().
  ShapeDtype.dtype: ShapeDtype#dtype().
  DummyStream.write: DummyStream#write().
  DummyStream.flush: DummyStream#flush().
---
# Module: [`ejkernel/callib/_utils.py`](../../../../../../raw/code/ejkernel/ejkernel/callib/_utils.py)

## Classes
### `DummyStream`
- def: [`ejkernel/callib/_utils.py:185`](../../../../../../raw/code/ejkernel/ejkernel/callib/_utils.py#L185)
- doc: A null device-like stream that discards all writes.
- signature: `class DummyStream:`
- members:
  - `flush(self, *args, **kwargs)` — [`L201`](../../../../../../raw/code/ejkernel/ejkernel/callib/_utils.py#L201) — Discard all flush operations.
  - `write(self, *args, **kwargs)` — [`L192`](../../../../../../raw/code/ejkernel/ejkernel/callib/_utils.py#L192) — Discard all write operations.
- used by: [`quiet`](_utils.md#quiet)

### `ShapeDtype`  ·  implements/extends Protocol
- def: [`ejkernel/callib/_utils.py:122`](../../../../../../raw/code/ejkernel/ejkernel/callib/_utils.py#L122)
- doc: Protocol for objects that have shape and dtype attributes.
- signature: `class ShapeDtype(Protocol):`
- members:
  - `dtype(self)` — [`L147`](../../../../../../raw/code/ejkernel/ejkernel/callib/_utils.py#L147) — Return the data type of the array-like object.
  - `shape(self)` — [`L138`](../../../../../../raw/code/ejkernel/ejkernel/callib/_utils.py#L138) — Return the dimensions of the array-like object.
- used by: [`build_tilelang_call`](_tilelang_call.md#build_tilelang_call), [`tilelang_call`](_tilelang_call.md#tilelang_call), [`cute_call`](_cute_call.md#cute_call), [`triton_call`](_triton_call.md#triton_call)

## Functions
- `cdiv(a: int, b: int)` — [`L48`](../../../../../../raw/code/ejkernel/ejkernel/callib/_utils.py#L48)
- `check_bool_flag(name: str, default: bool = True)` — [`L252`](../../../../../../raw/code/ejkernel/ejkernel/callib/_utils.py#L252) — Parse boolean environment variable.
- `get_cache_dir()` — [`L156`](../../../../../../raw/code/ejkernel/ejkernel/callib/_utils.py#L156) — Get the EasyDeL cache directory.
- `next_power_of_2(x: int)` — [`L105`](../../../../../../raw/code/ejkernel/ejkernel/callib/_utils.py#L105) — Returns the next power of two greater than or equal to `x`.
- `quiet(suppress_stdout=True, suppress_stderr=True)` — [`L212`](../../../../../../raw/code/ejkernel/ejkernel/callib/_utils.py#L212) — Context manager to temporarily suppress stdout and/or stderr output.
- `strides_from_shape(shape: tuple[int, ...])` — [`L80`](../../../../../../raw/code/ejkernel/ejkernel/callib/_utils.py#L80) — Calculate the strides for a contiguous array with the given shape.

