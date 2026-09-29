---
title: 'Module: aqt/jax/v2/flax/freezer.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/flax/freezer.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.flax.freezer`/
symbols:
  FreezerMode: FreezerMode#
  FreezerMode.NONE: FreezerMode#NONE.
  FreezerMode.WRITE: FreezerMode#WRITE.
  FreezerMode.READ: FreezerMode#READ.
  _FREEZE_VAR_NAME: _FREEZE_VAR_NAME.
  Freezer: Freezer#
  Freezer.collection: Freezer#collection.
  Freezer.mode: Freezer#mode.
  Freezer.axis_metadata_wrapper: Freezer#axis_metadata_wrapper.
  Freezer._get_or_set: Freezer#_get_or_set().
  Freezer.get: Freezer#get().
  Freezer.set: Freezer#set().
---
# Module: [`aqt/jax/v2/flax/freezer.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer.py)

## Classes
### `Freezer`
- def: [`aqt/jax/v2/flax/freezer.py:32`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer.py#L32)
- doc: Flax module that can freeze its input.
- signature: `class Freezer(nn.Module):`
- members:
  - `get(self)` — [`L96`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer.py#L96)
  - `set(self, inputs: Any)` — [`L99`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer.py#L99)
  - `axis_metadata_wrapper` — [`L48`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer.py#L48)
  - `collection` — [`L46`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer.py#L46)
  - `mode` — [`L47`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer.py#L47)
- protocol/private: `_get_or_set`[`L51`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer.py#L51)

### `FreezerMode`
- def: [`aqt/jax/v2/flax/freezer.py:23`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer.py#L23)
- signature: `class FreezerMode(enum.Enum):`
- members:
  - `NONE` — [`L24`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer.py#L24)
  - `READ` — [`L26`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer.py#L26)
  - `WRITE` — [`L25`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer.py#L25)

## Module values
- `_FREEZE_VAR_NAME` — [`L29`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/freezer.py#L29)

