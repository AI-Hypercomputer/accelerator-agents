---
title: 'Module: jax/_src/pallas/fuser/fusion.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/fuser/fusion.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.fuser.fusion`/
symbols:
  Fusion: Fusion#
  Fusion.__call__: Fusion#__call__().
  Fusion.func: Fusion#func.
  A: A.
  Fusion.out_type: Fusion#out_type.
  K: K.
  Fusion.in_type: Fusion#in_type.
  safe_map: safe_map.
  Fusion.shape: Fusion#shape().
  Fusion.dtype: Fusion#dtype().
  Fusion.type: Fusion#type().
  Fusion.in_shape: Fusion#in_shape().
  Fusion.in_dtype: Fusion#in_dtype().
  Fusion.strict_mode: Fusion#strict_mode.
---
# Module: [`jax/_src/pallas/fuser/fusion.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusion.py)

## Classes
### `Fusion`  ·  implements/extends Generic
- def: [`jax/_src/pallas/fuser/fusion.py:33`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusion.py#L33)
- signature: `class Fusion(Generic[A, K]):`
- members:
  - `dtype(self)` — [`L48`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusion.py#L48)
  - `in_dtype(self)` — [`L60`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusion.py#L60)
  - `in_shape(self)` — [`L56`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusion.py#L56)
  - `shape(self)` — [`L44`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusion.py#L44)
  - `type(self)` — [`L52`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusion.py#L52)
  - `func` — [`L35`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusion.py#L35)
  - `in_type` — [`L36`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusion.py#L36)
  - `out_type` — [`L37`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusion.py#L37)
  - `strict_mode` — [`L38`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusion.py#L38)
- protocol/private: `__call__`[`L40`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusion.py#L40)
- uses (calls/refs, reference-scoped): [`A`](fusion.md#A), [`K`](fusion.md#K)
- used by: [`_construct_output_fusions`](jaxpr_fusion.md#_construct_output_fusions), [`_make_trivial_fusion`](fusible.md#_make_trivial_fusion), [`construct_input_fusion`](jaxpr_fusion.md#construct_input_fusion)

## Module values
- `A` — [`L28`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusion.py#L28)
- `K` — [`L29`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusion.py#L29)
- `safe_map` — [`L26`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/fusion.py#L26)

