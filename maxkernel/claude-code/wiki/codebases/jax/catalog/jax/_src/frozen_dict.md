---
title: 'Module: jax/_src/frozen_dict.py'
type: catalog
provenance: extracted
module: jax/_src/frozen_dict.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.frozen_dict`/
symbols:
  FrozenDict: FrozenDict#
  FrozenDict._d: FrozenDict#_d.
  FrozenDict.__getitem__: FrozenDict#__getitem__().
  FrozenDict.get: FrozenDict#get().
  K: K.
  FrozenDict.__init__: FrozenDict#__init__().
  FrozenDict.__eq__: FrozenDict#__eq__().
  FrozenDict.__iter__: FrozenDict#__iter__().
  V: V.
  FrozenDict.__repr__: FrozenDict#__repr__().
  FrozenDict.__str__: FrozenDict#__str__().
  FrozenDict.__hash__: FrozenDict#__hash__().
  FrozenDict.__len__: FrozenDict#__len__().
---
# Module: [`jax/_src/frozen_dict.py`](../../../../../../raw/code/jax/jax/_src/frozen_dict.py)

## Classes
### `FrozenDict`  ·  implements/extends Mapping
- def: [`jax/_src/frozen_dict.py:22`](../../../../../../raw/code/jax/jax/_src/frozen_dict.py#L22)
- signature: `class FrozenDict(Mapping[K, V]):`
- members:
  - `get(self, key: K)` — [`L51`](../../../../../../raw/code/jax/jax/_src/frozen_dict.py#L51)
- protocol/private: `__eq__`[`L40`](../../../../../../raw/code/jax/jax/_src/frozen_dict.py#L40), `__getitem__`[`L33`](../../../../../../raw/code/jax/jax/_src/frozen_dict.py#L33), `__hash__`[`L36`](../../../../../../raw/code/jax/jax/_src/frozen_dict.py#L36), `__init__`[`L24`](../../../../../../raw/code/jax/jax/_src/frozen_dict.py#L24), `__iter__`[`L45`](../../../../../../raw/code/jax/jax/_src/frozen_dict.py#L45), `__len__`[`L48`](../../../../../../raw/code/jax/jax/_src/frozen_dict.py#L48), `__repr__`[`L27`](../../../../../../raw/code/jax/jax/_src/frozen_dict.py#L27), `__str__`[`L30`](../../../../../../raw/code/jax/jax/_src/frozen_dict.py#L30), `_d`[`L25`](../../../../../../raw/code/jax/jax/_src/frozen_dict.py#L25)
- uses (calls/refs, reference-scoped): [`K`](frozen_dict.md#K), [`V`](frozen_dict.md#V)
- used by: [`interpret_pallas_call`](pallas/mosaic/interpret/interpret_pallas_call.md#interpret_pallas_call), [`_pallas_call_batching_rule`](pallas/pallas_call.md#_pallas_call_batching_rule), [`_pallas_call_state_discharge_rule`](pallas/pallas_call.md#_pallas_call_state_discharge_rule), [`pallas_call_hlo_interpret`](pallas/hlo_interpreter.md#pallas_call_hlo_interpret), [`_lower_to_custom_call`](pallas/mosaic/pallas_call_registration.md#_lower_to_custom_call), [`pallas_call_lowering`](pallas/triton/pallas_call_registration.md#pallas_call_lowering), [`wrapper`](pallas/mpmd.md#_mpmd_map.wrapper), [`_pallas_call_to_lojax`](pallas/pallas_call.md#_pallas_call_to_lojax), [`_pallas_call_jvp_rule`](pallas/pallas_call.md#_pallas_call_jvp_rule), [`pallas_call_lowering`](pallas/mosaic_gpu/pallas_call_registration.md#pallas_call_lowering), [`wrapped`](pallas/pallas_call.md#_pallas_call.wrapped), [`pallas_call_tpu_lowering_rule`](pallas/mosaic/pallas_call_registration.md#pallas_call_tpu_lowering_rule), [`_batch_with_explicit_loop`](pallas/pallas_call.md#_batch_with_explicit_loop), [`wrapped`](pallas/core.md#core_map.wrapped), [`_tensorcore_mesh_discharge_rule`](pallas/mosaic/core.md#_tensorcore_mesh_discharge_rule), [`_mpmd_map_discharge_rule`](pallas/mpmd.md#_mpmd_map_discharge_rule), [`_mpmd_map_to_lojax`](pallas/mpmd.md#_mpmd_map_to_lojax), [`extend_pure`](core.md#AxisEnv.extend_pure), [`axis_sizes`](core.md#AxisEnv.axis_sizes), [`pop_pure`](core.md#AxisEnv.pop_pure), [`__init__`](pallas/mosaic/core.md#CompilerParams.__init__), [`assert_is_tile_preserving`](pallas/einshape.md#Einshape.assert_is_tile_preserving), [`__post_init__`](pallas/mosaic_gpu/core.md#ParameterizedLayout.__post_init__), [`__post_init__`](tpu_custom_call.md#CustomCallBackendConfig.__post_init__), [`top_axis_env`](core.md#top_axis_env), [`__new__`](core.md#AxisEnv.__new__)

## Module values
- `K` — [`L18`](../../../../../../raw/code/jax/jax/_src/frozen_dict.py#L18)
- `V` — [`L19`](../../../../../../raw/code/jax/jax/_src/frozen_dict.py#L19)

