---
title: 'Module: axlearn/common/factorized_rms.py'
type: catalog
provenance: extracted
module: axlearn/common/factorized_rms.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.factorized_rms`/
symbols:
  scale_by_factored_rms: scale_by_factored_rms().
  scale_by_factored_rms.get_vx_spec: scale_by_factored_rms().get_vx_spec().
  scale_by_factored_rms.partition_fn: scale_by_factored_rms().partition_fn().
  scale_by_factored_rms.init_fn._init: scale_by_factored_rms().init_fn()._init().
  _factored_dims: _factored_dims().
  scale_by_factored_rms.init_fn: scale_by_factored_rms().init_fn().
  scale_by_factored_rms.update_fn: scale_by_factored_rms().update_fn().
  scale_by_factored_rms.update_fn._update: scale_by_factored_rms().update_fn()._update().
  _UpdateResult.update: _UpdateResult#update.
  _UpdateResult.v_row: _UpdateResult#v_row.
  _UpdateResult.v_col: _UpdateResult#v_col.
  _UpdateResult.v: _UpdateResult#v.
  scale_by_factored_rms._to_state: scale_by_factored_rms()._to_state().
  scale_by_factored_rms.VxSpec.v_row: scale_by_factored_rms().VxSpec#v_row.
  scale_by_factored_rms.VxSpec.v_col: scale_by_factored_rms().VxSpec#v_col.
  scale_by_factored_rms.VxSpec.v: scale_by_factored_rms().VxSpec#v.
  _UpdateResult: _UpdateResult#
  scale_by_factored_rms.VxSpec: scale_by_factored_rms().VxSpec#
---
# Module: [`axlearn/common/factorized_rms.py`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py)

## Classes
### `VxSpec`
- def: [`axlearn/common/factorized_rms.py:182`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L182)
- signature: `class VxSpec:`
- members:
  - `v` — [`L185`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L185)
  - `v_col` — [`L184`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L184)
  - `v_row` — [`L183`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L183)
- uses (calls/refs, reference-scoped): [`OptStateSpec`](optimizer_base.md#OptStateSpec)
- used by: [`get_vx_spec`](factorized_rms.md#scale_by_factored_rms.get_vx_spec)

### `_UpdateResult`
- def: [`axlearn/common/factorized_rms.py:64`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L64)
- doc: Opaque container that is not traversed by jax.tree.map.
- signature: `class _UpdateResult:`
- members:
  - `update` — [`L67`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L67)
  - `v` — [`L70`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L70)
  - `v_col` — [`L69`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L69)
  - `v_row` — [`L68`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L68)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor)
- used by: [`_init`](factorized_rms.md#scale_by_factored_rms.init_fn._init), [`_update`](factorized_rms.md#scale_by_factored_rms.update_fn._update)

## Functions
- `_factored_dims(factored: bool, factorization_spec: Optional[FactorizationSpec])` — [`L30`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L30) — Whether to use a factored second moment estimator.
- `_init(param)` — [`L109`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L109)
- `_to_state(count: Tensor, result_tree)` — [`L97`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L97) — Maps from a tree of (factored) values to separate trees of values.
- `_update(grad, v_row, v_col, v, param)` — [`L139`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L139)
- `get_vx_spec(param_spec: ParameterSpec)` — [`L187`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L187)
- `init_fn(params)` — [`L106`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L106) — Initialise the optimiser's state.
- `partition_fn(param_specs: NestedParameterSpec)` — [`L226`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L226)
- `scale_by_factored_rms(factored: bool = True, decay_rate: Schedule = config_for_function(adafactor_decay_rate), epsilon: float = 1e-30)` — [`L73`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L73) — Scaling by a factored estimate of the gradient rms (as in Adafactor).
- `update_fn(grads, state, params)` — [`L132`](../../../../../../raw/code/axlearn/axlearn/common/factorized_rms.py#L132) — Apply gradient transformation.

