---
title: 'Module: src/maxtext/common/train_state_nnx.py'
type: catalog
provenance: extracted
module: src/maxtext/common/train_state_nnx.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.common.train_state_nnx`/
symbols:
  to_linen_checkpoint_dict: to_linen_checkpoint_dict().
  _strip_rng_state: _strip_rng_state().
  _opt_state_to_linen: _opt_state_to_linen().
  TrainStateNNX.apply_gradients: TrainStateNNX#apply_gradients().
  from_linen_checkpoint_dict: from_linen_checkpoint_dict().
  _opt_state_from_linen: _opt_state_from_linen().
  TrainStateNNX.optimizer: TrainStateNNX#optimizer.
  _cast_step: _cast_step().
  _wrap_mu_nu_with_params: _wrap_mu_nu_with_params().
  _strip_mu_nu_params: _strip_mu_nu_params().
  TrainStateNNX.model: TrainStateNNX#model.
  _NNX_RNG_STATE_KEYS: _NNX_RNG_STATE_KEYS.
  _as_chain_index: _as_chain_index().
  TrainStateNNX: TrainStateNNX#
  TrainStateNNX.__init__: TrainStateNNX#__init__().
---
# Module: [`src/maxtext/common/train_state_nnx.py`](../../../../../../../raw/code/maxtext/src/maxtext/common/train_state_nnx.py)

## Classes
### `TrainStateNNX`  ·  implements/extends Module
- def: [`src/maxtext/common/train_state_nnx.py:24`](../../../../../../../raw/code/maxtext/src/maxtext/common/train_state_nnx.py#L24)
- doc: A unified container for NNX models and optimizers.
- signature: `class TrainStateNNX(nnx.Module):`
- members:
  - `apply_gradients(self, grads: Any, **kwargs)` — [`L43`](../../../../../../../raw/code/maxtext/src/maxtext/common/train_state_nnx.py#L43) — Mimics the Linen apply_gradients function.
  - `model` — [`L40`](../../../../../../../raw/code/maxtext/src/maxtext/common/train_state_nnx.py#L40)
  - `optimizer` — [`L41`](../../../../../../../raw/code/maxtext/src/maxtext/common/train_state_nnx.py#L41)
- protocol/private: `__init__`[`L35`](../../../../../../../raw/code/maxtext/src/maxtext/common/train_state_nnx.py#L35)

## Functions
- `_as_chain_index(key)` — [`L112`](../../../../../../../raw/code/maxtext/src/maxtext/common/train_state_nnx.py#L112) — Returns the int index for an int or digit-string key, else None.
- `_cast_step(step, dtype)` — [`L74`](../../../../../../../raw/code/maxtext/src/maxtext/common/train_state_nnx.py#L74) — Casts the step's dtype, handling both concrete arrays and abstract ShapeDtypeStruct.
- `_opt_state_from_linen(opt_state)` — [`L165`](../../../../../../../raw/code/maxtext/src/maxtext/common/train_state_nnx.py#L165) — Inverse of `_opt_state_to_linen`: Linen list-with-None -> NNX int-keyed dict.
- `_opt_state_to_linen(opt_state)` — [`L121`](../../../../../../../raw/code/maxtext/src/maxtext/common/train_state_nnx.py#L121) — Reshapes the NNX optax-chain opt_state to Linen's list-with-None layout.
- `_strip_mu_nu_params(state)` — [`L156`](../../../../../../../raw/code/maxtext/src/maxtext/common/train_state_nnx.py#L156) — Inverse of `_wrap_mu_nu_with_params`: removes the inner 'params' wrap from mu/nu.
- `_strip_rng_state(tree)` — [`L86`](../../../../../../../raw/code/maxtext/src/maxtext/common/train_state_nnx.py#L86) — Removes the NNX-only 'rngs'/'dropout' subtrees that Linen doesn't carry.
- `_wrap_mu_nu_with_params(state)` — [`L105`](../../../../../../../raw/code/maxtext/src/maxtext/common/train_state_nnx.py#L105) — Wraps mu/nu under an inner 'params' key (the Linen collection).
- `from_linen_checkpoint_dict(linen_pure_dict)` — [`L174`](../../../../../../../raw/code/maxtext/src/maxtext/common/train_state_nnx.py#L174) — Inverse of `to_linen_checkpoint_dict`: Linen on-disk layout -> NNX layout.
- `to_linen_checkpoint_dict(nnx_pure_dict)` — [`L139`](../../../../../../../raw/code/maxtext/src/maxtext/common/train_state_nnx.py#L139) — Reshapes a TrainStateNNX pure dict ({model, optimizer}) into the Linen on-disk layout.

## Module values
- `_NNX_RNG_STATE_KEYS` — [`L71`](../../../../../../../raw/code/maxtext/src/maxtext/common/train_state_nnx.py#L71)

