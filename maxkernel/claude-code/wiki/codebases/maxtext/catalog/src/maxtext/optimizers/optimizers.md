---
title: 'Module: src/maxtext/optimizers/optimizers.py'
type: catalog
provenance: extracted
module: src/maxtext/optimizers/optimizers.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.optimizers.optimizers`/
symbols:
  get_optimizer: get_optimizer().
  skip_step_on_spikes.update_fn: skip_step_on_spikes().update_fn().
  skip_step_on_spikes: skip_step_on_spikes().
  adam_pax: adam_pax().
  adam_pax.update_fn._update_momentum: adam_pax().update_fn()._update_momentum().
  _get_path_mask_fn: _get_path_mask_fn().
  get_adamw_mask: get_adamw_mask().
  _get_path_mask_fn.mask_fn: _get_path_mask_fn().mask_fn().
  adam_pax.update_fn: adam_pax().update_fn().
  _compute_rolling_stats: _compute_rolling_stats().
  adam_pax.bias_corrected_decay: adam_pax().bias_corrected_decay().
  _get_path_mask_fn.mask_fn._is_masked: _get_path_mask_fn().mask_fn()._is_masked().
  skip_step_on_spikes.init_fn: skip_step_on_spikes().init_fn().
  skip_step_on_spikes.update_fn.do_update: skip_step_on_spikes().update_fn().do_update().
  skip_step_on_spikes.update_fn.skip_update: skip_step_on_spikes().update_fn().skip_update().
  adam_pax.init_fn: adam_pax().init_fn().
  adam_pax.update_fn._slot_opt_state: adam_pax().update_fn()._slot_opt_state#
  adam_pax.update_fn._slot_opt_state.__init__: adam_pax().update_fn()._slot_opt_state#__init__().
  _slot_opt_state.mu: _slot_opt_state#mu.
  _slot_opt_state.nu: _slot_opt_state#nu.
---
# Module: [`src/maxtext/optimizers/optimizers.py`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py)

## Classes
### `_slot_opt_state`
- def: [`src/maxtext/optimizers/optimizers.py:307`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L307)
- signature: `class _slot_opt_state:`
- members:
  - `mu` — [`L310`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L310)
  - `nu` — [`L311`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L311)
- protocol/private: `__init__`[`L309`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L309)
- used by: [`_update_momentum`](optimizers.md#adam_pax.update_fn._update_momentum)

## Functions
- `_compute_rolling_stats(arr: jax.Array, count: jax.Array, interval: int)` — [`L51`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L51) — Computes mean and unbiased std (Bessel's correction) over a rolling window.
- `_get_path_mask_fn(patterns, match_returns_true=True)` — [`L27`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L27) — Helper to create a mask function from a list of regex patterns.
- `_is_masked(path, _)` — [`L35`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L35)
- `_update_momentum(update, mu, nu)` — [`L313`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L313)
- `adam_pax(learning_rate_fn: optax.Schedule, beta1: float, beta2: float, epsilon: float, epsilon_root: float, weight_decay: float, mask=None)` — [`L244`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L244) — Standard Adam optimizer that supports weight decay.
- `bias_corrected_decay(step: jnp.int32, decay: float)` — [`L278`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L278) — Incorporates bias correction into decay.
- `do_update()` — [`L138`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L138)
- `get_adamw_mask(config)` — [`L46`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L46) — Create a mask function for AdamW optimizer to exclude certain parameters from weight decay.
- `get_optimizer(config, learning_rate_schedule, model=None)` — [`L168`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L168) — Create optimizer.
- `init_fn(params)` — [`L89`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L89)
- `init_fn(params)` — [`L273`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L273)
- `mask_fn(params)` — [`L34`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L34)
- `skip_step_on_spikes(inner_opt: optax.GradientTransformation, interval: int, scaling_factor: float)` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L66) — Wrapper that skips updates when loss or grad_norm spike.
- `skip_update()` — [`L141`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L141)
- `update_fn(updates, state, params=None, **extra_args)` — [`L98`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L98)
- `update_fn(updates, state, params=None)` — [`L301`](../../../../../../../raw/code/maxtext/src/maxtext/optimizers/optimizers.py#L301)

