---
title: 'Module: tokamax/_src/ops/normalization/pallas_triton_vjp_config.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/normalization/pallas_triton_vjp_config.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.normalization.pallas_triton_vjp_config`/
symbols:
  get_key: get_key().
  get_heuristics_config: get_heuristics_config().
  _canonicalize_shape: _canonicalize_shape.
  _maybe_shape: _maybe_shape().
  Key.Key: Key.Key.
  Config: Config#
  Config.block_n: Config#block_n.
  Config.block_m: Config#block_m.
  Config.num_warps: Config#num_warps.
---
# Module: [`tokamax/_src/ops/normalization/pallas_triton_vjp_config.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp_config.py)

## Classes
### `Config`
- def: [`tokamax/_src/ops/normalization/pallas_triton_vjp_config.py:29`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp_config.py#L29)
- signature: `class Config:`
- members:
  - `block_m` — [`L30`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp_config.py#L30)
  - `block_n` — [`L31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp_config.py#L31)
  - `num_warps` — [`L32`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp_config.py#L32)
- used by: [`_get_autotuning_configs`](pallas_triton_vjp.md#PallasTritonNormalizationVjp._get_autotuning_configs), [`_fwd`](pallas_triton_vjp.md#PallasTritonNormalizationVjp._fwd), [`Config`](pallas_triton_vjp.md#Config.Config), [`get_heuristics_config`](pallas_triton_vjp_config.md#get_heuristics_config)

## Functions
- `_maybe_shape(x, axis)` — [`L59`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp_config.py#L59)
- `get_heuristics_config(residuals: base.Residuals, out: jax.Array, dout: jax.Array, x: jax.Array, scale: jax.Array | None, offset: jax.Array | None, **kwargs)` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp_config.py#L38) — Returns a config based on heuristics.
- `get_key(residuals: base.Residuals, out: jax.Array, dout: jax.Array, x: jax.Array, scale: jax.Array | None, offset: jax.Array | None, *, axis: int, scale_offset: float, **kwargs)` — [`L67`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp_config.py#L67) — Returns the lookup key for the given args.

## Module values
- `Key` — [`L35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp_config.py#L35)
- `_canonicalize_shape` — [`L56`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp_config.py#L56)

