---
title: 'Module: tokamax/_src/ops/normalization/pallas_triton_config.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/normalization/pallas_triton_config.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.normalization.pallas_triton_config`/
symbols:
  get_heuristics_config: get_heuristics_config().
  get_key: get_key().
  canonicalize_shape: canonicalize_shape().
  Config.block_n: Config#block_n.
  Config.block_m: Config#block_m.
  Config.num_warps: Config#num_warps.
  canonicalize_shape_3d: canonicalize_shape_3d().
  Config: Config#
  Key.Key: Key.Key.
  _maybe_shape: _maybe_shape().
---
# Module: [`tokamax/_src/ops/normalization/pallas_triton_config.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_config.py)

## Classes
### `Config`
- def: [`tokamax/_src/ops/normalization/pallas_triton_config.py:31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_config.py#L31)
- signature: `class Config:`
- members:
  - `block_m` — [`L32`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_config.py#L32)
  - `block_n` — [`L33`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_config.py#L33)
  - `num_warps` — [`L34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_config.py#L34)
- used by: [`_fwd`](pallas_triton.md#PallasTritonNormalization._fwd), [`_get_autotuning_configs`](pallas_triton.md#PallasTritonNormalization._get_autotuning_configs), [`get_heuristics_config`](pallas_triton_config.md#get_heuristics_config), [`Config`](pallas_triton.md#Config.Config)

## Functions
- `_maybe_shape(x)` — [`L54`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_config.py#L54)
- `canonicalize_shape(shape: Sequence[int], axis: int)` — [`L46`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_config.py#L46)
- `canonicalize_shape_3d(shape: Sequence[int], axis: int)` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_config.py#L40)
- `get_heuristics_config(x: jax.Array, scale: jax.Array | None, offset: jax.Array | None, *, axis: int, block_size_per_warp: int = 1024, vmap_axis_sizes: tuple[int, ...], **_)` — [`L82`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_config.py#L82) — Returns a config based on heuristics.
- `get_key(x: jax.Array, scale: jax.Array | None, offset: jax.Array | None, *, axis: int, return_residuals: bool = False, **kwargs)` — [`L58`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_config.py#L58) — Returns the lookup key for the given args.

## Module values
- `Key` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_config.py#L37)

