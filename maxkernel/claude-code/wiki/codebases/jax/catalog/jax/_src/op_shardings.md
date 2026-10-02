---
title: 'Module: jax/_src/op_shardings.py'
type: catalog
provenance: extracted
module: jax/_src/op_shardings.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.op_shardings`/
symbols:
  are_hlo_shardings_equal: are_hlo_shardings_equal().
  op_sharding_to_numpy_indices: op_sharding_to_numpy_indices().
  get_num_ways_dim_sharded: get_num_ways_dim_sharded().
  is_hlo_sharding_replicated: is_hlo_sharding_replicated().
  op_sharding_to_indices: op_sharding_to_indices().
  _Index: _Index.
---
# Module: [`jax/_src/op_shardings.py`](../../../../../../raw/code/jax/jax/_src/op_shardings.py)

## Functions
- `are_hlo_shardings_equal(hc1: xc.HloSharding, hc2: xc.HloSharding)` — [`L58`](../../../../../../raw/code/jax/jax/_src/op_shardings.py#L58)
- `get_num_ways_dim_sharded(hlo_sharding: xc.HloSharding)` — [`L27`](../../../../../../raw/code/jax/jax/_src/op_shardings.py#L27)
- `is_hlo_sharding_replicated(hc: xc.HloSharding)` — [`L54`](../../../../../../raw/code/jax/jax/_src/op_shardings.py#L54)
- `op_sharding_to_indices(op_sharding: xc.HloSharding, shape: Sequence[int], num_devices: int)` — [`L106`](../../../../../../raw/code/jax/jax/_src/op_shardings.py#L106)
- `op_sharding_to_numpy_indices(hlo_sharding: xc.HloSharding, shape: Sequence[int], num_devices: int)` — [`L69`](../../../../../../raw/code/jax/jax/_src/op_shardings.py#L69)

## Module values
- `_Index` — [`L66`](../../../../../../raw/code/jax/jax/_src/op_shardings.py#L66)

