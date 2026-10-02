---
title: 'Module: tokamax/_src/pallas/grid.py'
type: catalog
provenance: extracted
module: tokamax/_src/pallas/grid.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.pallas.grid`/
symbols:
  get_cheapest_grid_pids: get_cheapest_grid_pids().
  get_grid_pids: get_grid_pids().
  get_cheapest_grid_pids.group_size_m_usage: get_cheapest_grid_pids().group_size_m_usage().
  get_cheapest_grid_pids.group_size_n_usage: get_cheapest_grid_pids().group_size_n_usage().
  _get_group_size_cost: _get_group_size_cost().
---
# Module: [`tokamax/_src/pallas/grid.py`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/grid.py)

## Functions
- `_get_group_size_cost(group_size_m: int, grid_m: int, grid_n: int, block_m_cost: int, block_n_cost: int)` — [`L23`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/grid.py#L23) — Returns the total cost for the given group size.
- `get_cheapest_grid_pids(pid: ScalarInt, *, grid_m: int, grid_n: int, block_m_cost: int, block_n_cost: int)` — [`L66`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/grid.py#L66) — Returns the grouped program IDs that minimize the total cost.
- `get_grid_pids(pid: ScalarInt, grid_m: int, grid_n: int, group_size_m: int)` — [`L38`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/grid.py#L38) — Returns the program IDs in each grid axis.
- `group_size_m_usage(group_size_m)` — [`L77`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/grid.py#L77)
- `group_size_n_usage(group_size_n)` — [`L86`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/grid.py#L86)

