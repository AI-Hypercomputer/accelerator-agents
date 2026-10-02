---
title: 'Module: jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic_gpu.interpret.interpret_pallas_call`/
symbols:
  interpret_pallas_call: interpret_pallas_call().
  _allocate_buffers_for_outputs: _allocate_buffers_for_outputs().
  interpret_pallas_call._kernel: interpret_pallas_call()._kernel().
  _get_kernel_buffers: _get_kernel_buffers().
  _allocate_buffers_for_inputs: _allocate_buffers_for_inputs().
  _get_grid_and_cluster_dims_and_num_threads: _get_grid_and_cluster_dims_and_num_threads().
  _get_outputs: _get_outputs().
  _load_and_store_between_allocation_keys: _load_and_store_between_allocation_keys().
  _get_grid_bounds: _get_grid_bounds().
  _copy_from_gmem_buffers: _copy_from_gmem_buffers().
  _copy_to_gmem_buffers: _copy_to_gmem_buffers().
  interpret_pallas_call._grid_loop_body: interpret_pallas_call()._grid_loop_body().
  AllocationKeyAndValue.key: AllocationKeyAndValue#key.
  AllocationKeyAndValue.value: AllocationKeyAndValue#value.
  get_races: get_races().
  AllocationKeyAndValue.shape: AllocationKeyAndValue#shape().
  AllocationKeyAndValue: AllocationKeyAndValue#
  reset_gpu_interpret_mode_state: reset_gpu_interpret_mode_state().
---
# Module: [`jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py)

## Classes
### `AllocationKeyAndValue`
- def: [`jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py:116`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L116)
- signature: `class AllocationKeyAndValue:`
- members:
  - `shape(self)` — [`L121`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L121)
  - `key` — [`L117`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L117)
  - `value` — [`L118`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L118)
- uses (calls/refs, reference-scoped): [`Array`](../../../basearray.md#Array), [`shape`](../../../basearray.md#Array.shape)
- used by: [`interpret_pallas_call`](interpret_pallas_call.md#interpret_pallas_call), [`_allocate_buffers_for_outputs`](interpret_pallas_call.md#_allocate_buffers_for_outputs), [`_kernel`](interpret_pallas_call.md#interpret_pallas_call._kernel), [`_get_outputs`](interpret_pallas_call.md#_get_outputs)

## Functions
- `_allocate_buffers_for_inputs(token: jax.Array, device_id: int, invars: Sequence[Any], inputs: Sequence[jax.Array])` — [`L84`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L84) — Allocates `GMEM` buffers for the `inputs` of a `pallas_call`.
- `_allocate_buffers_for_outputs(token, device_id: int, num_threads: int, input_output_aliases: tuple[tuple[int, int], ...], grid_mapping: pallas_core.GridMapping, input_buffer_keys: Sequence[jax.Array], input_vals: Sequence[jax.Array], interpret_params: InterpretGPUParams)` — [`L125`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L125) — Allocates `GMEM` buffers for `pallas_call` outputs, respecting aliased inputs.
- `_copy_from_gmem_buffers(token, device_id: int, grid_point_coords: jax.Array, thread_id: jax.Array, avals: Sequence[Any], gmem_buffer_keys: Sequence[jax.Array], target_buffer_keys: Sequence[jax.Array], transforms)` — [`L311`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L311)
- `_copy_to_gmem_buffers(token, device_id: int, grid_point_coords: jax.Array, thread_id: jax.Array, avals: Sequence[Any], source_buffer_keys: Sequence[jax.Array], gmem_buffer_keys: Sequence[jax.Array], transforms)` — [`L339`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L339)
- `_get_grid_and_cluster_dims_and_num_threads(grid_mapping: pallas_core.GridMapping, mesh: plgpu.Mesh | None)` — [`L60`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L60)
- `_get_grid_bounds(grid_mapping: pallas_core.GridMapping)` — [`L46`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L46)
- `_get_kernel_buffers(token, device_id: int, num_threads: int, grid_mapping: pallas_core.GridMapping, invars: Sequence[Any], arg_transforms: tuple[tuple[state_types.Transform, ...], ...], input_buffer_keys: Sequence[jax.Array], output_buffer_keys: Sequence[jax.Array], interpret_params: InterpretGPUParams)` — [`L190`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L190) — Collects buffers to be passed to the kernel from `pallas_call` input/output buffers.
- `_get_outputs(token, device_id: int, output_buffers: Sequence[AllocationKeyAndValue])` — [`L255`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L255) — Reads and returns values from the allocated output buffers.
- `_grid_loop_body(loop_idx: int, token)` — [`L533`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L533)
- `_kernel(thread_id, token, grid_point_coords)` — [`L470`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L470)
- `_load_and_store_between_allocation_keys(*, token: jax.Array, device_id: int, grid_point_coords: jax.Array, thread_id: jax.Array, share_and_dtype: Any, load_allocation_key: jax.Array, store_allocation_key: jax.Array, transform)` — [`L277`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L277)
- `get_races()` — [`L38`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L38)
- `interpret_pallas_call(*args, jaxpr: jax_core.Jaxpr, debug: bool, input_output_aliases: tuple[tuple[int, int], ...], grid_mapping: pallas_core.GridMapping, mesh: plgpu.Mesh | None, compiler_params: Mapping[str, Any], cost_estimate: pallas_core.CostEstimate, out_avals: tuple[jax_core.AbstractValue, ...], interpret_params: InterpretGPUParams, metadata: Mapping[str, str] | None, kernel_arg_transforms: tuple[tuple[state_types.Transform, ...], ...] = (), **kwargs)` — [`L367`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L367)
- `reset_gpu_interpret_mode_state()` — [`L42`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/interpret_pallas_call.py#L42)

