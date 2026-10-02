---
title: 'Module: jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic_gpu.interpret.jaxpr_interpret`/
symbols:
  JaxprInterpreter.interpret: JaxprInterpreter#interpret().
  JaxprInterpreter._allocate_for_aval: JaxprInterpreter#_allocate_for_aval().
  DeviceInfo.device_id: DeviceInfo#device_id().
  JaxprInterpreter._deallocate_for_aval: JaxprInterpreter#_deallocate_for_aval().
  JaxprInterpreter._interpret_axis_index_p: JaxprInterpreter#_interpret_axis_index_p().
  JaxprInterpreter._interpret_run_scoped_p: JaxprInterpreter#_interpret_run_scoped_p().
  JaxprInterpreter._interpret_copy_gmem_to_smem_p: JaxprInterpreter#_interpret_copy_gmem_to_smem_p().
  JaxprInterpreter._interpret_copy_smem_to_gmem_p: JaxprInterpreter#_interpret_copy_smem_to_gmem_p().
  JaxprInterpreter._interpret_tcgen05_mma_p: JaxprInterpreter#_interpret_tcgen05_mma_p().
  JaxprInterpreter._interpret_wgmma_ref_p: JaxprInterpreter#_interpret_wgmma_ref_p().
  JaxprInterpreter._interpret_wgmma_accumulator_deref_p: JaxprInterpreter#_interpret_wgmma_accumulator_deref_p().
  JaxprInterpreter._interpret_async_load_tmem_p: JaxprInterpreter#_interpret_async_load_tmem_p().
  JaxprInterpreter._interpret_wait_smem_to_gmem_p: JaxprInterpreter#_interpret_wait_smem_to_gmem_p().
  JaxprInterpreter._interpret_commit_smem_p: JaxprInterpreter#_interpret_commit_smem_p().
  JaxprInterpreter.thread_id: JaxprInterpreter#thread_id.
  JaxprInterpreter.device_info: JaxprInterpreter#device_info.
  JaxprInterpreter._interpret_barrier_primitive: JaxprInterpreter#_interpret_barrier_primitive().
  JaxprInterpreter._interpret_get_p: JaxprInterpreter#_interpret_get_p().
  JaxprInterpreter._interpret_swap_p: JaxprInterpreter#_interpret_swap_p().
  JaxprInterpreter.grid_point_coords: JaxprInterpreter#grid_point_coords.
  JaxprInterpreter.mesh: JaxprInterpreter#mesh.
  _get_barrier_allocation_key_from_inval: _get_barrier_allocation_key_from_inval().
  _raise_if_unsupported_memory_space: _raise_if_unsupported_memory_space().
  apply_unswizzle_and_untile: apply_unswizzle_and_untile().
  get_uninitialized_array: get_uninitialized_array().
  JaxprInterpreter.thread_id_in_block: JaxprInterpreter#thread_id_in_block().
  JaxprInterpreter.are_thread_cluster_axes_collective: JaxprInterpreter#are_thread_cluster_axes_collective().
  JaxprInterpreter.thread_cluster_coords: JaxprInterpreter#thread_cluster_coords().
  JaxprInterpreter.cluster_coords: JaxprInterpreter#cluster_coords().
  JaxprInterpreter._interpret_scan_p: JaxprInterpreter#_interpret_scan_p().
  JaxprInterpreter._interpret_while_p: JaxprInterpreter#_interpret_while_p().
  JaxprInterpreter._interpret_arithmetic_primitive: JaxprInterpreter#_interpret_arithmetic_primitive().
  JaxprInterpreter.num_threads_per_block: JaxprInterpreter#num_threads_per_block().
  JaxprInterpreter.thread_cluster_shape: JaxprInterpreter#thread_cluster_shape().
  JaxprInterpreter._interpret_cond_p: JaxprInterpreter#_interpret_cond_p().
  JaxprInterpreter._interpret_barrier_arrive_p: JaxprInterpreter#_interpret_barrier_arrive_p().
  JaxprInterpreter._interpret_barrier_wait_p: JaxprInterpreter#_interpret_barrier_wait_p().
  DeviceInfo.__init__: DeviceInfo#__init__().
  DeviceInfo.axis_indices: DeviceInfo#axis_indices.
  JaxprInterpreter.interpret_params: JaxprInterpreter#interpret_params.
  _raise_if_unsupported_collective_axes: _raise_if_unsupported_collective_axes().
  _get_index_for_barrier_allocation_key: _get_index_for_barrier_allocation_key().
  JaxprInterpreter.num_concurrent_threads: JaxprInterpreter#num_concurrent_threads().
  JaxprInterpreter.is_thread_block_axis_collective: JaxprInterpreter#is_thread_block_axis_collective().
  JaxprInterpreter._scan_body: JaxprInterpreter#_scan_body().
  DeviceInfo.axis_sizes: DeviceInfo#axis_sizes.
  JaxprInterpreter.cluster_dims: JaxprInterpreter#cluster_dims.
  DeviceInfo.num_devices: DeviceInfo#num_devices().
  _SENTINEL: _SENTINEL.
  JaxprInterpreter._body: JaxprInterpreter#_body().
  DeviceInfo: DeviceInfo#
  _get_num_threads_sharing_collective_allocation: _get_num_threads_sharing_collective_allocation().
  JaxprInterpreter: JaxprInterpreter#
  JaxprInterpreter.compiler_params: JaxprInterpreter#compiler_params.
---
# Module: [`jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py)

## Classes
### `DeviceInfo`
- def: [`jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py:41`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L41)
- doc: Information about the device that is being interpreted.
- signature: `class DeviceInfo:`
- members:
  - `device_id(self)` — [`L60`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L60) — Computes the logical ID of the device being interpreted.
  - `num_devices(self)` — [`L67`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L67) — Computes the number of (SPMD) devices.
  - `axis_indices` — [`L45`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L45)
  - `axis_sizes` — [`L47`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L47)
- protocol/private: `__init__`[`L49`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L49)
- uses (calls/refs, reference-scoped): [`axis_index`](../../../lax/parallel.md#axis_index), [`AxisName`](../../../core.md#AxisName), [`get_axis_env`](../../../core.md#get_axis_env), [`device_coords_to_logical_id`](../../mosaic/interpret/utils.md#device_coords_to_logical_id)
- used by: [`_allocate_for_aval`](jaxpr_interpret.md#JaxprInterpreter._allocate_for_aval), [`interpret_pallas_call`](interpret_pallas_call.md#interpret_pallas_call), [`_kernel`](interpret_pallas_call.md#interpret_pallas_call._kernel), [`_deallocate_for_aval`](jaxpr_interpret.md#JaxprInterpreter._deallocate_for_aval), [`_interpret_axis_index_p`](jaxpr_interpret.md#JaxprInterpreter._interpret_axis_index_p), [`_interpret_copy_gmem_to_smem_p`](jaxpr_interpret.md#JaxprInterpreter._interpret_copy_gmem_to_smem_p), [`_interpret_copy_smem_to_gmem_p`](jaxpr_interpret.md#JaxprInterpreter._interpret_copy_smem_to_gmem_p), [`_interpret_tcgen05_mma_p`](jaxpr_interpret.md#JaxprInterpreter._interpret_tcgen05_mma_p), [`_interpret_wgmma_ref_p`](jaxpr_interpret.md#JaxprInterpreter._interpret_wgmma_ref_p), [`_interpret_async_load_tmem_p`](jaxpr_interpret.md#JaxprInterpreter._interpret_async_load_tmem_p), [`_interpret_commit_smem_p`](jaxpr_interpret.md#JaxprInterpreter._interpret_commit_smem_p), [`_interpret_wait_smem_to_gmem_p`](jaxpr_interpret.md#JaxprInterpreter._interpret_wait_smem_to_gmem_p), [`_interpret_wgmma_accumulator_deref_p`](jaxpr_interpret.md#JaxprInterpreter._interpret_wgmma_accumulator_deref_p), [`device_info`](jaxpr_interpret.md#JaxprInterpreter.device_info), [`_interpret_barrier_primitive`](jaxpr_interpret.md#JaxprInterpreter._interpret_barrier_primitive), [`_interpret_get_p`](jaxpr_interpret.md#JaxprInterpreter._interpret_get_p), [`_interpret_swap_p`](jaxpr_interpret.md#JaxprInterpreter._interpret_swap_p)

### `JaxprInterpreter`
- def: [`jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py:209`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L209)
- doc: Interprets a jaxpr by replacing memory operations with (GPU) callbacks.
- signature: `class JaxprInterpreter:`
- members:
  - `are_thread_cluster_axes_collective(self, collective_axes: tuple[jax_core.AxisName, ...])` — [`L271`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L271) — Returns a tuple of booleans indicating whether each thread cluster axis is collective.
  - `cluster_coords(self)` — [`L254`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L254)
  - `interpret(self, jaxpr, token, *args)` — [`L948`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L948)
  - `is_thread_block_axis_collective(self, collective_axes: tuple[jax_core.AxisName, ...])` — [`L310`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L310) — Returns whether the axis corresponding to the threads in a block is collective.
  - `num_concurrent_threads(self)` — [`L246`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L246)
  - `num_threads_per_block(self)` — [`L239`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L239)
  - `thread_cluster_coords(self)` — [`L264`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L264) — Returns the coordinates of the thread along the cluster axes *and* within the block.
  - `thread_cluster_shape(self)` — [`L259`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L259) — Returns the number of threads along the cluster axes *and* within a block.
  - `thread_id_in_block(self)` — [`L250`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L250)
  - `cluster_dims` — [`L213`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L213)
  - `compiler_params` — [`L235`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L235)
  - `device_info` — [`L234`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L234)
  - `grid_point_coords` — [`L212`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L212)
  - `interpret_params` — [`L236`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L236)
  - `mesh` — [`L233`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L233)
  - `thread_id` — [`L231`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L231)
- protocol/private: `_allocate_for_aval`[`L379`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L379), `_body`[`L625`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L625), `_deallocate_for_aval`[`L495`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L495), `_interpret_arithmetic_primitive`[`L690`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L690), `_interpret_async_load_tmem_p`[`L899`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L899), `_interpret_axis_index_p`[`L316`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L316), `_interpret_barrier_arrive_p`[`L674`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L674), `_interpret_barrier_primitive`[`L640`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L640), `_interpret_barrier_wait_p`[`L682`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L682), `_interpret_commit_smem_p`[`L933`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L933), `_interpret_cond_p`[`L578`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L578), `_interpret_copy_gmem_to_smem_p`[`L711`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L711), `_interpret_copy_smem_to_gmem_p`[`L749`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L749), `_interpret_get_p`[`L338`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L338), `_interpret_run_scoped_p`[`L374`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L374), `_interpret_scan_p`[`L593`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L593), `_interpret_swap_p`[`L355`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L355), `_interpret_tcgen05_mma_p`[`L837`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L837), `_interpret_wait_smem_to_gmem_p`[`L914`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L914), `_interpret_wgmma_accumulator_deref_p`[`L820`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L820), `_interpret_wgmma_ref_p`[`L789`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L789), `_interpret_while_p`[`L615`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L615), `_scan_body`[`L602`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L602)
- uses (calls/refs, reference-scoped): [`Array`](../../../basearray.md#Array), [`shape`](../../../basearray.md#Array.shape), [`AbstractRef`](../../../state/types.md#AbstractRef), [`ShapedArray`](../../../core.md#ShapedArray), [`split_list`](../../../util.md#split_list), [`int32`](../../../../numpy/__init__.pyi.md#int32.int32), [`Jaxpr`](../../../core.md#Jaxpr), [`safe_zip`](../../../util.md#safe_zip), [`Transform`](../../../state/types.md#Transform), [`scan_p`](../../../lax/control_flow/loops.md#scan_p), [`cond_p`](../../../lax/control_flow/conditionals.md#cond_p), [`scan`](../../../lax/control_flow/loops.md#scan), [`while_p`](../../../lax/control_flow/loops.md#while_p), [`io_callback`](../../../callback.md#io_callback), [`while_loop`](../../../lax/control_flow/loops.md#while_loop), [`PyTreeDef`](../../../tree_util.md#PyTreeDef), [`reshape`](../../../basearray.md#Array.reshape), [`rem`](../../../lax/lax.md#rem), [`array`](../../../../numpy/__init__.pyi.md#array), [`MemorySpace`](../core.md#MemorySpace), [`HostAllocationKey`](shared_memory.md#HostAllocationKey), [`get_p`](../../../state/primitives.md#get_p), [`swap_p`](../../../state/primitives.md#swap_p), [`tcgen05_mma`](gpu_callbacks.md#tcgen05_mma), [`SourceInfo`](../../../source_info_util.md#SourceInfo), [`axis_index_p`](../../../lax/parallel.md#axis_index_p), [`wgmma`](gpu_callbacks.md#wgmma), [`async_load_tmem`](gpu_callbacks.md#async_load_tmem), [`device_id`](jaxpr_interpret.md#DeviceInfo.device_id), [`AxisName`](../../../core.md#AxisName), [`TOKEN_SHAPE_DTYPE`](gpu_callbacks.md#TOKEN_SHAPE_DTYPE), [`copy_smem_to_gmem`](gpu_callbacks.md#copy_smem_to_gmem), [`run_scoped_p`](../../primitives.md#run_scoped_p), [`copy_gmem_to_smem`](gpu_callbacks.md#copy_gmem_to_smem), [`wgmma_accumulator_deref`](gpu_callbacks.md#wgmma_accumulator_deref), [`call_allocate_barriers`](gpu_callbacks.md#call_allocate_barriers), [`call_allocate_cluster_barriers`](gpu_callbacks.md#call_allocate_cluster_barriers), [`shape`](../../../core.md#ShapeDtypeStruct.shape), [`copy_gmem_to_smem_p`](../primitives.md#copy_gmem_to_smem_p), [`copy_smem_to_gmem_p`](../primitives.md#copy_smem_to_gmem_p)  (+60 more)
- used by: [`_kernel`](interpret_pallas_call.md#interpret_pallas_call._kernel)

## Functions
- `_get_barrier_allocation_key_from_inval(inval, transforms_treedef, transforms_leaves)` — [`L137`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L137)
- `_get_index_for_barrier_allocation_key(transforms_treedef, transforms_leaves)` — [`L114`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L114)
- `_get_num_threads_sharing_collective_allocation(axes_dims: tuple[int, ...], is_last_thread_axis_collective: bool)` — [`L171`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L171) — Returns the number of threads that share a collective allocation.
- `_raise_if_unsupported_collective_axes(mesh: plgpu.Mesh | None, is_collective_by_thread_cluster_axis: tuple[bool, ...])` — [`L85`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L85)
- `_raise_if_unsupported_memory_space(space: mosaic_gpu_core.MemorySpace | None)` — [`L72`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L72)
- `apply_unswizzle_and_untile(transforms: tuple[state_types.Transform, ...], aval: jax_core.AbstractValue)` — [`L185`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L185)
- `get_uninitialized_array(shape: tuple[int, ...], dtype: jnp.dtype, memory_space: mosaic_gpu_core.MemorySpace, uninitialized_memory: Literal["nan", "zero"])` — [`L196`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L196)

## Module values
- `_SENTINEL` — [`L182`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/jaxpr_interpret.py#L182)

