---
title: 'Module: jax/_src/pallas/mosaic/interpret/utils.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/interpret/utils.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.interpret.utils`/
symbols:
  LoggingInfo.source_info: LoggingInfo#source_info.
  LoggingInfo.device_id: LoggingInfo#device_id.
  GPULoggingInfo: GPULoggingInfo#
  LoggingInfo.format: LoggingInfo#format().
  to_range: to_range().
  JaxprEnv.read: JaxprEnv#read().
  GPULoggingInfo.grid_point_coords: GPULoggingInfo#grid_point_coords.
  GPULoggingInfo.thread_id: GPULoggingInfo#thread_id.
  get_uninitialized_value: get_uninitialized_value().
  LoggingInfo: LoggingInfo#
  TPULoggingInfo: TPULoggingInfo#
  _device_id_to_logical: _device_id_to_logical().
  get_uninitialized_array: get_uninitialized_array().
  TPULoggingInfo.local_core_id: TPULoggingInfo#local_core_id.
  JaxprEnv.write: JaxprEnv#write().
  JaxprEnv.read_many: JaxprEnv#read_many().
  pad_to_block_dimension: pad_to_block_dimension().
  is_float: is_float().
  JaxprEnv.write_many: JaxprEnv#write_many().
  GPULoggingInfo.get_location_str: GPULoggingInfo#get_location_str().
  LoggingInfo.get_location_str: LoggingInfo#get_location_str().
  JaxprEnv._env: JaxprEnv#_env.
  LoggingInfo.get_source_info_str: LoggingInfo#get_source_info_str().
  Counter.get_next: Counter#get_next().
  is_int: is_int().
  device_coords_to_logical_id: device_coords_to_logical_id().
  TPULoggingInfo.get_location_str: TPULoggingInfo#get_location_str().
  Placeholder.dtype: Placeholder#dtype.
  get_next_indices: get_next_indices().
  get_indices: get_indices().
  JaxprEnv.__init__: JaxprEnv#__init__().
  Counter: Counter#
  Counter.value: Counter#value.
  Placeholder: Placeholder#
  JaxprEnv: JaxprEnv#
  JaxprEnv._sentinel_for_floating_point_values: JaxprEnv#_sentinel_for_floating_point_values.
  clip_range_to_shape: clip_range_to_shape().
  Counter.lock: Counter#lock.
  _device_id_dict_to_mesh: _device_id_dict_to_mesh().
  Placeholder.shape: Placeholder#shape.
  _transform_slice_or_index: _transform_slice_or_index().
  _compose_slice_or_index: _compose_slice_or_index().
  is_range_out_of_bounds_for_shape: is_range_out_of_bounds_for_shape().
  Counter.__init__: Counter#__init__().
---
# Module: [`jax/_src/pallas/mosaic/interpret/utils.py`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py)

## Classes
### `Counter`
- def: [`jax/_src/pallas/mosaic/interpret/utils.py:162`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L162)
- doc: A simple counter that is thread-safe.
- signature: `class Counter:`
- members:
  - `get_next(self)` — [`L169`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L169)
  - `lock` — [`L167`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L167)
  - `value` — [`L166`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L166)
- protocol/private: `__init__`[`L165`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L165)
- used by: [`dma_start`](interpret_pallas_call.md#dma_start), [`_initialize_shared_memory`](interpret_pallas_call.md#_initialize_shared_memory), [`dma_id_counter`](interpret_pallas_call.md#dma_id_counter.dma_id_counter)

### `GPULoggingInfo`  ·  implements/extends LoggingInfo
- def: [`jax/_src/pallas/mosaic/interpret/utils.py:140`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L140)
- doc: Logging info for GPU interpret mode.
- signature: `class GPULoggingInfo(LoggingInfo):`
- members:
  - `get_location_str(self)` — [`L155`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L155)
  - `grid_point_coords` — [`L148`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L148)
  - `thread_id` — [`L153`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L153)
- uses (calls/refs, reference-scoped): [`device_id`](utils.md#LoggingInfo.device_id), [`LoggingInfo`](utils.md#LoggingInfo)
- used by: [`_allocate_buffer`](../../mosaic_gpu/interpret/gpu_callbacks.md#_allocate_buffer), [`_allocate_buffer_for_all_threads`](../../mosaic_gpu/interpret/gpu_callbacks.md#_allocate_buffer_for_all_threads), [`_allocate_barriers`](../../mosaic_gpu/interpret/gpu_callbacks.md#_allocate_barriers), [`_allocate_cluster_barriers`](../../mosaic_gpu/interpret/gpu_callbacks.md#_allocate_cluster_barriers), [`__init__`](../../mosaic_gpu/interpret/gpu_callbacks.md#AsyncCopyTask.__init__), [`_get`](../../mosaic_gpu/interpret/gpu_callbacks.md#_get), [`_swap`](../../mosaic_gpu/interpret/gpu_callbacks.md#_swap), [`tcgen05_mma`](../../mosaic_gpu/interpret/gpu_callbacks.md#tcgen05_mma), [`arrive`](../../mosaic_gpu/interpret/shared_memory.md#Barrier.arrive), [`wait`](../../mosaic_gpu/interpret/shared_memory.md#Barrier.wait), [`wgmma`](../../mosaic_gpu/interpret/gpu_callbacks.md#wgmma), [`_deallocate_barrier`](../../mosaic_gpu/interpret/gpu_callbacks.md#_deallocate_barrier), [`_barrier_arrive`](../../mosaic_gpu/interpret/gpu_callbacks.md#_barrier_arrive), [`async_load_tmem`](../../mosaic_gpu/interpret/gpu_callbacks.md#async_load_tmem), [`_barrier_wait`](../../mosaic_gpu/interpret/gpu_callbacks.md#_barrier_wait), [`allocate_barrier`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.allocate_barrier), [`_deallocate_buffer`](../../mosaic_gpu/interpret/gpu_callbacks.md#_deallocate_buffer), [`allocate_cluster_barrier`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.allocate_cluster_barrier), [`wgmma_accumulator_deref`](../../mosaic_gpu/interpret/gpu_callbacks.md#wgmma_accumulator_deref), [`deallocate_barrier`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.deallocate_barrier), [`deallocate_cluster_barrier`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.deallocate_cluster_barrier), [`arrive`](../../mosaic_gpu/interpret/shared_memory.md#ClusterBarrier.arrive), [`wait`](../../mosaic_gpu/interpret/shared_memory.md#ClusterBarrier.wait), [`LoggingInfo`](utils.md#LoggingInfo), [`get_location_str`](utils.md#LoggingInfo.get_location_str), [`logging_info`](../../mosaic_gpu/interpret/gpu_callbacks.md#AsyncCopyTask.logging_info)

### `JaxprEnv`
- def: [`jax/_src/pallas/mosaic/interpret/utils.py:262`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L262)
- doc: An environment for interpreting jaxprs, mapping variables to values.
- signature: `class JaxprEnv:`
- members:
  - `read(self, var)` — [`L284`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L284)
  - `read_many(self, vars)` — [`L295`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L295)
  - `write(self, var, value)` — [`L298`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L298)
  - `write_many(self, vars, values)` — [`L303`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L303)
- protocol/private: `__init__`[`L265`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L265), `_env`[`L275`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L275), `_sentinel_for_floating_point_values`[`L272`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L272)
- uses (calls/refs, reference-scoped): [`safe_map`](../../../util.md#safe_map), [`Var`](../../../core.md#Var), [`full`](../../../lax/lax.md#full), [`val`](../../../core.md#Literal.val), [`is_float`](utils.md#is_float), [`dtype`](utils.md#Placeholder.dtype), [`Placeholder`](utils.md#Placeholder), [`shape`](utils.md#Placeholder.shape)
- used by: [`_interpret_jaxpr`](interpret_pallas_call.md#_interpret_jaxpr), [`interpret`](../../mosaic_gpu/interpret/jaxpr_interpret.md#JaxprInterpreter.interpret)

### `LoggingInfo`  ·  implements/extends ABC
- def: [`jax/_src/pallas/mosaic/interpret/utils.py:84`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L84)
- doc: Holds information for logging.
- signature: `class LoggingInfo(abc.ABC):`
- members:
  - `format(self, message: str, line_prefix: str | None = None)` — [`L100`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L100) — Formats a message for logging, across multiple lines.
  - `get_location_str(self)` — [`L91`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L91) — Returns a string representation of the location (device/core/thread).
  - `get_source_info_str(self)` — [`L95`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L95)
  - `device_id` — [`L88`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L88)
  - `source_info` — [`L87`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L87)
- uses (calls/refs, reference-scoped): [`SourceInfo`](../../../source_info_util.md#SourceInfo), [`summarize`](../../../source_info_util.md#summarize), [`GPULoggingInfo`](utils.md#GPULoggingInfo), [`TPULoggingInfo`](utils.md#TPULoggingInfo), [`get_location_str`](utils.md#GPULoggingInfo.get_location_str), [`get_location_str`](utils.md#TPULoggingInfo.get_location_str)
- used by: [`execute_read`](interpret_pallas_call.md#DMA.execute_read), [`execute_write`](interpret_pallas_call.md#DMA.execute_write), [`_allocate_buffer`](../../mosaic_gpu/interpret/gpu_callbacks.md#_allocate_buffer), [`_allocate_buffer_for_all_threads`](../../mosaic_gpu/interpret/gpu_callbacks.md#_allocate_buffer_for_all_threads), [`get`](interpret_pallas_call.md#get), [`_allocate_barriers`](../../mosaic_gpu/interpret/gpu_callbacks.md#_allocate_barriers), [`_allocate_cluster_barriers`](../../mosaic_gpu/interpret/gpu_callbacks.md#_allocate_cluster_barriers), [`get_buffer_content`](shared_memory.md#GenericSharedMemory.get_buffer_content), [`_allocate_buffer`](interpret_pallas_call.md#_allocate_buffer), [`__init__`](../../mosaic_gpu/interpret/gpu_callbacks.md#AsyncCopyTask.__init__), [`wait`](shared_memory.md#Semaphore.wait), [`_get`](../../mosaic_gpu/interpret/gpu_callbacks.md#_get), [`_swap`](../../mosaic_gpu/interpret/gpu_callbacks.md#_swap), [`tcgen05_mma`](../../mosaic_gpu/interpret/gpu_callbacks.md#tcgen05_mma), [`arrive`](../../mosaic_gpu/interpret/shared_memory.md#Barrier.arrive), [`swap_buffer_content`](shared_memory.md#GenericSharedMemory.swap_buffer_content), [`wait`](../../mosaic_gpu/interpret/shared_memory.md#Barrier.wait), [`store_buffer_content`](shared_memory.md#GenericSharedMemory.store_buffer_content), [`wgmma`](../../mosaic_gpu/interpret/gpu_callbacks.md#wgmma), [`store`](interpret_pallas_call.md#store), [`_deallocate_barrier`](../../mosaic_gpu/interpret/gpu_callbacks.md#_deallocate_barrier), [`swap`](interpret_pallas_call.md#swap), [`_barrier_arrive`](../../mosaic_gpu/interpret/gpu_callbacks.md#_barrier_arrive), [`async_load_tmem`](../../mosaic_gpu/interpret/gpu_callbacks.md#async_load_tmem), [`deallocate_buffer`](shared_memory.md#GenericSharedMemory.deallocate_buffer), [`_barrier_wait`](../../mosaic_gpu/interpret/gpu_callbacks.md#_barrier_wait), [`allocate_barrier`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.allocate_barrier), [`signal`](shared_memory.md#Semaphore.signal), [`_deallocate_buffer`](../../mosaic_gpu/interpret/gpu_callbacks.md#_deallocate_buffer), [`allocate_cluster_barrier`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.allocate_cluster_barrier), [`wgmma_accumulator_deref`](../../mosaic_gpu/interpret/gpu_callbacks.md#wgmma_accumulator_deref), [`allocate_buffer`](shared_memory.md#GenericSharedMemory.allocate_buffer), [`GPULoggingInfo`](utils.md#GPULoggingInfo), [`deallocate_barrier`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.deallocate_barrier), [`deallocate_cluster_barrier`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.deallocate_cluster_barrier), [`arrive`](../../mosaic_gpu/interpret/shared_memory.md#ClusterBarrier.arrive), [`_deallocate_buffer`](interpret_pallas_call.md#_deallocate_buffer), [`dma_wait`](interpret_pallas_call.md#dma_wait), [`semaphore_signal`](interpret_pallas_call.md#semaphore_signal), [`semaphore_wait`](interpret_pallas_call.md#semaphore_wait)  (+4 more)

### `Placeholder`
- def: [`jax/_src/pallas/mosaic/interpret/utils.py:255`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L255)
- doc: Placeholder for use in `JaxprEnv` below instead of storing a concrete value.
- signature: `class Placeholder:`
- members:
  - `dtype` — [`L259`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L259)
  - `shape` — [`L258`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L258)
- uses (calls/refs, reference-scoped): [`dtype`](../../../../numpy/__init__.pyi.md#dtype)
- used by: [`read`](utils.md#JaxprEnv.read), [`write`](utils.md#JaxprEnv.write)

### `TPULoggingInfo`  ·  implements/extends LoggingInfo
- def: [`jax/_src/pallas/mosaic/interpret/utils.py:130`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L130)
- doc: Logging info for TPU interpret mode.
- signature: `class TPULoggingInfo(LoggingInfo):`
- members:
  - `get_location_str(self)` — [`L135`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L135)
  - `local_core_id` — [`L133`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L133)
- uses (calls/refs, reference-scoped): [`device_id`](utils.md#LoggingInfo.device_id), [`LoggingInfo`](utils.md#LoggingInfo)
- used by: [`execute_read`](interpret_pallas_call.md#DMA.execute_read), [`execute_write`](interpret_pallas_call.md#DMA.execute_write), [`get`](interpret_pallas_call.md#get), [`_allocate_buffer`](interpret_pallas_call.md#_allocate_buffer), [`store`](interpret_pallas_call.md#store), [`swap`](interpret_pallas_call.md#swap), [`_deallocate_buffer`](interpret_pallas_call.md#_deallocate_buffer), [`dma_wait`](interpret_pallas_call.md#dma_wait), [`semaphore_signal`](interpret_pallas_call.md#semaphore_signal), [`semaphore_wait`](interpret_pallas_call.md#semaphore_wait), [`LoggingInfo`](utils.md#LoggingInfo), [`get_location_str`](utils.md#LoggingInfo.get_location_str)

## Functions
- `_compose_slice_or_index(slice_or_idx1, slice_or_idx2)` — [`L317`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L317)
- `_device_id_dict_to_mesh(device_id_dict, axis_sizes, axis_indices)` — [`L177`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L177)
- `_device_id_to_logical(device_id, device_id_type, axis_sizes, axis_indices)` — [`L235`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L235)
- `_transform_slice_or_index(slice_or_idx)` — [`L307`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L307)
- `clip_range_to_shape(rnge: tuple[slice | int, ...], shape: tuple[int, ...])` — [`L389`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L389) — Clips `slice`s in `rnge` to the `shape`. Returns None if `rnge` is entirely out of bounds.
- `device_coords_to_logical_id(device_coords, axis_sizes, axis_indices)` — [`L218`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L218)
- `get_indices(grid, loop_index)` — [`L418`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L418)
- `get_next_indices(grid, indices)` — [`L408`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L408)
- `get_uninitialized_array(shape, dtype, uninitialized_memory: Literal["nan", "zero"])` — [`L46`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L46)
- `get_uninitialized_value(dtype, uninitialized_memory: Literal["nan", "zero"])` — [`L31`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L31)
- `is_float(dtype)` — [`L250`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L250)
- `is_int(dtype)` — [`L246`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L246)
- `is_range_out_of_bounds_for_shape(rnge: tuple[slice | int, ...], shape: tuple[int, ...])` — [`L361`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L361) — Returns whether `rnge` is at least partially out of bounds for `shape`.
- `pad_to_block_dimension(value, block_shape, uninitialized_memory: Literal["nan", "zero"])` — [`L56`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L56) — Pads values so the shape evenly divides into block dimensions.
- `to_range(transforms)` — [`L351`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/utils.py#L351)

