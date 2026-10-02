---
title: 'Module: jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic_gpu.interpret.gpu_callbacks`/
symbols:
  _allocate_buffer_for_all_threads: _allocate_buffer_for_all_threads().
  _allocate_buffer: _allocate_buffer().
  _allocate_barriers: _allocate_barriers().
  _allocate_cluster_barriers: _allocate_cluster_barriers().
  AsyncCopyTask.__init__: AsyncCopyTask#__init__().
  _get: _get().
  _swap: _swap().
  tcgen05_mma: tcgen05_mma().
  wgmma: wgmma().
  AsyncCopySmemToGmemTask.pre_read: AsyncCopySmemToGmemTask#pre_read().
  AsyncCopySmemToGmemTask.post_read: AsyncCopySmemToGmemTask#post_read().
  _initialize_shared_memory: _initialize_shared_memory().
  _deallocate_barrier: _deallocate_barrier().
  AsyncCopyTask.__call__: AsyncCopyTask#__call__().
  AsyncCopyGmemToSmemTask.pre_read: AsyncCopyGmemToSmemTask#pre_read().
  AsyncCopyGmemToSmemTask.post_read: AsyncCopyGmemToSmemTask#post_read().
  _barrier_arrive: _barrier_arrive().
  async_load_tmem: async_load_tmem().
  _get_shared_memory: _get_shared_memory().
  TOKEN_SHAPE_DTYPE: TOKEN_SHAPE_DTYPE.
  _barrier_wait: _barrier_wait().
  copy_smem_to_gmem: copy_smem_to_gmem().
  _deallocate_buffer: _deallocate_buffer().
  wgmma_accumulator_deref: wgmma_accumulator_deref().
  copy_gmem_to_smem: copy_gmem_to_smem().
  call_allocate_barriers: call_allocate_barriers().
  call_allocate_cluster_barriers: call_allocate_cluster_barriers().
  AsyncCopyGmemToSmemTask.__init__: AsyncCopyGmemToSmemTask#__init__().
  call_allocate_buffer_for_all_threads: call_allocate_buffer_for_all_threads().
  call_make_allocation_request_array: call_make_allocation_request_array().
  get_memory_space_idx: get_memory_space_idx().
  _make_allocation_request_array: _make_allocation_request_array().
  call_allocate_buffer: call_allocate_buffer().
  AsyncCopyGmemToSmemTask.post_write: AsyncCopyGmemToSmemTask#post_write().
  AsyncCopySmemToGmemTask.__init__: AsyncCopySmemToGmemTask#__init__().
  AsyncCopySmemToGmemTask.post_write: AsyncCopySmemToGmemTask#post_write().
  get_races: get_races().
  _handle_out_of_bounds_read: _handle_out_of_bounds_read().
  _remove_noop_transforms: _remove_noop_transforms().
  call_initialize_shared_memory: call_initialize_shared_memory().
  call_deallocate_buffer: call_deallocate_buffer().
  _is_dynamic: _is_dynamic().
  call_deallocate_barrier: call_deallocate_barrier().
  call_barrier_wait: call_barrier_wait().
  call_barrier_arrive: call_barrier_arrive().
  _races._races: _races._races.
  call_get: call_get().
  call_update_clocks_for_device_barrier: call_update_clocks_for_device_barrier().
  call_swap: call_swap().
  _shared_memory._shared_memory: _shared_memory._shared_memory.
  wait_smem_to_gmem: wait_smem_to_gmem().
  commit_smem: commit_smem().
  IDX_BY_GPU_MEMORY_SPACE.IDX_BY_GPU_MEMORY_SPACE: IDX_BY_GPU_MEMORY_SPACE.IDX_BY_GPU_MEMORY_SPACE.
  AsyncCopyGmemToSmemTask.clock: AsyncCopyGmemToSmemTask#clock.
  AsyncCopySmemToGmemTask.clock: AsyncCopySmemToGmemTask#clock.
  reset_gpu_interpret_mode_state: reset_gpu_interpret_mode_state().
  call_clean_up_shared_memory: call_clean_up_shared_memory().
  _update_clocks_for_device_barrier: _update_clocks_for_device_barrier().
  call_assert_no_barriers_allocated: call_assert_no_barriers_allocated().
  AsyncCopyTask.pre_read: AsyncCopyTask#pre_read().
  AsyncCopyTask.post_read: AsyncCopyTask#post_read().
  AsyncCopyTask.post_write: AsyncCopyTask#post_write().
  AsyncCopyGmemToSmemTask.smem_commit_clock: AsyncCopyGmemToSmemTask#smem_commit_clock.
  AsyncCopySmemToGmemTask.smem_commit_clock: AsyncCopySmemToGmemTask#smem_commit_clock.
  kernel_thread_finished: kernel_thread_finished().
  is_gmem_memory_space: is_gmem_memory_space().
  AsyncCopyTask.device_id: AsyncCopyTask#device_id.
  AsyncCopyTask.thread_id: AsyncCopyTask#thread_id.
  AsyncCopyTask.source_info: AsyncCopyTask#source_info.
  _validate_transforms: _validate_transforms().
  AsyncCopyTask: AsyncCopyTask#
  AsyncCopyTask.src_allocation_key: AsyncCopyTask#src_allocation_key.
  AsyncCopyTask.dst_allocation_key: AsyncCopyTask#dst_allocation_key.
  AsyncCopyTask.logging_info: AsyncCopyTask#logging_info.
  _clear_shared_memory: _clear_shared_memory().
  _clean_up_shared_memory: _clean_up_shared_memory().
  _assert_no_barriers_allocated: _assert_no_barriers_allocated().
  NOOP_TRANSFORMS: NOOP_TRANSFORMS.
  GPU_MEMORY_SPACE_BY_IDX.GPU_MEMORY_SPACE_BY_IDX: GPU_MEMORY_SPACE_BY_IDX.GPU_MEMORY_SPACE_BY_IDX.
  AsyncCopySmemToGmemTask.read_clock: AsyncCopySmemToGmemTask#read_clock.
  AsyncCopySmemToGmemTask.write_clock: AsyncCopySmemToGmemTask#write_clock.
  AsyncCopyGmemToSmemTask.barrier: AsyncCopyGmemToSmemTask#barrier.
  AsyncCopyTask.src_transforms: AsyncCopyTask#src_transforms.
  AsyncCopyTask.dst_transforms: AsyncCopyTask#dst_transforms.
  AsyncCopyTask.grid_point_coords: AsyncCopyTask#grid_point_coords.
  AsyncCopyGmemToSmemTask: AsyncCopyGmemToSmemTask#
  AsyncCopySmemToGmemTask: AsyncCopySmemToGmemTask#
  _shared_memory_init_lock: _shared_memory_init_lock.
  ordering_barrier: ordering_barrier().
  get_thread_id_for_collective_allocation_key: get_thread_id_for_collective_allocation_key().
  AsyncCopyTask.tma_thread_id: AsyncCopyTask#tma_thread_id.
  AsyncCopyTask.data: AsyncCopyTask#data.
---
# Module: [`jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py)

## Classes
### `AsyncCopyGmemToSmemTask`  ·  implements/extends AsyncCopyTask
- def: [`jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py:1331`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1331)
- doc: An async task representing a GMEM -&gt; SMEM TMA memory copy.
- signature: `class AsyncCopyGmemToSmemTask(AsyncCopyTask):`
- members:
  - `post_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory)` — [`L1381`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1381)
  - `post_write(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory)` — [`L1396`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1396)
  - `pre_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory)` — [`L1365`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1365)
  - `barrier` — [`L1334`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1334)
  - `clock` — [`L1335`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1335)
  - `smem_commit_clock` — [`L1336`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1336)
- protocol/private: `__init__`[`L1338`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1338)
- uses (calls/refs, reference-scoped): [`Array`](../../../basearray.md#Array), [`HostAllocationKey`](shared_memory.md#HostAllocationKey), [`__init__`](gpu_callbacks.md#AsyncCopyTask.__init__), [`arrive`](shared_memory.md#Barrier.arrive), [`SourceInfo`](../../../source_info_util.md#SourceInfo), [`VectorClock`](../../mosaic/interpret/vector_clock.md#VectorClock), [`_get_shared_memory`](gpu_callbacks.md#_get_shared_memory), [`check_write`](../../mosaic/interpret/race_detection_state.md#RaceDetectionState.check_write), [`check_read`](../../mosaic/interpret/race_detection_state.md#RaceDetectionState.check_read), [`detect_races`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.detect_races), [`to_range`](../../mosaic/interpret/utils.md#to_range), [`GPUSharedMemory`](shared_memory.md#GPUSharedMemory), [`copy_vector_clock`](../../mosaic/interpret/vector_clock.md#copy_vector_clock), [`get_races`](gpu_callbacks.md#get_races), [`inc_vector_clock`](../../mosaic/interpret/vector_clock.md#inc_vector_clock), [`Barrier`](shared_memory.md#Barrier), [`get_barrier`](shared_memory.md#GPUSharedMemory.get_barrier), [`device_id`](gpu_callbacks.md#AsyncCopyTask.device_id), [`thread_id`](gpu_callbacks.md#AsyncCopyTask.thread_id), [`source_info`](gpu_callbacks.md#AsyncCopyTask.source_info), [`AsyncCopyTask`](gpu_callbacks.md#AsyncCopyTask), [`dst_allocation_key`](gpu_callbacks.md#AsyncCopyTask.dst_allocation_key), [`logging_info`](gpu_callbacks.md#AsyncCopyTask.logging_info), [`src_allocation_key`](gpu_callbacks.md#AsyncCopyTask.src_allocation_key), [`dst_transforms`](gpu_callbacks.md#AsyncCopyTask.dst_transforms), [`src_transforms`](gpu_callbacks.md#AsyncCopyTask.src_transforms)
- used by: [`copy_gmem_to_smem`](gpu_callbacks.md#copy_gmem_to_smem), [`post_read`](gpu_callbacks.md#AsyncCopyTask.post_read), [`post_write`](gpu_callbacks.md#AsyncCopyTask.post_write), [`pre_read`](gpu_callbacks.md#AsyncCopyTask.pre_read), [`AsyncCopyTask`](gpu_callbacks.md#AsyncCopyTask)

### `AsyncCopySmemToGmemTask`  ·  implements/extends AsyncCopyTask
- def: [`jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py:1405`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1405)
- doc: An async task representing a SMEM -&gt; GMEM TMA memory copy.
- signature: `class AsyncCopySmemToGmemTask(AsyncCopyTask):`
- members:
  - `post_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory)` — [`L1453`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1453)
  - `post_write(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory)` — [`L1467`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1467)
  - `pre_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory)` — [`L1438`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1438)
  - `clock` — [`L1408`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1408)
  - `read_clock` — [`L1410`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1410)
  - `smem_commit_clock` — [`L1409`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1409)
  - `write_clock` — [`L1411`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1411)
- protocol/private: `__init__`[`L1413`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1413)
- uses (calls/refs, reference-scoped): [`Array`](../../../basearray.md#Array), [`HostAllocationKey`](shared_memory.md#HostAllocationKey), [`__init__`](gpu_callbacks.md#AsyncCopyTask.__init__), [`SourceInfo`](../../../source_info_util.md#SourceInfo), [`VectorClock`](../../mosaic/interpret/vector_clock.md#VectorClock), [`check_write`](../../mosaic/interpret/race_detection_state.md#RaceDetectionState.check_write), [`check_read`](../../mosaic/interpret/race_detection_state.md#RaceDetectionState.check_read), [`detect_races`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.detect_races), [`to_range`](../../mosaic/interpret/utils.md#to_range), [`GPUSharedMemory`](shared_memory.md#GPUSharedMemory), [`copy_vector_clock`](../../mosaic/interpret/vector_clock.md#copy_vector_clock), [`get_races`](gpu_callbacks.md#get_races), [`add_copy_smem_to_gmem_clocks`](shared_memory.md#GPUSharedMemory.add_copy_smem_to_gmem_clocks), [`inc_vector_clock`](../../mosaic/interpret/vector_clock.md#inc_vector_clock), [`device_id`](gpu_callbacks.md#AsyncCopyTask.device_id), [`thread_id`](gpu_callbacks.md#AsyncCopyTask.thread_id), [`source_info`](gpu_callbacks.md#AsyncCopyTask.source_info), [`AsyncCopyTask`](gpu_callbacks.md#AsyncCopyTask), [`dst_allocation_key`](gpu_callbacks.md#AsyncCopyTask.dst_allocation_key), [`src_allocation_key`](gpu_callbacks.md#AsyncCopyTask.src_allocation_key), [`dst_transforms`](gpu_callbacks.md#AsyncCopyTask.dst_transforms), [`src_transforms`](gpu_callbacks.md#AsyncCopyTask.src_transforms)
- used by: [`copy_smem_to_gmem`](gpu_callbacks.md#copy_smem_to_gmem), [`post_read`](gpu_callbacks.md#AsyncCopyTask.post_read), [`post_write`](gpu_callbacks.md#AsyncCopyTask.post_write), [`pre_read`](gpu_callbacks.md#AsyncCopyTask.pre_read), [`AsyncCopyTask`](gpu_callbacks.md#AsyncCopyTask)

### `AsyncCopyTask`
- def: [`jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py:1232`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1232)
- doc: An async task representing a TMA memory copy.
- signature: `class AsyncCopyTask:`
- members:
  - `post_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory)` — [`L1324`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1324)
  - `post_write(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory)` — [`L1327`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1327)
  - `pre_read(self, tma_thread_id: int, shared_memory: memory.GPUSharedMemory)` — [`L1321`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1321)
  - `data` — [`L1267`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1267)
  - `device_id` — [`L1245`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1245)
  - `dst_allocation_key` — [`L1260`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1260)
  - `dst_transforms` — [`L1261`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1261)
  - `grid_point_coords` — [`L1247`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1247)
  - `logging_info` — [`L1265`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1265)
  - `source_info` — [`L1263`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1263)
  - `src_allocation_key` — [`L1256`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1256)
  - `src_transforms` — [`L1257`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1257)
  - `thread_id` — [`L1250`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1250)
  - `tma_thread_id` — [`L1253`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1253)
- protocol/private: `__call__`[`L1296`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1296), `__init__`[`L1269`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1269)
- uses (calls/refs, reference-scoped): [`Array`](../../../basearray.md#Array), [`HostAllocationKey`](shared_memory.md#HostAllocationKey), [`get_buffer_content`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.get_buffer_content), [`SourceInfo`](../../../source_info_util.md#SourceInfo), [`store_buffer_content`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.store_buffer_content), [`post_read`](gpu_callbacks.md#AsyncCopySmemToGmemTask.post_read), [`pre_read`](gpu_callbacks.md#AsyncCopySmemToGmemTask.pre_read), [`post_read`](gpu_callbacks.md#AsyncCopyGmemToSmemTask.post_read), [`pre_read`](gpu_callbacks.md#AsyncCopyGmemToSmemTask.pre_read), [`_get_shared_memory`](gpu_callbacks.md#_get_shared_memory), [`source_info`](../../mosaic/interpret/utils.md#LoggingInfo.source_info), [`device_id`](../../mosaic/interpret/utils.md#LoggingInfo.device_id), [`GPULoggingInfo`](../../mosaic/interpret/utils.md#GPULoggingInfo), [`to_range`](../../mosaic/interpret/utils.md#to_range), [`GPUSharedMemory`](shared_memory.md#GPUSharedMemory), [`post_write`](gpu_callbacks.md#AsyncCopyGmemToSmemTask.post_write), [`post_write`](gpu_callbacks.md#AsyncCopySmemToGmemTask.post_write), [`grid_point_coords`](../../mosaic/interpret/utils.md#GPULoggingInfo.grid_point_coords), [`thread_id`](../../mosaic/interpret/utils.md#GPULoggingInfo.thread_id), [`AsyncCopyGmemToSmemTask`](gpu_callbacks.md#AsyncCopyGmemToSmemTask), [`AsyncCopySmemToGmemTask`](gpu_callbacks.md#AsyncCopySmemToGmemTask)
- used by: [`post_read`](gpu_callbacks.md#AsyncCopySmemToGmemTask.post_read), [`pre_read`](gpu_callbacks.md#AsyncCopySmemToGmemTask.pre_read), [`post_read`](gpu_callbacks.md#AsyncCopyGmemToSmemTask.post_read), [`pre_read`](gpu_callbacks.md#AsyncCopyGmemToSmemTask.pre_read), [`__init__`](gpu_callbacks.md#AsyncCopyGmemToSmemTask.__init__), [`__init__`](gpu_callbacks.md#AsyncCopySmemToGmemTask.__init__), [`post_write`](gpu_callbacks.md#AsyncCopyGmemToSmemTask.post_write), [`post_write`](gpu_callbacks.md#AsyncCopySmemToGmemTask.post_write), [`AsyncCopyGmemToSmemTask`](gpu_callbacks.md#AsyncCopyGmemToSmemTask), [`AsyncCopySmemToGmemTask`](gpu_callbacks.md#AsyncCopySmemToGmemTask)

## Functions
- `_allocate_barriers(*, token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, axes_dims: tuple[int, ...], num_arrivals: jax.Array, flat_num_barriers: jax.Array, ref_count: jax.Array, source_info: source_info_util.SourceInfo | None = None)` — [`L837`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L837)
- `_allocate_buffer(token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, allocation_request_as_array: jax.Array, value: jax.Array, source_info: source_info_util.SourceInfo | None = None)` — [`L373`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L373) — Allocates a buffer for the given `allocation_request`.
- `_allocate_buffer_for_all_threads(token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array | None, allocation_request_as_array: jax.Array, value: jax.Array, source_info: source_info_util.SourceInfo | None = None)` — [`L263`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L263) — Allocates a buffer for the given `allocation_request`.
- `_allocate_cluster_barriers(*, token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, axes_dims: tuple[int, ...], is_axis_collective: tuple[bool, ...], num_arrivals: jax.Array, flat_num_barriers: jax.Array, ref_count: jax.Array, source_info: source_info_util.SourceInfo | None = None)` — [`L1127`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1127)
- `_assert_no_barriers_allocated(token)` — [`L1116`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1116)
- `_barrier_arrive(token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, allocation_key_as_array: jax.Array, source_info: source_info_util.SourceInfo | None = None)` — [`L1060`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1060)
- `_barrier_wait(token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, allocation_key_as_array: jax.Array, source_info: source_info_util.SourceInfo | None = None)` — [`L1008`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1008)
- `_clean_up_shared_memory(token)` — [`L191`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L191)
- `_clear_shared_memory()` — [`L75`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L75)
- `_deallocate_barrier(token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, allocation_key_as_array: jax.Array, source_info: source_info_util.SourceInfo | None = None, cluster_barrier: bool = False)` — [`L936`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L936)
- `_deallocate_buffer(token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, allocation_key_as_array: jax.Array, source_info: source_info_util.SourceInfo | None = None)` — [`L459`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L459) — Decreases the reference count of the buffer with `allocation_key` (Deallocates the buffer if its reference count becomes zero).
- `_get(token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array | None, thread_id: jax.Array, allocation_key_as_array: jax.Array, transforms, block_indices=None, grid_loop_idx=None, clock=None, increment_clock: bool = True, source_info=None, input_name=None)` — [`L576`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L576) — Performs a read from the buffer for `allocation_key_as_array` from the given device and thread.
- `_get_shared_memory()` — [`L70`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L70)
- `_handle_out_of_bounds_read(ret: np.ndarray | None, full_read_shape: tuple[int, ...], shape: Sequence[int], dtype: np.dtype, allocation_key: HostAllocationKey, read_range: tuple[int | slice, ...], shared_memory: memory.GPUSharedMemory, source_info, input_name: str | None, block_indices: tuple[int, ...] | None, grid_loop_idx: tuple[int, ...] | None)` — [`L506`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L506) — Handles out-of-bounds read based on shared_memory configuration.
- `_initialize_shared_memory(*, token: jax.Array, num_gpus: jax.Array, num_threads_per_block: jax.Array, num_blocks_per_cluster: jax.Array, interpret_params: InterpretGPUParams)` — [`L127`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L127)
- `_is_dynamic(indexer: indexing.NDIndexer)` — [`L556`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L556)
- `_make_allocation_request_array(*, token: jax.Array, memory_space_id: int, device_id: jax.Array, thread_id: jax.Array | None = None, initial_ref_count: int = 1)` — [`L221`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L221)
- `_remove_noop_transforms(transforms: tuple[Any, ...])` — [`L1484`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1484)
- `_swap(token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, allocation_key_as_array: jax.Array, transforms, val: np.ndarray, mask: jax.Array | None, *, clock=None, increment_clock: bool = True, source_info=None)` — [`L708`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L708) — Performs a swap into the buffer for `allocation_key` from the given device and thread.
- `_update_clocks_for_device_barrier(token, device_id: jax.Array)` — [`L203`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L203)
- `_validate_transforms(transforms)` — [`L564`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L564)
- `async_load_tmem(*, token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, src_allocation_key_as_array: jax.Array, src_transforms: tuple[Any, ...], source_info: source_info_util.SourceInfo | None = None)` — [`L1845`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1845)
- `call_allocate_barriers(*, token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, axes_dims: tuple[int, ...], num_arrivals: jax.Array, flat_num_barriers: int | jax.Array, ref_count: jax.Array, source_info: source_info_util.SourceInfo | None = None)` — [`L902`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L902)
- `call_allocate_buffer(token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, allocation_request_as_array: jax.Array, value: jax.Array, source_info: source_info_util.SourceInfo | None = None)` — [`L438`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L438)
- `call_allocate_buffer_for_all_threads(token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array | None, allocation_request_as_array: jax.Array, value: jax.Array, source_info: source_info_util.SourceInfo | None = None)` — [`L352`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L352)
- `call_allocate_cluster_barriers(*, token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, axes_dims: tuple[int, ...], is_axis_collective: tuple[bool, ...], num_arrivals: jax.Array, flat_num_barriers: int | jax.Array, ref_count: jax.Array, source_info: source_info_util.SourceInfo | None = None)` — [`L1196`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1196)
- `call_assert_no_barriers_allocated(token)` — [`L1121`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1121)
- `call_barrier_arrive(token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, allocation_key_as_array: jax.Array, source_info: source_info_util.SourceInfo | None = None)` — [`L1097`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1097)
- `call_barrier_wait(token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, allocation_key_as_array: jax.Array, source_info: source_info_util.SourceInfo | None = None)` — [`L1041`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1041)
- `call_clean_up_shared_memory(token)` — [`L197`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L197)
- `call_deallocate_barrier(token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, allocation_key_as_array: jax.Array, source_info: source_info_util.SourceInfo | None = None, cluster_barrier: bool = False)` — [`L987`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L987)
- `call_deallocate_buffer(token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, allocation_key_as_array: jax.Array, source_info: source_info_util.SourceInfo | None = None)` — [`L487`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L487)
- `call_get(*, token: jax.Array, result_shape_and_dtype, device_id: jax.Array, grid_point_coords: jax.Array | None, thread_id: jax.Array, allocation_key_as_array: jax.Array, transforms, block_indices=None, grid_loop_idx=None, clock=None, source_info=None, input_name=None)` — [`L678`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L678)
- `call_initialize_shared_memory(*, token: jax.Array, num_gpus: jax.Array, num_threads_per_block: jax.Array, num_blocks_per_cluster: jax.Array, interpret_params: InterpretGPUParams)` — [`L170`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L170)
- `call_make_allocation_request_array(*, token: jax.Array, memory_space_id: int, device_id: jax.Array, thread_id: jax.Array | None = None, initial_ref_count: int = 1)` — [`L241`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L241)
- `call_swap(*, token: jax.Array, result_shape_and_dtype, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, allocation_key_as_array: jax.Array, transforms, val: jax.Array, mask: jax.Array | None, clock=None, source_info=None)` — [`L781`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L781)
- `call_update_clocks_for_device_barrier(token, device_id: jax.Array)` — [`L212`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L212)
- `commit_smem(*, token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, source_info: source_info_util.SourceInfo | None = None)` — [`L1730`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1730)
- `copy_gmem_to_smem(*, token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, src_allocation_key_as_array: jax.Array, src_transforms: tuple[Any, ...], dst_allocation_key_as_array: jax.Array, dst_transforms: tuple[Any, ...], barrier_allocation_key_as_array: jax.Array, source_info: source_info_util.SourceInfo | None = None)` — [`L1676`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1676)
- `copy_smem_to_gmem(*, token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, src_allocation_key_as_array: jax.Array, src_transforms: tuple[Any, ...], dst_allocation_key_as_array: jax.Array, dst_transforms: tuple[Any, ...], predicate: jax.Array | None, source_info: source_info_util.SourceInfo, commit_group: bool, reduction_op: mgpu.TMAReductionOp)` — [`L1597`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1597)
- `get_memory_space_idx(space: mosaic_gpu_core.MemorySpace)` — [`L55`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L55)
- `get_races()` — [`L81`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L81)
- `get_thread_id_for_collective_allocation_key(thread_id: int, axes_dims: tuple[int, ...], is_last_thread_axis_collective: bool)` — [`L810`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L810) — Returns the thread ID to use for the allocation key in a collective allocation.
- `is_gmem_memory_space(space: mosaic_gpu_core.MemorySpace | None)` — [`L61`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L61)
- `kernel_thread_finished(*, token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array)` — [`L1880`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1880)
- `ordering_barrier(token)` — [`L107`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L107)
- `reset_gpu_interpret_mode_state()` — [`L86`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L86) — Resets all global, shared state used by GPU interpret mode.
- `tcgen05_mma(*, token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, acc_allocation_key_as_array: jax.Array, acc_transforms: tuple[Any, ...], acc_dtype: jnp.dtype, a_allocation_key_as_array: jax.Array, a_transforms: tuple[Any, ...], b_allocation_key_as_array: jax.Array, b_transforms: tuple[Any, ...], accumulate: jax.Array, barrier_allocation_key_as_array: jax.Array | None = None, a_scale_allocation_key_as_array: jax.Array | None = None, a_scale_transforms: tuple[Any, ...] | None = None, b_scale_allocation_key_as_array: jax.Array | None = None, b_scale_transforms: tuple[Any, ...] | None = None, a_sparse_metadata_allocation_key_as_array: jax.Array | None = None, a_sparse_metadata_transforms: tuple[Any, ...] | None = None, source_info: source_info_util.SourceInfo | None = None)` — [`L1747`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1747)
- `wait_smem_to_gmem(*, token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, n: int, wait_read_only: bool, source_info: source_info_util.SourceInfo | None = None)` — [`L1657`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1657)
- `wgmma(*, token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, acc_allocation_key_as_array: jax.Array, acc_transforms: tuple[Any, ...], acc_dtype: jnp.dtype, a_allocation_key_as_array: jax.Array, a_transforms: tuple[Any, ...], b_allocation_key_as_array: jax.Array, b_transforms: tuple[Any, ...], source_info: source_info_util.SourceInfo | None = None)` — [`L1492`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1492)
- `wgmma_accumulator_deref(*, token: jax.Array, device_id: jax.Array, grid_point_coords: jax.Array, thread_id: jax.Array, acc_allocation_key_as_array: jax.Array, wait_n: int | None, source_info: source_info_util.SourceInfo | None = None)` — [`L1564`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1564)

## Module values
- `GPU_MEMORY_SPACE_BY_IDX` — [`L49`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L49)
- `IDX_BY_GPU_MEMORY_SPACE` — [`L43`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L43)
- `NOOP_TRANSFORMS` — [`L1478`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L1478)
- `TOKEN_SHAPE_DTYPE` — [`L104`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L104)
- `_races` — [`L67`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L67)
- `_shared_memory` — [`L65`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L65)
- `_shared_memory_init_lock` — [`L66`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/gpu_callbacks.py#L66)

