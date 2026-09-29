---
title: 'Module: jax/_src/pallas/mosaic/interpret/shared_memory.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/interpret/shared_memory.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.interpret.shared_memory`/
symbols:
  GenericSharedMemory.get_buffer_content: GenericSharedMemory#get_buffer_content().
  Semaphore.wait: Semaphore#wait().
  GenericSharedMemory.swap_buffer_content: GenericSharedMemory#swap_buffer_content().
  GenericSharedMemory.store_buffer_content: GenericSharedMemory#store_buffer_content().
  GenericSharedMemory.lock: GenericSharedMemory#lock.
  GenericSharedMemory.deallocate_buffer: GenericSharedMemory#deallocate_buffer().
  Semaphore.signal: Semaphore#signal().
  GenericSharedMemory.allocate_buffer: GenericSharedMemory#allocate_buffer().
  GenericSharedMemory.detect_races: GenericSharedMemory#detect_races.
  GenericSharedMemory.mem: GenericSharedMemory#mem.
  GenericSharedMemory.incr_clock: GenericSharedMemory#incr_clock().
  SharedMemory.get_semaphores_and_increment_clock: SharedMemory#get_semaphores_and_increment_clock().
  GenericSharedMemory: GenericSharedMemory#
  GenericSharedMemory.enable_logging: GenericSharedMemory#enable_logging().
  SharedMemory.get_sempahores_with_nonzero_count: SharedMemory#get_sempahores_with_nonzero_count().
  SharedMemory.allocate_semaphores: SharedMemory#allocate_semaphores().
  GenericSharedMemory.clocks: GenericSharedMemory#clocks.
  SharedMemory.guarantee_semaphore_with_fixed_id: SharedMemory#guarantee_semaphore_with_fixed_id().
  Semaphore.count_by_core: Semaphore#count_by_core.
  SharedMemory.__init__: SharedMemory#__init__().
  Buffer.logical_shape: Buffer#logical_shape().
  SharedMemory.set_failed: SharedMemory#set_failed().
  GenericSharedMemory._log: GenericSharedMemory#_log().
  SharedMemory.get_global_core_id: SharedMemory#get_global_core_id().
  Semaphore.tasks: Semaphore#tasks.
  GenericSharedMemory.update_clocks: GenericSharedMemory#update_clocks().
  SharedMemory.append_semaphore_task: SharedMemory#append_semaphore_task().
  GenericSharedMemory.logging_mode: GenericSharedMemory#logging_mode.
  GenericSharedMemory.get_next_buffer_id: GenericSharedMemory#get_next_buffer_id().
  Semaphore.shared_memory: Semaphore#shared_memory.
  Buffer._logical_shape: Buffer#_logical_shape.
  Semaphore.clocks: Semaphore#clocks.
  Allocation: Allocation#
  Buffer.content: Buffer#content().
  Buffer.shape: Buffer#shape().
  SharedMemory.num_cores: SharedMemory#num_cores().
  Buffer.__getitem__: Buffer#__getitem__().
  Buffer.__setitem__: Buffer#__setitem__().
  SharedMemory.num_cores_per_device: SharedMemory#num_cores_per_device.
  Buffer._set_within_logical_shape: Buffer#_set_within_logical_shape().
  SharedMemory._unsafe_get_semaphore: SharedMemory#_unsafe_get_semaphore().
  SharedMemory.check_failed: SharedMemory#check_failed().
  Semaphore: Semaphore#
  Semaphore.enqueue_task: Semaphore#enqueue_task().
  GenericSharedMemory.thread_to_vc_position: GenericSharedMemory#thread_to_vc_position().
  SharedMemory.sem: SharedMemory#sem.
  SharedMemory.fixed_id_sem: SharedMemory#fixed_id_sem.
  Semaphore.cv: Semaphore#cv.
  Semaphore.detect_races: Semaphore#detect_races().
  Buffer._normalize_range: Buffer#_normalize_range().
  Buffer: Buffer#
  Buffer.dtype: Buffer#dtype().
  SharedMemory: SharedMemory#
  Semaphore.get_global_core_id: Semaphore#get_global_core_id().
  SharedMemory.get_global_core_ids: SharedMemory#get_global_core_ids().
  SharedMemory.global_core_id_to_thread: SharedMemory#global_core_id_to_thread().
  Buffer.decrease_ref_count: Buffer#decrease_ref_count().
  Buffer.set_in_bounds_portion: Buffer#set_in_bounds_portion().
  ShapeAndDtype: ShapeAndDtype#
  SharedMemory.get_random_virtual_device_id: SharedMemory#get_random_virtual_device_id().
  SharedMemory.update_clocks_for_device_barrier: SharedMemory#update_clocks_for_device_barrier().
  Semaphore.num_cores: Semaphore#num_cores().
  Semaphore.dma_execution_mode: Semaphore#dma_execution_mode().
  Semaphore.read: Semaphore#read().
  ShapeAndDtype.__iter__: ShapeAndDtype#__iter__().
  GenericSharedMemory.num_devices: GenericSharedMemory#num_devices.
  SharedMemory.thread_to_vc_position: SharedMemory#thread_to_vc_position().
  SharedMemory.print: SharedMemory#print().
  Semaphore._log: Semaphore#_log().
  Buffer.ref_count: Buffer#ref_count().
  GenericSharedMemory.next_buffer_id: GenericSharedMemory#next_buffer_id.
  GenericSharedMemory._failed_thread: GenericSharedMemory#_failed_thread.
  GenericSharedMemory.dma_execution_mode: GenericSharedMemory#dma_execution_mode.
  GenericSharedMemory._failure: GenericSharedMemory#_failure.
  Buffer.has_zero_ref_count: Buffer#has_zero_ref_count().
  Buffer.size: Buffer#size().
  GenericSharedMemory.output_ranges: GenericSharedMemory#output_ranges.
  Semaphore.id: Semaphore#id.
  SemaphoreTask: SemaphoreTask.
  GenericSharedMemory.uninitialized_memory: GenericSharedMemory#uninitialized_memory.
  GenericSharedMemory.clean_up_barrier: GenericSharedMemory#clean_up_barrier.
  GenericSharedMemory.deallocated_bytes: GenericSharedMemory#deallocated_bytes.
  SharedMemory.ThreadKey: SharedMemory#ThreadKey.
  SharedMemory.next_semaphore_id: SharedMemory#next_semaphore_id.
  Semaphore.__init__: Semaphore#__init__().
  logger: logger.
  Semaphore.enable_logging: Semaphore#enable_logging.
  Buffer._content: Buffer#_content.
  Buffer._ref_count: Buffer#_ref_count.
  GenericSharedMemory.out_of_bounds_reads: GenericSharedMemory#out_of_bounds_reads.
  GenericSharedMemory.barrier: GenericSharedMemory#barrier.
  ShapeAndDtype.shape: ShapeAndDtype#shape.
  ShapeAndDtype.dtype: ShapeAndDtype#dtype.
  GenericSharedMemory.vector_clock_size: GenericSharedMemory#vector_clock_size.
  GenericSharedMemory.buffer_bounds: GenericSharedMemory#buffer_bounds.
  Buffer.__init__: Buffer#__init__().
  SharedMemory.MemKey: SharedMemory#MemKey.
---
# Module: [`jax/_src/pallas/mosaic/interpret/shared_memory.py`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py)

## Classes
### `Allocation`
- def: [`jax/_src/pallas/mosaic/interpret/shared_memory.py:231`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L231)
- signature: `class Allocation:`
- uses (calls/refs, reference-scoped): [`Barrier`](../../mosaic_gpu/interpret/shared_memory.md#Barrier), [`Buffer`](shared_memory.md#Buffer), [`ClusterBarrier`](../../mosaic_gpu/interpret/shared_memory.md#ClusterBarrier)
- used by: [`mem`](shared_memory.md#GenericSharedMemory.mem), [`Barrier`](../../mosaic_gpu/interpret/shared_memory.md#Barrier), [`Buffer`](shared_memory.md#Buffer), [`ClusterBarrier`](../../mosaic_gpu/interpret/shared_memory.md#ClusterBarrier)

### `Buffer`  ·  implements/extends Allocation
- def: [`jax/_src/pallas/mosaic/interpret/shared_memory.py:235`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L235)
- signature: `class Buffer(Allocation):`
- members:
  - `__getitem__(self, rnge: tuple[slice | int, ...])` — [`L299`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L299) — Returns the portion of `self.content` specified by `rnge`.
  - `__setitem__(self, rnge: tuple[slice | int, ...], value: np.ndarray)` — [`L337`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L337) — Updates `self.content` with `value`, if `rnge` is fully within `self.shape`.
  - `_normalize_range(self, rnge: tuple[slice | int, ...])` — [`L292`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L292) — Normalizes `rnge` by adding slices to match the `self.logical_shape`.
  - `_set_within_logical_shape(self, rnge: tuple[slice | int, ...], value: np.ndarray)` — [`L325`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L325) — Updates `self.content` with `value` for the portion of `rnge` within `self.logical_shape`.
  - `content(self)` — [`L261`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L261)
  - `decrease_ref_count(self)` — [`L272`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L272)
  - `dtype(self)` — [`L289`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L289)
  - `has_zero_ref_count(self)` — [`L277`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L277)
  - `logical_shape(self)` — [`L269`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L269)
  - `ref_count(self)` — [`L265`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L265)
  - `set_in_bounds_portion(self, rnge: tuple[slice | int, ...], value: np.ndarray)` — [`L358`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L358) — Updates `self.content` with `value` for the portion of `rnge` within `self.logical_shape`.
  - `shape(self)` — [`L285`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L285)
  - `size(self)` — [`L281`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L281)
- protocol/private: `__init__`[`L237`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L237), `_content`[`L245`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L245), `_logical_shape`[`L249`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L249), `_ref_count`[`L246`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L246)
- uses (calls/refs, reference-scoped): [`Allocation`](shared_memory.md#Allocation), [`clip_range_to_shape`](utils.md#clip_range_to_shape), [`is_range_out_of_bounds_for_shape`](utils.md#is_range_out_of_bounds_for_shape)
- used by: [`get_buffer_content`](shared_memory.md#GenericSharedMemory.get_buffer_content), [`swap_buffer_content`](shared_memory.md#GenericSharedMemory.swap_buffer_content), [`store_buffer_content`](shared_memory.md#GenericSharedMemory.store_buffer_content), [`deallocate_buffer`](shared_memory.md#GenericSharedMemory.deallocate_buffer), [`allocate_buffer`](shared_memory.md#GenericSharedMemory.allocate_buffer), [`Allocation`](shared_memory.md#Allocation)

### `GenericSharedMemory`
- def: [`jax/_src/pallas/mosaic/interpret/shared_memory.py:376`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L376)
- members:
  - `_log(self, message: str)` — [`L418`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L418) — Logs a message. To be called while holding `self.lock`.
  - `allocate_buffer(self, key: MemKey, ref_count: int, value: np.ndarray, logical_shape: tuple[int, ...] | None = None, logging_info: interpret_utils.LoggingInfo | None = None)` — [`L445`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L445) — Allocates a memory buffer with the given key unless it already exists.
  - `deallocate_buffer(self, key: MemKey, logging_info: interpret_utils.LoggingInfo | None = None)` — [`L471`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L471) — Decreases the ref count for the buffer with `key` and deallocates the buffer if the ref count is zero.
  - `enable_logging(self)` — [`L412`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L412)
  - `get_buffer_content(self, key: MemKey, rnge: tuple[slice | int, ...], thread: ThreadKey, increment_clock: bool = True, logging_info: interpret_utils.LoggingInfo | None = None)` — [`L503`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L503) — Reads contents of a memory buffer.
  - `get_next_buffer_id(self, key: ThreadKey)` — [`L438`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L438) — Returns the next buffer ID for the given device and thread.
  - `incr_clock(self, thread: ThreadKey, take_lock: bool = True)` — [`L431`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L431) — Increments a threads's own index within its clock by one.
  - `store_buffer_content(self, key: MemKey, rnge: tuple[slice | int, ...], value: np.ndarray, thread: ThreadKey, increment_clock: bool = True, logging_info: interpret_utils.LoggingInfo | None = None)` — [`L560`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L560) — Stores contents into a memory buffer.
  - `swap_buffer_content(self, key: MemKey, rnge: tuple[slice | int, ...], value: np.ndarray, mask: np.ndarray | None, thread: ThreadKey, increment_clock: bool = True, logging_info: interpret_utils.LoggingInfo | None = None)` — [`L621`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L621) — Swaps contents of a memory buffer.
  - `thread_to_vc_position(self, thread: ThreadKey)` — [`L428`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L428)
  - `update_clocks(self, threads: Sequence[ThreadKey])` — [`L714`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L714) — Synchronizes the vector clocks for all of the given threads.
  - `barrier` — [`L385`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L385)
  - `buffer_bounds` — [`L388`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L388)
  - `clean_up_barrier` — [`L386`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L386)
  - `clocks` — [`L384`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L384)
  - `deallocated_bytes` — [`L401`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L401)
  - `detect_races` — [`L381`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L381)
  - `dma_execution_mode` — [`L379`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L379)
  - `lock` — [`L396`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L396)
  - `logging_mode` — [`L390`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L390)
  - `mem` — [`L392`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L392)
  - `next_buffer_id` — [`L398`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L398)
  - `num_devices` — [`L377`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L377)
  - `out_of_bounds_reads` — [`L378`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L378)
  - `output_ranges` — [`L404`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L404)
  - `uninitialized_memory` — [`L380`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L380)
  - `vector_clock_size` — [`L382`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L382)
- protocol/private: `_failed_thread`[`L409`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L409), `_failure`[`L408`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L408)
- uses (calls/refs, reference-scoped): [`VectorClock`](vector_clock.md#VectorClock), [`format`](utils.md#LoggingInfo.format), [`copy_vector_clock`](vector_clock.md#copy_vector_clock), [`logical_shape`](shared_memory.md#Buffer.logical_shape), [`update_vector_clock`](vector_clock.md#update_vector_clock), [`LoggingInfo`](utils.md#LoggingInfo), [`inc_vector_clock`](vector_clock.md#inc_vector_clock), [`Allocation`](shared_memory.md#Allocation), [`shape`](shared_memory.md#Buffer.shape), [`LoggingMode`](params.md#LoggingMode), [`Buffer`](shared_memory.md#Buffer), [`dtype`](shared_memory.md#Buffer.dtype), [`ShapeAndDtype`](shared_memory.md#ShapeAndDtype), [`decrease_ref_count`](shared_memory.md#Buffer.decrease_ref_count), [`set_in_bounds_portion`](shared_memory.md#Buffer.set_in_bounds_portion), [`thread_to_vc_position`](shared_memory.md#SharedMemory.thread_to_vc_position), [`thread_to_vc_position`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.thread_to_vc_position), [`has_zero_ref_count`](shared_memory.md#Buffer.has_zero_ref_count), [`size`](shared_memory.md#Buffer.size), [`logger`](shared_memory.md#logger), [`SHARED_MEMORY`](params.md#LoggingMode.SHARED_MEMORY)
- used by: [`_allocate_buffer`](../../mosaic_gpu/interpret/gpu_callbacks.md#_allocate_buffer), [`_allocate_buffer_for_all_threads`](../../mosaic_gpu/interpret/gpu_callbacks.md#_allocate_buffer_for_all_threads), [`get`](interpret_pallas_call.md#get), [`_allocate_barriers`](../../mosaic_gpu/interpret/gpu_callbacks.md#_allocate_barriers), [`_allocate_cluster_barriers`](../../mosaic_gpu/interpret/gpu_callbacks.md#_allocate_cluster_barriers), [`_allocate_buffer`](interpret_pallas_call.md#_allocate_buffer), [`wait`](shared_memory.md#Semaphore.wait), [`_get`](../../mosaic_gpu/interpret/gpu_callbacks.md#_get), [`_swap`](../../mosaic_gpu/interpret/gpu_callbacks.md#_swap), [`tcgen05_mma`](../../mosaic_gpu/interpret/gpu_callbacks.md#tcgen05_mma), [`wait`](../../mosaic_gpu/interpret/shared_memory.md#Barrier.wait), [`dma_start`](interpret_pallas_call.md#dma_start), [`wgmma`](../../mosaic_gpu/interpret/gpu_callbacks.md#wgmma), [`store`](interpret_pallas_call.md#store), [`post_read`](../../mosaic_gpu/interpret/gpu_callbacks.md#AsyncCopySmemToGmemTask.post_read), [`pre_read`](../../mosaic_gpu/interpret/gpu_callbacks.md#AsyncCopySmemToGmemTask.pre_read), [`__call__`](../../mosaic_gpu/interpret/gpu_callbacks.md#AsyncCopyTask.__call__), [`post_read`](../../mosaic_gpu/interpret/gpu_callbacks.md#AsyncCopyGmemToSmemTask.post_read), [`pre_read`](../../mosaic_gpu/interpret/gpu_callbacks.md#AsyncCopyGmemToSmemTask.pre_read), [`swap`](interpret_pallas_call.md#swap), [`async_load_tmem`](../../mosaic_gpu/interpret/gpu_callbacks.md#async_load_tmem), [`allocate_barrier`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.allocate_barrier), [`copy_smem_to_gmem`](../../mosaic_gpu/interpret/gpu_callbacks.md#copy_smem_to_gmem), [`_deallocate_buffer`](../../mosaic_gpu/interpret/gpu_callbacks.md#_deallocate_buffer), [`allocate_cluster_barrier`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.allocate_cluster_barrier), [`copy_gmem_to_smem`](../../mosaic_gpu/interpret/gpu_callbacks.md#copy_gmem_to_smem), [`wgmma_accumulator_deref`](../../mosaic_gpu/interpret/gpu_callbacks.md#wgmma_accumulator_deref), [`deallocate_barrier`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.deallocate_barrier), [`deallocate_cluster_barrier`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.deallocate_cluster_barrier), [`_deallocate_buffer`](interpret_pallas_call.md#_deallocate_buffer), [`get_barrier_and_increment_clock`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.get_barrier_and_increment_clock), [`GPUSharedMemory`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory), [`get_semaphores_and_increment_clock`](shared_memory.md#SharedMemory.get_semaphores_and_increment_clock), [`allocate_semaphores`](shared_memory.md#SharedMemory.allocate_semaphores), [`get_sempahores_with_nonzero_count`](shared_memory.md#SharedMemory.get_sempahores_with_nonzero_count), [`get_smem_commit_clock`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.get_smem_commit_clock), [`num_blocks_per_cluster`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.num_blocks_per_cluster), [`post_write`](../../mosaic_gpu/interpret/gpu_callbacks.md#AsyncCopySmemToGmemTask.post_write), [`_handle_out_of_bounds_read`](../../mosaic_gpu/interpret/gpu_callbacks.md#_handle_out_of_bounds_read), [`guarantee_semaphore_with_fixed_id`](shared_memory.md#SharedMemory.guarantee_semaphore_with_fixed_id)  (+31 more)

### `Semaphore`
- def: [`jax/_src/pallas/mosaic/interpret/shared_memory.py:37`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L37)
- signature: `class Semaphore:`
- members:
  - `_log(self, message: str)` — [`L87`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L87) — Logs a message. To be called while holding the lock on `self.cv`.
  - `detect_races(self)` — [`L80`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L80)
  - `dma_execution_mode(self)` — [`L84`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L84)
  - `enqueue_task(self, task: SemaphoreTask, global_core_id: int)` — [`L102`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L102)
  - `get_global_core_id(self, device_id: int, local_core_id: int)` — [`L99`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L99)
  - `num_cores(self)` — [`L76`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L76)
  - `read(self, global_core_id)` — [`L144`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L144)
  - `signal(self, inc, global_core_id, clock, logging_info: interpret_utils.LoggingInfo | None = None)` — [`L107`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L107) — Signal the semaphore on `(device_id, core_id)` by `inc`.
  - `wait(self, value, global_core_id, *, has_tasks=False, logging_info: interpret_utils.LoggingInfo | None = None)` — [`L148`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L148)
  - `clocks` — [`L71`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L71)
  - `count_by_core` — [`L53`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L53)
  - `cv` — [`L51`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L51)
  - `enable_logging` — [`L47`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L47)
  - `id` — [`L46`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L46)
  - `shared_memory` — [`L45`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L45)
  - `tasks` — [`L58`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L58)
- protocol/private: `__init__`[`L39`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L39)
- uses (calls/refs, reference-scoped): [`VectorClock`](vector_clock.md#VectorClock), [`lock`](shared_memory.md#GenericSharedMemory.lock), [`format`](utils.md#LoggingInfo.format), [`detect_races`](shared_memory.md#GenericSharedMemory.detect_races), [`clocks`](shared_memory.md#GenericSharedMemory.clocks), [`copy_vector_clock`](vector_clock.md#copy_vector_clock), [`update_vector_clock`](vector_clock.md#update_vector_clock), [`LoggingInfo`](utils.md#LoggingInfo), [`get_global_core_id`](shared_memory.md#SharedMemory.get_global_core_id), [`num_cores`](shared_memory.md#SharedMemory.num_cores), [`check_failed`](shared_memory.md#SharedMemory.check_failed), [`SharedMemory`](shared_memory.md#SharedMemory), [`global_core_id_to_thread`](shared_memory.md#SharedMemory.global_core_id_to_thread), [`dma_execution_mode`](shared_memory.md#GenericSharedMemory.dma_execution_mode), [`SemaphoreTask`](shared_memory.md#SemaphoreTask), [`logger`](shared_memory.md#logger)
- used by: [`execute_read`](interpret_pallas_call.md#DMA.execute_read), [`execute_write`](interpret_pallas_call.md#DMA.execute_write), [`dma_wait`](interpret_pallas_call.md#dma_wait), [`semaphore_signal`](interpret_pallas_call.md#semaphore_signal), [`semaphore_wait`](interpret_pallas_call.md#semaphore_wait), [`get_semaphores_and_increment_clock`](shared_memory.md#SharedMemory.get_semaphores_and_increment_clock), [`allocate_semaphores`](shared_memory.md#SharedMemory.allocate_semaphores), [`get_sempahores_with_nonzero_count`](shared_memory.md#SharedMemory.get_sempahores_with_nonzero_count), [`guarantee_semaphore_with_fixed_id`](shared_memory.md#SharedMemory.guarantee_semaphore_with_fixed_id), [`_validate`](interpret_pallas_call.md#_validate), [`append_semaphore_task`](shared_memory.md#SharedMemory.append_semaphore_task), [`dst_global_core_id`](interpret_pallas_call.md#DMA.dst_global_core_id), [`src_global_core_id`](interpret_pallas_call.md#DMA.src_global_core_id), [`_unsafe_get_semaphore`](shared_memory.md#SharedMemory._unsafe_get_semaphore), [`detect_races`](interpret_pallas_call.md#DMA.detect_races), [`fixed_id_sem`](shared_memory.md#SharedMemory.fixed_id_sem), [`sem`](shared_memory.md#SharedMemory.sem), [`dst_sem`](interpret_pallas_call.md#DMA.dst_sem), [`src_sem`](interpret_pallas_call.md#DMA.src_sem)

### `ShapeAndDtype`
- def: [`jax/_src/pallas/mosaic/interpret/shared_memory.py:367`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L367)
- signature: `class ShapeAndDtype:`
- members:
  - `dtype` — [`L369`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L369)
  - `shape` — [`L368`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L368)
- protocol/private: `__iter__`[`L371`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L371)
- used by: [`get_buffer_content`](shared_memory.md#GenericSharedMemory.get_buffer_content), [`swap_buffer_content`](shared_memory.md#GenericSharedMemory.swap_buffer_content), [`store_buffer_content`](shared_memory.md#GenericSharedMemory.store_buffer_content)

### `SharedMemory`
- def: [`jax/_src/pallas/mosaic/interpret/shared_memory.py:729`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L729)
- members:
  - `_unsafe_get_semaphore(self, sem_id: int)` — [`L815`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L815) — Returns the semaphore with the given ID. `self.lock` must be held.
  - `allocate_semaphores(self, key: Any, num_semaphores: int)` — [`L969`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L969) — Returns the next semaphore ID and ensures that the next `num_semaphores` are allocated.
  - `append_semaphore_task(self, semaphore_id: int, global_core_id: int, task: SemaphoreTask)` — [`L878`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L878) — Appends a task to be executed if the semaphore with the given sempahore ID is waiting to be signalled on the core with the given global core ID.
  - `check_failed(self)` — [`L860`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L860)
  - `get_global_core_id(self, device_id: int, local_core_id: int)` — [`L790`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L790) — Computes the global core ID from the given device and local core ID.
  - `get_global_core_ids(self, device_id: int)` — [`L794`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L794) — Computes the global core IDs for all cores in the given device.
  - `get_random_virtual_device_id(self)` — [`L889`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L889)
  - `get_semaphores_and_increment_clock(self, sem_ids: Sequence[int | None], global_core_id: int)` — [`L913`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L913) — Returns the semaphores with the given `sem_ids` and increments the vector clock for the core with `global_core_id`.
  - `get_sempahores_with_nonzero_count(self, device_id: int)` — [`L955`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L955) — Returns tuples (semaphore, global_core_id) for all semaphores with a nonzero count for the core with `global_core_id`.
  - `global_core_id_to_thread(self, global_core_id: int)` — [`L804`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L804)
  - `guarantee_semaphore_with_fixed_id(self, semaphore_id: int)` — [`L988`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L988) — Ensures that a semaphore with the given `semaphore_id` exists.
  - `num_cores(self)` — [`L787`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L787)
  - `print(self, device_id: int)` — [`L809`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L809)
  - `set_failed(self, exception: Exception, device_id: int | None = None, local_core_id: int | None = None, top_level: bool = True)` — [`L830`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L830)
  - `thread_to_vc_position(self, thread: ThreadKey)` — [`L801`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L801)
  - `update_clocks_for_device_barrier(self, device_id: int)` — [`L1015`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L1015) — Synchronizes the vector clocks for the cores on the given device.
  - `MemKey` — [`L733`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L733)
  - `ThreadKey` — [`L735`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L735)
  - `fixed_id_sem` — [`L746`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L746)
  - `next_semaphore_id` — [`L743`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L743)
  - `num_cores_per_device` — [`L737`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L737)
  - `sem` — [`L740`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L740)
- protocol/private: `__init__`[`L748`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L748)
- uses (calls/refs, reference-scoped): [`VectorClock`](vector_clock.md#VectorClock), [`lock`](shared_memory.md#GenericSharedMemory.lock), [`detect_races`](shared_memory.md#GenericSharedMemory.detect_races), [`incr_clock`](shared_memory.md#GenericSharedMemory.incr_clock), [`mem`](shared_memory.md#GenericSharedMemory.mem), [`GenericSharedMemory`](shared_memory.md#GenericSharedMemory), [`count_by_core`](shared_memory.md#Semaphore.count_by_core), [`update_clocks`](shared_memory.md#GenericSharedMemory.update_clocks), [`logging_mode`](shared_memory.md#GenericSharedMemory.logging_mode), [`Semaphore`](shared_memory.md#Semaphore), [`enqueue_task`](shared_memory.md#Semaphore.enqueue_task), [`LoggingMode`](params.md#LoggingMode), [`cv`](shared_memory.md#Semaphore.cv), [`make_vector_clock`](vector_clock.md#make_vector_clock), [`num_devices`](shared_memory.md#GenericSharedMemory.num_devices), [`_failed_thread`](shared_memory.md#GenericSharedMemory._failed_thread), [`_failure`](shared_memory.md#GenericSharedMemory._failure), [`SemaphoreTask`](shared_memory.md#SemaphoreTask), [`clean_up_barrier`](shared_memory.md#GenericSharedMemory.clean_up_barrier), [`SEMAPHORE`](params.md#LoggingMode.SEMAPHORE), [`barrier`](shared_memory.md#GenericSharedMemory.barrier), [`vector_clock_size`](shared_memory.md#GenericSharedMemory.vector_clock_size)
- used by: [`_allocate_buffer`](interpret_pallas_call.md#_allocate_buffer), [`wait`](shared_memory.md#Semaphore.wait), [`dma_start`](interpret_pallas_call.md#dma_start), [`_initialize_shared_memory`](interpret_pallas_call.md#_initialize_shared_memory), [`_get_shared_memory`](interpret_pallas_call.md#_get_shared_memory), [`dma_wait`](interpret_pallas_call.md#dma_wait), [`semaphore_signal`](interpret_pallas_call.md#semaphore_signal), [`semaphore_wait`](interpret_pallas_call.md#semaphore_wait), [`races`](interpret_pallas_call.md#races.races), [`_allocate_semaphores`](interpret_pallas_call.md#_allocate_semaphores), [`count_by_core`](shared_memory.md#Semaphore.count_by_core), [`_validate`](interpret_pallas_call.md#_validate), [`tasks`](shared_memory.md#Semaphore.tasks), [`_shared_memory`](interpret_pallas_call.md#_shared_memory._shared_memory), [`clocks`](shared_memory.md#Semaphore.clocks), [`_update_clocks_for_global_barrier`](interpret_pallas_call.md#_update_clocks_for_global_barrier), [`wrapper`](interpret_pallas_call.md#fail_on_exception.wrapper), [`_update_clocks_for_device_barrier`](interpret_pallas_call.md#_update_clocks_for_device_barrier), [`thread_to_vc_position`](shared_memory.md#GenericSharedMemory.thread_to_vc_position), [`get_global_core_id`](shared_memory.md#Semaphore.get_global_core_id), [`fail`](interpret_pallas_call.md#fail), [`get_barrier_semaphore`](interpret_pallas_call.md#get_barrier_semaphore), [`num_cores`](shared_memory.md#Semaphore.num_cores), [`__init__`](shared_memory.md#Semaphore.__init__)

## Module values
- `SemaphoreTask` — [`L227`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L227)
- `logger` — [`L34`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/shared_memory.py#L34)

