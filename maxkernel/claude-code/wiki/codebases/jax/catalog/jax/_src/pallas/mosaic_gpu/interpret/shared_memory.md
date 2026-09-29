---
title: 'Module: jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic_gpu.interpret.shared_memory`/
symbols:
  HostAllocationKey: HostAllocationKey#
  Barrier.arrive: Barrier#arrive().
  Barrier.wait: Barrier#wait().
  HostAllocationRequest.from_array: HostAllocationRequest#from_array().
  GPUSharedMemory.allocate_barrier: GPUSharedMemory#allocate_barrier().
  GPUSharedMemory.allocate_cluster_barrier: GPUSharedMemory#allocate_cluster_barrier().
  GPUSharedMemory.deallocate_barrier: GPUSharedMemory#deallocate_barrier().
  GPUSharedMemory.deallocate_cluster_barrier: GPUSharedMemory#deallocate_cluster_barrier().
  ClusterBarrier.arrive: ClusterBarrier#arrive().
  GPUSharedMemory.get_barrier_and_increment_clock: GPUSharedMemory#get_barrier_and_increment_clock().
  GPUSharedMemory: GPUSharedMemory#
  ClusterBarrier.wait: ClusterBarrier#wait().
  GPUSharedMemory.get_smem_commit_clock: GPUSharedMemory#get_smem_commit_clock().
  GPUSharedMemory.num_blocks_per_cluster: GPUSharedMemory#num_blocks_per_cluster.
  Barrier.phase: Barrier#phase.
  HostAllocationRequest.shape_and_dtype: HostAllocationRequest#shape_and_dtype().
  GPUSharedMemory.update_smem_commit_clock: GPUSharedMemory#update_smem_commit_clock().
  GPUSharedMemory.wait_smem_to_gmem: GPUSharedMemory#wait_smem_to_gmem().
  Barrier.deallocate: Barrier#deallocate().
  GPUSharedMemory.get_next_tma_thread_id: GPUSharedMemory#get_next_tma_thread_id().
  GPUSharedMemory.add_copy_smem_to_gmem_clocks: GPUSharedMemory#add_copy_smem_to_gmem_clocks().
  GPUSharedMemory.__init__: GPUSharedMemory#__init__().
  Barrier: Barrier#
  Barrier.cv: Barrier#cv.
  HostAllocationRequest.__iter__: HostAllocationRequest#__iter__().
  ClusterBarrier.deallocate: ClusterBarrier#deallocate().
  GPUSharedMemory.num_concurrent_threads: GPUSharedMemory#num_concurrent_threads.
  GPUSharedMemory.get_barrier: GPUSharedMemory#get_barrier().
  HostAllocationRequest.memory_space_id: HostAllocationRequest#memory_space_id.
  HostAllocationRequest.initial_ref_count: HostAllocationRequest#initial_ref_count.
  GPUSharedMemory.ThreadKey: GPUSharedMemory#ThreadKey.
  GPUSharedMemory.pending_smem_to_gmem_read_clocks: GPUSharedMemory#pending_smem_to_gmem_read_clocks.
  GPUSharedMemory.update_clock: GPUSharedMemory#update_clock().
  GPUSharedMemory.get_clock: GPUSharedMemory#get_clock().
  GPUSharedMemory.execute_async_task: GPUSharedMemory#execute_async_task().
  Barrier.detect_races: Barrier#detect_races().
  GPUSharedMemory.smem_commit_clocks: GPUSharedMemory#smem_commit_clocks.
  GPUSharedMemory.pending_smem_to_gmem_write_clocks: GPUSharedMemory#pending_smem_to_gmem_write_clocks.
  GPUSharedMemory.MemKey: GPUSharedMemory#MemKey.
  GPUSharedMemory.assert_no_barriers_allocated: GPUSharedMemory#assert_no_barriers_allocated().
  GPUSharedMemory.kernel_thread_finished: GPUSharedMemory#kernel_thread_finished().
  HostAllocationRequest.device_id: HostAllocationRequest#device_id.
  HostAllocationRequest.thread_id: HostAllocationRequest#thread_id.
  ClusterBarrier: ClusterBarrier#
  ClusterBarrier.lock: ClusterBarrier#lock.
  Barrier.clock: Barrier#clock.
  Barrier.smem_commit_clock: Barrier#smem_commit_clock.
  ClusterBarrier.barriers: ClusterBarrier#barriers.
  Barrier.arrivals_count: Barrier#arrivals_count.
  Barrier.has_zero_ref_count: Barrier#has_zero_ref_count().
  HostAllocationRequest: HostAllocationRequest#
  ClusterBarrier.has_zero_ref_count: ClusterBarrier#has_zero_ref_count().
  GPUSharedMemory.get_next_wgmma_accumulator_id: GPUSharedMemory#get_next_wgmma_accumulator_id().
  GPUSharedMemory.update_clocks_for_device_barrier: GPUSharedMemory#update_clocks_for_device_barrier().
  Barrier._log: Barrier#_log().
  GPUSharedMemory.next_regs_id: GPUSharedMemory#next_regs_id.
  Barrier.shared_memory: Barrier#shared_memory.
  Barrier.last_observed_phase_by_thread: Barrier#last_observed_phase_by_thread.
  Barrier.__repr__: Barrier#__repr__().
  HostAllocationRequest.as_np_array: HostAllocationRequest#as_np_array().
  HostAllocationKey.buffer_id: HostAllocationKey#buffer_id.
  HostAllocationKey.__iter__: HostAllocationKey#__iter__().
  ClusterBarrier.axes_dims: ClusterBarrier#axes_dims.
  GPUSharedMemory.num_total_threads_per_device: GPUSharedMemory#num_total_threads_per_device().
  GPUSharedMemory.thread_to_vc_position: GPUSharedMemory#thread_to_vc_position().
  GPUSharedMemory.get_global_thread_id: GPUSharedMemory#get_global_thread_id().
  Barrier.ref_count: Barrier#ref_count.
  Barrier.enable_logging: Barrier#enable_logging.
  ClusterBarrier.ref_count: ClusterBarrier#ref_count.
  GPUSharedMemory.num_tma_threads_per_device: GPUSharedMemory#num_tma_threads_per_device.
  GPUSharedMemory.next_tma_thread_id_per_device: GPUSharedMemory#next_tma_thread_id_per_device.
  Barrier.__init__: Barrier#__init__().
  ClusterBarrier.__init__: ClusterBarrier#__init__().
  Barrier.num_arrivals: Barrier#num_arrivals.
  ClusterBarrier.enable_logging: ClusterBarrier#enable_logging.
  ClusterBarrier._log: ClusterBarrier#_log().
  logger: logger.
  Barrier.num_pallas_threads_per_block: Barrier#num_pallas_threads_per_block.
  AsyncTask: AsyncTask#
  ClusterBarrier.is_axis_collective: ClusterBarrier#is_axis_collective.
  GPUSharedMemory.num_pallas_threads_per_block: GPUSharedMemory#num_pallas_threads_per_block.
  AsyncTask.__call__: AsyncTask#__call__().
  ClusterBarrier.num_arrivals: ClusterBarrier#num_arrivals.
---
# Module: [`jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py)

## Classes
### `AsyncTask`  ·  implements/extends Protocol
- def: [`jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py:379`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L379)
- doc: Async task to be run on some non-main thread (e.g. TMA or TensorCore)
- signature: `class AsyncTask(Protocol):`
- members:
  - `__call__(self, tma_thread_id: int)` — [`L382`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L382) — Execute the async task on the given thread.
- used by: [`execute_async_task`](shared_memory.md#GPUSharedMemory.execute_async_task)

### `Barrier`  ·  implements/extends Allocation
- def: [`jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py:35`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L35)
- signature: `class Barrier(memory.Allocation):`
- members:
  - `arrive(self, *, thread_id: int, clock: vc.VectorClock | None = None, smem_commit_clock: vc.VectorClock | None = None, logging_info: interpret_utils.GPULoggingInfo | None = None)` — [`L158`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L158)
  - `deallocate(self)` — [`L139`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L139) — Deallocates the `Barrier`.
  - `detect_races(self)` — [`L132`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L132)
  - `has_zero_ref_count(self)` — [`L135`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L135)
  - `wait(self, *, device_id: int, thread_id: int, logging_info: interpret_utils.GPULoggingInfo | None = None)` — [`L216`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L216)
  - `arrivals_count` — [`L76`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L76)
  - `clock` — [`L109`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L109)
  - `cv` — [`L106`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L106)
  - `enable_logging` — [`L77`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L77)
  - `last_observed_phase_by_thread` — [`L91`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L91)
  - `num_arrivals` — [`L75`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L75)
  - `num_pallas_threads_per_block` — [`L74`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L74)
  - `phase` — [`L86`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L86)
  - `ref_count` — [`L73`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L73)
  - `shared_memory` — [`L72`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L72)
  - `smem_commit_clock` — [`L110`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L110)
- protocol/private: `__init__`[`L63`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L63), `__repr__`[`L114`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L114), `_log`[`L120`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L120)
- uses (calls/refs, reference-scoped): [`VectorClock`](../../mosaic/interpret/vector_clock.md#VectorClock), [`lock`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.lock), [`GPULoggingInfo`](../../mosaic/interpret/utils.md#GPULoggingInfo), [`format`](../../mosaic/interpret/utils.md#LoggingInfo.format), [`detect_races`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.detect_races), [`GPUSharedMemory`](shared_memory.md#GPUSharedMemory), [`clocks`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.clocks), [`copy_vector_clock`](../../mosaic/interpret/vector_clock.md#copy_vector_clock), [`update_vector_clock`](../../mosaic/interpret/vector_clock.md#update_vector_clock), [`Allocation`](../../mosaic/interpret/shared_memory.md#Allocation), [`smem_commit_clocks`](shared_memory.md#GPUSharedMemory.smem_commit_clocks), [`logger`](shared_memory.md#logger)
- used by: [`tcgen05_mma`](gpu_callbacks.md#tcgen05_mma), [`_barrier_arrive`](gpu_callbacks.md#_barrier_arrive), [`_barrier_wait`](gpu_callbacks.md#_barrier_wait), [`allocate_barrier`](shared_memory.md#GPUSharedMemory.allocate_barrier), [`deallocate_barrier`](shared_memory.md#GPUSharedMemory.deallocate_barrier), [`arrive`](shared_memory.md#ClusterBarrier.arrive), [`get_barrier_and_increment_clock`](shared_memory.md#GPUSharedMemory.get_barrier_and_increment_clock), [`wait`](shared_memory.md#ClusterBarrier.wait), [`post_write`](gpu_callbacks.md#AsyncCopyGmemToSmemTask.post_write), [`Allocation`](../../mosaic/interpret/shared_memory.md#Allocation), [`deallocate`](shared_memory.md#ClusterBarrier.deallocate), [`get_barrier`](shared_memory.md#GPUSharedMemory.get_barrier), [`assert_no_barriers_allocated`](shared_memory.md#GPUSharedMemory.assert_no_barriers_allocated), [`barriers`](shared_memory.md#ClusterBarrier.barriers), [`barrier`](gpu_callbacks.md#AsyncCopyGmemToSmemTask.barrier)

### `ClusterBarrier`  ·  implements/extends Allocation
- def: [`jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py:387`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L387)
- signature: `class ClusterBarrier(memory.Allocation):`
- members:
  - `__init__(self, shared_memory: GPUSharedMemory, *, axes_dims: tuple[int, ...], is_axis_collective: tuple[bool, ...], ref_count: int, num_arrivals: int, enable_logging: bool = False)` — [`L389`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L389) — Initializes the ClusterBarrier.
  - `arrive(self, *, thread_id: int, clock: vc.VectorClock | None = None, smem_commit_clock: vc.VectorClock | None = None, logging_info: interpret_utils.GPULoggingInfo | None = None)` — [`L466`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L466)
  - `deallocate(self)` — [`L560`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L560) — Deallocates the `ClusterBarrier`.
  - `has_zero_ref_count(self)` — [`L462`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L462)
  - `wait(self, *, device_id: int, thread_id: int, logging_info: interpret_utils.GPULoggingInfo | None = None)` — [`L534`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L534)
  - `axes_dims` — [`L413`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L413)
  - `barriers` — [`L438`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L438)
  - `enable_logging` — [`L417`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L417)
  - `is_axis_collective` — [`L414`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L414)
  - `lock` — [`L419`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L419)
  - `num_arrivals` — [`L416`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L416)
  - `ref_count` — [`L415`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L415)
- protocol/private: `_log`[`L451`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L451)
- uses (calls/refs, reference-scoped): [`arrive`](shared_memory.md#Barrier.arrive), [`VectorClock`](../../mosaic/interpret/vector_clock.md#VectorClock), [`wait`](shared_memory.md#Barrier.wait), [`GPULoggingInfo`](../../mosaic/interpret/utils.md#GPULoggingInfo), [`format`](../../mosaic/interpret/utils.md#LoggingInfo.format), [`GPUSharedMemory`](shared_memory.md#GPUSharedMemory), [`deallocate`](shared_memory.md#Barrier.deallocate), [`Allocation`](../../mosaic/interpret/shared_memory.md#Allocation), [`Barrier`](shared_memory.md#Barrier)
- used by: [`allocate_cluster_barrier`](shared_memory.md#GPUSharedMemory.allocate_cluster_barrier), [`deallocate_cluster_barrier`](shared_memory.md#GPUSharedMemory.deallocate_cluster_barrier), [`get_barrier_and_increment_clock`](shared_memory.md#GPUSharedMemory.get_barrier_and_increment_clock), [`Allocation`](../../mosaic/interpret/shared_memory.md#Allocation), [`assert_no_barriers_allocated`](shared_memory.md#GPUSharedMemory.assert_no_barriers_allocated)

### `GPUSharedMemory`
- def: [`jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py:572`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L572)
- members:
  - `add_copy_smem_to_gmem_clocks(self, thread: ThreadKey, read_clock: vc.VectorClock, write_clock: vc.VectorClock)` — [`L940`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L940) — Records read and write clocks for a completed copy from SMEM to GMEM.
  - `allocate_barrier(self, key: MemKey, ref_count: int, num_arrivals: int, logging_info: interpret_utils.GPULoggingInfo | None = None)` — [`L743`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L743) — Allocates a barrier with the given key unless it already exists.
  - `allocate_cluster_barrier(self, key: MemKey, axes_dims: tuple[int, ...], is_axis_collective: tuple[bool, ...], ref_count: int, num_arrivals: int, logging_info: interpret_utils.GPULoggingInfo | None = None)` — [`L839`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L839) — Allocates a cluster barrier with the given key unless it already exists.
  - `assert_no_barriers_allocated(self)` — [`L909`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L909)
  - `deallocate_barrier(self, key: MemKey, logging_info: interpret_utils.GPULoggingInfo | None = None)` — [`L804`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L804)
  - `deallocate_cluster_barrier(self, key: MemKey, logging_info: interpret_utils.GPULoggingInfo | None = None)` — [`L874`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L874)
  - `execute_async_task(self, task: AsyncTask, device_id: int, thread_id: int)` — [`L933`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L933) — Executes an async task immediately (intiated by the given thread).
  - `get_barrier(self, key: MemKey)` — [`L794`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L794)
  - `get_barrier_and_increment_clock(self, key: MemKey, device_id: int, thread_id: int)` — [`L774`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L774)
  - `get_clock(self, vector_clock_idx)` — [`L739`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L739)
  - `get_global_thread_id(self, device_id: int, local_thread_id: int)` — [`L705`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L705) — Computes the global thread ID from the given device and local thread ID.
  - `get_next_tma_thread_id(self, device_id: int)` — [`L715`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L715)
  - `get_next_wgmma_accumulator_id(self, device_id: int, thread_id: int)` — [`L726`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L726)
  - `get_smem_commit_clock(self, thread: ThreadKey)` — [`L927`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L927)
  - `kernel_thread_finished(self, device_id: int, thread_id: int)` — [`L976`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L976) — Called when a thread completes execution of a kernel.
  - `num_total_threads_per_device(self)` — [`L702`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L702)
  - `thread_to_vc_position(self, thread: ThreadKey)` — [`L712`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L712)
  - `update_clock(self, vector_clock_idx, clock: vc.VectorClock)` — [`L734`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L734)
  - `update_clocks_for_device_barrier(self, device_id)` — [`L989`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L989) — Synchronizes the vector clocks for the cores on the given device.
  - `update_smem_commit_clock(self, thread: ThreadKey)` — [`L918`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L918) — Sets the smem commit clock for the given core to its current clock.
  - `wait_smem_to_gmem(self, device_id: int, thread_id: int, n: int, wait_read_only: bool)` — [`L953`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L953) — Ensures no more than n SMEM to GMEM copies are outstanding.
  - `MemKey` — [`L575`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L575)
  - `ThreadKey` — [`L577`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L577)
  - `next_regs_id` — [`L588`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L588)
  - `next_tma_thread_id_per_device` — [`L582`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L582)
  - `num_blocks_per_cluster` — [`L684`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L684)
  - `num_concurrent_threads` — [`L579`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L579)
  - `num_pallas_threads_per_block` — [`L683`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L683)
  - `num_tma_threads_per_device` — [`L580`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L580)
  - `pending_smem_to_gmem_read_clocks` — [`L596`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L596)
  - `pending_smem_to_gmem_write_clocks` — [`L597`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L597)
  - `smem_commit_clocks` — [`L592`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L592)
- protocol/private: `__init__`[`L599`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L599)
- uses (calls/refs, reference-scoped): [`HostAllocationKey`](shared_memory.md#HostAllocationKey), [`VectorClock`](../../mosaic/interpret/vector_clock.md#VectorClock), [`lock`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.lock), [`GPULoggingInfo`](../../mosaic/interpret/utils.md#GPULoggingInfo), [`format`](../../mosaic/interpret/utils.md#LoggingInfo.format), [`detect_races`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.detect_races), [`incr_clock`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.incr_clock), [`mem`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.mem), [`GenericSharedMemory`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory), [`enable_logging`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.enable_logging), [`clocks`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.clocks), [`copy_vector_clock`](../../mosaic/interpret/vector_clock.md#copy_vector_clock), [`deallocate`](shared_memory.md#Barrier.deallocate), [`update_vector_clock`](../../mosaic/interpret/vector_clock.md#update_vector_clock), [`_log`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory._log), [`update_clocks`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.update_clocks), [`logging_mode`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.logging_mode), [`Barrier`](shared_memory.md#Barrier), [`deallocate`](shared_memory.md#ClusterBarrier.deallocate), [`ClusterBarrier`](shared_memory.md#ClusterBarrier), [`has_zero_ref_count`](shared_memory.md#Barrier.has_zero_ref_count), [`has_zero_ref_count`](shared_memory.md#ClusterBarrier.has_zero_ref_count), [`LoggingMode`](params.md#LoggingMode), [`make_vector_clock`](../../mosaic/interpret/vector_clock.md#make_vector_clock), [`num_devices`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.num_devices), [`dma_execution_mode`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.dma_execution_mode), [`BARRIER`](../../mosaic/interpret/params.md#LoggingMode.BARRIER), [`AsyncTask`](shared_memory.md#AsyncTask)
- used by: [`_allocate_buffer`](gpu_callbacks.md#_allocate_buffer), [`_allocate_buffer_for_all_threads`](gpu_callbacks.md#_allocate_buffer_for_all_threads), [`_allocate_barriers`](gpu_callbacks.md#_allocate_barriers), [`_allocate_cluster_barriers`](gpu_callbacks.md#_allocate_cluster_barriers), [`tcgen05_mma`](gpu_callbacks.md#tcgen05_mma), [`wait`](shared_memory.md#Barrier.wait), [`post_read`](gpu_callbacks.md#AsyncCopySmemToGmemTask.post_read), [`pre_read`](gpu_callbacks.md#AsyncCopySmemToGmemTask.pre_read), [`_deallocate_barrier`](gpu_callbacks.md#_deallocate_barrier), [`_initialize_shared_memory`](gpu_callbacks.md#_initialize_shared_memory), [`post_read`](gpu_callbacks.md#AsyncCopyGmemToSmemTask.post_read), [`pre_read`](gpu_callbacks.md#AsyncCopyGmemToSmemTask.pre_read), [`_barrier_arrive`](gpu_callbacks.md#_barrier_arrive), [`_barrier_wait`](gpu_callbacks.md#_barrier_wait), [`_get_shared_memory`](gpu_callbacks.md#_get_shared_memory), [`copy_smem_to_gmem`](gpu_callbacks.md#copy_smem_to_gmem), [`copy_gmem_to_smem`](gpu_callbacks.md#copy_gmem_to_smem), [`__init__`](gpu_callbacks.md#AsyncCopyGmemToSmemTask.__init__), [`post_write`](gpu_callbacks.md#AsyncCopyGmemToSmemTask.post_write), [`post_write`](gpu_callbacks.md#AsyncCopySmemToGmemTask.post_write), [`get_races`](gpu_callbacks.md#get_races), [`_handle_out_of_bounds_read`](gpu_callbacks.md#_handle_out_of_bounds_read), [`_races`](gpu_callbacks.md#_races._races), [`_shared_memory`](gpu_callbacks.md#_shared_memory._shared_memory), [`commit_smem`](gpu_callbacks.md#commit_smem), [`wait_smem_to_gmem`](gpu_callbacks.md#wait_smem_to_gmem), [`_update_clocks_for_device_barrier`](gpu_callbacks.md#_update_clocks_for_device_barrier), [`kernel_thread_finished`](gpu_callbacks.md#kernel_thread_finished), [`post_read`](gpu_callbacks.md#AsyncCopyTask.post_read), [`post_write`](gpu_callbacks.md#AsyncCopyTask.post_write), [`pre_read`](gpu_callbacks.md#AsyncCopyTask.pre_read), [`thread_to_vc_position`](../../mosaic/interpret/shared_memory.md#GenericSharedMemory.thread_to_vc_position), [`_assert_no_barriers_allocated`](gpu_callbacks.md#_assert_no_barriers_allocated), [`__init__`](shared_memory.md#Barrier.__init__), [`__init__`](shared_memory.md#ClusterBarrier.__init__)

### `HostAllocationKey`
- def: [`jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py:367`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L367)
- members:
  - `buffer_id` — [`L370`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L370)
- protocol/private: `__iter__`[`L372`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L372)
- uses (calls/refs, reference-scoped): [`__iter__`](shared_memory.md#HostAllocationRequest.__iter__), [`HostAllocationRequest`](shared_memory.md#HostAllocationRequest)
- used by: [`_allocate_for_aval`](jaxpr_interpret.md#JaxprInterpreter._allocate_for_aval), [`_allocate_buffer`](gpu_callbacks.md#_allocate_buffer), [`_allocate_buffer_for_all_threads`](gpu_callbacks.md#_allocate_buffer_for_all_threads), [`_allocate_barriers`](gpu_callbacks.md#_allocate_barriers), [`_allocate_cluster_barriers`](gpu_callbacks.md#_allocate_cluster_barriers), [`__init__`](gpu_callbacks.md#AsyncCopyTask.__init__), [`_get`](gpu_callbacks.md#_get), [`_swap`](gpu_callbacks.md#_swap), [`tcgen05_mma`](gpu_callbacks.md#tcgen05_mma), [`wgmma`](gpu_callbacks.md#wgmma), [`_deallocate_barrier`](gpu_callbacks.md#_deallocate_barrier), [`_barrier_arrive`](gpu_callbacks.md#_barrier_arrive), [`async_load_tmem`](gpu_callbacks.md#async_load_tmem), [`_barrier_wait`](gpu_callbacks.md#_barrier_wait), [`copy_smem_to_gmem`](gpu_callbacks.md#copy_smem_to_gmem), [`_deallocate_buffer`](gpu_callbacks.md#_deallocate_buffer), [`copy_gmem_to_smem`](gpu_callbacks.md#copy_gmem_to_smem), [`wgmma_accumulator_deref`](gpu_callbacks.md#wgmma_accumulator_deref), [`call_allocate_barriers`](gpu_callbacks.md#call_allocate_barriers), [`call_allocate_cluster_barriers`](gpu_callbacks.md#call_allocate_cluster_barriers), [`__init__`](gpu_callbacks.md#AsyncCopyGmemToSmemTask.__init__), [`GPUSharedMemory`](shared_memory.md#GPUSharedMemory), [`call_allocate_buffer_for_all_threads`](gpu_callbacks.md#call_allocate_buffer_for_all_threads), [`call_allocate_buffer`](gpu_callbacks.md#call_allocate_buffer), [`__init__`](gpu_callbacks.md#AsyncCopySmemToGmemTask.__init__), [`_handle_out_of_bounds_read`](gpu_callbacks.md#_handle_out_of_bounds_read), [`_get_barrier_allocation_key_from_inval`](jaxpr_interpret.md#_get_barrier_allocation_key_from_inval), [`MemKey`](shared_memory.md#GPUSharedMemory.MemKey), [`dst_allocation_key`](gpu_callbacks.md#AsyncCopyTask.dst_allocation_key), [`src_allocation_key`](gpu_callbacks.md#AsyncCopyTask.src_allocation_key)

### `HostAllocationRequest`
- def: [`jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py:316`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L316)
- doc: Request for an allocation on a device/thread and in a memory space.
- signature: `class HostAllocationRequest:`
- members:
  - `as_np_array(self)` — [`L348`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L348)
  - `from_array(cls, request: jax.Array | np.ndarray)` — [`L352`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L352)
  - `shape_and_dtype(cls)` — [`L343`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L343)
  - `device_id` — [`L320`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L320)
  - `initial_ref_count` — [`L326`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L326)
  - `memory_space_id` — [`L319`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L319)
  - `thread_id` — [`L322`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L322)
- protocol/private: `__iter__`[`L328`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L328)
- uses (calls/refs, reference-scoped): [`Array`](../../../basearray.md#Array), [`shape`](../../../basearray.md#Array.shape), [`dtype`](../../../basearray.md#Array.dtype), [`int32`](../../../../numpy/__init__.pyi.md#int32.int32), [`ShapeDtypeStruct`](../../../core.md#ShapeDtypeStruct), [`shape`](../../../core.md#ShapeDtypeStruct.shape), [`is_int`](../../mosaic/interpret/utils.md#is_int)
- used by: [`_allocate_for_aval`](jaxpr_interpret.md#JaxprInterpreter._allocate_for_aval), [`_allocate_buffer`](gpu_callbacks.md#_allocate_buffer), [`_allocate_buffer_for_all_threads`](gpu_callbacks.md#_allocate_buffer_for_all_threads), [`_allocate_barriers`](gpu_callbacks.md#_allocate_barriers), [`HostAllocationKey`](shared_memory.md#HostAllocationKey), [`_allocate_cluster_barriers`](gpu_callbacks.md#_allocate_cluster_barriers), [`_get`](gpu_callbacks.md#_get), [`_swap`](gpu_callbacks.md#_swap), [`tcgen05_mma`](gpu_callbacks.md#tcgen05_mma), [`wgmma`](gpu_callbacks.md#wgmma), [`_deallocate_barrier`](gpu_callbacks.md#_deallocate_barrier), [`_barrier_arrive`](gpu_callbacks.md#_barrier_arrive), [`async_load_tmem`](gpu_callbacks.md#async_load_tmem), [`_barrier_wait`](gpu_callbacks.md#_barrier_wait), [`copy_smem_to_gmem`](gpu_callbacks.md#copy_smem_to_gmem), [`_deallocate_buffer`](gpu_callbacks.md#_deallocate_buffer), [`copy_gmem_to_smem`](gpu_callbacks.md#copy_gmem_to_smem), [`wgmma_accumulator_deref`](gpu_callbacks.md#wgmma_accumulator_deref), [`call_allocate_barriers`](gpu_callbacks.md#call_allocate_barriers), [`call_allocate_cluster_barriers`](gpu_callbacks.md#call_allocate_cluster_barriers), [`call_allocate_buffer_for_all_threads`](gpu_callbacks.md#call_allocate_buffer_for_all_threads), [`call_make_allocation_request_array`](gpu_callbacks.md#call_make_allocation_request_array), [`_make_allocation_request_array`](gpu_callbacks.md#_make_allocation_request_array), [`call_allocate_buffer`](gpu_callbacks.md#call_allocate_buffer), [`_get_barrier_allocation_key_from_inval`](jaxpr_interpret.md#_get_barrier_allocation_key_from_inval), [`__iter__`](shared_memory.md#HostAllocationKey.__iter__)

## Module values
- `logger` — [`L32`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/shared_memory.py#L32)

