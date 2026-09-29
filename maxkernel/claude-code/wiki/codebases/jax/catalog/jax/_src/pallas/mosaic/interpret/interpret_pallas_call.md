---
title: 'Module: jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.interpret.interpret_pallas_call`/
symbols:
  _interpret_jaxpr: _interpret_jaxpr().
  interpret_pallas_call: interpret_pallas_call().
  interpret_pallas_call._execute_grid_for_core._body: interpret_pallas_call()._execute_grid_for_core()._body().
  DMA.execute_read: DMA#execute_read().
  DMA.execute_write: DMA#execute_write().
  get: get().
  _allocate_buffer: _allocate_buffer().
  interpret_pallas_call._execute_grid_for_core: interpret_pallas_call()._execute_grid_for_core().
  dma_start: dma_start().
  store: store().
  TOKEN_SHAPE_DTYPE: TOKEN_SHAPE_DTYPE.
  interpret_pallas_call._execute_grid_for_core._body._store_slice_to_kernel_input: interpret_pallas_call()._execute_grid_for_core()._body()._store_slice_to_kernel_input().
  swap: swap().
  _initialize_shared_memory: _initialize_shared_memory().
  TPU_MEMORY_SPACE_IDXS.TPU_MEMORY_SPACE_IDXS: TPU_MEMORY_SPACE_IDXS.TPU_MEMORY_SPACE_IDXS.
  _get_randomized_grid_coordinates: _get_randomized_grid_coordinates().
  host_idx: host_idx.
  _get_shared_memory: _get_shared_memory().
  _deallocate_buffer: _deallocate_buffer().
  dma_wait: dma_wait().
  semaphore_signal: semaphore_signal().
  semaphore_wait: semaphore_wait().
  _remove_memory_space_abstract_eval: _remove_memory_space_abstract_eval().
  interpret_pallas_call._execute_grid_for_core._body._store_to_output_buffer: interpret_pallas_call()._execute_grid_for_core()._body()._store_to_output_buffer().
  fail_on_exception: fail_on_exception().
  races.races: races.races.
  _allocate_semaphores: _allocate_semaphores().
  any_idx: any_idx.
  _validate: _validate().
  _forward_any_to_hbm: _forward_any_to_hbm().
  InterpretContext.device_id: InterpretContext#device_id.
  InterpretContext.local_core_id: InterpretContext#local_core_id.
  _shared_memory._shared_memory: _shared_memory._shared_memory.
  _compute_start_indices: _compute_start_indices().
  _get_grid_point: _get_grid_point().
  TPU_MEMORY_SPACE_NAMES: TPU_MEMORY_SPACE_NAMES.
  DMA.clock: DMA#clock.
  remove_memory_space_p: remove_memory_space_p.
  fail_on_exception.wrapper: fail_on_exception().wrapper().
  reset_tpu_interpret_mode_state: reset_tpu_interpret_mode_state().
  _update_clocks_for_global_barrier: _update_clocks_for_global_barrier().
  _barrier: _barrier().
  _check_for_revisiting: _check_for_revisiting().
  DMA.src_global_core_id: DMA#src_global_core_id().
  DMA.dst_global_core_id: DMA#dst_global_core_id().
  _compute_start_indices._get_start_index: _compute_start_indices()._get_start_index().
  DMA.state: DMA#state.
  _HBM: _HBM.
  _get_memory_space_and_raise_if_hbm: _get_memory_space_and_raise_if_hbm().
  InterpretContext.axis_indices: InterpretContext#axis_indices.
  _update_clocks_for_device_barrier: _update_clocks_for_device_barrier().
  _get_parallel_dim_semantics: _get_parallel_dim_semantics().
  interpret_pallas_call._get_local_grid_env: interpret_pallas_call()._get_local_grid_env().
  DMA.detect_races: DMA#detect_races().
  _clean_up_shared_memory: _clean_up_shared_memory().
  dma_id_counter.dma_id_counter: dma_id_counter.dma_id_counter.
  DmaState: DmaState#
  DMA.execute_read_and_write: DMA#execute_read_and_write().
  _SEMAPHORE: _SEMAPHORE.
  DMA.source_info: DMA#source_info.
  register_tpu_interpret_impl: register_tpu_interpret_impl().
  InterpretContext.interpret_params: InterpretContext#interpret_params.
  DMA.data: DMA#data.
  _to_int: _to_int().
  DMA.dst_sem: DMA#dst_sem.
  InterpretContext.grid_mapping: InterpretContext#grid_mapping.
  fail: fail().
  force_tpu_interpret_mode: force_tpu_interpret_mode().
  set_tpu_interpret_mode: set_tpu_interpret_mode().
  _clear_shared_memory: _clear_shared_memory().
  _get_padded_shape: _get_padded_shape().
  _get_with_padding: _get_with_padding().
  get_barrier_semaphore: get_barrier_semaphore().
  _ANY: _ANY.
  register_tpu_interpret_impl.decorator: register_tpu_interpret_impl().decorator().
  InterpretContext.mesh: InterpretContext#mesh.
  DMA.virtual_device_id: DMA#virtual_device_id.
  _remove_memory_space_impl: _remove_memory_space_impl().
  DMA.src_sem: DMA#src_sem.
  DMA.data_size: DMA#data_size().
  InterpretContext.mosaic_params: InterpretContext#mosaic_params.
  _GridPointCoordinatesPerDim: _GridPointCoordinatesPerDim.
  _interpret_impls._interpret_impls: _interpret_impls._interpret_impls.
  _local_core_id_or_zero_if_hbm: _local_core_id_or_zero_if_hbm().
  T: T.
  InterpretContext: InterpretContext#
  _SENTINEL: _SENTINEL.
  InterpretContext.replace: InterpretContext#replace().
  _interpret_jaxpr._scan_body: _interpret_jaxpr()._scan_body().
  _shared_memory_init_lock: _shared_memory_init_lock.
  DmaState.COMPLETED: DmaState#COMPLETED.
  DMA.src_device_id: DMA#src_device_id.
  DMA.src_local_core_id: DMA#src_local_core_id.
  DMA.dst_device_id: DMA#dst_device_id.
  DMA.dst_local_core_id: DMA#dst_local_core_id.
  InterpretContext.axis_sizes: InterpretContext#axis_sizes.
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  TOP_LEVEL_TOKEN_VALUE: TOP_LEVEL_TOKEN_VALUE.
  ordering_barrier: ordering_barrier().
  DmaState.STARTED: DmaState#STARTED.
  DmaState.READ: DmaState#READ.
  DMA.id: DMA#id.
  DMA.lock: DMA#lock.
  InterpretContext.grid_point: InterpretContext#grid_point.
  Token: Token.
  DMA: DMA#
  DMA.src_memory_space: DMA#src_memory_space.
  DMA.src_buffer_id: DMA#src_buffer_id.
  DMA.src_transforms: DMA#src_transforms.
  DMA.dst_memory_space: DMA#dst_memory_space.
  DMA.dst_buffer_id: DMA#dst_buffer_id.
  DMA.dst_transforms: DMA#dst_transforms.
  _interpret_jaxpr._make_branch: _interpret_jaxpr()._make_branch().
  _interpret_jaxpr._body: _interpret_jaxpr()._body().
  _get_parallel_subgrid_size: _get_parallel_subgrid_size().
  _remove_memory_space_lowering: _remove_memory_space_lowering().
---
# Module: [`jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py)

## Classes
### `DMA`
- def: [`jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:892`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L892)
- signature: `class DMA:`
- members:
  - `data_size(self)` — [`L917`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L917)
  - `detect_races(self)` — [`L922`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L922)
  - `dst_global_core_id(self)` — [`L932`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L932)
  - `execute_read(self)` — [`L937`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L937) — Executes the reading part of this DMA.
  - `execute_read_and_write(self)` — [`L1029`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1029) — Executes this DMA, both the reading and writing parts.
  - `execute_write(self)` — [`L982`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L982) — Executes the writing part of this DMA.
  - `src_global_core_id(self)` — [`L926`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L926)
  - `clock` — [`L908`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L908)
  - `data` — [`L913`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L913)
  - `dst_buffer_id` — [`L903`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L903)
  - `dst_device_id` — [`L900`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L900)
  - `dst_local_core_id` — [`L901`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L901)
  - `dst_memory_space` — [`L902`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L902)
  - `dst_sem` — [`L906`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L906)
  - `dst_transforms` — [`L904`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L904)
  - `id` — [`L893`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L893)
  - `lock` — [`L914`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L914)
  - `source_info` — [`L910`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L910)
  - `src_buffer_id` — [`L898`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L898)
  - `src_device_id` — [`L895`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L895)
  - `src_local_core_id` — [`L896`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L896)
  - `src_memory_space` — [`L897`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L897)
  - `src_sem` — [`L905`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L905)
  - `src_transforms` — [`L899`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L899)
  - `state` — [`L912`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L912)
  - `virtual_device_id` — [`L907`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L907)
- uses (calls/refs, reference-scoped): [`get`](interpret_pallas_call.md#get), [`SourceInfo`](../../../source_info_util.md#SourceInfo), [`VectorClock`](vector_clock.md#VectorClock), [`store`](interpret_pallas_call.md#store), [`signal`](shared_memory.md#Semaphore.signal), [`source_info`](utils.md#LoggingInfo.source_info), [`device_id`](utils.md#LoggingInfo.device_id), [`copy_vector_clock`](vector_clock.md#copy_vector_clock), [`inc_vector_clock`](vector_clock.md#inc_vector_clock), [`TPULoggingInfo`](utils.md#TPULoggingInfo), [`local_core_id`](utils.md#TPULoggingInfo.local_core_id), [`Semaphore`](shared_memory.md#Semaphore), [`DmaState`](interpret_pallas_call.md#DmaState), [`detect_races`](shared_memory.md#Semaphore.detect_races), [`get_global_core_id`](shared_memory.md#Semaphore.get_global_core_id), [`COMPLETED`](interpret_pallas_call.md#DmaState.COMPLETED), [`READ`](interpret_pallas_call.md#DmaState.READ), [`STARTED`](interpret_pallas_call.md#DmaState.STARTED)
- used by: [`dma_start`](interpret_pallas_call.md#dma_start)

### `DmaState`  ·  implements/extends Enum
- def: [`jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:885`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L885)
- signature: `class DmaState(enum.Enum):`
- members:
  - `COMPLETED` — [`L888`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L888)
  - `READ` — [`L887`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L887)
  - `STARTED` — [`L886`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L886)
- used by: [`execute_read`](interpret_pallas_call.md#DMA.execute_read), [`execute_write`](interpret_pallas_call.md#DMA.execute_write), [`state`](interpret_pallas_call.md#DMA.state)

### `InterpretContext`
- def: [`jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py:1277`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1277)
- signature: `class InterpretContext:`
- members:
  - `replace(self, **changes)` — [`L1288`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1288)
  - `axis_indices` — [`L1282`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1282)
  - `axis_sizes` — [`L1281`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1281)
  - `device_id` — [`L1283`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1283)
  - `grid_mapping` — [`L1279`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1279)
  - `grid_point` — [`L1278`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1278)
  - `interpret_params` — [`L1286`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1286)
  - `local_core_id` — [`L1284`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1284)
  - `mesh` — [`L1280`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1280)
  - `mosaic_params` — [`L1285`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1285)
- uses (calls/refs, reference-scoped): [`Array`](../../../basearray.md#Array), [`Mesh`](../../core.md#Mesh), [`AxisName`](../../../core.md#AxisName), [`CompilerParams`](../core.md#CompilerParams), [`GridMapping`](../../core.md#GridMapping), [`InterpretParams`](params.md#InterpretParams)
- used by: [`_interpret_jaxpr`](interpret_pallas_call.md#_interpret_jaxpr), [`_body`](interpret_pallas_call.md#interpret_pallas_call._execute_grid_for_core._body), [`_execute_grid_for_core`](interpret_pallas_call.md#interpret_pallas_call._execute_grid_for_core)

## Functions
- `_allocate_buffer(token: Array, device_id: Array, local_core_id: Array | None, memory_space: Array, val: Array, source_info: source_info_util.SourceInfo | None = None)` — [`L358`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L358) — Allocates a memory buffer on the device with id `device_id` and core with id `local_core_id`.
- `_allocate_semaphores(token: Array, device_id: Array, local_core_id: Array | None, shape: Array)` — [`L479`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L479) — Allocates semaphores on the device with id `device_id` and core with id `local_core_id`.
- `_barrier(token, device_id)` — [`L238`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L238)
- `_body(val)` — [`L1437`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1437)
- `_body(carry: tuple[jnp.int32, tuple[jnp.int32, ...], jnp.ndarray, tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], jnp.int32,], ctx: InterpretContext)` — [`L2174`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L2174) — Performs one execution of the kernel body.
- `_check_for_revisiting(token, device_id, local_core_id, loop_idx, output_blocks)` — [`L259`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L259)
- `_clean_up_shared_memory(token, device_id)` — [`L247`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L247)
- `_clear_shared_memory()` — [`L172`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L172)
- `_compute_start_indices(block_mapping, loop_idx, *args, ctx, token)` — [`L1713`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1713)
- `_deallocate_buffer(token, device_id, local_core_id, memory_space, buffer_id, source_info=None)` — [`L455`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L455)
- `_execute_grid_for_core(core_index, token)` — [`L2153`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L2153)
- `_forward_any_to_hbm(memory_space)` — [`L1235`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1235)
- `_get_grid_point(loop_indices: tuple[Array, ...], grid_point_coordinates: _GridPointCoordinatesPerDim)` — [`L1852`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1852) — Indexes each entry in `grid_point_coordinates` with the corresponding entry in `loop_indices`.
- `_get_local_grid_env(grid_point)` — [`L2143`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L2143)
- `_get_memory_space_and_raise_if_hbm(aval, primitive_name, message=None)` — [`L1244`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1244)
- `_get_padded_shape(logical_shape: tuple[int, ...], dtype: jnp.dtype)` — [`L321`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L321)
- `_get_parallel_dim_semantics(mosaic_params: mosaic_core.CompilerParams, num_dimensions_in_grid: int)` — [`L1743`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1743) — Returns a tuple indicating which grid dimensions have parallel semantics.
- `_get_parallel_subgrid_size(parallel_semantics_per_dim: tuple[bool, ...], grid: tuple[int, ...])` — [`L1772`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1772) — Returns the size of the subgrid along the parallel dimensions.
- `_get_randomized_grid_coordinates(grid: tuple[int, ...], mosaic_params: mosaic_core.CompilerParams, random_seed: int | None)` — [`L1783`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1783) — Returns a tuple of randomized coordinates for each 'parallel' dimension in `grid`.
- `_get_shared_memory()` — [`L167`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L167)
- `_get_start_index(i, b)` — [`L1723`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1723)
- `_get_with_padding(x: np.ndarray, uninitialized_memory: Literal['nan', 'zero'])` — [`L345`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L345)
- `_initialize_shared_memory(token, device_id, num_devices, num_cores_per_device, *, interpret_params)` — [`L182`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L182)
- `_interpret_jaxpr(jaxpr, *args, ctx: InterpretContext, token: Array)` — [`L1292`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1292)
- `_local_core_id_or_zero_if_hbm(local_core_id: int, memory_space: str)` — [`L448`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L448)
- `_make_branch(jaxpr)` — [`L1405`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1405)
- `_remove_memory_space_abstract_eval(x)` — [`L1834`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1834)
- `_remove_memory_space_impl(x)` — [`L1844`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1844)
- `_remove_memory_space_lowering(_, x)` — [`L1847`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1847)
- `_scan_body(c, a)` — [`L1419`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1419)
- `_store_slice_to_kernel_input(index, input_var, memory_space, token)` — [`L2259`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L2259)
- `_store_to_output_buffer(index, output_var, transform, memory_space, token)` — [`L2340`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L2340)
- `_to_int(x: int | Array | None)` — [`L580`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L580) — Converts a value to an integer, or returns None if the value is None.
- `_update_clocks_for_device_barrier(token, device_id)` — [`L219`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L219) — Synchronizes the vector clocks for the cores on the given device.
- `_update_clocks_for_global_barrier()` — [`L226`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L226) — Synchronizes all vector clocks.
- `_validate(token, device_id)` — [`L304`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L304)
- `decorator(impl: T)` — [`L1269`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1269)
- `dma_start(token, device_id, src_local_core_id, src_memory_space, src_id, src_transforms, dst_memory_space, dst_id, dst_transforms, dst_sem_id, src_sem_id, dst_device_id, source_info=None)` — [`L1038`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1038)
- `dma_wait(token, device_id, local_core_id, sem_id, size, source_info=None)` — [`L1132`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1132)
- `fail(e: Exception, device_id: int | None)` — [`L66`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L66)
- `fail_on_exception(func)` — [`L71`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L71)
- `force_tpu_interpret_mode(params: InterpretParams = InterpretParams())` — [`L113`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L113) — Context manager that forces TPU interpret mode under its dynamic context.
- `get(token, device_id, local_core_id, memory_space, buffer_id, transforms, block_indices=None, grid_loop_idx=None, *, src_device_id=None, src_local_core_id=None, clock=None, source_info=None, input_name=None)` — [`L588`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L588)
- `get_barrier_semaphore(token, device_id, collective_id)` — [`L572`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L572)
- `interpret_pallas_call(*args, jaxpr: jax_core.Jaxpr, debug: bool, input_output_aliases: tuple[tuple[int, int], ...], grid_mapping: pallas_core.GridMapping, mesh: pallas_core.Mesh | None, compiler_params: pallas_core.CompilerParams | None, cost_estimate: pallas_core.CostEstimate, out_avals: tuple[jax_core.AbstractValue, ...], interpret_params: InterpretParams, metadata: frozen_dict.FrozenDict[str, str] | None, name: str | None)` — [`L1883`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1883)
- `ordering_barrier(token)` — [`L178`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L178)
- `register_tpu_interpret_impl(prim: jax_core.Primitive)` — [`L1262`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1262) — Registers an alternate primitive implementation for TPU Interpret Mode.
- `reset_tpu_interpret_mode_state()` — [`L148`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L148) — Resets all global, shared state used by TPU interpret mode.
- `semaphore_signal(token, device_id, local_core_id, sem_id, inc, target_device_id, target_local_core_id, source_info=None)` — [`L1159`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1159)
- `semaphore_wait(token, device_id, local_core_id, sem_id, value, source_info=None)` — [`L1203`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1203)
- `set_tpu_interpret_mode(params: InterpretParams = InterpretParams())` — [`L136`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L136)
- `store(token, device_id, local_core_id, memory_space, buffer_id, transforms, val, block_indices=None, grid_loop_idx=None, *, src_device_id=None, src_local_core_id=None, clock=None, source_info=None, output_name=None)` — [`L718`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L718)
- `swap(token, device_id, local_core_id, memory_space, buffer_id, transforms, val, mask, *, source_info=None)` — [`L810`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L810)
- `wrapper(*args, **kwargs)` — [`L73`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L73)

## Module values
- `T` — [`L1259`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1259)
- `TOKEN_SHAPE_DTYPE` — [`L1880`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1880)
- `TOP_LEVEL_TOKEN_VALUE` — [`L63`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L63)
- `TPU_MEMORY_SPACE_IDXS` — [`L545`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L545)
- `TPU_MEMORY_SPACE_NAMES` — [`L552`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L552)
- `Token` — [`L58`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L58)
- `_ANY` — [`L1232`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1232)
- `_GridPointCoordinatesPerDim` — [`L1781`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1781)
- `_HBM` — [`L1231`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1231)
- `_SEMAPHORE` — [`L1230`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1230)
- `_SENTINEL` — [`L1241`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1241)
- `_interpret_impls` — [`L1257`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1257)
- `_shared_memory` — [`L142`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L142)
- `_shared_memory_init_lock` — [`L143`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L143)
- `any_idx` — [`L557`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L557)
- `dma_id_counter` — [`L145`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L145)
- `host_idx` — [`L562`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L562)
- `races` — [`L144`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L144)
- `remove_memory_space_p` — [`L1831`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L1831)
- `unsafe_map` — [`L60`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L60)
- `unsafe_zip` — [`L61`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L61)

