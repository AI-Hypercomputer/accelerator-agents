---
title: 'Module: jax/_src/pallas/mosaic/interpret/params.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/interpret/params.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.interpret.params`/
symbols:
  InterpretParams: InterpretParams#
  SharedInterpretParams.uninitialized_memory: SharedInterpretParams#uninitialized_memory.
  SharedInterpretParams.vector_clock_size: SharedInterpretParams#vector_clock_size.
  LoggingMode: LoggingMode#
  InterpretParams.num_cores_per_device: InterpretParams#num_cores_per_device().
  SharedInterpretParams: SharedInterpretParams#
  SharedInterpretParams.__post_init__: SharedInterpretParams#__post_init__().
  get_interpret_effects: get_interpret_effects().
  SharedInterpretParams.get_vector_clock_size: SharedInterpretParams#get_vector_clock_size().
  SharedInterpretParams.skip_floating_point_ops: SharedInterpretParams#skip_floating_point_ops.
  SharedInterpretParams.num_cores_or_threads: SharedInterpretParams#num_cores_or_threads.
  SharedInterpretParams.logging_mode: SharedInterpretParams#logging_mode.
  LoggingMode.BARRIER: LoggingMode#BARRIER.
  LoggingMode.SEMAPHORE: LoggingMode#SEMAPHORE.
  InterpretParams.grid_point_recorder: InterpretParams#grid_point_recorder.
  LoggingMode.SHARED_MEMORY: LoggingMode#SHARED_MEMORY.
  SharedInterpretParams.detect_races: SharedInterpretParams#detect_races.
  SharedInterpretParams.out_of_bounds_reads: SharedInterpretParams#out_of_bounds_reads.
  InterpretParams.random_seed: InterpretParams#random_seed.
  InterpretParams.allow_hbm_allocation_in_run_scoped: InterpretParams#allow_hbm_allocation_in_run_scoped.
  InterpretParams.dma_execution_mode: InterpretParams#dma_execution_mode.
  InterpretParams.buffer_bounds: InterpretParams#buffer_bounds.
---
# Module: [`jax/_src/pallas/mosaic/interpret/params.py`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py)

## Classes
### `InterpretParams`  ·  implements/extends SharedInterpretParams
- def: [`jax/_src/pallas/mosaic/interpret/params.py:121`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L121)
- doc: Parameters for TPU interpret mode.
- signature: `class InterpretParams(SharedInterpretParams):`
- members:
  - `num_cores_per_device(self)` — [`L187`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L187)
  - `allow_hbm_allocation_in_run_scoped` — [`L183`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L183)
  - `buffer_bounds` — [`L184`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L184)
  - `dma_execution_mode` — [`L178`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L178)
  - `grid_point_recorder` — [`L180`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L180)
  - `random_seed` — [`L179`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L179)
- uses (calls/refs, reference-scoped): [`SharedInterpretParams`](params.md#SharedInterpretParams), [`num_cores_or_threads`](params.md#SharedInterpretParams.num_cores_or_threads)
- used by: [`_interpret_jaxpr`](interpret_pallas_call.md#_interpret_jaxpr), [`interpret_pallas_call`](interpret_pallas_call.md#interpret_pallas_call), [`_body`](interpret_pallas_call.md#interpret_pallas_call._execute_grid_for_core._body), [`_pallas_call_lowering`](../../pallas_call.md#_pallas_call_lowering), [`get_interpret_effects`](../../core.md#get_interpret_effects), [`interpret_params`](interpret_pallas_call.md#InterpretContext.interpret_params), [`SharedInterpretParams`](params.md#SharedInterpretParams), [`force_tpu_interpret_mode`](interpret_pallas_call.md#force_tpu_interpret_mode), [`set_tpu_interpret_mode`](interpret_pallas_call.md#set_tpu_interpret_mode)

### `LoggingMode`  ·  implements/extends Flag
- def: [`jax/_src/pallas/mosaic/interpret/params.py:24`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L24)
- doc: Logging mode for the kernel interpreter.
- signature: `class LoggingMode(enum.Flag):`
- members:
  - `BARRIER` — [`L33`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L33)
  - `SEMAPHORE` — [`L34`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L34)
  - `SHARED_MEMORY` — [`L35`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L35)
- used by: [`allocate_barrier`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.allocate_barrier), [`allocate_cluster_barrier`](../../mosaic_gpu/interpret/shared_memory.md#GPUSharedMemory.allocate_cluster_barrier), [`allocate_semaphores`](shared_memory.md#SharedMemory.allocate_semaphores), [`enable_logging`](shared_memory.md#GenericSharedMemory.enable_logging), [`guarantee_semaphore_with_fixed_id`](shared_memory.md#SharedMemory.guarantee_semaphore_with_fixed_id), [`__init__`](shared_memory.md#SharedMemory.__init__), [`logging_mode`](shared_memory.md#GenericSharedMemory.logging_mode), [`LoggingMode`](../../mosaic_gpu/interpret/params.md#LoggingMode), [`logging_mode`](../../mosaic_gpu/interpret/params.md#InterpretGPUParams.logging_mode), [`logging_mode`](params.md#SharedInterpretParams.logging_mode)

### `SharedInterpretParams`
- def: [`jax/_src/pallas/mosaic/interpret/params.py:39`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L39)
- doc: Parameters for kernel interpret mode.
- signature: `class SharedInterpretParams:`
- members:
  - `get_vector_clock_size(self, num_devices)` — [`L105`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L105) — Returns the number of vector clocks to use for TPU interpret mode.`
  - `detect_races` — [`L84`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L84)
  - `logging_mode` — [`L90`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L90)
  - `num_cores_or_threads` — [`L88`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L88)
  - `out_of_bounds_reads` — [`L85`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L85)
  - `skip_floating_point_ops` — [`L86`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L86)
  - `uninitialized_memory` — [`L87`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L87)
  - `vector_clock_size` — [`L89`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L89)
- protocol/private: `__post_init__`[`L92`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L92)
- uses (calls/refs, reference-scoped): [`InterpretGPUParams`](../../mosaic_gpu/interpret/params.md#InterpretGPUParams), [`InterpretParams`](params.md#InterpretParams), [`LoggingMode`](params.md#LoggingMode)
- used by: [`_interpret_jaxpr`](interpret_pallas_call.md#_interpret_jaxpr), [`interpret_pallas_call`](interpret_pallas_call.md#interpret_pallas_call), [`interpret`](../../mosaic_gpu/interpret/jaxpr_interpret.md#JaxprInterpreter.interpret), [`_allocate_for_aval`](../../mosaic_gpu/interpret/jaxpr_interpret.md#JaxprInterpreter._allocate_for_aval), [`_allocate_buffers_for_outputs`](../../mosaic_gpu/interpret/interpret_pallas_call.md#_allocate_buffers_for_outputs), [`_get_kernel_buffers`](../../mosaic_gpu/interpret/interpret_pallas_call.md#_get_kernel_buffers), [`_initialize_shared_memory`](../../mosaic_gpu/interpret/gpu_callbacks.md#_initialize_shared_memory), [`InterpretGPUParams`](../../mosaic_gpu/interpret/params.md#InterpretGPUParams), [`InterpretParams`](params.md#InterpretParams), [`_interpret_arithmetic_primitive`](../../mosaic_gpu/interpret/jaxpr_interpret.md#JaxprInterpreter._interpret_arithmetic_primitive), [`__post_init__`](../../mosaic_gpu/interpret/params.md#InterpretGPUParams.__post_init__), [`num_cores_per_device`](params.md#InterpretParams.num_cores_per_device)

## Functions
- `get_interpret_effects()` — [`L191`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/params.py#L191)

