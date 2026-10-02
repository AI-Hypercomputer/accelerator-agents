---
title: 'Module: jax/_src/pallas/mosaic_gpu/interpret/params.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic_gpu/interpret/params.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic_gpu.interpret.params`/
symbols:
  InterpretGPUParams: InterpretGPUParams#
  InterpretGPUParams.__post_init__: InterpretGPUParams#__post_init__().
  LoggingMode: LoggingMode.
  force_gpu_interpret_mode: force_gpu_interpret_mode().
  set_gpu_interpret_mode: set_gpu_interpret_mode().
  InterpretGPUParams.logging_mode: InterpretGPUParams#logging_mode.
  get_interpret_effects: get_interpret_effects().
  InterpretGPUParams.num_tma_threads_per_device: InterpretGPUParams#num_tma_threads_per_device.
---
# Module: [`jax/_src/pallas/mosaic_gpu/interpret/params.py`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/params.py)

## Classes
### `InterpretGPUParams`  ·  implements/extends SharedInterpretParams
- def: [`jax/_src/pallas/mosaic_gpu/interpret/params.py:27`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/params.py#L27)
- doc: Parameters for GPU interpret mode.
- signature: `class InterpretGPUParams(interpret_params.SharedInterpretParams):`
- members:
  - `logging_mode` — [`L52`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/params.py#L52)
  - `num_tma_threads_per_device` — [`L51`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/params.py#L51)
- protocol/private: `__post_init__`[`L54`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/params.py#L54)
- uses (calls/refs, reference-scoped): [`vector_clock_size`](../../mosaic/interpret/params.md#SharedInterpretParams.vector_clock_size), [`LoggingMode`](../../mosaic/interpret/params.md#LoggingMode), [`SharedInterpretParams`](../../mosaic/interpret/params.md#SharedInterpretParams), [`__post_init__`](../../mosaic/interpret/params.md#SharedInterpretParams.__post_init__)
- used by: [`interpret_pallas_call`](interpret_pallas_call.md#interpret_pallas_call), [`_allocate_buffers_for_outputs`](interpret_pallas_call.md#_allocate_buffers_for_outputs), [`_get_kernel_buffers`](interpret_pallas_call.md#_get_kernel_buffers), [`_initialize_shared_memory`](gpu_callbacks.md#_initialize_shared_memory), [`_pallas_call_lowering`](../../pallas_call.md#_pallas_call_lowering), [`call_initialize_shared_memory`](gpu_callbacks.md#call_initialize_shared_memory), [`get_interpret_effects`](../../core.md#get_interpret_effects), [`SharedInterpretParams`](../../mosaic/interpret/params.md#SharedInterpretParams), [`interpret_params`](jaxpr_interpret.md#JaxprInterpreter.interpret_params), [`force_gpu_interpret_mode`](params.md#force_gpu_interpret_mode), [`set_gpu_interpret_mode`](params.md#set_gpu_interpret_mode)

## Functions
- `force_gpu_interpret_mode(params: InterpretGPUParams = InterpretGPUParams())` — [`L73`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/params.py#L73) — Context manager that forces GPU interpret mode under its dynamic context.
- `get_interpret_effects()` — [`L68`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/params.py#L68)
- `set_gpu_interpret_mode(params: InterpretGPUParams = InterpretGPUParams())` — [`L95`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/params.py#L95)

## Module values
- `LoggingMode` — [`L23`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/interpret/params.py#L23)

