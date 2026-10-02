---
title: 'Module: jax/_src/pallas/mosaic_gpu/torch.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic_gpu/torch.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic_gpu.torch`/
symbols:
  _find_mgpu_call: _find_mgpu_call().
  _compile_fn: _compile_fn().
  as_torch_kernel.wrapper: as_torch_kernel().wrapper().
  _compile_fn.apply: _compile_fn().apply().
  _compile_fn.as_torch_dtype: _compile_fn().as_torch_dtype().
  as_torch_kernel: as_torch_kernel().
  _find_mgpu_call_in_module: _find_mgpu_call_in_module().
  _find_mgpu_call.allocate_torch_buffer: _find_mgpu_call().allocate_torch_buffer().
  _mlir_to_torch_dtype: _mlir_to_torch_dtype().
  _is_custom_call: _is_custom_call().
  _find_mgpu_call.run_broadcast: _find_mgpu_call().run_broadcast().
  _find_mgpu_call.prepare_args: _find_mgpu_call().prepare_args().
  _find_mgpu_call.prepare_outputs: _find_mgpu_call().prepare_outputs().
---
# Module: [`jax/_src/pallas/mosaic_gpu/torch.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/torch.py)

## Functions
- `_compile_fn(fn, in_structs)` — [`L226`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/torch.py#L226)
- `_find_mgpu_call(block: ir.Block, args: list[ir.Value])` — [`L107`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/torch.py#L107)
- `_find_mgpu_call_in_module(module: ir.Module)` — [`L77`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/torch.py#L77)
- `_is_custom_call(op: ir.Operation, name: str)` — [`L221`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/torch.py#L221)
- `_mlir_to_torch_dtype(torch, mlir_dtype: ir.Type)` — [`L91`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/torch.py#L91)
- `allocate_torch_buffer(env, device, _shape=result_type.shape, _dtype=_mlir_to_torch_dtype(torch, result_type.element_type), _result_name=value_names[op.result])` — [`L119`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/torch.py#L119)
- `apply(*user_args)` — [`L264`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/torch.py#L264)
- `as_torch_dtype(dtype)` — [`L260`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/torch.py#L260)
- `as_torch_kernel(fn)` — [`L38`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/torch.py#L38) — Makes a Mosaic GPU kernel callable with PyTorch tensors.
- `prepare_args(*user_args, device)` — [`L187`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/torch.py#L187)
- `prepare_outputs(*all_args, device)` — [`L209`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/torch.py#L209)
- `run_broadcast(env, device, _target_shape=target_shape, _dtype=dtype, _operand_name=operand_name, _result_name=result_name)` — [`L164`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/torch.py#L164)
- `wrapper(*args)` — [`L63`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/torch.py#L63)

