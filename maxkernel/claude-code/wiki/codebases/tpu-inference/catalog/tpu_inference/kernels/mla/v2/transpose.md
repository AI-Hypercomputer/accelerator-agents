---
title: 'Module: tpu_inference/kernels/mla/v2/transpose.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/mla/v2/transpose.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.mla.v2.transpose`/
symbols:
  xpose_pipeline: xpose_pipeline().
  logger: logger.
  identity_fn_generator: identity_fn_generator().
  xpose_full: xpose_full().
  get_reshape_dimension: get_reshape_dimension().
  pin_vmem_custom_call: pin_vmem_custom_call().
  prev_closest_valid_divisor: prev_closest_valid_divisor().
  xpose_pipeline.get_grid_index: xpose_pipeline().get_grid_index().
  xpose_full.xpose_kernel: xpose_full().xpose_kernel().
  get_reshape_dimension._reshape: get_reshape_dimension()._reshape().
  identity_fn_generator.identity: identity_fn_generator().identity().
  xpose_pipeline.xpose_kernel: xpose_pipeline().xpose_kernel().
---
# Module: [`tpu_inference/kernels/mla/v2/transpose.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/transpose.py)

## Functions
- `_reshape(inp)` — [`L93`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/transpose.py#L93)
- `get_grid_index(i: int, j: int, input_grid: bool)` — [`L215`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/transpose.py#L215)
- `get_reshape_dimension(shape, reshape_axes, dtype=jnp.float32)` — [`L90`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/transpose.py#L90)
- `identity(*arg)` — [`L102`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/transpose.py#L102)
- `identity_fn_generator(num_scalars: int = 0)` — [`L99`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/transpose.py#L99) — Method to copy input content into outputs.
- `pin_vmem_custom_call(input_tensor: jax.Array, num_scalars: int = 0)` — [`L121`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/transpose.py#L121) — Prefetches buffers to VMEM.
- `prev_closest_valid_divisor(number: int, divider: int, multiple_of: int = 1)` — [`L55`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/transpose.py#L55) — Finds the largest divisor of 'number' that is <= 'divider' and divisible
- `xpose_full(input, *, transpose_axes)` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/transpose.py#L33)
- `xpose_kernel(input_ref, output_ref)` — [`L35`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/transpose.py#L35)
- `xpose_kernel(input_ref, output_ref)` — [`L160`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/transpose.py#L160)
- `xpose_pipeline(input: jax.Array, *, transpose_axes: Sequence[int], n_tile: int = 128, m_tile: int = 128, parallel_axis: int = 0, pipeline_axis: int = 1)` — [`L141`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/transpose.py#L141) — Double buffer transpose custom call implementation.

## Module values
- `logger` — [`L27`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/transpose.py#L27)

