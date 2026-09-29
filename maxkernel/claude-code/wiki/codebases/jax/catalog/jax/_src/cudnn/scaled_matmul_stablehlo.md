---
title: 'Module: jax/_src/cudnn/scaled_matmul_stablehlo.py'
type: catalog
provenance: extracted
module: jax/_src/cudnn/scaled_matmul_stablehlo.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.cudnn.scaled_matmul_stablehlo`/
symbols:
  scaled_dot_general_wrapper: scaled_dot_general_wrapper().
  quantize: quantize().
  _scaled_matmul_p: _scaled_matmul_p.
  _scaled_matmul_p_wrapper: _scaled_matmul_p_wrapper.
  scaled_dot_general_transpose_lhs: scaled_dot_general_transpose_lhs().
  scaled_dot_impl: scaled_dot_impl().
  _scaled_matmul_lower: _scaled_matmul_lower.
  scaled_matmul_wrapper: scaled_matmul_wrapper().
  _scaled_matmul_partition: _scaled_matmul_partition().
  _scaled_matmul: _scaled_matmul().
  scaled_dot_bwd: scaled_dot_bwd().
  _get_output_sharding: _get_output_sharding().
  _scaled_matmul_batcher: _scaled_matmul_batcher().
  shape_normalization: shape_normalization().
  _scaled_matmul_gpu_lowering: _scaled_matmul_gpu_lowering().
  _scaled_matmul_infer_sharding_from_operands: _scaled_matmul_infer_sharding_from_operands().
  _supported_in_out_sharding: _supported_in_out_sharding().
  _scaled_matmul_partition._scaled_matmul_impl_partition: _scaled_matmul_partition()._scaled_matmul_impl_partition().
  e8m0_to_dtype: e8m0_to_dtype().
  scaled_dot_general_transpose_rhs: scaled_dot_general_transpose_rhs().
  _scaled_matmul_impl: _scaled_matmul_impl().
  _supported_in_out_sharding.named_sharding: _supported_in_out_sharding().named_sharding().
  BlockScaleConfig: BlockScaleConfig#
  quantize.get_scales_per_block: quantize().get_scales_per_block().
  scaled_dot_general_fn: scaled_dot_general_fn().
  cast_to_e8m0_with_rounding_up: cast_to_e8m0_with_rounding_up().
  BlockScaleConfig.data_type: BlockScaleConfig#data_type.
  BlockScaleConfig.scale_type: BlockScaleConfig#scale_type.
  BlockScaleConfig.global_scale: BlockScaleConfig#global_scale.
  _scaled_matmul_abstract: _scaled_matmul_abstract().
  _scaled_matmul_rocm_lowering: _scaled_matmul_rocm_lowering().
  _scaled_matmul_rocm_lowering._scaled_dot_lowering_impl: _scaled_matmul_rocm_lowering()._scaled_dot_lowering_impl().
  scaled_dot_fwd: scaled_dot_fwd().
  _ensure_batch_dim: _ensure_batch_dim().
  BlockScaleConfig.mode: BlockScaleConfig#mode.
  BlockScaleConfig.block_size: BlockScaleConfig#block_size.
  BlockScaleConfig.infer_only: BlockScaleConfig#infer_only.
  default_layouts: default_layouts().
  _check_shardings: _check_shardings().
  _enable_all_reduce: _enable_all_reduce().
  _are_specs_overlapping: _are_specs_overlapping().
  compute_dot_output_shape: compute_dot_output_shape().
  block_scaled_dot_name: block_scaled_dot_name.
  element_type_to_backend_config_type: element_type_to_backend_config_type().
  _enable_reduce_scatter: _enable_reduce_scatter().
  _get_reduce_scatter_dim: _get_reduce_scatter_dim().
  ensure_tuple: ensure_tuple().
---
# Module: [`jax/_src/cudnn/scaled_matmul_stablehlo.py`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py)

## Classes
### `BlockScaleConfig`
- def: [`jax/_src/cudnn/scaled_matmul_stablehlo.py:44`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L44)
- signature: `class BlockScaleConfig:`
- members:
  - `block_size` — [`L46`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L46)
  - `data_type` — [`L47`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L47)
  - `global_scale` — [`L49`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L49)
  - `infer_only` — [`L50`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L50)
  - `mode` — [`L45`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L45)
  - `scale_type` — [`L48`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L48)
- uses (calls/refs, reference-scoped): [`Array`](../basearray.md#Array), [`DTypeLike`](../typing.md#DTypeLike)
- used by: [`scaled_dot_general_wrapper`](scaled_matmul_stablehlo.md#scaled_dot_general_wrapper), [`get_scaled_dot_general_config`](../nn/functions.md#get_scaled_dot_general_config), [`scaled_dot_general_transpose_rhs`](scaled_matmul_stablehlo.md#scaled_dot_general_transpose_rhs), [`scaled_dot_general`](../nn/functions.md#scaled_dot_general)

## Functions
- `_are_specs_overlapping(lhs, rhs)` — [`L210`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L210)
- `_check_shardings(shardings)` — [`L179`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L179)
- `_enable_all_reduce(lhs, rhs)` — [`L205`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L205)
- `_enable_reduce_scatter(lhs, rhs)` — [`L194`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L194)
- `_ensure_batch_dim(lhs, rhs, dimension_numbers)` — [`L748`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L748)
- `_get_output_sharding(shardings)` — [`L217`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L217)
- `_get_reduce_scatter_dim(lhs, rhs, output)` — [`L244`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L244)
- `_scaled_dot_lowering_impl(lhs, rhs, lhs_scales, rhs_scales)` — [`L113`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L113)
- `_scaled_matmul(lhs: Array, rhs: Array, lhs_scales: Array, rhs_scales: Array, preferred_element_type: DTypeLike = np.dtype('float32'))` — [`L390`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L390)
- `_scaled_matmul_abstract(a, b, a_scale, b_scale, *, preferred_element_type)` — [`L129`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L129)
- `_scaled_matmul_batcher(batched_args, batch_dims, *, preferred_element_type)` — [`L341`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L341)
- `_scaled_matmul_gpu_lowering(ctx, a, b, a_scales, b_scales, preferred_element_type)` — [`L70`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L70)
- `_scaled_matmul_impl(a, b, a_scale, b_scale, preferred_element_type)` — [`L64`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L64)
- `_scaled_matmul_impl_partition(a, b, a_scale, b_scale)` — [`L316`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L316)
- `_scaled_matmul_infer_sharding_from_operands(preferred_element_type, mesh, shapes, output_shape)` — [`L232`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L232)
- `_scaled_matmul_partition(preferred_element_type, mesh, shapes, output_shape)` — [`L304`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L304)
- `_scaled_matmul_rocm_lowering(ctx, a, b, a_scales, b_scales, preferred_element_type)` — [`L108`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L108)
- `_supported_in_out_sharding(lhs_sharding, rhs_sharding, out_sharding, reduce_scatter_dim)` — [`L259`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L259)
- `cast_to_e8m0_with_rounding_up(x)` — [`L527`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L527)
- `compute_dot_output_shape(lhs_shape, rhs_shape, lhs_dimension_numbers, rhs_dimension_numbers)` — [`L501`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L501) — Computes the output shape for a `lax.dot_general`-like operation.
- `default_layouts(*shapes)` — [`L52`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L52)
- `e8m0_to_dtype(x, dtype)` — [`L540`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L540)
- `element_type_to_backend_config_type(dtype)` — [`L55`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L55)
- `ensure_tuple(dimension_numbers)` — [`L737`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L737)
- `get_scales_per_block(values)` — [`L559`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L559)
- `named_sharding(lhs_specs, rhs_specs, out_specs)` — [`L268`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L268)
- `quantize(x, config)` — [`L550`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L550)
- `scaled_dot_bwd(dimension_numbers, preferred_element_type, configs, res, g)` — [`L702`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L702)
- `scaled_dot_fwd(lhs, rhs, dimension_numbers, preferred_element_type, configs)` — [`L694`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L694)
- `scaled_dot_general_fn(lhs, rhs, dimension_numbers, preferred_element_type, configs)` — [`L688`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L688)
- `scaled_dot_general_transpose_lhs(g, x, y, *, dimension_numbers, preferred_element_type, configs, swap_ans=False)` — [`L630`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L630)
- `scaled_dot_general_transpose_rhs(g, x, y, *, dimension_numbers, preferred_element_type: DTypeLike, configs: list[BlockScaleConfig])` — [`L669`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L669)
- `scaled_dot_general_wrapper(lhs, rhs, dimension_numbers, preferred_element_type=np.float32, configs: list[BlockScaleConfig] | None = None)` — [`L763`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L763)
- `scaled_dot_impl(lhs, rhs, dimension_numbers, preferred_element_type, configs)` — [`L591`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L591)
- `scaled_matmul_wrapper(lhs: Array, rhs: Array, lhs_scales: Array, rhs_scales: Array, preferred_element_type: DTypeLike = np.dtype('float32'))` — [`L403`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L403) — Performs scaled matrix multiplication between two 3D arrays, with scaling
- `shape_normalization(x, dimension_numbers)` — [`L455`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L455) — Normalizes the shape of the input tensor `x` to `(B, M, K)`.

## Module values
- `_scaled_matmul_lower` — [`L330`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L330)
- `_scaled_matmul_p` — [`L136`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L136)
- `_scaled_matmul_p_wrapper` — [`L158`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L158)
- `block_scaled_dot_name` — [`L41`](../../../../../../../raw/code/jax/jax/_src/cudnn/scaled_matmul_stablehlo.py#L41)

