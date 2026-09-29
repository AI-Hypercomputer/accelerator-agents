---
title: 'Module: tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.gmm_v2_kernel_tests.pallas_mosaic_tpu_v2_kernel_test`/
symbols:
  GmmTest.test_tgmm_with_tile_info: GmmTest#test_tgmm_with_tile_info().
  GmmTest.test_tgmm_with_rhs_scale_n_padding: GmmTest#test_tgmm_with_rhs_scale_n_padding().
  GmmTest.test_gmm_weight_quantized_block_larger_than_tile_k: GmmTest#test_gmm_weight_quantized_block_larger_than_tile_k().
  GmmTest.test_gmm_activation_weight_quantized_block_larger_than_tile_k: GmmTest#test_gmm_activation_weight_quantized_block_larger_than_tile_k().
  GmmTest.test_tgmm_explicitly_exercises_all_branches: GmmTest#test_tgmm_explicitly_exercises_all_branches().
  get_group_sizes: get_group_sizes().
  GmmTest.test_tgmm_basic: GmmTest#test_tgmm_basic().
  GmmTest.test_tgmm_implicit_padding: GmmTest#test_tgmm_implicit_padding().
  GmmTest.test_tgmm_empty_group: GmmTest#test_tgmm_empty_group().
  GmmTest.test_tgmm_with_rhs_scale: GmmTest#test_tgmm_with_rhs_scale().
  GmmTest.test_gmm_nonlocal_groups_produce_zeros: GmmTest#test_gmm_nonlocal_groups_produce_zeros().
  GmmTest.test_gmm_fused_activation: GmmTest#test_gmm_fused_activation().
  assert_arrays_all_close: assert_arrays_all_close().
  GmmTest.test_gmm_basic: GmmTest#test_gmm_basic().
  GmmTest.test_gmm_weight_quantized: GmmTest#test_gmm_weight_quantized().
  GmmTest.test_gmm_activation_weight_quantized: GmmTest#test_gmm_activation_weight_quantized().
  GmmTest.test_gmm_implicit_padding: GmmTest#test_gmm_implicit_padding().
  GmmTest.test_gmm_weight_quantized_padding: GmmTest#test_gmm_weight_quantized_padding().
  quantize_tensor: quantize_tensor().
  reference_gmm: reference_gmm().
  reference_tgmm: reference_tgmm().
  _lookup_tol: _lookup_tol().
  GmmTest.test_gmm_uninitialized_memory_robustness: GmmTest#test_gmm_uninitialized_memory_robustness().
  poison_tpu_memory: poison_tpu_memory().
  _GroupConfig: _GroupConfig.
  GmmTest.test_gmm_security_isolation: GmmTest#test_gmm_security_isolation().
  _DTYPE_TOL: _DTYPE_TOL.
  poison_tpu_memory.poison_kernel: poison_tpu_memory().poison_kernel().
  GmmTest: GmmTest#
  GmmTest.setUp: GmmTest#setUp().
---
# Module: [`tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py)

## Classes
### `GmmTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py:239`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L239)
- signature: `class GmmTest(parameterized.TestCase):`
- members:
  - `setUp(self)` — [`L241`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L241)
  - `test_gmm_activation_weight_quantized(self, batch_size, in_size, out_size, num_groups, weight_dtype, block_size, group_offset)` — [`L862`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L862)
  - `test_gmm_activation_weight_quantized_block_larger_than_tile_k(self, batch_size, in_size, out_size, num_groups, weight_dtype, block_size, tile_k, group_offset)` — [`L803`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L803) — Test activation+weight quantized path with quant_block_size > tile_k.
  - `test_gmm_basic(self, batch_size, in_size, out_size, num_groups, has_bias, group_offset)` — [`L254`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L254)
  - `test_gmm_fused_activation(self, batch_size, in_size, out_size, num_groups, has_bias, use_weight_scale, maybe_quantize_lhs, fuse_act, group_offset, block_size)` — [`L1084`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L1084)
  - `test_gmm_implicit_padding(self, batch_size, in_size, out_size, num_groups, has_bias, group_offset)` — [`L917`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L917)
  - `test_gmm_nonlocal_groups_produce_zeros(self, batch_size, in_size, out_size, group_config)` — [`L1036`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L1036)
  - `test_gmm_security_isolation(self)` — [`L664`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L664) — Verifies that sequences (experts) are isolated from each other.
  - `test_gmm_uninitialized_memory_robustness(self)` — [`L704`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L704) — Verifies that the kernel is robust against uninitialized scratchpads.
  - `test_gmm_weight_quantized(self, batch_size, in_size, out_size, num_groups, has_bias, weight_dtype, block_size, group_offset)` — [`L609`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L609)
  - `test_gmm_weight_quantized_block_larger_than_tile_k(self, batch_size, in_size, out_size, num_groups, weight_dtype, block_size, tile_k, group_offset)` — [`L741`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L741) — Test that quant_block_size > tile_k is handled correctly.
  - `test_gmm_weight_quantized_padding(self, batch_size, in_size, out_size, num_groups, has_bias, weight_dtype, block_size, group_offset)` — [`L965`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L965)
  - `test_tgmm_basic(self, batch_size, in_size, out_size, num_groups, group_offset)` — [`L295`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L295)
  - `test_tgmm_empty_group(self, batch_size, in_size, out_size, num_groups, group_offset, empty_group_index)` — [`L433`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L433) — Test that TGMM correctly zeros output for empty groups.
  - `test_tgmm_explicitly_exercises_all_branches(self)` — [`L473`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L473)
  - `test_tgmm_implicit_padding(self, batch_size, in_size, out_size, num_groups, group_offset)` — [`L340`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L340)
  - `test_tgmm_with_rhs_scale(self, batch_size, in_size, out_size, num_groups, group_offset, dtype_pair)` — [`L521`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L521)
  - `test_tgmm_with_rhs_scale_n_padding(self)` — [`L559`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L559)
  - `test_tgmm_with_tile_info(self, batch_size, in_size, out_size, num_groups, group_offset, tile_k, tile_n)` — [`L387`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L387)
- uses (calls/refs, reference-scoped): [`gmm_v2`](../pallas_mosaic_tpu_v2_gmm_kernel.md#gmm_v2), [`tgmm_v2`](../pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_v2), [`tile_k`](../pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes.tile_k), [`tile_n`](../pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes.tile_n), [`TileSizes`](../pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes), [`tile_m`](../pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes.tile_m), [`get_group_sizes`](pallas_mosaic_tpu_v2_kernel_test.md#get_group_sizes), [`assert_arrays_all_close`](pallas_mosaic_tpu_v2_kernel_test.md#assert_arrays_all_close), [`quantize_tensor`](pallas_mosaic_tpu_v2_kernel_test.md#quantize_tensor), [`reference_gmm`](pallas_mosaic_tpu_v2_kernel_test.md#reference_gmm), [`reference_tgmm`](pallas_mosaic_tpu_v2_kernel_test.md#reference_tgmm), [`validate_tgmm_inputs`](../pallas_mosaic_tpu_v2_tgmm_kernel.md#validate_tgmm_inputs), [`apply_act_fn`](../pallas_mosaic_tpu_v2_gmm_kernel.md#apply_act_fn), [`poison_tpu_memory`](pallas_mosaic_tpu_v2_kernel_test.md#poison_tpu_memory), [`_GroupConfig`](pallas_mosaic_tpu_v2_kernel_test.md#_GroupConfig)

## Functions
- `_lookup_tol(dtype)` — [`L221`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L221)
- `assert_arrays_all_close(actual, desired, *, atol=None, rtol=None)` — [`L231`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L231)
- `get_group_sizes(batch_size: int, num_groups: int)` — [`L62`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L62)
- `poison_kernel(in_ref, out_ref, v_scratch, s_scratch)` — [`L39`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L39)
- `poison_tpu_memory()` — [`L32`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L32) — Fills TPU scratchpad memory with NaNs to simulate garbage state.
- `quantize_tensor(x: jax.Array, dtype: jnp.dtype, axis: int = -1, block_size: int = 256)` — [`L71`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L71)
- `reference_gmm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, rhs_scale: jax.Array | None = None, rhs_bias: jax.Array | None = None, group_offset: jax.Array | None = None)` — [`L97`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L97)
- `reference_tgmm(lhs, rhs, group_sizes, num_actual_groups, rhs_scale: jax.Array | None = None, group_offset=None, out_dtype: jnp.dtype | None = None)` — [`L162`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L162)

## Module values
- `_DTYPE_TOL` — [`L216`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L216)
- `_GroupConfig` — [`L57`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.py#L57)

