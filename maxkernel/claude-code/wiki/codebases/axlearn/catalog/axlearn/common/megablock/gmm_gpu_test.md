---
title: 'Module: axlearn/common/megablock/gmm_gpu_test.py'
type: catalog
provenance: extracted
module: axlearn/common/megablock/gmm_gpu_test.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.megablock.gmm_gpu_test`/
symbols:
  GmmTest.test_gmm_lib: GmmTest#test_gmm_lib().
  GmmTest.test_gmm_benchmark_fn: GmmTest#test_gmm_benchmark_fn().
  tolerances_map: tolerances_map().
  generate_random_data: generate_random_data().
  GmmTest.reference_fn: GmmTest#reference_fn().
  gmm_lax_scan: gmm_lax_scan().
  gmm_lax_scan.compute_group_contribution: gmm_lax_scan().compute_group_contribution().
  generate_group_sizes: generate_group_sizes().
  reference_gmm: reference_gmm().
  matmul_with_preferred_dtype: matmul_with_preferred_dtype().
  generate_perfect_group_sizes: generate_perfect_group_sizes().
  GmmTest: GmmTest#
  GROUPED_MATMUL_TESTS: GROUPED_MATMUL_TESTS.
---
# Module: [`axlearn/common/megablock/gmm_gpu_test.py`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu_test.py)

## Classes
### `GmmTest`  ·  implements/extends TestCase
- def: [`axlearn/common/megablock/gmm_gpu_test.py:228`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu_test.py#L228)
- doc: Tests GPU and TPU decoding.
- signature: `class GmmTest(TestCase):`
- members:
  - `reference_fn(lhs, rhs, group_sizes, preferred_element_type)` — [`L273`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu_test.py#L273)
  - `test_gmm_benchmark_fn(self, m: int, k: int, n: int, tm: int, tk: int, tn: int, num_groups: int, out_dtype: jnp.dtype, transpose_rhs: bool, interpret: bool = False)` — [`L318`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu_test.py#L318)
  - `test_gmm_lib(self, m: int, k: int, n: int, tm: int, tk: int, tn: int, num_groups: int, out_dtype: jnp.dtype, transpose_rhs: bool, interpret: bool = False)` — [`L242`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu_test.py#L242)
- uses (calls/refs, reference-scoped): [`tolerances_map`](gmm_gpu_test.md#tolerances_map), [`gmm`](ops.md#gmm), [`generate_random_data`](gmm_gpu_test.md#generate_random_data), [`gmm_lax_scan`](gmm_gpu_test.md#gmm_lax_scan), [`reference_gmm`](gmm_gpu_test.md#reference_gmm), [`GROUPED_MATMUL_TESTS`](gmm_gpu_test.md#GROUPED_MATMUL_TESTS)  (3 test-only)
- used by: (1 test-only callers)

## Functions
- `compute_group_contribution(carry_start_index: int, group_data: Tuple[int, Tensor])` — [`L166`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu_test.py#L166) — Scan body: Computes the contribution of one group to the final output.
- `generate_group_sizes(num_groups: int, maxval: int)` — [`L45`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu_test.py#L45)
- `generate_perfect_group_sizes(num_groups: int, maxval: int, block_size: int)` — [`L62`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu_test.py#L62) — Generate group sizes which perfectly match the tile size, so that one tile only processed
- `generate_random_data(m: int, k: int, n: int, num_groups: int, test_dtype: jnp.dtype, transpose: bool)` — [`L88`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu_test.py#L88)
- `gmm_lax_scan(lhs: Tensor, rhs: Tensor, group_sizes: Tensor, preferred_element_type: jnp.dtype = jnp.float32)` — [`L130`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu_test.py#L130) — Computes Grouped Matrix Multiplication using lax.scan, this function is used for gmm
- `matmul_with_preferred_dtype(a: Tensor, b: Tensor, preferred_element_type: jnp.dtype)` — [`L123`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu_test.py#L123) — Performs matmul, potentially casting inputs for accumulation.
- `reference_gmm(lhs: Tensor, rhs: Tensor, group_sizes: Tensor, preferred_element_type: jnp.dtype = jnp.float32)` — [`L103`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu_test.py#L103)
- `tolerances_map(dtype: jnp.dtype)` — [`L24`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu_test.py#L24)

## Module values
- `GROUPED_MATMUL_TESTS` — [`L217`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu_test.py#L217)

