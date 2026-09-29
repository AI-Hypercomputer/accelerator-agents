---
title: 'Module: tests/test_matmul.py'
type: catalog
provenance: extracted
module: tests/test_matmul.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `tests.test_matmul`/TestTiledMatmul#
symbols:
  TestTiledMatmul.test_small_aligned: test_small_aligned().
  TestTiledMatmul.test_medium_aligned: test_medium_aligned().
  TestTiledMatmul.test_unaligned_dimensions: test_unaligned_dimensions().
  TestTiledMatmul.test_rectangular: test_rectangular().
  TestTiledMatmul.test_identity_matrix: test_identity_matrix().
  TestTiledMatmul.test_zero_input: test_zero_input().
  TestTiledMatmul.test_different_block_sizes: test_different_block_sizes().
  TestTiledMatmul.test_invalid_inputs: test_invalid_inputs().
  TestTiledMatmul: ''
---
# Module: [`tests/test_matmul.py`](../../../../../raw/code/pallas-forge/tests/test_matmul.py)

## Classes
### `TestTiledMatmul`
- def: [`tests/test_matmul.py:12`](../../../../../raw/code/pallas-forge/tests/test_matmul.py#L12)
- doc: Correctness tests for tiled_matmul (run on CPU via interpret mode).
- signature: `class TestTiledMatmul:`
- members:
  - `test_different_block_sizes(self, rng_key)` — [`L86`](../../../../../raw/code/pallas-forge/tests/test_matmul.py#L86) — Various block size combinations should all produce correct results.
  - `test_identity_matrix(self, rng_key)` — [`L67`](../../../../../raw/code/pallas-forge/tests/test_matmul.py#L67) — Multiply by identity should return the original.
  - `test_invalid_inputs(self)` — [`L100`](../../../../../raw/code/pallas-forge/tests/test_matmul.py#L100) — Invalid inputs should raise ValueError.
  - `test_medium_aligned(self, rng_key, dtype, tolerance)` — [`L29`](../../../../../raw/code/pallas-forge/tests/test_matmul.py#L29) — Medium matrix with default block sizes.
  - `test_rectangular(self, rng_key)` — [`L54`](../../../../../raw/code/pallas-forge/tests/test_matmul.py#L54) — Tall and wide matrices.
  - `test_small_aligned(self, rng_key, dtype, tolerance)` — [`L15`](../../../../../raw/code/pallas-forge/tests/test_matmul.py#L15) — Small matrix that's perfectly aligned to block boundaries.
  - `test_unaligned_dimensions(self, rng_key)` — [`L41`](../../../../../raw/code/pallas-forge/tests/test_matmul.py#L41) — Non-aligned dimensions trigger padding/unpadding.
  - `test_zero_input(self, rng_key)` — [`L77`](../../../../../raw/code/pallas-forge/tests/test_matmul.py#L77) — Zero input should produce zero output.
- uses (calls/refs, reference-scoped): [`tiled_matmul`](../pallas_forge/kernels/matmul.md#tiled_matmul)

