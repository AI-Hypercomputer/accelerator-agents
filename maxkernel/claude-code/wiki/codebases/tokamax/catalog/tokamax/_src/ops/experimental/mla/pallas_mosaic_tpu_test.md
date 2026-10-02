---
title: 'Module: tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.experimental.mla.pallas_mosaic_tpu_test`/
symbols:
  MultiHeadLatentAttentionTest.test_mla_benchmark_correctness: MultiHeadLatentAttentionTest#test_mla_benchmark_correctness().
  PallasTpuMhla: PallasTpuMhla.
  ReferenceMhla: ReferenceMhla.
  default_batched_decode_benchmark_params: default_batched_decode_benchmark_params().
  _generate_mla_params: _generate_mla_params().
  MultiHeadLatentAttentionTest: MultiHeadLatentAttentionTest#
---
# Module: [`tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_test.py`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_test.py)

## Classes
### `MultiHeadLatentAttentionTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_test.py:61`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_test.py#L61)
- signature: `class MultiHeadLatentAttentionTest(parameterized.TestCase):`
- members:
  - `test_mla_benchmark_correctness(self)` — [`L62`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_test.py#L62)
- uses (calls/refs, reference-scoped): [`generate_mla_inputs`](utils.md#generate_mla_inputs), [`cdiv`](utils.md#cdiv), [`PallasTpuMhla`](pallas_mosaic_tpu_test.md#PallasTpuMhla), [`ReferenceMhla`](pallas_mosaic_tpu_test.md#ReferenceMhla), [`default_batched_decode_benchmark_params`](pallas_mosaic_tpu_test.md#default_batched_decode_benchmark_params)

## Functions
- `_generate_mla_params()` — [`L53`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_test.py#L53)
- `default_batched_decode_benchmark_params()` — [`L38`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_test.py#L38)

## Module values
- `PallasTpuMhla` — [`L34`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_test.py#L34)
- `ReferenceMhla` — [`L35`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_test.py#L35)

