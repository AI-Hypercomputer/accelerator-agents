---
title: 'Module: tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.experimental.mla.pallas_mosaic_tpu_kernel_test`/MlaKernelTest#
symbols:
  MlaKernelTest.test_mla_output_shapes: test_mla_output_shapes().
  MlaKernelTest.test_mla_correctness: test_mla_correctness().
  MlaKernelTest: ''
---
# Module: [`tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel_test.py`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel_test.py)

## Classes
### `MlaKernelTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel_test.py:43`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel_test.py#L43)
- signature: `class MlaKernelTest(parameterized.TestCase):`
- members:
  - `test_mla_correctness(self, data)` — [`L125`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel_test.py#L125)
  - `test_mla_output_shapes(self, data)` — [`L46`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel_test.py#L46)
- uses (calls/refs, reference-scoped): [`mla_ragged_paged_attention`](pallas_mosaic_tpu_kernel.md#mla_ragged_paged_attention), [`generate_mla_inputs`](utils.md#generate_mla_inputs), [`mla_attention`](reference.md#mla_attention), [`cdiv`](utils.md#cdiv), [`get_dtype_packing`](utils.md#get_dtype_packing), [`align_to`](utils.md#align_to)

