---
title: 'Module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.linear_softmax_cross_entropy_loss.pallas_mosaic_tpu_kernel_test`/FlashLcePallasMosaicTpuKernelTest#
symbols:
  FlashLcePallasMosaicTpuKernelTest.test_kernel_forward_matches_reference: test_kernel_forward_matches_reference().
  FlashLcePallasMosaicTpuKernelTest.test_kernel_bwd_matches_reference: test_kernel_bwd_matches_reference().
  FlashLcePallasMosaicTpuKernelTest.test_validation_errors: test_validation_errors().
  FlashLcePallasMosaicTpuKernelTest: ''
  FlashLcePallasMosaicTpuKernelTest.setUp: setUp().
---
# Module: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel_test.py)

## Classes
### `FlashLcePallasMosaicTpuKernelTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel_test.py:27`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel_test.py#L27)
- signature: `class FlashLcePallasMosaicTpuKernelTest(parameterized.TestCase):`
- members:
  - `setUp(self)` — [`L29`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel_test.py#L29)
  - `test_kernel_bwd_matches_reference(self, b_dim, h_dim, v_dim, reduction)` — [`L287`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel_test.py#L287)
  - `test_kernel_forward_matches_reference(self, b_dim, h_dim, v_dim, reduction, dtype=jnp.float32)` — [`L150`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel_test.py#L150)
  - `test_validation_errors(self, b_dim, h_dim, v_dim)` — [`L345`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/linear_softmax_cross_entropy_loss/pallas_mosaic_tpu_kernel_test.py#L345)
- uses (calls/refs, reference-scoped): [`get_tpu_specific_default_config`](pallas_mosaic_tpu.md#get_tpu_specific_default_config), [`b_block_size`](pallas_mosaic_tpu.md#Config.b_block_size), [`h_block_size`](pallas_mosaic_tpu.md#Config.h_block_size), [`v_block_size`](pallas_mosaic_tpu.md#Config.v_block_size), [`linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu`](pallas_mosaic_tpu_kernel.md#linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu), [`linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu`](pallas_mosaic_tpu_kernel.md#linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu), [`linear_softmax_cross_entropy_loss_fwd_reference`](reference.md#linear_softmax_cross_entropy_loss_fwd_reference), [`linear_softmax_cross_entropy_loss_bwd_reference`](reference.md#linear_softmax_cross_entropy_loss_bwd_reference)

