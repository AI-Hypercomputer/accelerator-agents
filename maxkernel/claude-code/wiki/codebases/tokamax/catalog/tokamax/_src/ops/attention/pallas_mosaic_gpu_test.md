---
title: 'Module: tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.pallas_mosaic_gpu_test`/PallasMosaicGpuFlashAttentionTest#
symbols:
  PallasMosaicGpuFlashAttentionTest.test_autotune_configs: test_autotune_configs().
  PallasMosaicGpuFlashAttentionTest.__init__: __init__().
  PallasMosaicGpuFlashAttentionTest.test_vjp_autotune_configs: test_vjp_autotune_configs().
  PallasMosaicGpuFlashAttentionTest._run_test_with_inputs: _run_test_with_inputs().
  PallasMosaicGpuFlashAttentionTest.test_split_k: test_split_k().
  PallasMosaicGpuFlashAttentionTest.test_normalize_output: test_normalize_output().
  PallasMosaicGpuFlashAttentionTest._test_op_parameters: _test_op_parameters().
  PallasMosaicGpuFlashAttentionTest.test_causal_mask: test_causal_mask().
  PallasMosaicGpuFlashAttentionTest._test_bench: _test_bench().
  PallasMosaicGpuFlashAttentionTest._test_small_sequences: _test_small_sequences().
  PallasMosaicGpuFlashAttentionTest: ''
  PallasMosaicGpuFlashAttentionTest.test_causal_mask_cross_attention0: test_causal_mask_cross_attention0().
  PallasMosaicGpuFlashAttentionTest.test_op_parameters: test_op_parameters().
  PallasMosaicGpuFlashAttentionTest.setUp: setUp().
  PallasMosaicGpuFlashAttentionTest.test_causal_mask_cross_attention1: test_causal_mask_cross_attention1().
  PallasMosaicGpuFlashAttentionTest.test_padding_mask_with_nans: test_padding_mask_with_nans().
---
# Module: [`tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py)

## Classes
### `PallasMosaicGpuFlashAttentionTest`  ·  implements/extends AttentionTestBase
- def: [`tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py:36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py#L36)
- signature: `class PallasMosaicGpuFlashAttentionTest(test_base.AttentionTestBase):`
- members:
  - `setUp(self)` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py#L38)
  - `test_autotune_configs(self)` — [`L169`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py#L169)
  - `test_causal_mask(self)` — [`L109`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py#L109)
  - `test_causal_mask_cross_attention0(self)` — [`L116`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py#L116)
  - `test_causal_mask_cross_attention1(self)` — [`L122`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py#L122)
  - `test_normalize_output(self)` — [`L128`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py#L128)
  - `test_op_parameters(self, use_stable_softmax, rescale_threshold)` — [`L135`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py#L135)
  - `test_padding_mask_with_nans(self)` — [`L125`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py#L125)
  - `test_split_k(self)` — [`L215`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py#L215)
  - `test_vjp_autotune_configs(self)` — [`L190`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py#L190)
- protocol/private: `__init__`[`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py#L43), `_run_test_with_inputs`[`L77`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py#L77), `_test_bench`[`L157`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py#L157), `_test_op_parameters`[`L138`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py#L138), `_test_small_sequences`[`L228`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_test.py#L228)
- uses (calls/refs, reference-scoped): [`DotProductAttention`](base.md#DotProductAttention), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`vjp`](../op.md#Op.vjp), [`bind`](base.md#DotProductAttention.bind), [`PallasMosaicGpuFlashAttention`](pallas_mosaic_gpu.md#PallasMosaicGpuFlashAttention), [`PallasMosaicGpuFlashAttentionVjp`](pallas_mosaic_gpu_vjp.md#PallasMosaicGpuFlashAttentionVjp), [`dbias_intermediate_dtype`](pallas_mosaic_gpu_vjp.md#PallasMosaicGpuFlashAttentionVjp.dbias_intermediate_dtype), [`_get_kernel_module`](pallas_mosaic_gpu.md#_get_kernel_module)  (12 test-only)
- used by: (7 test-only callers)

