---
title: 'Module: tokamax/_src/ops/attention/pallas_mosaic_tpu_test.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/pallas_mosaic_tpu_test.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.pallas_mosaic_tpu_test`/PallasMosaicTpuFlashAttentionTest#
symbols:
  PallasMosaicTpuFlashAttentionTest._test_attention: _test_attention().
  PallasMosaicTpuFlashAttentionTest.test_autotune_vjp: test_autotune_vjp().
  PallasMosaicTpuFlashAttentionTest.test_autotune_configs: test_autotune_configs().
  PallasMosaicTpuFlashAttentionTest.f_base: f_base().
  PallasMosaicTpuFlashAttentionTest.test_simple: test_simple().
  PallasMosaicTpuFlashAttentionTest.f: f().
  PallasMosaicTpuFlashAttentionTest: ''
  PallasMosaicTpuFlashAttentionTest.setUp: setUp().
---
# Module: [`tokamax/_src/ops/attention/pallas_mosaic_tpu_test.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_test.py)

## Classes
### `PallasMosaicTpuFlashAttentionTest`  ·  implements/extends TestCase
- def: [`tokamax/_src/ops/attention/pallas_mosaic_tpu_test.py:30`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_test.py#L30)
- signature: `class PallasMosaicTpuFlashAttentionTest(parameterized.TestCase):`
- members:
  - `f(query, key, value, do)` — [`L60`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_test.py#L60)
  - `f_base(query, key, value, do)` — [`L50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_test.py#L50)
  - `setUp(self)` — [`L32`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_test.py#L32)
  - `test_autotune_configs(self)` — [`L123`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_test.py#L123)
  - `test_autotune_vjp(self)` — [`L160`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_test.py#L160)
  - `test_simple(self, dtype, num_kv_heads, masking, logits_soft_cap)` — [`L94`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_test.py#L94)
- protocol/private: `_test_attention`[`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_test.py#L37)
- uses (calls/refs, reference-scoped): [`DotProductAttention`](base.md#DotProductAttention), [`Mask`](base.md#Mask), [`bind`](../op.md#Op.bind), [`_get_autotuning_configs`](pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention._get_autotuning_configs), [`vjp`](../op.md#Op.vjp), [`bind`](base.md#DotProductAttention.bind), [`config`](../op.md#Op.config), [`_get_autotuning_configs`](pallas_mosaic_tpu_vjp.md#PallasMosaicTpuFlashAttentionVjp._get_autotuning_configs), [`PallasMosaicTpuFlashAttention`](pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention), [`PallasMosaicTpuFlashAttentionVjp`](pallas_mosaic_tpu_vjp.md#PallasMosaicTpuFlashAttentionVjp)

