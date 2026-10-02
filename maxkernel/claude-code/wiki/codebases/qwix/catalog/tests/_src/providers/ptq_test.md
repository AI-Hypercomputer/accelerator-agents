---
title: 'Module: tests/_src/providers/ptq_test.py'
type: catalog
provenance: extracted
module: tests/_src/providers/ptq_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.providers.ptq_test`/
symbols:
  PtqTest.test_nnx_srq: PtqTest#test_nnx_srq().
  PtqTest.test_nn_srq: PtqTest#test_nn_srq().
  PtqTest.test_nnx_einsum_sharding_ptq: PtqTest#test_nnx_einsum_sharding_ptq().
  PtqTest.test_dot_pallas_call: PtqTest#test_dot_pallas_call().
  PtqTest.test_nnx_scan: PtqTest#test_nnx_scan().
  PtqTest.test_nn_ptq: PtqTest#test_nn_ptq().
  PtqTest.test_nnx_ptq: PtqTest#test_nnx_ptq().
  PtqTest.test_reshape_weight: PtqTest#test_reshape_weight().
  PtqTest.test_asarray_interception_with_aux: PtqTest#test_asarray_interception_with_aux().
  PtqTest.test_symbolic_export: PtqTest#test_symbolic_export().
  PtqTest.test_allow_extra_params: PtqTest#test_allow_extra_params().
  PtqTest._get_quantized_asarray_model: PtqTest#_get_quantized_asarray_model().
  PtqTest.test_asarray_interception_linen_module: PtqTest#test_asarray_interception_linen_module().
  PtqTest.test_nnx_multi_head_attention: PtqTest#test_nnx_multi_head_attention().
  PtqTest.test_asarray_with_aux_qarray: PtqTest#test_asarray_with_aux_qarray().
  PtqTest.test_asarray_nnx_param_with_aux: PtqTest#test_asarray_nnx_param_with_aux().
  PtqTest.test_dot_pallas_call.Model.__call__: PtqTest#test_dot_pallas_call().Model#__call__().
  PtqTest.test_nnx_scan.ScanModel.__call__: PtqTest#test_nnx_scan().ScanModel#__call__().
  PtqTest.test_asarray_interception_qarray: PtqTest#test_asarray_interception_qarray().
  PtqTest.test_asarray_interception_nnx_state: PtqTest#test_asarray_interception_nnx_state().
  ScanModel.layers: ScanModel#layers.
  PtqTest.test_reshape_weight.ReshapeDense.out_features: PtqTest#test_reshape_weight().ReshapeDense#out_features.
  PtqTest.init_fn: PtqTest#init_fn().
  PtqTest.test_reshape_weight.ReshapeDense.__call__: PtqTest#test_reshape_weight().ReshapeDense#__call__().
  PtqTest.test_allow_extra_params.TestModel.__call__: PtqTest#test_allow_extra_params().TestModel#__call__().
  PtqTest.test_allow_extra_params.TestModel.compute_loss: PtqTest#test_allow_extra_params().TestModel#compute_loss().
  PtqTest.get_canonical_pspec: PtqTest#get_canonical_pspec().
  PtqTest.test_dot_pallas_call.Model: PtqTest#test_dot_pallas_call().Model#
  PtqTest.test_dot_pallas_call.Model.pallas_dot: PtqTest#test_dot_pallas_call().Model#pallas_dot().
  PtqTest.test_nnx_scan.ScanModel: PtqTest#test_nnx_scan().ScanModel#
  PtqTest.test_nnx_scan.ScanModel.create_layer: PtqTest#test_nnx_scan().ScanModel#create_layer().
  PtqTest.test_nnx_scan.ScanModel.scan_fn: PtqTest#test_nnx_scan().ScanModel#scan_fn().
  PtqTest.test_reshape_weight.ReshapeDense: PtqTest#test_reshape_weight().ReshapeDense#
  PtqTest.test_allow_extra_params.TestModel: PtqTest#test_allow_extra_params().TestModel#
  TestModel.dense_layer: TestModel#dense_layer.
  TestModel.ssl_loss_layer: TestModel#ssl_loss_layer.
  PtqTest._get_quantized_asarray_model.AsArrayModule: PtqTest#_get_quantized_asarray_model().AsArrayModule#
  PtqTest.test_asarray_interception_linen_module.LinenAsArrayModule: PtqTest#test_asarray_interception_linen_module().LinenAsArrayModule#
  PtqTest: PtqTest#
  PtqTest.test_nnx_scan.ScanModel.__init__: PtqTest#test_nnx_scan().ScanModel#__init__().
  PtqTest.test_allow_extra_params.TestModel.setup: PtqTest#test_allow_extra_params().TestModel#setup().
  PtqTest.test_allow_extra_params.TestModel.predict: PtqTest#test_allow_extra_params().TestModel#predict().
  PtqTest._get_quantized_asarray_model.AsArrayModule.__call__: PtqTest#_get_quantized_asarray_model().AsArrayModule#__call__().
  PtqTest.test_asarray_interception_linen_module.LinenAsArrayModule.__call__: PtqTest#test_asarray_interception_linen_module().LinenAsArrayModule#__call__().
---
# Module: [`tests/_src/providers/ptq_test.py`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py)

## Classes
### `AsArrayModule`  ·  implements/extends Module
- def: [`tests/_src/providers/ptq_test.py:548`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L548)
- signature: `class AsArrayModule(nnx.Module):`
- protocol/private: `__call__`[`L550`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L550)
- used by: (1 test-only callers)

### `LinenAsArrayModule`  ·  implements/extends Module
- def: [`tests/_src/providers/ptq_test.py:636`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L636)
- signature: `class LinenAsArrayModule(nn.Module):`
- protocol/private: `__call__`[`L639`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L639)
- used by: (1 test-only callers)

### `Model`  ·  implements/extends Module
- def: [`tests/_src/providers/ptq_test.py:378`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L378)
- signature: `class Model(nn.Module):`
- members:
  - `pallas_dot(x, y, out)` — [`L388`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L388)
- protocol/private: `__call__`[`L381`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L381)
- uses (calls/refs, reference-scoped): [`get_current_rule`](../../../qwix/_src/qconfig.md#get_current_rule)
- used by: (1 test-only callers)

### `PtqTest`  ·  implements/extends TestCase
- def: [`tests/_src/providers/ptq_test.py:34`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L34)
- signature: `class PtqTest(parameterized.TestCase):`
- members:
  - `get_canonical_pspec(x: jax.Array)` — [`L260`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L260) — The sharding.spec may be shorter than the ndim.
  - `init_fn(prng, inputs)` — [`L527`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L527)
  - `test_allow_extra_params(self)` — [`L499`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L499)
  - `test_asarray_interception_linen_module(self)` — [`L630`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L630) — asarray should gracefully fallback for Linen Modules.
  - `test_asarray_interception_nnx_state(self)` — [`L608`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L608) — asarray should reconstruct QArray from nnx.State and fallback for others.
  - `test_asarray_interception_qarray(self)` — [`L600`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L600) — asarray should not dequantize QArray.
  - `test_asarray_interception_with_aux(self)` — [`L559`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L559) — asarray should not dequantize WithAux.
  - `test_asarray_nnx_param_with_aux(self)` — [`L664`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L664) — asarray should handle nnx.Param wrapping a WithAux.
  - `test_asarray_with_aux_qarray(self)` — [`L653`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L653) — asarray should handle WithAux wrapping a QArray.
  - `test_dot_pallas_call(self)` — [`L375`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L375) — pallas_call should not be intercepted.
  - `test_nn_ptq(self)` — [`L36`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L36)
  - `test_nn_srq(self, act_calibration_method)` — [`L89`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L89)
  - `test_nnx_einsum_sharding_ptq(self)` — [`L211`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L211)
  - `test_nnx_multi_head_attention(self)` — [`L675`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L675)
  - `test_nnx_ptq(self)` — [`L145`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L145)
  - `test_nnx_scan(self)` — [`L427`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L427) — Test nnx.scan with PTQ.
  - `test_nnx_srq(self, act_calibration_method)` — [`L312`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L312)
  - `test_reshape_weight(self)` — [`L463`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L463) — Test that reshape(weight) can be recognized as a param.
  - `test_symbolic_export(self)` — [`L408`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L408) — Test jax export with symbolic shape.
- protocol/private: `_get_quantized_asarray_model`[`L547`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L547)
- uses (calls/refs, reference-scoped): [`weight_qtype`](../../../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`quantize_model`](../../../qwix/_src/model.md#quantize_model), [`QuantizationRule`](../../../qwix/_src/qconfig.md#QuantizationRule), [`QArray`](../../../qwix/_src/core/qarray.md#QArray), [`act_qtype`](../../../qwix/_src/qconfig.md#QuantizationRule.act_qtype), [`module_path`](../../../qwix/_src/qconfig.md#QuantizationRule.module_path), [`PtqProvider`](../../../qwix/_src/providers/ptq.md#PtqProvider), [`WithAux`](../../../qwix/_src/providers/ptq.md#WithAux), [`quantize_params`](../../../qwix/_src/providers/ptq.md#quantize_params), [`tile_size`](../../../qwix/_src/qconfig.md#QuantizationRule.tile_size), [`act_static_scale`](../../../qwix/_src/qconfig.md#QuantizationRule.act_static_scale), [`QtProvider`](../../../qwix/_src/providers/qt.md#QtProvider), [`act_calibration_method`](../../../qwix/_src/qconfig.md#QuantizationRule.act_calibration_method), [`QuantStat`](../../../qwix/_src/utils/flax_util.md#QuantStat), [`astype`](../../../qwix/_src/providers/ptq.md#WithAux.astype), [`shape`](../../../qwix/_src/providers/ptq.md#WithAux.shape), `bias`, `kernel`, [`dtype`](../../../qwix/_src/providers/ptq.md#WithAux.dtype), [`ndim`](../../../qwix/_src/providers/ptq.md#WithAux.ndim), `kernel`  (8 test-only)

### `ReshapeDense`  ·  implements/extends Module
- def: [`tests/_src/providers/ptq_test.py:466`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L466)
- signature: `class ReshapeDense(nn.Module):`
- members:
  - `out_features` — [`L467`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L467)
- protocol/private: `__call__`[`L470`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L470)
- used by: (1 test-only callers)

### `ScanModel`  ·  implements/extends Module
- def: [`tests/_src/providers/ptq_test.py:430`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L430)
- signature: `class ScanModel(nnx.Module):`
- members:
  - `create_layer(rngs: nnx.Rngs)` — [`L435`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L435)
  - `scan_fn(x: jax.Array, layer)` — [`L442`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L442)
  - `layers` — [`L438`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L438)
- protocol/private: `__call__`[`L440`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L440), `__init__`[`L432`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L432)
- used by: (1 test-only callers)

### `TestModel`  ·  implements/extends Module
- def: [`tests/_src/providers/ptq_test.py:500`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L500)
- signature: `class TestModel(nn.Module):`
- members:
  - `compute_loss(self, x)` — [`L512`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L512)
  - `predict(self, x)` — [`L509`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L509)
  - `setup(self)` — [`L502`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L502)
  - `dense_layer` — [`L503`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L503)
  - `ssl_loss_layer` — [`L504`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L504)
- protocol/private: `__call__`[`L506`](../../../../../../../raw/code/qwix/tests/_src/providers/ptq_test.py#L506)
- used by: (1 test-only callers)

