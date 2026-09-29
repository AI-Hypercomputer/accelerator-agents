---
title: 'Module: tests/_src/providers/lora_test.py'
type: catalog
provenance: extracted
module: tests/_src/providers/lora_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.providers.lora_test`/
symbols:
  LoraTest.test_lora_einsum_nnx_sharded: LoraTest#test_lora_einsum_nnx_sharded().
  LoraTest.test_lora_dot_general_nnx: LoraTest#test_lora_dot_general_nnx().
  LoraTest.test_lora_einsum_nnx: LoraTest#test_lora_einsum_nnx().
  LoraTest.test_lora_dot_general_batch: LoraTest#test_lora_dot_general_batch().
  LoraTest.test_lora_conv_nnx: LoraTest#test_lora_conv_nnx().
  LoraTest.test_lora_dot_general_nn: LoraTest#test_lora_dot_general_nn().
  LoraTest.test_lora_dot_general_multi_out_dim: LoraTest#test_lora_dot_general_multi_out_dim().
  LoraTest.test_lora_dot_general_no_weight: LoraTest#test_lora_dot_general_no_weight().
  LoraTest.test_lora_dot_general_no_remain: LoraTest#test_lora_dot_general_no_remain().
  LoraTest.test_nnx_remat: LoraTest#test_nnx_remat().
  LoraTest.test_lora_conv_nn: LoraTest#test_lora_conv_nn().
  LoraTest.jit_apply: LoraTest#jit_apply().
  LoraTest._shard_nnx_model: LoraTest#_shard_nnx_model().
  LoraTest.test_parse_einsum_str_for_lora: LoraTest#test_parse_einsum_str_for_lora().
  LoraTest.test_parse_einsum_str_for_lora_invalid_einsum_str: LoraTest#test_parse_einsum_str_for_lora_invalid_einsum_str().
  LoraTest.test_lora_dot_general_batch.BatchLinear.__call__: LoraTest#test_lora_dot_general_batch().BatchLinear#__call__().
  LoraTest.test_lora_dot_general_multi_out_dim.MultiOutLinear.__call__: LoraTest#test_lora_dot_general_multi_out_dim().MultiOutLinear#__call__().
  LoraTest.test_lora_dot_general_no_remain.FullContractModel.__call__: LoraTest#test_lora_dot_general_no_remain().FullContractModel#__call__().
  LoraTest.test_nnx_remat.Model.__call__: LoraTest#test_nnx_remat().Model#__call__().
  LoraTest.loss_fn: LoraTest#loss_fn().
  BatchLinear.kernel: BatchLinear#kernel.
  Model.linear: Model#linear.
  LoraTest.test_lora_dot_general_batch.BatchLinear: LoraTest#test_lora_dot_general_batch().BatchLinear#
  LoraTest.test_lora_dot_general_multi_out_dim.MultiOutLinear: LoraTest#test_lora_dot_general_multi_out_dim().MultiOutLinear#
  MultiOutLinear.kernel: MultiOutLinear#kernel.
  LoraTest.test_lora_dot_general_no_weight.NoWeightModel: LoraTest#test_lora_dot_general_no_weight().NoWeightModel#
  LoraTest.test_lora_dot_general_no_remain.FullContractModel: LoraTest#test_lora_dot_general_no_remain().FullContractModel#
  FullContractModel.kernel: FullContractModel#kernel.
  LoraTest.test_nnx_remat.Model: LoraTest#test_nnx_remat().Model#
  LoraTest: LoraTest#
  LoraTest.test_lora_dot_general_batch.BatchLinear.__init__: LoraTest#test_lora_dot_general_batch().BatchLinear#__init__().
  LoraTest.test_lora_dot_general_multi_out_dim.MultiOutLinear.__init__: LoraTest#test_lora_dot_general_multi_out_dim().MultiOutLinear#__init__().
  LoraTest.test_lora_dot_general_no_weight.NoWeightModel.__call__: LoraTest#test_lora_dot_general_no_weight().NoWeightModel#__call__().
  LoraTest.test_lora_dot_general_no_remain.FullContractModel.__init__: LoraTest#test_lora_dot_general_no_remain().FullContractModel#__init__().
  LoraTest.test_nnx_remat.Model.__init__: LoraTest#test_nnx_remat().Model#__init__().
---
# Module: [`tests/_src/providers/lora_test.py`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py)

## Classes
### `BatchLinear`  ·  implements/extends Module
- def: [`tests/_src/providers/lora_test.py:363`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L363)
- signature: `class BatchLinear(nnx.Module):`
- members:
  - `kernel` — [`L367`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L367)
- protocol/private: `__call__`[`L371`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L371), `__init__`[`L365`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L365)
- used by: (1 test-only callers)

### `FullContractModel`  ·  implements/extends Module
- def: [`tests/_src/providers/lora_test.py:496`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L496)
- signature: `class FullContractModel(nnx.Module):`
- members:
  - `kernel` — [`L499`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L499)
- protocol/private: `__call__`[`L501`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L501), `__init__`[`L498`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L498)
- used by: (1 test-only callers)

### `LoraTest`  ·  implements/extends TestCase
- def: [`tests/_src/providers/lora_test.py:30`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L30)
- signature: `class LoraTest(parameterized.TestCase):`
- members:
  - `_shard_nnx_model(self, model: nnx.Module, mesh: jax.sharding.Mesh)` — [`L602`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L602) — Shards the model in-place with the given mesh.
  - `jit_apply(model, x)` — [`L140`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L140)
  - `loss_fn(model, x)` — [`L141`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L141)
  - `test_lora_conv_nn(self)` — [`L512`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L512) — Test LoRA on nn.Conv module.
  - `test_lora_conv_nnx(self, qtype)` — [`L544`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L544) — Test LoRA on nnx.Conv module.
  - `test_lora_dot_general_batch(self)` — [`L357`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L357) — Test LoRA on a dot_general operation with batch dimensions.
  - `test_lora_dot_general_multi_out_dim(self)` — [`L418`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L418) — Test LoRA on a dot_general operation with multiple output dimensions.
  - `test_lora_dot_general_nn(self, weight_qtype)` — [`L322`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L322) — Test LoRA on nn.Dense module.
  - `test_lora_dot_general_nnx(self, dropout_rate)` — [`L122`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L122) — Test LoRA on nnx.Linear module.
  - `test_lora_dot_general_no_remain(self)` — [`L493`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L493) — Tests LoRA where there are no remaining dimensions (full contraction).
  - `test_lora_dot_general_no_weight(self)` — [`L472`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L472) — Tests that LoRA skips operations where rhs is not a weight.
  - `test_lora_einsum_nnx(self, dropout_rate)` — [`L167`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L167) — Test LoRA on nnx.Einsum module.
  - `test_lora_einsum_nnx_sharded(self, weight_qtype, apply_sharding_to_base_model)` — [`L218`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L218) — Test QLoRA on nnx.Einsum module param with sharding.
  - `test_nnx_remat(self)` — [`L609`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L609) — Test nnx.remat with LoRA.
  - `test_parse_einsum_str_for_lora(self, einsum_str, lhs_shape, rhs_shape, lora_rank, expected_lora_einsum_str, expected_a_sharding_transpose, expected_b_sharding_transpose, expected_output_shape)` — [`L64`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L64)
  - `test_parse_einsum_str_for_lora_invalid_einsum_str(self, einsum_str, lhs_shape, rhs_shape)` — [`L110`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L110)
- uses (calls/refs, reference-scoped): [`weight_qtype`](../../../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`module_path`](../../../qwix/_src/qconfig.md#QuantizationRule.module_path), [`WithAux`](../../../qwix/_src/providers/ptq.md#WithAux), [`tile_size`](../../../qwix/_src/qconfig.md#QuantizationRule.tile_size), [`apply_lora_to_model`](../../../qwix/_src/providers/lora.md#apply_lora_to_model), [`_parse_einsum_str_for_lora`](../../../qwix/_src/providers/lora.md#_parse_einsum_str_for_lora), [`LoraProvider`](../../../qwix/_src/providers/lora.md#LoraProvider), [`LoraRule`](../../../qwix/_src/providers/lora.md#LoraRule), [`rank`](../../../qwix/_src/providers/lora.md#LoraRule.rank), [`dropout`](../../../qwix/_src/providers/lora.md#LoraRule.dropout), [`alpha`](../../../qwix/_src/providers/lora.md#LoraRule.alpha), `bias`, `kernel`, [`lora_b_initializer`](../../../qwix/_src/providers/lora.md#LoraRule.lora_b_initializer), `kernel`  (6 test-only)

### `Model`  ·  implements/extends Module
- def: [`tests/_src/providers/lora_test.py:612`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L612)
- signature: `class Model(nnx.Module):`
- members:
  - `linear` — [`L615`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L615)
- protocol/private: `__call__`[`L617`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L617), `__init__`[`L614`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L614)
- used by: (1 test-only callers)

### `MultiOutLinear`  ·  implements/extends Module
- def: [`tests/_src/providers/lora_test.py:424`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L424)
- signature: `class MultiOutLinear(nnx.Module):`
- members:
  - `kernel` — [`L428`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L428)
- protocol/private: `__call__`[`L434`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L434), `__init__`[`L426`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L426)
- used by: (1 test-only callers)

### `NoWeightModel`  ·  implements/extends Module
- def: [`tests/_src/providers/lora_test.py:475`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L475)
- signature: `class NoWeightModel(nnx.Module):`
- protocol/private: `__call__`[`L477`](../../../../../../../raw/code/qwix/tests/_src/providers/lora_test.py#L477)
- used by: (1 test-only callers)

