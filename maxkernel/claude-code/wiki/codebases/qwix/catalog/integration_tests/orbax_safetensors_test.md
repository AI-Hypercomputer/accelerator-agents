---
title: 'Module: integration_tests/orbax_safetensors_test.py'
type: catalog
provenance: extracted
module: integration_tests/orbax_safetensors_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `integration_tests.orbax_safetensors_test`/
symbols:
  OrbaxSafetensorsIntegrationTest.test_load_full_precision_weights: OrbaxSafetensorsIntegrationTest#test_load_full_precision_weights().
  OrbaxSafetensorsIntegrationTest.test_load_prequantized_checkpoints: OrbaxSafetensorsIntegrationTest#test_load_prequantized_checkpoints().
  OrbaxSafetensorsIntegrationTest.test_load_with_renaming: OrbaxSafetensorsIntegrationTest#test_load_with_renaming().
  OrbaxSafetensorsIntegrationTest.test_load_prequantized_2d_blocksize_checkpoints: OrbaxSafetensorsIntegrationTest#test_load_prequantized_2d_blocksize_checkpoints().
  OrbaxSafetensorsIntegrationTest.create_quantized_model: OrbaxSafetensorsIntegrationTest#create_quantized_model().
  load_nested_safetensors: load_nested_safetensors().
  OrbaxSafetensorsIntegrationTest.assert_quantized_weight: OrbaxSafetensorsIntegrationTest#assert_quantized_weight().
  LanguageModel.__call__: LanguageModel#__call__().
  _flatten_dict: _flatten_dict().
  CustomTestModel: CustomTestModel#
  LanguageModel.embed_tokens: LanguageModel#embed_tokens.
  LanguageModel.layers: LanguageModel#layers.
  QProj.__call__: QProj#__call__().
  LinearAttn.__init__: LinearAttn#__init__().
  LinearAttn.__call__: LinearAttn#__call__().
  Layer.__init__: Layer#__init__().
  Layer.__call__: Layer#__call__().
  EmbedTokens.__call__: EmbedTokens#__call__().
  InnerModel.__init__: InnerModel#__init__().
  InnerModel.__call__: InnerModel#__call__().
  CustomTestModel.__init__: CustomTestModel#__init__().
  CustomTestModel.__call__: CustomTestModel#__call__().
  QProj: QProj#
  QProj.weight: QProj#weight.
  LinearAttn: LinearAttn#
  LinearAttn.q_proj: LinearAttn#q_proj.
  Layer: Layer#
  Layer.linear_attn: Layer#linear_attn.
  EmbedTokens: EmbedTokens#
  EmbedTokens.weight: EmbedTokens#weight.
  LanguageModel: LanguageModel#
  InnerModel: InnerModel#
  InnerModel.language_model: InnerModel#language_model.
  CustomTestModel.model: CustomTestModel#model.
  QProj.__init__: QProj#__init__().
  EmbedTokens.__init__: EmbedTokens#__init__().
  LanguageModel.__init__: LanguageModel#__init__().
  OrbaxSafetensorsIntegrationTest: OrbaxSafetensorsIntegrationTest#
---
# Module: [`integration_tests/orbax_safetensors_test.py`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py)

## Classes
### `CustomTestModel`  ·  implements/extends Module
- def: [`integration_tests/orbax_safetensors_test.py:138`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L138)
- doc: Dummy custom test model.
- signature: `class CustomTestModel(nnx.Module):`
- members:
  - `model` — [`L142`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L142)
- protocol/private: `__call__`[`L144`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L144), `__init__`[`L141`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L141)
- uses (calls/refs, reference-scoped): [`InnerModel`](orbax_safetensors_test.md#InnerModel)
- used by: [`test_load_full_precision_weights`](orbax_safetensors_test.md#OrbaxSafetensorsIntegrationTest.test_load_full_precision_weights), [`test_load_prequantized_checkpoints`](orbax_safetensors_test.md#OrbaxSafetensorsIntegrationTest.test_load_prequantized_checkpoints), [`test_load_with_renaming`](orbax_safetensors_test.md#OrbaxSafetensorsIntegrationTest.test_load_with_renaming), [`create_quantized_model`](orbax_safetensors_test.md#OrbaxSafetensorsIntegrationTest.create_quantized_model)

### `EmbedTokens`  ·  implements/extends Module
- def: [`integration_tests/orbax_safetensors_test.py:109`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L109)
- signature: `class EmbedTokens(nnx.Module):`
- members:
  - `weight` — [`L112`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L112)
- protocol/private: `__call__`[`L114`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L114), `__init__`[`L111`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L111)
- used by: [`embed_tokens`](orbax_safetensors_test.md#LanguageModel.embed_tokens)

### `InnerModel`  ·  implements/extends Module
- def: [`integration_tests/orbax_safetensors_test.py:129`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L129)
- signature: `class InnerModel(nnx.Module):`
- members:
  - `language_model` — [`L132`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L132)
- protocol/private: `__call__`[`L134`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L134), `__init__`[`L131`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L131)
- uses (calls/refs, reference-scoped): [`LanguageModel`](orbax_safetensors_test.md#LanguageModel)
- used by: [`__init__`](orbax_safetensors_test.md#CustomTestModel.__init__)

### `LanguageModel`  ·  implements/extends Module
- def: [`integration_tests/orbax_safetensors_test.py:118`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L118)
- signature: `class LanguageModel(nnx.Module):`
- members:
  - `embed_tokens` — [`L121`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L121)
  - `layers` — [`L122`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L122)
- protocol/private: `__call__`[`L124`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L124), `__init__`[`L120`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L120)
- uses (calls/refs, reference-scoped): [`EmbedTokens`](orbax_safetensors_test.md#EmbedTokens), [`Layer`](orbax_safetensors_test.md#Layer)
- used by: [`__init__`](orbax_safetensors_test.md#InnerModel.__init__)

### `Layer`  ·  implements/extends Module
- def: [`integration_tests/orbax_safetensors_test.py:100`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L100)
- signature: `class Layer(nnx.Module):`
- members:
  - `linear_attn` — [`L103`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L103)
- protocol/private: `__call__`[`L105`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L105), `__init__`[`L102`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L102)
- uses (calls/refs, reference-scoped): [`LinearAttn`](orbax_safetensors_test.md#LinearAttn)
- used by: [`layers`](orbax_safetensors_test.md#LanguageModel.layers)

### `LinearAttn`  ·  implements/extends Module
- def: [`integration_tests/orbax_safetensors_test.py:91`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L91)
- signature: `class LinearAttn(nnx.Module):`
- members:
  - `q_proj` — [`L94`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L94)
- protocol/private: `__call__`[`L96`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L96), `__init__`[`L93`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L93)
- uses (calls/refs, reference-scoped): [`QProj`](orbax_safetensors_test.md#QProj)
- used by: [`__init__`](orbax_safetensors_test.md#Layer.__init__)

### `OrbaxSafetensorsIntegrationTest`  ·  implements/extends TestCase
- def: [`integration_tests/orbax_safetensors_test.py:148`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L148)
- signature: `class OrbaxSafetensorsIntegrationTest(absltest.TestCase):`
- members:
  - `assert_quantized_weight(self, weight: Any, expected_shape: Tuple[int, ...] = (128, 128), expected_qvalue_dtype: Any = jnp.float8_e4m3fn, expected_scale_dtype: Any = jnp.float32)` — [`L150`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L150)
  - `create_quantized_model()` — [`L330`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L330)
  - `test_load_full_precision_weights(self)` — [`L164`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L164)
  - `test_load_prequantized_2d_blocksize_checkpoints(self)` — [`L276`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L276)
  - `test_load_prequantized_checkpoints(self)` — [`L217`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L217)
  - `test_load_with_renaming(self)` — [`L357`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L357)
- uses (calls/refs, reference-scoped): [`weight_qtype`](../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`quantize_model`](../qwix/_src/model.md#quantize_model), [`QuantizationRule`](../qwix/_src/qconfig.md#QuantizationRule), [`act_qtype`](../qwix/_src/qconfig.md#QuantizationRule.act_qtype), [`module_path`](../qwix/_src/qconfig.md#QuantizationRule.module_path), [`PtqProvider`](../qwix/_src/providers/ptq.md#PtqProvider), [`quantize_params`](../qwix/_src/providers/ptq.md#quantize_params), [`process_prequantized_params`](../qwix/_src/utils/checkpoint_util.md#process_prequantized_params), [`tile_size`](../qwix/_src/qconfig.md#QuantizationRule.tile_size), [`act_static_scale`](../qwix/_src/qconfig.md#QuantizationRule.act_static_scale), [`load_nested_safetensors`](orbax_safetensors_test.md#load_nested_safetensors), [`CustomTestModel`](orbax_safetensors_test.md#CustomTestModel)

### `QProj`  ·  implements/extends Module
- def: [`integration_tests/orbax_safetensors_test.py:82`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L82)
- signature: `class QProj(nnx.Module):`
- members:
  - `weight` — [`L85`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L85)
- protocol/private: `__call__`[`L87`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L87), `__init__`[`L84`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L84)
- used by: [`__init__`](orbax_safetensors_test.md#LinearAttn.__init__)

## Functions
- `_flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".")` — [`L32`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L32) — Flattens a nested dictionary into a single-level dictionary.
- `load_nested_safetensors(directory: str | epath.Path, abstract_pytree: Any | None = None, mesh: jax.sharding.Mesh | None = None, rename_rules: Sequence[Tuple[str, str]] | None = None)` — [`L46`](../../../../../raw/code/qwix/integration_tests/orbax_safetensors_test.py#L46) — Loads a safetensors checkpoint as a nested PyTree.

