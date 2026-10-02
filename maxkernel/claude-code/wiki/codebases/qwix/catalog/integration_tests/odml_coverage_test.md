---
title: 'Module: integration_tests/odml_coverage_test.py'
type: catalog
provenance: extracted
module: integration_tests/odml_coverage_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `integration_tests.odml_coverage_test`/
symbols:
  OdmlTest.test_srq: OdmlTest#test_srq().
  OdmlTest.test_drq: OdmlTest#test_drq().
  OdmlTest.test_nnx: OdmlTest#test_nnx().
  OdmlTest.test_weight_only: OdmlTest#test_weight_only().
  OdmlTest.test_partial_quantization_unet_succeeds: OdmlTest#test_partial_quantization_unet_succeeds().
  OdmlTest.test_vae_separate_export: OdmlTest#test_vae_separate_export().
  Transformer.__call__: Transformer#__call__().
  OdmlTest.test_partial_quantization: OdmlTest#test_partial_quantization().
  OdmlTest.test_model_output_pytree: OdmlTest#test_model_output_pytree().
  srq_test_case: srq_test_case().
  print_diff: print_diff().
  CNN: CNN#
  DropoutModel.expected_quant_stats_keys: DropoutModel#expected_quant_stats_keys.
  UNet.__call__: UNet#__call__().
  VAE.__call__: VAE#__call__().
  OdmlTest.test_nnx.NnxModel.__call__: OdmlTest#test_nnx().NnxModel#__call__().
  drq_test_case: drq_test_case().
  Transformer.use_bias: Transformer#use_bias.
  OdmlTest._save_edge_model: OdmlTest#_save_edge_model().
  VAE.encode: VAE#encode().
  VAE.decode: VAE#decode().
  Transformer.vocab_size: Transformer#vocab_size.
  CNN.drq_expected_ops_summary: CNN#drq_expected_ops_summary.
  Transformer.create_input: Transformer#create_input().
  Transformer.drq_expected_ops_summary: Transformer#drq_expected_ops_summary.
  GroupNormSilu.expected_ops_summary: GroupNormSilu#expected_ops_summary.
  DenseConcatenateResidual.expected_ops_summary: DenseConcatenateResidual#expected_ops_summary.
  UNet.expected_ops_summary: UNet#expected_ops_summary.
  ConvBnRelu.expected_ops_summary: ConvBnRelu#expected_ops_summary.
  ValuePreservingPrimitivesModel: ValuePreservingPrimitivesModel#
  ValuePreservingPrimitivesModel.expected_ops_summary: ValuePreservingPrimitivesModel#expected_ops_summary.
  srq_test_cases: srq_test_cases.
  drq_test_cases: drq_test_cases.
  Transformer.embedding_dim: Transformer#embedding_dim.
  UNet: UNet#
  OdmlTest._summarize_ops_details: OdmlTest#_summarize_ops_details().
  Transformer.num_layers: Transformer#num_layers.
  Transformer.num_heads: Transformer#num_heads.
  Transformer.qkv_features: Transformer#qkv_features.
  Transformer.hidden_dim: Transformer#hidden_dim.
  GroupNormSilu: GroupNormSilu#
  UNet.encoder_filters_sequence: UNet#encoder_filters_sequence.
  UNet.decoder_filters_sequence: UNet#decoder_filters_sequence.
  UNet.create_input: UNet#create_input().
  ModelOutputPyTree: ModelOutputPyTree#
  ModelOutputPyTree.create_input: ModelOutputPyTree#create_input().
  VAE: VAE#
  VAE.encoder: VAE#encoder.
  VAE.decoder: VAE#decoder.
  VAE.create_input: VAE#create_input().
  OdmlTest.apply_with_rng: OdmlTest#apply_with_rng().
  OdmlTest.test_nnx.NnxModel: OdmlTest#test_nnx().NnxModel#
  NnxModel.linear1: NnxModel#linear1.
  NnxModel.linear2: NnxModel#linear2.
  CNN.__call__: CNN#__call__().
  CNN.create_input: CNN#create_input().
  CNN.expected_quant_stats_keys: CNN#expected_quant_stats_keys.
  CNN.expected_ops_summary: CNN#expected_ops_summary.
  DropoutModel: DropoutModel#
  DropoutModel.skip_export: DropoutModel#skip_export.
  DropoutModel.__call__: DropoutModel#__call__().
  DropoutModel.create_input: DropoutModel#create_input().
  Transformer: Transformer#
  Transformer.expected_quant_stats_keys: Transformer#expected_quant_stats_keys.
  Transformer.expected_ops_summary: Transformer#expected_ops_summary.
  GroupNormSilu.__call__: GroupNormSilu#__call__().
  GroupNormSilu.create_input: GroupNormSilu#create_input().
  GroupNormSilu.expected_quant_stats_keys: GroupNormSilu#expected_quant_stats_keys.
  DenseConcatenateResidual: DenseConcatenateResidual#
  DenseConcatenateResidual.__call__: DenseConcatenateResidual#__call__().
  DenseConcatenateResidual.create_input: DenseConcatenateResidual#create_input().
  DenseConcatenateResidual.expected_quant_stats_keys: DenseConcatenateResidual#expected_quant_stats_keys.
  UNet.expected_quant_stats_keys: UNet#expected_quant_stats_keys.
  UNet.additional_provider_args: UNet#additional_provider_args.
  UNet.expected_quant_stats_values: UNet#expected_quant_stats_values.
  ConvBnRelu: ConvBnRelu#
  ConvBnRelu.__call__: ConvBnRelu#__call__().
  ConvBnRelu.create_input: ConvBnRelu#create_input().
  ConvBnRelu.expected_quant_stats_keys: ConvBnRelu#expected_quant_stats_keys.
  RepeatNegative: RepeatNegative#
  RepeatNegative.__call__: RepeatNegative#__call__().
  RepeatNegative.create_input: RepeatNegative#create_input().
  ValuePreservingPrimitivesModel.__call__: ValuePreservingPrimitivesModel#__call__().
  ValuePreservingPrimitivesModel.create_input: ValuePreservingPrimitivesModel#create_input().
  ValuePreservingPrimitivesModel.expected_quant_stats_keys: ValuePreservingPrimitivesModel#expected_quant_stats_keys.
  ModelOutputPyTree.__call__: ModelOutputPyTree#__call__().
  ModelOutputPyTree.expected_quant_stats_keys: ModelOutputPyTree#expected_quant_stats_keys.
  ModelOutputPyTree.expected_ops_summary: ModelOutputPyTree#expected_ops_summary.
  VAE.setup: VAE#setup().
  OdmlTest: OdmlTest#
  OdmlTest.test_nnx.NnxModel.__init__: OdmlTest#test_nnx().NnxModel#__init__().
---
# Module: [`integration_tests/odml_coverage_test.py`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py)

## Classes
### `CNN`  ·  implements/extends Module
- def: [`integration_tests/odml_coverage_test.py:53`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L53)
- doc: A simple CNN model.
- signature: `class CNN(nn.Module):`
- members:
  - `create_input(self)` — [`L70`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L70)
  - `drq_expected_ops_summary` — [`L90`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L90)
  - `expected_ops_summary` — [`L83`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L83)
  - `expected_quant_stats_keys` — [`L73`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L73)
- protocol/private: `__call__`[`L57`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L57)
- uses (calls/refs, reference-scoped): [`srq_test_case`](odml_coverage_test.md#srq_test_case), [`drq_test_case`](odml_coverage_test.md#drq_test_case)
- used by: [`test_weight_only`](odml_coverage_test.md#OdmlTest.test_weight_only)

### `ConvBnRelu`  ·  implements/extends Module
- def: [`integration_tests/odml_coverage_test.py:357`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L357)
- doc: A single Conv + BN + Relu block.
- signature: `class ConvBnRelu(nn.Module):`
- members:
  - `create_input(self)` — [`L372`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L372)
  - `expected_ops_summary` — [`L381`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L381)
  - `expected_quant_stats_keys` — [`L375`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L375)
- protocol/private: `__call__`[`L361`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L361)
- uses (calls/refs, reference-scoped): [`srq_test_case`](odml_coverage_test.md#srq_test_case)

### `DenseConcatenateResidual`  ·  implements/extends Module
- def: [`integration_tests/odml_coverage_test.py:236`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L236)
- signature: `class DenseConcatenateResidual(nn.Module):`
- members:
  - `create_input(self)` — [`L248`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L248)
  - `expected_ops_summary` — [`L259`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L259)
  - `expected_quant_stats_keys` — [`L251`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L251)
- protocol/private: `__call__`[`L239`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L239)
- uses (calls/refs, reference-scoped): [`srq_test_case`](odml_coverage_test.md#srq_test_case)

### `DropoutModel`  ·  implements/extends Module
- def: [`integration_tests/odml_coverage_test.py:97`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L97)
- doc: A model with Dropout to verify metadata propagation.
- signature: `class DropoutModel(nn.Module):`
- members:
  - `create_input(self)` — [`L111`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L111)
  - `expected_quant_stats_keys` — [`L114`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L114)
  - `skip_export` — [`L100`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L100)
- protocol/private: `__call__`[`L103`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L103)
- uses (calls/refs, reference-scoped): [`srq_test_case`](odml_coverage_test.md#srq_test_case), [`drq_test_case`](odml_coverage_test.md#drq_test_case)

### `GroupNormSilu`  ·  implements/extends Module
- def: [`integration_tests/odml_coverage_test.py:208`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L208)
- signature: `class GroupNormSilu(nn.Module):`
- members:
  - `create_input(self)` — [`L216`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L216)
  - `expected_ops_summary` — [`L227`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L227)
  - `expected_quant_stats_keys` — [`L219`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L219)
- protocol/private: `__call__`[`L211`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L211)
- uses (calls/refs, reference-scoped): [`srq_test_case`](odml_coverage_test.md#srq_test_case)
- used by: [`test_partial_quantization`](odml_coverage_test.md#OdmlTest.test_partial_quantization)

### `ModelOutputPyTree`  ·  implements/extends Module
- def: [`integration_tests/odml_coverage_test.py:461`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L461)
- doc: Verify that tuple outputs are tagged correctly.
- signature: `class ModelOutputPyTree(nn.Module):`
- members:
  - `create_input(self)` — [`L473`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L473)
  - `expected_ops_summary` — [`L481`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L481)
  - `expected_quant_stats_keys` — [`L476`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L476)
- protocol/private: `__call__`[`L465`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L465)
- used by: [`test_model_output_pytree`](odml_coverage_test.md#OdmlTest.test_model_output_pytree)

### `NnxModel`  ·  implements/extends Module
- def: [`integration_tests/odml_coverage_test.py:898`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L898)
- signature: `class NnxModel(nnx.Module):`
- members:
  - `linear1` — [`L901`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L901)
  - `linear2` — [`L902`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L902)
- protocol/private: `__call__`[`L904`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L904), `__init__`[`L900`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L900)
- used by: [`test_nnx`](odml_coverage_test.md#OdmlTest.test_nnx)

### `OdmlTest`  ·  implements/extends TestCase
- def: [`integration_tests/odml_coverage_test.py:507`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L507)
- signature: `class OdmlTest(parameterized.TestCase):`
- members:
  - `_save_edge_model(self, edge_model)` — [`L612`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L612) — Save the edge model to test outputs.
  - `_summarize_ops_details(self, ops_details)` — [`L620`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L620) — Summarize the ops details.
  - `apply_with_rng(params, inputs)` — [`L584`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L584)
  - `test_drq(self, model: nn.Module)` — [`L815`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L815)
  - `test_model_output_pytree(self)` — [`L735`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L735)
  - `test_nnx(self)` — [`L897`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L897)
  - `test_partial_quantization(self)` — [`L709`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L709)
  - `test_partial_quantization_unet_succeeds(self, module_path: str)` — [`L775`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L775)
  - `test_srq(self, model: nn.Module)` — [`L510`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L510)
  - `test_vae_separate_export(self)` — [`L868`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L868)
  - `test_weight_only(self, model: nn.Module)` — [`L664`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L664)
- uses (calls/refs, reference-scoped): [`weight_qtype`](../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`quantize_model`](../qwix/_src/model.md#quantize_model), [`QuantizationRule`](../qwix/_src/qconfig.md#QuantizationRule), [`act_qtype`](../qwix/_src/qconfig.md#QuantizationRule.act_qtype), [`module_path`](../qwix/_src/qconfig.md#QuantizationRule.module_path), [`act_static_scale`](../qwix/_src/qconfig.md#QuantizationRule.act_static_scale), [`OdmlQatProvider`](../qwix/_src/providers/odml.md#OdmlQatProvider), [`OdmlConversionProvider`](../qwix/_src/providers/odml.md#OdmlConversionProvider), [`QuantStat`](../qwix/_src/utils/flax_util.md#QuantStat), [`print_diff`](odml_coverage_test.md#print_diff), [`CNN`](odml_coverage_test.md#CNN), [`op_names`](../qwix/_src/qconfig.md#QuantizationRule.op_names), [`UNet`](odml_coverage_test.md#UNet), [`drq_test_cases`](odml_coverage_test.md#drq_test_cases), [`srq_test_cases`](odml_coverage_test.md#srq_test_cases), [`GroupNormSilu`](odml_coverage_test.md#GroupNormSilu), [`ModelOutputPyTree`](odml_coverage_test.md#ModelOutputPyTree), [`NnxModel`](odml_coverage_test.md#OdmlTest.test_nnx.NnxModel), [`VAE`](odml_coverage_test.md#VAE), [`create_input`](odml_coverage_test.md#ModelOutputPyTree.create_input), [`create_input`](odml_coverage_test.md#UNet.create_input), [`create_input`](odml_coverage_test.md#VAE.create_input)

### `RepeatNegative`  ·  implements/extends Module
- def: [`integration_tests/odml_coverage_test.py:390`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L390)
- signature: `class RepeatNegative(nn.Module):`
- members:
  - `create_input(self)` — [`L397`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L397)
- protocol/private: `__call__`[`L393`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L393)

### `Transformer`  ·  implements/extends Module
- def: [`integration_tests/odml_coverage_test.py:123`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L123)
- doc: A simple Transformer model.
- signature: `class Transformer(nn.Module):`
- members:
  - `create_input(self)` — [`L152`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L152)
  - `drq_expected_ops_summary` — [`L201`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L201)
  - `embedding_dim` — [`L131`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L131)
  - `expected_ops_summary` — [`L195`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L195)
  - `expected_quant_stats_keys` — [`L161`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L161)
  - `hidden_dim` — [`L129`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L129)
  - `num_heads` — [`L127`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L127)
  - `num_layers` — [`L126`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L126)
  - `qkv_features` — [`L128`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L128)
  - `use_bias` — [`L132`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L132)
  - `vocab_size` — [`L130`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L130)
- protocol/private: `__call__`[`L135`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L135)
- uses (calls/refs, reference-scoped): [`srq_test_case`](odml_coverage_test.md#srq_test_case)

### `UNet`  ·  implements/extends Module
- def: [`integration_tests/odml_coverage_test.py:268`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L268)
- doc: A simple UNet model.
- signature: `class UNet(nn.Module):`
- members:
  - `create_input(self)` — [`L311`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L311)
  - `additional_provider_args` — [`L334`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L334)
  - `decoder_filters_sequence` — [`L272`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L272)
  - `encoder_filters_sequence` — [`L271`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L271)
  - `expected_ops_summary` — [`L348`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L348)
  - `expected_quant_stats_keys` — [`L317`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L317)
  - `expected_quant_stats_values` — [`L339`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L339)
- protocol/private: `__call__`[`L275`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L275)
- uses (calls/refs, reference-scoped): [`srq_test_case`](odml_coverage_test.md#srq_test_case)
- used by: [`test_partial_quantization_unet_succeeds`](odml_coverage_test.md#OdmlTest.test_partial_quantization_unet_succeeds), [`test_weight_only`](odml_coverage_test.md#OdmlTest.test_weight_only)

### `VAE`  ·  implements/extends Module
- def: [`integration_tests/odml_coverage_test.py:488`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L488)
- signature: `class VAE(nn.Module):`
- members:
  - `create_input(self)` — [`L503`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L503)
  - `decode(self, x)` — [`L497`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L497)
  - `encode(self, x)` — [`L494`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L494)
  - `setup(self)` — [`L490`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L490)
  - `decoder` — [`L492`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L492)
  - `encoder` — [`L491`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L491)
- protocol/private: `__call__`[`L500`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L500)
- used by: [`test_vae_separate_export`](odml_coverage_test.md#OdmlTest.test_vae_separate_export)

### `ValuePreservingPrimitivesModel`  ·  implements/extends Module
- def: [`integration_tests/odml_coverage_test.py:402`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L402)
- doc: A model with value-preserving primitives to verify metadata propagation.
- signature: `class ValuePreservingPrimitivesModel(nn.Module):`
- members:
  - `create_input(self)` — [`L442`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L442)
  - `expected_ops_summary` — [`L450`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L450)
  - `expected_quant_stats_keys` — [`L445`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L445)
- protocol/private: `__call__`[`L406`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L406)
- uses (calls/refs, reference-scoped): [`srq_test_case`](odml_coverage_test.md#srq_test_case)

## Functions
- `drq_test_case(cls)` — [`L46`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L46)
- `print_diff(name, x, y)` — [`L947`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L947)
- `srq_test_case(cls)` — [`L41`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L41)

## Module values
- `drq_test_cases` — [`L38`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L38)
- `srq_test_cases` — [`L37`](../../../../../raw/code/qwix/integration_tests/odml_coverage_test.py#L37)

