---
title: 'Module: tests/_src/providers/odml_test.py'
type: catalog
provenance: extracted
module: tests/_src/providers/odml_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.providers.odml_test`/
symbols:
  OdmlTest.test_metadata_propagation_reciprocal_division: OdmlTest#test_metadata_propagation_reciprocal_division().
  OdmlTest.test_nnx: OdmlTest#test_nnx().
  OdmlTest.test_mismatched_sibling_parameter_isolation: OdmlTest#test_mismatched_sibling_parameter_isolation().
  OdmlTest.test_linen: OdmlTest#test_linen().
  OdmlTest.test_matched_siblings_sharing_stats: OdmlTest#test_matched_siblings_sharing_stats().
  OdmlTest.test_mismatched_sibling_quantized_vs_float: OdmlTest#test_mismatched_sibling_quantized_vs_float().
  OdmlTest.test_matched_siblings_with_reshape_sharing: OdmlTest#test_matched_siblings_with_reshape_sharing().
  OdmlTest.test_immediate_matched_siblings_sharing_stats: OdmlTest#test_immediate_matched_siblings_sharing_stats().
  OdmlTest.test_metadata_propagation_linear_arithmetic: OdmlTest#test_metadata_propagation_linear_arithmetic().
  OdmlTest.test_linen_shared_scope: OdmlTest#test_linen_shared_scope().
  OdmlTest.test_linen_no_shared_scope: OdmlTest#test_linen_no_shared_scope().
  OdmlTest.test_odml_interception_stack: OdmlTest#test_odml_interception_stack().
  OdmlTest.test_mixed_tags_at_boundary: OdmlTest#test_mixed_tags_at_boundary().
  OdmlTest.test_linen_shared_scope.SharedScopeModel.__call__: OdmlTest#test_linen_shared_scope().SharedScopeModel#__call__().
  OdmlTest.test_linen_no_shared_scope.NoSharedScopeModel.__call__: OdmlTest#test_linen_no_shared_scope().NoSharedScopeModel#__call__().
  NamedParamModule.features: NamedParamModule#features.
  NamedParamModule.param_name: NamedParamModule#param_name.
  NamedParamModule.__call__: NamedParamModule#__call__().
  OdmlTest.test_nnx.NnxModel.__call__: OdmlTest#test_nnx().NnxModel#__call__().
  NamedParamModule: NamedParamModule#
  OdmlTest.test_linen.LinenModel: OdmlTest#test_linen().LinenModel#
  OdmlTest.test_linen_shared_scope.SharedScopeModel: OdmlTest#test_linen_shared_scope().SharedScopeModel#
  OdmlTest.test_linen_no_shared_scope.NoSharedScopeModel: OdmlTest#test_linen_no_shared_scope().NoSharedScopeModel#
  OdmlTest.test_nnx.NnxModel: OdmlTest#test_nnx().NnxModel#
  NnxModel.linear1: NnxModel#linear1.
  NnxModel.linear2: NnxModel#linear2.
  OdmlTest.test_mixed_tags_at_boundary.BranchModel: OdmlTest#test_mixed_tags_at_boundary().BranchModel#
  OdmlTest.test_matched_siblings_sharing_stats.SiblingModel: OdmlTest#test_matched_siblings_sharing_stats().SiblingModel#
  OdmlTest.test_mismatched_sibling_quantized_vs_float.MixedSiblingModel: OdmlTest#test_mismatched_sibling_quantized_vs_float().MixedSiblingModel#
  OdmlTest.test_mismatched_sibling_parameter_isolation.MultiQuantSiblingModel: OdmlTest#test_mismatched_sibling_parameter_isolation().MultiQuantSiblingModel#
  OdmlTest.test_matched_siblings_with_reshape_sharing.SiblingReshapeModel: OdmlTest#test_matched_siblings_with_reshape_sharing().SiblingReshapeModel#
  OdmlTest.test_immediate_matched_siblings_sharing_stats.SiblingModel: OdmlTest#test_immediate_matched_siblings_sharing_stats().SiblingModel#
  OdmlTest.test_metadata_propagation_linear_arithmetic.LinearPropagationModel: OdmlTest#test_metadata_propagation_linear_arithmetic().LinearPropagationModel#
  OdmlTest.test_metadata_propagation_reciprocal_division.ReciprocalDivisionModel: OdmlTest#test_metadata_propagation_reciprocal_division().ReciprocalDivisionModel#
  OdmlTest: OdmlTest#
  OdmlTest.test_linen.LinenModel.__call__: OdmlTest#test_linen().LinenModel#__call__().
  OdmlTest.test_nnx.NnxModel.__init__: OdmlTest#test_nnx().NnxModel#__init__().
  OdmlTest.test_mixed_tags_at_boundary.BranchModel.__call__: OdmlTest#test_mixed_tags_at_boundary().BranchModel#__call__().
  OdmlTest.test_matched_siblings_sharing_stats.SiblingModel.__call__: OdmlTest#test_matched_siblings_sharing_stats().SiblingModel#__call__().
  OdmlTest.test_mismatched_sibling_quantized_vs_float.MixedSiblingModel.__call__: OdmlTest#test_mismatched_sibling_quantized_vs_float().MixedSiblingModel#__call__().
  OdmlTest.test_mismatched_sibling_parameter_isolation.MultiQuantSiblingModel.__call__: OdmlTest#test_mismatched_sibling_parameter_isolation().MultiQuantSiblingModel#__call__().
  OdmlTest.test_matched_siblings_with_reshape_sharing.SiblingReshapeModel.__call__: OdmlTest#test_matched_siblings_with_reshape_sharing().SiblingReshapeModel#__call__().
  OdmlTest.test_immediate_matched_siblings_sharing_stats.SiblingModel.__call__: OdmlTest#test_immediate_matched_siblings_sharing_stats().SiblingModel#__call__().
  OdmlTest.test_metadata_propagation_linear_arithmetic.LinearPropagationModel.__call__: OdmlTest#test_metadata_propagation_linear_arithmetic().LinearPropagationModel#__call__().
  OdmlTest.test_metadata_propagation_reciprocal_division.ReciprocalDivisionModel.__call__: OdmlTest#test_metadata_propagation_reciprocal_division().ReciprocalDivisionModel#__call__().
---
# Module: [`tests/_src/providers/odml_test.py`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py)

## Classes
### `BranchModel`  ·  implements/extends Module
- def: [`tests/_src/providers/odml_test.py:234`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L234)
- signature: `class BranchModel(nn.Module):`
- protocol/private: `__call__`[`L237`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L237)
- used by: (1 test-only callers)

### `LinearPropagationModel`  ·  implements/extends Module
- def: [`tests/_src/providers/odml_test.py:529`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L529)
- signature: `class LinearPropagationModel(nn.Module):`
- protocol/private: `__call__`[`L532`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L532)
- used by: (1 test-only callers)

### `LinenModel`  ·  implements/extends Module
- def: [`tests/_src/providers/odml_test.py:46`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L46)
- signature: `class LinenModel(nn.Module):`
- protocol/private: `__call__`[`L49`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L49)
- used by: (1 test-only callers)

### `MixedSiblingModel`  ·  implements/extends Module
- def: [`tests/_src/providers/odml_test.py:326`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L326)
- signature: `class MixedSiblingModel(nn.Module):`
- protocol/private: `__call__`[`L329`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L329)
- used by: (1 test-only callers)

### `MultiQuantSiblingModel`  ·  implements/extends Module
- def: [`tests/_src/providers/odml_test.py:374`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L374)
- signature: `class MultiQuantSiblingModel(nn.Module):`
- protocol/private: `__call__`[`L377`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L377)
- used by: (1 test-only callers)

### `NamedParamModule`  ·  implements/extends Module
- def: [`tests/_src/providers/odml_test.py:31`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L31)
- signature: `class NamedParamModule(nn.Module):`
- members:
  - `features` — [`L32`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L32)
  - `param_name` — [`L33`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L33)
- protocol/private: `__call__`[`L36`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L36)
- used by: (2 test-only callers)

### `NnxModel`  ·  implements/extends Module
- def: [`tests/_src/providers/odml_test.py:158`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L158)
- signature: `class NnxModel(nnx.Module):`
- members:
  - `linear1` — [`L161`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L161)
  - `linear2` — [`L162`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L162)
- protocol/private: `__call__`[`L164`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L164), `__init__`[`L160`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L160)
- used by: (1 test-only callers)

### `NoSharedScopeModel`  ·  implements/extends Module
- def: [`tests/_src/providers/odml_test.py:126`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L126)
- signature: `class NoSharedScopeModel(nn.Module):`
- protocol/private: `__call__`[`L129`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L129)
- uses (calls/refs, reference-scoped): (3 test-only callers)
- used by: (1 test-only callers)

### `OdmlTest`  ·  implements/extends TestCase
- def: [`tests/_src/providers/odml_test.py:43`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L43)
- signature: `class OdmlTest(parameterized.TestCase):`
- members:
  - `test_immediate_matched_siblings_sharing_stats(self)` — [`L477`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L477) — Test that matched sibling branches using immediate quantization share the tracer and stats.
  - `test_linen(self)` — [`L45`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L45)
  - `test_linen_no_shared_scope(self)` — [`L123`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L123) — Test that standard submodules have separate counters and paths.
  - `test_linen_shared_scope(self)` — [`L91`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L91) — Test that shared scopes do not cause naming collisions.
  - `test_matched_siblings_sharing_stats(self)` — [`L268`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L268) — Test that matched sibling branches correctly share the tracer and resolve scales.
  - `test_matched_siblings_with_reshape_sharing(self)` — [`L426`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L426) — Test that sibling branches separated by reshape calibrate and convert separately.
  - `test_metadata_propagation_linear_arithmetic(self)` — [`L526`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L526) — Test that elementwise linear arithmetic propagates quantization rules and ALLOW_FUSION.
  - `test_metadata_propagation_reciprocal_division(self)` — [`L583`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L583) — Test that activation / const propagates metadata but const / activation does not.
  - `test_mismatched_sibling_parameter_isolation(self)` — [`L371`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L371) — Test that sibling branches with different quantized parameters are strictly isolated.
  - `test_mismatched_sibling_quantized_vs_float(self)` — [`L323`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L323) — Test that mismatched sibling branches (quantized vs float) are isolated.
  - `test_mixed_tags_at_boundary(self)` — [`L233`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L233)
  - `test_nnx(self)` — [`L157`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L157)
  - `test_odml_interception_stack(self)` — [`L207`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L207) — Verifies that ODML providers return interceptors in the correct order.
- uses (calls/refs, reference-scoped): [`weight_qtype`](../../../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`quantize_model`](../../../qwix/_src/model.md#quantize_model), [`QuantizationRule`](../../../qwix/_src/qconfig.md#QuantizationRule), [`act_qtype`](../../../qwix/_src/qconfig.md#QuantizationRule.act_qtype), [`module_path`](../../../qwix/_src/qconfig.md#QuantizationRule.module_path), [`AuxDataKey`](../../../qwix/_src/providers/odml_ops.md#AuxDataKey), [`get`](../../../qwix/_src/aux_data.md#get), [`mapping`](../../../qwix/_src/interception.md#Interceptor.mapping), [`OdmlQatProvider`](../../../qwix/_src/providers/odml.md#OdmlQatProvider), [`OdmlConversionProvider`](../../../qwix/_src/providers/odml.md#OdmlConversionProvider), [`act_calibration_method`](../../../qwix/_src/qconfig.md#QuantizationRule.act_calibration_method), [`get_interceptors`](../../../qwix/_src/providers/odml.md#OdmlQatProvider.get_interceptors), [`QuantStat`](../../../qwix/_src/utils/flax_util.md#QuantStat), [`FQ_RULE`](../../../qwix/_src/providers/odml_ops.md#AuxDataKey.FQ_RULE), [`PRIMITIVE_BIND_KEY`](../../../qwix/_src/interception.md#PRIMITIVE_BIND_KEY), [`PrimitiveBindOp`](../../../qwix/_src/providers/odml_ops.md#PrimitiveBindOp)  (12 test-only)

### `ReciprocalDivisionModel`  ·  implements/extends Module
- def: [`tests/_src/providers/odml_test.py:586`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L586)
- signature: `class ReciprocalDivisionModel(nn.Module):`
- protocol/private: `__call__`[`L589`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L589)
- used by: (1 test-only callers)

### `SharedScopeModel`  ·  implements/extends Module
- def: [`tests/_src/providers/odml_test.py:94`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L94)
- signature: `class SharedScopeModel(nn.Module):`
- protocol/private: `__call__`[`L97`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L97)
- uses (calls/refs, reference-scoped): (3 test-only callers)
- used by: (1 test-only callers)

### `SiblingModel`  ·  implements/extends Module
- def: [`tests/_src/providers/odml_test.py:480`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L480)
- signature: `class SiblingModel(nn.Module):`
- protocol/private: `__call__`[`L274`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L274), `__call__`[`L483`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L483)
- used by: (1 test-only callers)

### `SiblingReshapeModel`  ·  implements/extends Module
- def: [`tests/_src/providers/odml_test.py:429`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L429)
- signature: `class SiblingReshapeModel(nn.Module):`
- protocol/private: `__call__`[`L432`](../../../../../../../raw/code/qwix/tests/_src/providers/odml_test.py#L432)
- used by: (1 test-only callers)

