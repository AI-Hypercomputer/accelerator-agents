---
title: 'Module: tests/contrib/padded_ptq_test.py'
type: catalog
provenance: extracted
module: tests/contrib/padded_ptq_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests.contrib.padded_ptq_test`/
symbols:
  PaddedPtqTest.test_einsum_simple: PaddedPtqTest#test_einsum_simple().
  PaddedPtqTest.test_dot_general_simple: PaddedPtqTest#test_dot_general_simple().
  D: D.
  PaddedPtqTest.test_einsum_simple.EinsumModel.__call__: PaddedPtqTest#test_einsum_simple().EinsumModel#__call__().
  PaddedPtqTest.test_dot_general_simple.DotModel.__call__: PaddedPtqTest#test_dot_general_simple().DotModel#__call__().
  E: E.
  F: F.
  PaddedPtqTest.test_einsum_simple.EinsumModel: PaddedPtqTest#test_einsum_simple().EinsumModel#
  PaddedPtqTest.test_dot_general_simple.DotModel: PaddedPtqTest#test_dot_general_simple().DotModel#
  T: T.
  PaddedPtqTest.test_einsum_simple.EinsumModel.e: PaddedPtqTest#test_einsum_simple().EinsumModel#e.
  PaddedPtqTest.test_einsum_simple.EinsumModel.d: PaddedPtqTest#test_einsum_simple().EinsumModel#d.
  PaddedPtqTest.test_einsum_simple.EinsumModel.f: PaddedPtqTest#test_einsum_simple().EinsumModel#f.
  PaddedPtqTest.test_einsum_simple.EinsumModel.w_init: PaddedPtqTest#test_einsum_simple().EinsumModel#w_init.
  PaddedPtqTest.test_dot_general_simple.DotModel.e: PaddedPtqTest#test_dot_general_simple().DotModel#e.
  PaddedPtqTest.test_dot_general_simple.DotModel.d: PaddedPtqTest#test_dot_general_simple().DotModel#d.
  PaddedPtqTest.test_dot_general_simple.DotModel.f: PaddedPtqTest#test_dot_general_simple().DotModel#f.
  PaddedPtqTest.test_dot_general_simple.DotModel.w_init: PaddedPtqTest#test_dot_general_simple().DotModel#w_init.
  PaddedPtqTest: PaddedPtqTest#
---
# Module: [`tests/contrib/padded_ptq_test.py`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py)

## Classes
### `DotModel`  ·  implements/extends Module
- def: [`tests/contrib/padded_ptq_test.py:155`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L155)
- signature: `class DotModel(nn.Module):`
- members:
  - `d` — [`L157`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L157)
  - `e` — [`L156`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L156)
  - `f` — [`L158`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L158)
  - `w_init` — [`L159`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L159)
- protocol/private: `__call__`[`L162`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L162)
- used by: (1 test-only callers)

### `EinsumModel`  ·  implements/extends Module
- def: [`tests/contrib/padded_ptq_test.py:79`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L79)
- signature: `class EinsumModel(nn.Module):`
- members:
  - `d` — [`L81`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L81)
  - `e` — [`L80`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L80)
  - `f` — [`L82`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L82)
  - `w_init` — [`L83`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L83)
- protocol/private: `__call__`[`L86`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L86)
- used by: (1 test-only callers)

### `PaddedPtqTest`  ·  implements/extends TestCase
- def: [`tests/contrib/padded_ptq_test.py:30`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L30)
- signature: `class PaddedPtqTest(parameterized.TestCase):`
- members:
  - `test_dot_general_simple(self, weight_qtype, act_qtype, tile_size)` — [`L115`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L115) — documented in [qwix-_src-qconfig](../../../concepts/qwix-_src-qconfig.md)
  - `test_einsum_simple(self, weight_qtype, act_qtype, tile_size)` — [`L38`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L38) — documented in [qwix-_src-qconfig](../../../concepts/qwix-_src-qconfig.md)
- uses (calls/refs, reference-scoped): [`weight_qtype`](../../qwix/_src/qconfig.md#QuantizationRule.weight_qtype), [`quantize_model`](../../qwix/_src/model.md#quantize_model), [`QuantizationRule`](../../qwix/_src/qconfig.md#QuantizationRule), [`act_qtype`](../../qwix/_src/qconfig.md#QuantizationRule.act_qtype), [`module_path`](../../qwix/_src/qconfig.md#QuantizationRule.module_path), [`qvalue`](../../qwix/_src/core/qarray.md#QArray.qvalue), [`PtqProvider`](../../qwix/_src/providers/ptq.md#PtqProvider), [`scale`](../../qwix/_src/core/qarray.md#QArray.scale), [`tile_size`](../../qwix/_src/qconfig.md#QuantizationRule.tile_size), [`quantize_act`](../../qwix/_src/providers/ptq.md#quantize_act), [`get_how_to_quantize`](../../qwix/_src/core/einsum.md#get_how_to_quantize), [`get_how_to_quantize`](../../qwix/_src/core/dot_general.md#get_how_to_quantize), [`PaddedPtqProvider`](../../qwix/contrib/padded_ptq.md#PaddedPtqProvider), [`quantize_act`](../../qwix/contrib/padded_ptq.md#quantize_act)  (6 test-only)

## Module values
- `D` — [`L26`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L26)
- `E` — [`L27`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L27)
- `F` — [`L27`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L27)
- `T` — [`L27`](../../../../../../raw/code/qwix/tests/contrib/padded_ptq_test.py#L27)

