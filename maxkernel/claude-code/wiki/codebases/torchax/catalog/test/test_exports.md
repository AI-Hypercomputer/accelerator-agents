---
title: 'Module: test/test_exports.py'
type: catalog
provenance: extracted
module: test/test_exports.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_exports`/
symbols:
  ExportTest.test_interpolate: ExportTest#test_interpolate().
  ExportTest.test_constant: ExportTest#test_constant().
  ExportTest.test_interpolate_dynamic: ExportTest#test_interpolate_dynamic().
  ExportTest.test_export_dtypes: ExportTest#test_export_dtypes().
  ExportTest.setUp: ExportTest#setUp().
  Interpolate: Interpolate#
  TensorConstant: TensorConstant#
  Interpolate.forward: Interpolate#forward().
  TensorConstant.__init__: TensorConstant#__init__().
  TensorConstant.forward: TensorConstant#forward().
  ExportTest: ExportTest#
---
# Module: [`test/test_exports.py`](../../../../../raw/code/torchax/test/test_exports.py)

## Classes
### `ExportTest`  ·  implements/extends TestCase
- def: [`test/test_exports.py:47`](../../../../../raw/code/torchax/test/test_exports.py#L47)
- signature: `class ExportTest(unittest.TestCase):`
- members:
  - `setUp(self)` — [`L48`](../../../../../raw/code/torchax/test/test_exports.py#L48)
  - `test_constant(self)` — [`L75`](../../../../../raw/code/torchax/test/test_exports.py#L75)
  - `test_export_dtypes(self)` — [`L113`](../../../../../raw/code/torchax/test/test_exports.py#L113)
  - `test_interpolate(self)` — [`L52`](../../../../../raw/code/torchax/test/test_exports.py#L52)
  - `test_interpolate_dynamic(self)` — [`L96`](../../../../../raw/code/torchax/test/test_exports.py#L96)
- uses (calls/refs, reference-scoped): [`default_env`](../torchax/__init__.md#default_env), [`exported_program_to_jax`](../torchax/export.md#exported_program_to_jax), [`t2j_copy`](../torchax/tensor.md#Environment.t2j_copy), [`j2t_copy`](../torchax/tensor.md#Environment.j2t_copy), [`exported_program_to_stablehlo`](../torchax/export.md#exported_program_to_stablehlo), [`enable_accuracy_mode`](../torchax/__init__.md#enable_accuracy_mode)  (2 test-only)

### `Interpolate`  ·  implements/extends Module
- def: [`test/test_exports.py:28`](../../../../../raw/code/torchax/test/test_exports.py#L28)
- signature: `class Interpolate(torch.nn.Module):`
- members:
  - `forward(self, masks: torch.Tensor)` — [`L29`](../../../../../raw/code/torchax/test/test_exports.py#L29)
- used by: (2 test-only callers)

### `TensorConstant`  ·  implements/extends Module
- def: [`test/test_exports.py:39`](../../../../../raw/code/torchax/test/test_exports.py#L39)
- signature: `class TensorConstant(torch.nn.Module):`
- members:
  - `forward(self, a)` — [`L43`](../../../../../raw/code/torchax/test/test_exports.py#L43)
- protocol/private: `__init__`[`L40`](../../../../../raw/code/torchax/test/test_exports.py#L40)
- used by: (2 test-only callers)

