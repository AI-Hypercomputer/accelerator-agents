---
title: 'Module: test/test_symbolic_shapes.py'
type: catalog
provenance: extracted
module: test/test_symbolic_shapes.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_symbolic_shapes`/
symbols:
  SymbolicShapeTest.test_constraints_min_max: SymbolicShapeTest#test_constraints_min_max().
  SymbolicShapeTest.test_constraints_multiply: SymbolicShapeTest#test_constraints_multiply().
  SymbolicShapeTest.test_constraint_indirection: SymbolicShapeTest#test_constraint_indirection().
  SymbolicShapeTest: SymbolicShapeTest#
  AddOne: AddOne#
  ConcatAddModel: ConcatAddModel#
  AddOne.__init__: AddOne#__init__().
  AddOne.forward: AddOne#forward().
  ConcatAddModel.__init__: ConcatAddModel#__init__().
  ConcatAddModel.forward: ConcatAddModel#forward().
  SymbolicShapeTest.setUp: SymbolicShapeTest#setUp().
---
# Module: [`test/test_symbolic_shapes.py`](../../../../../raw/code/torchax/test/test_symbolic_shapes.py)

## Classes
### `AddOne`  ·  implements/extends Module
- def: [`test/test_symbolic_shapes.py:23`](../../../../../raw/code/torchax/test/test_symbolic_shapes.py#L23)
- signature: `class AddOne(torch.nn.Module):`
- members:
  - `forward(self, a)` — [`L27`](../../../../../raw/code/torchax/test/test_symbolic_shapes.py#L27)
- protocol/private: `__init__`[`L24`](../../../../../raw/code/torchax/test/test_symbolic_shapes.py#L24)
- used by: (2 test-only callers)

### `ConcatAddModel`  ·  implements/extends Module
- def: [`test/test_symbolic_shapes.py:31`](../../../../../raw/code/torchax/test/test_symbolic_shapes.py#L31)
- signature: `class ConcatAddModel(torch.nn.Module):`
- members:
  - `forward(self, a, b)` — [`L35`](../../../../../raw/code/torchax/test/test_symbolic_shapes.py#L35)
- protocol/private: `__init__`[`L32`](../../../../../raw/code/torchax/test/test_symbolic_shapes.py#L32)
- used by: (1 test-only callers)

### `SymbolicShapeTest`  ·  implements/extends TestCase
- def: [`test/test_symbolic_shapes.py:40`](../../../../../raw/code/torchax/test/test_symbolic_shapes.py#L40)
- doc: Test possible symbolic shape computations that upstream torch export can
- signature: `class SymbolicShapeTest(base_test_util.TestCase):`
- members:
  - `setUp(self)` — [`L46`](../../../../../raw/code/torchax/test/test_symbolic_shapes.py#L46)
  - `test_constraint_indirection(self)` — [`L90`](../../../../../raw/code/torchax/test/test_symbolic_shapes.py#L90) — Test a model where none of the shapes are directly symbolic variables
  - `test_constraints_min_max(self)` — [`L49`](../../../../../raw/code/torchax/test/test_symbolic_shapes.py#L49) — Test a model with basic min/max dimension restrictions
  - `test_constraints_multiply(self)` — [`L68`](../../../../../raw/code/torchax/test/test_symbolic_shapes.py#L68) — Test a model with a slightly more complex constraint, where the input
- uses (calls/refs, reference-scoped): [`exported_program_to_stablehlo`](../torchax/export.md#exported_program_to_stablehlo)  (3 test-only)

