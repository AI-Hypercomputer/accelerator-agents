---
title: 'Module: tests/_src/core/einsum_info_test.py'
type: catalog
provenance: extracted
module: tests/_src/core/einsum_info_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.core.einsum_info_test`/
symbols:
  EinsumInfoTest.test_parse: EinsumInfoTest#test_parse().
  BroadcastOperandsTest.test_broadcast_operands_concrete: BroadcastOperandsTest#test_broadcast_operands_concrete().
  SymbolicDim.name: SymbolicDim#name.
  EinsumInfoTest.test_dimension_numbers: EinsumInfoTest#test_dimension_numbers().
  EinsumInfoTest.test_output_perm: EinsumInfoTest#test_output_perm().
  BroadcastOperandsTest.test_broadcast_operands_symbolic_parameterized: BroadcastOperandsTest#test_broadcast_operands_symbolic_parameterized().
  SymbolicDim: SymbolicDim#
  SymbolicDim.__mul__: SymbolicDim#__mul__().
  SymbolicDim.__rmul__: SymbolicDim#__rmul__().
  SymbolicDim.__eq__: SymbolicDim#__eq__().
  EinsumInfoTest.test_parse_validation: EinsumInfoTest#test_parse_validation().
  SymbolicEinsumTest.test_einsum_contract_path_symbolic: SymbolicEinsumTest#test_einsum_contract_path_symbolic().
  SymbolicEinsumTest.test_sanitize_shape: SymbolicEinsumTest#test_sanitize_shape().
  BroadcastOperandsTest.create_operand: BroadcastOperandsTest#create_operand().
  BroadcastOperandsTest.create_op: BroadcastOperandsTest#create_op().
  SymbolicDim.__int__: SymbolicDim#__int__().
  SymbolicDim.__index__: SymbolicDim#__index__().
  SymbolicDim.__str__: SymbolicDim#__str__().
  SymbolicDim.__repr__: SymbolicDim#__repr__().
  SymbolicDim.__hash__: SymbolicDim#__hash__().
  BroadcastOperandsTest.test_broadcast_operands_error: BroadcastOperandsTest#test_broadcast_operands_error().
  SymbolicDim.__init__: SymbolicDim#__init__().
  SymbolicDim.__gt__: SymbolicDim#__gt__().
  EinsumInfoTest: EinsumInfoTest#
  BroadcastOperandsTest: BroadcastOperandsTest#
  SymbolicEinsumTest: SymbolicEinsumTest#
---
# Module: [`tests/_src/core/einsum_info_test.py`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py)

## Classes
### `BroadcastOperandsTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/einsum_info_test.py:186`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L186)
- signature: `class BroadcastOperandsTest(parameterized.TestCase):`
- members:
  - `create_op(info)` — [`L298`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L298)
  - `create_operand(shape, is_qarray)` — [`L230`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L230)
  - `test_broadcast_operands_concrete(self, op1_shape, op2_shape, expected_shape, lhs_is_qarray=False, rhs_is_qarray=False)` — [`L222`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L222)
  - `test_broadcast_operands_error(self)` — [`L255`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L255)
  - `test_broadcast_operands_symbolic_parameterized(self, op1_info, op2_info, expected_shape, updates)` — [`L291`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L291)
- uses (calls/refs, reference-scoped): [`QArray`](../../../qwix/_src/core/qarray.md#QArray), [`qvalue`](../../../qwix/_src/core/qarray.md#QArray.qvalue), [`scale`](../../../qwix/_src/core/qarray.md#QArray.scale), [`broadcast_operands`](../../../qwix/_src/core/einsum_info.md#broadcast_operands)  (1 test-only)

### `EinsumInfoTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/einsum_info_test.py:64`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L64)
- signature: `class EinsumInfoTest(parameterized.TestCase):`
- members:
  - `test_dimension_numbers(self)` — [`L164`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L164)
  - `test_output_perm(self)` — [`L176`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L176)
  - `test_parse(self, einsum_str, expected_lhs, expected_rhs, expected_out, expected_batch, expected_contract, ndims=None)` — [`L134`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L134)
  - `test_parse_validation(self, einsum_str)` — [`L160`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L160)
- uses (calls/refs, reference-scoped): [`batch_chars`](../../../qwix/_src/core/einsum_info.md#EinsumInfo.batch_chars), [`contract_chars`](../../../qwix/_src/core/einsum_info.md#EinsumInfo.contract_chars), [`output_perm`](../../../qwix/_src/core/einsum_info.md#EinsumInfo.output_perm), [`parse`](../../../qwix/_src/core/einsum_info.md#EinsumInfo.parse), [`dimension_numbers`](../../../qwix/_src/core/einsum_info.md#EinsumInfo.dimension_numbers), [`rhs`](../../../qwix/_src/core/einsum_info.md#EinsumInfo.rhs), [`lhs`](../../../qwix/_src/core/einsum_info.md#EinsumInfo.lhs), [`EinsumInfo`](../../../qwix/_src/core/einsum_info.md#EinsumInfo), [`out`](../../../qwix/_src/core/einsum_info.md#EinsumInfo.out)

### `SymbolicDim`
- def: [`tests/_src/core/einsum_info_test.py:26`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L26)
- doc: Mock class for symbolic dimension.
- signature: `class SymbolicDim:`
- members:
  - `name` — [`L30`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L30)
- protocol/private: `__eq__`[`L54`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L54), `__gt__`[`L60`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L60), `__hash__`[`L57`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L57), `__index__`[`L37`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L37), `__init__`[`L29`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L29), `__int__`[`L32`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L32), `__mul__`[`L48`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L48), `__repr__`[`L45`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L45), `__rmul__`[`L51`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L51), `__str__`[`L42`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L42)
- used by: (3 test-only callers)

### `SymbolicEinsumTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/einsum_info_test.py:330`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L330)
- signature: `class SymbolicEinsumTest(absltest.TestCase):`
- members:
  - `test_einsum_contract_path_symbolic(self)` — [`L332`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L332)
  - `test_sanitize_shape(self)` — [`L352`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_info_test.py#L352)
- uses (calls/refs, reference-scoped): [`sanitize_shape`](../../../qwix/_src/core/einsum_info.md#sanitize_shape)  (1 test-only)

