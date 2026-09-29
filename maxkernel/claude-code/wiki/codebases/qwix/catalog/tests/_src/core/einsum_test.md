---
title: 'Module: tests/_src/core/einsum_test.py'
type: catalog
provenance: extracted
module: tests/_src/core/einsum_test.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `tests._src.core.einsum_test`/
symbols:
  EinsumTest._make_array: EinsumTest#_make_array().
  EinsumTest.test_fake_quantization: EinsumTest#test_fake_quantization().
  EinsumTest.test_dequant_on_inputs: EinsumTest#test_dequant_on_inputs().
  EinsumTest.test_broadcasting_and_mixed_types: EinsumTest#test_broadcasting_and_mixed_types().
  EinsumTest.test_innermost_tiling_heuristic: EinsumTest#test_innermost_tiling_heuristic().
  EinsumTest.test_einsum_and_benchmark: EinsumTest#test_einsum_and_benchmark().
  EinsumTest.test_einsum: EinsumTest#test_einsum().
  EinsumTest.quantize: EinsumTest#quantize().
  EinsumTest._einsum: EinsumTest#_einsum().
  EinsumTest.test_generic_broadcasting_incompatible: EinsumTest#test_generic_broadcasting_incompatible().
  time_it: time_it().
  EinsumTest.fq_einsum: EinsumTest#fq_einsum().
  EinsumTest.q_einsum: EinsumTest#q_einsum().
  EinsumTest: EinsumTest#
---
# Module: [`tests/_src/core/einsum_test.py`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_test.py)

## Classes
### `EinsumTest`  ·  implements/extends TestCase
- def: [`tests/_src/core/einsum_test.py:34`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_test.py#L34)
- signature: `class EinsumTest(parameterized.TestCase):`
- members:
  - `fq_einsum(lhs, rhs)` — [`L286`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_test.py#L286)
  - `q_einsum(lhs, rhs)` — [`L295`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_test.py#L295)
  - `quantize(array, for_lhs)` — [`L143`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_test.py#L143)
  - `test_broadcasting_and_mixed_types(self, einsum_str, lhs_shape, rhs_shape, expected_shape, lhs_qtype=None, rhs_qtype=None, rhs_is_scalar=False, expected_rel_error=None, skip_reference_check=False)` — [`L383`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_test.py#L383)
  - `test_dequant_on_inputs(self)` — [`L317`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_test.py#L317)
  - `test_einsum(self, *, einsum_str, qtype, lhs_shape, rhs_shape, tile_size=None, lhs_asymmetric=False, expected_rel_mae)` — [`L228`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_test.py#L228)
  - `test_einsum_and_benchmark(self, rhs_qtype: jax.typing.DTypeLike, expected_mae: float, lhs_qtype: jax.typing.DTypeLike | None = None, tile_size: int | None = None)` — [`L128`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_test.py#L128)
  - `test_fake_quantization(self)` — [`L270`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_test.py#L270)
  - `test_generic_broadcasting_incompatible(self)` — [`L444`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_test.py#L444)
  - `test_innermost_tiling_heuristic(self)` — [`L450`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_test.py#L450)
- protocol/private: `_einsum`[`L157`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_test.py#L157), `_make_array`[`L36`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_test.py#L36)
- uses (calls/refs, reference-scoped): [`HowToQuantize`](../../../qwix/_src/core/qarray.md#HowToQuantize), [`qtype`](../../../qwix/_src/core/qarray.md#HowToQuantize.qtype), [`quantize`](../../../qwix/_src/core/qarray.md#quantize), [`tiled_axes`](../../../qwix/_src/core/qarray.md#HowToQuantize.tiled_axes), [`channelwise_axes`](../../../qwix/_src/core/qarray.md#HowToQuantize.channelwise_axes), [`einsum`](../../../qwix/_src/core/einsum.md#einsum), [`get_how_to_quantize`](../../../qwix/_src/core/einsum.md#get_how_to_quantize)  (1 test-only)

## Functions
- `time_it(f, *args)` — [`L27`](../../../../../../../raw/code/qwix/tests/_src/core/einsum_test.py#L27)

