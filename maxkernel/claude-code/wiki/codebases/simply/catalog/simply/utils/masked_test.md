---
title: 'Module: simply/utils/masked_test.py'
type: catalog
provenance: extracted
module: simply/utils/masked_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.masked_test`/
symbols:
  _make_random_input_and_mask: _make_random_input_and_mask().
  MaskedTest.test_masked: MaskedTest#test_masked().
  MaskedTest.test_masked_max: MaskedTest#test_masked_max().
  MaskedTest.test_masked_min: MaskedTest#test_masked_min().
  MaskedTest.test_masked_sum: MaskedTest#test_masked_sum().
  MaskedTest.test_masked_mean: MaskedTest#test_masked_mean().
  MaskedTest.test_masked_var: MaskedTest#test_masked_var().
  MaskedTest.test_masked_std: MaskedTest#test_masked_std().
  MaskedTest.test_masked_mean_var: MaskedTest#test_masked_mean_var().
  MaskedTest.test_masked_mean_std: MaskedTest#test_masked_mean_std().
  MaskedTest: MaskedTest#
---
# Module: [`simply/utils/masked_test.py`](../../../../../../raw/code/simply/simply/utils/masked_test.py)

## Classes
### `MaskedTest`  ·  implements/extends TestCase
- def: [`simply/utils/masked_test.py:33`](../../../../../../raw/code/simply/simply/utils/masked_test.py#L33)
- signature: `class MaskedTest(parameterized.TestCase):`
- members:
  - `test_masked(self, padding_value: float)` — [`L38`](../../../../../../raw/code/simply/simply/utils/masked_test.py#L38)
  - `test_masked_max(self, axis: int | tuple[int, ...] | None, keepdims: bool)` — [`L53`](../../../../../../raw/code/simply/simply/utils/masked_test.py#L53)
  - `test_masked_mean(self, axis: int | tuple[int, ...] | None, keepdims: bool)` — [`L116`](../../../../../../raw/code/simply/simply/utils/masked_test.py#L116)
  - `test_masked_mean_std(self, axis: int | tuple[int, ...] | None, ddof: int, keepdims: bool)` — [`L215`](../../../../../../raw/code/simply/simply/utils/masked_test.py#L215)
  - `test_masked_mean_var(self, axis: int | tuple[int, ...] | None, ddof: int, keepdims: bool)` — [`L187`](../../../../../../raw/code/simply/simply/utils/masked_test.py#L187)
  - `test_masked_min(self, axis: int | tuple[int, ...] | None, keepdims: bool)` — [`L74`](../../../../../../raw/code/simply/simply/utils/masked_test.py#L74)
  - `test_masked_std(self, axis: int | tuple[int, ...] | None, ddof: int, keepdims: bool)` — [`L163`](../../../../../../raw/code/simply/simply/utils/masked_test.py#L163)
  - `test_masked_sum(self, axis: int | tuple[int, ...] | None, keepdims: bool)` — [`L95`](../../../../../../raw/code/simply/simply/utils/masked_test.py#L95)
  - `test_masked_var(self, axis: int | tuple[int, ...] | None, ddof: int, keepdims: bool)` — [`L139`](../../../../../../raw/code/simply/simply/utils/masked_test.py#L139)
- uses (calls/refs, reference-scoped): [`masked`](masked.md#masked), [`masked_mean`](masked.md#masked_mean), [`masked_mean_var`](masked.md#masked_mean_var), [`_make_random_input_and_mask`](masked_test.md#_make_random_input_and_mask), [`masked_mean_std`](masked.md#masked_mean_std), [`masked_std`](masked.md#masked_std), [`masked_var`](masked.md#masked_var), [`masked_max`](masked.md#masked_max), [`masked_min`](masked.md#masked_min), [`masked_sum`](masked.md#masked_sum)

## Functions
- `_make_random_input_and_mask(size: int | tuple[int, ...])` — [`L23`](../../../../../../raw/code/simply/simply/utils/masked_test.py#L23)

