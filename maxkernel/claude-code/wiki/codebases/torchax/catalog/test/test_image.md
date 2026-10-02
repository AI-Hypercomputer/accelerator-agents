---
title: 'Module: test/test_image.py'
type: catalog
provenance: extracted
module: test/test_image.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_image`/
symbols:
  TestResampling.test_resampling_combinations_bicubic: TestResampling#test_resampling_combinations_bicubic().
  upsample_jit: upsample_jit().
  TestResampling: TestResampling#
---
# Module: [`test/test_image.py`](../../../../../raw/code/torchax/test/test_image.py)

## Classes
### `TestResampling`  ·  implements/extends TestCase
- def: [`test/test_image.py:46`](../../../../../raw/code/torchax/test/test_image.py#L46)
- signature: `class TestResampling(parameterized.TestCase):`
- members:
  - `test_resampling_combinations_bicubic(self, antialias, align_corners)` — [`L57`](../../../../../raw/code/torchax/test/test_image.py#L57)
- uses (calls/refs, reference-scoped): [`default_env`](../torchax/__init__.md#default_env), [`jax_view`](../torchax/interop.md#jax_view), [`j2t_copy`](../torchax/tensor.md#Environment.j2t_copy), [`to_xla`](../torchax/tensor.md#Environment.to_xla)  (1 test-only)

## Functions
- `upsample_jit(tensor, output_size: tuple[int, int], align_corners: bool, antialias: bool, method: str)` — [`L28`](../../../../../raw/code/torchax/test/test_image.py#L28)

