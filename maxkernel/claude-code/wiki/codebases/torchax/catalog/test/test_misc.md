---
title: 'Module: test/test_misc.py'
type: catalog
provenance: extracted
module: test/test_misc.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_misc`/MiscTest#
symbols:
  MiscTest.test_random_with_tensor_input: test_random_with_tensor_input().
  MiscTest.test_extract_jax_kwargs: test_extract_jax_kwargs().
  MiscTest.rand_plus_one: rand_plus_one().
  MiscTest.test_to_device: test_to_device().
  MiscTest.test_to_device_twice: test_to_device_twice().
  MiscTest.test_zeros_with_explicit_size: test_zeros_with_explicit_size().
  MiscTest.test_extract_jax_kwargs.M: test_extract_jax_kwargs().M#
  MiscTest: ''
  MiscTest.test_extract_jax_kwargs.M.forward: test_extract_jax_kwargs().M#forward().
---
# Module: [`test/test_misc.py`](../../../../../raw/code/torchax/test/test_misc.py)

## Classes
### `M`  ·  implements/extends Module
- def: [`test/test_misc.py:30`](../../../../../raw/code/torchax/test/test_misc.py#L30)
- signature: `class M(torch.nn.Module):`
- members:
  - `forward(self, a, b)` — [`L31`](../../../../../raw/code/torchax/test/test_misc.py#L31)
- used by: (1 test-only callers)

### `MiscTest`  ·  implements/extends TestCase
- def: [`test/test_misc.py:28`](../../../../../raw/code/torchax/test/test_misc.py#L28)
- signature: `class MiscTest(unittest.TestCase):`
- members:
  - `rand_plus_one(rng)` — [`L83`](../../../../../raw/code/torchax/test/test_misc.py#L83)
  - `test_extract_jax_kwargs(self)` — [`L29`](../../../../../raw/code/torchax/test/test_misc.py#L29)
  - `test_random_with_tensor_input(self)` — [`L65`](../../../../../raw/code/torchax/test/test_misc.py#L65)
  - `test_to_device(self)` — [`L42`](../../../../../raw/code/torchax/test/test_misc.py#L42)
  - `test_to_device_twice(self)` — [`L53`](../../../../../raw/code/torchax/test/test_misc.py#L53)
  - `test_zeros_with_explicit_size(self)` — [`L92`](../../../../../raw/code/torchax/test/test_misc.py#L92)
- uses (calls/refs, reference-scoped): [`default_env`](../torchax/__init__.md#default_env), [`manual_seed`](../torchax/tensor.md#Environment.manual_seed), [`extract_jax`](../torchax/__init__.md#extract_jax)  (1 test-only)

