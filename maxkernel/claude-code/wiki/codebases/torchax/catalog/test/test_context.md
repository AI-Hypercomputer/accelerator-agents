---
title: 'Module: test/test_context.py'
type: catalog
provenance: extracted
module: test/test_context.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_context`/
symbols:
  xla_env: xla_env.
  TestContext.test_override_property: TestContext#test_override_property().
  TestContext.test_jit_with_rng: TestContext#test_jit_with_rng().
  TestContext.test_buffer: TestContext#test_buffer().
  TestContext.test_same_manual_seed: TestContext#test_same_manual_seed().
  TestContext.test_different_manual_seed: TestContext#test_different_manual_seed().
  TestContext.test_mode_context_manager: TestContext#test_mode_context_manager().
  TestContext.test_mode_decorator: TestContext#test_mode_decorator().
  TestContext._test_mode_decorator: TestContext#_test_mode_decorator().
  TestContext.test_generator_seed: TestContext#test_generator_seed().
  TestContext.test_buffer.M: TestContext#test_buffer().M#
  TestContext.random_op: TestContext#random_op().
  TestContext: TestContext#
  TestContext.test_buffer.M.__init__: TestContext#test_buffer().M#__init__().
---
# Module: [`test/test_context.py`](../../../../../raw/code/torchax/test/test_context.py)

## Classes
### `M`  ·  implements/extends Module
- def: [`test/test_context.py:105`](../../../../../raw/code/torchax/test/test_context.py#L105)
- signature: `class M(torch.nn.Module):`
- protocol/private: `__init__`[`L106`](../../../../../raw/code/torchax/test/test_context.py#L106)
- used by: (1 test-only callers)

### `TestContext`  ·  implements/extends TestCase
- def: [`test/test_context.py:27`](../../../../../raw/code/torchax/test/test_context.py#L27)
- signature: `class TestContext(unittest.TestCase):`
- members:
  - `random_op()` — [`L83`](../../../../../raw/code/torchax/test/test_context.py#L83)
  - `test_buffer(self)` — [`L104`](../../../../../raw/code/torchax/test/test_context.py#L104)
  - `test_different_manual_seed(self)` — [`L68`](../../../../../raw/code/torchax/test/test_context.py#L68)
  - `test_generator_seed(self)` — [`L96`](../../../../../raw/code/torchax/test/test_context.py#L96)
  - `test_jit_with_rng(self)` — [`L80`](../../../../../raw/code/torchax/test/test_context.py#L80)
  - `test_mode_context_manager(self)` — [`L28`](../../../../../raw/code/torchax/test/test_context.py#L28)
  - `test_mode_decorator(self)` — [`L43`](../../../../../raw/code/torchax/test/test_context.py#L43)
  - `test_override_property(self)` — [`L48`](../../../../../raw/code/torchax/test/test_context.py#L48)
  - `test_same_manual_seed(self)` — [`L56`](../../../../../raw/code/torchax/test/test_context.py#L56)
- protocol/private: `_test_mode_decorator`[`L37`](../../../../../raw/code/torchax/test/test_context.py#L37)
- uses (calls/refs, reference-scoped): [`Tensor`](../torchax/tensor.md#Tensor), [`manual_seed`](../torchax/tensor.md#Environment.manual_seed), [`param`](../torchax/tensor.md#Environment.param), [`override_property`](../torchax/tensor.md#Environment.override_property), [`jax_jit`](../torchax/interop.md#jax_jit), [`to_xla`](../torchax/tensor.md#Environment.to_xla), [`Environment`](../torchax/tensor.md#Environment), [`prng`](../torchax/tensor.md#RuntimeProperty.prng)  (2 test-only)

## Module values
- `xla_env` — [`L24`](../../../../../raw/code/torchax/test/test_context.py#L24)

