---
title: 'Module: test/test_flax.py'
type: catalog
provenance: extracted
module: test/test_flax.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_flax`/
symbols:
  FlaxTest.test_flax_simple: FlaxTest#test_flax_simple().
  FlaxTest.test_flax_functional_call: FlaxTest#test_flax_functional_call().
  FlaxTest.test_flax_module_nested: FlaxTest#test_flax_module_nested().
  Parent.cnn: Parent#cnn.
  FlaxTest.test_flax_module_nested.Parent.forward: FlaxTest#test_flax_module_nested().Parent#forward().
  FlaxTest.jitted: FlaxTest#jitted().
  CNN: CNN#
  FlaxTest.test_flax_module_nested.Parent: FlaxTest#test_flax_module_nested().Parent#
  Parent.a: Parent#a.
  CNN.__call__: CNN#__call__().
  FlaxTest: FlaxTest#
  FlaxTest.test_flax_module_nested.Parent.__init__: FlaxTest#test_flax_module_nested().Parent#__init__().
---
# Module: [`test/test_flax.py`](../../../../../raw/code/torchax/test/test_flax.py)

## Classes
### `CNN`  ·  implements/extends Module
- def: [`test/test_flax.py:27`](../../../../../raw/code/torchax/test/test_flax.py#L27)
- doc: A simple CNN model.
- signature: `class CNN(nn.Module):`
- protocol/private: `__call__`[`L31`](../../../../../raw/code/torchax/test/test_flax.py#L31)
- used by: (3 test-only callers)

### `FlaxTest`  ·  implements/extends TestCase
- def: [`test/test_flax.py:45`](../../../../../raw/code/torchax/test/test_flax.py#L45)
- signature: `class FlaxTest(unittest.TestCase):`
- members:
  - `jitted(weights, args)` — [`L72`](../../../../../raw/code/torchax/test/test_flax.py#L72)
  - `test_flax_functional_call(self)` — [`L60`](../../../../../raw/code/torchax/test/test_flax.py#L60)
  - `test_flax_module_nested(self)` — [`L81`](../../../../../raw/code/torchax/test/test_flax.py#L81)
  - `test_flax_simple(self)` — [`L46`](../../../../../raw/code/torchax/test/test_flax.py#L46)
- uses (calls/refs, reference-scoped): [`default_env`](../torchax/__init__.md#default_env), [`jax_jit`](../torchax/interop.md#jax_jit), [`forward`](../torchax/flax.md#FlaxNNModule.forward), [`prng_key`](../torchax/tensor.md#Environment.prng_key), [`FlaxNNModule`](../torchax/flax.md#FlaxNNModule)  (2 test-only)

### `Parent`  ·  implements/extends Module
- def: [`test/test_flax.py:84`](../../../../../raw/code/torchax/test/test_flax.py#L84)
- signature: `class Parent(torch.nn.Module):`
- members:
  - `forward(self, x)` — [`L91`](../../../../../raw/code/torchax/test/test_flax.py#L91)
  - `a` — [`L87`](../../../../../raw/code/torchax/test/test_flax.py#L87)
  - `cnn` — [`L89`](../../../../../raw/code/torchax/test/test_flax.py#L89)
- protocol/private: `__init__`[`L85`](../../../../../raw/code/torchax/test/test_flax.py#L85)
- uses (calls/refs, reference-scoped): [`FlaxNNModule`](../torchax/flax.md#FlaxNNModule)  (1 test-only)
- used by: (1 test-only callers)

