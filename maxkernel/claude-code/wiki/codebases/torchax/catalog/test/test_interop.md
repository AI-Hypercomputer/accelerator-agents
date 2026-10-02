---
title: 'Module: test/test_interop.py'
type: catalog
provenance: extracted
module: test/test_interop.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_interop`/
symbols:
  ModuleWithUnregisteredTensor.m1: ModuleWithUnregisteredTensor#m1.
  InteropTest.test_module_with_shared_weights: InteropTest#test_module_with_shared_weights().
  InteropTest.test_j2t_autograd_forward: InteropTest#test_j2t_autograd_forward().
  InteropTest.test_j2t_autograd_backward: InteropTest#test_j2t_autograd_backward().
  InteropTest.test_module_with_shared_weights.ModuleWithSharedWeights.forward: InteropTest#test_module_with_shared_weights().ModuleWithSharedWeights#forward().
  InteropTest.test_torch_jax_view_dtype: InteropTest#test_torch_jax_view_dtype().
  ModuleWithSharedWeights.b: ModuleWithSharedWeights#b.
  InteropTest.setUp: InteropTest#setUp().
  InteropTest.test_mod_attr.ModuleWithUnregisteredTensor.__init__: InteropTest#test_mod_attr().ModuleWithUnregisteredTensor#__init__().
  InteropTest.test_to_jax_device: InteropTest#test_to_jax_device().
  is_tpu_available: is_tpu_available().
  InteropTest.fn: InteropTest#fn().
  ModuleWithSharedWeights.a: ModuleWithSharedWeights#a.
  InteropTest.test_mod_attr.Child: InteropTest#test_mod_attr().Child#
  Child.x: Child#x.
  InteropTest.test_mod_attr.ModuleWithUnregisteredTensor: InteropTest#test_mod_attr().ModuleWithUnregisteredTensor#
  ModuleWithUnregisteredTensor.a: ModuleWithUnregisteredTensor#a.
  InteropTest.test_module_with_shared_weights.ModuleWithSharedWeights: InteropTest#test_module_with_shared_weights().ModuleWithSharedWeights#
  InteropTest: InteropTest#
  InteropTest.test_mod_attr: InteropTest#test_mod_attr().
  InteropTest.test_mod_attr.Child.__init__: InteropTest#test_mod_attr().Child#__init__().
  ModuleWithUnregisteredTensor.b: ModuleWithUnregisteredTensor#b.
  ModuleWithUnregisteredTensor.d: ModuleWithUnregisteredTensor#d.
  InteropTest.test_module_with_shared_weights.ModuleWithSharedWeights.__init__: InteropTest#test_module_with_shared_weights().ModuleWithSharedWeights#__init__().
---
# Module: [`test/test_interop.py`](../../../../../raw/code/torchax/test/test_interop.py)

## Classes
### `Child`  ·  implements/extends Module
- def: [`test/test_interop.py:43`](../../../../../raw/code/torchax/test/test_interop.py#L43)
- signature: `class Child(torch.nn.Module):`
- members:
  - `x` — [`L46`](../../../../../raw/code/torchax/test/test_interop.py#L46)
- protocol/private: `__init__`[`L44`](../../../../../raw/code/torchax/test/test_interop.py#L44)
- used by: (2 test-only callers)

### `InteropTest`  ·  implements/extends TestCase
- def: [`test/test_interop.py:38`](../../../../../raw/code/torchax/test/test_interop.py#L38)
- signature: `class InteropTest(unittest.TestCase):`
- members:
  - `fn(x)` — [`L73`](../../../../../raw/code/torchax/test/test_interop.py#L73)
  - `setUp(self)` — [`L39`](../../../../../raw/code/torchax/test/test_interop.py#L39)
  - `test_j2t_autograd_backward(self)` — [`L86`](../../../../../raw/code/torchax/test/test_interop.py#L86)
  - `test_j2t_autograd_forward(self)` — [`L70`](../../../../../raw/code/torchax/test/test_interop.py#L70)
  - `test_mod_attr(self)` — [`L42`](../../../../../raw/code/torchax/test/test_interop.py#L42)
  - `test_module_with_shared_weights(self)` — [`L103`](../../../../../raw/code/torchax/test/test_interop.py#L103)
  - `test_to_jax_device(self)` — [`L144`](../../../../../raw/code/torchax/test/test_interop.py#L144)
  - `test_torch_jax_view_dtype(self)` — [`L170`](../../../../../raw/code/torchax/test/test_interop.py#L170)
- uses (calls/refs, reference-scoped): [`default_env`](../torchax/__init__.md#default_env), [`jax_view`](../torchax/interop.md#jax_view), [`torch_view`](../torchax/interop.md#torch_view), [`params`](../torchax/interop.md#JittableModule.params), [`enable_globally`](../torchax/__init__.md#enable_globally), [`jax_jit`](../torchax/interop.md#jax_jit), [`JittableModule`](../torchax/interop.md#JittableModule), [`functional_call`](../torchax/interop.md#JittableModule.functional_call), [`j2t_autograd`](../torchax/interop.md#j2t_autograd)  (2 test-only)

### `ModuleWithSharedWeights`  ·  implements/extends Module
- def: [`test/test_interop.py:105`](../../../../../raw/code/torchax/test/test_interop.py#L105)
- signature: `class ModuleWithSharedWeights(torch.nn.Module):`
- members:
  - `forward(self, x)` — [`L111`](../../../../../raw/code/torchax/test/test_interop.py#L111)
  - `a` — [`L108`](../../../../../raw/code/torchax/test/test_interop.py#L108)
  - `b` — [`L109`](../../../../../raw/code/torchax/test/test_interop.py#L109)
- protocol/private: `__init__`[`L106`](../../../../../raw/code/torchax/test/test_interop.py#L106)
- used by: (1 test-only callers)

### `ModuleWithUnregisteredTensor`  ·  implements/extends Module
- def: [`test/test_interop.py:48`](../../../../../raw/code/torchax/test/test_interop.py#L48)
- signature: `class ModuleWithUnregisteredTensor(torch.nn.Module):`
- members:
  - `a` — [`L51`](../../../../../raw/code/torchax/test/test_interop.py#L51)
  - `b` — [`L52`](../../../../../raw/code/torchax/test/test_interop.py#L52)
  - `d` — [`L56`](../../../../../raw/code/torchax/test/test_interop.py#L56)
  - `m1` — [`L57`](../../../../../raw/code/torchax/test/test_interop.py#L57)
- protocol/private: `__init__`[`L49`](../../../../../raw/code/torchax/test/test_interop.py#L49)
- uses (calls/refs, reference-scoped): [`extract_all_buffers`](../torchax/interop.md#extract_all_buffers), [`set_all_buffers`](../torchax/interop.md#set_all_buffers)  (2 test-only)

## Functions
- `is_tpu_available()` — [`L26`](../../../../../raw/code/torchax/test/test_interop.py#L26) — Checks if any TPU devices are available to JAX.

