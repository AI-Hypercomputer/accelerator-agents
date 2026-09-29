---
title: 'Module: test/test_jittable_module.py'
type: catalog
provenance: extracted
module: test/test_jittable_module.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.test_jittable_module`/
symbols:
  JittableModuleTest.test_functional_call_callable: JittableModuleTest#test_functional_call_callable().
  JittableModuleTest.test_isinstance_does_not_mix: JittableModuleTest#test_isinstance_does_not_mix().
  MyAwesomeModel: MyAwesomeModel#
  JittableModuleTest.test_isinstance_works: JittableModuleTest#test_isinstance_works().
  EvenMoreAwesomeModel: EvenMoreAwesomeModel#
  JittableModuleTest.outer_function: JittableModuleTest#outer_function().
  JittableModuleTest: JittableModuleTest#
---
# Module: [`test/test_jittable_module.py`](../../../../../raw/code/torchax/test/test_jittable_module.py)

## Classes
### `EvenMoreAwesomeModel`  ·  implements/extends Module
- def: [`test/test_jittable_module.py:26`](../../../../../raw/code/torchax/test/test_jittable_module.py#L26)
- signature: `class EvenMoreAwesomeModel(torch.nn.Module):`
- used by: (1 test-only callers)

### `JittableModuleTest`  ·  implements/extends TestCase
- def: [`test/test_jittable_module.py:30`](../../../../../raw/code/torchax/test/test_jittable_module.py#L30)
- signature: `class JittableModuleTest(unittest.TestCase):`
- members:
  - `outer_function(model, x)` — [`L51`](../../../../../raw/code/torchax/test/test_jittable_module.py#L51)
  - `test_functional_call_callable(self)` — [`L50`](../../../../../raw/code/torchax/test/test_jittable_module.py#L50)
  - `test_isinstance_does_not_mix(self)` — [`L39`](../../../../../raw/code/torchax/test/test_jittable_module.py#L39)
  - `test_isinstance_works(self)` — [`L31`](../../../../../raw/code/torchax/test/test_jittable_module.py#L31)
- uses (calls/refs, reference-scoped): [`params`](../torchax/interop.md#JittableModule.params), [`JittableModule`](../torchax/interop.md#JittableModule), [`functional_call`](../torchax/interop.md#JittableModule.functional_call)  (2 test-only)

### `MyAwesomeModel`  ·  implements/extends Module
- def: [`test/test_jittable_module.py:22`](../../../../../raw/code/torchax/test/test_jittable_module.py#L22)
- signature: `class MyAwesomeModel(torch.nn.Module):`
- used by: (3 test-only callers)

