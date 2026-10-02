---
title: 'Module: simply/utils/registry_test.py'
type: catalog
provenance: extracted
module: simply/utils/registry_test.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.registry_test`/
symbols:
  RegistryTest.test_dummy_registry: RegistryTest#test_dummy_registry().
  RegistryTest._dummy_fn: RegistryTest#_dummy_fn().
  RegistryTest.test_dummy_registry._DummyClass: RegistryTest#test_dummy_registry()._DummyClass#
  DummyClassRegistry: DummyClassRegistry#
  DummyClassRegistry.namespace: DummyClassRegistry#namespace.
  RegistryTest: RegistryTest#
---
# Module: [`simply/utils/registry_test.py`](../../../../../../raw/code/simply/simply/utils/registry_test.py)

## Classes
### `DummyClassRegistry`  ·  implements/extends RootRegistry
- def: [`simply/utils/registry_test.py:20`](../../../../../../raw/code/simply/simply/utils/registry_test.py#L20)
- doc: Registry for dummy class.
- signature: `class DummyClassRegistry(registry.RootRegistry):`
- members:
  - `namespace` — [`L23`](../../../../../../raw/code/simply/simply/utils/registry_test.py#L23)
- uses (calls/refs, reference-scoped): [`RootRegistry`](registry.md#RootRegistry)
- used by: [`RootRegistry`](registry.md#RootRegistry), [`test_dummy_registry`](registry_test.md#RegistryTest.test_dummy_registry), [`_DummyClass`](registry_test.md#RegistryTest.test_dummy_registry._DummyClass)

### `RegistryTest`  ·  implements/extends TestCase
- def: [`simply/utils/registry_test.py:26`](../../../../../../raw/code/simply/simply/utils/registry_test.py#L26)
- signature: `class RegistryTest(absltest.TestCase):`
- members:
  - `test_dummy_registry(self)` — [`L28`](../../../../../../raw/code/simply/simply/utils/registry_test.py#L28)
- protocol/private: `_dummy_fn`[`L34`](../../../../../../raw/code/simply/simply/utils/registry_test.py#L34)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`RootRegistry`](registry.md#RootRegistry), [`get`](registry.md#RootRegistry.get), [`FunctionRegistry`](registry.md#FunctionRegistry), [`registry`](registry.md#RootRegistry.registry), [`_DummyClass`](registry_test.md#RegistryTest.test_dummy_registry._DummyClass), [`DummyClassRegistry`](registry_test.md#DummyClassRegistry), [`reset`](registry.md#RootRegistry.reset)

### `_DummyClass`
- def: [`simply/utils/registry_test.py:38`](../../../../../../raw/code/simply/simply/utils/registry_test.py#L38)
- signature: `class _DummyClass:`
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`DummyClassRegistry`](registry_test.md#DummyClassRegistry)
- used by: [`test_dummy_registry`](registry_test.md#RegistryTest.test_dummy_registry)

