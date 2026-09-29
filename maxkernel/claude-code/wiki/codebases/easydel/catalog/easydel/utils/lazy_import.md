---
title: 'Module: easydel/utils/lazy_import.py'
type: catalog
provenance: extracted
module: easydel/utils/lazy_import.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.utils.lazy_import`/
symbols:
  LazyModule._import_structure: LazyModule#_import_structure.
  LazyModule.__all__: LazyModule#__all__.
  LazyModule.__getattr__.Placeholder._backends: LazyModule#__getattr__().Placeholder#_backends.
  LazyModule.__getattr__.Placeholder: LazyModule#__getattr__().Placeholder#
  LazyModule._class_to_module: LazyModule#_class_to_module.
  LazyModule.__getattr__: LazyModule#__getattr__().
  LazyModule.__reduce__: LazyModule#__reduce__().
  is_package_available: is_package_available().
  LazyModule._modules: LazyModule#_modules.
  IMPORT_STRUCTURE_T: IMPORT_STRUCTURE_T.
  LazyModule: LazyModule#
  LazyModule._objects: LazyModule#_objects.
  LazyModule.__init__: LazyModule#__init__().
  LazyModule.__dir__: LazyModule#__dir__().
  LazyModule._object_missing_backend: LazyModule#_object_missing_backend.
  LazyModule._name: LazyModule#_name.
  LazyModule._get_module: LazyModule#_get_module().
  BACKENDS_T: BACKENDS_T.
  DummyObject: DummyObject#
  DummyObject.__getattribute__: DummyObject#__getattribute__().
---
# Module: [`easydel/utils/lazy_import.py`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py)

## Classes
### `DummyObject`
- def: [`easydel/utils/lazy_import.py:135`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L135)
- protocol/private: `__getattribute__`[`L141`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L141)
- used by: [`Placeholder`](lazy_import.md#LazyModule.__getattr__.Placeholder)

### `LazyModule`  ·  implements/extends ModuleType
- def: [`easydel/utils/lazy_import.py:25`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L25)
- doc: Module subclass that defers sub-module imports until first access.
- signature: `class LazyModule(ModuleType):`
- protocol/private: `__all__`[`L53`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L53), `__dir__`[`L91`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L91), `__getattr__`[`L98`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L98), `__init__`[`L39`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L39), `__reduce__`[`L131`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L131), `_class_to_module`[`L52`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L52), `_get_module`[`L122`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L122), `_import_structure`[`L73`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L73), `_modules`[`L51`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L51), `_name`[`L72`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L72), `_object_missing_backend`[`L49`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L49), `_objects`[`L71`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L71)
- uses (calls/refs, reference-scoped): [`IMPORT_STRUCTURE_T`](lazy_import.md#IMPORT_STRUCTURE_T)
- used by: [`_distributed_msg`](../__init__.md#_distributed_msg), [`__version__`](../__init__.md#__version__), [`_backends`](lazy_import.md#LazyModule.__getattr__.Placeholder._backends)

### `Placeholder`
- def: [`easydel/utils/lazy_import.py:104`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L104)
- signature: `class Placeholder(metaclass=DummyObject):`
- protocol/private: `_backends`[`L105`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L105)
- uses (calls/refs, reference-scoped): [`_class_to_module`](lazy_import.md#LazyModule._class_to_module), [`_modules`](lazy_import.md#LazyModule._modules), [`_get_module`](lazy_import.md#LazyModule._get_module), [`DummyObject`](lazy_import.md#DummyObject)

## Functions
- `is_package_available(package_name: str)` — [`L146`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L146) — Checks if a package is available in the current Python environment.

## Module values
- `BACKENDS_T` — [`L21`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L21)
- `IMPORT_STRUCTURE_T` — [`L22`](../../../../../../raw/code/EasyDeL/easydel/utils/lazy_import.py#L22)

