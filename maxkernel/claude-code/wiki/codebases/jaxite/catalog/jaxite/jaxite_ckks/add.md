---
title: 'Module: jaxite/jaxite_ckks/add.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/add.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.add`/
symbols:
  AddBase: AddBase#
  AddBase.add: AddBase#add().
  AddSimple.add: AddSimple#add().
  AddModularBarrett.add: AddModularBarrett#add().
  AddSimple: AddSimple#
  AddModularBarrett: AddModularBarrett#
  AddModularSubtract: AddModularSubtract#
  AddModularBarrett.barrett_constants: AddModularBarrett#barrett_constants.
  AddModularBarrett.__init__: AddModularBarrett#__init__().
  AddModularBarrett.tree_flatten: AddModularBarrett#tree_flatten().
  AddModularSubtract.add: AddModularSubtract#add().
  AddModularSubtract.tree_flatten: AddModularSubtract#tree_flatten().
  AddModularSubtract.moduli: AddModularSubtract#moduli.
  ABC: ABC.
  abstractmethod: abstractmethod.
  AddSimple.__init__: AddSimple#__init__().
  AddSimple.tree_flatten: AddSimple#tree_flatten().
  AddSimple.tree_unflatten: AddSimple#tree_unflatten().
  AddModularBarrett.tree_unflatten: AddModularBarrett#tree_unflatten().
  AddModularSubtract.__init__: AddModularSubtract#__init__().
  AddModularSubtract.tree_unflatten: AddModularSubtract#tree_unflatten().
---
# Module: [`jaxite/jaxite_ckks/add.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py)

## Classes
### `AddBase`  ·  implements/extends ABC
- def: [`jaxite/jaxite_ckks/add.py:13`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L13)
- doc: Abstract base class for addition kernels.
- signature: `class AddBase(ABC):`
- members:
  - `add(self, a: jnp.ndarray, b: jnp.ndarray)` — [`L17`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L17) — Performs addition.
- uses (calls/refs, reference-scoped): [`add`](add.md#AddModularBarrett.add), [`add`](add.md#AddSimple.add), [`AddModularBarrett`](add.md#AddModularBarrett), [`AddModularSubtract`](add.md#AddModularSubtract), [`AddSimple`](add.md#AddSimple), [`add`](add.md#AddModularSubtract.add), [`ABC`](add.md#ABC), [`abstractmethod`](add.md#abstractmethod)
- used by: [`AddModularBarrett`](add.md#AddModularBarrett), [`AddModularSubtract`](add.md#AddModularSubtract), [`AddSimple`](add.md#AddSimple)

### `AddModularBarrett`  ·  implements/extends AddBase
- def: [`jaxite/jaxite_ckks/add.py:44`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L44)
- doc: Kernel for modular addition using Barrett reduction.
- signature: `class AddModularBarrett(AddBase):`
- members:
  - `add(self, a: jnp.ndarray, b: jnp.ndarray)` — [`L60`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L60)
  - `tree_flatten(self)` — [`L66`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L66)
  - `tree_unflatten(cls, _, children)` — [`L70`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L70)
  - `barrett_constants` — [`L58`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L58)
- protocol/private: `__init__`[`L57`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L57)
- uses (calls/refs, reference-scoped): [`modular_reduction`](barrett.md#modular_reduction), [`AddBase`](add.md#AddBase), [`BarrettConstants`](barrett.md#BarrettConstants)
- used by: [`_get_kernel`](add_test.md#_get_kernel), [`AddBase`](add.md#AddBase), [`add`](add.md#AddBase.add)

### `AddModularSubtract`  ·  implements/extends AddBase
- def: [`jaxite/jaxite_ckks/add.py:76`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L76)
- doc: Kernel for modular addition using simple subtraction.
- signature: `class AddModularSubtract(AddBase):`
- members:
  - `add(self, a: jnp.ndarray, b: jnp.ndarray)` — [`L86`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L86)
  - `tree_flatten(self)` — [`L94`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L94)
  - `tree_unflatten(cls, _, children)` — [`L98`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L98)
  - `moduli` — [`L84`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L84)
- protocol/private: `__init__`[`L83`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L83)
- uses (calls/refs, reference-scoped): [`AddBase`](add.md#AddBase)
- used by: [`_get_kernel`](add_test.md#_get_kernel), [`AddBase`](add.md#AddBase), [`add`](add.md#AddBase.add)

### `AddSimple`  ·  implements/extends AddBase
- def: [`jaxite/jaxite_ckks/add.py:22`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L22)
- doc: Kernel for raw addition without reduction.
- signature: `class AddSimple(AddBase):`
- members:
  - `add(self, a: jnp.ndarray, b: jnp.ndarray)` — [`L32`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L32)
  - `tree_flatten(self)` — [`L35`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L35)
  - `tree_unflatten(cls, _, children)` — [`L39`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L39)
- protocol/private: `__init__`[`L29`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L29)
- uses (calls/refs, reference-scoped): [`AddBase`](add.md#AddBase)
- used by: [`test_full_pipeline_add`](add_test.md#AddTest.test_full_pipeline_add), [`_get_kernel`](add_test.md#_get_kernel), [`AddBase`](add.md#AddBase), [`add`](add.md#AddBase.add), [`test_add_modular_barrett`](add_test.md#AddTest.test_add_modular_barrett), [`test_add_modular_subtract`](add_test.md#AddTest.test_add_modular_subtract), [`test_add_modular_subtract_hypothesis`](add_test.md#AddTest.test_add_modular_subtract_hypothesis), [`test_add_simple`](add_test.md#AddTest.test_add_simple)

## Module values
- `ABC` — [`L9`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L9)
- `abstractmethod` — [`L10`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add.py#L10)

