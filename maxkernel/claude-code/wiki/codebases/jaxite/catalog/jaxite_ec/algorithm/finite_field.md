---
title: 'Module: jaxite_ec/algorithm/finite_field.py'
type: catalog
provenance: extracted
module: jaxite_ec/algorithm/finite_field.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite_ec.algorithm.finite_field`/
symbols:
  FiniteFieldElement.prime: FiniteFieldElement#prime.
  FiniteFieldElement.value: FiniteFieldElement#value.
  BigInt: BigInt.
  FiniteFieldElementMontgomery.value: FiniteFieldElementMontgomery#value.
  FiniteFieldElementMontgomery.__truediv__: FiniteFieldElementMontgomery#__truediv__().
  FiniteFieldElementMontgomery.montgomery_reduction: FiniteFieldElementMontgomery#montgomery_reduction().
  FiniteFieldElementMontgomery.copy: FiniteFieldElementMontgomery#copy().
  FiniteFieldElement: FiniteFieldElement#
  FiniteFieldElementBarrett.copy: FiniteFieldElementBarrett#copy().
  FiniteFieldElementBarrett.barrett_reduction: FiniteFieldElementBarrett#barrett_reduction().
  FiniteFieldElement.copy: FiniteFieldElement#copy().
  FiniteFieldElementBarrett.__mul__: FiniteFieldElementBarrett#__mul__().
  FiniteFieldElementBarrett.__truediv__: FiniteFieldElementBarrett#__truediv__().
  FiniteFieldElementMontgomery.montgomeryize: FiniteFieldElementMontgomery#montgomeryize().
  FiniteFieldElementMontgomery.change_montgomery_form: FiniteFieldElementMontgomery#change_montgomery_form().
  FiniteFieldElementMontgomery.__mul__: FiniteFieldElementMontgomery#__mul__().
  FiniteFieldElementBarrett.two_k: FiniteFieldElementBarrett#two_k.
  FiniteFieldElementMontgomery.k: FiniteFieldElementMontgomery#k.
  FiniteFieldElementMontgomery.r: FiniteFieldElementMontgomery#r.
  FiniteFieldElementMontgomery.n_prime: FiniteFieldElementMontgomery#n_prime.
  FiniteFieldElement.set_value: FiniteFieldElement#set_value().
  FiniteFieldElement.__add__: FiniteFieldElement#__add__().
  FiniteFieldElement.__sub__: FiniteFieldElement#__sub__().
  FiniteFieldElement.__mul__: FiniteFieldElement#__mul__().
  FiniteFieldElement.__truediv__: FiniteFieldElement#__truediv__().
  FiniteFieldElement.__pow__: FiniteFieldElement#__pow__().
  FiniteFieldElementBarrett.__add__: FiniteFieldElementBarrett#__add__().
  FiniteFieldElementBarrett.__sub__: FiniteFieldElementBarrett#__sub__().
  FiniteFieldElementMontgomery.__add__: FiniteFieldElementMontgomery#__add__().
  FiniteFieldElementMontgomery.__sub__: FiniteFieldElementMontgomery#__sub__().
  FiniteFieldElementBarrett.mu: FiniteFieldElementBarrett#mu.
  FiniteFieldElementMontgomery.r_inverse: FiniteFieldElementMontgomery#r_inverse.
  FiniteFieldElement.__eq__: FiniteFieldElement#__eq__().
  FiniteFieldElement.__str__: FiniteFieldElement#__str__().
  FiniteFieldElement.__repr__: FiniteFieldElement#__repr__().
  FiniteFieldElementBarrett.__init__: FiniteFieldElementBarrett#__init__().
  FiniteFieldElementMontgomery.__init__: FiniteFieldElementMontgomery#__init__().
  FiniteFieldElementMontgomery.one_bar: FiniteFieldElementMontgomery#one_bar.
  FiniteFieldElement.__init__: FiniteFieldElement#__init__().
  FiniteFieldElementMontgomery.r_mask: FiniteFieldElementMontgomery#r_mask.
  FiniteFieldElementMontgomery.de_montgomeryize: FiniteFieldElementMontgomery#de_montgomeryize().
  FiniteFieldElementMontgomery.montgomeryized: FiniteFieldElementMontgomery#montgomeryized.
  FiniteFieldElement.get_value: FiniteFieldElement#get_value().
  FiniteFieldElement.get_prime: FiniteFieldElement#get_prime().
  FiniteFieldElement.__hex__: FiniteFieldElement#__hex__().
  FiniteFieldElement.hex_value_str: FiniteFieldElement#hex_value_str().
  FiniteFieldElementBarrett: FiniteFieldElementBarrett#
  FiniteFieldElementMontgomery: FiniteFieldElementMontgomery#
---
# Module: [`jaxite_ec/algorithm/finite_field.py`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py)

## Classes
### `FiniteFieldElement`
- def: [`jaxite_ec/algorithm/finite_field.py:11`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L11)
- doc: Finite field element for elliptic curve cryptography.
- signature: `class FiniteFieldElement:`
- members:
  - `copy(self, value=None, transform=False, reduction=False)` — [`L46`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L46) — Create a deep copy of the current finite field element. — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
  - `get_prime(self)` — [`L43`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L43)
  - `get_value(self)` — [`L40`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L40)
  - `hex_value_str(self)` — [`L114`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L114)
  - `set_value(self, value)` — [`L28`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L28) — Set the value of the finite field element, with validation.
  - `prime` — [`L21`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L21) — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
  - `value` — [`L16`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L16) — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
- protocol/private: `__add__`[`L69`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L69), `__eq__`[`L99`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L99), `__hex__`[`L111`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L111), `__init__`[`L14`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L14), `__mul__`[`L81`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L81), `__pow__`[`L95`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L95), `__repr__`[`L105`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L105), `__str__`[`L102`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L102), `__sub__`[`L75`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L75), `__truediv__`[`L87`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L87)
- uses (calls/refs, reference-scoped): [`BigInt`](finite_field.md#BigInt), [`copy`](finite_field.md#FiniteFieldElementMontgomery.copy), [`copy`](finite_field.md#FiniteFieldElementBarrett.copy), [`FiniteFieldElementBarrett`](finite_field.md#FiniteFieldElementBarrett), [`FiniteFieldElementMontgomery`](finite_field.md#FiniteFieldElementMontgomery)
- used by: [`__truediv__`](finite_field.md#FiniteFieldElementMontgomery.__truediv__), [`montgomery_reduction`](finite_field.md#FiniteFieldElementMontgomery.montgomery_reduction), [`copy`](finite_field.md#FiniteFieldElementBarrett.copy), [`barrett_reduction`](finite_field.md#FiniteFieldElementBarrett.barrett_reduction), [`__mul__`](finite_field.md#FiniteFieldElementBarrett.__mul__), [`__mul__`](finite_field.md#FiniteFieldElementMontgomery.__mul__), [`__truediv__`](finite_field.md#FiniteFieldElementBarrett.__truediv__), [`montgomeryize`](finite_field.md#FiniteFieldElementMontgomery.montgomeryize), [`n_prime`](finite_field.md#FiniteFieldElementMontgomery.n_prime), [`__add__`](finite_field.md#FiniteFieldElementBarrett.__add__), [`__add__`](finite_field.md#FiniteFieldElementMontgomery.__add__), [`__sub__`](finite_field.md#FiniteFieldElementBarrett.__sub__), [`__sub__`](finite_field.md#FiniteFieldElementMontgomery.__sub__), [`r_inverse`](finite_field.md#FiniteFieldElementMontgomery.r_inverse), [`__init__`](finite_field.md#FiniteFieldElementBarrett.__init__), [`__init__`](finite_field.md#FiniteFieldElementMontgomery.__init__), [`FiniteFieldElementBarrett`](finite_field.md#FiniteFieldElementBarrett), [`FiniteFieldElementMontgomery`](finite_field.md#FiniteFieldElementMontgomery)

### `FiniteFieldElementBarrett`  ·  implements/extends FiniteFieldElement
- def: [`jaxite_ec/algorithm/finite_field.py:118`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L118)
- doc: Finite field element for elliptic curve cryptography using Barrett reduction.
- signature: `class FiniteFieldElementBarrett(FiniteFieldElement):`
- members:
  - `barrett_reduction(self, x)` — [`L132`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L132) — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
  - `copy(self, value=None, transform=False, reduction=False)` — [`L181`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L181) — Create a deep copy of the current finite field element. — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
  - `mu` — [`L130`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L130) — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
  - `two_k` — [`L125`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L125) — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
- protocol/private: `__add__`[`L141`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L141), `__init__`[`L121`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L121), `__mul__`[`L162`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L162), `__sub__`[`L151`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L151), `__truediv__`[`L171`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L171)
- uses (calls/refs, reference-scoped): [`prime`](finite_field.md#FiniteFieldElement.prime), [`value`](finite_field.md#FiniteFieldElement.value), [`BigInt`](finite_field.md#BigInt), [`FiniteFieldElement`](finite_field.md#FiniteFieldElement), [`__init__`](finite_field.md#FiniteFieldElement.__init__)
- used by: [`FiniteFieldElement`](finite_field.md#FiniteFieldElement), [`copy`](finite_field.md#FiniteFieldElement.copy)

### `FiniteFieldElementMontgomery`  ·  implements/extends FiniteFieldElement
- def: [`jaxite_ec/algorithm/finite_field.py:193`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L193)
- doc: Finite field element for elliptic curve cryptography using Montgomery reduction.
- signature: `class FiniteFieldElementMontgomery(FiniteFieldElement):`
- members:
  - `change_montgomery_form(self)` — [`L230`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L230) — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
  - `copy(self, value=None, transform=False, reduction=False)` — [`L291`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L291) — Create a deep copy of the current finite field element. — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
  - `de_montgomeryize(self, x_bar)` — [`L226`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L226)
  - `montgomery_reduction(self, x)` — [`L215`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L215) — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
  - `montgomeryize(self, x)` — [`L222`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L222) — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
  - `k` — [`L200`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L200) — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
  - `montgomeryized` — [`L212`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L212)
  - `n_prime` — [`L209`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L209) — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
  - `one_bar` — [`L213`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L213) — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
  - `r` — [`L206`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L206) — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
  - `r_inverse` — [`L208`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L208) — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
  - `r_mask` — [`L210`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L210)
  - `value` — [`L211`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L211) — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)
- protocol/private: `__add__`[`L239`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L239), `__init__`[`L196`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L196), `__mul__`[`L262`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L262), `__sub__`[`L250`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L250), `__truediv__`[`L272`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L272)
- uses (calls/refs, reference-scoped): [`prime`](finite_field.md#FiniteFieldElement.prime), [`BigInt`](finite_field.md#BigInt), [`FiniteFieldElement`](finite_field.md#FiniteFieldElement), [`__init__`](finite_field.md#FiniteFieldElement.__init__)
- used by: [`FiniteFieldElement`](finite_field.md#FiniteFieldElement), [`copy`](finite_field.md#FiniteFieldElement.copy)

## Module values
- `BigInt` — [`L8`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/finite_field.py#L8) — documented in [jaxite_ec-algorithm-finite_field](../../../concepts/jaxite_ec-algorithm-finite_field.md)

