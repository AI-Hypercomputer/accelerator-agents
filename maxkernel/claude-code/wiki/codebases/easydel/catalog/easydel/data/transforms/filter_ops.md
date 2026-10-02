---
title: 'Module: easydel/data/transforms/filter_ops.py'
type: catalog
provenance: extracted
module: easydel/data/transforms/filter_ops.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.transforms.filter_ops`/Filter
symbols:
  FilterByField.__call__: ByField#__call__().
  FilterTransform.__call__: Transform#__call__().
  FilterByField.__repr__: ByField#__repr__().
  FilterNonEmpty.__call__: NonEmpty#__call__().
  FilterTransform: Transform#
  FilterByField: ByField#
  FilterNonEmpty: NonEmpty#
  FilterByField._field: ByField#_field.
  FilterTransform.__init__: Transform#__init__().
  FilterTransform.__repr__: Transform#__repr__().
  FilterNonEmpty.__repr__: NonEmpty#__repr__().
  FilterTransform._predicate: Transform#_predicate.
  FilterByField._predicate: ByField#_predicate.
  FilterNonEmpty._fields: NonEmpty#_fields.
  FilterTransform.is_filter: Transform#is_filter().
  FilterByField.__init__: ByField#__init__().
  FilterByField.is_filter: ByField#is_filter().
  FilterNonEmpty.__init__: NonEmpty#__init__().
  FilterNonEmpty.is_filter: NonEmpty#is_filter().
---
# Module: [`easydel/data/transforms/filter_ops.py`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py)

## Classes
### `FilterByField`  ·  implements/extends Transform
- def: [`easydel/data/transforms/filter_ops.py:69`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L69)
- doc: Filter examples based on a specific field value.
- signature: `class FilterByField(Transform):`
- members:
  - `__call__(self, example: Example)` — [`L92`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L92) — Return example if the specified field's value matches the predicate.
  - `__init__(self, field: str, predicate: tp.Callable[[tp.Any], bool])` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L82) — Initialize FilterByField.
  - `is_filter(self)` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L108)
- protocol/private: `__repr__`[`L111`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L111), `_field`[`L89`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L89), `_predicate`[`L90`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L90)
- uses (calls/refs, reference-scoped): [`Transform`](base.md#Transform), [`Example`](base.md#Example)
- used by: [`Transform`](base.md#Transform), [`is_filter`](base.md#Transform.is_filter)

### `FilterNonEmpty`  ·  implements/extends Transform
- def: [`easydel/data/transforms/filter_ops.py:116`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L116)
- doc: Filter out examples where specified fields are empty.
- signature: `class FilterNonEmpty(Transform):`
- members:
  - `__call__(self, example: Example)` — [`L136`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L136) — Return example if all specified fields are non-empty.
  - `__init__(self, fields: list[str])` — [`L128`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L128) — Initialize FilterNonEmpty.
  - `is_filter(self)` — [`L154`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L154)
- protocol/private: `__repr__`[`L157`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L157), `_fields`[`L134`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L134)
- uses (calls/refs, reference-scoped): [`Transform`](base.md#Transform), [`Example`](base.md#Example)
- used by: [`Transform`](base.md#Transform), [`is_filter`](base.md#Transform.is_filter)

### `FilterTransform`  ·  implements/extends Transform
- def: [`easydel/data/transforms/filter_ops.py:30`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L30)
- doc: Filter examples based on a predicate function.
- signature: `class FilterTransform(Transform):`
- members:
  - `__call__(self, example: Example)` — [`L49`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L49) — Return example if predicate is True, else None.
  - `__init__(self, predicate: tp.Callable[[Example], bool])` — [`L41`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L41) — Initialize FilterTransform.
  - `is_filter(self)` — [`L61`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L61)
- protocol/private: `__repr__`[`L64`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L64), `_predicate`[`L47`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/filter_ops.py#L47)
- uses (calls/refs, reference-scoped): [`Transform`](base.md#Transform), [`Example`](base.md#Example)
- used by: [`Transform`](base.md#Transform), [`is_filter`](base.md#Transform.is_filter)

