---
title: 'Module: easydel/data/transforms/field_ops.py'
type: catalog
provenance: extracted
module: easydel/data/transforms/field_ops.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.transforms.field_ops`/
symbols:
  CombineFields.__call__: CombineFields#__call__().
  ExtractField.__call__: ExtractField#__call__().
  RenameFields: RenameFields#
  AddField.__call__: AddField#__call__().
  AddField._value: AddField#_value.
  RenameFields.__call__: RenameFields#__call__().
  SelectFields.__call__: SelectFields#__call__().
  DropFields.__call__: DropFields#__call__().
  ExtractField.__repr__: ExtractField#__repr__().
  CombineFields.__repr__: CombineFields#__repr__().
  AddField.__repr__: AddField#__repr__().
  SelectFields: SelectFields#
  DropFields: DropFields#
  ExtractField: ExtractField#
  CombineFields: CombineFields#
  AddField: AddField#
  AddField._field: AddField#_field.
  CombineFields._source_fields: CombineFields#_source_fields.
  RenameFields.__repr__: RenameFields#__repr__().
  SelectFields.__repr__: SelectFields#__repr__().
  DropFields.__repr__: DropFields#__repr__().
  AddField.__init__: AddField#__init__().
  RenameFields._mapping: RenameFields#_mapping.
  SelectFields._fields: SelectFields#_fields.
  DropFields._fields: DropFields#_fields.
  ExtractField._source_path: ExtractField#_source_path.
  ExtractField._target_field: ExtractField#_target_field.
  CombineFields._target_field: CombineFields#_target_field.
  CombineFields._combiner: CombineFields#_combiner.
  ExtractField._default: ExtractField#_default.
  ExtractField._extract_path: ExtractField#_extract_path().
  CombineFields._separator: CombineFields#_separator.
  CombineFields._drop_sources: CombineFields#_drop_sources.
  RenameFields.__init__: RenameFields#__init__().
  SelectFields.__init__: SelectFields#__init__().
  DropFields.__init__: DropFields#__init__().
  ExtractField.__init__: ExtractField#__init__().
  CombineFields.__init__: CombineFields#__init__().
---
# Module: [`easydel/data/transforms/field_ops.py`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py)

## Classes
### `AddField`  ·  implements/extends Transform
- def: [`easydel/data/transforms/field_ops.py:278`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L278)
- doc: Add a new field with a constant value or computed value.
- signature: `class AddField(Transform):`
- members:
  - `__call__(self, example: Example)` — [`L312`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L312) — Add the new field to the example.
  - `__init__(self, field: str, value: tp.Any | tp.Callable[[Example], tp.Any])` — [`L297`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L297) — Initialize AddField.
- protocol/private: `__repr__`[`L330`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L330), `_field`[`L309`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L309), `_value`[`L310`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L310)
- uses (calls/refs, reference-scoped): [`Transform`](base.md#Transform), [`Example`](base.md#Example)
- used by: [`Transform`](base.md#Transform)

### `CombineFields`  ·  implements/extends Transform
- def: [`easydel/data/transforms/field_ops.py:207`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L207)
- doc: Combine multiple fields into one.
- signature: `class CombineFields(Transform):`
- members:
  - `__call__(self, example: Example)` — [`L248`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L248) — Combine the specified fields into a single target field.
  - `__init__(self, source_fields: list[str], target_field: str, combiner: tp.Callable[[list[tp.Any]], tp.Any] | None = None, separator: str = " ", drop_sources: bool = False)` — [`L224`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L224) — Initialize CombineFields.
- protocol/private: `__repr__`[`L274`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L274), `_combiner`[`L244`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L244), `_drop_sources`[`L246`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L246), `_separator`[`L245`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L245), `_source_fields`[`L242`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L242), `_target_field`[`L243`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L243)
- uses (calls/refs, reference-scoped): [`Transform`](base.md#Transform), [`Example`](base.md#Example)
- used by: [`Transform`](base.md#Transform)

### `DropFields`  ·  implements/extends Transform
- def: [`easydel/data/transforms/field_ops.py:101`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L101)
- doc: Drop specified fields from examples.
- signature: `class DropFields(Transform):`
- members:
  - `__call__(self, example: Example)` — [`L118`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L118) — Remove the specified fields from the example.
  - `__init__(self, fields: list[str])` — [`L110`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L110) — Initialize DropFields.
- protocol/private: `__repr__`[`L129`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L129), `_fields`[`L116`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L116)
- uses (calls/refs, reference-scoped): [`Transform`](base.md#Transform), [`Example`](base.md#Example)
- used by: [`Transform`](base.md#Transform)

### `ExtractField`  ·  implements/extends Transform
- def: [`easydel/data/transforms/field_ops.py:133`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L133)
- doc: Extract a nested field to a new top-level field.
- signature: `class ExtractField(Transform):`
- members:
  - `__call__(self, example: Example)` — [`L166`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L166) — Extract the nested value and add it as a new top-level field.
  - `__init__(self, source_path: str, target_field: str, default: tp.Any = None)` — [`L149`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L149) — Initialize ExtractField.
  - `_extract_path(self, data: tp.Any, path: str)` — [`L180`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L180) — Extract value from nested path like 'a.b[0].c'.
- protocol/private: `__repr__`[`L203`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L203), `_default`[`L164`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L164), `_source_path`[`L162`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L162), `_target_field`[`L163`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L163)
- uses (calls/refs, reference-scoped): [`Transform`](base.md#Transform), [`Example`](base.md#Example)
- used by: [`Transform`](base.md#Transform)

### `RenameFields`  ·  implements/extends Transform
- def: [`easydel/data/transforms/field_ops.py:33`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L33)
- doc: Rename fields in examples.
- signature: `class RenameFields(Transform):`
- members:
  - `__call__(self, example: Example)` — [`L50`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L50) — Rename fields according to the mapping.
  - `__init__(self, mapping: dict[str, str])` — [`L42`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L42) — Initialize RenameFields.
- protocol/private: `__repr__`[`L65`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L65), `_mapping`[`L48`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L48)
- uses (calls/refs, reference-scoped): [`Transform`](base.md#Transform), [`Example`](base.md#Example)
- used by: [`Transform`](base.md#Transform), [`build_sharded_source`](../../infra/elarge/builders.md#build_sharded_source)

### `SelectFields`  ·  implements/extends Transform
- def: [`easydel/data/transforms/field_ops.py:69`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L69)
- doc: Select only specified fields, dropping all others.
- signature: `class SelectFields(Transform):`
- members:
  - `__call__(self, example: Example)` — [`L86`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L86) — Keep only the specified fields, dropping all others.
  - `__init__(self, fields: list[str])` — [`L78`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L78) — Initialize SelectFields.
- protocol/private: `__repr__`[`L97`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L97), `_fields`[`L84`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/field_ops.py#L84)
- uses (calls/refs, reference-scoped): [`Transform`](base.md#Transform), [`Example`](base.md#Example)
- used by: [`Transform`](base.md#Transform)

