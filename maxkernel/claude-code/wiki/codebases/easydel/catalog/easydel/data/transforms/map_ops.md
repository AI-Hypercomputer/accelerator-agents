---
title: 'Module: easydel/data/transforms/map_ops.py'
type: catalog
provenance: extracted
module: easydel/data/transforms/map_ops.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.transforms.map_ops`/Map
symbols:
  MapField.__call__: Field#__call__().
  MapField.__repr__: Field#__repr__().
  MapTransform: Transform#
  MapTransform.__call__: Transform#__call__().
  MapField._field: Field#_field.
  MapField: Field#
  MapField._output_field: Field#_output_field.
  MapTransform.__init__: Transform#__init__().
  MapTransform.__repr__: Transform#__repr__().
  MapTransform._fn: Transform#_fn.
  MapField._fn: Field#_fn.
  MapField.__init__: Field#__init__().
---
# Module: [`easydel/data/transforms/map_ops.py`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/map_ops.py)

## Classes
### `MapField`  ·  implements/extends Transform
- def: [`easydel/data/transforms/map_ops.py:62`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/map_ops.py#L62)
- doc: Apply a function to a specific field.
- signature: `class MapField(Transform):`
- members:
  - `__call__(self, example: Example)` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/map_ops.py#L94) — Apply the function to the specified field.
  - `__init__(self, field: str, fn: tp.Callable[[tp.Any], tp.Any], output_field: str | None = None)` — [`L76`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/map_ops.py#L76) — Initialize MapField.
- protocol/private: `__repr__`[`L112`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/map_ops.py#L112), `_field`[`L90`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/map_ops.py#L90), `_fn`[`L91`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/map_ops.py#L91), `_output_field`[`L92`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/map_ops.py#L92)
- uses (calls/refs, reference-scoped): [`Transform`](base.md#Transform), [`Example`](base.md#Example)
- used by: [`Transform`](base.md#Transform)

### `MapTransform`  ·  implements/extends Transform
- def: [`easydel/data/transforms/map_ops.py:29`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/map_ops.py#L29)
- doc: Apply a function to each example.
- signature: `class MapTransform(Transform):`
- members:
  - `__call__(self, example: Example)` — [`L46`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/map_ops.py#L46) — Apply the mapping function to the example.
  - `__init__(self, fn: tp.Callable[[Example], Example])` — [`L38`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/map_ops.py#L38) — Initialize MapTransform.
- protocol/private: `__repr__`[`L57`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/map_ops.py#L57), `_fn`[`L44`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/map_ops.py#L44)
- uses (calls/refs, reference-scoped): [`Transform`](base.md#Transform), [`Example`](base.md#Example)
- used by: [`Transform`](base.md#Transform), [`build_sharded_source`](../../infra/elarge/builders.md#build_sharded_source)

