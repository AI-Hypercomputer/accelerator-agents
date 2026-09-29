---
title: 'Module: jaxlib/mosaic/python/tpu.py'
type: catalog
provenance: extracted
module: jaxlib/mosaic/python/tpu.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jaxlib.mosaic.python.tpu`/
symbols:
  vector_load: vector_load().
  vector_store: vector_store().
  RegionOp: RegionOp#
  RegionOp.body: RegionOp#body().
  TraceOp: TraceOp#
  TraceOp.__init__: TraceOp#__init__().
  TraceOp.body: TraceOp#body().
  RegionOp.__init__: RegionOp#__init__().
  reinterpret_cast: reinterpret_cast().
---
# Module: [`jaxlib/mosaic/python/tpu.py`](../../../../../../../raw/code/jax/jaxlib/mosaic/python/tpu.py)

## Classes
### `RegionOp`
- def: [`jaxlib/mosaic/python/tpu.py:48`](../../../../../../../raw/code/jax/jaxlib/mosaic/python/tpu.py#L48)
- doc: An extension to the automatically generated RegionOp bindings.
- signature: `class RegionOp(_tpu_ops_gen.RegionOp):`
- members:
  - `body(self)` — [`L56`](../../../../../../../raw/code/jax/jaxlib/mosaic/python/tpu.py#L56)
- protocol/private: `__init__`[`L51`](../../../../../../../raw/code/jax/jaxlib/mosaic/python/tpu.py#L51)
- used by: [`_run_scoped_lowering_rule`](../../../jax/_src/pallas/mosaic/lowering.md#_run_scoped_lowering_rule), [`_debug_print_rule`](../../../jax/_src/pallas/mosaic/lowering.md#_debug_print_rule)

### `TraceOp`
- def: [`jaxlib/mosaic/python/tpu.py:35`](../../../../../../../raw/code/jax/jaxlib/mosaic/python/tpu.py#L35)
- doc: An extension to the automatically generated TraceOp bindings.
- signature: `class TraceOp(_tpu_ops_gen.TraceOp):`
- members:
  - `body(self)` — [`L43`](../../../../../../../raw/code/jax/jaxlib/mosaic/python/tpu.py#L43)
- protocol/private: `__init__`[`L38`](../../../../../../../raw/code/jax/jaxlib/mosaic/python/tpu.py#L38)

## Functions
- `reinterpret_cast(result, input, dynamic_sizes=None, *, dynamic_offset=None, loc=None, ip=None)` — [`L95`](../../../../../../../raw/code/jax/jaxlib/mosaic/python/tpu.py#L95)
- `vector_load(result, base, indices, *, strides=None, mask=None, loc=None, ip=None)` — [`L60`](../../../../../../../raw/code/jax/jaxlib/mosaic/python/tpu.py#L60)
- `vector_store(value_to_store, base, indices, *, strides=None, add=False, mask=None, loc=None, ip=None)` — [`L77`](../../../../../../../raw/code/jax/jaxlib/mosaic/python/tpu.py#L77)

