---
title: 'Module: jax/_src/pallas/mosaic/sc_lowering.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/sc_lowering.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.sc_lowering`/
symbols:
  _load_lowering_rule: _load_lowering_rule().
  _store_lowering_rule: _store_lowering_rule().
  _extract_indirect_offsets_from_indices: _extract_indirect_offsets_from_indices().
  _dma_start_lowering_rule: _dma_start_lowering_rule().
  register_lowering_rule: register_lowering_rule.
  _dma_wait_lowering_rule: _dma_wait_lowering_rule().
  LoweringRuleContext: LoweringRuleContext.
  _prepare_dma_refs: _prepare_dma_refs().
  _iota_lowering_rule_sc: _iota_lowering_rule_sc().
  _extract_indirect_offsets: _extract_indirect_offsets().
  _sort_lowering_rule: _sort_lowering_rule().
  _has_indirect_offsets: _has_indirect_offsets().
  MemorySpace: MemorySpace.
  _rev_lowering_rule: _rev_lowering_rule().
  _debug_print_lowering_rule: _debug_print_lowering_rule().
  _check_aval_is_supported: _check_aval_is_supported().
  _empty_ref_lowering_rule: _empty_ref_lowering_rule().
  _transform_ref: _transform_ref.
  _get_lowering_rule: _get_lowering_rule().
  _swap_lowering_rule: _swap_lowering_rule().
  _dtype_to_ir_type: _dtype_to_ir_type.
  _get_ref_and_transforms: _get_ref_and_transforms.
  _dma_unflatten: _dma_unflatten.
  CoreMemorySpace: CoreMemorySpace.
  _debug_print_lowering_rule.fail: _debug_print_lowering_rule().fail().
  ShapedAbstractValue: ShapedAbstractValue.
  LoweringContext: LoweringContext.
  MosaicGridMapping: MosaicGridMapping.
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  _dma_start_lowering_rule._dma_start: _dma_start_lowering_rule()._dma_start().
  _dma_wait_lowering_rule._dma_wait: _dma_wait_lowering_rule()._dma_wait().
---
# Module: [`jax/_src/pallas/mosaic/sc_lowering.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py)

## Functions
- `_check_aval_is_supported(caller: str, aval: jax_core.ShapedArray)` — [`L287`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L287)
- `_debug_print_lowering_rule(ctx: LoweringRuleContext, *args, fmt: str, ordered, partitioned, in_tree, static_args, np_printoptions, has_placeholders, logging_record)` — [`L301`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L301)
- `_dma_start(src_ref, dst_ref, sem, src_sem)` — [`L472`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L472)
- `_dma_start_lowering_rule(ctx: LoweringRuleContext, *args, tree, device_id_type: pallas_primitives.DeviceIdType, priority: int, add: bool)` — [`L428`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L428)
- `_dma_wait(src_ref, dst_ref, sem)` — [`L564`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L564)
- `_dma_wait_lowering_rule(ctx: LoweringRuleContext, *args, tree, device_id_type: pallas_primitives.DeviceIdType, insert_dummy_device: bool)` — [`L519`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L519)
- `_empty_ref_lowering_rule(ctx: LoweringRuleContext, ty, memory_space)` — [`L721`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L721)
- `_extract_indirect_offsets(transforms: Sequence[state.Transform], expected_shape: tuple[int, ...], transforms_aval: Sequence[state.Transform], core_type: tpu_core.CoreType)` — [`L669`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L669)
- `_extract_indirect_offsets_from_indices(indices: Sequence[Any], indices_aval: Sequence[Any], core_type: tpu_core.CoreType, indexer_shape: tuple[int | Any, ...], expected_shape: tuple[int, ...] | None = None)` — [`L588`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L588) — Extracts the indirect offsets from the indices, if there are any.
- `_get_lowering_rule(ctx: LoweringRuleContext, ref, *flat_transforms, tree)` — [`L71`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L71)
- `_has_indirect_offsets(transforms: Sequence[state.Transform], transforms_aval: Sequence[state.Transform], core_type: tpu_core.CoreType)` — [`L702`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L702)
- `_iota_lowering_rule_sc(ctx: LoweringRuleContext, dtype, shape, dimension, sharding)` — [`L272`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L272)
- `_load_lowering_rule(ctx: LoweringRuleContext, ref, mask, *flat_transforms, tree)` — [`L75`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L75)
- `_prepare_dma_refs(src_ref, dst_ref, src_aval, dst_aval, core_type: tpu_core.CoreType, is_add: bool = False)` — [`L336`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L336) — Prepares the DMA source and destination references.
- `_rev_lowering_rule(ctx: LoweringRuleContext, x, dimensions)` — [`L771`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L771)
- `_sort_lowering_rule(ctx: LoweringRuleContext, *xs, dimension, is_stable, num_keys)` — [`L730`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L730)
- `_store_lowering_rule(ctx: LoweringRuleContext, ref, val, mask, *flat_transforms, tree, add)` — [`L174`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L174)
- `_swap_lowering_rule(ctx: LoweringRuleContext, ref, val, *flat_transforms, tree)` — [`L166`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L166)
- `fail(reason: str)` — [`L314`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L314)

## Module values
- `CoreMemorySpace` — [`L47`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L47)
- `LoweringContext` — [`L51`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L51)
- `LoweringRuleContext` — [`L52`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L52)
- `MemorySpace` — [`L46`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L46)
- `MosaicGridMapping` — [`L53`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L53)
- `ShapedAbstractValue` — [`L49`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L49)
- `_dma_unflatten` — [`L57`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L57)
- `_dtype_to_ir_type` — [`L55`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L55)
- `_get_ref_and_transforms` — [`L58`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L58)
- `_transform_ref` — [`L56`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L56)
- `register_lowering_rule` — [`L61`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L61)
- `unsafe_map` — [`L42`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L42)
- `unsafe_zip` — [`L43`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_lowering.py#L43)

