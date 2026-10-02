---
title: 'Module: tokamax/_src/pallas/block.py'
type: catalog
provenance: extracted
module: tokamax/_src/pallas/block.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.pallas.block`/
symbols:
  BlockRef.inbounds_masks: BlockRef#inbounds_masks().
  ds: ds.
  pallas_call: pallas_call().
  BlockRef.spec: BlockRef#spec.
  BlockRef._visible_axes: BlockRef#_visible_axes().
  pallas_call.helper.wrapped_kernel: pallas_call().helper().wrapped_kernel().
  BlockRef.load: BlockRef#load().
  BlockRefIndexer.__getitem__: BlockRefIndexer#__getitem__().
  BlockRef.ref: BlockRef#ref.
  BlockRef.bounds_checked: BlockRef#bounds_checked().
  BlockRef.inbounds_mask: BlockRef#inbounds_mask().
  pallas_call.helper.wrapped_kernel.ld_st: pallas_call().helper().wrapped_kernel().ld_st().
  BlockRef.bounds: BlockRef#bounds().
  pallas_call.helper: pallas_call().helper().
  _zip: _zip.
  BlockRef.store: BlockRef#store().
  _is_scalar_indexer: _is_scalar_indexer().
  BlockRef._ndindexer: BlockRef#_ndindexer().
  BlockRefIndexer.ref: BlockRefIndexer#ref.
  _block_ref: _block_ref().
  BlockRef: BlockRef#
  Indexer.Indexer: Indexer.Indexer.
  BlockRef.at: BlockRef#at().
  BlockRef.__getattr__: BlockRef#__getattr__().
  _pids: _pids().
  BlockRef.full_shape: BlockRef#full_shape.
  _as_tuple: _as_tuple().
  BlockRefIndexer: BlockRefIndexer#
  dslice: dslice().
  _PL_LOAD_STORE_PATCH_LOCK: _PL_LOAD_STORE_PATCH_LOCK.
---
# Module: [`tokamax/_src/pallas/block.py`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py)

## Classes
### `BlockRef`
- def: [`tokamax/_src/pallas/block.py:60`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L60) — documented in [tokamax-_src-pallas-block](../../../../concepts/tokamax-_src-pallas-block.md)
- doc: A reference to data with a `BlockSpec`.
- signature: `class BlockRef:`
- members:
  - `_ndindexer(self)` — [`L234`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L234) — Returns the indexer into the block, or `None`.
  - `_visible_axes(self)` — [`L224`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L224) — Returns the axes of the referenced array that are visible.
  - `at(self)` — [`L73`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L73)
  - `bounds(self)` — [`L77`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L77) — Returns the bounds of the reference in each dimension. — documented in [tokamax-_src-pallas-block](../../../../concepts/tokamax-_src-pallas-block.md)
  - `bounds_checked(self)` — [`L89`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L89) — Indicates which dimensions require bounds checking. — documented in [tokamax-_src-pallas-block](../../../../concepts/tokamax-_src-pallas-block.md)
  - `inbounds_mask(self, *, bounds_check: bool | Sequence[bool] = True)` — [`L157`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L157) — Returns a boolean mask indicating if each element is in-bounds. — documented in [tokamax-_src-pallas-block](../../../../concepts/tokamax-_src-pallas-block.md)
  - `inbounds_masks(self)` — [`L98`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L98) — Returns a boolean mask for each dimension indicating if in-bounds. — documented in [tokamax-_src-pallas-block](../../../../concepts/tokamax-_src-pallas-block.md)
  - `load(self, *, bounds_check: bool | Sequence[bool] = True, other: Any = None, **kwargs)` — [`L181`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L181) — Loads a block with `mask=inbounds_mask(bounds_check=bounds_check)`. — documented in [tokamax-_src-pallas-block](../../../../concepts/tokamax-_src-pallas-block.md)
  - `store(self, val: jax.Array, **kwargs)` — [`L212`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L212) — Stores a block with `mask=inbounds_mask()`.
  - `full_shape` — [`L69`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L69) — documented in [tokamax-_src-pallas-block](../../../../concepts/tokamax-_src-pallas-block.md)
  - `ref` — [`L68`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L68) — documented in [tokamax-_src-pallas-block](../../../../concepts/tokamax-_src-pallas-block.md)
  - `spec` — [`L70`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L70) — documented in [tokamax-_src-pallas-block](../../../../concepts/tokamax-_src-pallas-block.md)
- protocol/private: `__getattr__`[`L220`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L220)
- uses (calls/refs, reference-scoped): [`_zip`](block.md#_zip), [`_is_scalar_indexer`](block.md#_is_scalar_indexer), [`BlockRefIndexer`](block.md#BlockRefIndexer), [`_pids`](block.md#_pids)
- used by: [`__getitem__`](block.md#BlockRefIndexer.__getitem__), [`ld_st`](block.md#pallas_call.helper.wrapped_kernel.ld_st), [`_block_ref`](block.md#_block_ref), [`ref`](block.md#BlockRefIndexer.ref)

### `BlockRefIndexer`
- def: [`tokamax/_src/pallas/block.py:249`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L249)
- members:
  - `ref` — [`L252`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L252)
- protocol/private: `__getitem__`[`L254`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L254)
- uses (calls/refs, reference-scoped): [`ref`](block.md#BlockRef.ref), [`BlockRef`](block.md#BlockRef), [`Indexer`](block.md#Indexer.Indexer)
- used by: [`at`](block.md#BlockRef.at)

## Functions
- `_as_tuple(x)` — [`L267`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L267)
- `_block_ref(ref, value, spec)` — [`L261`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L261)
- `_is_scalar_indexer(idx: Indexer)` — [`L45`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L45)
- `_pids()` — [`L49`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L49)
- `dslice(idx: int | jax.Array, size: int)` — [`L37`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L37) — Returns a block-aligned `pl.Slice`.
- `helper(*args, grid_spec=grid_spec)` — [`L309`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L309)
- `ld_st(fn, ref, *args, **kwargs)` — [`L327`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L327)
- `pallas_call(kernel: Callable[..., None], out_shape: Any, *, grid: tuple[int, ...] = (), grid_spec: pl.GridSpec | None = None, in_specs: Any = pl.no_block_spec, out_specs: Any = pl.no_block_spec, filter_specs: bool = False, compiler_params: pl.CompilerParams | None = plgpu.CompilerParams(), **kwargs)` — [`L271`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L271) — Invokes `pallas_call`, wrapping refs with a `BlockSpec` as `BlockRef`s.
- `wrapped_kernel(*refs)` — [`L319`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L319)

## Module values
- `Indexer` — [`L34`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L34)
- `_PL_LOAD_STORE_PATCH_LOCK` — [`L258`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L258)
- `_zip` — [`L31`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L31)
- `ds` — [`L42`](../../../../../../../raw/code/tokamax/tokamax/_src/pallas/block.py#L42)

