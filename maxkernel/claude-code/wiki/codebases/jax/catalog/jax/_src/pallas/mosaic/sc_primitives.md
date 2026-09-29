---
title: 'Module: jax/_src/pallas/mosaic/sc_primitives.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/sc_primitives.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.sc_primitives`/
symbols:
  fetch_and_add: fetch_and_add().
  _scatter_lowering_rule: _scatter_lowering_rule().
  _gather_lowering_rule: _gather_lowering_rule().
  _gather_abstract_eval: _gather_abstract_eval().
  _masked_sort_lowering_rule: _masked_sort_lowering_rule().
  _parallel_loop_lowering_rule: _parallel_loop_lowering_rule().
  _pack_abstract_eval: _pack_abstract_eval().
  _unpack_abstract_eval: _unpack_abstract_eval().
  load_expanded: load_expanded().
  store_compressed: store_compressed().
  addupdate: addupdate().
  addupdate_compressed: addupdate_compressed().
  _scatter_abstract_eval: _scatter_abstract_eval().
  masked_cumsum_p: masked_cumsum_p.
  scatter_p: scatter_p.
  cummax: cummax().
  cumsum: cumsum().
  all_reduce_population_count_p: all_reduce_population_count_p.
  all_reduce_ffs_p: all_reduce_ffs_p.
  Ref.Ref: Ref.Ref.
  cummin: cummin().
  _masked_sort_abstract_eval: _masked_sort_abstract_eval().
  parallel_loop_p: parallel_loop_p.
  _indexed_shape: _indexed_shape().
  _masked_cumop_lowering_rule: _masked_cumop_lowering_rule().
  swap_p: swap_p.
  bitcast: bitcast().
  parallel_loop.decorator: parallel_loop().decorator().
  pack: pack().
  unpack: unpack().
  _pack_lowering_rule: _pack_lowering_rule().
  _unpack_lowering_rule: _unpack_lowering_rule().
  load_p: load_p.
  gather_p: gather_p.
  scan_count_p: scan_count_p.
  masked_sort_p: masked_sort_p.
  unpack_p: unpack_p.
  fetch_and_add_p: fetch_and_add_p.
  load_gather: load_gather().
  store_scatter: store_scatter().
  addupdate_scatter: addupdate_scatter().
  scan_count: scan_count().
  _reduce_op_lowering_rule: _reduce_op_lowering_rule().
  _swap_abstract_eval: _swap_abstract_eval().
  _bitcast_lowering_rule: _bitcast_lowering_rule().
  _scan_count_abstract_eval: _scan_count_abstract_eval().
  _cumsum_lowering_rule: _cumsum_lowering_rule().
  _fetch_and_add_abstract_eval: _fetch_and_add_abstract_eval().
  TransformedRef.TransformedRef: TransformedRef.TransformedRef.
  barrier_p: barrier_p.
  masked_cummax_p: masked_cummax_p.
  masked_cummin_p: masked_cummin_p.
  bitcast_p: bitcast_p.
  pack_p: pack_p.
  _load_abstract_eval: _load_abstract_eval().
  _load_lowering_rule: _load_lowering_rule().
  _swap_lowering_rule: _swap_lowering_rule().
  _masked_cummax_abstract_eval: _masked_cummax_abstract_eval().
  _parallel_loop_abstract_eval: _parallel_loop_abstract_eval().
  _mask_all_reduce_lowering_rule: _mask_all_reduce_lowering_rule().
  sort_key_val: sort_key_val().
  all_reduce_population_count: all_reduce_population_count().
  all_reduce_ffs: all_reduce_ffs().
  parallel_loop: parallel_loop().
  PackFormat: PackFormat#
  _barrier_abstract_eval: _barrier_abstract_eval().
  _barrier_lowering_rule: _barrier_lowering_rule().
  _scan_count_lowering_rule: _scan_count_lowering_rule().
  _fetch_and_add_lowering_rule: _fetch_and_add_lowering_rule().
  _mask_all_reduce_abstract_eval: _mask_all_reduce_abstract_eval().
  subcore_barrier: subcore_barrier().
  _ensure_ir_value: _ensure_ir_value.
  MemoryEffect: MemoryEffect#
  _format_to_ir_attribute: _format_to_ir_attribute().
  _bitcast_abstract_eval: _bitcast_abstract_eval().
  _T: _T.
  _memory_effect: _memory_effect.
  PackFormat.COMPRESSED: PackFormat#COMPRESSED.
  PackFormat.INTERLEAVED: PackFormat#INTERLEAVED.
  parallel_loop.decorator.wrapped: parallel_loop().decorator().wrapped().
---
# Module: [`jax/_src/pallas/mosaic/sc_primitives.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py)

## Classes
### `MemoryEffect`  ·  implements/extends Effect
- def: [`jax/_src/pallas/mosaic/sc_primitives.py:459`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L459)
- signature: `class MemoryEffect(jax_core.Effect):`
- uses (calls/refs, reference-scoped): [`Effect`](../../core.md#Effect)
- used by: [`_memory_effect`](sc_primitives.md#_memory_effect)

### `PackFormat`  ·  implements/extends Enum
- def: [`jax/_src/pallas/mosaic/sc_primitives.py:947`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L947)
- signature: `class PackFormat(enum.Enum):`
- members:
  - `COMPRESSED` — [`L949`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L949)
  - `INTERLEAVED` — [`L951`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L951)
- used by: [`_pack_abstract_eval`](sc_primitives.md#_pack_abstract_eval), [`_unpack_abstract_eval`](sc_primitives.md#_unpack_abstract_eval), [`pack`](sc_primitives.md#pack), [`unpack`](sc_primitives.md#unpack), [`_format_to_ir_attribute`](sc_primitives.md#_format_to_ir_attribute)

## Functions
- `_barrier_abstract_eval()` — [`L471`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L471)
- `_barrier_lowering_rule(ctx: sc_lowering.LoweringRuleContext)` — [`L476`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L476)
- `_bitcast_abstract_eval(x, dtype)` — [`L420`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L420)
- `_bitcast_lowering_rule(ctx: sc_lowering.LoweringRuleContext, x, *, dtype)` — [`L441`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L441)
- `_cumsum_lowering_rule(ctx: sc_lowering.LoweringRuleContext, x, axis, reverse)` — [`L651`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L651)
- `_fetch_and_add_abstract_eval(*args)` — [`L1210`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L1210)
- `_fetch_and_add_lowering_rule(ctx: sc_lowering.LoweringRuleContext, *args)` — [`L1237`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L1237)
- `_format_to_ir_attribute(format: PackFormat)` — [`L954`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L954)
- `_gather_abstract_eval(*flat_args, tree)` — [`L248`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L248)
- `_gather_lowering_rule(ctx: sc_lowering.LoweringRuleContext, *flat_args, tree)` — [`L264`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L264)
- `_indexed_shape(ref: Ref, indices: Sequence[jax.Array])` — [`L225`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L225)
- `_load_abstract_eval(ref, *args, has_mask, tree)` — [`L60`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L60)
- `_load_lowering_rule(ctx: sc_lowering.LoweringRuleContext, ref, *args, has_mask, tree)` — [`L75`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L75)
- `_mask_all_reduce_abstract_eval(x, *, reduce)` — [`L1137`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L1137)
- `_mask_all_reduce_lowering_rule(ctx: sc_lowering.LoweringRuleContext, x, *, reduce, kind: str)` — [`L1147`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L1147)
- `_masked_cummax_abstract_eval(x, mask)` — [`L547`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L547)
- `_masked_cumop_lowering_rule(ctx: sc_lowering.LoweringRuleContext, x, mask, *, reduction_kind: str)` — [`L558`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L558)
- `_masked_sort_abstract_eval(keys, values, *maybe_mask, descending)` — [`L687`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L687)
- `_masked_sort_lowering_rule(ctx: sc_lowering.LoweringRuleContext, keys, values, *maybe_mask, descending)` — [`L710`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L710)
- `_pack_abstract_eval(a, b, *, format, preferred_element_type)` — [`L962`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L962)
- `_pack_lowering_rule(ctx: sc_lowering.LoweringRuleContext, a, b, *, format, preferred_element_type)` — [`L1004`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L1004)
- `_parallel_loop_abstract_eval(*args, jaxpr, tree, **params)` — [`L765`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L765)
- `_parallel_loop_lowering_rule(ctx: sc_lowering.LoweringRuleContext, *flat_args, tree, unroll, jaxpr)` — [`L781`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L781)
- `_reduce_op_lowering_rule(ctx: sc_lowering.LoweringRuleContext, x, axes, *, reduction_kind, out_sharding=None)` — [`L584`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L584)
- `_scan_count_abstract_eval(x, mask)` — [`L496`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L496)
- `_scan_count_lowering_rule(ctx: sc_lowering.LoweringRuleContext, x, mask)` — [`L508`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L508)
- `_scatter_abstract_eval(*flat_args, tree, add)` — [`L317`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L317)
- `_scatter_lowering_rule(ctx: sc_lowering.LoweringRuleContext, *flat_args, tree, add)` — [`L341`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L341)
- `_swap_abstract_eval(ref, x, *args, has_mask, tree, add)` — [`L120`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L120)
- `_swap_lowering_rule(ctx: sc_lowering.LoweringRuleContext, ref, x, *args, has_mask, tree, add)` — [`L143`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L143)
- `_unpack_abstract_eval(ab, *, format, preferred_element_type)` — [`L1057`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L1057)
- `_unpack_lowering_rule(ctx: sc_lowering.LoweringRuleContext, ab, *, format, preferred_element_type)` — [`L1094`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L1094)
- `addupdate(ref: Ref, x: jax.Array)` — [`L189`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L189) — Performs an atomic add to a ref.
- `addupdate_compressed(ref: Ref, x: jax.Array, *, mask: jax.Array)` — [`L208`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L208) — Performs a masked atomic add to a ref.
- `addupdate_scatter(ref: Ref, indices: Sequence[jax.Array], x: jax.Array, *, mask: jax.Array | None = None)` — [`L399`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L399) — Scatters an array to a ref atomically adding to existing values.
- `all_reduce_ffs(x: jax.Array, *, reduce: int = 1)` — [`L1191`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L1191) — Computes the index of the first true element in the array.
- `all_reduce_population_count(x: jax.Array, *, reduce: int = 1)` — [`L1171`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L1171) — Computes the number of nonzero elements in the array.
- `bitcast(x: jax.Array, dtype: jax.typing.DTypeLike)` — [`L447`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L447) — Bitcasts an array to a different dtype.
- `cummax(x: jax.Array, *, mask: jax.Array | None = None)` — [`L610`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L610) — Returns the cumulative max of the array along its innermost axis.
- `cummin(x: jax.Array, *, mask: jax.Array | None = None)` — [`L630`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L630) — Returns the cumulative min of the array along its innermost axis.
- `cumsum(x: jax.Array, *, mask: jax.Array | None = None)` — [`L666`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L666) — Returns the cumulative sum of the array along its innermost axis.
- `decorator(body)` — [`L902`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L902)
- `fetch_and_add(x_ref: jax.Ref | state_types.TransformedRef, value: jax.typing.ArrayLike, *, subcore_id: jax.typing.ArrayLike)` — [`L1246`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L1246) — Adds value to the ``x_ref`` on another subcore.
- `load_expanded(ref: Ref, *, mask: jax.Array)` — [`L87`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L87) — Performs and expanded masked load from a ref.
- `load_gather(ref: Ref, indices: Sequence[jax.Array], *, mask: jax.Array | None = None)` — [`L288`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L288) — Gathers an array from a ref.
- `pack(a: jax.Array, b: jax.Array,, *, format: PackFormat, preferred_element_type: jax.typing.DTypeLike | None = None)` — [`L1022`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L1022) — Packs two arrays according to the given format.
- `parallel_loop(lower: jax.typing.ArrayLike, upper: jax.typing.ArrayLike, step: jax.typing.ArrayLike = ..., *, unroll: int = ..., carry: None = None)` — [`L816`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L816)
- `scan_count(x: jax.Array, mask: jax.Array | None = None)` — [`L514`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L514) — Computes the running duplicate occurrence count of the array.
- `sort_key_val(keys: jax.Array, values: jax.Array, *, mask: jax.Array | None = None, descending: bool = False)` — [`L738`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L738) — Sorts keys and values, pushing invalid elements to the last positions.
- `store_compressed(ref: Ref, x: jax.Array, *, mask: jax.Array)` — [`L155`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L155) — Performs a compressed masked store to a ref.
- `store_scatter(ref: Ref, indices: Sequence[jax.Array], x: jax.Array, *, mask: jax.Array | None = None)` — [`L371`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L371) — Scatters an array to a ref.
- `subcore_barrier()` — [`L482`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L482) — Blocks until all subcores on the same core reach this instruction.
- `unpack(ab: jax.Array,, *, format: PackFormat, preferred_element_type: jax.typing.DTypeLike | None = None)` — [`L1106`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L1106) — Unpacks two arrays according to the given format.
- `wrapped(idx, *carries)` — [`L904`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L904)

## Module values
- `Ref` — [`L51`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L51)
- `TransformedRef` — [`L50`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L50)
- `_T` — [`L53`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L53)
- `_ensure_ir_value` — [`L48`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L48)
- `_memory_effect` — [`L465`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L465)
- `all_reduce_ffs_p` — [`L1184`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L1184)
- `all_reduce_population_count_p` — [`L1162`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L1162)
- `barrier_p` — [`L467`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L467)
- `bitcast_p` — [`L416`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L416)
- `fetch_and_add_p` — [`L1205`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L1205)
- `gather_p` — [`L243`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L243)
- `load_p` — [`L55`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L55)
- `masked_cummax_p` — [`L534`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L534)
- `masked_cummin_p` — [`L537`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L537)
- `masked_cumsum_p` — [`L540`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L540)
- `masked_sort_p` — [`L683`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L683)
- `pack_p` — [`L958`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L958)
- `parallel_loop_p` — [`L759`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L759)
- `scan_count_p` — [`L491`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L491)
- `scatter_p` — [`L311`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L311)
- `swap_p` — [`L115`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L115)
- `unpack_p` — [`L1052`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/sc_primitives.py#L1052)

