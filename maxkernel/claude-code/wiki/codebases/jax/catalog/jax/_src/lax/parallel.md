---
title: 'Module: jax/_src/lax/parallel.py'
type: catalog
provenance: extracted
module: jax/_src/lax/parallel.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.lax.parallel`/
symbols:
  done_p: done_p.
  axis_index: axis_index().
  psum_p: psum_p.
  axis_index_p: axis_index_p.
  _psum: _psum().
  ragged_all_to_all_p: ragged_all_to_all_p.
  all_to_all_p: all_to_all_p.
  pmax_p: pmax_p.
  pmin_p: pmin_p.
  _ragged_all_to_all_transpose: _ragged_all_to_all_transpose().
  all_gather_p: all_gather_p.
  p: p.
  reduce_scatter_p: reduce_scatter_p.
  ppermute_p: ppermute_p.
  pbroadcast_p: pbroadcast_p.
  _build_axis_index_lowering_hlo: _build_axis_index_lowering_hlo().
  _try_mesh_axes_replica_group: _try_mesh_axes_replica_group().
  _all_to_all_batched_collective: _all_to_all_batched_collective().
  _axis_index_effectful_abstract_eval: _axis_index_effectful_abstract_eval().
  vary_unreduced_cast: vary_unreduced_cast().
  psend_p: psend_p.
  preduced: preduced().
  axis_size: axis_size().
  _allreduce_effectful_abstract_eval: _allreduce_effectful_abstract_eval().
  _check_axis_names: _check_axis_names().
  _all_to_all_is_async.bind: _all_to_all_is_async().bind().
  _allreduce_lowering: _allreduce_lowering().
  _ragged_all_to_all_batched_collective: _ragged_all_to_all_batched_collective().
  psum_invariant_p: psum_invariant_p.
  precv_p: precv_p.
  _axis_index_of_val: _axis_index_of_val().
  insert_collective_pvary: insert_collective_pvary().
  _axis_size: _axis_size().
  all_gather_invariant.bind: all_gather_invariant().bind().
  _reduce_scatter_collective: _reduce_scatter_collective().
  bind_psum_invariant: bind_psum_invariant().
  preduced_p: preduced_p.
  vary_unreduced_cast_p: vary_unreduced_cast_p.
  _pbroadcast_lowering: _pbroadcast_lowering().
  _all_gather_batched_collective: _all_gather_batched_collective().
  _batched_reduction_collective: _batched_reduction_collective().
  _constant_reduction: _constant_reduction().
  _all_to_all_effectful_abstract_eval: _all_to_all_effectful_abstract_eval().
  _all_gather.bind: _all_gather().bind().
  _all_gather_lowering: _all_gather_lowering().
  pmin: pmin().
  _pcollectives_lowering_common: _pcollectives_lowering_common().
  all_gather_invariant_p: all_gather_invariant_p.
  pmax: pmax().
  _pbroadcast_is_async: _pbroadcast_is_async().
  _all_to_all_lowering: _all_to_all_lowering().
  _psum_invariant_abstract_eval: _psum_invariant_abstract_eval().
  unreduced_psum: unreduced_psum().
  _psum_is_async.bind: _psum_is_async().bind().
  _ppermute_is_async.bind: _ppermute_is_async().bind().
  _psend_abstract_eval: _psend_abstract_eval().
  _pbroadcast_transpose_rule: _pbroadcast_transpose_rule().
  _ragged_all_to_all_lowering: _ragged_all_to_all_lowering().
  _ragged_all_to_all_jvp: _ragged_all_to_all_jvp().
  _all_gather: _all_gather().
  _all_gather_effectful_abstract_eval: _all_gather_effectful_abstract_eval().
  _reduce_scatter_effectful_abstract_eval: _reduce_scatter_effectful_abstract_eval().
  _psum_scatter_is_async.bind: _psum_scatter_is_async().bind().
  _psum_scatter: _psum_scatter().
  _psum_scatter.bind: _psum_scatter().bind().
  all_gather_reduced.bind: all_gather_reduced().bind().
  _pcast_funcs: _pcast_funcs.
  psum: psum().
  _reduce_scatter_lowering: _reduce_scatter_lowering().
  all_gather_reduced_p: all_gather_reduced_p.
  _psum_is_async: _psum_is_async().
  _maybe_skip_one_sized_axes: _maybe_skip_one_sized_axes().
  _all_to_all_is_async: _all_to_all_is_async().
  _pmin_pmax_abstract_eval: _pmin_pmax_abstract_eval().
  _raise_to_shaped_abstract_eval: _raise_to_shaped_abstract_eval().
  all_gather: all_gather().
  _all_gather_is_async: _all_gather_is_async().
  _psum_scatter_is_async: _psum_scatter_is_async().
  _psum_invariant_lowering_rule: _psum_invariant_lowering_rule().
  all_gather_reduced: all_gather_reduced().
  unreduced_psum_scatter: unreduced_psum_scatter().
  unreduced_reduce_scatter_p: unreduced_reduce_scatter_p.
  _unreduced_psum_lowering: _unreduced_psum_lowering().
  pmax.bind: pmax().bind().
  pmin.bind: pmin().bind().
  psend.bind: psend().bind().
  ragged_all_to_all: ragged_all_to_all().
  _allreduce_lowering.all_reduce: _allreduce_lowering().all_reduce().
  _psum_transpose_rule: _psum_transpose_rule().
  _ragged_all_to_all_effectful_abstract_eval: _ragged_all_to_all_effectful_abstract_eval().
  _all_gather_batcher: _all_gather_batcher().
  all_gather_invariant: all_gather_invariant().
  _all_gather_invariant_effectful_abstract_eval: _all_gather_invariant_effectful_abstract_eval().
  _all_gather_invariant_transpose_rule: _all_gather_invariant_transpose_rule().
  _axis_index_batcher: _axis_index_batcher().
  _psum_invariant_impl: _psum_invariant_impl().
  _pvary_abstract_eval: _pvary_abstract_eval().
  unreduced_psum_scatter.bind: unreduced_psum_scatter().bind().
  unreduced_psum_p: unreduced_psum_p.
  _unreduced_psum_abstract_eval: _unreduced_psum_abstract_eval().
  pcast.bind: pcast().bind().
  _reduce_scatter_start_lowering: _reduce_scatter_start_lowering().
  _unreduced_reduce_scatter_start_lowering: _unreduced_reduce_scatter_start_lowering().
  _canonicalize_axis_index_groups: _canonicalize_axis_index_groups().
  _ppermute_batcher: _ppermute_batcher().
  pmean: pmean().
  precv: precv().
  SingleSideCollectiveEffect: SingleSideCollectiveEffect#
  _moveaxis: _moveaxis().
  pcast: pcast().
  _ppermute_is_async: _ppermute_is_async().
  _reduction_with_positional_batcher: _reduction_with_positional_batcher().
  _async_start_lowering: _async_start_lowering().
  all_to_all: all_to_all().
  collective_vma_rule: collective_vma_rule().
  _all_gather_reduced_lowering: _all_gather_reduced_lowering().
  all_gather_start_p: all_gather_start_p.
  all_gather_reduced_start_p: all_gather_reduced_start_p.
  psum_done_p: psum_done_p.
  _psum.pos_reduce: _psum().pos_reduce().
  _reduction_batcher: _reduction_batcher().
  _psum_transpose_rule.broadcast_positional: _psum_transpose_rule().broadcast_positional().
  _ppermute_transpose_rule: _ppermute_transpose_rule().
  _psend_lowering_gpu: _psend_lowering_gpu().
  _precv_lowering_gpu: _precv_lowering_gpu().
  _precv_abstract_eval: _precv_abstract_eval().
  _pbroadcast_batcher: _pbroadcast_batcher().
  _foldaxis: _foldaxis().
  _all_to_all_batcher: _all_to_all_batcher().
  _all_gather_invariant_batched_collective: _all_gather_invariant_batched_collective().
  _reduce_scatter_batcher: _reduce_scatter_batcher().
  _build_axis_index_lowering_hlo.f: _build_axis_index_lowering_hlo().f().
  _psum_invariant_batching_rule: _psum_invariant_batching_rule().
  _pvary_transpose_rule: _pvary_transpose_rule().
  _pvary_batcher: _pvary_batcher().
  _all_gather_reduced_effectful_abstract_eval: _all_gather_reduced_effectful_abstract_eval().
  _unreduced_reduce_scatter_effectful_abstract_eval: _unreduced_reduce_scatter_effectful_abstract_eval().
  _vary_unreduced_cast_abstract_eval: _vary_unreduced_cast_abstract_eval().
  _async_done_abstract_eval: _async_done_abstract_eval().
  psum_scatter: psum_scatter().
  _get_from: _get_from().
  psum_start_p: psum_start_p.
  psum_invariant_start_p: psum_invariant_start_p.
  unreduced_psum_start_p: unreduced_psum_start_p.
  reduce_scatter_start_p: reduce_scatter_start_p.
  unreduced_reduce_scatter_start_p: unreduced_reduce_scatter_start_p.
  all_to_all_start_p: all_to_all_start_p.
  pbroadcast_start_p: pbroadcast_start_p.
  ppermute_start_p: ppermute_start_p.
  all_gather_done_p: all_gather_done_p.
  reduce_scatter_done_p: reduce_scatter_done_p.
  pargmin: pargmin().
  pargmax: pargmax().
  psend: psend().
  _replica_groups: _replica_groups().
  ppermute: ppermute().
  _ppermute_lowering: _ppermute_lowering().
  single_side_collective_effect: single_side_collective_effect.
  _all_gather_invariant_lowering: _all_gather_invariant_lowering().
  _unreduced_reduce_scatter_lowering: _unreduced_reduce_scatter_lowering().
  all_to_all_done_p: all_to_all_done_p.
  pbroadcast_done_p: pbroadcast_done_p.
  ppermute_done_p: ppermute_done_p.
  _allreduce_impl: _allreduce_impl().
  _splitaxis: _splitaxis().
  _raise_valueerror: _raise_valueerror().
  _all_to_all_transpose_rule: _all_to_all_transpose_rule().
  _all_gather_transpose_rule: _all_gather_transpose_rule().
  _reduce_scatter_transpose_rule: _reduce_scatter_transpose_rule().
  _axis_index_lowering: _axis_index_lowering().
  _psum_invariant_transpose_rule: _psum_invariant_transpose_rule().
  _all_gather_reduced_transpose_rule: _all_gather_reduced_transpose_rule().
  _unreduced_reduce_scatter_transpose_rule: _unreduced_reduce_scatter_transpose_rule().
  _unreduced_psum_transpose_rule: _unreduced_psum_transpose_rule().
  _preduced_abstract_eval: _preduced_abstract_eval().
  _preduced_transpose_rule: _preduced_transpose_rule().
  _vary_unreduced_cast_transpose_rule: _vary_unreduced_cast_transpose_rule().
  _reduced_vary_cast_abstract_eval: _reduced_vary_cast_abstract_eval().
  _reduced_vary_cast_transpose_rule: _reduced_vary_cast_transpose_rule().
  _async_start_abstract_eval: _async_start_abstract_eval().
  _validate_reduce_axis_index_groups: _validate_reduce_axis_index_groups().
  _device_list_replica_groups_hlo: _device_list_replica_groups_hlo().
  pbroadcast: pbroadcast().
  pshuffle: pshuffle().
  pswapaxes: pswapaxes().
  SingleSideCollectiveEffect.__hash__: SingleSideCollectiveEffect#__hash__().
  SingleSideCollectiveEffect.__eq__: SingleSideCollectiveEffect#__eq__().
  all_gather_start: all_gather_start().
  psum_start: psum_start().
  psum_scatter_start: psum_scatter_start().
  all_to_all_start: all_to_all_start().
  pbroadcast_start: pbroadcast_start().
  ppermute_start: ppermute_start().
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  _allowed_pcast_to: _allowed_pcast_to.
  _allreduce_lowering._positional_reduce: _allreduce_lowering()._positional_reduce().
  _psend_lowering: _psend_lowering().
  _precv_lowering: _precv_lowering().
  _pbroadcast_lowering.source_to_front: _pbroadcast_lowering().source_to_front().
  _all_to_all_impl: _all_to_all_impl().
  _ragged_all_to_all_batched_collective.bdim_at_second: _ragged_all_to_all_batched_collective().bdim_at_second().
  _ragged_all_to_all_batched_collective.merge: _ragged_all_to_all_batched_collective().merge().
  _ragged_all_to_all_batched_collective.split: _ragged_all_to_all_batched_collective().split().
  _ragged_all_to_all_impl: _ragged_all_to_all_impl().
  _all_gather_impl: _all_gather_impl().
  _all_gather_invariant_impl: _all_gather_invariant_impl().
  _all_gather_reduced_impl: _all_gather_reduced_impl().
  _all_gather_reduced_batched_collective: _all_gather_reduced_batched_collective().
  _unreduced_reduce_scatter_impl: _unreduced_reduce_scatter_impl().
  _unreduced_reduce_scatter_batcher: _unreduced_reduce_scatter_batcher().
  _unreduced_psum_batcher: _unreduced_psum_batcher().
  _preduced_batcher: _preduced_batcher().
  _vary_unreduced_cast_batcher: _vary_unreduced_cast_batcher().
  _reduced_vary_cast_batcher: _reduced_vary_cast_batcher().
  async_prim: async_prim.
  sync_prim: sync_prim.
  SingleSideCollectiveEffect.__str__: SingleSideCollectiveEffect#__str__.
---
# Module: [`jax/_src/lax/parallel.py`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py)

## Classes
### `SingleSideCollectiveEffect`  ·  implements/extends Effect
- def: [`jax/_src/lax/parallel.py:1189`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1189)
- signature: `class SingleSideCollectiveEffect(core.Effect):`
- protocol/private: `__eq__`[`L1193`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1193), `__hash__`[`L1191`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1191), `__str__`[`L1190`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1190)
- uses (calls/refs, reference-scoped): [`Effect`](../core.md#Effect)
- used by: [`single_side_collective_effect`](parallel.md#single_side_collective_effect)

## Functions
- `_all_gather(x, axis_name, *, axis_index_groups, axis, tiled, is_async)` — [`L1800`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1800)
- `_all_gather_batched_collective(prim, axis_data, vals_in, dims_in, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled)` — [`L1934`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1934)
- `_all_gather_batcher(prim, vals_in, dims_in, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled)` — [`L1913`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1913)
- `_all_gather_effectful_abstract_eval(x_aval, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled)` — [`L1889`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1889)
- `_all_gather_impl(x, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled)` — [`L1818`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1818)
- `_all_gather_invariant_batched_collective(axis_data, vals_in, dims_in, all_gather_dimension, axis_name, axis_size, tiled)` — [`L2065`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2065)
- `_all_gather_invariant_effectful_abstract_eval(x_aval, *, all_gather_dimension, axis_name, axis_size, tiled)` — [`L2018`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2018)
- `_all_gather_invariant_impl(x, *, all_gather_dimension, axis_name, axis_size, tiled)` — [`L2036`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2036)
- `_all_gather_invariant_lowering(ctx, x, *, all_gather_dimension, axis_name, axis_size, tiled, platform=None)` — [`L2042`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2042)
- `_all_gather_invariant_transpose_rule(cts, x, *, all_gather_dimension, axis_name, axis_size, tiled)` — [`L2055`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2055)
- `_all_gather_is_async(x, axis_name, *, axis_index_groups=None, axis=0, tiled=False, to: str = 'varying', is_async: bool = False)` — [`L1777`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1777)
- `_all_gather_lowering(ctx, x, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled, platform=None, is_async=False)` — [`L1821`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1821)
- `_all_gather_reduced_batched_collective(axis_data, vals_in, dims_in, all_gather_dimension, axis_name, axis_size, tiled)` — [`L2592`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2592)
- `_all_gather_reduced_effectful_abstract_eval(x_aval, *, all_gather_dimension, axis_name, axis_size, tiled)` — [`L2532`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2532)
- `_all_gather_reduced_impl(x, *, all_gather_dimension, axis_name, axis_size, tiled)` — [`L2565`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2565)
- `_all_gather_reduced_lowering(ctx, x, *, all_gather_dimension, axis_name, axis_size, tiled, platform=None, is_async=False)` — [`L2571`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2571)
- `_all_gather_reduced_transpose_rule(cts, x, *, all_gather_dimension, axis_name, axis_size, tiled)` — [`L2585`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2585)
- `_all_gather_transpose_rule(cts, x, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled)` — [`L1906`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1906)
- `_all_to_all_batched_collective(axis_data, vals_in, dims_in, axis_name, split_axis, concat_axis, axis_index_groups, tiled)` — [`L1441`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1441)
- `_all_to_all_batcher(vals_in, dims_in, *, axis_name, split_axis, concat_axis, axis_index_groups, tiled)` — [`L1427`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1427)
- `_all_to_all_effectful_abstract_eval(input_aval, axis_name, split_axis, concat_axis, axis_index_groups, tiled)` — [`L1513`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1513)
- `_all_to_all_impl(*args, **kwargs)` — [`L1539`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1539)
- `_all_to_all_is_async(x, axis_name, split_axis, concat_axis, *, axis_index_groups=None, tiled=False, is_async=False)` — [`L568`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L568)
- `_all_to_all_lowering(ctx, x, *, split_axis, concat_axis, axis_name, axis_index_groups, tiled, is_async=False)` — [`L1354`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1354)
- `_all_to_all_transpose_rule(cts, x, axis_name, split_axis, concat_axis, axis_index_groups, tiled)` — [`L1416`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1416)
- `_allreduce_effectful_abstract_eval(aval, *, axes, axis_index_groups)` — [`L978`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L978)
- `_allreduce_impl(prim, pos_reducer, arg, *, axes, axis_index_groups)` — [`L970`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L970)
- `_allreduce_lowering(prim, pos_fn, ctx, arg, *, axes, axis_index_groups)` — [`L1009`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1009)
- `_async_done_abstract_eval(aval)` — [`L3005`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L3005)
- `_async_start_abstract_eval(sync_prim, done_fun, *args, **kwargs)` — [`L2986`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2986)
- `_async_start_lowering(sync_lower, ctx, x, **kwargs)` — [`L3016`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L3016) — Returns an async start lowering function given a synchronous lowering.
- `_axis_index_batcher(axis_data, vals_in, dims_in, *, axis_name)` — [`L2408`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2408)
- `_axis_index_effectful_abstract_eval(*, axis_name)` — [`L2396`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2396)
- `_axis_index_lowering(ctx, *, axis_name)` — [`L2392`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2392)
- `_axis_index_of_val(x, val, axis_name)` — [`L302`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L302)
- `_axis_size(axis_name: AxisName, axis_index_groups: Sequence[Sequence[int]] | None = None,)` — [`L832`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L832)
- `_batched_reduction_collective(prim, if_unmapped, axis_data, vals_in, dims_in, axes, axis_index_groups)` — [`L890`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L890)
- `_build_axis_index_lowering_hlo(ctx, axis_name, axis_ctx)` — [`L2342`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2342)
- `_canonicalize_axis_index_groups(axis_index_groups)` — [`L317`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L317)
- `_check_axis_names(axes, api_name)` — [`L1000`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1000)
- `_constant_reduction(prim, axis_data, arg, axes, axis_index_groups)` — [`L843`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L843)
- `_device_list_replica_groups_hlo(replica_groups: Sequence[Sequence[int]])` — [`L932`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L932)
- `_foldaxis(axis, x)` — [`L1349`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1349)
- `_get_from(aval, axes: tuple[AxisName, ...], name)` — [`L2893`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2893)
- `_maybe_skip_one_sized_axes(axes)` — [`L188`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L188)
- `_moveaxis(src, dst, x)` — [`L1338`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1338)
- `_pbroadcast_batcher(axis_data, vals_in, dims_in, axis_name, source)` — [`L1287`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1287)
- `_pbroadcast_is_async(x, axis_name, source, is_async=False)` — [`L350`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L350)
- `_pbroadcast_lowering(ctx, x, *, axis_name, source)` — [`L1307`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1307)
- `_pbroadcast_transpose_rule(t, x, source, axis_name)` — [`L1282`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1282)
- `_pcollectives_lowering_common(ctx, *, axis_name, perm, op_name)` — [`L1109`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1109)
- `_pmin_pmax_abstract_eval(name, aval, *, axes, axis_index_groups)` — [`L994`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L994)
- `_positional_reduce(aval, arg)` — [`L1021`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1021)
- `_ppermute_batcher(axis_data, vals_in, dims_in, axis_name, perm)` — [`L1154`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1154)
- `_ppermute_is_async(x, axis_name, perm, is_async=False)` — [`L383`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L383)
- `_ppermute_lowering(ctx, x, *, axis_name, perm)` — [`L1141`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1141)
- `_ppermute_transpose_rule(t, x, perm, axis_name)` — [`L1149`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1149)
- `_precv_abstract_eval(token, *, out_shape, axis_name, **params)` — [`L1266`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1266)
- `_precv_lowering(ctx, token, *, out_shape, axis_name, perm)` — [`L1276`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1276)
- `_precv_lowering_gpu(ctx, token, *, out_shape, axis_name, perm)` — [`L1245`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1245)
- `_preduced_abstract_eval(aval, *, axes)` — [`L2768`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2768)
- `_preduced_batcher(vals_in, dims_in, *, axes)` — [`L2790`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2790)
- `_preduced_transpose_rule(cts, arg, *, axes)` — [`L2785`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2785)
- `_psend_abstract_eval(x, *, axis_name, **params)` — [`L1225`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1225)
- `_psend_lowering(ctx, x, *, axis_name, perm)` — [`L1238`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1238)
- `_psend_lowering_gpu(ctx, x, *, axis_name, perm)` — [`L1200`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1200)
- `_psum(x, axis_name, *, axis_index_groups, is_async)` — [`L146`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L146)
- `_psum_invariant_abstract_eval(name, aval, *, axes)` — [`L2438`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2438)
- `_psum_invariant_batching_rule(axis_data, vals_in, dims_in, axes)` — [`L2464`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2464)
- `_psum_invariant_impl(arg, *, axes)` — [`L2433`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2433)
- `_psum_invariant_lowering_rule(ctx, arg, *, axes)` — [`L2459`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2459)
- `_psum_invariant_transpose_rule(cts, arg, *, axes)` — [`L2470`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2470)
- `_psum_is_async(x, axis_name, *, axis_index_groups=None, is_async=False)` — [`L127`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L127)
- `_psum_scatter(x, axis_name, *, scatter_dimension, axis_index_groups, tiled, is_async)` — [`L2325`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2325)
- `_psum_scatter_is_async(x, axis_name, *, scatter_dimension=0, axis_index_groups=None, tiled=False, is_async=False)` — [`L2303`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2303)
- `_psum_transpose_rule(cts, arg, *, axes, axis_index_groups)` — [`L1063`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1063)
- `_pvary_abstract_eval(aval, *, axes)` — [`L2483`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2483)
- `_pvary_batcher(vals_in, dims_in, *, axes)` — [`L2503`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2503)
- `_pvary_transpose_rule(cts, arg, *, axes)` — [`L2498`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2498)
- `_ragged_all_to_all_batched_collective(axis_data, vals_in, dims_in, axis_name, axis_index_groups)` — [`L1656`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1656)
- `_ragged_all_to_all_effectful_abstract_eval(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, axis_name, axis_index_groups)` — [`L1582`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1582)
- `_ragged_all_to_all_impl(*args, **kwargs)` — [`L1685`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1685)
- `_ragged_all_to_all_jvp(primals, tangents, **params)` — [`L1621`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1621)
- `_ragged_all_to_all_lowering(ctx, operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name, axis_index_groups)` — [`L1551`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1551)
- `_ragged_all_to_all_transpose(t, operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name, axis_index_groups)` — [`L1635`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1635)
- `_raise_to_shaped_abstract_eval(x, *, axis_name, **params)` — [`L1175`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1175)
- `_raise_valueerror(name, arg, *, axes)` — [`L2477`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2477)
- `_reduce_scatter_batcher(vals_in, dims_in, *, scatter_dimension, axis_name, axis_index_groups, axis_size, tiled)` — [`L2162`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2162)
- `_reduce_scatter_collective(axis_data, vals_in, dims_in, scatter_dimension, axis_name, axis_index_groups, axis_size, tiled)` — [`L2178`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2178)
- `_reduce_scatter_effectful_abstract_eval(x_aval, *, axis_name, scatter_dimension, axis_index_groups, axis_size, tiled)` — [`L2128`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2128)
- `_reduce_scatter_lowering(prim, ctx, x, *, scatter_dimension, axis_name, axis_index_groups, axis_size, tiled)` — [`L2074`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2074)
- `_reduce_scatter_start_lowering(ctx, x, *, tiled, **kwargs)` — [`L3052`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L3052)
- `_reduce_scatter_transpose_rule(cts, x, *, axis_name, scatter_dimension, axis_index_groups, axis_size, tiled)` — [`L2155`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2155)
- `_reduced_vary_cast_abstract_eval(aval, *, axes)` — [`L2857`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2857)
- `_reduced_vary_cast_batcher(vals_in, dims_in, *, axes)` — [`L2887`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2887)
- `_reduced_vary_cast_transpose_rule(cts, x, *, axes)` — [`L2882`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2882)
- `_reduction_batcher(prim, v, d, *, axes, axis_index_groups)` — [`L875`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L875)
- `_reduction_with_positional_batcher(prim, v, d, axis_index_groups, transform_unmapped, transform_mapped)` — [`L858`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L858)
- `_replica_groups(axis_ctx, axis_name, axis_index_groups)` — [`L923`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L923)
- `_splitaxis(axis, factor, x)` — [`L1343`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1343)
- `_try_mesh_axes_replica_group(ctx, axis_names, axis_index_groups)` — [`L941`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L941)
- `_unreduced_psum_abstract_eval(aval, *, axes)` — [`L2707`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2707)
- `_unreduced_psum_batcher(axis_data, vals_in, dims_in, axes)` — [`L2740`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2740)
- `_unreduced_psum_lowering(ctx, arg, *, axes)` — [`L2735`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2735)
- `_unreduced_psum_transpose_rule(cts, arg, *, axes)` — [`L2744`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2744)
- `_unreduced_reduce_scatter_batcher(axis_data, vals_in, dims_in, axis_name, scatter_dimension, axis_size, tiled)` — [`L2678`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2678)
- `_unreduced_reduce_scatter_effectful_abstract_eval(x_aval, *, axis_name, scatter_dimension, axis_size, tiled)` — [`L2622`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2622)
- `_unreduced_reduce_scatter_impl(x, *, axis_name, scatter_dimension, axis_size, tiled)` — [`L2667`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2667)
- `_unreduced_reduce_scatter_lowering(prim, ctx, x, *, axis_name, scatter_dimension, axis_size, tiled)` — [`L2685`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2685)
- `_unreduced_reduce_scatter_start_lowering(ctx, x, *, tiled, **kwargs)` — [`L3063`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L3063)
- `_unreduced_reduce_scatter_transpose_rule(cts, x, *, axis_name, scatter_dimension, axis_size, tiled)` — [`L2672`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2672)
- `_validate_reduce_axis_index_groups(axis_index_groups)` — [`L310`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L310)
- `_vary_unreduced_cast_abstract_eval(aval, *, axes)` — [`L2814`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2814)
- `_vary_unreduced_cast_batcher(vals_in, dims_in, *, axes)` — [`L2845`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2845)
- `_vary_unreduced_cast_transpose_rule(cts, x, *, axes)` — [`L2840`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2840)
- `all_gather(x, axis_name, *, axis_index_groups=None, axis=0, tiled=False, to: str = 'varying')` — [`L1707`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1707) — Gather values of x across all replicas.
- `all_gather_invariant(x, axis_name, *, axis: int = 0, tiled: bool = False)` — [`L1982`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1982) — Gather values of x across all replicas.
- `all_gather_reduced(x, axis_name, *, axis: int = 0, tiled: bool = False, is_async: bool = False)` — [`L2514`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2514)
- `all_gather_start(*args, **kwargs)` — [`L2967`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2967)
- `all_reduce(aval, x)` — [`L1038`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1038)
- `all_to_all(x, axis_name, split_axis, concat_axis, *, axis_index_groups=None, tiled=False)` — [`L521`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L521) — Materialize the mapped axis and map a different axis.
- `all_to_all_start(*args, **kwargs)` — [`L2976`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2976)
- `axis_index(axis_name: AxisName)` — [`L761`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L761) — Return the index along the mapped axis ``axis_name``.
- `axis_size(axis_name: AxisName)` — [`L802`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L802) — Return the size of the mapped axis ``axis_name``.
- `bdim_at_second(x, d)` — [`L1668`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1668)
- `bind(leaf)` — [`L135`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L135)
- `bind(leaf)` — [`L254`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L254)
- `bind(leaf)` — [`L285`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L285)
- `bind(leaf)` — [`L386`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L386)
- `bind(leaf)` — [`L421`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L421)
- `bind(x, split_axis=split_axis, concat_axis=concat_axis)` — [`L571`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L571)
- `bind(leaf)` — [`L1807`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1807)
- `bind(leaf)` — [`L2005`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2005)
- `bind(leaf)` — [`L2311`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2311)
- `bind(leaf)` — [`L2333`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2333)
- `bind(leaf)` — [`L2521`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2521)
- `bind(leaf)` — [`L2609`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2609)
- `bind(leaf)` — [`L2935`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2935)
- `bind_psum_invariant(leaf, *, axes, axis_index_groups, is_async)` — [`L2422`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2422)
- `broadcast_positional(ct, arg)` — [`L1069`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1069)
- `collective_vma_rule(prim_name, axis_name, x_aval)` — [`L1876`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1876)
- `f()` — [`L2370`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2370)
- `insert_collective_pvary(axis_name, x)` — [`L1698`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1698)
- `merge(x)` — [`L1672`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1672)
- `pargmax(x, axis_name)` — [`L297`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L297)
- `pargmin(x, axis_name)` — [`L291`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L291)
- `pbroadcast(x, axis_name, source)` — [`L323`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L323) — Perform a collective broadcast and replicate from ``source``.
- `pbroadcast_start(*args, **kwargs)` — [`L2979`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2979)
- `pcast(x, axis_name, *, to: str)` — [`L2923`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2923)
- `pmax(x, axis_name, *, axis_index_groups=None)` — [`L229`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L229) — Compute an all-reduce max on ``x`` over the pmapped axis ``axis_name``.
- `pmean(x, axis_name, *, axis_index_groups=None)` — [`L196`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L196) — Compute an all-reduce mean on ``x`` over the pmapped axis ``axis_name``.
- `pmin(x, axis_name, *, axis_index_groups=None)` — [`L260`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L260) — Compute an all-reduce min on ``x`` over the pmapped axis ``axis_name``.
- `pos_reduce(x)` — [`L163`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L163)
- `ppermute(x, axis_name, perm)` — [`L356`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L356) — Perform a collective permutation according to the permutation ``perm``.
- `ppermute_start(*args, **kwargs)` — [`L2982`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2982)
- `precv(token, out_shape, axis_name, perm)` — [`L428`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L428) — Perform a collective recv according to the permutation ``perm``.
- `preduced(x, axis_name)` — [`L2752`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2752)
- `psend(x, axis_name, perm)` — [`L393`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L393) — Perform a collective send according to the permutation ``perm``.
- `pshuffle(x, axis_name, perm)` — [`L466`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L466) — Convenience wrapper of jax.lax.ppermute with alternate permutation encoding
- `psum(x, axis_name, *, axis_index_groups=None)` — [`L59`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L59) — Compute an all-reduce sum on ``x`` over the pmapped axis ``axis_name``.
- `psum_scatter(x, axis_name, *, scatter_dimension=0, axis_index_groups=None, tiled=False)` — [`L2222`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2222) — Like ``psum(x, axis_name)`` but each device retains only part of the result.
- `psum_scatter_start(*args, **kwargs)` — [`L2973`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2973)
- `psum_start(*args, **kwargs)` — [`L2970`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2970)
- `pswapaxes(x, axis_name, axis, *, axis_index_groups=None)` — [`L490`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L490) — Swap the pmapped axis ``axis_name`` with the unmapped axis ``axis``.
- `ragged_all_to_all(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name, axis_index_groups=None)` — [`L603`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L603) — Ragged version of :func:`all_to_all` collective.
- `source_to_front(group)` — [`L1309`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1309)
- `split(x)` — [`L1673`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1673)
- `unreduced_psum(x, axis_name, is_async=False)` — [`L2696`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2696)
- `unreduced_psum_scatter(x, axis_name, *, scatter_dimension=0, tiled=False, is_async=False)` — [`L2602`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2602)
- `vary_unreduced_cast(x, axis_name)` — [`L2797`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2797)

## Module values
- `_allowed_pcast_to` — [`L2921`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2921)
- `_pcast_funcs` — [`L2914`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2914)
- `all_gather_done_p` — [`L2958`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2958)
- `all_gather_invariant_p` — [`L2016`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2016)
- `all_gather_p` — [`L1969`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1969)
- `all_gather_reduced_p` — [`L2530`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2530)
- `all_gather_reduced_start_p` — [`L2947`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2947)
- `all_gather_start_p` — [`L2946`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2946)
- `all_to_all_done_p` — [`L2961`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2961)
- `all_to_all_p` — [`L1543`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1543)
- `all_to_all_start_p` — [`L2953`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2953)
- `async_prim` — [`L2990`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2990)
- `axis_index_p` — [`L2414`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2414)
- `done_p` — [`L2990`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2990)
- `p` — [`L1973`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1973)
- `pbroadcast_done_p` — [`L2962`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2962)
- `pbroadcast_p` — [`L1331`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1331)
- `pbroadcast_start_p` — [`L2954`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2954)
- `pmax_p` — [`L1091`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1091)
- `pmin_p` — [`L1100`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1100)
- `ppermute_done_p` — [`L2963`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2963)
- `ppermute_p` — [`L1181`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1181)
- `ppermute_start_p` — [`L2955`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2955)
- `precv_p` — [`L1272`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1272)
- `preduced_p` — [`L2764`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2764)
- `psend_p` — [`L1233`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1233)
- `psum_done_p` — [`L2959`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2959)
- `psum_invariant_p` — [`L2431`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2431)
- `psum_invariant_start_p` — [`L2949`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2949)
- `psum_p` — [`L1081`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1081)
- `psum_start_p` — [`L2948`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2948)
- `ragged_all_to_all_p` — [`L1689`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1689)
- `reduce_scatter_done_p` — [`L2960`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2960)
- `reduce_scatter_p` — [`L2212`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2212)
- `reduce_scatter_start_p` — [`L2951`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2951)
- `single_side_collective_effect` — [`L1197`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L1197)
- `sync_prim` — [`L2990`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2990)
- `unreduced_psum_p` — [`L2705`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2705)
- `unreduced_psum_start_p` — [`L2950`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2950)
- `unreduced_reduce_scatter_p` — [`L2620`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2620)
- `unreduced_reduce_scatter_start_p` — [`L2952`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2952)
- `unsafe_map` — [`L53`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L53)
- `unsafe_zip` — [`L54`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L54)
- `vary_unreduced_cast_p` — [`L2810`](../../../../../../../raw/code/jax/jax/_src/lax/parallel.py#L2810)

