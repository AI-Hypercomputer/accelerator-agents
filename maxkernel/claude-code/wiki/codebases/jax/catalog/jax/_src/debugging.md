---
title: 'Module: jax/_src/debugging.py'
type: catalog
provenance: extracted
module: jax/_src/debugging.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.debugging`/
symbols:
  debug_callback_lowering: debug_callback_lowering().
  _inspect_sharding_lowering_rule: _inspect_sharding_lowering_rule().
  debug_print_p: debug_print_p.
  debug_callback_p: debug_callback_p.
  debug_callback_impl: debug_callback_impl().
  visualize_sharding: visualize_sharding().
  inspect_sharding_p: inspect_sharding_p.
  debug_callback._debug_callback: debug_callback()._debug_callback().
  _make_logging_record: _make_logging_record().
  debug_print_lowering_rule: debug_print_lowering_rule().
  _inspect_sharding_lowering_rule._hlo_sharding_callback: _inspect_sharding_lowering_rule()._hlo_sharding_callback().
  debug_print._debug_print: debug_print()._debug_print().
  debug_print_impl: debug_print_impl().
  _debug_callback_partial_auto.f: _debug_callback_partial_auto().f().
  DebugEffect: DebugEffect#
  debug_print_abstract_eval: debug_print_abstract_eval().
  ordered_debug_effect: ordered_debug_effect.
  debug_effect: debug_effect.
  inspect_array_sharding: inspect_array_sharding().
  _debug_print_eager_rule: _debug_print_eager_rule().
  debug_print: debug_print().
  debug_callback_abstract_eval: debug_callback_abstract_eval().
  _debug_callback_state_discharge_rule: _debug_callback_state_discharge_rule().
  _debug_print_state_discharge_rule: _debug_print_state_discharge_rule().
  unsafe_map: unsafe_map.
  OrderedDebugEffect: OrderedDebugEffect#
  debug_batching_rule: debug_batching_rule().
  debug_callback_lowering._callback: debug_callback_lowering()._callback().
  _split_callback_args: _split_callback_args().
  merge_callback_args: merge_callback_args().
  debug_callback_jvp_rule: debug_callback_jvp_rule().
  _debug_callback_partial_auto: _debug_callback_partial_auto().
  debug_print_jvp_rule: debug_print_jvp_rule().
  _inspect_sharding_batching_rule: _inspect_sharding_batching_rule().
  _inspect_sharding_jvp_rule: _inspect_sharding_jvp_rule().
  inspect_array_sharding._inspect: inspect_array_sharding()._inspect().
  _debug_callback_eager_rule: _debug_callback_eager_rule().
  _make_flat_callback: _make_flat_callback().
  _format_print_callback: _format_print_callback().
  visualize_array_sharding: visualize_array_sharding().
  logger: logger.
  debug_callback: debug_callback().
  debug_batching_rule.get_arg_at_dim: debug_batching_rule().get_arg_at_dim().
  debug_callback_transpose_rule: debug_callback_transpose_rule().
  _make_flat_callback._flat_callback: _make_flat_callback()._flat_callback().
  debug_callback._debug_callback._flat_callback: debug_callback()._debug_callback()._flat_callback().
  formatter: formatter.
  _inspect_sharding_abstract_eval: _inspect_sharding_abstract_eval().
  _canonicalize_color: _canonicalize_color().
  visualize_array_sharding._visualize: visualize_array_sharding()._visualize().
  debug_log: debug_log.
  _debug_partial_eval_custom: _debug_partial_eval_custom().
  debug_print_transpose_rule: debug_print_transpose_rule().
  _DebugPrintFormatChecker: _DebugPrintFormatChecker#
  _inspect_sharding_impl: _inspect_sharding_impl().
  _INSPECT_SHARDING_CALL_NAME: _INSPECT_SHARDING_CALL_NAME.
  _slice_to_chunk_idx: _slice_to_chunk_idx().
  _raise_to_slice: _raise_to_slice().
  Color: Color.
  ColorMap: ColorMap.
  _get_text_color: _get_text_color().
  make_color_iter: make_color_iter().
  DebugEffect.__str__: DebugEffect#__str__.
  OrderedDebugEffect.__str__: OrderedDebugEffect#__str__.
  _DebugPrintFormatChecker.format_field: _DebugPrintFormatChecker#format_field().
  _DebugPrintFormatChecker.check_unused_args: _DebugPrintFormatChecker#check_unused_args().
---
# Module: [`jax/_src/debugging.py`](../../../../../../raw/code/jax/jax/_src/debugging.py)

## Classes
### `DebugEffect`  ·  implements/extends Effect
- def: [`jax/_src/debugging.py:58`](../../../../../../raw/code/jax/jax/_src/debugging.py#L58)
- signature: `class DebugEffect(effects.Effect):`
- protocol/private: `__str__`[`L59`](../../../../../../raw/code/jax/jax/_src/debugging.py#L59)
- used by: [`debug_callback_impl`](debugging.md#debug_callback_impl), [`ordered_debug_effect`](debugging.md#ordered_debug_effect), [`debug_effect`](debugging.md#debug_effect), [`debug_callback_abstract_eval`](debugging.md#debug_callback_abstract_eval), [`_debug_callback_eager_rule`](debugging.md#_debug_callback_eager_rule), [`debug_callback_transpose_rule`](debugging.md#debug_callback_transpose_rule)

### `OrderedDebugEffect`  ·  implements/extends Effect
- def: [`jax/_src/debugging.py:62`](../../../../../../raw/code/jax/jax/_src/debugging.py#L62)
- signature: `class OrderedDebugEffect(effects.Effect):`
- protocol/private: `__str__`[`L63`](../../../../../../raw/code/jax/jax/_src/debugging.py#L63)
- used by: [`ordered_debug_effect`](debugging.md#ordered_debug_effect)

### `_DebugPrintFormatChecker`  ·  implements/extends Formatter
- def: [`jax/_src/debugging.py:528`](../../../../../../raw/code/jax/jax/_src/debugging.py#L528)
- signature: `class _DebugPrintFormatChecker(string.Formatter):`
- members:
  - `check_unused_args(self, used_args, args, kwargs)` — [`L534`](../../../../../../raw/code/jax/jax/_src/debugging.py#L534)
  - `format_field(self, value, format_spec)` — [`L530`](../../../../../../raw/code/jax/jax/_src/debugging.py#L530)
- used by: [`formatter`](debugging.md#formatter)

## Functions
- `_callback(*flat_args)` — [`L199`](../../../../../../raw/code/jax/jax/_src/debugging.py#L199)
- `_canonicalize_color(color: Color)` — [`L790`](../../../../../../raw/code/jax/jax/_src/debugging.py#L790)
- `_debug_callback(callback: Callable[..., None], *c_args: Any, **c_kwargs: Any)` — [`L485`](../../../../../../raw/code/jax/jax/_src/debugging.py#L485)
- `_debug_callback_eager_rule(mesh, *args, callback: Callable[..., Any], effect: DebugEffect, partitioned: bool)` — [`L984`](../../../../../../raw/code/jax/jax/_src/debugging.py#L984)
- `_debug_callback_partial_auto(axis_context, *args, **params)` — [`L146`](../../../../../../raw/code/jax/jax/_src/debugging.py#L146)
- `_debug_callback_state_discharge_rule(in_avals, out_avals, *args, effect, partitioned, callback, **params)` — [`L266`](../../../../../../raw/code/jax/jax/_src/debugging.py#L266)
- `_debug_partial_eval_custom(saveable, unks_in, inst_in, eqn, primitive)` — [`L231`](../../../../../../raw/code/jax/jax/_src/debugging.py#L231)
- `_debug_print(fmt: str, *c_args, **c_kwargs)` — [`L650`](../../../../../../raw/code/jax/jax/_src/debugging.py#L650)
- `_debug_print_eager_rule(mesh, *args, fmt: str, ordered, partitioned, in_tree, static_args, np_printoptions, has_placeholders, logging_record)` — [`L1000`](../../../../../../raw/code/jax/jax/_src/debugging.py#L1000)
- `_debug_print_state_discharge_rule(in_avals, out_avals, *args, **kwargs)` — [`L402`](../../../../../../raw/code/jax/jax/_src/debugging.py#L402)
- `_flat_callback(*dyn_args)` — [`L302`](../../../../../../raw/code/jax/jax/_src/debugging.py#L302)
- `_flat_callback(*dyn_args_flat)` — [`L495`](../../../../../../raw/code/jax/jax/_src/debugging.py#L495)
- `_format_print_callback(fmt: str, np_printoptions, has_placeholders, logging_record, *args, **kwargs)` — [`L549`](../../../../../../raw/code/jax/jax/_src/debugging.py#L549)
- `_get_text_color(color: str)` — [`L796`](../../../../../../raw/code/jax/jax/_src/debugging.py#L796)
- `_hlo_sharding_callback(hlo_sharding: xc.HloSharding)` — [`L743`](../../../../../../raw/code/jax/jax/_src/debugging.py#L743)
- `_inspect(val)` — [`L972`](../../../../../../raw/code/jax/jax/_src/debugging.py#L972)
- `_inspect_sharding_abstract_eval(aval, **_)` — [`L700`](../../../../../../raw/code/jax/jax/_src/debugging.py#L700)
- `_inspect_sharding_batching_rule(args, _, *, callback)` — [`L706`](../../../../../../raw/code/jax/jax/_src/debugging.py#L706)
- `_inspect_sharding_impl(value, *, callback)` — [`L695`](../../../../../../raw/code/jax/jax/_src/debugging.py#L695)
- `_inspect_sharding_jvp_rule(primals, _, **params)` — [`L713`](../../../../../../raw/code/jax/jax/_src/debugging.py#L713)
- `_inspect_sharding_lowering_rule(ctx: mlir.LoweringRuleContext, value, *, callback)` — [`L719`](../../../../../../raw/code/jax/jax/_src/debugging.py#L719)
- `_make_flat_callback(in_tree, callback, static_args)` — [`L301`](../../../../../../raw/code/jax/jax/_src/debugging.py#L301)
- `_make_logging_record(level)` — [`L566`](../../../../../../raw/code/jax/jax/_src/debugging.py#L566)
- `_raise_to_slice(slc: slice | int)` — [`L782`](../../../../../../raw/code/jax/jax/_src/debugging.py#L782)
- `_slice_to_chunk_idx(size: int, slc: slice)` — [`L774`](../../../../../../raw/code/jax/jax/_src/debugging.py#L774)
- `_split_callback_args(args, kwargs)` — [`L276`](../../../../../../raw/code/jax/jax/_src/debugging.py#L276)
- `_visualize(sharding)` — [`L978`](../../../../../../raw/code/jax/jax/_src/debugging.py#L978)
- `debug_batching_rule(args, dims, *, primitive, **params)` — [`L114`](../../../../../../raw/code/jax/jax/_src/debugging.py#L114) — Unrolls the debug callback across the mapped axis.
- `debug_callback(callback: Callable[..., None], *args: Any, ordered: bool = False, partitioned: bool = False, **kwargs: Any)` — [`L409`](../../../../../../raw/code/jax/jax/_src/debugging.py#L409)
- `debug_callback_abstract_eval(*flat_avals, callback: Callable[..., Any], effect: DebugEffect, partitioned: bool)` — [`L108`](../../../../../../raw/code/jax/jax/_src/debugging.py#L108)
- `debug_callback_impl(*args, callback: Callable[..., Any], effect: DebugEffect, partitioned: bool)` — [`L85`](../../../../../../raw/code/jax/jax/_src/debugging.py#L85)
- `debug_callback_jvp_rule(primals, tangents, **params)` — [`L136`](../../../../../../raw/code/jax/jax/_src/debugging.py#L136)
- `debug_callback_lowering(ctx, *args, effect, partitioned, callback, **params)` — [`L155`](../../../../../../raw/code/jax/jax/_src/debugging.py#L155)
- `debug_callback_transpose_rule(_, *flat_args, callback: Callable[..., Any], effect: DebugEffect, partitioned)` — [`L140`](../../../../../../raw/code/jax/jax/_src/debugging.py#L140)
- `debug_print(fmt: str, *args: Any, ordered: bool = False, partitioned: bool = False, skip_format_check: bool = False, _use_logging: bool = False, **kwargs: Any)` — [`L581`](../../../../../../raw/code/jax/jax/_src/debugging.py#L581)
- `debug_print_abstract_eval(*avals: Any, fmt: str, ordered, **kwargs)` — [`L338`](../../../../../../raw/code/jax/jax/_src/debugging.py#L338)
- `debug_print_impl(*args: Any, fmt: str, ordered, partitioned, in_tree, static_args, np_printoptions, has_placeholders, logging_record)` — [`L314`](../../../../../../raw/code/jax/jax/_src/debugging.py#L314)
- `debug_print_jvp_rule(primals, tangents, **params)` — [`L349`](../../../../../../raw/code/jax/jax/_src/debugging.py#L349)
- `debug_print_lowering_rule(ctx, *dyn_args, fmt, ordered, partitioned, in_tree, static_args, np_printoptions, has_placeholders, logging_record)` — [`L364`](../../../../../../raw/code/jax/jax/_src/debugging.py#L364)
- `debug_print_transpose_rule(_, *args, **kwargs)` — [`L356`](../../../../../../raw/code/jax/jax/_src/debugging.py#L356)
- `f()` — [`L148`](../../../../../../raw/code/jax/jax/_src/debugging.py#L148)
- `get_arg_at_dim(i, dim, arg)` — [`L119`](../../../../../../raw/code/jax/jax/_src/debugging.py#L119)
- `inspect_array_sharding(value, *, callback: Callable[[Sharding], None])` — [`L927`](../../../../../../raw/code/jax/jax/_src/debugging.py#L927) — Enables inspecting array sharding inside JIT-ted functions.
- `make_color_iter(color_map, num_rows, num_cols)` — [`L802`](../../../../../../raw/code/jax/jax/_src/debugging.py#L802)
- `merge_callback_args(in_tree, dyn_args, static_args)` — [`L288`](../../../../../../raw/code/jax/jax/_src/debugging.py#L288)
- `visualize_array_sharding(arr, **kwargs)` — [`L976`](../../../../../../raw/code/jax/jax/_src/debugging.py#L976) — Visualizes an array's sharding.
- `visualize_sharding(shape: Sequence[int], sharding: Sharding, *, use_color: bool = True, scale: float = 1, min_width: int = 9, max_width: int = 80, color_map: ColorMap | None = None)` — [`L810`](../../../../../../raw/code/jax/jax/_src/debugging.py#L810) — Visualizes a ``Sharding`` using ``rich``.

## Module values
- `Color` — [`L787`](../../../../../../raw/code/jax/jax/_src/debugging.py#L787)
- `ColorMap` — [`L788`](../../../../../../raw/code/jax/jax/_src/debugging.py#L788)
- `_INSPECT_SHARDING_CALL_NAME` — [`L717`](../../../../../../raw/code/jax/jax/_src/debugging.py#L717)
- `debug_callback_p` — [`L79`](../../../../../../raw/code/jax/jax/_src/debugging.py#L79)
- `debug_effect` — [`L60`](../../../../../../raw/code/jax/jax/_src/debugging.py#L60)
- `debug_log` — [`L687`](../../../../../../raw/code/jax/jax/_src/debugging.py#L687)
- `debug_print_p` — [`L309`](../../../../../../raw/code/jax/jax/_src/debugging.py#L309)
- `formatter` — [`L546`](../../../../../../raw/code/jax/jax/_src/debugging.py#L546)
- `inspect_sharding_p` — [`L691`](../../../../../../raw/code/jax/jax/_src/debugging.py#L691)
- `logger` — [`L56`](../../../../../../raw/code/jax/jax/_src/debugging.py#L56)
- `ordered_debug_effect` — [`L65`](../../../../../../raw/code/jax/jax/_src/debugging.py#L65)
- `unsafe_map` — [`L82`](../../../../../../raw/code/jax/jax/_src/debugging.py#L82)

