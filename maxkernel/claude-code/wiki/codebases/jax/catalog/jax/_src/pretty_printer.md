---
title: 'Module: jax/_src/pretty_printer.py'
type: catalog
provenance: extracted
module: jax/_src/pretty_printer.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pretty_printer`/
symbols:
  text: text().
  Doc: Doc#
  concat: concat().
  join: join().
  brk: brk().
  Doc.format: Doc#format().
  color: color().
  group: group().
  nest: nest().
  keyword: keyword.
  type_annotation: type_annotation.
  Color: Color.
  Doc._format: Doc#_format().
  OutputFormat: OutputFormat.
  Doc.__add__: Doc#__add__().
  nil: nil().
  Intensity: Intensity.
  _PPRINT_USE_COLOR: _PPRINT_USE_COLOR.
  CAN_USE_COLOR: CAN_USE_COLOR.
  source_map: source_map().
  Doc.__repr__: Doc#__repr__().
  Doc.__str__: Doc#__str__().
  _can_use_color: _can_use_color().
---
# Module: [`jax/_src/pretty_printer.py`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py)

## Classes
### `Doc`
- def: [`jax/_src/pretty_printer.py:71`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L71)
- signature: `class Doc:`
- members:
  - `format(self, width: int = 80, *, use_color: bool | None = None, output_format: OutputFormat | None = None, separable_lines: bool = False, annotation_prefix: str = " # ", source_map: list[list[tuple[int, int, Any]]] | None = None)` — [`L97`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L97) — Formats a pretty-printer document as a string.
- protocol/private: `__add__`[`L74`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L74), `__repr__`[`L78`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L78), `__str__`[`L81`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L81), `_format`[`L85`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L85)
- uses (calls/refs, reference-scoped): [`use_cpp_method`](util.md#use_cpp_method), [`OutputFormat`](pretty_printer.md#OutputFormat), [`CAN_USE_COLOR`](pretty_printer.md#CAN_USE_COLOR), [`_PPRINT_USE_COLOR`](pretty_printer.md#_PPRINT_USE_COLOR)
- used by: [`text`](pretty_printer.md#text), [`_pp_eqn`](core.md#_pp_eqn), [`pp_jaxpr_skeleton`](core.md#pp_jaxpr_skeleton), [`concat`](pretty_printer.md#concat), [`pp_toplevel_jaxpr`](core.md#pp_toplevel_jaxpr), [`pp_vars`](core.md#pp_vars), [`_pretty_print`](core.md#Tracer._pretty_print), [`pp_var`](core.md#pp_var), [`pp_jaxprs`](core.md#pp_jaxprs), [`pp_eqn`](core.md#pp_eqn), [`pp_kv_pairs`](core.md#pp_kv_pairs), [`_pjit_pp_rule`](pjit.md#_pjit_pp_rule), [`pp_jaxpr`](core.md#pp_jaxpr), [`_pp_slice`](state/indexing.md#_pp_slice), [`pp_shared_jaxpr`](core.md#pp_shared_jaxpr), [`pretty_print`](state/types.md#Transform.pretty_print), [`_pretty_print`](state/types.md#RefEffect._pretty_print), [`pretty_print`](state/indexing.md#NDIndexer.pretty_print), [`pp_kv_pair`](core.md#pp_kv_pair), [`_swap_pp_rule`](state/primitives.md#_swap_pp_rule), [`join`](pretty_printer.md#join), [`pp_eqns`](core.md#pp_eqns), [`pp_jaxpr_eqn_range`](core.md#pp_jaxpr_eqn_range), [`_addupdate_pp_rule`](state/primitives.md#_addupdate_pp_rule), [`_custom_jvp_call_pp_rule`](custom_derivatives.md#_custom_jvp_call_pp_rule), [`_custom_vjp_call_pp_rule`](custom_derivatives.md#_custom_vjp_call_pp_rule), [`_get_pp_rule`](state/primitives.md#_get_pp_rule), [`brk`](pretty_printer.md#brk), [`color`](pretty_printer.md#color), [`CustomPpEqnRule`](core.md#CustomPpEqnRule), [`pp_effect`](core.md#pp_effect), [`group`](pretty_printer.md#group), [`nest`](pretty_printer.md#nest), [`pretty_print`](pallas/mosaic_gpu/core.md#UnswizzleRef.pretty_print), [`pretty_print`](pallas/mosaic_gpu/core.md#UntilingTransform.pretty_print), [`pretty_print`](state/types.md#BitcastTransform.pretty_print), [`pretty_print`](state/types.md#ReshapeTransform.pretty_print), [`pretty_print`](state/types.md#SelectTransform.pretty_print), [`pretty_print`](state/types.md#TransposeTransform.pretty_print), [`pretty_print`](core.md#Jaxpr.pretty_print)  (+4 more)

## Functions
- `_can_use_color()` — [`L46`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L46)
- `brk(text: str = " ")` — [`L159`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L159) — A break.
- `color(child: Doc, foreground: Color | None = None, background: Color | None = None, intensity: Intensity | None = None)` — [`L182`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L182) — ANSI colors.
- `concat(children: Sequence[Doc])` — [`L154`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L154) — Concatenation of documents.
- `group(doc: Doc)` — [`L167`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L167) — Layout alternative groups.
- `join(sep: Doc, docs: Sequence[Doc])` — [`L213`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L213) — Concatenates `docs`, separated by `sep`.
- `nest(n: int, doc: Doc)` — [`L177`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L177) — Increases the indentation level by `n`.
- `nil()` — [`L130`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L130) — An empty document.
- `source_map(doc: Doc, source: Any)` — [`L196`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L196) — Source mapping.
- `text(text: str, annotation: str | None = None, anchor: str | None = None, href: str | None = None)` — [`L135`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L135) — Literal text.

## Module values
- `CAN_USE_COLOR` — [`L62`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L62)
- `Color` — [`L64`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L64)
- `Intensity` — [`L65`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L65)
- `OutputFormat` — [`L67`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L67)
- `_PPRINT_USE_COLOR` — [`L40`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L40)
- `keyword` — [`L210`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L210)
- `type_annotation` — [`L208`](../../../../../../raw/code/jax/jax/_src/pretty_printer.py#L208)

