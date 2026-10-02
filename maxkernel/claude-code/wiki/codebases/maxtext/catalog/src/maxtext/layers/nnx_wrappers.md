---
title: 'Module: src/maxtext/layers/nnx_wrappers.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/nnx_wrappers.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.nnx_wrappers`/
symbols:
  ToNNX.__call__: ToNNX#__call__().
  ToLinen.__call__: ToLinen#__call__().
  to_linen: to_linen().
  to_linen_class: to_linen_class().
  ToLinen._module_kwargs: ToLinen#_module_kwargs().
  to_linen_class.__init__: to_linen_class().__init__().
  _fix_for_qwix_quantization.wrap.wrapped: _fix_for_qwix_quantization().wrap().wrapped().
  ToLinen.__getattr__: ToLinen#__getattr__().
  to_linen_class.ToLinenPartial: to_linen_class().ToLinenPartial#
  _fix_for_qwix_quantization: _fix_for_qwix_quantization().
  to_linen_var: to_linen_var().
  ToNNX.to_nnx__rngs: ToNNX#to_nnx__rngs.
  ToLinen: ToLinen#
  linen_vars_to_nnx_attrs: linen_vars_to_nnx_attrs().
  linen_rngs_dict: linen_rngs_dict().
  ToLinen.metadata_fn: ToLinen#metadata_fn.
  ToNNX.__getattr__: ToNNX#__getattr__().
  current_linen_module: current_linen_module().
  ToLinen._update_variables: ToLinen#_update_variables().
  _MISSING: _MISSING.
  ToLinen.kwargs: ToLinen#kwargs.
  ToLinen.nnx_module_augment_fn: ToLinen#nnx_module_augment_fn.
  nnx_attrs_to_linen_vars: nnx_attrs_to_linen_vars().
  lazy_init: lazy_init().
  _fix_for_qwix_quantization.wrap: _fix_for_qwix_quantization().wrap().
  _fix_for_qwix_quantization.wrap_setattr: _fix_for_qwix_quantization().wrap_setattr().
  ToLinen._to_linen_var: ToLinen#_to_linen_var().
  ToNNX.to_nnx__module: ToNNX#to_nnx__module.
  ToLinen.nnx_class: ToLinen#nnx_class.
  ToLinen.args: ToLinen#args.
  _Missing: _Missing#
  is_linen_initializing: is_linen_initializing().
  ToNNX.lazy_init: ToNNX#lazy_init().
  to_linen_class.ToLinenPartial.__init_subclass__: to_linen_class().ToLinenPartial#__init_subclass__().
  _set_initializing: _set_initializing().
  _get_module_method: _get_module_method().
  ToLinen.skip_rng: ToLinen#skip_rng.
  M: M.
  is_vanilla_variable: is_vanilla_variable().
  get_col_name: get_col_name().
  to_nnx_var: to_nnx_var().
  _recursive_merge: _recursive_merge().
  _fix_for_qwix_quantization.wrap_setattr.wrapped_setattr: _fix_for_qwix_quantization().wrap_setattr().wrapped_setattr().
  ToLinen.maybe_unbox: ToLinen#maybe_unbox().
  ToNNX: ToNNX#
  ToNNX.__init__: ToNNX#__init__().
---
# Module: [`src/maxtext/layers/nnx_wrappers.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py)

## Classes
### `ToLinen`  ·  implements/extends Module
- def: [`src/maxtext/layers/nnx_wrappers.py:401`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L401)
- doc: A wrapper to turn any NNX module into a Linen module.
- signature: `class ToLinen(linen.Module):`
- members:
  - `_update_variables(self, module)` — [`L563`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L563) — Store the NNX module's graph def and state inside Linen module variables.
  - `maybe_unbox(x)` — [`L482`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L482)
  - `args` — [`L441`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L441)
  - `kwargs` — [`L442`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L442)
  - `metadata_fn` — [`L444`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L444)
  - `nnx_class` — [`L440`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L440)
  - `nnx_module_augment_fn` — [`L447`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L447)
  - `skip_rng` — [`L443`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L443)
- protocol/private: `__call__`[`L450`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L450), `__getattr__`[`L551`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L551), `_module_kwargs`[`L451`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L451), `_to_linen_var`[`L581`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L581)
- uses (calls/refs, reference-scoped): [`ToLinenPartial`](nnx_wrappers.md#to_linen_class.ToLinenPartial), [`_fix_for_qwix_quantization`](nnx_wrappers.md#_fix_for_qwix_quantization), [`to_linen_var`](nnx_wrappers.md#to_linen_var), [`linen_rngs_dict`](nnx_wrappers.md#linen_rngs_dict), [`_get_module_method`](nnx_wrappers.md#_get_module_method)
- used by: [`to_linen`](nnx_wrappers.md#to_linen), [`to_linen_class`](nnx_wrappers.md#to_linen_class), [`__init__`](nnx_wrappers.md#to_linen_class.__init__), [`ToLinenPartial`](nnx_wrappers.md#to_linen_class.ToLinenPartial)

### `ToLinenPartial`  ·  implements/extends ToLinen
- def: [`src/maxtext/layers/nnx_wrappers.py:699`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L699)
- doc: A dynamically created Linen Module that wraps a specific NNX Module.
- signature: `class ToLinenPartial(ToLinen):`
- protocol/private: `__init_subclass__`[`L702`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L702)
- uses (calls/refs, reference-scoped): [`__init__`](nnx_wrappers.md#to_linen_class.__init__), [`ToLinen`](nnx_wrappers.md#ToLinen)
- used by: [`to_linen_class`](nnx_wrappers.md#to_linen_class), [`ToLinen`](nnx_wrappers.md#ToLinen)

### `ToNNX`  ·  implements/extends Module
- def: [`src/maxtext/layers/nnx_wrappers.py:185`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L185)
- doc: A wrapper to turn any Linen module into an NNX module.
- signature: `class ToNNX(Module):`
- members:
  - `lazy_init(self, *args, **kwargs)` — [`L230`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L230) — A shortcut of calling `nnx.bridge.lazy_init()` upon this module.
  - `to_nnx__module` — [`L220`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L220)
  - `to_nnx__rngs` — [`L222`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L222)
- protocol/private: `__call__`[`L244`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L244), `__getattr__`[`L234`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L234), `__init__`[`L215`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L215)
- uses (calls/refs, reference-scoped): [`linen_vars_to_nnx_attrs`](nnx_wrappers.md#linen_vars_to_nnx_attrs), `scope`, [`current_linen_module`](nnx_wrappers.md#current_linen_module), [`lazy_init`](nnx_wrappers.md#lazy_init), [`nnx_attrs_to_linen_vars`](nnx_wrappers.md#nnx_attrs_to_linen_vars), [`_recursive_merge`](nnx_wrappers.md#_recursive_merge), `mutable`, `mutable`

### `_Missing`
- def: [`src/maxtext/layers/nnx_wrappers.py:599`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L599)
- signature: `class _Missing:`
- used by: [`__init__`](nnx_wrappers.md#to_linen_class.__init__), [`_MISSING`](nnx_wrappers.md#_MISSING)

## Functions
- `__init__(self, args=None, kwargs=None, nnx_class=None, skip_rng=None, metadata_fn=None, name=_MISSING, parent=_MISSING, nnx_module_augment_fn=nnx_module_augment_fn, **other_kwargs)` — [`L664`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L664)
- `_fix_for_qwix_quantization(module: Module)` — [`L343`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L343) — Process the nnx module to make it compatible with QWIX quantization.
- `_get_module_method(module, method: tp.Callable[..., Any] | str | None)` — [`L325`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L325) — Get a callable method from the module, or raise TypeError.
- `_recursive_merge(dict1, dict2)` — [`L93`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L93) — Recursively merge two dicts.
- `_set_initializing(module: Module, initializing: bool)` — [`L141`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L141)
- `current_linen_module()` — [`L166`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L166) — Get the current Linen module from the Linen context.
- `get_col_name(keypath: tp.Sequence[Any])` — [`L71`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L71) — Given the keypath of a Flax variable type, return its Linen collection name.
- `is_linen_initializing()` — [`L173`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L173) — Returns True if currently inside a Linen ``init()`` call.
- `is_vanilla_variable(vs: variablelib.Variable)` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L45) — A variables state is vanilla if its metadata is essentially blank.
- `lazy_init(fn: Module | tp.Callable[..., tp.Any], *args, **kwargs)` — [`L147`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L147) — To run through an arbitrary nnx.Module method and initialize all its needed state.
- `linen_rngs_dict(linen_module: linen.Module, add_default: bool = False)` — [`L316`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L316) — Given a module, split out one of its every active RNG key collections.
- `linen_vars_to_nnx_attrs(variables: tp.Mapping[str, Any])` — [`L100`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L100) — Convert a dict of Linen-style variables to NNX variables.
- `nnx_attrs_to_linen_vars(nnx_attrs: dict)` — [`L124`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L124) — Convert a dict of NNX variables (or variable states) to Linen-style variables.
- `to_linen(nnx_class: tp.Callable[..., Module], *args, metadata_fn: tp.Callable[[variablelib.Variable], tp.Any] | None = to_linen_var, name: str | None = None, skip_rng: bool = False, abstract_init: bool = True, **kwargs)` — [`L606`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L606) — Shortcut of `nnx.bridge.ToLinen` if user is not changing any of its default fields.
- `to_linen_class(base_nnx_class: type[M], base_metadata_fn: tp.Callable[[variablelib.Variable], tp.Any] | None = to_linen_var, base_skip_rng: bool = False, nnx_module_augment_fn: tp.Callable[[Module, str | None], Module] | None = None, **partial_kwargs: tp.Any)` — [`L626`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L626) — A dynamically created Linen Module that wraps a specific NNX Module.
- `to_linen_var(vs: variablelib.Variable)` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L59)
- `to_nnx_var(col: str, x: meta.AxisMetadata | Any)` — [`L78`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L78) — Convert a Linen variable to an NNX variable.
- `wrap(call_fn, name: str)` — [`L359`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L359)
- `wrap_setattr(old_setattr)` — [`L375`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L375)
- `wrapped(*args, **kwargs)` — [`L360`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L360)
- `wrapped_setattr(self, name: str, value: Any)` — [`L376`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L376)

## Module values
- `M` — [`L38`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L38)
- `_MISSING` — [`L603`](../../../../../../../raw/code/maxtext/src/maxtext/layers/nnx_wrappers.py#L603)

