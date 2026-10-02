---
title: 'Module: qwix/_src/interception.py'
type: catalog
provenance: extracted
module: qwix/_src/interception.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.interception`/
symbols:
  Interceptor: Interceptor#
  Interceptor.mapping: Interceptor#mapping.
  Interceptor.id: Interceptor#id.
  _InterceptionManager.activate_interceptor: _InterceptionManager#activate_interceptor().
  wrap_func_intercepted: wrap_func_intercepted().
  _InterceptionManager.deactivate_interceptor: _InterceptionManager#deactivate_interceptor().
  interception_manager: interception_manager.
  _InterceptionManager._maybe_apply_interception: _InterceptionManager#_maybe_apply_interception().
  _InterceptionManager._intercepted_threads: _InterceptionManager#_intercepted_threads.
  wrap_func_intercepted.wrapper: wrap_func_intercepted().wrapper().
  _InterceptionManager._on_intercepted_called: _InterceptionManager#_on_intercepted_called().
  _preprocess_interceptor: _preprocess_interceptor().
  _InterceptionManager.is_active: _InterceptionManager#is_active().
  _InterceptionManager._interceptors: _InterceptionManager#_interceptors.
  _InterceptionManager._maybe_remove_interception: _InterceptionManager#_maybe_remove_interception().
  disable_interceptions: disable_interceptions().
  disable_interceptions.wrapper: disable_interceptions().wrapper().
  _InterceptionManager._lock: _InterceptionManager#_lock.
  has_attribute: has_attribute().
  _InterceptionManager._original_fns: _InterceptionManager#_original_fns.
  _InterceptionManager.disable_interception: _InterceptionManager#disable_interception().
  _InterceptionManager.enable_interception: _InterceptionManager#enable_interception().
  _fn_to_code: _fn_to_code().
  Function.Function: Function.Function.
  PRIMITIVE_BIND_KEY: PRIMITIVE_BIND_KEY.
  Interceptor.__getitem__: Interceptor#__getitem__().
  get_attribute: get_attribute().
  _fn_to_code.wrapper: _fn_to_code().wrapper().
  _resolve_path: _resolve_path().
  Interceptor.__iter__: Interceptor#__iter__().
  Interceptor.__len__: Interceptor#__len__().
  Interceptor.__hash__: Interceptor#__hash__().
  _copy_fn: _copy_fn().
  _InterceptionManager: _InterceptionManager#
  _InterceptionManager.__init__: _InterceptionManager#__init__().
---
# Module: [`qwix/_src/interception.py`](../../../../../../raw/code/qwix/qwix/_src/interception.py)

## Classes
### `Interceptor`  ·  implements/extends Mapping
- def: [`qwix/_src/interception.py:35`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L35) — documented in [qwix-_src-interception](../../../concepts/qwix-_src-interception.md)
- doc: A container for interception mappings with a stable identity.
- signature: `class Interceptor(collections.abc.Mapping[str, Function]):`
- members:
  - `id` — [`L49`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L49) — documented in [qwix-_src-interception](../../../concepts/qwix-_src-interception.md)
  - `mapping` — [`L48`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L48) — documented in [qwix-_src-interception](../../../concepts/qwix-_src-interception.md)
- protocol/private: `__getitem__`[`L51`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L51), `__hash__`[`L60`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L60), `__iter__`[`L54`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L54), `__len__`[`L57`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L57)
- uses (calls/refs, reference-scoped): [`Function`](interception.md#Function.Function)
- used by: [`activate_interceptor`](interception.md#_InterceptionManager.activate_interceptor), [`wrap_func_intercepted`](interception.md#wrap_func_intercepted), [`deactivate_interceptor`](interception.md#_InterceptionManager.deactivate_interceptor), [`get_interceptors`](providers/odml.md#OdmlQatProvider.get_interceptors), [`_on_intercepted_called`](interception.md#_InterceptionManager._on_intercepted_called), [`get_interceptors`](qconfig.md#QuantizationProvider.get_interceptors), [`_preprocess_interceptor`](interception.md#_preprocess_interceptor), [`is_active`](interception.md#_InterceptionManager.is_active), [`_interceptors`](interception.md#_InterceptionManager._interceptors)  (13 test-only)

### `_InterceptionManager`
- def: [`qwix/_src/interception.py:187`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L187)
- members:
  - `_maybe_apply_interception(self, name: str)` — [`L301`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L301) — Tries to patch a specific Python attribute. — documented in [qwix-_src-interception](../../../concepts/qwix-_src-interception.md)
  - `_maybe_remove_interception(self, name: str)` — [`L340`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L340) — Tries to remove the patch for one name.
  - `_on_intercepted_called(self, name: str, args, kwargs)` — [`L361`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L361) — Called when an intercepted function is called. — documented in [qwix-_src-interception](../../../concepts/qwix-_src-interception.md)
  - `activate_interceptor(self, interceptor: Interceptor)` — [`L240`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L240) — Activates the interceptor for the current thread. — documented in [qwix-_src-interception](../../../concepts/qwix-_src-interception.md)
  - `deactivate_interceptor(self, interceptor: Interceptor)` — [`L276`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L276) — Deactivates the interceptor for the current thread. — documented in [qwix-_src-interception](../../../concepts/qwix-_src-interception.md)
  - `disable_interception(self)` — [`L389`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L389) — Disables all interceptions for the current thread and returns the list of disabled interceptors.
  - `enable_interception(self, interceptor_ids: list[int])` — [`L400`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L400) — Enables the given interceptions for the current thread.
  - `is_active(self, interceptor: Interceptor)` — [`L234`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L234) — Returns whether the interceptor is active for the current thread. — documented in [qwix-_src-interception](../../../concepts/qwix-_src-interception.md)
- protocol/private: `__init__`[`L214`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L214), `_intercepted_threads`[`L232`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L232), `_interceptors`[`L226`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L226), `_lock`[`L216`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L216), `_original_fns`[`L221`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L221)
- uses (calls/refs, reference-scoped): [`get`](aux_data.md#get), [`Interceptor`](interception.md#Interceptor), [`id`](interception.md#Interceptor.id), [`_fn_to_code`](interception.md#_fn_to_code), [`_resolve_path`](interception.md#_resolve_path), [`_copy_fn`](interception.md#_copy_fn)
- used by: [`interception_manager`](interception.md#interception_manager), [`wrapper`](interception.md#wrap_func_intercepted.wrapper), [`wrapper`](interception.md#disable_interceptions.wrapper)  (3 test-only)

## Functions
- `_copy_fn(fn: types.FunctionType)` — [`L436`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L436) — Constructs a new function object with the same attributes as the given one.
- `_fn_to_code(fn: Function)` — [`L413`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L413) — Returns a code object that takes 0 freevars. — documented in [qwix-_src-interception](../../../concepts/qwix-_src-interception.md)
- `_preprocess_interceptor(interceptor: Interceptor, disable_jit: bool)` — [`L64`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L64) — Preprocesses the interceptor by rewriting keys. — documented in [qwix-_src-interception](../../../concepts/qwix-_src-interception.md)
- `_resolve_path(name: str)` — [`L483`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L483) — Resolves a dot-separated name into (parent_object, attribute_name).
- `disable_interceptions(fn)` — [`L454`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L454) — Return the function with interceptions disabled when called. — documented in [qwix-_src-core-conv_general_qt](../../../concepts/qwix-_src-core-conv_general_qt.md)
- `get_attribute(name: str)` — [`L468`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L468) — Returns the attribute from the name.
- `has_attribute(name: str)` — [`L474`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L474) — Returns if the module exists and has the attribute.
- `wrap_func_intercepted(func: Function, get_interceptor: Callable[[], Interceptor], *, disable_jit: bool, input_transform: Callable[[Any, Any], tuple[Any, Any]] = lambda *x: x, output_transform: Callable[[Any], Any] = lambda x: x, should_intercept: Callable[[], bool] = lambda : True)` — [`L124`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L124) — Wraps a function to execute within an active interception scope. — documented in [qwix-_src-interception](../../../concepts/qwix-_src-interception.md)
- `wrapper(*args, **kwargs)` — [`L158`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L158)
- `wrapper(*args, **kwargs)` — [`L422`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L422)
- `wrapper(*args, **kwargs)` — [`L458`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L458)

## Module values
- `Function` — [`L28`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L28)
- `PRIMITIVE_BIND_KEY` — [`L31`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L31)
- `interception_manager` — [`L410`](../../../../../../raw/code/qwix/qwix/_src/interception.py#L410)

