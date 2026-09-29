---
title: 'Module: jax/_src/api.py'
type: catalog
provenance: extracted
module: jax/_src/api.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.api`/
symbols:
  jit: jit().
  vmap: vmap().
  device_put: device_put().
  vmap.vmap_f: vmap().vmap_f().
  named_scope: named_scope().
  _check_sharding: _check_sharding().
  jacrev.jacfun: jacrev().jacfun().
  _vjp3_callable: _vjp3_callable().
  device_put_replicated._device_put_replicated: device_put_replicated()._device_put_replicated().
  NotSpecified: NotSpecified#
  value_and_grad.value_and_grad_f: value_and_grad().value_and_grad_f().
  _std_basis: _std_basis().
  linear_transpose: linear_transpose().
  jacfwd.jacfun: jacfwd().jacfun().
  _nan_check_posthook: _nan_check_posthook().
  _mapped_axis_size: _mapped_axis_size().
  _device_get: _device_get().
  clear_backends: clear_backends().
  VJP.__eq__: VJP#__eq__.
  _infer_src_sharding: _infer_src_sharding().
  make_jaxpr.make_jaxpr_f: make_jaxpr().make_jaxpr_f().
  _lift_linearized: _lift_linearized().
  _vjp3_bwd: _vjp3_bwd().
  device_put_sharded._device_put_sharded: device_put_sharded()._device_put_sharded().
  VJP.__call__: VJP#__call__().
  value_and_grad: value_and_grad().
  vjp: vjp().
  _possible_downcast: _possible_downcast().
  make_jaxpr: make_jaxpr().
  eval_shape: eval_shape().
  _check_input_dtype_revderiv: _check_input_dtype_revderiv().
  grad: grad().
  fwd_and_bwd: fwd_and_bwd().
  jvp: jvp().
  linear_transpose.transposed_fun: linear_transpose().transposed_fun().
  _unravel_array_into_pytree: _unravel_array_into_pytree().
  pspec_to_sharding: pspec_to_sharding().
  device_get: device_get().
  VJP.with_refs: VJP#with_refs().
  _update_debug_special_thread_local: _update_debug_special_thread_local().
  jacfwd: jacfwd().
  _jvp: _jvp().
  _insert_pvary: _insert_pvary().
  _jacfwd_unravel: _jacfwd_unravel().
  _jacrev_unravel: _jacrev_unravel().
  _temporary_dtype_exception: _temporary_dtype_exception().
  _vjp_check_ct_avals: _vjp_check_ct_avals().
  jacrev: jacrev().
  device_put_sharded: device_put_sharded().
  device_put_replicated: device_put_replicated().
  named_call: named_call().
  AxisName: AxisName.
  _post_hook_state: _post_hook_state.
  _dtype: _dtype.
  _update_debug_special_global: _update_debug_special_global().
  _check_output_dtype_revderiv: _check_output_dtype_revderiv().
  clear_caches: clear_caches().
  Inline: Inline#
  disable_jit: disable_jit().
  _check_scalar: _check_scalar().
  _check_input_dtype_jacfwd: _check_input_dtype_jacfwd().
  _check_string_compatible_sharding: _check_string_compatible_sharding().
  block_until_ready: block_until_ready().
  fwd_and_bwd.fwd: fwd_and_bwd().fwd().
  clean_up: clean_up().
  _mapped_axis_size._get_argument_type: _mapped_axis_size()._get_argument_type().
  grad.grad_f: grad().grad_f().
  grad.grad_f_aux: grad().grad_f_aux().
  _check_output_dtype_grad: _check_output_dtype_grad.
  _check_output_dtype_jacfwd: _check_output_dtype_jacfwd().
  _check_output_dtype_jacrev: _check_output_dtype_jacrev.
  hessian: hessian().
  _mapped_axis_spec._get_spec: _mapped_axis_spec()._get_spec().
  _check_ema_unmapped_args: _check_ema_unmapped_args().
  _mapped_axis_size._get_axis_size: _mapped_axis_size()._get_axis_size().
  _is_ref: _is_ref().
  _vjp_ct_tree_error: _vjp_ct_tree_error().
  ShapeDtypeStruct: ShapeDtypeStruct.
  VJP.in_tree: VJP#in_tree.
  VJP.out_tree: VJP#out_tree.
  F: F.
  linearize: linearize().
  _mapped_axis_size._all_sizes_index: _mapped_axis_size()._all_sizes_index().
  NotNeeded: NotNeeded#
  GradRef: GradRef#
  DidntWant: DidntWant#
  T: T.
  config_ext: config_ext.
  Inline.AUTO: Inline#AUTO.
  float0: float0.
  _check_input_dtype_grad: _check_input_dtype_grad.
  jacobian: jacobian().
  _check_input_dtype_jacrev: _check_input_dtype_jacrev.
  _mapped_axis_spec: _mapped_axis_spec().
  _mapped_axis_size.arg_name: _mapped_axis_size().arg_name().
  check_accum: check_accum().
  _ref_aval: _ref_aval().
  copy_to_host_async: copy_to_host_async().
  live_arrays: live_arrays().
  fwd_and_bwd.bwd: fwd_and_bwd().bwd().
  VJP.fun: VJP#fun.
  VJP.args_res: VJP#args_res.
  VJP.opaque_residuals: VJP#opaque_residuals.
  VJP: VJP#
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  U: U.
  effects_barrier: effects_barrier().
  block_until_ready.try_to_block: block_until_ready().try_to_block().
  RSpec: RSpec#
  GradValue: GradValue#
  DontWant: DontWant#
  Inline.JAX_EARLY: Inline#JAX_EARLY.
  _split: _split().
  tuptree_map: tuptree_map().
  _vjp_too_many_args: _vjp_too_many_args.
  VJP.jaxpr: VJP#jaxpr.
  Device: Device.
  Inline.XLA_EARLY: Inline#XLA_EARLY.
  Inline.XLA_LATE: Inline#XLA_LATE.
  NotSpecified.__repr__: NotSpecified#__repr__().
  RSpec.idx: RSpec#idx.
  RSpec.primal: RSpec#primal.
  VJP.__hash__: VJP#__hash__.
---
# Module: [`jax/_src/api.py`](../../../../../../raw/code/jax/jax/_src/api.py)

## Classes
### `DidntWant`
- def: [`jax/_src/api.py:1763`](../../../../../../raw/code/jax/jax/_src/api.py#L1763)
- uses (calls/refs, reference-scoped): [`register_dataclass`](tree_util.md#register_dataclass)
- used by: [`Device`](../__init__.md#Device), [`_vjp3_bwd`](api.md#_vjp3_bwd)

### `DontWant`
- def: [`jax/_src/api.py:1758`](../../../../../../raw/code/jax/jax/_src/api.py#L1758)
- used by: [`Device`](../__init__.md#Device), [`_vjp3_callable`](api.md#_vjp3_callable)

### `GradRef`
- def: [`jax/_src/api.py:1754`](../../../../../../raw/code/jax/jax/_src/api.py#L1754)
- uses (calls/refs, reference-scoped): [`register_dataclass`](tree_util.md#register_dataclass)
- used by: [`Device`](../__init__.md#Device), [`_vjp3_bwd`](api.md#_vjp3_bwd)

### `GradValue`
- def: [`jax/_src/api.py:1749`](../../../../../../raw/code/jax/jax/_src/api.py#L1749)
- used by: [`Device`](../__init__.md#Device), [`_vjp3_callable`](api.md#_vjp3_callable)

### `Inline`  ·  implements/extends Enum
- def: [`jax/_src/api.py:107`](../../../../../../raw/code/jax/jax/_src/api.py#L107)
- signature: `class Inline(enum.Enum):`
- members:
  - `AUTO` — [`L111`](../../../../../../raw/code/jax/jax/_src/api.py#L111)
  - `JAX_EARLY` — [`L108`](../../../../../../raw/code/jax/jax/_src/api.py#L108)
  - `XLA_EARLY` — [`L109`](../../../../../../raw/code/jax/jax/_src/api.py#L109)
  - `XLA_LATE` — [`L110`](../../../../../../raw/code/jax/jax/_src/api.py#L110)
- uses (calls/refs, reference-scoped): [`api_boundary`](traceback_util.md#api_boundary)
- used by: [`jit`](api.md#jit), [`_parse_jit_arguments`](pjit.md#_parse_jit_arguments), [`_pjit_lowering`](pjit.md#_pjit_lowering), [`pjit_staging_rule`](pjit.md#pjit_staging_rule), [`make_jit`](pjit.md#make_jit), [`inline`](pjit.md#PjitInfo.inline)

### `NotNeeded`
- def: [`jax/_src/api.py:1745`](../../../../../../raw/code/jax/jax/_src/api.py#L1745)
- uses (calls/refs, reference-scoped): [`register_dataclass`](tree_util.md#register_dataclass)
- used by: [`_vjp3_callable`](api.md#_vjp3_callable)

### `NotSpecified`
- def: [`jax/_src/api.py:162`](../../../../../../raw/code/jax/jax/_src/api.py#L162)
- doc: Sentinel for use in jax.jit
- signature: `class NotSpecified:`
- protocol/private: `__repr__`[`L164`](../../../../../../raw/code/jax/jax/_src/api.py#L164)
- used by: [`kernel`](pallas/mosaic_gpu/core.md#kernel), [`load`](pallas/mosaic_gpu/primitives.md#load), [`kernel`](pallas/helpers.md#kernel)

### `RSpec`
- def: [`jax/_src/api.py:1672`](../../../../../../raw/code/jax/jax/_src/api.py#L1672)
- members:
  - `idx` — [`L1673`](../../../../../../raw/code/jax/jax/_src/api.py#L1673)
  - `primal` — [`L1674`](../../../../../../raw/code/jax/jax/_src/api.py#L1674)

### `VJP`
- def: [`jax/_src/api.py:1768`](../../../../../../raw/code/jax/jax/_src/api.py#L1768)
- members:
  - `with_refs(self, *maybe_ct_refs)` — [`L1783`](../../../../../../raw/code/jax/jax/_src/api.py#L1783)
  - `args_res` — [`L1772`](../../../../../../raw/code/jax/jax/_src/api.py#L1772)
  - `fun` — [`L1769`](../../../../../../raw/code/jax/jax/_src/api.py#L1769)
  - `in_tree` — [`L1770`](../../../../../../raw/code/jax/jax/_src/api.py#L1770)
  - `jaxpr` — [`L1774`](../../../../../../raw/code/jax/jax/_src/api.py#L1774)
  - `opaque_residuals` — [`L1773`](../../../../../../raw/code/jax/jax/_src/api.py#L1773)
  - `out_tree` — [`L1771`](../../../../../../raw/code/jax/jax/_src/api.py#L1771)
- protocol/private: `__call__`[`L1776`](../../../../../../raw/code/jax/jax/_src/api.py#L1776), `__eq__`[`L1789`](../../../../../../raw/code/jax/jax/_src/api.py#L1789), `__hash__`[`L1788`](../../../../../../raw/code/jax/jax/_src/api.py#L1788)
- uses (calls/refs, reference-scoped): [`api_boundary`](traceback_util.md#api_boundary), [`PyTreeDef`](tree_util.md#PyTreeDef), [`register_pytree_node`](tree_util.md#register_pytree_node), [`_vjp_too_many_args`](api.md#_vjp_too_many_args)

## Functions
- `_all_sizes_index(sz)` — [`L1329`](../../../../../../raw/code/jax/jax/_src/api.py#L1329)
- `_check_ema_unmapped_args(ema, args_flat, in_axes_flat)` — [`L1237`](../../../../../../raw/code/jax/jax/_src/api.py#L1237)
- `_check_input_dtype_jacfwd(holomorphic: bool, x: Any)` — [`L745`](../../../../../../raw/code/jax/jax/_src/api.py#L745)
- `_check_input_dtype_revderiv(name, holomorphic, allow_int, x)` — [`L569`](../../../../../../raw/code/jax/jax/_src/api.py#L569)
- `_check_output_dtype_jacfwd(holomorphic, x)` — [`L762`](../../../../../../raw/code/jax/jax/_src/api.py#L762)
- `_check_output_dtype_revderiv(name, holomorphic, x)` — [`L590`](../../../../../../raw/code/jax/jax/_src/api.py#L590)
- `_check_scalar(x)` — [`L556`](../../../../../../raw/code/jax/jax/_src/api.py#L556)
- `_check_sharding(aval, s)` — [`L2017`](../../../../../../raw/code/jax/jax/_src/api.py#L2017)
- `_check_string_compatible_sharding(s)` — [`L2004`](../../../../../../raw/code/jax/jax/_src/api.py#L2004) — Checks if target devices are compatible with string arrays.
- `_device_get(x)` — [`L2271`](../../../../../../raw/code/jax/jax/_src/api.py#L2271)
- `_device_put_replicated(x)` — [`L2253`](../../../../../../raw/code/jax/jax/_src/api.py#L2253)
- `_device_put_sharded(*xs)` — [`L2195`](../../../../../../raw/code/jax/jax/_src/api.py#L2195)
- `_get_argument_type(x)` — [`L1291`](../../../../../../raw/code/jax/jax/_src/api.py#L1291)
- `_get_axis_size(name: str, x, axis: int)` — [`L1261`](../../../../../../raw/code/jax/jax/_src/api.py#L1261)
- `_get_spec(arg, i)` — [`L1215`](../../../../../../raw/code/jax/jax/_src/api.py#L1215)
- `_infer_src_sharding(src, x, x_aval)` — [`L1988`](../../../../../../raw/code/jax/jax/_src/api.py#L1988)
- `_insert_pvary(basis, leaf)` — [`L907`](../../../../../../raw/code/jax/jax/_src/api.py#L907)
- `_is_ref(x)` — [`L1679`](../../../../../../raw/code/jax/jax/_src/api.py#L1679)
- `_jacfwd_unravel(input_pytree, output_pytree_leaf, arr)` — [`L926`](../../../../../../raw/code/jax/jax/_src/api.py#L926)
- `_jacrev_unravel(output_pytree, input_pytree_leaf, arr)` — [`L935`](../../../../../../raw/code/jax/jax/_src/api.py#L935)
- `_jvp(fun: Callable, primals, tangents, has_aux=False)` — [`L1399`](../../../../../../raw/code/jax/jax/_src/api.py#L1399)
- `_lift_linearized(jaxpr, in_avals, out_avals, out_known, consts, *tangents)` — [`L1511`](../../../../../../raw/code/jax/jax/_src/api.py#L1511)
- `_mapped_axis_size(fn, tree, vals, dims, name, axis_size=None)` — [`L1250`](../../../../../../raw/code/jax/jax/_src/api.py#L1250)
- `_mapped_axis_spec(args_flat, in_axes_flat)` — [`L1214`](../../../../../../raw/code/jax/jax/_src/api.py#L1214)
- `_nan_check_posthook(fun, args, kwargs, output)` — [`L114`](../../../../../../raw/code/jax/jax/_src/api.py#L114) — Hook function called by the C++ jit/pmap to perform NaN checking.
- `_possible_downcast(x, example, spec)` — [`L942`](../../../../../../raw/code/jax/jax/_src/api.py#L942)
- `_ref_aval(a)` — [`L1686`](../../../../../../raw/code/jax/jax/_src/api.py#L1686)
- `_split(x, indices, axis)` — [`L974`](../../../../../../raw/code/jax/jax/_src/api.py#L974)
- `_std_basis(pytree)` — [`L912`](../../../../../../raw/code/jax/jax/_src/api.py#L912)
- `_temporary_dtype_exception(a, a_)` — [`L1550`](../../../../../../raw/code/jax/jax/_src/api.py#L1550)
- `_unravel_array_into_pytree(pytree, axis, example, arr, specs)` — [`L953`](../../../../../../raw/code/jax/jax/_src/api.py#L953) — Unravel an array into a PyTree with a given structure.
- `_update_debug_special_global(_)` — [`L141`](../../../../../../raw/code/jax/jax/_src/api.py#L141)
- `_update_debug_special_thread_local(_)` — [`L147`](../../../../../../raw/code/jax/jax/_src/api.py#L147)
- `_vjp3_bwd(in_tree, out_tree, out_known, jaxpr, out_primal_avals, residuals, maybe_accums, out_ct)` — [`L1656`](../../../../../../raw/code/jax/jax/_src/api.py#L1656)
- `_vjp3_callable(spec, out_known, jaxpr, out_primal_avals, in_tree, out_tree, args_res, opaque_res, *maybe_ct_refs)` — [`L1633`](../../../../../../raw/code/jax/jax/_src/api.py#L1633)
- `_vjp_check_ct_avals(cts, primal_avals)` — [`L1728`](../../../../../../raw/code/jax/jax/_src/api.py#L1728)
- `_vjp_ct_tree_error(jaxpr, out_tree, ct_tree)` — [`L1713`](../../../../../../raw/code/jax/jax/_src/api.py#L1713)
- `arg_name(key_path)` — [`L1304`](../../../../../../raw/code/jax/jax/_src/api.py#L1304)
- `block_until_ready(x)` — [`L2490`](../../../../../../raw/code/jax/jax/_src/api.py#L2490) — Tries to call a ``block_until_ready`` method on pytree leaves.
- `bwd(f_vjp, outgrad)` — [`L674`](../../../../../../raw/code/jax/jax/_src/api.py#L674)
- `check_accum(aval, acc)` — [`L1651`](../../../../../../raw/code/jax/jax/_src/api.py#L1651)
- `clean_up()` — [`L2587`](../../../../../../raw/code/jax/jax/_src/api.py#L2587)
- `clear_backends(_crash=False)` — [`L2553`](../../../../../../raw/code/jax/jax/_src/api.py#L2553) — Clear all backend clients so that new backend clients can be created later.
- `clear_caches()` — [`L2603`](../../../../../../raw/code/jax/jax/_src/api.py#L2603) — Clear all compilation and staging caches.
- `copy_to_host_async(x)` — [`L2527`](../../../../../../raw/code/jax/jax/_src/api.py#L2527) — Tries to call a ``copy_to_host_async`` method on pytree leaves.
- `device_get(x: Any)` — [`L2288`](../../../../../../raw/code/jax/jax/_src/api.py#L2288) — Transfer ``x`` to host.
- `device_put(x, device: None | xc.Device | Sharding | P | Format | Any = None, *, src: None | xc.Device | Sharding | P | Format | Any = None, donate: bool | Any = False, may_alias: bool | None | Any = None)` — [`L2046`](../../../../../../raw/code/jax/jax/_src/api.py#L2046) — Transfers ``x`` to ``device``. — documented in [jax-_src-api](../../../concepts/jax-_src-api.md)
- `device_put_replicated(x: Any, devices: Sequence[xc.Device])` — [`L2219`](../../../../../../raw/code/jax/jax/_src/api.py#L2219) — Transfer array(s) to each specified device and form Array(s).
- `device_put_sharded(shards: Sequence[Any], devices: Sequence[xc.Device])` — [`L2139`](../../../../../../raw/code/jax/jax/_src/api.py#L2139) — Transfer array shards to specified devices and form Array(s).
- `disable_jit(disable: bool = True)` — [`L370`](../../../../../../raw/code/jax/jax/_src/api.py#L370) — Context manager that disables :py:func:`jit` behavior under its dynamic context.
- `effects_barrier()` — [`L2486`](../../../../../../raw/code/jax/jax/_src/api.py#L2486) — Waits until existing functions have completed any side-effects.
- `eval_shape(fun: Callable, *args, **kwargs)` — [`L2329`](../../../../../../raw/code/jax/jax/_src/api.py#L2329) — Compute the shape/dtype of ``fun`` without any FLOPs.
- `fwd(*args, **kwargs)` — [`L671`](../../../../../../raw/code/jax/jax/_src/api.py#L671)
- `fwd_and_bwd(fun: Callable, argnums: int | Sequence[int], has_aux: bool = False, jitted: bool = True)` — [`L613`](../../../../../../raw/code/jax/jax/_src/api.py#L613) — Creates functions ``fwd`` and ``bwd`` corresponding to the forward and
- `grad(fun: Callable, argnums: int | Sequence[int] = 0, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False, reduce_axes: Sequence[AxisName] = ())` — [`L421`](../../../../../../raw/code/jax/jax/_src/api.py#L421) — Creates a function that evaluates the gradient of ``fun``.
- `grad_f(*args, **kwargs)` — [`L475`](../../../../../../raw/code/jax/jax/_src/api.py#L475)
- `grad_f_aux(*args, **kwargs)` — [`L481`](../../../../../../raw/code/jax/jax/_src/api.py#L481)
- `hessian(fun: Callable, argnums: int | Sequence[int] = 0, has_aux: bool = False, holomorphic: bool = False)` — [`L840`](../../../../../../raw/code/jax/jax/_src/api.py#L840) — Hessian of ``fun`` as a dense array.
- `jacfun(*args, **kwargs)` — [`L726`](../../../../../../raw/code/jax/jax/_src/api.py#L726)
- `jacfun(*args, **kwargs)` — [`L814`](../../../../../../raw/code/jax/jax/_src/api.py#L814)
- `jacfwd(fun: Callable, argnums: int | Sequence[int] = 0, has_aux: bool = False, holomorphic: bool = False)` — [`L685`](../../../../../../raw/code/jax/jax/_src/api.py#L685) — Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.
- `jacobian(fun: Callable, argnums: int | Sequence[int] = 0, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False)` — [`L829`](../../../../../../raw/code/jax/jax/_src/api.py#L829) — Alias of :func:`jax.jacrev`.
- `jacrev(fun: Callable, argnums: int | Sequence[int] = 0, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False)` — [`L770`](../../../../../../raw/code/jax/jax/_src/api.py#L770) — Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.
- `jit(fun: Callable,, *, in_shardings: Any = ..., out_shardings: Any = ..., static_argnums: int | Sequence[int] | None = ..., static_argnames: str | Iterable[str] | None = ..., donate_argnums: int | Sequence[int] | None = ..., donate_argnames: str | Iterable[str] | None = ..., keep_unused: bool = ..., device: xc.Device | None = ..., backend: str | None = ..., inline: bool | Inline = ..., compiler_options: dict[str, Any] | None = ...)` — [`L168`](../../../../../../raw/code/jax/jax/_src/api.py#L168)
- `jvp(fun: Callable, primals, tangents, has_aux: bool = False)` — [`L1352`](../../../../../../raw/code/jax/jax/_src/api.py#L1352) — Computes a (forward-mode) Jacobian-vector product of ``fun``.
- `linear_transpose(fun: Callable, *primals, reduce_axes=())` — [`L1799`](../../../../../../raw/code/jax/jax/_src/api.py#L1799) — Transpose a function that is promised to be linear.
- `linearize(fun: Callable, *primals, has_aux: Literal[False] = False)` — [`L1423`](../../../../../../raw/code/jax/jax/_src/api.py#L1423)
- `live_arrays(platform=None)` — [`L2596`](../../../../../../raw/code/jax/jax/_src/api.py#L2596) — Return all live arrays in the backend for `platform`.
- `make_jaxpr(fun: Callable, static_argnums: int | Sequence[int] = (), axis_env: Sequence[tuple[AxisName, int]] | None = None, return_shape: Literal[False] = ...)` — [`L1878`](../../../../../../raw/code/jax/jax/_src/api.py#L1878)
- `make_jaxpr_f(*args, **kwargs)` — [`L1966`](../../../../../../raw/code/jax/jax/_src/api.py#L1966)
- `named_call(fun: F, *, name: str | None = None)` — [`L2401`](../../../../../../raw/code/jax/jax/_src/api.py#L2401) — Adds a user specified name to a function when staging out JAX computations.
- `named_scope(name: str)` — [`L2435`](../../../../../../raw/code/jax/jax/_src/api.py#L2435) — A context manager that adds a user specified name to the JAX name stack.
- `pspec_to_sharding(name, val)` — [`L2035`](../../../../../../raw/code/jax/jax/_src/api.py#L2035)
- `transposed_fun(const, out_cotangent)` — [`L1860`](../../../../../../raw/code/jax/jax/_src/api.py#L1860)
- `try_to_block(x)` — [`L2501`](../../../../../../raw/code/jax/jax/_src/api.py#L2501)
- `tuptree_map(f, treedef, x)` — [`L1676`](../../../../../../raw/code/jax/jax/_src/api.py#L1676)
- `value_and_grad(fun: Callable, argnums: int | Sequence[int] = 0, has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False, reduce_axes: Sequence[AxisName] = ())` — [`L489`](../../../../../../raw/code/jax/jax/_src/api.py#L489) — Create a function that evaluates both ``fun`` and the gradient of ``fun``.
- `value_and_grad_f(*args, **kwargs)` — [`L537`](../../../../../../raw/code/jax/jax/_src/api.py#L537)
- `vjp(fun: Callable[..., T], *primals: Any, has_aux: Literal[False] = False, reduce_axes: Sequence[AxisName] = ())` — [`L1556`](../../../../../../raw/code/jax/jax/_src/api.py#L1556)
- `vmap(fun: F, in_axes: int | None | Sequence[Any] = 0, out_axes: Any = 0, axis_name: AxisName | None = None, axis_size: int | None = None, spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None, sum_match: bool = False)` — [`L982`](../../../../../../raw/code/jax/jax/_src/api.py#L982) — Vectorizing map. Creates a function which maps ``fun`` over argument axes. — documented in [jax-_src-api](../../../concepts/jax-_src-api.md)
- `vmap_f(*args, **kwargs)` — [`L1155`](../../../../../../raw/code/jax/jax/_src/api.py#L1155) — documented in [jax-_src-api](../../../concepts/jax-_src-api.md)

## Module values
- `AxisName` — [`L92`](../../../../../../raw/code/jax/jax/_src/api.py#L92)
- `Device` — [`L94`](../../../../../../raw/code/jax/jax/_src/api.py#L94)
- `F` — [`L98`](../../../../../../raw/code/jax/jax/_src/api.py#L98)
- `ShapeDtypeStruct` — [`L105`](../../../../../../raw/code/jax/jax/_src/api.py#L105)
- `T` — [`L99`](../../../../../../raw/code/jax/jax/_src/api.py#L99)
- `U` — [`L100`](../../../../../../raw/code/jax/jax/_src/api.py#L100)
- `_check_input_dtype_grad` — [`L588`](../../../../../../raw/code/jax/jax/_src/api.py#L588)
- `_check_input_dtype_jacrev` — [`L835`](../../../../../../raw/code/jax/jax/_src/api.py#L835)
- `_check_output_dtype_grad` — [`L610`](../../../../../../raw/code/jax/jax/_src/api.py#L610)
- `_check_output_dtype_jacrev` — [`L836`](../../../../../../raw/code/jax/jax/_src/api.py#L836)
- `_dtype` — [`L90`](../../../../../../raw/code/jax/jax/_src/api.py#L90)
- `_post_hook_state` — [`L136`](../../../../../../raw/code/jax/jax/_src/api.py#L136)
- `_vjp_too_many_args` — [`L1691`](../../../../../../raw/code/jax/jax/_src/api.py#L1691)
- `config_ext` — [`L85`](../../../../../../raw/code/jax/jax/_src/api.py#L85)
- `float0` — [`L160`](../../../../../../raw/code/jax/jax/_src/api.py#L160)
- `unsafe_map` — [`L102`](../../../../../../raw/code/jax/jax/_src/api.py#L102)
- `unsafe_zip` — [`L103`](../../../../../../raw/code/jax/jax/_src/api.py#L103)

