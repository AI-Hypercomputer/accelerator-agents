---
title: 'Module: torchax/tensor.py'
type: catalog
provenance: extracted
module: torchax/tensor.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.tensor`/
symbols:
  Tensor: Tensor#
  Environment.dispatch: Environment#dispatch().
  Environment.load_ops: Environment#load_ops().
  Tensor._elem: Tensor#_elem.
  Environment._to_copy: Environment#_to_copy().
  Environment.manual_seed: Environment#manual_seed().
  Environment.t2j_copy: Environment#t2j_copy().
  Environment.override_op_definition: Environment#override_op_definition().
  Environment.config: Environment#config.
  Environment.param: Environment#param().
  Environment.j2t_copy: Environment#j2t_copy().
  Environment._get_op_or_decomp: Environment#_get_op_or_decomp().
  Environment.override_property: Environment#override_property().
  Environment.to_jax: Environment#to_jax().
  XLAFunctionMode.__torch_function__: XLAFunctionMode#__torch_function__().
  Environment._property: Environment#_property.
  Environment._handle_tensor_constructor: Environment#_handle_tensor_constructor().
  Tensor.torch: Tensor#torch().
  Environment.to_xla: Environment#to_xla().
  Tensor.flatten: Tensor#flatten().
  Tensor.__setitem__: Tensor#__setitem__().
  _none: _none.
  Environment.enable_torch_modes: Environment#enable_torch_modes().
  Environment.disable_torch_modes: Environment#disable_torch_modes().
  Environment.t2j_iso: Environment#t2j_iso().
  log_nested: log_nested().
  Tensor.apply_jax: Tensor#apply_jax().
  Environment._should_use_torchax_tensor: Environment#_should_use_torchax_tensor().
  Environment.j2t_iso: Environment#j2t_iso().
  Tensor.detach: Tensor#detach().
  Tensor.jax: Tensor#jax().
  XLADispatchMode.__torch_dispatch__: XLADispatchMode#__torch_dispatch__().
  RuntimeProperty.__init__: RuntimeProperty#__init__().
  _make_debug_msg: _make_debug_msg().
  RuntimeProperty.prng: RuntimeProperty#prng.
  RuntimeProperty.override: RuntimeProperty#override().
  Environment.prng_key: Environment#prng_key().
  Environment: Environment#
  Environment._dispatch_mode: Environment#_dispatch_mode.
  Tensor.dtype: Tensor#dtype().
  RuntimeProperty: RuntimeProperty#
  _PropertyHolder.content: _PropertyHolder#content.
  Environment._is_same_device: Environment#_is_same_device().
  Environment.is_not_torchax_tensor: Environment#is_not_torchax_tensor().
  Environment._move_one_value: Environment#_move_one_value().
  Environment.v2t_iso: Environment#v2t_iso().
  Environment.to_tensor: Environment#to_tensor().
  RuntimeProperty.with_default: RuntimeProperty#with_default().
  Environment._decomps: Environment#_decomps.
  Tensor._env: Tensor#_env.
  _name_of_func: _name_of_func().
  Environment.get_and_rotate_prng_key: Environment#get_and_rotate_prng_key().
  Tensor.ndim: Tensor#ndim().
  Tensor.data: Tensor#data().
  Environment._function_mode: Environment#_function_mode.
  Environment.enabled: Environment#enabled.
  OperatorNotFound: OperatorNotFound#
  XLAFunctionMode.env: XLAFunctionMode#env.
  Tensor.__str__: Tensor#__str__().
  Tensor.apply_jax_: Tensor#apply_jax_().
  RuntimeProperty.get_and_rotate_prng_key: RuntimeProperty#get_and_rotate_prng_key().
  Environment._torch_Tensor_to: Environment#_torch_Tensor_to().
  XLADispatchMode.env: XLADispatchMode#env.
  RuntimeProperty.autocast_dtype: RuntimeProperty#autocast_dtype.
  Environment._ops: Environment#_ops.
  Tensor.__new__: Tensor#__new__().
  Tensor.__init__: Tensor#__init__().
  Tensor.__repr__: Tensor#__repr__.
  Tensor.shape: Tensor#shape().
  Tensor.type_as: Tensor#type_as().
  Tensor.numpy: Tensor#numpy().
  Tensor.dim: Tensor#dim().
  Tensor.jax_device: Tensor#jax_device().
  Tensor.tolist: Tensor#tolist().
  Tensor.shard_: Tensor#shard_().
  _PropertyHolder.__init__: _PropertyHolder#__init__().
  Environment.__enter__: Environment#__enter__().
  Environment.__exit__: Environment#__exit__().
  _make_debug_msg._display: _make_debug_msg()._display().
  XLADispatchMode: XLADispatchMode#
  Environment._get_from_dict: Environment#_get_from_dict().
  logger: logger.
  debug_accuracy: debug_accuracy().
  XLAFunctionMode: XLAFunctionMode#
  TENSOR_CONSTRUCTORS: TENSOR_CONSTRUCTORS.
  _None: _None#
  RuntimeProperty.mesh: RuntimeProperty#mesh.
  _PropertyHolder: _PropertyHolder#
  Tensor.__torch_function__: Tensor#__torch_function__.
  Tensor.__torch_dispatch__: Tensor#__torch_dispatch__().
  Tensor.device: Tensor#device().
  XLAFunctionMode.__init__: XLAFunctionMode#__init__().
  XLADispatchMode.__init__: XLADispatchMode#__init__().
  SUPPORTED_JAX_PLATFROM: SUPPORTED_JAX_PLATFROM.
  Environment.__init__: Environment#__init__().
---
# Module: [`torchax/tensor.py`](../../../../../raw/code/torchax/torchax/tensor.py)

## Classes
### `Environment`
- def: [`torchax/tensor.py:371`](../../../../../raw/code/torchax/torchax/tensor.py#L371) — documented in [docs-docs-tutorials-trainingyt](../../concepts/docs-docs-tutorials-trainingyt.md)
- members:
  - `disable_torch_modes(self, *exc)` — [`L655`](../../../../../raw/code/torchax/torchax/tensor.py#L655)
  - `dispatch(self, func, types, args, kwargs)` — [`L572`](../../../../../raw/code/torchax/torchax/tensor.py#L572) — documented in [torchax-tensor](../../concepts/torchax-tensor.md)
  - `enable_torch_modes(self)` — [`L650`](../../../../../raw/code/torchax/torchax/tensor.py#L650) — documented in [torchax](../../concepts/torchax.md)
  - `get_and_rotate_prng_key(self, generator: torch.Generator | None = None)` — [`L530`](../../../../../raw/code/torchax/torchax/tensor.py#L530)
  - `is_not_torchax_tensor(x)` — [`L591`](../../../../../raw/code/torchax/torchax/tensor.py#L591)
  - `j2t_copy(self, args)` — [`L726`](../../../../../raw/code/torchax/torchax/tensor.py#L726) — Convert torch.Tensor in cpu to a jax array — documented in [torchax-tensor](../../concepts/torchax-tensor.md)
  - `j2t_iso(self, jaxarray)` — [`L718`](../../../../../raw/code/torchax/torchax/tensor.py#L718) — Convert jax array to torchax Tensor. — documented in [torchax-tensor](../../concepts/torchax-tensor.md)
  - `load_ops(self)` — [`L453`](../../../../../raw/code/torchax/torchax/tensor.py#L453) — documented in [torchax-tensor](../../concepts/torchax-tensor.md)
  - `manual_seed(self, key)` — [`L413`](../../../../../raw/code/torchax/torchax/tensor.py#L413) — documented in [torchax-tensor](../../concepts/torchax-tensor.md)
  - `override_op_definition(self, op_to_override, op_impl, is_view_op=False)` — [`L748`](../../../../../raw/code/torchax/torchax/tensor.py#L748)
  - `override_property(self, *, mesh=_none, prng=_none, autocast_dtype=_none)` — [`L759`](../../../../../raw/code/torchax/torchax/tensor.py#L759)
  - `param(self)` — [`L410`](../../../../../raw/code/torchax/torchax/tensor.py#L410)
  - `prng_key(self)` — [`L427`](../../../../../raw/code/torchax/torchax/tensor.py#L427)
  - `t2j_copy(self, args)` — [`L737`](../../../../../raw/code/torchax/torchax/tensor.py#L737) — Convert jax array to torch.Tensor in cpu. — documented in [torchax-tensor](../../concepts/torchax-tensor.md)
  - `t2j_iso(self, torchtensors)` — [`L684`](../../../../../raw/code/torchax/torchax/tensor.py#L684) — Convert torchax Tensor to jax array. — documented in [torchax-tensor](../../concepts/torchax-tensor.md)
  - `to_jax(x)` — [`L691`](../../../../../raw/code/torchax/torchax/tensor.py#L691)
  - `to_tensor(x)` — [`L710`](../../../../../raw/code/torchax/torchax/tensor.py#L710)
  - `to_xla(self, torchvalues)` — [`L679`](../../../../../raw/code/torchax/torchax/tensor.py#L679)
  - `v2t_iso(self, views)` — [`L709`](../../../../../raw/code/torchax/torchax/tensor.py#L709) — documented in [torchax-tensor](../../concepts/torchax-tensor.md)
  - `config` — [`L395`](../../../../../raw/code/torchax/torchax/tensor.py#L395) — documented in [torchax-tensor](../../concepts/torchax-tensor.md)
  - `enabled` — [`L397`](../../../../../raw/code/torchax/torchax/tensor.py#L397)
- protocol/private: `__enter__`[`L662`](../../../../../raw/code/torchax/torchax/tensor.py#L662), `__exit__`[`L666`](../../../../../raw/code/torchax/torchax/tensor.py#L666), `__init__`[`L384`](../../../../../raw/code/torchax/torchax/tensor.py#L384), `_decomps`[`L390`](../../../../../raw/code/torchax/torchax/tensor.py#L390), `_dispatch_mode`[`L386`](../../../../../raw/code/torchax/torchax/tensor.py#L386), `_function_mode`[`L385`](../../../../../raw/code/torchax/torchax/tensor.py#L385), `_get_from_dict`[`L475`](../../../../../raw/code/torchax/torchax/tensor.py#L475), `_get_op_or_decomp`[`L474`](../../../../../raw/code/torchax/torchax/tensor.py#L474), `_handle_tensor_constructor`[`L535`](../../../../../raw/code/torchax/torchax/tensor.py#L535), `_is_same_device`[`L496`](../../../../../raw/code/torchax/torchax/tensor.py#L496), `_move_one_value`[`L669`](../../../../../raw/code/torchax/torchax/tensor.py#L669), `_ops`[`L389`](../../../../../raw/code/torchax/torchax/tensor.py#L389), `_property`[`L401`](../../../../../raw/code/torchax/torchax/tensor.py#L401), `_should_use_torchax_tensor`[`L430`](../../../../../raw/code/torchax/torchax/tensor.py#L430), `_to_copy`[`L507`](../../../../../raw/code/torchax/torchax/tensor.py#L507), `_torch_Tensor_to`[`L553`](../../../../../raw/code/torchax/torchax/tensor.py#L553)
- uses (calls/refs, reference-scoped): [`Tensor`](tensor.md#Tensor), [`View`](view.md#View), [`t2j_dtype`](ops/mappings.md#t2j_dtype), [`_elem`](tensor.md#Tensor._elem), [`execute_policy`](amp.md#execute_policy), [`torch`](view.md#View.torch), [`all_aten_ops`](ops/ops_registry.md#all_aten_ops.all_aten_ops), [`autocast_policy`](amp.md#autocast_policy), [`_none`](tensor.md#_none), [`apply_jax`](tensor.md#Tensor.apply_jax), [`jax`](tensor.md#Tensor.jax), [`all_torch_functions`](ops/ops_registry.md#all_torch_functions.all_torch_functions), [`override`](tensor.md#RuntimeProperty.override), [`prng`](tensor.md#RuntimeProperty.prng), [`Operator`](ops/ops_registry.md#Operator), [`RuntimeProperty`](tensor.md#RuntimeProperty), [`content`](tensor.md#_PropertyHolder.content), [`dtype`](tensor.md#Tensor.dtype), [`is_jax_function`](ops/ops_registry.md#Operator.is_jax_function), [`_name_of_func`](tensor.md#_name_of_func), [`OperatorNotFound`](tensor.md#OperatorNotFound), [`is_user_defined`](ops/ops_registry.md#Operator.is_user_defined), [`is_view_op`](ops/ops_registry.md#Operator.is_view_op), [`needs_env`](ops/ops_registry.md#Operator.needs_env), [`get_and_rotate_prng_key`](tensor.md#RuntimeProperty.get_and_rotate_prng_key), [`j2t`](ops/mappings.md#j2t), [`t2j`](ops/mappings.md#t2j), [`DECOMPOSITIONS`](decompositions.md#DECOMPOSITIONS), [`autocast_dtype`](tensor.md#RuntimeProperty.autocast_dtype), [`XLADispatchMode`](tensor.md#XLADispatchMode), [`MUTABLE_DECOMPOSITION`](decompositions.md#MUTABLE_DECOMPOSITION), [`treat_cuda_as_jax_device`](config.md#Configuration.treat_cuda_as_jax_device), [`use_dlpack_for_data_conversion`](config.md#Configuration.use_dlpack_for_data_conversion), [`Configuration`](config.md#Configuration), [`XLAFunctionMode`](tensor.md#XLAFunctionMode), [`_PropertyHolder`](tensor.md#_PropertyHolder), [`TENSOR_CONSTRUCTORS`](tensor.md#TENSOR_CONSTRUCTORS), [`allow_mixed_math_with_scalar_tensor`](config.md#Configuration.allow_mixed_math_with_scalar_tensor), [`debug_accuracy`](tensor.md#debug_accuracy), [`debug_accuracy_for_each_op`](config.md#Configuration.debug_accuracy_for_each_op)  (+2 more)
- used by: [`default_env`](__init__.md#default_env), [`extract_jax`](__init__.md#extract_jax), [`enable_globally`](__init__.md#enable_globally), [`exported_program_to_jax`](export.md#exported_program_to_jax), [`enable_accuracy_mode`](__init__.md#enable_accuracy_mode), [`disable_temporarily`](__init__.md#disable_temporarily), [`jax_func`](__init__.md#extract_jax.jax_func), [`_env`](__init__.md#_env._env), [`torch`](tensor.md#Tensor.torch), [`__setitem__`](tensor.md#Tensor.__setitem__), [`apply_jax`](tensor.md#Tensor.apply_jax), [`enable_performance_mode`](__init__.md#enable_performance_mode), [`disable_globally`](__init__.md#disable_globally), [`autocast`](amp.md#autocast), [`__init__`](tensor.md#Tensor.__init__)  (29 test-only)

### `OperatorNotFound`  ·  implements/extends Exception
- def: [`torchax/tensor.py:41`](../../../../../raw/code/torchax/torchax/tensor.py#L41)
- signature: `class OperatorNotFound(Exception):`
- used by: [`_get_op_or_decomp`](tensor.md#Environment._get_op_or_decomp), [`__torch_function__`](tensor.md#XLAFunctionMode.__torch_function__), [`functional_interpolate`](ops/jtorch.md#functional_interpolate)

### `RuntimeProperty`
- def: [`torchax/tensor.py:333`](../../../../../raw/code/torchax/torchax/tensor.py#L333)
- signature: `class RuntimeProperty:`
- members:
  - `get_and_rotate_prng_key(self)` — [`L354`](../../../../../raw/code/torchax/torchax/tensor.py#L354)
  - `override(cls, base, *, mesh=_none, prng=_none, autocast_dtype=_none)` — [`L344`](../../../../../raw/code/torchax/torchax/tensor.py#L344)
  - `with_default(val, default)` — [`L345`](../../../../../raw/code/torchax/torchax/tensor.py#L345)
  - `autocast_dtype` — [`L336`](../../../../../raw/code/torchax/torchax/tensor.py#L336)
  - `mesh` — [`L334`](../../../../../raw/code/torchax/torchax/tensor.py#L334)
  - `prng` — [`L335`](../../../../../raw/code/torchax/torchax/tensor.py#L335)
- protocol/private: `__init__`[`L338`](../../../../../raw/code/torchax/torchax/tensor.py#L338)
- uses (calls/refs, reference-scoped): [`_none`](tensor.md#_none)
- used by: [`dispatch`](tensor.md#Environment.dispatch), [`manual_seed`](tensor.md#Environment.manual_seed), [`param`](tensor.md#Environment.param), [`override_property`](tensor.md#Environment.override_property), [`_property`](tensor.md#Environment._property), [`prng_key`](tensor.md#Environment.prng_key), [`get_and_rotate_prng_key`](tensor.md#Environment.get_and_rotate_prng_key), [`__init__`](tensor.md#_PropertyHolder.__init__)  (1 test-only)

### `Tensor`  ·  implements/extends Tensor
- def: [`torchax/tensor.py:57`](../../../../../raw/code/torchax/torchax/tensor.py#L57) — documented in [docs-docs-tutorials-trainingyt](../../concepts/docs-docs-tutorials-trainingyt.md)
- signature: `class Tensor(torch.Tensor):`
- members:
  - `apply_jax(self, jax_function, *args, **kwargs)` — [`L181`](../../../../../raw/code/torchax/torchax/tensor.py#L181)
  - `apply_jax_(self, jax_function, *args, **kwargs)` — [`L186`](../../../../../raw/code/torchax/torchax/tensor.py#L186)
  - `data(self)` — [`L172`](../../../../../raw/code/torchax/torchax/tensor.py#L172) — documented in [torchax-interop](../../concepts/torchax-interop.md)
  - `detach(self)` — [`L142`](../../../../../raw/code/torchax/torchax/tensor.py#L142) — documented in [torchax-tensor](../../concepts/torchax-tensor.md)
  - `device(self)` — [`L164`](../../../../../raw/code/torchax/torchax/tensor.py#L164)
  - `dim(self)` — [`L160`](../../../../../raw/code/torchax/torchax/tensor.py#L160)
  - `dtype(self)` — [`L157`](../../../../../raw/code/torchax/torchax/tensor.py#L157)
  - `flatten(self, start_dim=0, end_dim=-1)` — [`L97`](../../../../../raw/code/torchax/torchax/tensor.py#L97)
  - `jax(self)` — [`L150`](../../../../../raw/code/torchax/torchax/tensor.py#L150)
  - `jax_device(self)` — [`L168`](../../../../../raw/code/torchax/torchax/tensor.py#L168)
  - `ndim(self)` — [`L94`](../../../../../raw/code/torchax/torchax/tensor.py#L94)
  - `numpy(self)` — [`L145`](../../../../../raw/code/torchax/torchax/tensor.py#L145)
  - `shape(self)` — [`L90`](../../../../../raw/code/torchax/torchax/tensor.py#L90)
  - `shard_(self, sharding)` — [`L193`](../../../../../raw/code/torchax/torchax/tensor.py#L193)
  - `tolist(self)` — [`L190`](../../../../../raw/code/torchax/torchax/tensor.py#L190)
  - `torch(self)` — [`L153`](../../../../../raw/code/torchax/torchax/tensor.py#L153)
  - `type_as(self, other)` — [`L123`](../../../../../raw/code/torchax/torchax/tensor.py#L123)
- protocol/private: `__init__`[`L79`](../../../../../raw/code/torchax/torchax/tensor.py#L79), `__new__`[`L59`](../../../../../raw/code/torchax/torchax/tensor.py#L59), `__repr__`[`L87`](../../../../../raw/code/torchax/torchax/tensor.py#L87), `__setitem__`[`L108`](../../../../../raw/code/torchax/torchax/tensor.py#L108), `__str__`[`L84`](../../../../../raw/code/torchax/torchax/tensor.py#L84), `__torch_dispatch__`[`L130`](../../../../../raw/code/torchax/torchax/tensor.py#L130), `__torch_function__`[`L127`](../../../../../raw/code/torchax/torchax/tensor.py#L127), `_elem`[`L81`](../../../../../raw/code/torchax/torchax/tensor.py#L81), `_env`[`L82`](../../../../../raw/code/torchax/torchax/tensor.py#L82)
- uses (calls/refs, reference-scoped): [`j2t_copy`](tensor.md#Environment.j2t_copy), [`t2j_iso`](tensor.md#Environment.t2j_iso), [`j2t_iso`](tensor.md#Environment.j2t_iso), [`Environment`](tensor.md#Environment), [`j2t_dtype`](ops/mappings.md#j2t_dtype), [`_shape_static_boolean_index_put`](ops/jaten.md#_shape_static_boolean_index_put), [`logger`](tensor.md#logger)
- used by: [`_to_copy`](tensor.md#Environment._to_copy), [`update`](view.md#View.update), [`manual_seed`](tensor.md#Environment.manual_seed), [`_jax_view`](interop.md#_jax_view), [`_torch_view`](interop.md#_torch_view), [`to_jax`](tensor.md#Environment.to_jax), [`__call__`](ops/op_base.md#OutVariant.__call__), [`getitem`](ops/jtorch.md#getitem), [`torch`](view.md#View.torch), [`_handle_tensor_constructor`](tensor.md#Environment._handle_tensor_constructor), [`_as_tensor`](ops/jtorch.md#_as_tensor), [`j2t_iso`](tensor.md#Environment.j2t_iso), [`_move_one_value`](tensor.md#Environment._move_one_value), [`is_not_torchax_tensor`](tensor.md#Environment.is_not_torchax_tensor), [`to_jax_array`](checkpoint.md#_to_jax.to_jax_array)  (48 test-only)

### `XLADispatchMode`  ·  implements/extends TorchDispatchMode
- def: [`torchax/tensor.py:273`](../../../../../raw/code/torchax/torchax/tensor.py#L273) — documented in [docs-docs-tutorials-trainingyt](../../concepts/docs-docs-tutorials-trainingyt.md)
- signature: `class XLADispatchMode(torch_dispatch.TorchDispatchMode):`
- members:
  - `env` — [`L275`](../../../../../raw/code/torchax/torchax/tensor.py#L275)
- protocol/private: `__init__`[`L274`](../../../../../raw/code/torchax/torchax/tensor.py#L274), `__torch_dispatch__`[`L277`](../../../../../raw/code/torchax/torchax/tensor.py#L277)
- uses (calls/refs, reference-scoped): [`log_nested`](tensor.md#log_nested), [`_make_debug_msg`](tensor.md#_make_debug_msg)
- used by: [`_dispatch_mode`](tensor.md#Environment._dispatch_mode)  (1 test-only)

### `XLAFunctionMode`  ·  implements/extends TorchFunctionMode
- def: [`torchax/tensor.py:248`](../../../../../raw/code/torchax/torchax/tensor.py#L248) — documented in [docs-docs-tutorials-trainingyt](../../concepts/docs-docs-tutorials-trainingyt.md)
- doc: Context manager that dispatches torch function calls to JAX.
- signature: `class XLAFunctionMode(torch.overrides.TorchFunctionMode):`
- members:
  - `env` — [`L252`](../../../../../raw/code/torchax/torchax/tensor.py#L252)
- protocol/private: `__init__`[`L251`](../../../../../raw/code/torchax/torchax/tensor.py#L251), `__torch_function__`[`L254`](../../../../../raw/code/torchax/torchax/tensor.py#L254)
- uses (calls/refs, reference-scoped): [`log_nested`](tensor.md#log_nested), [`_make_debug_msg`](tensor.md#_make_debug_msg), [`_name_of_func`](tensor.md#_name_of_func), [`OperatorNotFound`](tensor.md#OperatorNotFound)
- used by: [`_function_mode`](tensor.md#Environment._function_mode)

### `_None`
- def: [`torchax/tensor.py:324`](../../../../../raw/code/torchax/torchax/tensor.py#L324)
- doc: A sentinal type to allow None as normal value.
- signature: `class _None:`
- used by: [`_none`](tensor.md#_none)

### `_PropertyHolder`  ·  implements/extends local
- def: [`torchax/tensor.py:361`](../../../../../raw/code/torchax/torchax/tensor.py#L361)
- doc: Thread local storage for runtime properties.
- signature: `class _PropertyHolder(threading.local):`
- members:
  - `content` — [`L368`](../../../../../raw/code/torchax/torchax/tensor.py#L368)
- protocol/private: `__init__`[`L367`](../../../../../raw/code/torchax/torchax/tensor.py#L367)
- uses (calls/refs, reference-scoped): [`RuntimeProperty`](tensor.md#RuntimeProperty)
- used by: [`manual_seed`](tensor.md#Environment.manual_seed), [`param`](tensor.md#Environment.param), [`override_property`](tensor.md#Environment.override_property), [`_property`](tensor.md#Environment._property)

## Functions
- `_display(a)` — [`L229`](../../../../../raw/code/torchax/torchax/tensor.py#L229)
- `_make_debug_msg(is_dispatch, log_args, func, args, kwargs)` — [`L228`](../../../../../raw/code/torchax/torchax/tensor.py#L228)
- `_name_of_func(func)` — [`L296`](../../../../../raw/code/torchax/torchax/tensor.py#L296)
- `debug_accuracy(func, args, kwargs, current_output)` — [`L197`](../../../../../raw/code/torchax/torchax/tensor.py#L197)
- `log_nested(env, message)` — [`L46`](../../../../../raw/code/torchax/torchax/tensor.py#L46)

## Module values
- `SUPPORTED_JAX_PLATFROM` — [`L321`](../../../../../raw/code/torchax/torchax/tensor.py#L321)
- `TENSOR_CONSTRUCTORS` — [`L303`](../../../../../raw/code/torchax/torchax/tensor.py#L303)
- `_none` — [`L330`](../../../../../raw/code/torchax/torchax/tensor.py#L330)
- `logger` — [`L38`](../../../../../raw/code/torchax/torchax/tensor.py#L38)

