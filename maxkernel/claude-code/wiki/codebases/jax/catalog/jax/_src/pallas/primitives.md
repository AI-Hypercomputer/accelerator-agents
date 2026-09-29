---
title: 'Module: jax/_src/pallas/primitives.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/primitives.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.primitives`/
symbols:
  _swap_discharge_rule: _swap_discharge_rule().
  run_scoped: run_scoped().
  program_id: program_id().
  _load_discharge_rule: _load_discharge_rule().
  dot: dot().
  DeviceIdType: DeviceIdType#
  _semaphore_signal_pp_eqn: _semaphore_signal_pp_eqn().
  semaphore_signal: semaphore_signal().
  run_scoped_p: run_scoped_p.
  _jaxpr_call: _jaxpr_call().
  _semaphore_wait_pp_eqn: _semaphore_wait_pp_eqn().
  _jaxpr_call_pp_eqn: _jaxpr_call_pp_eqn().
  _semaphore_signal_abstract_eval: _semaphore_signal_abstract_eval().
  semaphore_wait: semaphore_wait().
  check_sem_avals: check_sem_avals().
  get_global_p: get_global_p.
  _run_scoped_discharge_rule: _run_scoped_discharge_rule().
  semaphore_signal_p: semaphore_signal_p.
  jaxpr_call_p: jaxpr_call_p.
  _jaxpr_call_discharge: _jaxpr_call_discharge().
  semaphore_wait_p: semaphore_wait_p.
  device_id_to_logical: device_id_to_logical().
  Slice: Slice.
  _num_programs_bind_with_trace: _num_programs_bind_with_trace().
  load_p: load_p.
  _load_pp_rule: _load_pp_rule().
  uninitialized_value: uninitialized_value().
  swap_p: swap_p.
  _swap_pp_rule: _swap_pp_rule().
  _run_scoped_lowering_rule: _run_scoped_lowering_rule().
  multiple_of_p: multiple_of_p.
  _pp_device_id: _pp_device_id().
  program_id_bind_with_trace: program_id_bind_with_trace().
  _get_ref_and_transforms: _get_ref_and_transforms().
  _load_abstract_eval: _load_abstract_eval().
  DeviceIdType.MESH: DeviceIdType#MESH.
  load: load().
  _handle_small: _handle_small().
  get_global: get_global().
  semaphore_read_p: semaphore_read_p.
  delay_p: delay_p.
  num_programs: num_programs().
  _swap_abstract_eval: _swap_abstract_eval().
  swap: swap().
  _run_scoped_to_lojax: _run_scoped_to_lojax().
  semaphore_read: semaphore_read().
  _device_id_dict_to_mesh: _device_id_dict_to_mesh().
  program_id_p: program_id_p.
  _semaphore_wait_abstract_eval: _semaphore_wait_abstract_eval().
  _load_jvp: _load_jvp().
  _swap_jvp: _swap_jvp().
  multiple_of: multiple_of().
  _program_id_abstract_eval: _program_id_abstract_eval().
  _num_programs_abstract_eval: _num_programs_abstract_eval().
  num_programs_p: num_programs_p.
  _unpad_values_to_avoid_dynamic_slice_oob_shift: _unpad_values_to_avoid_dynamic_slice_oob_shift.
  _reciprocal_lowering_rule._reciprocal: _reciprocal_lowering_rule()._reciprocal().
  wrap_with_transforms: wrap_with_transforms().
  _transform_semaphore: _transform_semaphore().
  _semaphore_read_discharge_rule: _semaphore_read_discharge_rule().
  _semaphore_signal_discharge_rule: _semaphore_signal_discharge_rule().
  _semaphore_wait_discharge_rule: _semaphore_wait_discharge_rule().
  delay: delay().
  _pad_values_to_avoid_dynamic_slice_oob_shift: _pad_values_to_avoid_dynamic_slice_oob_shift().
  _run_scoped_abstract_eval: _run_scoped_abstract_eval().
  _semaphore_read_abstract_eval: _semaphore_read_abstract_eval().
  _delay_abstract_eval: _delay_abstract_eval().
  _jaxpr_call_abstract_eval: _jaxpr_call_abstract_eval().
  sem_effect: sem_effect.
  reciprocal_p: reciprocal_p.
  reciprocal: reciprocal().
  debug_print: debug_print().
  wrap_with_transforms.wrapped: wrap_with_transforms().wrapped().
  _run_scoped_lowering_rule._lower_fun: _run_scoped_lowering_rule()._lower_fun().
  delay_effect: delay_effect.
  DeviceId: DeviceId.
  _multiple_of_abstract_eval: _multiple_of_abstract_eval().
  _reciprocal_abstract_eval: _reciprocal_abstract_eval().
  _get_global_abstract_eval: _get_global_abstract_eval().
  NDIndexer: NDIndexer.
  store: store().
  check_debug_print_format: check_debug_print_format().
  DeviceIdType.LOGICAL: DeviceIdType#LOGICAL.
  SemaphoreEffect: SemaphoreEffect#
  _reciprocal_lowering_rule: _reciprocal_lowering_rule().
  wrap_with_transforms_old: wrap_with_transforms_old().
  DelayEffect: DelayEffect#
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  _run_scoped_is_high: _run_scoped_is_high().
  _get_global_discharge_rule: _get_global_discharge_rule().
---
# Module: [`jax/_src/pallas/primitives.py`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py)

## Classes
### `DelayEffect`  ·  implements/extends Effect
- def: [`jax/_src/pallas/primitives.py:1322`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1322)
- signature: `class DelayEffect(effects.Effect):`
- used by: [`delay_effect`](primitives.md#delay_effect)

### `DeviceIdType`  ·  implements/extends Enum
- def: [`jax/_src/pallas/primitives.py:891`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L891)
- signature: `class DeviceIdType(enum.Enum):`
- members:
  - `LOGICAL` — [`L893`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L893)
  - `MESH` — [`L892`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L892)
- used by: [`_semaphore_signal_lowering_rule`](mosaic_gpu/primitives.md#_semaphore_signal_lowering_rule), [`_dma_start_lowering_rule`](mosaic/sc_lowering.md#_dma_start_lowering_rule), [`dma_start_partial_discharge_rule`](mosaic/primitives.md#dma_start_partial_discharge_rule), [`_dma_start_lowering_rule`](mosaic/lowering.md#_dma_start_lowering_rule), [`_dma_wait_lowering_rule`](mosaic/sc_lowering.md#_dma_wait_lowering_rule), [`_semaphore_signal_lowering_rule`](mosaic/lowering.md#_semaphore_signal_lowering_rule), [`_device_id_to_logical`](mosaic/lowering.md#_device_id_to_logical), [`_dma_wait_lowering_rule`](mosaic/lowering.md#_dma_wait_lowering_rule), [`semaphore_signal`](primitives.md#semaphore_signal), [`remote_ref`](mosaic_gpu/core.md#remote_ref), [`semaphore_signal`](mosaic_gpu/primitives.md#semaphore_signal), [`_semaphore_signal_abstract_eval`](primitives.md#_semaphore_signal_abstract_eval), [`_get_dma_effects`](mosaic/primitives.md#_get_dma_effects), [`device_id_to_logical`](primitives.md#device_id_to_logical), [`make_async_remote_copy`](mosaic/primitives.md#make_async_remote_copy), [`make_async_copy`](mosaic/primitives.md#make_async_copy), [`async_remote_copy`](mosaic/primitives.md#async_remote_copy), [`_device_id_to_logical`](mosaic_gpu/lowering.md#_device_id_to_logical), [`_device_id_to_logical`](mosaic/interpret/utils.md#_device_id_to_logical), [`device_id_type`](mosaic/primitives.md#AsyncCopyDescriptor.device_id_type), [`device_id_type`](mosaic_gpu/core.md#PeerMemRef.device_id_type)

### `SemaphoreEffect`  ·  implements/extends Effect
- def: [`jax/_src/pallas/primitives.py:985`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L985)
- signature: `class SemaphoreEffect(effects.Effect):`
- used by: [`sem_effect`](primitives.md#sem_effect)

## Functions
- `_delay_abstract_eval(nanos)` — [`L1330`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1330)
- `_device_id_dict_to_mesh(mesh_context: pallas_utils.MeshInfo | None, device_id_dict, get_axis_index)` — [`L1223`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1223)
- `_get_global_abstract_eval(*, what)` — [`L869`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L869)
- `_get_global_discharge_rule(in_avals, out_avals, *, what)` — [`L873`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L873)
- `_get_ref_and_transforms(ref)` — [`L885`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L885)
- `_handle_small(dtype: jax_typing.DTypeLike)` — [`L495`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L495) — Ugly workaround to support types that don't allow automatic promotion.
- `_jaxpr_call(jaxpr: jax_core.Jaxpr, *refs: state_types.AbstractRef | state_types.TransformedRef, program_ids: Sequence[jax_typing.Array | None])` — [`L1430`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1430) — Internal primitive for calling a kernel jaxpr inside ``emit_pipeline``.
- `_jaxpr_call_abstract_eval(*args, jaxpr: jax_core.Jaxpr, **params)` — [`L1345`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1345)
- `_jaxpr_call_discharge(flat_should_discharge, in_avals, out_avals, *flat_args, jaxpr, ref_treedefs, program_ids_treedef)` — [`L1390`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1390)
- `_jaxpr_call_pp_eqn(eqn: jax_core.JaxprEqn, context: jax_core.JaxprPpContext, settings: jax_core.JaxprPpSettings)` — [`L1355`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1355)
- `_load_abstract_eval(*avals_flat, args_tree, **_)` — [`L149`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L149)
- `_load_discharge_rule(in_avals, out_avals, *args_flat, args_tree, **_)` — [`L266`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L266)
- `_load_jvp(primals, tangents, args_tree, **params)` — [`L197`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L197)
- `_load_pp_rule(eqn, context, settings)` — [`L168`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L168)
- `_lower_fun(*lower_fun_args)` — [`L830`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L830)
- `_multiple_of_abstract_eval(aval, **_)` — [`L142`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L142)
- `_num_programs_abstract_eval(**_)` — [`L117`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L117)
- `_num_programs_bind_with_trace(trace, _, avals, params)` — [`L101`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L101)
- `_pad_values_to_avoid_dynamic_slice_oob_shift(value, slice_sizes, unpad=False)` — [`L236`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L236) — DynamicSlice and DynamicUpdateSlice adjust the start index in cases where the
- `_pp_device_id(device_id, context)` — [`L1069`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1069)
- `_program_id_abstract_eval(**_)` — [`L92`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L92)
- `_reciprocal(x, *, approx=False)` — [`L583`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L583)
- `_reciprocal_abstract_eval(x, *, approx, full_range)` — [`L569`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L569)
- `_reciprocal_lowering_rule(ctx: mlir.LoweringRuleContext, x, *, approx=False, full_range=True)` — [`L574`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L574)
- `_run_scoped_abstract_eval(*args, jaxpr, collective_axes, **_)` — [`L743`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L743)
- `_run_scoped_discharge_rule(should_discharge, in_avals, out_avals, *args_flat, jaxpr, collective_axes, ref_transforms, **_)` — [`L760`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L760)
- `_run_scoped_is_high(*avals, jaxpr, **params)` — [`L678`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L678)
- `_run_scoped_lowering_rule(ctx, *args, jaxpr, collective_axes, **_)` — [`L815`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L815)
- `_run_scoped_to_lojax(*args, jaxpr, **params)` — [`L683`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L683)
- `_semaphore_read_abstract_eval(*avals, args_tree)` — [`L956`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L956)
- `_semaphore_read_discharge_rule(in_avals, out_avals, *flat_args, args_tree)` — [`L963`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L963)
- `_semaphore_signal_abstract_eval(*avals, args_tree, device_id_type: DeviceIdType)` — [`L1035`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1035)
- `_semaphore_signal_discharge_rule(in_avals, out_avals, *flat_args, args_tree, device_id_type)` — [`L1124`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1124)
- `_semaphore_signal_pp_eqn(eqn: jax_core.JaxprEqn, context: jax_core.JaxprPpContext, settings: jax_core.JaxprPpSettings)` — [`L1094`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1094)
- `_semaphore_wait_abstract_eval(*avals, args_tree)` — [`L1168`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1168)
- `_semaphore_wait_discharge_rule(in_avals, out_avals, *flat_args, args_tree)` — [`L1203`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1203)
- `_semaphore_wait_pp_eqn(eqn: jax_core.JaxprEqn, context: jax_core.JaxprPpContext, settings: jax_core.JaxprPpSettings)` — [`L1177`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1177)
- `_swap_abstract_eval(*avals_flat, args_tree, **_)` — [`L311`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L311)
- `_swap_discharge_rule(in_avals, out_avals, *args_flat, args_tree, **_)` — [`L390`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L390)
- `_swap_jvp(primals, tangents, *, args_tree, **params)` — [`L368`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L368)
- `_swap_pp_rule(eqn, context, settings)` — [`L333`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L333)
- `_transform_semaphore(ref_value, transforms, ref_aval)` — [`L924`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L924) — Helper function for indexing into a semaphore during state_discharge.
- `check_debug_print_format(fmt: str, *args: jax_typing.ArrayLike)` — [`L620`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L620)
- `check_sem_avals(sem_aval, sem_transforms_avals, name, allowed_semaphore_types=None)` — [`L896`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L896)
- `debug_print(fmt: str, *args: jax_typing.ArrayLike)` — [`L596`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L596) — Prints values from inside a Pallas kernel.
- `delay(nanos: int | jax_typing.Array)` — [`L1335`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1335) — Sleeps for the given number of nanoseconds.
- `device_id_to_logical(mesh_context: pallas_utils.MeshInfo | None, device_id: Any, device_id_type: DeviceIdType, get_axis_index: Callable[[Any], Any])` — [`L1275`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1275) — Normalizes a device id into a logical device id and axes that don't correspond to JAX mesh axes.
- `dot(a, b, trans_a: bool = False, trans_b: bool = False, allow_tf32: bool | None = None, precision=None)` — [`L504`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L504) — Computes the dot product of two arrays.
- `get_global(what: pallas_core.ScratchShape)` — [`L848`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L848) — Returns a global reference that persists across all kernel invocations.
- `load(x_ref_or_view, idx, *, mask=None, other=None, cache_modifier=None, eviction_policy=None, volatile=False)` — [`L439`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L439) — Returns an array loaded from the given index.
- `multiple_of(x: jax_typing.Array, values: Sequence[int] | int)` — [`L125`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L125) — A compiler hint that asserts a value is a static multiple of another.
- `num_programs(axis: int)` — [`L97`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L97) — Returns the size of the grid along the given axis.
- `program_id(axis: int)` — [`L63`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L63) — Returns the kernel execution position along the given axis of the grid.
- `program_id_bind_with_trace(trace, _, avals, params)` — [`L77`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L77)
- `reciprocal(x, *, approx=False, full_range=True)` — [`L551`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L551) — Computes the reciprocal of an array.
- `run_scoped(f: Callable[..., Any], *types: Any, collective_axes: Hashable | tuple[Hashable, ...] = (), **kw_types: Any)` — [`L690`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L690) — Calls the function with allocated references and returns the result.
- `semaphore_read(sem_or_view)` — [`L941`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L941) — Reads the value of a semaphore.
- `semaphore_signal(sem_or_view, inc: int | jax_typing.Array = 1, *, device_id: DeviceId = None, device_id_type: DeviceIdType = DeviceIdType.MESH, core_index: int | jax_typing.Array | None = None)` — [`L997`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L997) — Increments the value of a semaphore.
- `semaphore_wait(sem_or_view, value: int | jax_typing.Array = 1, *, decrement: bool = True)` — [`L1150`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1150) — Blocks execution of the current thread until a semaphore reaches a value.
- `store(x_ref_or_view, idx, val, *, mask=None, eviction_policy=None)` — [`L486`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L486) — Stores a value at the given index.
- `swap(x_ref_or_view, idx, val, *, mask=None, eviction_policy=None, _function_name="swap")` — [`L469`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L469) — Swaps the value at the given index and returns the old value.
- `uninitialized_value(shape, dtype)` — [`L220`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L220)
- `wrap_with_transforms(fun: Callable, ref_transforms: tuple[tuple[state_types.Transform, ...], ...])` — [`L658`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L658)
- `wrap_with_transforms_old(f, transforms, *args)` — [`L649`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L649)
- `wrapped(*args, **kwargs)` — [`L662`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L662)

## Module values
- `DeviceId` — [`L977`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L977)
- `NDIndexer` — [`L56`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L56)
- `Slice` — [`L55`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L55)
- `_unpad_values_to_avoid_dynamic_slice_oob_shift` — [`L260`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L260)
- `delay_effect` — [`L1324`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1324)
- `delay_p` — [`L1318`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1318)
- `get_global_p` — [`L843`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L843)
- `jaxpr_call_p` — [`L1340`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1340)
- `load_p` — [`L145`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L145)
- `multiple_of_p` — [`L120`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L120)
- `num_programs_p` — [`L95`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L95)
- `program_id_p` — [`L61`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L61)
- `reciprocal_p` — [`L548`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L548)
- `run_scoped_p` — [`L675`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L675)
- `sem_effect` — [`L987`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L987)
- `semaphore_read_p` — [`L937`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L937)
- `semaphore_signal_p` — [`L993`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L993)
- `semaphore_wait_p` — [`L1146`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L1146)
- `swap_p` — [`L307`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L307)
- `unsafe_map` — [`L58`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L58)
- `unsafe_zip` — [`L59`](../../../../../../../raw/code/jax/jax/_src/pallas/primitives.py#L59)

