---
title: 'Module: jax/_src/source_info_util.py'
type: catalog
provenance: extracted
module: jax/_src/source_info_util.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.source_info_util`/
symbols:
  SourceInfo: SourceInfo#
  current: current().
  summarize: summarize().
  NameStack: NameStack#
  SourceInfo.traceback: SourceInfo#traceback.
  UserContextManager.__exit__: UserContextManager#__exit__().
  SourceInfo.replace: SourceInfo#replace().
  SourceInfo.name_stack: SourceInfo#name_stack.
  raw_frame_to_frame: raw_frame_to_frame().
  _SourceInfoContext.context: _SourceInfoContext#context.
  current_name_stack: current_name_stack().
  _source_info_context: _source_info_context.
  NameStack.stack: NameStack#stack.
  user_context: user_context.
  ExtendNameStackContextManager.__enter__: ExtendNameStackContextManager#__enter__().
  TransformNameStackContextManager.__enter__: TransformNameStackContextManager#__enter__().
  _summarize_frame: _summarize_frame().
  user_frames: user_frames().
  NameStack.extend: NameStack#extend().
  SourceInfo.__init__: SourceInfo#__init__().
  UserContextManager.__enter__: UserContextManager#__enter__().
  SetNameStackContextManager.__enter__: SetNameStackContextManager#__enter__().
  user_frame: user_frame().
  UserContextManager: UserContextManager#
  _F: _F.
  register_exclusion: register_exclusion().
  register_inclusion: register_inclusion().
  NameStack.transform: NameStack#transform().
  new_name_stack: new_name_stack().
  is_user_filename: is_user_filename().
  ExtendNameStackContextManager: ExtendNameStackContextManager#
  UserContextManager.__init__: UserContextManager#__init__().
  ExtendNameStackContextManager.__exit__: ExtendNameStackContextManager#__exit__().
  SetNameStackContextManager.__exit__: SetNameStackContextManager#__exit__().
  TransformNameStackContextManager.__exit__: TransformNameStackContextManager#__exit__().
  Traceback: Traceback.
  new_source_info: new_source_info().
  SetNameStackContextManager: SetNameStackContextManager#
  TransformNameStackContextManager: TransformNameStackContextManager#
  extend_name_stack: extend_name_stack.
  reset_name_stack: reset_name_stack().
  Frame: Frame#
  Frame.file_name: Frame#file_name.
  Frame.start_line: Frame#start_line.
  NameStack.__getitem__: NameStack#__getitem__().
  NameStack.__add__: NameStack#__add__().
  NameStack.__radd__: NameStack#__radd__().
  NameStack.__str__: NameStack#__str__().
  _SourceInfoContext.__init__: _SourceInfoContext#__init__().
  ExtendNameStackContextManager.__call__: ExtendNameStackContextManager#__call__().
  SetNameStackContextManager.__init__: SetNameStackContextManager#__init__().
  SetNameStackContextManager.__call__: SetNameStackContextManager#__call__().
  TransformNameStackContextManager.__call__: TransformNameStackContextManager#__call__().
  _exclude_path_regex: _exclude_path_regex().
  _include_path_regex: _include_path_regex().
  Frame.start_column: Frame#start_column.
  Scope.name: Scope#name.
  Scope.wrap: Scope#wrap().
  has_user_context: has_user_context().
  ExtendNameStackContextManager.wrapper: ExtendNameStackContextManager#wrapper().
  SetNameStackContextManager.wrapper: SetNameStackContextManager#wrapper().
  set_name_stack: set_name_stack.
  TransformNameStackContextManager.wrapper: TransformNameStackContextManager#wrapper().
  Frame.function_name: Frame#function_name.
  _exclude_paths._exclude_paths: _exclude_paths._exclude_paths.
  Transform.wrap: Transform#wrap().
  NameStack.__len__: NameStack#__len__().
  ExtendNameStackContextManager.__init__: ExtendNameStackContextManager#__init__().
  TransformNameStackContextManager.__init__: TransformNameStackContextManager#__init__().
  transform_name_stack: transform_name_stack.
  Scope: Scope#
  Transform: Transform#
  Transform.name: Transform#name.
  JaxStackTraceBeforeTransformation: JaxStackTraceBeforeTransformation#
  _include_paths._include_paths: _include_paths._include_paths.
  Frame.end_line: Frame#end_line.
  Frame.end_column: Frame#end_column.
  _SourceInfoContext: _SourceInfoContext#
  _message: _message.
  SourceInfo.__slots__: SourceInfo#__slots__.
  UserContextManager.__slots__: UserContextManager#__slots__.
  ExtendNameStackContextManager.__slots__: ExtendNameStackContextManager#__slots__.
  SetNameStackContextManager.__slots__: SetNameStackContextManager#__slots__.
  TransformNameStackContextManager.__slots__: TransformNameStackContextManager#__slots__.
---
# Module: [`jax/_src/source_info_util.py`](../../../../../../raw/code/jax/jax/_src/source_info_util.py)

## Classes
### `ExtendNameStackContextManager`
- def: [`jax/_src/source_info_util.py:272`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L272)
- signature: `class ExtendNameStackContextManager:`
- members:
  - `wrapper(*args, **kwargs)` — [`L289`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L289)
- protocol/private: `__call__`[`L287`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L287), `__enter__`[`L278`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L278), `__exit__`[`L284`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L284), `__init__`[`L275`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L275), `__slots__`[`L273`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L273)
- uses (calls/refs, reference-scoped): [`replace`](source_info_util.md#SourceInfo.replace), [`context`](source_info_util.md#_SourceInfoContext.context), [`name_stack`](source_info_util.md#SourceInfo.name_stack), [`_source_info_context`](source_info_util.md#_source_info_context), [`extend`](source_info_util.md#NameStack.extend), [`_F`](source_info_util.md#_F)
- used by: [`named_scope`](api.md#named_scope), [`extend_name_stack`](source_info_util.md#extend_name_stack)

### `Frame`  ·  implements/extends NamedTuple
- def: [`jax/_src/source_info_util.py:37`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L37)
- signature: `class Frame(NamedTuple):`
- members:
  - `end_column` — [`L43`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L43)
  - `end_line` — [`L42`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L42)
  - `file_name` — [`L38`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L38)
  - `function_name` — [`L39`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L39)
  - `start_column` — [`L41`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L41)
  - `start_line` — [`L40`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L40)
- used by: [`lower_jaxpr_to_module`](pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_module), [`raw_frame_to_frame`](source_info_util.md#raw_frame_to_frame), [`_make_logging_record`](debugging.md#_make_logging_record), [`_summarize_frame`](source_info_util.md#_summarize_frame), [`user_frames`](source_info_util.md#user_frames), [`user_frame`](source_info_util.md#user_frame)

### `JaxStackTraceBeforeTransformation`  ·  implements/extends Exception
- def: [`jax/_src/source_info_util.py:216`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L216)
- signature: `class JaxStackTraceBeforeTransformation(Exception):`
- used by: [`__exit__`](source_info_util.md#UserContextManager.__exit__), [`has_user_context`](source_info_util.md#has_user_context)

### `NameStack`
- def: [`jax/_src/source_info_util.py:101`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L101)
- signature: `class NameStack:`
- members:
  - `extend(self, name: str)` — [`L104`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L104)
  - `transform(self, transform_name: str)` — [`L107`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L107)
  - `stack` — [`L102`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L102)
- protocol/private: `__add__`[`L116`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L116), `__getitem__`[`L110`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L110), `__len__`[`L113`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L113), `__radd__`[`L119`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L119), `__str__`[`L122`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L122)
- uses (calls/refs, reference-scoped): [`wrap`](source_info_util.md#Scope.wrap), [`Scope`](source_info_util.md#Scope), [`Transform`](source_info_util.md#Transform)
- used by: [`jaxpr_subcomp`](pallas/mosaic/lowering.md#jaxpr_subcomp), [`lower_jaxpr_to_mosaic_gpu`](pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_mosaic_gpu), [`body`](pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_module.body), [`body_func`](pallas/mosaic/lowering.md#lower_jaxpr_to_transform_func.body_func), [`_pjit_lowering`](pjit.md#_pjit_lowering), [`from_mosaic_grid_mapping`](pallas/mosaic/lowering.md#PipelinedLoweringContext.from_mosaic_grid_mapping), [`from_mesh`](pallas/mosaic/lowering.md#UnpipelinedLoweringContext.from_mesh), [`replace`](source_info_util.md#SourceInfo.replace), [`name_stack`](source_info_util.md#SourceInfo.name_stack), [`current_name_stack`](source_info_util.md#current_name_stack), [`__enter__`](source_info_util.md#ExtendNameStackContextManager.__enter__), [`__enter__`](source_info_util.md#TransformNameStackContextManager.__enter__), [`_remat_lowering`](ad_checkpoint.md#_remat_lowering), [`__init__`](source_info_util.md#SourceInfo.__init__), [`new_name_stack`](source_info_util.md#new_name_stack), [`__init__`](source_info_util.md#UserContextManager.__init__), [`new_source_info`](source_info_util.md#new_source_info), [`reset_name_stack`](source_info_util.md#reset_name_stack), [`name_stack`](pallas/mosaic/lowering.md#LoweringContext.name_stack), [`__init__`](source_info_util.md#SetNameStackContextManager.__init__), [`name_stack`](pallas/mosaic_gpu/lowering.md#ModuleContext.name_stack)

### `Scope`  ·  implements/extends NamedTuple
- def: [`jax/_src/source_info_util.py:84`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L84)
- signature: `class Scope(NamedTuple):`
- members:
  - `wrap(self, stack: list[str])` — [`L87`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L87)
  - `name` — [`L85`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L85)
- used by: [`jaxpr_subcomp`](pallas/mosaic/lowering.md#jaxpr_subcomp), [`lower_jaxpr_to_mosaic_gpu`](pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_mosaic_gpu), [`stack`](source_info_util.md#NameStack.stack), [`extend`](source_info_util.md#NameStack.extend), [`__str__`](source_info_util.md#NameStack.__str__)

### `SetNameStackContextManager`
- def: [`jax/_src/source_info_util.py:297`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L297)
- signature: `class SetNameStackContextManager:`
- members:
  - `wrapper(*args, **kwargs)` — [`L312`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L312)
- protocol/private: `__call__`[`L310`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L310), `__enter__`[`L303`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L303), `__exit__`[`L307`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L307), `__init__`[`L300`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L300), `__slots__`[`L298`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L298)
- uses (calls/refs, reference-scoped): [`NameStack`](source_info_util.md#NameStack), [`replace`](source_info_util.md#SourceInfo.replace), [`context`](source_info_util.md#_SourceInfoContext.context), [`_source_info_context`](source_info_util.md#_source_info_context), [`_F`](source_info_util.md#_F)
- used by: [`set_name_stack`](source_info_util.md#set_name_stack)

### `SourceInfo`
- def: [`jax/_src/source_info_util.py:136`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L136)
- signature: `class SourceInfo:`
- members:
  - `replace(self, *, traceback: Traceback | None = None, name_stack: NameStack | None = None)` — [`L147`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L147)
  - `name_stack` — [`L138`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L138)
  - `traceback` — [`L137`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L137)
- protocol/private: `__init__`[`L143`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L143), `__slots__`[`L141`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L141)
- uses (calls/refs, reference-scoped): [`NameStack`](source_info_util.md#NameStack), [`Traceback`](source_info_util.md#Traceback)
- used by: [`jaxpr_subcomp`](pallas/mosaic/lowering.md#jaxpr_subcomp), [`lower_jaxpr_to_mosaic_gpu`](pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_mosaic_gpu), [`eval_jaxpr`](core.md#eval_jaxpr), [`_eval_jaxpr_discharge_state`](state/discharge.md#_eval_jaxpr_discharge_state), [`jaxpr_type_signature`](../experimental/key_reuse/_core.md#jaxpr_type_signature), [`lower_jaxpr_to_triton_ir`](pallas/triton/lowering.md#lower_jaxpr_to_triton_ir), [`physicalize_interp`](pallas/fuser/fusible_dtype.md#physicalize_interp), [`_custom_evaluate_jaxpr`](pallas/fuser/custom_evaluate.md#_custom_evaluate_jaxpr), [`eval_jaxpr_recursive`](pallas/hlo_interpreter.md#eval_jaxpr_recursive), [`_pp_eqn`](core.md#_pp_eqn), [`_allocate_buffer`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#_allocate_buffer), [`_allocate_buffer_for_all_threads`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#_allocate_buffer_for_all_threads), [`_allocate_barriers`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#_allocate_barriers), [`_allocate_cluster_barriers`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#_allocate_cluster_barriers), [`source_info`](core.md#JaxprEqn.source_info), [`_allocate_buffer`](pallas/mosaic/interpret/interpret_pallas_call.md#_allocate_buffer), [`_scan_partial_eval`](lax/control_flow/loops.md#_scan_partial_eval), [`__init__`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#AsyncCopyTask.__init__), [`tcgen05_mma`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#tcgen05_mma), [`wgmma`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#wgmma), [`replace`](core.md#JaxprEqn.replace), [`_cond_partial_eval`](lax/control_flow/conditionals.md#_cond_partial_eval), [`_deallocate_barrier`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#_deallocate_barrier), [`_barrier_arrive`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#_barrier_arrive), [`async_load_tmem`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#async_load_tmem), [`pp_eqn`](core.md#pp_eqn), [`_barrier_wait`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#_barrier_wait), [`copy_smem_to_gmem`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#copy_smem_to_gmem), [`current`](source_info_util.md#current), [`source_info`](pallas/mosaic/interpret/utils.md#LoggingInfo.source_info), [`summarize`](source_info_util.md#summarize), [`_deallocate_buffer`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#_deallocate_buffer), [`copy_gmem_to_smem`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#copy_gmem_to_smem), [`wgmma_accumulator_deref`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#wgmma_accumulator_deref), [`call_allocate_barriers`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#call_allocate_barriers), [`call_allocate_cluster_barriers`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#call_allocate_cluster_barriers), [`__init__`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#AsyncCopyGmemToSmemTask.__init__), [`_interpret_barrier_primitive`](pallas/mosaic_gpu/interpret/jaxpr_interpret.md#JaxprInterpreter._interpret_barrier_primitive), [`call_allocate_buffer_for_all_threads`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#call_allocate_buffer_for_all_threads), [`call_allocate_buffer`](pallas/mosaic_gpu/interpret/gpu_callbacks.md#call_allocate_buffer)  (+19 more)

### `Transform`  ·  implements/extends NamedTuple
- def: [`jax/_src/source_info_util.py:90`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L90)
- signature: `class Transform(NamedTuple):`
- members:
  - `wrap(self, stack: list[str])` — [`L93`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L93)
  - `name` — [`L91`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L91)
- used by: [`stack`](source_info_util.md#NameStack.stack), [`transform`](source_info_util.md#NameStack.transform)

### `TransformNameStackContextManager`
- def: [`jax/_src/source_info_util.py:331`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L331)
- signature: `class TransformNameStackContextManager:`
- members:
  - `wrapper(*args, **kwargs)` — [`L348`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L348)
- protocol/private: `__call__`[`L346`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L346), `__enter__`[`L337`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L337), `__exit__`[`L343`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L343), `__init__`[`L334`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L334), `__slots__`[`L332`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L332)
- uses (calls/refs, reference-scoped): [`replace`](source_info_util.md#SourceInfo.replace), [`context`](source_info_util.md#_SourceInfoContext.context), [`name_stack`](source_info_util.md#SourceInfo.name_stack), [`_source_info_context`](source_info_util.md#_source_info_context), [`_F`](source_info_util.md#_F), [`transform`](source_info_util.md#NameStack.transform)
- used by: [`transform_name_stack`](source_info_util.md#transform_name_stack)

### `UserContextManager`
- def: [`jax/_src/source_info_util.py:230`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L230)
- signature: `class UserContextManager:`
- protocol/private: `__enter__`[`L238`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L238), `__exit__`[`L243`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L243), `__init__`[`L233`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L233), `__slots__`[`L231`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L231)
- uses (calls/refs, reference-scoped): [`NameStack`](source_info_util.md#NameStack), [`replace`](source_info_util.md#SourceInfo.replace), [`context`](source_info_util.md#_SourceInfoContext.context), [`_source_info_context`](source_info_util.md#_source_info_context), [`filter_traceback`](traceback_util.md#filter_traceback), [`Traceback`](source_info_util.md#Traceback), [`has_user_context`](source_info_util.md#has_user_context), [`JaxStackTraceBeforeTransformation`](source_info_util.md#JaxStackTraceBeforeTransformation), [`format_exception_only`](traceback_util.md#format_exception_only), [`_message`](source_info_util.md#_message)
- used by: [`user_context`](source_info_util.md#user_context)

### `_SourceInfoContext`  ·  implements/extends local
- def: [`jax/_src/source_info_util.py:202`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L202)
- signature: `class _SourceInfoContext(threading.local):`
- members:
  - `context` — [`L203`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L203)
- protocol/private: `__init__`[`L205`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L205)
- uses (calls/refs, reference-scoped): [`SourceInfo`](source_info_util.md#SourceInfo), [`new_source_info`](source_info_util.md#new_source_info)
- used by: [`current`](source_info_util.md#current), [`__exit__`](source_info_util.md#UserContextManager.__exit__), [`current_name_stack`](source_info_util.md#current_name_stack), [`_source_info_context`](source_info_util.md#_source_info_context), [`__enter__`](source_info_util.md#ExtendNameStackContextManager.__enter__), [`__enter__`](source_info_util.md#TransformNameStackContextManager.__enter__), [`__enter__`](source_info_util.md#SetNameStackContextManager.__enter__), [`__enter__`](source_info_util.md#UserContextManager.__enter__), [`__exit__`](source_info_util.md#ExtendNameStackContextManager.__exit__), [`__exit__`](source_info_util.md#SetNameStackContextManager.__exit__), [`__exit__`](source_info_util.md#TransformNameStackContextManager.__exit__)

## Functions
- `_exclude_path_regex()` — [`L57`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L57)
- `_include_path_regex()` — [`L73`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L73)
- `_summarize_frame(frame: Frame)` — [`L189`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L189)
- `current()` — [`L210`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L210)
- `current_name_stack()` — [`L265`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L265)
- `has_user_context(e)` — [`L223`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L223)
- `is_user_filename(filename: str)` — [`L158`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L158) — Heuristic that guesses the identity of the user's code in a stack trace.
- `new_name_stack(name: str = '')` — [`L129`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L129)
- `new_source_info()` — [`L154`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L154)
- `raw_frame_to_frame(code: types.CodeType, lasti: int)` — [`L164`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L164)
- `register_exclusion(path: str)` — [`L63`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L63)
- `register_inclusion(path: str)` — [`L78`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L78)
- `reset_name_stack()` — [`L326`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L326)
- `summarize(source_info: SourceInfo, num_frames=1)` — [`L196`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L196)
- `user_frame(traceback: Traceback | None)` — [`L186`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L186)
- `user_frames(traceback: Traceback | None)` — [`L173`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L173) — Iterator over the user's frames, filtering jax-internal frames.

## Module values
- `Traceback` — [`L35`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L35)
- `_F` — [`L269`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L269)
- `_exclude_paths` — [`L46`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L46)
- `_include_paths` — [`L70`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L70)
- `_message` — [`L218`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L218)
- `_source_info_context` — [`L208`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L208)
- `extend_name_stack` — [`L294`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L294)
- `set_name_stack` — [`L318`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L318)
- `transform_name_stack` — [`L353`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L353)
- `user_context` — [`L262`](../../../../../../raw/code/jax/jax/_src/source_info_util.py#L262)

