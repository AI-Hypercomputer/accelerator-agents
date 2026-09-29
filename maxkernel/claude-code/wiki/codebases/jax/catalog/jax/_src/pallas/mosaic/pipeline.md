---
title: 'Module: jax/_src/pallas/mosaic/pipeline.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/pipeline.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.pipeline`/
symbols:
  emit_pipeline.wrapped: emit_pipeline().wrapped().
  BufferedRef.create: BufferedRef#create().
  _emit_pipeline.pipeline: _emit_pipeline().pipeline().
  _emit_pipeline.pipeline.loop_body: _emit_pipeline().pipeline().loop_body().
  _emit_pipeline.pipeline._loop_body: _emit_pipeline().pipeline()._loop_body().
  _make_block_slice: _make_block_slice().
  BufferedRef.copy_out: BufferedRef#copy_out().
  BufferedRef.wait_out: BufferedRef#wait_out().
  _emit_pipeline_effectful_abstract_eval: _emit_pipeline_effectful_abstract_eval().
  BufferedRef.copy_in: BufferedRef#copy_in().
  BufferedRef.wait_in: BufferedRef#wait_in().
  _spec_has_trivial_windowing: _spec_has_trivial_windowing().
  BufferedRef: BufferedRef#
  _partition_grid: _partition_grid().
  sync_copy: sync_copy().
  _emit_pipeline_lowering_rule: _emit_pipeline_lowering_rule().
  _emit_pipeline_lowering_rule.wrapped_pipeline_fun: _emit_pipeline_lowering_rule().wrapped_pipeline_fun().
  BufferedRef.current_ref: BufferedRef#current_ref().
  BufferedRef._window_ref_at: BufferedRef#_window_ref_at().
  Scheduler.initialize_step: Scheduler#initialize_step().
  BufferedRef.initialize_slots: BufferedRef#initialize_slots().
  emit_pipeline: emit_pipeline().
  _ref_to_value_aval: _ref_to_value_aval().
  _pipeline_body_effectful_abstract_eval: _pipeline_body_effectful_abstract_eval().
  _pipeline_body_lowering_rule: _pipeline_body_lowering_rule().
  fetch_with_lookahead: fetch_with_lookahead().
  map_brefs: map_brefs.
  Scheduler.fetch_indices: Scheduler#fetch_indices.
  BufferedRef.compute_slice: BufferedRef#compute_slice().
  Scheduler.copy_in: Scheduler#copy_in().
  emit_pipeline_p: emit_pipeline_p.
  pipeline_body_p: pipeline_body_p.
  _create_blocked_slice: _create_blocked_slice().
  _emit_pipeline: _emit_pipeline().
  BufferedRef.is_buffered: BufferedRef#is_buffered().
  Scheduler.grid_env: Scheduler#grid_env().
  Scheduler._named_scope: Scheduler#_named_scope().
  BufferedRefBase.get_dma_slice: BufferedRefBase#get_dma_slice().
  BufferedRef.buffer_count: BufferedRef#buffer_count().
  BufferedRef.advance_copy_in_slot: BufferedRef#advance_copy_in_slot().
  BufferedRef.advance_wait_in_slot: BufferedRef#advance_wait_in_slot().
  BufferedRef.advance_copy_out_slot: BufferedRef#advance_copy_out_slot().
  BufferedRef.advance_wait_out_slot: BufferedRef#advance_wait_out_slot().
  _create_bounded_slice: _create_bounded_slice().
  BufferedRefBase.block_shape: BufferedRefBase#block_shape().
  BufferedRef.window_ref: BufferedRef#window_ref.
  BufferType: BufferType#
  _get_block_shape._get_dim_size: _get_block_shape()._get_dim_size().
  _next_index: _next_index().
  _make_pipeline_allocations.make_input_bref: _make_pipeline_allocations().make_input_bref().
  _make_pipeline_allocations.make_output_bref: _make_pipeline_allocations().make_output_bref().
  BufferedRefBase._to_window_slice: BufferedRefBase#_to_window_slice().
  BufferedRef.current_copy_out_slot: BufferedRef#current_copy_out_slot().
  BufferedRef.current_wait_in_slot: BufferedRef#current_wait_in_slot().
  _make_pipeline_allocations: _make_pipeline_allocations().
  when: when.
  ds: ds.
  BufferedRef._advance_slot: BufferedRef#_advance_slot().
  BufferedRefBase.is_output: BufferedRefBase#is_output().
  BufferedRef.current_copy_in_slot: BufferedRef#current_copy_in_slot().
  BufferedRef.current_wait_out_slot: BufferedRef#current_wait_out_slot().
  Scheduler.wait_in: Scheduler#wait_in().
  BufferType.is_input: BufferType#is_input().
  BufferType.is_output: BufferType#is_output().
  BufferedRefBase: BufferedRefBase#
  Scheduler.has_changed: Scheduler#has_changed().
  BufferedRef.bind_existing_ref: BufferedRef#bind_existing_ref().
  Scheduler.indices: Scheduler#indices.
  Scheduler._wait: Scheduler#_wait().
  is_transformed_ref: is_transformed_ref.
  _prev_index: _prev_index().
  Scheduler.will_change_current: Scheduler#will_change_current().
  HBM: HBM.
  BufferedRefBase.buffer_type: BufferedRefBase#buffer_type().
  BufferedRefBase.is_input: BufferedRefBase#is_input().
  Tiling: Tiling.
  _get_block_shape: _get_block_shape().
  BufferedRef.input: BufferedRef#input().
  BufferedRef.output: BufferedRef#output().
  Scheduler.will_change_fetch: Scheduler#will_change_fetch().
  Scheduler._start: Scheduler#_start().
  Scheduler.wait_out: Scheduler#wait_out().
  Scheduler.copy_out: Scheduler#copy_out().
  _normalize_specs: _normalize_specs().
  _emit_pipeline_lowering_rule.wrapped_pipeline_fun.new_body: _emit_pipeline_lowering_rule().wrapped_pipeline_fun().new_body().
  ANY: ANY.
  Scheduler.first_step: Scheduler#first_step.
  Squeezed: Squeezed.
  REF: REF.
  BufferedRefBase.is_input_output: BufferedRefBase#is_input_output().
  BufferedRefBase.is_manual: BufferedRefBase#is_manual().
  BufferedRefBase.with_spec: BufferedRefBase#with_spec().
  BufferedRef.__post_init__: BufferedRef#__post_init__().
  BufferedRef.input_output: BufferedRef#input_output().
  BufferedRef.advance_next_fetch: BufferedRef#advance_next_fetch().
  Scheduler._send: Scheduler#_send().
  Scheduler._copy_out: Scheduler#_copy_out().
  Scheduler._end: Scheduler#_end().
  PipelineStep.tree_flatten: PipelineStep#tree_flatten().
  VMEM: VMEM.
  ArrayRef: ArrayRef.
  BufferedRefBase.spec: BufferedRefBase#spec().
  BufferedRefBase.compute_index: BufferedRefBase#compute_index().
  map_inputs: map_inputs().
  map_outputs: map_outputs().
  Scheduler.prev_indices: Scheduler#prev_indices.
  Indirect: Indirect.
  Element: Element.
  BoundedSlice: BoundedSlice.
  BufferedRef.next_fetch: BufferedRef#next_fetch.
  BufferedRef.sem_recvs: BufferedRef#sem_recvs.
  BufferedRef.sem_sends: BufferedRef#sem_sends.
  BufferedRef.buffer_type: BufferedRef#buffer_type().
  _filter_indices: _filter_indices().
  _emit_pipeline.pipeline.make_scheduler: _emit_pipeline().pipeline().make_scheduler().
  SMEM: SMEM.
  BufferedRef.with_slot_index: BufferedRef#with_slot_index().
  fetch_with_lookahead._loop_body: fetch_with_lookahead()._loop_body().
  Scheduler.out_of_fetch: Scheduler#out_of_fetch().
  Scheduler.advance_slots: Scheduler#advance_slots().
  emit_pipeline_with_allocations: emit_pipeline_with_allocations().
  Blocked: Blocked.
  GridDimensionSemantics: GridDimensionSemantics.
  BufferedRef.copy_in_slot: BufferedRef#copy_in_slot.
  BufferedRef.wait_in_slot: BufferedRef#wait_in_slot.
  BufferedRef.copy_out_slot: BufferedRef#copy_out_slot.
  BufferedRef.wait_out_slot: BufferedRef#wait_out_slot.
  Scheduler.last_step: Scheduler#last_step.
  PipelineStep: PipelineStep#
  _tuples_differ: _tuples_differ().
  BufferedRefBase.advance_copy_in_slot: BufferedRefBase#advance_copy_in_slot().
  BufferedRefBase.advance_wait_in_slot: BufferedRefBase#advance_wait_in_slot().
  BufferedRefBase.advance_copy_out_slot: BufferedRefBase#advance_copy_out_slot().
  BufferedRefBase.advance_wait_out_slot: BufferedRefBase#advance_wait_out_slot().
  BufferedRefBase.has_indirect: BufferedRefBase#has_indirect().
  BufferedRef.with_spec: BufferedRef#with_spec().
  BufferedRef.next_fetch_indices: BufferedRef#next_fetch_indices().
  BufferedRef.unbind_refs: BufferedRef#unbind_refs().
  align_to: align_to.
  multiple_of: multiple_of.
  BlockDim: BlockDim.
  Slice: Slice.
  SemaphoreType: SemaphoreType.
  SemaphoreTuple: SemaphoreTuple.
  Scheduler.alias_local_refs: Scheduler#alias_local_refs().
  PipelineStep.index: PipelineStep#index.
  PipelineStep.local_index: PipelineStep#local_index.
  BufferType.INPUT_OUTPUT: BufferType#INPUT_OUTPUT.
  BufferedRef.prefetched_count: BufferedRef#prefetched_count.
  Scheduler.step: Scheduler#step.
  Scheduler.grid: Scheduler#grid.
  cdiv: cdiv.
  program_id: program_id.
  num_programs: num_programs.
  PARALLEL: PARALLEL.
  ARBITRARY: ARBITRARY.
  _tuple_lt: _tuple_lt.
  BufferedRef._spec: BufferedRef#_spec.
  BufferedRef._buffer_type: BufferedRef#_buffer_type.
  BufferedRef.use_lookahead: BufferedRef#use_lookahead().
  BufferedRef.with_next_fetch: BufferedRef#with_next_fetch().
  BufferedRef.cumulative_copy_in: BufferedRef#cumulative_copy_in().
  BufferedRef.cumulative_copy_out: BufferedRef#cumulative_copy_out().
  BufferedRef.cumulative_wait_in: BufferedRef#cumulative_wait_in().
  BufferedRef.cumulative_wait_out: BufferedRef#cumulative_wait_out().
  fetch_with_lookahead.as_uint32: fetch_with_lookahead().as_uint32().
  Scheduler.finalize: Scheduler#finalize().
  _emit_pipeline.pipeline._._sync_copy_in: _emit_pipeline().pipeline()._()._sync_copy_in().
  _emit_pipeline.pipeline._._sync_copy_out: _emit_pipeline().pipeline()._()._sync_copy_out().
  BufferType.INPUT: BufferType#INPUT.
  BufferType.OUTPUT: BufferType#OUTPUT.
  BufferedRef._grid_rank: BufferedRef#_grid_rank.
  BufferedRef.is_trivial_windowing: BufferedRef#is_trivial_windowing.
  GridIndices: GridIndices.
  CondVal: CondVal.
  PipelineBlockSpecs: PipelineBlockSpecs.
  PipelineRefs: PipelineRefs.
  BufferedRefBase.is_buffered: BufferedRefBase#is_buffered().
  BufferedRefBase.initialize_slots: BufferedRefBase#initialize_slots().
  BufferedRefBase.bind_existing_ref: BufferedRefBase#bind_existing_ref().
  BufferedRefBase.unbind_refs: BufferedRefBase#unbind_refs().
  BufferedRef.tiling: BufferedRef#tiling.
  BufferedRef.spec: BufferedRef#spec().
  BufferedRef.with_window_ref: BufferedRef#with_window_ref().
  fetch_with_lookahead._loop_body._start: fetch_with_lookahead()._loop_body()._start().
  Scheduler.__init__: Scheduler#__init__().
  BufferedRef._buffer_count: BufferedRef#_buffer_count.
  BufferedRef.has_allocated_buffer: BufferedRef#has_allocated_buffer.
  Scheduler.grid_offsets: Scheduler#grid_offsets.
  Scheduler.num_stages: Scheduler#num_stages.
  Scheduler.num_steps: Scheduler#num_steps.
  Scheduler.next_indices: Scheduler#next_indices.
  Scheduler.unalias_local_refs: Scheduler#unalias_local_refs().
  contextmanager: contextmanager.
  _tuple_all_binop: _tuple_all_binop().
  BufferType.MANUAL: BufferType#MANUAL.
  fetch_with_lookahead._loop_cond: fetch_with_lookahead()._loop_cond().
  map_inputs.fmap: map_inputs().fmap().
  map_outputs.fmap: map_outputs().fmap().
  Scheduler: Scheduler#
  Scheduler.trace_scopes: Scheduler#trace_scopes.
  Scheduler._explicit_indices: Scheduler#_explicit_indices.
  Scheduler.add_offset: Scheduler#add_offset.
  BufferedRefBase.is_trivial_windowing: BufferedRefBase#is_trivial_windowing().
  BufferedRefBase.prefetched_count: BufferedRefBase#prefetched_count().
  BufferedRefBase.has_allocated_buffer: BufferedRefBase#has_allocated_buffer().
  PipelineStep.tree_unflatten: PipelineStep#tree_unflatten().
---
# Module: [`jax/_src/pallas/mosaic/pipeline.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py)

## Classes
### `BufferType`  ·  implements/extends Enum
- def: [`jax/_src/pallas/mosaic/pipeline.py:226`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L226)
- doc: Buffer type for the arguments to an emitted pipeline.
- signature: `class BufferType(enum.Enum):`
- members:
  - `is_input(self)` — [`L234`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L234)
  - `is_output(self)` — [`L241`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L241)
  - `INPUT` — [`L228`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L228)
  - `INPUT_OUTPUT` — [`L230`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L230)
  - `MANUAL` — [`L231`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L231)
  - `OUTPUT` — [`L229`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L229)
- used by: [`create`](pipeline.md#BufferedRef.create), [`initialize_slots`](pipeline.md#BufferedRef.initialize_slots), [`is_output`](pipeline.md#BufferedRefBase.is_output), [`buffer_type`](pipeline.md#BufferedRefBase.buffer_type), [`is_input`](pipeline.md#BufferedRefBase.is_input), [`input`](pipeline.md#BufferedRef.input), [`output`](pipeline.md#BufferedRef.output), [`input_output`](pipeline.md#BufferedRef.input_output), [`is_input_output`](pipeline.md#BufferedRefBase.is_input_output), [`is_manual`](pipeline.md#BufferedRefBase.is_manual), [`_buffer_type`](pipeline.md#BufferedRef._buffer_type)

### `BufferedRef`  ·  implements/extends BufferedRefBase
- def: [`jax/_src/pallas/mosaic/pipeline.py:454`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L454)
- doc: A helper class to automate VMEM double buffering in pallas pipelines.
- signature: `class BufferedRef(BufferedRefBase):`
- members:
  - `advance_copy_in_slot(self, predicate: bool | jax.Array = True)` — [`L854`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L854) — Switch to the next copy slot.
  - `advance_copy_out_slot(self, predicate: bool | jax.Array = True)` — [`L870`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L870) — Switch to the next copy slot.
  - `advance_next_fetch(self, grid)` — [`L969`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L969)
  - `advance_wait_in_slot(self, predicate: bool | jax.Array = True)` — [`L862`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L862) — Switch to the next wait slot.
  - `advance_wait_out_slot(self, predicate: bool | jax.Array = True)` — [`L878`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L878) — Switch to the next wait slot.
  - `bind_existing_ref(self, window_ref, indices)` — [`L787`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L787) — For handling VMEM references, the pipeline aliases the existing ref.
  - `buffer_count(self)` — [`L530`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L530) — Returns the number of buffers used for multiple buffering.
  - `buffer_type(self)` — [`L516`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L516)
  - `compute_slice(self, grid_indices)` — [`L802`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L802) — Compute DMA slice from grid indices.
  - `copy_in(self, src_ref, grid_indices)` — [`L903`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L903) — Starts copy of HBM dma slice into the current slot.
  - `copy_out(self, dst_ref, grid_indices)` — [`L917`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L917) — Starts copy of HBM dma slice from the current slot.
  - `create(cls, spec: pallas_core.BlockSpec, dtype_or_type, buffer_type, buffer_count, grid_rank=None, use_lookahead=False, source_memory_space: tpu_core.MemorySpace | Literal[ANY] = ANY, tiling: Tiling | None = None, is_trivial_windowing: bool = False, prefetched_count: int = 0)` — [`L537`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L537) — Create a BufferedRef.
  - `cumulative_copy_in(self)` — [`L736`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L736) — The cumulative number of copy_ins issued on this buffer.
  - `cumulative_copy_out(self)` — [`L747`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L747) — The cumulative number of copy_outs issued on this buffer.
  - `cumulative_wait_in(self)` — [`L758`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L758) — The cumulative number of wait_ins issued on this buffer.
  - `cumulative_wait_out(self)` — [`L769`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L769) — The cumulative number of wait_outs issued on this buffer.
  - `current_copy_in_slot(self)` — [`L742`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L742) — Index in multiple buffer corresponding to the current slot.
  - `current_copy_out_slot(self)` — [`L753`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L753) — Index in multiple buffer corresponding to the current copy slot.
  - `current_ref(self)` — [`L712`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L712) — Returns the current working slice of the double-buffer.
  - `current_wait_in_slot(self)` — [`L764`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L764) — Index in multiple buffer corresponding to the current wait slot.
  - `current_wait_out_slot(self)` — [`L775`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L775) — Index in multiple buffer corresponding to the current wait slot.
  - `initialize_slots(self)` — [`L829`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L829)
  - `input(cls, spec, dtype_or_type, buffer_count=2, **kwargs)` — [`L662`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L662)
  - `input_output(cls, spec, dtype_or_type, buffer_count=2, **kwargs)` — [`L674`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L674)
  - `is_buffered(self)` — [`L520`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L520) — Whether this buffer is multiple-buffered.
  - `next_fetch_indices(self)` — [`L780`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L780) — Returns the next grid indices to fetch from if using lookahead.
  - `output(cls, spec, dtype_or_type, buffer_count=2, **kwargs)` — [`L668`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L668)
  - `spec(self)` — [`L512`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L512)
  - `unbind_refs(self)` — [`L797`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L797)
  - `use_lookahead(self)` — [`L525`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L525) — Whether this buffer allows lookahead for fetching blocks.
  - `wait_in(self, src_ref, grid_indices)` — [`L937`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L937) — Waits for input copy to finish.
  - `wait_out(self, dst_ref, grid_indices)` — [`L953`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L953) — Waits for output copy to finish.
  - `with_next_fetch(self, next_fetch: Sequence[jax.Array | int] | None = None)` — [`L683`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L683)
  - `with_slot_index(self, copy_in_slot: int | jax.Array | None = None, copy_out_slot: int | jax.Array | None = None, wait_in_slot: int | jax.Array | None = None, wait_out_slot: int | jax.Array | None = None)` — [`L692`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L692) — Returns a new BufferedRef with the given slot index.
  - `with_spec(self, spec: pallas_core.BlockSpec)` — [`L679`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L679) — Returns a new BufferedRef with the given block spec.
  - `with_window_ref(self, window_ref: ArrayRef | None)` — [`L689`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L689)
  - `copy_in_slot` — [`L482`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L482)
  - `copy_out_slot` — [`L484`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L484)
  - `has_allocated_buffer` — [`L493`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L493)
  - `is_trivial_windowing` — [`L490`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L490)
  - `next_fetch` — [`L486`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L486)
  - `prefetched_count` — [`L496`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L496)
  - `sem_recvs` — [`L487`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L487)
  - `sem_sends` — [`L488`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L488)
  - `tiling` — [`L489`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L489)
  - `wait_in_slot` — [`L483`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L483)
  - `wait_out_slot` — [`L485`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L485)
  - `window_ref` — [`L481`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L481)
- protocol/private: `__post_init__`[`L500`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L500), `_advance_slot`[`L849`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L849), `_buffer_count`[`L479`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L479), `_buffer_type`[`L478`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L478), `_grid_rank`[`L480`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L480), `_spec`[`L477`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L477), `_window_ref_at`[`L886`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L886)
- uses (calls/refs, reference-scoped): [`Array`](../../basearray.md#Array), [`AbstractRef`](../../state/types.md#AbstractRef), [`ShapedArray`](../../core.md#ShapedArray), [`AbstractValue`](../../core.md#AbstractValue), [`BlockSpec`](../core.md#BlockSpec), [`int32`](../../../numpy/__init__.pyi.md#int32.int32), [`select`](../../lax/lax.md#select), [`CoreType`](core.md#CoreType), [`MemorySpace`](core.md#MemorySpace), [`rem`](../../lax/lax.md#rem), [`memory_space_to_tpu_memory_space`](core.md#memory_space_to_tpu_memory_space), [`at`](../../basearray.md#Array.at), [`uint32`](../../../numpy/__init__.pyi.md#uint32.uint32), [`register_dataclass`](../../tree_util.md#register_dataclass), [`memory_space`](../core.md#BlockSpec.memory_space), [`_ref_to_value_aval`](pipeline.md#_ref_to_value_aval), [`start`](primitives.md#AsyncCopyDescriptor.start), [`make_async_copy`](primitives.md#make_async_copy), [`TC`](core.md#CoreType.TC), [`sync_copy`](helpers.md#sync_copy), [`wait`](primitives.md#AsyncCopyDescriptor.wait), [`get_dma_slice`](pipeline.md#BufferedRefBase.get_dma_slice), [`block_shape`](pipeline.md#BufferedRefBase.block_shape), [`BufferType`](pipeline.md#BufferType), [`DMA`](core.md#SemaphoreType.DMA), [`_next_index`](pipeline.md#_next_index), [`_to_window_slice`](pipeline.md#BufferedRefBase._to_window_slice), [`ds`](pipeline.md#ds), [`is_output`](pipeline.md#BufferedRefBase.is_output), [`update`](../../core.md#AbstractValue.update), [`BufferedRefBase`](pipeline.md#BufferedRefBase), [`is_input`](pipeline.md#BufferType.is_input), [`is_output`](pipeline.md#BufferType.is_output), [`HBM`](pipeline.md#HBM), [`is_input`](pipeline.md#BufferedRefBase.is_input), [`Tiling`](pipeline.md#Tiling), [`_get_block_shape`](pipeline.md#_get_block_shape), [`ANY`](pipeline.md#ANY), [`Squeezed`](pipeline.md#Squeezed), [`ArrayRef`](pipeline.md#ArrayRef)  (+16 more)
- used by: [`sync_copy`](pipeline.md#sync_copy), [`copy_in`](pipeline.md#Scheduler.copy_in), [`make_input_bref`](pipeline.md#_make_pipeline_allocations.make_input_bref), [`make_output_bref`](pipeline.md#_make_pipeline_allocations.make_output_bref), [`wait_in`](pipeline.md#Scheduler.wait_in), [`BufferedRefBase`](pipeline.md#BufferedRefBase), [`buffer_type`](pipeline.md#BufferedRefBase.buffer_type), [`copy_out`](pipeline.md#Scheduler.copy_out), [`wait_out`](pipeline.md#Scheduler.wait_out), [`with_spec`](pipeline.md#BufferedRefBase.with_spec), [`spec`](pipeline.md#BufferedRefBase.spec), [`advance_copy_in_slot`](pipeline.md#BufferedRefBase.advance_copy_in_slot), [`advance_copy_out_slot`](pipeline.md#BufferedRefBase.advance_copy_out_slot), [`advance_wait_in_slot`](pipeline.md#BufferedRefBase.advance_wait_in_slot), [`advance_wait_out_slot`](pipeline.md#BufferedRefBase.advance_wait_out_slot), [`bind_existing_ref`](pipeline.md#BufferedRefBase.bind_existing_ref), [`initialize_slots`](pipeline.md#BufferedRefBase.initialize_slots), [`is_buffered`](pipeline.md#BufferedRefBase.is_buffered), [`unbind_refs`](pipeline.md#BufferedRefBase.unbind_refs)

### `BufferedRefBase`
- def: [`jax/_src/pallas/mosaic/pipeline.py:271`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L271)
- doc: Abstract interface for BufferedRefs.
- signature: `class BufferedRefBase:`
- members:
  - `advance_copy_in_slot(self, predicate: bool = True)` — [`L323`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L323) — Advance the copy in slot.
  - `advance_copy_out_slot(self, predicate: bool = True)` — [`L331`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L331) — Advance the copy out slot.
  - `advance_wait_in_slot(self, predicate: bool = True)` — [`L327`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L327) — Advance the wait in slot.
  - `advance_wait_out_slot(self, predicate: bool = True)` — [`L335`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L335) — Advance the wait out slot.
  - `bind_existing_ref(self, window_ref, indices)` — [`L430`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L430) — For handling VMEM references, the pipeline aliases the existing ref.
  - `block_shape(self)` — [`L340`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L340)
  - `buffer_type(self)` — [`L279`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L279)
  - `compute_index(self)` — [`L356`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L356)
  - `get_dma_slice(self, src_ty, grid_indices)` — [`L359`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L359)
  - `has_allocated_buffer(self)` — [`L351`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L351) — Returns True if the reference has an allocated buffer outside loop.
  - `has_indirect(self)` — [`L344`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L344) — Whether any block dimension uses indirect indexing.
  - `initialize_slots(self)` — [`L319`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L319) — Initializes slots to 0.
  - `is_buffered(self)` — [`L283`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L283)
  - `is_input(self)` — [`L287`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L287)
  - `is_input_output(self)` — [`L295`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L295)
  - `is_manual(self)` — [`L299`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L299)
  - `is_output(self)` — [`L291`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L291)
  - `is_trivial_windowing(self)` — [`L303`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L303) — Whether the reference uses trivial windowing.
  - `prefetched_count(self)` — [`L316`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L316)
  - `spec(self)` — [`L275`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L275)
  - `unbind_refs(self)` — [`L435`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L435)
  - `with_spec(self, spec: pallas_core.BlockSpec)` — [`L438`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L438) — Returns a new BufferedRefBase with the given block spec.
- protocol/private: `_to_window_slice`[`L423`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L423)
- uses (calls/refs, reference-scoped): [`BlockSpec`](../core.md#BlockSpec), [`block_shape`](../core.md#BlockSpec.block_shape), [`index_map`](../core.md#BlockSpec.index_map), [`_make_block_slice`](pipeline.md#_make_block_slice), [`BufferedRef`](pipeline.md#BufferedRef), [`infer_tiling`](tpu_info.md#infer_tiling), [`initialize_slots`](pipeline.md#BufferedRef.initialize_slots), [`is_buffered`](pipeline.md#BufferedRef.is_buffered), [`advance_copy_in_slot`](pipeline.md#BufferedRef.advance_copy_in_slot), [`advance_copy_out_slot`](pipeline.md#BufferedRef.advance_copy_out_slot), [`advance_wait_in_slot`](pipeline.md#BufferedRef.advance_wait_in_slot), [`advance_wait_out_slot`](pipeline.md#BufferedRef.advance_wait_out_slot), [`BufferType`](pipeline.md#BufferType), [`ds`](pipeline.md#ds), [`is_input`](pipeline.md#BufferType.is_input), [`is_output`](pipeline.md#BufferType.is_output), [`bind_existing_ref`](pipeline.md#BufferedRef.bind_existing_ref), [`Squeezed`](pipeline.md#Squeezed), [`Indirect`](pipeline.md#Indirect), [`buffer_type`](pipeline.md#BufferedRef.buffer_type), [`unbind_refs`](pipeline.md#BufferedRef.unbind_refs), [`with_spec`](pipeline.md#BufferedRef.with_spec), [`BlockDim`](pipeline.md#BlockDim), [`INPUT_OUTPUT`](pipeline.md#BufferType.INPUT_OUTPUT), [`spec`](pipeline.md#BufferedRef.spec), [`MANUAL`](pipeline.md#BufferType.MANUAL)
- used by: [`copy_out`](pipeline.md#BufferedRef.copy_out), [`wait_out`](pipeline.md#BufferedRef.wait_out), [`copy_in`](pipeline.md#BufferedRef.copy_in), [`wait_in`](pipeline.md#BufferedRef.wait_in), [`BufferedRef`](pipeline.md#BufferedRef), [`sync_copy`](pipeline.md#sync_copy), [`current_ref`](pipeline.md#BufferedRef.current_ref), [`map_brefs`](pipeline.md#map_brefs), [`compute_slice`](pipeline.md#BufferedRef.compute_slice), [`advance_copy_in_slot`](pipeline.md#BufferedRef.advance_copy_in_slot), [`advance_copy_out_slot`](pipeline.md#BufferedRef.advance_copy_out_slot), [`advance_wait_in_slot`](pipeline.md#BufferedRef.advance_wait_in_slot), [`advance_wait_out_slot`](pipeline.md#BufferedRef.advance_wait_out_slot), [`__post_init__`](pipeline.md#BufferedRef.__post_init__)

### `PipelineStep`
- def: [`jax/_src/pallas/mosaic/pipeline.py:1637`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1637)
- signature: `class PipelineStep:`
- members:
  - `tree_flatten(self)` — [`L1641`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1641)
  - `tree_unflatten(cls, aux, children)` — [`L1653`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1653)
  - `index` — [`L1638`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1638)
  - `local_index` — [`L1639`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1639)
- uses (calls/refs, reference-scoped): [`Array`](../../basearray.md#Array), [`register_pytree_node_class`](../../tree_util.md#register_pytree_node_class)
- used by: [`wrapped`](pipeline.md#emit_pipeline.wrapped), [`loop_body`](pipeline.md#_emit_pipeline.pipeline.loop_body), [`_loop_body`](pipeline.md#_emit_pipeline.pipeline._loop_body)

### `Scheduler`
- def: [`jax/_src/pallas/mosaic/pipeline.py:1126`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1126)
- doc: Sequences input and output copies and waits for a pipeline.
- signature: `class Scheduler:`
- members:
  - `__init__(self, step: jax.Array, indices: tuple[int | jax.Array, ...], grid: tuple[int | jax.Array, ...], grid_offsets: tuple[int | jax.Array, ...], num_stages: int, trace_scopes=True, _explicit_indices: bool = False)` — [`L1129`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1129) — Initializes scheduler.
  - `advance_slots(self, buffered_ref)` — [`L1391`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1391)
  - `alias_local_refs(self, buffered_ref, ref)` — [`L1246`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1246)
  - `copy_in(self, buffered_ref, src_ref)` — [`L1322`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1322)
  - `copy_out(self, buffered_ref, dst_ref)` — [`L1367`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1367)
  - `finalize(self, buffered_ref, dst_ref)` — [`L1380`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1380)
  - `grid_env(self)` — [`L1201`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1201)
  - `has_changed(self, buffered_ref)` — [`L1215`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1215)
  - `initialize_step(self, buffered_ref, src_ref, step=0)` — [`L1257`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1257)
  - `out_of_fetch(self, buffered_ref)` — [`L1207`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1207) — Returns whether there are no more blocks to fetch.
  - `unalias_local_refs(self, buffered_ref)` — [`L1249`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1249)
  - `wait_in(self, buffered_ref, src_ref)` — [`L1309`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1309)
  - `wait_out(self, buffered_ref, dst_ref)` — [`L1350`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1350)
  - `will_change_current(self, buffered_ref)` — [`L1224`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1224)
  - `will_change_fetch(self, buffered_ref)` — [`L1233`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1233)
  - `add_offset` — [`L1178`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1178)
  - `fetch_indices` — [`L1184`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1184)
  - `first_step` — [`L1161`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1161)
  - `grid` — [`L1151`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1151)
  - `grid_offsets` — [`L1152`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1152)
  - `indices` — [`L1165`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1165)
  - `last_step` — [`L1162`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1162)
  - `next_indices` — [`L1174`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1174)
  - `num_stages` — [`L1153`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1153)
  - `num_steps` — [`L1158`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1158)
  - `prev_indices` — [`L1169`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1169)
  - `step` — [`L1150`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1150)
  - `trace_scopes` — [`L1154`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1154)
- protocol/private: `_copy_out`[`L1374`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1374), `_end`[`L1387`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1387), `_explicit_indices`[`L1155`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1155), `_named_scope`[`L1194`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1194), `_send`[`L1342`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1342), `_start`[`L1279`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1279), `_wait`[`L1317`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1317)
- uses (calls/refs, reference-scoped): [`Array`](../../basearray.md#Array), [`named_scope`](../../api.md#named_scope), [`BufferedRef`](pipeline.md#BufferedRef), [`fetch_with_lookahead`](pipeline.md#fetch_with_lookahead), [`_next_index`](pipeline.md#_next_index), [`grid_env`](../core.md#grid_env), [`when`](pipeline.md#when), [`_prev_index`](pipeline.md#_prev_index), [`_tuples_differ`](pipeline.md#_tuples_differ), [`GridAxis`](../core.md#GridAxis), [`contextmanager`](pipeline.md#contextmanager)
- used by: [`pipeline`](pipeline.md#_emit_pipeline.pipeline), [`loop_body`](pipeline.md#_emit_pipeline.pipeline.loop_body), [`_loop_body`](pipeline.md#_emit_pipeline.pipeline._loop_body), [`make_scheduler`](pipeline.md#_emit_pipeline.pipeline.make_scheduler)

## Functions
- `_create_blocked_slice(block_index: jax.Array | int, block_size: int, dim_size: int, tiling: int | None)` — [`L92`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L92)
- `_create_bounded_slice(slice_start: jax.Array | int, slice_size: jax.Array | int, block_size: int, dim_size: int, tiling: int | None)` — [`L114`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L114)
- `_emit_pipeline(body, *, grid: tuple[int | jax.Array, ...], in_specs=(), out_specs=(), tiling: Tiling | None = None, core_axis: tuple[int, ...] | int | None = None, core_axis_name: tuple[str, ...] | str | None = None, dimension_semantics: tuple[GridDimensionSemantics, ...] | None = None, trace_scopes: bool = True, no_pipelining: bool = False, _explicit_indices: bool = False, _grid_offsets: tuple[int | jax.Array, ...] | None = None)` — [`L1660`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1660) — Creates a function to emit a manual pallas pipeline.
- `_emit_pipeline_effectful_abstract_eval(*avals, body_jaxpr: core.Jaxpr, body_consts_len, grid_mapping, _num_extra_dynamic, args_tree, _explicit_indices, **params)` — [`L2074`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L2074)
- `_emit_pipeline_lowering_rule(ctx, *args, grid_mapping, _num_extra_dynamic, _static_grid_offsets, args_tree, body_jaxpr, body_consts_len, _explicit_indices, **params)` — [`L2241`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L2241)
- `_filter_indices(indices: tuple[int | jax.Array, ...], grid: tuple[int | jax.Array, ...])` — [`L1072`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1072)
- `_get_block_shape(spec: pallas_core.BlockSpec)` — [`L248`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L248) — Get the block shape for a given block spec.
- `_get_dim_size(bd)` — [`L250`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L250)
- `_loop_body(carry)` — [`L1020`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1020)
- `_loop_body(step, carry)` — [`L1820`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1820)
- `_loop_cond(carry)` — [`L1012`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1012)
- `_make_block_slice(block_index: jax.Array, block_size: BlockDim | int | None, size: int, tiling: int | None)` — [`L136`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L136)
- `_make_pipeline_allocations(*refs, in_specs=(), out_specs=(), tiling: Tiling | None = None, grid=())` — [`L1413`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1413) — Create BufferedRefs for the pipeline.
- `_next_index(indices: tuple[int | jax.Array, ...], grid: tuple[int | jax.Array, ...], allow_overflow: bool = False)` — [`L1081`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1081) — Increments the grid indices by one.
- `_normalize_specs(specs: Any)` — [`L1405`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1405)
- `_partition_grid(grid: tuple[int | jax.Array, ...], core_axis: tuple[int | str, ...] | int | str | None, dimension_semantics: tuple[GridDimensionSemantics, ...] | None)` — [`L1491`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1491)
- `_pipeline_body_effectful_abstract_eval(*avals, jaxpr, in_tree, num_inputs, _explicit_indices=False, **params)` — [`L2149`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L2149)
- `_pipeline_body_lowering_rule(ctx, *args_flat, jaxpr, in_tree, _explicit_indices=False, **_)` — [`L2195`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L2195)
- `_prev_index(indices: tuple[int | jax.Array, ...], grid: tuple[int | jax.Array, ...])` — [`L1114`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1114)
- `_ref_to_value_aval(ref)` — [`L443`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L443) — Return the inner of a ref, or a ShapedArray for TransformedRefs.
- `_spec_has_trivial_windowing(spec, grid, full_shape)` — [`L185`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L185)
- `_start()` — [`L1030`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1030)
- `_sync_copy_in(bref, ref)` — [`L1854`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1854)
- `_sync_copy_out(bref, ref)` — [`L1883`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1883)
- `_tuple_all_binop(binop, xs, ys)` — [`L178`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L178) — Dynamic reduce_all calculation with a user-provided comparison op.
- `_tuples_differ(xs, ys)` — [`L173`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L173) — Dynamic index-tuple comparison calculation.
- `as_uint32(x)` — [`L998`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L998)
- `emit_pipeline(body, *, grid: tuple[int | jax.Array, ...], in_specs=(), out_specs=(), tiling: Tiling | None = None, core_axis: tuple[int, ...] | int | None = None, core_axis_name: tuple[str, ...] | str | None = None, dimension_semantics: tuple[GridDimensionSemantics, ...] | None = None, trace_scopes: bool = True, no_pipelining: bool = False, _explicit_indices: bool = False)` — [`L1926`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1926)
- `emit_pipeline_with_allocations(body, *, grid, in_specs=(), out_specs=())` — [`L1892`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1892) — Creates pallas pipeline and top-level allocation preparation functions.
- `fetch_with_lookahead(buffered_ref, src_ref, grid, grid_offsets, predicate: jax.Array | bool = True, max_num_fetches: int | None = None, update_slots: bool = True)` — [`L975`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L975) — Fetch future blocks using unbounded lookahead.
- `fmap(bref, *f_args)` — [`L1056`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1056)
- `fmap(bref, *f_args)` — [`L1065`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1065)
- `loop_body(step, carry)` — [`L1774`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1774)
- `make_input_bref(in_spec, in_ref)` — [`L1441`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1441)
- `make_output_bref(out_spec, out_ref)` — [`L1468`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1468)
- `make_scheduler(step, indices)` — [`L1763`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1763)
- `map_inputs(f, *args)` — [`L1054`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1054) — Maps over all input BufferedRefs.
- `map_outputs(f, *args)` — [`L1063`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1063) — Maps over all output BufferedRefs.
- `new_body(ps, *args)` — [`L2280`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L2280)
- `pipeline(*refs: Any, scratches=None, allocations=None, body_prologue=None)` — [`L1725`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1725) — Run the pipeline.
- `sync_copy(src: REF | BufferedRef, dst: REF | BufferedRef, indices)` — [`L1607`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1607) — Perform a synchronous copy from src to dst.
- `wrapped(*args, allocations=None)` — [`L1965`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1965)
- `wrapped_pipeline_fun(*all_args, grid_mapping=grid_mapping)` — [`L2248`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L2248)

## Module values
- `ANY` — [`L74`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L74)
- `ARBITRARY` — [`L78`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L78)
- `ArrayRef` — [`L81`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L81)
- `BlockDim` — [`L68`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L68)
- `Blocked` — [`L67`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L67)
- `BoundedSlice` — [`L66`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L66)
- `CondVal` — [`L85`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L85)
- `Element` — [`L65`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L65)
- `GridDimensionSemantics` — [`L76`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L76)
- `GridIndices` — [`L84`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L84)
- `HBM` — [`L73`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L73)
- `Indirect` — [`L64`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L64)
- `PARALLEL` — [`L77`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L77)
- `PipelineBlockSpecs` — [`L86`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L86)
- `PipelineRefs` — [`L87`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L87)
- `REF` — [`L75`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L75)
- `SMEM` — [`L71`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L71)
- `SemaphoreTuple` — [`L80`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L80)
- `SemaphoreType` — [`L79`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L79)
- `Slice` — [`L69`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L69)
- `Squeezed` — [`L63`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L63)
- `Tiling` — [`L82`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L82)
- `VMEM` — [`L72`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L72)
- `_tuple_lt` — [`L183`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L183)
- `align_to` — [`L58`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L58)
- `cdiv` — [`L56`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L56)
- `contextmanager` — [`L57`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L57)
- `ds` — [`L70`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L70)
- `emit_pipeline_p` — [`L2067`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L2067)
- `is_transformed_ref` — [`L89`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L89)
- `map_brefs` — [`L1048`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L1048)
- `multiple_of` — [`L61`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L61)
- `num_programs` — [`L60`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L60)
- `pipeline_body_p` — [`L2141`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L2141)
- `program_id` — [`L59`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L59)
- `when` — [`L62`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/pipeline.py#L62)

