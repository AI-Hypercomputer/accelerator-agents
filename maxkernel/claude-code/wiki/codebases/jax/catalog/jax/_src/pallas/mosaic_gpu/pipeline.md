---
title: 'Module: jax/_src/pallas/mosaic_gpu/pipeline.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic_gpu/pipeline.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic_gpu.pipeline`/
symbols:
  emit_pipeline.scoped_pipeline: emit_pipeline().scoped_pipeline().
  emit_pipeline.scoped_pipeline.loop_body: emit_pipeline().scoped_pipeline().loop_body().
  emit_pipeline_warp_specialized.scoped_pipeline.compute_block: emit_pipeline_warp_specialized().scoped_pipeline().compute_block().
  emit_pipeline_warp_specialized.scoped_pipeline.compute_block.compute_loop_body: emit_pipeline_warp_specialized().scoped_pipeline().compute_block().compute_loop_body().
  emit_pipeline_warp_specialized: emit_pipeline_warp_specialized().
  emit_pipeline.pipeline: emit_pipeline().pipeline().
  emit_pipeline: emit_pipeline().
  BufferedRef.compute_gmem_slice: BufferedRef#compute_gmem_slice().
  emit_pipeline_warp_specialized.scoped_pipeline.memory_block: emit_pipeline_warp_specialized().scoped_pipeline().memory_block().
  emit_pipeline_warp_specialized._get_scoped_allocs: emit_pipeline_warp_specialized()._get_scoped_allocs().
  _inc_grid_by_1: _inc_grid_by_1().
  BufferedRef.copy_in: BufferedRef#copy_in().
  BufferedRef.copy_out: BufferedRef#copy_out().
  _downcast_spec: _downcast_spec().
  _in_smem: _in_smem().
  emit_pipeline_warp_specialized.scoped_pipeline.memory_block.memory_loop_body: emit_pipeline_warp_specialized().scoped_pipeline().memory_block().memory_loop_body().
  BufferedRef.spec: BufferedRef#spec.
  emit_pipeline_warp_specialized.scoped_pipeline: emit_pipeline_warp_specialized().scoped_pipeline().
  BufferedRef._make_block_slice: BufferedRef#_make_block_slice().
  _is_index_invariant: _is_index_invariant().
  BufferedRef.get_ref_for_slot: BufferedRef#get_ref_for_slot().
  BufferedRef.is_index_invariant: BufferedRef#is_index_invariant.
  _get_block_shape: _get_block_shape().
  emit_pipeline_warp_specialized.pipeline: emit_pipeline_warp_specialized().pipeline().
  zip: zip.
  _get_block_size: _get_block_size().
  emit_pipeline_warp_specialized.scoped_pipeline.memory_block._init_step: emit_pipeline_warp_specialized().scoped_pipeline().memory_block()._init_step().
  map: map.
  BufferedRef: BufferedRef#
  BufferedRef.smem_ref: BufferedRef#smem_ref.
  _uses_arguments: _uses_arguments().
  emit_pipeline_warp_specialized._get_slot: emit_pipeline_warp_specialized()._get_slot().
  emit_pipeline_warp_specialized.scoped_pipeline.memory_block._quiesce: emit_pipeline_warp_specialized().scoped_pipeline().memory_block()._quiesce().
  emit_pipeline_warp_specialized.scoped_pipeline.memory_block._quiesce._epi_step: emit_pipeline_warp_specialized().scoped_pipeline().memory_block()._quiesce()._epi_step().
  BufferedRef.gmem_ref: BufferedRef#gmem_ref.
  map_brefs: map_brefs.
  emit_pipeline.scoped_pipeline.prologue: emit_pipeline().scoped_pipeline().prologue().
  emit_pipeline.scoped_pipeline.loop_body.do_fetch: emit_pipeline().scoped_pipeline().loop_body().do_fetch().
  PipelinePipeline: PipelinePipeline#
  emit_pipeline_warp_specialized.scoped_pipeline.compute_block.pipeline_callback: emit_pipeline_warp_specialized().scoped_pipeline().compute_block().pipeline_callback().
  BlockSpecPytree.BlockSpecPytree: BlockSpecPytree.BlockSpecPytree.
  AbstractRefPytree.AbstractRefPytree: AbstractRefPytree.AbstractRefPytree.
  PipelineCallback: PipelineCallback#
  PipelineCallback.__call__: PipelineCallback#__call__().
  ComputeContext.__call__: ComputeContext#__call__().
  PipelinePipeline.START: PipelinePipeline#START.
  WarpSpecializedPipeline.get_allocations: WarpSpecializedPipeline#get_allocations().
  ComputeContext: ComputeContext#
  PipelinePipeline.STOP: PipelinePipeline#STOP.
  WarpSpecializedPipeline: WarpSpecializedPipeline#
  _compute_registers: _compute_registers().
  PipelinePipeline.STEADY: PipelinePipeline#STEADY.
  WarpSpecializedPipeline.__call__: WarpSpecializedPipeline#__call__().
---
# Module: [`jax/_src/pallas/mosaic_gpu/pipeline.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py)

## Classes
### `BufferedRef`
- def: [`jax/_src/pallas/mosaic_gpu/pipeline.py:82`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L82)
- members:
  - `compute_gmem_slice(self, grid_indices)` — [`L97`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L97)
  - `copy_in(self, slot, grid_indices, barrier_ref, barrier_slot=None)` — [`L125`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L125)
  - `copy_out(self, slot, grid_indices, predicate=None)` — [`L138`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L138)
  - `get_ref_for_slot(self, slot: int | jax.Array)` — [`L90`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L90)
  - `gmem_ref` — [`L85`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L85)
  - `is_index_invariant` — [`L84`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L84)
  - `smem_ref` — [`L88`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L88)
  - `spec` — [`L83`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L83)
- protocol/private: `_make_block_slice`[`L106`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L106)
- uses (calls/refs, reference-scoped): [`Array`](../../basearray.md#Array), [`AbstractRef`](../../state/types.md#AbstractRef), [`BlockSpec`](../core.md#BlockSpec), [`int32`](../../../numpy/__init__.pyi.md#int32.int32), [`dtype`](../../../numpy/__init__.pyi.md#dtype), [`Slice`](../../indexing.md#Slice), [`block_shape`](../core.md#BlockSpec.block_shape), [`Squeezed`](../core.md#Squeezed), [`shape`](../../state/types.md#AbstractRef.shape), [`copy_gmem_to_smem`](primitives.md#copy_gmem_to_smem), [`index_map`](../core.md#BlockSpec.index_map), [`Blocked`](../core.md#Blocked), [`BlockDim`](../core.md#BlockDim.BlockDim), [`copy_smem_to_gmem`](primitives.md#copy_smem_to_gmem), [`Element`](../core.md#Element), [`_in_smem`](pipeline.md#_in_smem), [`at`](../../state/types.md#AbstractRef.at), [`map`](pipeline.md#map)
- used by: [`scoped_pipeline`](pipeline.md#emit_pipeline.scoped_pipeline), [`compute_block`](pipeline.md#emit_pipeline_warp_specialized.scoped_pipeline.compute_block), [`loop_body`](pipeline.md#emit_pipeline.scoped_pipeline.loop_body), [`compute_loop_body`](pipeline.md#emit_pipeline_warp_specialized.scoped_pipeline.compute_block.compute_loop_body), [`memory_loop_body`](pipeline.md#emit_pipeline_warp_specialized.scoped_pipeline.memory_block.memory_loop_body), [`scoped_pipeline`](pipeline.md#emit_pipeline_warp_specialized.scoped_pipeline), [`_init_step`](pipeline.md#emit_pipeline_warp_specialized.scoped_pipeline.memory_block._init_step), [`do_fetch`](pipeline.md#emit_pipeline.scoped_pipeline.loop_body.do_fetch), [`map_brefs`](pipeline.md#map_brefs), [`prologue`](pipeline.md#emit_pipeline.scoped_pipeline.prologue)

### `ComputeContext`  ·  implements/extends Protocol
- def: [`jax/_src/pallas/mosaic_gpu/pipeline.py:487`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L487)
- doc: Protocol for a compute context for the warp specialized pipeline.
- signature: `class ComputeContext(Protocol):`
- protocol/private: `__call__`[`L511`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L511)
- uses (calls/refs, reference-scoped): [`PipelineCallback`](pipeline.md#PipelineCallback)
- used by: [`emit_pipeline_warp_specialized`](pipeline.md#emit_pipeline_warp_specialized)

### `PipelineCallback`  ·  implements/extends Protocol
- def: [`jax/_src/pallas/mosaic_gpu/pipeline.py:44`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L44)
- doc: A callback that returns the same type as the input.
- signature: `class PipelineCallback(Protocol):`
- protocol/private: `__call__`[`L46`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L46)
- used by: [`__call__`](pipeline.md#ComputeContext.__call__)

### `PipelinePipeline`  ·  implements/extends IntEnum
- def: [`jax/_src/pallas/mosaic_gpu/pipeline.py:515`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L515)
- signature: `class PipelinePipeline(enum.IntEnum):`
- members:
  - `START` — [`L516`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L516)
  - `STEADY` — [`L517`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L517)
  - `STOP` — [`L518`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L518)
- used by: [`emit_pipeline_warp_specialized`](pipeline.md#emit_pipeline_warp_specialized), [`memory_block`](pipeline.md#emit_pipeline_warp_specialized.scoped_pipeline.memory_block), [`_quiesce`](pipeline.md#emit_pipeline_warp_specialized.scoped_pipeline.memory_block._quiesce)

### `WarpSpecializedPipeline`  ·  implements/extends Protocol
- def: [`jax/_src/pallas/mosaic_gpu/pipeline.py:521`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L521)
- doc: Protocol for a warp specialized pipeline.
- signature: `class WarpSpecializedPipeline(Protocol):`
- members:
  - `get_allocations(self, *gmem_refs: Any)` — [`L528`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L528)
- protocol/private: `__call__`[`L523`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L523)
- used by: [`emit_pipeline_warp_specialized`](pipeline.md#emit_pipeline_warp_specialized), [`_pipeline_scope`](../../../experimental/pallas/ops/gpu/hopper_matmul_mgpu.md#kernel._pipeline_scope), [`_pipeline_scope`](../../../experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.md#mixed_matmul_kernel.kernel._pipeline_scope)

## Functions
- `_compute_registers(memory_registers: int, num_compute_wgs: int)` — [`L1064`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L1064) — Returns the max number of registers to use in compute threads.
- `_downcast_spec(spec: gpu_core.BlockSpec | pallas_core.BlockSpec)` — [`L195`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L195)
- `_epi_step(step)` — [`L1050`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L1050)
- `_get_block_shape(spec: pallas_core.BlockSpec, ref_shape: tuple[int, ...])` — [`L64`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L64)
- `_get_block_size(bd: pl.BlockDim | int | None)` — [`L54`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L54)
- `_get_scoped_allocs(*gmem_refs: AbstractRefPytree)` — [`L688`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L688)
- `_get_slot(step, has_seq_dim)` — [`L676`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L676) — Returns the buffer slot given the pipeline step.
- `_in_smem(spec: pallas_core.BlockSpec)` — [`L192`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L192)
- `_inc_grid_by_1(indices: tuple[jax.Array, ...], grid: pallas_core.TupleGrid)` — [`L178`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L178)
- `_init_step(step, indices)` — [`L1005`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L1005)
- `_is_index_invariant(spec: pallas_core.BlockSpec, grid: pallas_core.TupleGrid)` — [`L170`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L170)
- `_quiesce()` — [`L1044`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L1044)
- `_uses_arguments(index_map: Callable[..., Any], num_args: int)` — [`L152`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L152)
- `compute_block()` — [`L825`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L825)
- `compute_loop_body(step, carry)` — [`L850`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L850)
- `do_fetch()` — [`L422`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L422)
- `emit_pipeline(body: Callable[..., T], *, grid: pallas_core.TupleGrid, in_specs: Sequence[pallas_core.BlockSpec] = (), out_specs: Sequence[pallas_core.BlockSpec] = (), max_concurrent_steps: int = 1, init_carry: T | None = None)` — [`L215`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L215) — Creates a function to emit a manual pipeline within a Pallas kernel.
- `emit_pipeline_warp_specialized(body: Callable[..., None], *, grid: pallas_core.TupleGrid, memory_registers: int, in_specs: BlockSpecPytree = (), out_specs: BlockSpecPytree = (), max_concurrent_steps: int = 2, wg_axis: str, num_compute_wgs: int, pipeline_state: jax.Array | PipelinePipeline | None = None, manual_consumed_barriers: bool = False, compute_context: ComputeContext | None = None, memory_thread_idx: int | None = None)` — [`L532`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L532) — Creates a function to emit a warp-specialized pipeline.
- `loop_body(step, carry)` — [`L362`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L362)
- `memory_block()` — [`L975`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L975)
- `memory_loop_body(step, carry)` — [`L1016`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L1016)
- `pipeline(*gmem_refs: state.AbstractRef)` — [`L288`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L288)
- `pipeline(*gmem_refs: AbstractRefPytree, allocations: Any | None = None)` — [`L755`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L755) — Run the pipeline.
- `pipeline_callback(user_init_carry)` — [`L933`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L933)
- `prologue(step, fetch_indices)` — [`L348`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L348)
- `scoped_pipeline(*, in_gmem_refs, out_gmem_refs, in_smem_refs, out_smem_refs, barrier_ref)` — [`L324`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L324)
- `scoped_pipeline(*, flat_in_gmem_refs, flat_out_gmem_refs, flat_in_smem_refs, flat_out_smem_refs, in_smem_barrier_ref, flat_consumed_barrier_refs)` — [`L803`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L803)

## Module values
- `AbstractRefPytree` — [`L49`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L49)
- `BlockSpecPytree` — [`L48`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L48)
- `map` — [`L42`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L42)
- `map_brefs` — [`L76`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L76)
- `zip` — [`L43`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/pipeline.py#L43)

