---
title: 'Module: jax/_src/pallas/mosaic_gpu/core.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic_gpu/core.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic_gpu.core`/
symbols:
  Layout.to_mgpu: Layout#to_mgpu().
  MemorySpace: MemorySpace#
  kernel: kernel().
  GPUMemoryRef.get_ref_aval: GPUMemoryRef#get_ref_aval().
  flatten_ref_union: flatten_ref_union().
  LANExWARP_SEMANTICS: LANExWARP_SEMANTICS.
  WGxWARP_SEMANTICS: WGxWARP_SEMANTICS.
  MemorySpace.__call__: MemorySpace#__call__().
  Layout: Layout#
  PrimitiveSemantics: PrimitiveSemantics#
  flatten_ref_union.unflatten: flatten_ref_union().unflatten().
  BlockSpec.to_block_mapping: BlockSpec#to_block_mapping().
  UntilingTransform.commute_ndindexer: UntilingTransform#commute_ndindexer().
  SMEM: SMEM.
  UntilingTransform: UntilingTransform#
  ExpandLeadingBatchDimensionsTransform.commute_ndindexer: ExpandLeadingBatchDimensionsTransform#commute_ndindexer().
  ClusterBarrier.get_ref_aval: ClusterBarrier#get_ref_aval().
  UnswizzleRef: UnswizzleRef#
  PrimitiveSemantics.Warp: PrimitiveSemantics#Warp.
  remote_ref: remote_ref().
  BlockSpec: BlockSpec#
  Barrier: Barrier#
  Barrier.num_barriers: Barrier#num_barriers.
  UntilingTransform.tiling: UntilingTransform#tiling.
  commute_transpose_indexer: commute_transpose_indexer().
  SwizzleTransform.transform_type: SwizzleTransform#transform_type().
  ExpandLeadingBatchDimensionsTransform.transform_type: ExpandLeadingBatchDimensionsTransform#transform_type().
  RefUnion.__init__: RefUnion#__init__().
  WGMMAAccumulatorRef.get_ref_aval: WGMMAAccumulatorRef#get_ref_aval().
  layout_cast: layout_cast().
  CompilerParams: CompilerParams#
  _ref_group_size: _ref_group_size().
  cluster_ref: cluster_ref().
  multicast_ref: multicast_ref().
  CollapseLeadingBatchDimensionsTransform.transform_type: CollapseLeadingBatchDimensionsTransform#transform_type().
  _wgmma_ref_type_mapping: _wgmma_ref_type_mapping().
  UntilingTransform.transform_type: UntilingTransform#transform_type().
  _memory_effect: _memory_effect.
  SemaphoreType.__call__: SemaphoreType#__call__().
  UnswizzleRef.commute_ndindexer: UnswizzleRef#commute_ndindexer().
  ExtractAliasedRef.transform_type: ExtractAliasedRef#transform_type().
  AbstractTMEMRef: AbstractTMEMRef#
  transpose_ref: transpose_ref().
  GMEM: GMEM.
  TilingTransform.transform_type: TilingTransform#transform_type().
  UntilingTransform.commute_reshape: UntilingTransform#commute_reshape().
  TMEM: TMEM.
  Barrier.num_arrivals: Barrier#num_arrivals.
  TilingTransform: TilingTransform#
  Barrier.get_ref_aval: Barrier#get_ref_aval().
  SwizzleTransform: SwizzleTransform#
  to_gpu_transform: to_gpu_transform().
  WGMMAAbstractAccumulatorRef: WGMMAAbstractAccumulatorRef#
  CompilerParams.lowering_semantics: CompilerParams#lowering_semantics.
  LANExWG_SEMANTICS: LANExWG_SEMANTICS.
  WGxWG_SEMANTICS: WGxWG_SEMANTICS.
  transform_ref: transform_ref().
  UnswizzleRef.commute_reshape: UnswizzleRef#commute_reshape().
  Mesh.shape: Mesh#shape().
  Layout.WGMMA: Layout#WGMMA.
  BlockSpec.transforms: BlockSpec#transforms.
  Mesh.__post_init__: Mesh#__post_init__().
  ReducedLayout.to_mgpu: ReducedLayout#to_mgpu().
  kernel.wrapper: kernel().wrapper().
  Mesh: Mesh#
  PeerMemRef: PeerMemRef#
  UntilingTransform.commute_transpose: UntilingTransform#commute_transpose().
  batch_transform: batch_transform().
  TMEMLayout.to_mgpu: TMEMLayout#to_mgpu().
  UnswizzleRef.swizzle_elems: UnswizzleRef#swizzle_elems().
  SomeLayout.to_mgpu: SomeLayout#to_mgpu().
  GPUMemoryRef.__post_init__: GPUMemoryRef#__post_init__().
  RefUnion.get_ref_aval: RefUnion#get_ref_aval().
  ParameterizedLayout.to_mgpu: ParameterizedLayout#to_mgpu().
  UntilingTransform._maybe_cdiv_with_cast: UntilingTransform#_maybe_cdiv_with_cast().
  to_transform_attr: to_transform_attr().
  MulticastRef: MulticastRef#
  ExtractAliasedRef.from_transformed_ref: ExtractAliasedRef#from_transformed_ref().
  Mesh.thread_name: Mesh#thread_name.
  _Ref: _Ref.
  MemorySpace.TMEM: MemorySpace#TMEM.
  _ref_group_tmem_col_size: _ref_group_tmem_col_size().
  UntilingTransform.undo: UntilingTransform#undo().
  BlockSpec.collective_axes: BlockSpec#collective_axes.
  BarrierType: BarrierType#
  ClusterBarrierType: ClusterBarrierType#
  _gpu_mesh_discharge_rule: _gpu_mesh_discharge_rule().
  SomeLayout.reduce: SomeLayout#reduce().
  kernel._vmap_rule: kernel()._vmap_rule().
  AbstractRefUnion.update: AbstractRefUnion#update().
  AbstractRefUnion.__hash__: AbstractRefUnion#__hash__().
  TilingTransform.undo: TilingTransform#undo().
  UntilingTransform.pretty_print: UntilingTransform#pretty_print().
  ClusterRefTransform: ClusterRefTransform#
  SwizzleTransform.undo: SwizzleTransform#undo().
  UnswizzleRef.undo: UnswizzleRef#undo().
  UnswizzleRef.commute_transpose: UnswizzleRef#commute_transpose().
  UnswizzleRef.pretty_print: UnswizzleRef#pretty_print().
  Mesh.cluster_names: Mesh#cluster_names.
  Mesh.num_threads: Mesh#num_threads.
  WarpMesh.supported_memory_spaces: WarpMesh#supported_memory_spaces().
  WarpMesh.axis_name: WarpMesh#axis_name.
  _wgmma_pipeline_effect: _wgmma_pipeline_effect.
  TryClusterCancelResult: TryClusterCancelResult().
  SomeLayout: SomeLayout#
  MemorySpace.REGS: MemorySpace#REGS.
  align_to: align_to().
  BlockSpec.delay_release: BlockSpec#delay_release.
  Mesh.cluster: Mesh#cluster.
  ExpandLeadingBatchDimensionsTransform: ExpandLeadingBatchDimensionsTransform#
  kernel._vmap_rule.batched_body: kernel()._vmap_rule().batched_body().
  AbstractRefUnion.refs: AbstractRefUnion#refs.
  AbstractRefUnion.collective: AbstractRefUnion#collective().
  WarpMesh: WarpMesh#
  CompilerParams.profile_trace_scope: CompilerParams#profile_trace_scope.
  MemorySpace.GMEM: MemorySpace#GMEM.
  Mesh.grid_names: Mesh#grid_names.
  Layout.check_no_args: Layout#check_no_args().
  GPUMemoryRef: GPUMemoryRef#
  ExtractAliasedRef: ExtractAliasedRef#
  CompilerParams.dimension_semantics: CompilerParams#dimension_semantics.
  CompilerParams.__post_init__: CompilerParams#__post_init__().
  AbstractRefUnion.__init__: AbstractRefUnion#__init__().
  PeerMemRef.commute_ndindexer: PeerMemRef#commute_ndindexer().
  MulticastRef.commute_ndindexer: MulticastRef#commute_ndindexer().
  ClusterRefTransform.commute_ndindexer: ClusterRefTransform#commute_ndindexer().
  CollapseLeadingBatchDimensionsTransform.undo: CollapseLeadingBatchDimensionsTransform#undo().
  WGMMAAbstractAccumulatorRef.update: WGMMAAbstractAccumulatorRef#update().
  AbstractTMEMRef.update: AbstractTMEMRef#update().
  Mesh.discharges_effect: Mesh#discharges_effect().
  layout_cast_p: layout_cast_p.
  ParameterizedLayout.__post_init__: ParameterizedLayout#__post_init__().
  AbstractRefUnion: AbstractRefUnion#
  SMEM_ALIGNMENT: SMEM_ALIGNMENT.
  Barrier.orders_tensor_core: Barrier#orders_tensor_core.
  WGMMAAccumulatorRef.dtype: WGMMAAccumulatorRef#dtype.
  Mesh.grid: Mesh#grid.
  IdentityTransform: IdentityTransform#
  TMEMLayout: TMEMLayout#
  TransposeTransform: TransposeTransform.
  GPUMemoryRef.transforms: GPUMemoryRef#transforms.
  ParameterizedLayout: ParameterizedLayout#
  MemorySpace.SMEM: MemorySpace#SMEM.
  PrimitiveSemantics.Warpgroup: PrimitiveSemantics#Warpgroup.
  _GPUMemoryRefTree: _GPUMemoryRefTree.
  infer_tmem_layout: infer_tmem_layout().
  TilingTransform.tiling: TilingTransform#tiling.
  REGS: REGS.
  ClusterBarrier.num_barriers: ClusterBarrier#num_barriers.
  WGMMAAccumulatorRef: WGMMAAccumulatorRef#
  ParameterizedLayout.layout_cls: ParameterizedLayout#layout_cls.
  CollapseLeadingBatchDimensionsTransform: CollapseLeadingBatchDimensionsTransform#
  RefUnion.refs: RefUnion#refs.
  WGMMAAccumulatorRef._init: WGMMAAccumulatorRef#_init.
  MemoryEffect: MemoryEffect#
  PdlEffect: PdlEffect#
  TraceScope: TraceScope#
  CompilerParams.approx_math: CompilerParams#approx_math.
  SemaphoreType: SemaphoreType#
  SemaphoreType.get_array_aval: SemaphoreType#get_array_aval().
  SemaphoreType.get_ref_aval: SemaphoreType#get_ref_aval().
  GPUMemoryRef.layout: GPUMemoryRef#layout.
  GPUMemoryRef.collective: GPUMemoryRef#collective.
  AbstractRefUnion.layout: AbstractRefUnion#layout().
  AbstractRefUnion.__eq__: AbstractRefUnion#__eq__().
  PeerMemRef.undo: PeerMemRef#undo().
  MulticastRef.undo: MulticastRef#undo().
  ClusterRefTransform.__post_init__: ClusterRefTransform#__post_init__().
  ClusterRefTransform.undo: ClusterRefTransform#undo().
  SwizzleTransform.swizzle: SwizzleTransform#swizzle.
  UnswizzleRef.swizzle: UnswizzleRef#swizzle.
  ExpandLeadingBatchDimensionsTransform.batch_shape: ExpandLeadingBatchDimensionsTransform#batch_shape.
  IdentityTransform.undo: IdentityTransform#undo().
  Barrier.__post_init__: Barrier#__post_init__().
  ClusterBarrier: ClusterBarrier#
  ClusterBarrier.__post_init__: ClusterBarrier#__post_init__().
  WGMMAAccumulatorRef.init: WGMMAAccumulatorRef#init().
  WGMMAAbstractAccumulatorRef._getitem: WGMMAAbstractAccumulatorRef#_getitem().
  WGMMAAbstractAccumulatorRef._setitem: WGMMAAbstractAccumulatorRef#_setitem().
  AbstractTMEMRef.__init__: AbstractTMEMRef#__init__().
  Mesh.default_memory_space: Mesh#default_memory_space().
  Mesh.tracing_context: Mesh#tracing_context().
  WarpMesh.shape: WarpMesh#shape().
  _layout_cast_abstract_eval: _layout_cast_abstract_eval().
  ScratchShapeTree: ScratchShapeTree.
  is_trivial_index: is_trivial_index().
  is_trivial_index._slices: is_trivial_index()._slices().
  barrier_dtype: barrier_dtype#
  _pdl_effect: _pdl_effect.
  ClusterBarrier.num_arrivals: ClusterBarrier#num_arrivals.
  kernel.wrapper.kernel_body: kernel().wrapper().kernel_body().
  RefUnion: RefUnion#
  PeerMemRef.device_id_type: PeerMemRef#device_id_type.
  ClusterRefTransform.dims: ClusterRefTransform#dims.
  ExtractAliasedRef.dtype: ExtractAliasedRef#dtype.
  ReducedLayout.layout: ReducedLayout#layout.
  ReducedLayout: ReducedLayout#
  TMEM_COL_ALIGNMENT: TMEM_COL_ALIGNMENT.
  CompilerParams.profile_space: CompilerParams#profile_space.
  SemaphoreType.REGULAR: SemaphoreType#REGULAR.
  ClusterBarrierType.collective_axes: ClusterBarrierType#collective_axes.
  ClusterBarrierType.num_arrivals: ClusterBarrierType#num_arrivals.
  ClusterBarrierType.orders_tensor_core: ClusterBarrierType#orders_tensor_core.
  ClusterBarrierType.leader_tracked: ClusterBarrierType#leader_tracked.
  ClusterBarrier.collective_axes: ClusterBarrier#collective_axes.
  _WGMMAPipelineEffect: _WGMMAPipelineEffect#
  Layout.WG_STRIDED: Layout#WG_STRIDED.
  AbstractRefUnion._iter: AbstractRefUnion#_iter().
  Index: Index.
  SwizzleTransform.__post_init__: SwizzleTransform#__post_init__().
  UnswizzleRef.transform_type: UnswizzleRef#transform_type().
  IdentityTransform.transform_type: IdentityTransform#transform_type().
  BarrierType.__str__: BarrierType#__str__().
  ClusterBarrierType.__str__: ClusterBarrierType#__str__().
  Barrier.get_array_aval: Barrier#get_array_aval().
  ClusterBarrier.get_array_aval: ClusterBarrier#get_array_aval().
  WGMMAAbstractAccumulatorRef.__repr__: WGMMAAbstractAccumulatorRef#__repr__().
  AbstractTMEMRef.__repr__: AbstractTMEMRef#__repr__().
  Mesh.supported_memory_spaces: Mesh#supported_memory_spaces().
  WarpMesh.default_memory_space: WarpMesh#default_memory_space().
  WarpMesh.discharges_effect: WarpMesh#discharges_effect().
  Layout.__call__: Layout#__call__().
  TMEMLayout.__call__: TMEMLayout#__call__().
  TraceScope.WARPGROUP: TraceScope#WARPGROUP.
  CompilerParams.unsafe_no_auto_barriers: CompilerParams#unsafe_no_auto_barriers.
  CompilerParams.reduction_scratch_bytes: CompilerParams#reduction_scratch_bytes.
  CompilerParams.profile_dir: CompilerParams#profile_dir.
  PeerMemRef.device_id: PeerMemRef#device_id.
  MulticastRef.collective_axes: MulticastRef#collective_axes.
  BarrierType.num_arrivals: BarrierType#num_arrivals.
  ClusterBarrier.orders_tensor_core: ClusterBarrier#orders_tensor_core.
  WGMMAAccumulatorRef.shape: WGMMAAccumulatorRef#shape.
  Mesh.kernel_name: Mesh#kernel_name.
  ParameterizedLayout.args: ParameterizedLayout#args.
  ParameterizedLayout.kwargs: ParameterizedLayout#kwargs.
  Layout.TCGEN05: Layout#TCGEN05.
  DimensionSemantics: DimensionSemantics.
  TraceScope.WARP: TraceScope#WARP.
  CompilerParams.max_concurrent_steps: CompilerParams#max_concurrent_steps.
  SemaphoreType.BARRIER: SemaphoreType#BARRIER.
  ClusterRefTransform.idxs: ClusterRefTransform#idxs.
  ExtractAliasedRef.shape: ExtractAliasedRef#shape.
  BarrierType.name: BarrierType#name.
  BarrierType.orders_tensor_core: BarrierType#orders_tensor_core.
  ClusterBarrierType.name: ClusterBarrierType#name.
  ClusterBarrier.leader_tracked: ClusterBarrier#leader_tracked.
  WarpMesh._NUM_WARPS_PER_WARPGROUP: WarpMesh#_NUM_WARPS_PER_WARPGROUP.
  ReducedLayout.axes: ReducedLayout#axes.
  Layout.WGMMA_8BIT: Layout#WGMMA_8BIT.
  Layout.WGMMA_UPCAST_2X: Layout#WGMMA_UPCAST_2X.
  Layout.WGMMA_UPCAST_4X: Layout#WGMMA_UPCAST_4X.
  Layout.WGMMA_TRANSPOSED: Layout#WGMMA_TRANSPOSED.
  Layout.WG_SPLAT: Layout#WG_SPLAT.
  Layout.TILED: Layout#TILED.
  Layout.TCGEN05_TRANSPOSED: Layout#TCGEN05_TRANSPOSED.
  Layout.TCGEN05_M64_COLLECTIVE: Layout#TCGEN05_M64_COLLECTIVE.
  Layout.TCGEN05_TMEM_NATIVE: Layout#TCGEN05_TMEM_NATIVE.
  Layout.TCGEN05_M64_COLLECTIVE_NATIVE: Layout#TCGEN05_M64_COLLECTIVE_NATIVE.
  Layout.SMEM_GMEM_COPY: Layout#SMEM_GMEM_COPY.
  Layout.TMA_INDICES: Layout#TMA_INDICES.
  Layout.TMA_INDICES_4: Layout#TMA_INDICES_4.
  Layout._WGMMA_ACC_32BIT: Layout#_WGMMA_ACC_32BIT.
  TMEMLayout.SCALES_LAYOUT: TMEMLayout#SCALES_LAYOUT.
  TMEMLayout.SPARSE_METADATA_LAYOUT: TMEMLayout#SPARSE_METADATA_LAYOUT.
  TMEMLayout.M64_COLLECTIVE_LAYOUT: TMEMLayout#M64_COLLECTIVE_LAYOUT.
  TMEMLayout.SCALES_M64_COLLECTIVE_LAYOUT: TMEMLayout#SCALES_M64_COLLECTIVE_LAYOUT.
  CompilerParams.replace: CompilerParams#replace.
  MemorySpace.__str__: MemorySpace#__str__().
  MemorySpace.like: MemorySpace#like().
  AbstractRefUnion._getitem: AbstractRefUnion#_getitem().
  AbstractRefUnion._setitem: AbstractRefUnion#_setitem().
  PeerMemRef.transform_type: PeerMemRef#transform_type().
  MulticastRef.transform_type: MulticastRef#transform_type().
  ClusterRefTransform.transform_type: ClusterRefTransform#transform_type().
  ExtractAliasedRef.offset: ExtractAliasedRef#offset.
  ExtractAliasedRef.alias_group_idx: ExtractAliasedRef#alias_group_idx.
  ExtractAliasedRef.layout: ExtractAliasedRef#layout.
  WGMMAAbstractAccumulatorRef.__slots__: WGMMAAbstractAccumulatorRef#__slots__.
  AbstractTMEMRef.__slots__: AbstractTMEMRef#__slots__.
  _WARPGROUP_AXIS_NAME: _WARPGROUP_AXIS_NAME.
  Mesh.check_is_compatible_with: Mesh#check_is_compatible_with().
  Mesh.core_type: Mesh#core_type().
  WarpMesh.check_is_compatible_with: WarpMesh#check_is_compatible_with().
  WarpMesh.core_type: WarpMesh#core_type().
  WarpMesh.tracing_context: WarpMesh#tracing_context().
---
# Module: [`jax/_src/pallas/mosaic_gpu/core.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py)

## Classes
### `AbstractRefUnion`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:591`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L591)
- members:
  - `collective(self)` — [`L624`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L624)
  - `layout(self)` — [`L618`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L618)
  - `update(self, inner_aval=None, memory_space=None, kind=None)` — [`L613`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L613)
  - `refs` — [`L592`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L592)
- protocol/private: `__eq__`[`L632`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L632), `__hash__`[`L640`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L640), `__init__`[`L594`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L594), `_getitem`[`L606`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L606), `_iter`[`L603`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L603), `_setitem`[`L609`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L609)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../../state/types.md#AbstractRef), [`TransformedRef`](../../state/types.md#TransformedRef), [`transforms`](../../state/types.md#TransformedRef.transforms), [`flatten_ref_union`](core.md#flatten_ref_union), [`update`](../../state/types.md#AbstractRef.update), [`__init__`](../../state/types.md#AbstractRef.__init__), [`TMEM`](core.md#TMEM), [`_GPUMemoryRefTree`](core.md#_GPUMemoryRefTree)
- used by: [`_run_scoped_resource_estimator`](lowering.md#_run_scoped_resource_estimator), [`flatten_ref_union`](core.md#flatten_ref_union), [`update`](../../state/types.md#AbstractRef.update), [`get_ref_aval`](core.md#RefUnion.get_ref_aval), [`_getitem`](../../state/types.md#AbstractRef._getitem), [`_setitem`](../../state/types.md#AbstractRef._setitem)

### `AbstractTMEMRef`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1616`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1616)
- members:
  - `update(self, inner_aval=None, memory_space=None, kind=None)` — [`L1627`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1627)
- protocol/private: `__init__`[`L1619`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1619), `__repr__`[`L1624`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1624), `__slots__`[`L1617`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1617)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../../state/types.md#AbstractRef), [`update`](../../state/types.md#AbstractRef.update), [`__init__`](../../state/types.md#AbstractRef.__init__)
- used by: [`get_ref_aval`](core.md#GPUMemoryRef.get_ref_aval), [`ref_for_aval`](lowering.md#lower_pipelined_jaxpr_to_module.ref_for_aval), [`update`](../../state/types.md#AbstractRef.update), [`_tcgen05_mma_abstract_eval`](primitives.md#_tcgen05_mma_abstract_eval)

### `Barrier`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1494`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1494)
- doc: Describes a barrier reference.
- signature: `class Barrier:`
- members:
  - `get_array_aval(self)` — [`L1519`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1519)
  - `get_ref_aval(self)` — [`L1522`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1522)
  - `num_arrivals` — [`L1506`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1506)
  - `num_barriers` — [`L1507`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1507)
  - `orders_tensor_core` — [`L1508`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1508)
- protocol/private: `__post_init__`[`L1510`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1510)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../../state/types.md#AbstractRef), [`ShapedArray`](../../core.md#ShapedArray), [`SMEM`](core.md#SMEM), [`BarrierType`](core.md#BarrierType)
- used by: [`matmul_kernel`](../../../experimental/pallas/ops/gpu/blackwell_matmul_mgpu.md#matmul_kernel), [`_attention_bwd`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_bwd), [`attention_with_pipeline_emitter`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#attention_with_pipeline_emitter), [`pipeline`](pipeline.md#emit_pipeline.pipeline), [`decorator`](helpers.md#dynamic_scheduling_loop.decorator), [`entry`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_forward.entry), [`_get_scoped_allocs`](pipeline.md#emit_pipeline_warp_specialized._get_scoped_allocs), [`_scoped`](../../../experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.md#ragged_dot_kernel.kernel._scoped), [`_pipeline_scope`](../../../experimental/pallas/ops/gpu/hopper_matmul_mgpu.md#kernel._pipeline_scope)

### `BarrierType`  ·  implements/extends ExtendedDType
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1468`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1468)
- signature: `class BarrierType(dtypes.ExtendedDType):`
- members:
  - `name` — [`L1470`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1470)
  - `num_arrivals` — [`L1472`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1472)
  - `orders_tensor_core` — [`L1473`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1473)
- protocol/private: `__str__`[`L1475`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1475)
- uses (calls/refs, reference-scoped): [`ExtendedDType`](../../dtypes.md#ExtendedDType), [`barrier_dtype`](core.md#barrier_dtype)
- used by: [`_run_scoped_lowering_rule`](lowering.md#_run_scoped_lowering_rule), [`_allocate_for_aval`](interpret/jaxpr_interpret.md#JaxprInterpreter._allocate_for_aval), [`_run_scoped_resource_estimator`](lowering.md#_run_scoped_resource_estimator), [`ExtendedDType`](../../dtypes.md#ExtendedDType), [`_deallocate_for_aval`](interpret/jaxpr_interpret.md#JaxprInterpreter._deallocate_for_aval), [`_get_barrier`](lowering.md#_get_barrier), [`get_ref_aval`](core.md#Barrier.get_ref_aval)

### `BlockSpec`  ·  implements/extends BlockSpec
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1400`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1400)
- doc: A GPU-specific `BlockSpec`.
- signature: `class BlockSpec(pallas_core.BlockSpec):`
- members:
  - `to_block_mapping(self, origin: pallas_core.OriginStr, array_aval: jax_core.ShapedArray, *, index_map_avals: Sequence[jax_core.AbstractValue], index_map_tree: tree_util.PyTreeDef, grid: pallas_core.GridMappingGrid, vmapped_dims: tuple[int, ...], allow_captured_consts: bool = False, debug: bool = False)` — [`L1419`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1419)
  - `collective_axes` — [`L1417`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1417)
  - `delay_release` — [`L1416`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1416)
  - `transforms` — [`L1415`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1415)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../../state/types.md#AbstractRef), [`ShapedArray`](../../core.md#ShapedArray), [`AbstractValue`](../../core.md#AbstractValue), [`BlockSpec`](../core.md#BlockSpec), [`to_block_mapping`](../core.md#BlockSpec.to_block_mapping), [`Transform`](../../state/types.md#Transform), [`PyTreeDef`](../../tree_util.md#PyTreeDef), [`transform_type`](../../state/types.md#Transform.transform_type), [`update`](../../state/types.md#AbstractRef.update), [`BlockMapping`](../core.md#BlockMapping), [`block_aval`](../core.md#BlockMapping.block_aval), [`GridMappingGrid`](../core.md#GridMappingGrid), [`replace`](../core.md#BlockMapping.replace), [`OriginStr`](../core.md#OriginStr)
- used by: [`BlockSpec`](../core.md#BlockSpec), [`to_block_mapping`](../core.md#BlockSpec.to_block_mapping), [`emit_pipeline_warp_specialized`](pipeline.md#emit_pipeline_warp_specialized), [`_block_spec_from_block_mapping`](lowering.md#_block_spec_from_block_mapping), [`emit_pipeline`](pipeline.md#emit_pipeline), [`kernel_dq`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_bwd.kernel_dq), [`get_pipeline`](../../../experimental/pallas/ops/gpu/hopper_matmul_mgpu.md#kernel.get_pipeline), [`get_pipeline`](../../../experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.md#mixed_matmul_kernel.kernel.get_pipeline), [`fa3_kernel`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#attention_with_pipeline_emitter.fa3_kernel), [`kernel_dkv`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_bwd.kernel_dkv), [`_downcast_spec`](pipeline.md#_downcast_spec), [`acc_scope`](../../../experimental/pallas/ops/gpu/ragged_dot_mgpu.md#ragged_dot.body.mn_loop.acc_scope), [`acc_scope`](../../../experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.md#transposed_ragged_dot.body.mn_loop.acc_scope)

### `ClusterBarrier`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1528`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1528)
- signature: `class ClusterBarrier:`
- members:
  - `get_array_aval(self)` — [`L1544`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1544)
  - `get_ref_aval(self)` — [`L1547`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1547)
  - `collective_axes` — [`L1529`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1529)
  - `leader_tracked` — [`L1533`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1533)
  - `num_arrivals` — [`L1531`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1531)
  - `num_barriers` — [`L1530`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1530)
  - `orders_tensor_core` — [`L1532`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1532)
- protocol/private: `__post_init__`[`L1535`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1535)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../../state/types.md#AbstractRef), [`ShapedArray`](../../core.md#ShapedArray), [`SMEM`](core.md#SMEM), [`ClusterBarrierType`](core.md#ClusterBarrierType), [`collective_axes`](core.md#ClusterBarrierType.collective_axes), [`leader_tracked`](core.md#ClusterBarrierType.leader_tracked), [`num_arrivals`](core.md#ClusterBarrierType.num_arrivals), [`orders_tensor_core`](core.md#ClusterBarrierType.orders_tensor_core)
- used by: [`matmul_kernel`](../../../experimental/pallas/ops/gpu/blackwell_matmul_mgpu.md#matmul_kernel), [`decorator`](helpers.md#dynamic_scheduling_loop.decorator), [`_get_scoped_allocs`](pipeline.md#emit_pipeline_warp_specialized._get_scoped_allocs)

### `ClusterBarrierType`  ·  implements/extends ExtendedDType
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1480`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1480)
- signature: `class ClusterBarrierType(dtypes.ExtendedDType):`
- members:
  - `collective_axes` — [`L1484`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1484)
  - `leader_tracked` — [`L1487`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1487)
  - `name` — [`L1482`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1482)
  - `num_arrivals` — [`L1485`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1485)
  - `orders_tensor_core` — [`L1486`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1486)
- protocol/private: `__str__`[`L1489`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1489)
- uses (calls/refs, reference-scoped): [`ExtendedDType`](../../dtypes.md#ExtendedDType), [`barrier_dtype`](core.md#barrier_dtype)
- used by: [`_run_scoped_lowering_rule`](lowering.md#_run_scoped_lowering_rule), [`_allocate_for_aval`](interpret/jaxpr_interpret.md#JaxprInterpreter._allocate_for_aval), [`_run_scoped_resource_estimator`](lowering.md#_run_scoped_resource_estimator), [`ExtendedDType`](../../dtypes.md#ExtendedDType), [`_get_cluster_barrier`](lowering.md#_get_cluster_barrier), [`get_ref_aval`](core.md#ClusterBarrier.get_ref_aval), [`_deallocate_for_aval`](interpret/jaxpr_interpret.md#JaxprInterpreter._deallocate_for_aval)

### `ClusterRefTransform`  ·  implements/extends Transform
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1060`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1060)
- signature: `class ClusterRefTransform(state_types.Transform):`
- members:
  - `commute_ndindexer(self, _: jax_core.AbstractValue, indexer: indexing.NDIndexer)` — [`L1076`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1076)
  - `transform_type(self, x)` — [`L1070`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1070)
  - `undo(self, x: jax_core.AbstractValue)` — [`L1073`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1073)
  - `dims` — [`L1061`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1061)
  - `idxs` — [`L1064`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1064)
- protocol/private: `__post_init__`[`L1066`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1066)
- uses (calls/refs, reference-scoped): [`AbstractValue`](../../core.md#AbstractValue), [`Transform`](../../state/types.md#Transform), [`NDIndexer`](../../state/indexing.md#NDIndexer), [`register_dataclass`](../../tree_util.md#register_dataclass), [`AxisName`](../../core.md#AxisName)
- used by: [`_handle_transforms`](lowering.md#_handle_transforms), [`Transform`](../../state/types.md#Transform), [`_commute_transform`](lowering.md#_commute_transform), [`transform_type`](../../state/types.md#Transform.transform_type), [`undo`](../../state/types.md#Transform.undo), [`_bubble_up_transforms_for_lowering`](lowering.md#_bubble_up_transforms_for_lowering), [`cluster_ref`](core.md#cluster_ref)

### `CollapseLeadingBatchDimensionsTransform`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1284`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1284)
- members:
  - `transform_type(self, x: jax_core.AbstractValue)` — [`L1291`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1291)
  - `undo(self, x: jax_core.AbstractValue)` — [`L1306`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1306)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../../state/types.md#AbstractRef), [`ShapedArray`](../../core.md#ShapedArray), [`AbstractValue`](../../core.md#AbstractValue), [`shape`](../../core.md#ShapedArray.shape), [`Transform`](../../state/types.md#Transform), [`update`](../../core.md#ShapedArray.update), [`ndim`](../../core.md#ShapedArray.ndim), [`register_dataclass`](../../tree_util.md#register_dataclass), [`update`](../../state/types.md#AbstractRef.update), [`ExpandLeadingBatchDimensionsTransform`](core.md#ExpandLeadingBatchDimensionsTransform)
- used by: [`transform_type`](../../state/types.md#Transform.transform_type), [`undo`](../../state/types.md#Transform.undo), [`__call__`](core.md#MemorySpace.__call__)

### `CompilerParams`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:90`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L90)
- doc: Mosaic GPU compiler parameters.
- signature: `class CompilerParams:`
- members:
  - `approx_math` — [`L127`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L127)
  - `dimension_semantics` — [`L128`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L128)
  - `lowering_semantics` — [`L135`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L135)
  - `max_concurrent_steps` — [`L129`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L129)
  - `profile_dir` — [`L133`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L133)
  - `profile_space` — [`L132`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L132)
  - `profile_trace_scope` — [`L134`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L134)
  - `reduction_scratch_bytes` — [`L131`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L131)
  - `replace` — [`L147`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L147)
  - `unsafe_no_auto_barriers` — [`L130`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L130)
- protocol/private: `__post_init__`[`L137`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L137)
- uses (calls/refs, reference-scoped): [`LoweringSemantics`](../../../experimental/mosaic/gpu/core.md#LoweringSemantics), [`Lane`](../../../experimental/mosaic/gpu/core.md#LoweringSemantics.Lane), [`TraceScope`](core.md#TraceScope), [`WARPGROUP`](core.md#TraceScope.WARPGROUP), [`DimensionSemantics`](core.md#DimensionSemantics)
- used by: [`lower_jaxpr_to_module`](lowering.md#lower_jaxpr_to_module), [`body`](lowering.md#lower_jaxpr_to_module.body), [`lower_pipelined_jaxpr_to_module`](lowering.md#lower_pipelined_jaxpr_to_module), [`mixed_matmul_kernel`](../../../experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.md#mixed_matmul_kernel), [`matmul_kernel`](../../../experimental/pallas/ops/gpu/blackwell_matmul_mgpu.md#matmul_kernel), [`all_gather_lhs_matmul`](../../../experimental/pallas/ops/gpu/collective_matmul_mgpu.md#all_gather_lhs_matmul), [`pallas_call_lowering`](pallas_call_registration.md#pallas_call_lowering), [`_attention_bwd`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_bwd), [`attention_with_pipeline_emitter`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#attention_with_pipeline_emitter), [`ragged_dot_kernel`](../../../experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.md#ragged_dot_kernel), [`transposed_ragged_dot`](../../../experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.md#transposed_ragged_dot), [`all_gather`](../../../experimental/pallas/ops/gpu/all_gather_mgpu.md#all_gather), [`reduce_scatter`](../../../experimental/pallas/ops/gpu/reduce_scatter_mgpu.md#reduce_scatter), [`_attention_forward`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_forward), [`matmul`](../../../experimental/pallas/ops/gpu/hopper_matmul_mgpu.md#matmul), [`gpu_lowering`](../pallas_call.md#_pallas_call_lowering.gpu_lowering), [`ragged_dot`](../../../experimental/pallas/ops/gpu/ragged_dot_mgpu.md#ragged_dot), [`_gpu_mesh_discharge_rule`](core.md#_gpu_mesh_discharge_rule), [`scoped_pipeline_fn`](lowering.md#lower_pipelined_jaxpr_to_module.scoped_pipeline_fn)

### `ExpandLeadingBatchDimensionsTransform`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1313`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1313)
- members:
  - `commute_ndindexer(self, aval: jax_core.AbstractValue, indexer: indexing.NDIndexer)` — [`L1345`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1345)
  - `transform_type(self, x: jax_core.AbstractValue)` — [`L1322`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1322)
  - `batch_shape` — [`L1320`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1320)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../../state/types.md#AbstractRef), [`ShapedArray`](../../core.md#ShapedArray), [`AbstractValue`](../../core.md#AbstractValue), [`shape`](../../core.md#ShapedArray.shape), [`Transform`](../../state/types.md#Transform), [`NDIndexer`](../../state/indexing.md#NDIndexer), [`Slice`](../../indexing.md#Slice), [`indices`](../../state/indexing.md#NDIndexer.indices), [`start`](../../indexing.md#Slice.start), [`update`](../../core.md#ShapedArray.update), [`size`](../../indexing.md#Slice.size), [`ndim`](../../core.md#ShapedArray.ndim), [`register_dataclass`](../../tree_util.md#register_dataclass), [`from_indices_shape`](../../state/indexing.md#NDIndexer.from_indices_shape), [`stride`](../../indexing.md#Slice.stride), [`update`](../../state/types.md#AbstractRef.update), [`shape`](../../state/indexing.md#NDIndexer.shape), [`IdentityTransform`](core.md#IdentityTransform), [`IntIndexer`](../../state/indexing.md#IntIndexer)
- used by: [`_commute_transform`](lowering.md#_commute_transform), [`transform_type`](../../state/types.md#Transform.transform_type), [`_async_store_tmem_lowering_rule_wg`](primitives.md#_async_store_tmem_lowering_rule_wg), [`_async_store_tmem_lowering_rule`](primitives.md#_async_store_tmem_lowering_rule), [`undo`](core.md#CollapseLeadingBatchDimensionsTransform.undo)

### `ExtractAliasedRef`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1145`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1145)
- members:
  - `from_transformed_ref(cls, ref: pallas_core.TransformedRef, byte_offset: int, alias_group_idx: int, layout: tcgen05.TMEMLayout | None = None)` — [`L1160`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1160)
  - `transform_type(self, x)` — [`L1169`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1169)
  - `alias_group_idx` — [`L1152`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1152)
  - `dtype` — [`L1147`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1147)
  - `layout` — [`L1155`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1155)
  - `offset` — [`L1149`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1149)
  - `shape` — [`L1148`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1148)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../../state/types.md#AbstractRef), [`ShapedArray`](../../core.md#ShapedArray), [`TransformedRef`](../../state/types.md#TransformedRef), [`Transform`](../../state/types.md#Transform), [`DType`](../../typing.md#DType), [`dtype`](../../dtypes.md#dtype), [`ref`](../../state/types.md#TransformedRef.ref), [`update`](../../core.md#ShapedArray.update), [`register_dataclass`](../../tree_util.md#register_dataclass), [`update`](../../state/types.md#AbstractRef.update), [`dtype`](../../state/types.md#TransformedRef.dtype)
- used by: [`transform_type`](../../state/types.md#Transform.transform_type), [`flatten_ref_union`](core.md#flatten_ref_union), [`unflatten`](core.md#flatten_ref_union.unflatten)

### `GPUMemoryRef`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:437`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L437)
- members:
  - `get_ref_aval(self)` — [`L448`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L448) — documented in [jax-_src-pallas-mosaic_gpu-core](../../../../../concepts/jax-_src-pallas-mosaic_gpu-core.md)
  - `collective` — [`L441`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L441)
  - `layout` — [`L440`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L440)
  - `transforms` — [`L438`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L438)
- protocol/private: `__post_init__`[`L443`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L443)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../../state/types.md#AbstractRef), [`ShapedArray`](../../core.md#ShapedArray), [`TransformedRef`](../../state/types.md#TransformedRef), [`Transform`](../../state/types.md#Transform), [`transforms`](../../state/types.md#TransformedRef.transforms), [`ref`](../../state/types.md#TransformedRef.ref), [`MemorySpace`](core.md#MemorySpace), [`transform_type`](../../state/types.md#transform_type), [`MemoryRef`](../core.md#MemoryRef), [`AbstractTMEMRef`](core.md#AbstractTMEMRef), [`memory_space`](../core.md#MemoryRef.memory_space), [`undo_transforms`](../core.md#undo_transforms), [`get_ref_aval`](core.md#RefUnion.get_ref_aval), [`TMEM`](core.md#MemorySpace.TMEM), [`_Ref`](core.md#_Ref), [`dtype`](../core.md#MemoryRef.dtype), [`shape`](../core.md#MemoryRef.shape), [`RefUnion`](core.md#RefUnion)
- used by: [`ref_for_aval`](lowering.md#lower_pipelined_jaxpr_to_module.ref_for_aval), [`__call__`](core.md#MemorySpace.__call__), [`get_ref_aval`](../core.md#MemoryRef.get_ref_aval), [`RefUnion`](core.md#RefUnion)

### `IdentityTransform`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1389`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1389)
- members:
  - `transform_type(self, x: jax_core.AbstractValue)` — [`L1392`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1392)
  - `undo(self, x: jax_core.AbstractValue)` — [`L1395`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1395)
- uses (calls/refs, reference-scoped): [`AbstractValue`](../../core.md#AbstractValue), [`Transform`](../../state/types.md#Transform), [`register_dataclass`](../../tree_util.md#register_dataclass)
- used by: [`transform_type`](../../state/types.md#Transform.transform_type), [`undo`](../../state/types.md#Transform.undo), [`commute_ndindexer`](core.md#ExpandLeadingBatchDimensionsTransform.commute_ndindexer), [`_bubble_up_transform`](lowering.md#_bubble_up_transform)

### `Layout`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1895`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1895)
- members:
  - `check_no_args()` — [`L1925`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1925)
  - `to_mgpu(self, *args, **kwargs)` — [`L1924`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1924)
  - `SMEM_GMEM_COPY` — [`L1914`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1914)
  - `TCGEN05` — [`L1908`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1908)
  - `TCGEN05_M64_COLLECTIVE` — [`L1910`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1910)
  - `TCGEN05_M64_COLLECTIVE_NATIVE` — [`L1912`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1912)
  - `TCGEN05_TMEM_NATIVE` — [`L1911`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1911)
  - `TCGEN05_TRANSPOSED` — [`L1909`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1909)
  - `TILED` — [`L1906`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1906)
  - `TMA_INDICES` — [`L1915`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1915)
  - `TMA_INDICES_4` — [`L1916`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1916)
  - `WGMMA` — [`L1897`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1897)
  - `WGMMA_8BIT` — [`L1898`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1898)
  - `WGMMA_TRANSPOSED` — [`L1901`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1901)
  - `WGMMA_UPCAST_2X` — [`L1899`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1899)
  - `WGMMA_UPCAST_4X` — [`L1900`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1900)
  - `WG_SPLAT` — [`L1903`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1903)
  - `WG_STRIDED` — [`L1904`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1904)
- protocol/private: `_WGMMA_ACC_32BIT`[`L1919`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1919), `__call__`[`L1921`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1921)
- uses (calls/refs, reference-scoped): [`TiledLayout`](../../../experimental/mosaic/gpu/fragmented_array.md#TiledLayout), [`itemsize_bits`](../../dtypes.md#itemsize_bits), [`WGSplatFragLayout`](../../../experimental/mosaic/gpu/fragmented_array.md#WGSplatFragLayout), [`WGStridedFragLayout`](../../../experimental/mosaic/gpu/fragmented_array.md#WGStridedFragLayout), [`WGMMA_LAYOUT`](../../../experimental/mosaic/gpu/fragmented_array.md#WGMMA_LAYOUT), [`FragmentedLayout`](../../../experimental/mosaic/gpu/fragmented_array.md#FragmentedLayout.FragmentedLayout), [`tiled_copy_smem_gmem_layout`](../../../experimental/mosaic/gpu/fragmented_array.md#tiled_copy_smem_gmem_layout), [`TCGEN05_LAYOUT`](../../../experimental/mosaic/gpu/fragmented_array.md#TCGEN05_LAYOUT), [`WGMMA_TRANSPOSED_LAYOUT`](../../../experimental/mosaic/gpu/fragmented_array.md#WGMMA_TRANSPOSED_LAYOUT), [`TCGEN05_TRANSPOSED_LAYOUT`](../../../experimental/mosaic/gpu/fragmented_array.md#TCGEN05_TRANSPOSED_LAYOUT), [`TMA_INDICES_4_LAYOUT`](../../../experimental/mosaic/gpu/fragmented_array.md#TMA_INDICES_4_LAYOUT), [`TMA_INDICES_LAYOUT`](../../../experimental/mosaic/gpu/fragmented_array.md#TMA_INDICES_LAYOUT), [`WGMMA_LAYOUT_8BIT`](../../../experimental/mosaic/gpu/fragmented_array.md#WGMMA_LAYOUT_8BIT), [`WGMMA_LAYOUT_ACC_32BIT`](../../../experimental/mosaic/gpu/fragmented_array.md#WGMMA_LAYOUT_ACC_32BIT), [`WGMMA_LAYOUT_UPCAST_2X`](../../../experimental/mosaic/gpu/fragmented_array.md#WGMMA_LAYOUT_UPCAST_2X), [`tmem_native_layout`](../../../experimental/mosaic/gpu/fragmented_array.md#tmem_native_layout), [`WGMMA_LAYOUT_UPCAST_4X`](../../../experimental/mosaic/gpu/fragmented_array.md#WGMMA_LAYOUT_UPCAST_4X), [`SomeLayout`](core.md#SomeLayout), [`ParameterizedLayout`](core.md#ParameterizedLayout), [`TMEM_NATIVE_LAYOUT`](../../../experimental/mosaic/gpu/fragmented_array.md#TMEM_NATIVE_LAYOUT), [`tile_shape`](../../../experimental/mosaic/gpu/utils.md#tile_shape)
- used by: [`do_matmul`](../../../experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.md#do_matmul), [`_compute_wg`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_forward.kernel._compute_wg), [`kv_loop`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_forward.kernel._compute_wg.kv_loop), [`_compute_thread`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_bwd.kernel_dq._compute_thread), [`_compute_thread`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_bwd.kernel_dkv._compute_thread), [`q_pipeline`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_bwd.kernel_dkv.q_pipeline), [`block_matmul`](../../../experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.md#transposed_ragged_dot.body.mn_loop.acc_scope.block_matmul), [`_transfer_loop`](../../../experimental/pallas/ops/gpu/all_gather_mgpu.md#all_gather.kernel._transfer_loop), [`_transfer_loop`](../../../experimental/pallas/ops/gpu/reduce_scatter_mgpu.md#reduce_scatter.kernel._transfer_loop), [`to_mgpu`](core.md#SomeLayout.to_mgpu), [`to_mgpu`](core.md#ParameterizedLayout.to_mgpu), [`layout_cls`](core.md#ParameterizedLayout.layout_cls)

### `MemoryEffect`  ·  implements/extends Effect
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1804`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1804)
- signature: `class MemoryEffect(jax_core.Effect):`
- uses (calls/refs, reference-scoped): [`Effect`](../../core.md#Effect)
- used by: [`_memory_effect`](core.md#_memory_effect)

### `MemorySpace`  ·  implements/extends Enum
- def: [`jax/_src/pallas/mosaic_gpu/core.py:150`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L150) — documented in [jax-_src-pallas-mosaic_gpu-core](../../../../../concepts/jax-_src-pallas-mosaic_gpu-core.md)
- signature: `class MemorySpace(enum.Enum):`
- members:
  - `like(self, shape_dtype_like)` — [`L206`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L206)
  - `GMEM` — [`L152`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L152)
  - `REGS` — [`L158`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L158)
  - `SMEM` — [`L154`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L154)
  - `TMEM` — [`L156`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L156)
- protocol/private: `__call__`[`L163`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L163), `__str__`[`L160`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L160)
- uses (calls/refs, reference-scoped): [`ShapedArray`](../../core.md#ShapedArray), [`DTypeLike`](../../typing.md#DTypeLike), [`Transform`](../../state/types.md#Transform), [`itemsize_bits`](../../dtypes.md#itemsize_bits), [`MemoryRef`](../core.md#MemoryRef), [`memory_space`](../core.md#MemoryRef.memory_space), [`to_mgpu`](core.md#TMEMLayout.to_mgpu), [`GPUMemoryRef`](core.md#GPUMemoryRef), [`TMEMLayout`](core.md#TMEMLayout), [`transforms`](core.md#GPUMemoryRef.transforms), [`CollapseLeadingBatchDimensionsTransform`](core.md#CollapseLeadingBatchDimensionsTransform), [`infer_tmem_layout`](core.md#infer_tmem_layout), [`collective`](core.md#GPUMemoryRef.collective), [`layout`](core.md#GPUMemoryRef.layout)
- used by: [`_allocate_buffers_for_outputs`](interpret/interpret_pallas_call.md#_allocate_buffers_for_outputs), [`_allocate_buffer`](interpret/gpu_callbacks.md#_allocate_buffer), [`_allocate_buffer_for_all_threads`](interpret/gpu_callbacks.md#_allocate_buffer_for_all_threads), [`get_ref_aval`](core.md#GPUMemoryRef.get_ref_aval), [`SMEM`](core.md#SMEM), [`SMEM`](../../../experimental/pallas/mosaic_gpu.md#SMEM), [`_interpret_run_scoped_p`](interpret/jaxpr_interpret.md#JaxprInterpreter._interpret_run_scoped_p), [`_check_ref`](primitives.md#_check_ref), [`_allocate_buffers_for_inputs`](interpret/interpret_pallas_call.md#_allocate_buffers_for_inputs), [`get_ref_aval`](core.md#WGMMAAccumulatorRef.get_ref_aval), [`_wgmma_ref_type_mapping`](core.md#_wgmma_ref_type_mapping), [`__call__`](core.md#SemaphoreType.__call__), [`_async_copy_smem_to_tmem_abstract_eval`](primitives.md#_async_copy_smem_to_tmem_abstract_eval), [`_async_copy_to_tmem_abstract_eval`](primitives.md#_async_copy_to_tmem_abstract_eval), [`GMEM`](core.md#GMEM), [`transpose_ref`](core.md#transpose_ref), [`get_memory_space_idx`](interpret/gpu_callbacks.md#get_memory_space_idx), [`TMEM`](core.md#TMEM), [`_raise_if_unsupported_memory_space`](interpret/jaxpr_interpret.md#_raise_if_unsupported_memory_space), [`get_uninitialized_array`](interpret/jaxpr_interpret.md#get_uninitialized_array), [`__post_init__`](core.md#GPUMemoryRef.__post_init__), [`_async_load_tmem_abstract_eval`](primitives.md#_async_load_tmem_abstract_eval), [`_async_store_tmem_abstract_eval`](primitives.md#_async_store_tmem_abstract_eval), [`IDX_BY_GPU_MEMORY_SPACE`](interpret/gpu_callbacks.md#IDX_BY_GPU_MEMORY_SPACE.IDX_BY_GPU_MEMORY_SPACE), [`supported_memory_spaces`](core.md#WarpMesh.supported_memory_spaces), [`is_gmem_memory_space`](interpret/gpu_callbacks.md#is_gmem_memory_space), [`TMEM`](../../../experimental/pallas/mosaic_gpu.md#TMEM), [`GPU_MEMORY_SPACE_BY_IDX`](interpret/gpu_callbacks.md#GPU_MEMORY_SPACE_BY_IDX.GPU_MEMORY_SPACE_BY_IDX), [`REGS`](core.md#REGS), [`GMEM`](../../../experimental/pallas/mosaic_gpu.md#GMEM), [`REGS`](../../../experimental/pallas/mosaic_gpu.md#REGS), [`default_memory_space`](core.md#Mesh.default_memory_space), [`default_memory_space`](core.md#WarpMesh.default_memory_space), [`supported_memory_spaces`](core.md#Mesh.supported_memory_spaces)

### `Mesh`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1637`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1637)
- members:
  - `check_is_compatible_with(self, other_mesh)` — [`L1697`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1697)
  - `core_type(self)` — [`L1701`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1701)
  - `default_memory_space(self)` — [`L1676`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1676)
  - `discharges_effect(self, effect: jax_core.Effect)` — [`L1694`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1694)
  - `shape(self)` — [`L1680`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1680)
  - `supported_memory_spaces(self)` — [`L1705`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1705)
  - `tracing_context(self)` — [`L1709`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1709)
  - `cluster` — [`L1640`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1640)
  - `cluster_names` — [`L1641`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1641)
  - `grid` — [`L1638`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1638)
  - `grid_names` — [`L1639`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1639)
  - `kernel_name` — [`L1645`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1645)
  - `num_threads` — [`L1643`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1643)
  - `thread_name` — [`L1644`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1644)
- protocol/private: `__post_init__`[`L1647`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1647)
- uses (calls/refs, reference-scoped): [`MemorySpace`](core.md#MemorySpace), [`Mesh`](../core.md#Mesh), [`tracing_grid_env`](../core.md#tracing_grid_env), [`Effect`](../../core.md#Effect), [`_memory_effect`](core.md#_memory_effect), [`_wgmma_pipeline_effect`](core.md#_wgmma_pipeline_effect), [`GMEM`](core.md#MemorySpace.GMEM)
- used by: [`interpret_pallas_call`](interpret/interpret_pallas_call.md#interpret_pallas_call), [`lower_pipelined_jaxpr_to_module`](lowering.md#lower_pipelined_jaxpr_to_module), [`pallas_call_lowering`](pallas_call_registration.md#pallas_call_lowering), [`kernel`](core.md#kernel), [`core_type`](../core.md#Mesh.core_type), [`_interpret_axis_index_p`](interpret/jaxpr_interpret.md#JaxprInterpreter._interpret_axis_index_p), [`_get_grid_and_cluster_dims_and_num_threads`](interpret/interpret_pallas_call.md#_get_grid_and_cluster_dims_and_num_threads), [`mesh`](interpret/jaxpr_interpret.md#JaxprInterpreter.mesh), [`shape`](../core.md#Mesh.shape), [`check_is_compatible_with`](../core.md#Mesh.check_is_compatible_with), [`default_memory_space`](../core.md#Mesh.default_memory_space), [`discharges_effect`](../core.md#Mesh.discharges_effect), [`wrapper`](core.md#kernel.wrapper), [`supported_memory_spaces`](../core.md#Mesh.supported_memory_spaces), [`tracing_context`](../core.md#Mesh.tracing_context), [`are_thread_cluster_axes_collective`](interpret/jaxpr_interpret.md#JaxprInterpreter.are_thread_cluster_axes_collective), [`_gpu_mesh_discharge_rule`](core.md#_gpu_mesh_discharge_rule), [`num_threads_per_block`](interpret/jaxpr_interpret.md#JaxprInterpreter.num_threads_per_block), [`_raise_if_unsupported_collective_axes`](interpret/jaxpr_interpret.md#_raise_if_unsupported_collective_axes)

### `MulticastRef`  ·  implements/extends Transform
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1024`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1024)
- signature: `class MulticastRef(state_types.Transform):`
- members:
  - `commute_ndindexer(self, _: jax_core.AbstractValue, indexer: indexing.NDIndexer)` — [`L1035`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1035)
  - `transform_type(self, x)` — [`L1029`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1029)
  - `undo(self, x: jax_core.AbstractValue)` — [`L1032`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1032)
  - `collective_axes` — [`L1025`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1025)
- uses (calls/refs, reference-scoped): [`AbstractValue`](../../core.md#AbstractValue), [`Transform`](../../state/types.md#Transform), [`NDIndexer`](../../state/indexing.md#NDIndexer), [`register_dataclass`](../../tree_util.md#register_dataclass)
- used by: [`_handle_transforms`](lowering.md#_handle_transforms), [`Transform`](../../state/types.md#Transform), [`_commute_transform`](lowering.md#_commute_transform), [`transform_type`](../../state/types.md#Transform.transform_type), [`undo`](../../state/types.md#Transform.undo), [`_extract_gmem_copy_params`](primitives.md#_extract_gmem_copy_params), [`_bubble_up_transforms_for_lowering`](lowering.md#_bubble_up_transforms_for_lowering), [`remote_ref`](core.md#remote_ref), [`multicast_ref`](core.md#multicast_ref)

### `ParameterizedLayout`  ·  implements/extends SomeLayout
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1866`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1866)
- signature: `class ParameterizedLayout(SomeLayout):`
- members:
  - `to_mgpu(self, *args, **kwargs)` — [`L1875`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1875)
  - `args` — [`L1868`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1868)
  - `kwargs` — [`L1869`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1869)
  - `layout_cls` — [`L1867`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1867)
- protocol/private: `__post_init__`[`L1871`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1871)
- uses (calls/refs, reference-scoped): [`to_mgpu`](core.md#Layout.to_mgpu), [`Layout`](core.md#Layout), [`FrozenDict`](../../frozen_dict.md#FrozenDict), [`FragmentedLayout`](../../../experimental/mosaic/gpu/fragmented_array.md#FragmentedLayout.FragmentedLayout), [`SomeLayout`](core.md#SomeLayout), [`TMEMLayout`](core.md#TMEMLayout)
- used by: [`to_mgpu`](core.md#SomeLayout.to_mgpu), [`SomeLayout`](core.md#SomeLayout), [`__call__`](core.md#Layout.__call__), [`__call__`](core.md#TMEMLayout.__call__)

### `PdlEffect`  ·  implements/extends Effect
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1822`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1822)
- doc: Indicates that a kernel uses Programmatic Dependency Launch (PDL).
- signature: `class PdlEffect(jax_core.Effect):`
- uses (calls/refs, reference-scoped): [`Effect`](../../core.md#Effect)
- used by: [`_pdl_effect`](core.md#_pdl_effect)

### `PeerMemRef`  ·  implements/extends Transform
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1004`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1004)
- signature: `class PeerMemRef(state_types.Transform):`
- members:
  - `commute_ndindexer(self, _: jax_core.AbstractValue, indexer: indexing.NDIndexer)` — [`L1016`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1016)
  - `transform_type(self, x)` — [`L1013`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1013)
  - `undo(self, x: jax_core.AbstractValue)` — [`L1010`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1010)
  - `device_id` — [`L1005`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1005)
  - `device_id_type` — [`L1006`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1006)
- uses (calls/refs, reference-scoped): [`AbstractValue`](../../core.md#AbstractValue), [`Transform`](../../state/types.md#Transform), [`NDIndexer`](../../state/indexing.md#NDIndexer), [`register_dataclass`](../../tree_util.md#register_dataclass), [`DeviceIdType`](../primitives.md#DeviceIdType)
- used by: [`_handle_transforms`](lowering.md#_handle_transforms), [`Transform`](../../state/types.md#Transform), [`_commute_transform`](lowering.md#_commute_transform), [`transform_type`](../../state/types.md#Transform.transform_type), [`undo`](../../state/types.md#Transform.undo), [`_extract_gmem_copy_params`](primitives.md#_extract_gmem_copy_params), [`_bubble_up_transforms_for_lowering`](lowering.md#_bubble_up_transforms_for_lowering), [`remote_ref`](core.md#remote_ref), [`multicast_ref`](core.md#multicast_ref)

### `PrimitiveSemantics`  ·  implements/extends Enum
- def: [`jax/_src/pallas/mosaic_gpu/core.py:230`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L230)
- doc: Thread semantics for a primitives at the Pallas user-level.
- signature: `class PrimitiveSemantics(enum.Enum):`
- members:
  - `Warp` — [`L233`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L233)
  - `Warpgroup` — [`L234`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L234)
- used by: [`register_lowering_rule`](lowering.md#register_lowering_rule), [`body`](lowering.md#lower_jaxpr_to_module.body), [`_copy_gmem_to_smem_lowering`](primitives.md#_copy_gmem_to_smem_lowering), [`_swap_lowering_rule`](lowering.md#_swap_lowering_rule), [`_get_lowering_rule`](lowering.md#_get_lowering_rule), [`_semaphore_signal_multicast_lowering`](primitives.md#_semaphore_signal_multicast_lowering), [`_copy_smem_to_gmem_lowering`](primitives.md#_copy_smem_to_gmem_lowering), [`_semaphore_wait_lowering_rule`](primitives.md#_semaphore_wait_lowering_rule), [`_axis_index_rule`](lowering.md#_axis_index_rule), [`_core_map_lowering_rule`](lowering.md#_core_map_lowering_rule), [`_mpmd_map_lowering_rule`](lowering.md#_mpmd_map_lowering_rule), [`_barrier_arrive_lowering`](primitives.md#_barrier_arrive_lowering), [`_debug_print_lowering_rule`](lowering.md#_debug_print_lowering_rule), [`_inline_mgpu_flat_transformed_args`](primitives.md#_inline_mgpu_flat_transformed_args), [`_convert_element_type_lowering_rule_wg`](lowering.md#_convert_element_type_lowering_rule_wg), [`_select_n_lowering_rule`](lowering.md#_select_n_lowering_rule), [`LANExWARP_SEMANTICS`](core.md#LANExWARP_SEMANTICS), [`_convert_element_type_lowering_rule`](lowering.md#_convert_element_type_lowering_rule), [`_inline_mgpu_lowering_rule_wg_semantics`](primitives.md#_inline_mgpu_lowering_rule_wg_semantics), [`_swap_lowering_rule_wg`](lowering.md#_swap_lowering_rule_wg), [`_wait_smem_to_gmem_lowering`](primitives.md#_wait_smem_to_gmem_lowering), [`_binary_op_lowering_rule`](lowering.md#_binary_op_lowering_rule), [`WGxWARP_SEMANTICS`](core.md#WGxWARP_SEMANTICS), [`_inline_mgpu_lowering_rule`](primitives.md#_inline_mgpu_lowering_rule), [`_binary_op_lowering_rule_wg`](lowering.md#_binary_op_lowering_rule_wg), [`_comparison_lowering_rule_wg`](lowering.md#_comparison_lowering_rule_wg), [`_get_lowering_rule_wg`](lowering.md#_get_lowering_rule_wg), [`primitive_semantics`](lowering.md#ModuleContext.primitive_semantics), [`single_lane_predicate`](lowering.md#ModuleContext.single_lane_predicate), [`_binary_boolean_op_lowering_rule_wg`](lowering.md#_binary_boolean_op_lowering_rule_wg), [`LANExWG_SEMANTICS`](core.md#LANExWG_SEMANTICS), [`WGxWG_SEMANTICS`](core.md#WGxWG_SEMANTICS)

### `ReducedLayout`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1882`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1882)
- members:
  - `to_mgpu(self, *args, **kwargs)` — [`L1886`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1886)
  - `axes` — [`L1884`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1884)
  - `layout` — [`L1883`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1883)
- uses (calls/refs, reference-scoped): [`TiledLayout`](../../../experimental/mosaic/gpu/fragmented_array.md#TiledLayout), [`FragmentedLayout`](../../../experimental/mosaic/gpu/fragmented_array.md#FragmentedLayout.FragmentedLayout), [`to_mgpu`](core.md#SomeLayout.to_mgpu), [`reduce`](../../../experimental/mosaic/gpu/fragmented_array.md#TiledLayout.reduce), [`SomeLayout`](core.md#SomeLayout)
- used by: [`to_mgpu`](core.md#SomeLayout.to_mgpu), [`reduce`](core.md#SomeLayout.reduce)

### `RefUnion`  ·  implements/extends GPUMemoryRef
- def: [`jax/_src/pallas/mosaic_gpu/core.py:652`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L652)
- doc: A sequence of trees of refs that are allowed to reuse the same memory.
- signature: `class RefUnion(GPUMemoryRef):`
- members:
  - `get_ref_aval(self)` — [`L700`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L700)
  - `refs` — [`L663`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L663)
- protocol/private: `__init__`[`L665`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L665)
- uses (calls/refs, reference-scoped): [`ShapedArray`](../../core.md#ShapedArray), [`int32`](../../../numpy/__init__.pyi.md#int32.int32), [`SMEM`](core.md#SMEM), [`_ref_group_size`](core.md#_ref_group_size), [`TMEM`](core.md#TMEM), [`memory_space`](../core.md#MemoryRef.memory_space), [`_ref_group_tmem_col_size`](core.md#_ref_group_tmem_col_size), [`int8`](../../../numpy/__init__.pyi.md#int8.int8), [`GPUMemoryRef`](core.md#GPUMemoryRef), [`AbstractRefUnion`](core.md#AbstractRefUnion), [`_GPUMemoryRefTree`](core.md#_GPUMemoryRefTree), [`dtype`](../core.md#MemoryRef.dtype), [`shape`](../core.md#MemoryRef.shape)
- used by: [`get_ref_aval`](core.md#GPUMemoryRef.get_ref_aval), [`GPUMemoryRef`](core.md#GPUMemoryRef)

### `SemaphoreType`  ·  implements/extends Enum
- def: [`jax/_src/pallas/mosaic_gpu/core.py:210`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L210)
- signature: `class SemaphoreType(enum.Enum):`
- members:
  - `get_array_aval(self)` — [`L223`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L223)
  - `get_ref_aval(self)` — [`L226`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L226)
  - `BARRIER` — [`L212`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L212)
  - `REGULAR` — [`L211`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L211)
- protocol/private: `__call__`[`L214`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L214)
- uses (calls/refs, reference-scoped): [`ShapedArray`](../../core.md#ShapedArray), [`MemorySpace`](core.md#MemorySpace), [`MemoryRef`](../core.md#MemoryRef), [`get_array_aval`](../core.md#MemoryRef.get_array_aval), [`get_ref_aval`](../core.md#MemoryRef.get_ref_aval), [`_Ref`](core.md#_Ref), [`BarrierSemaphore`](../core.md#BarrierSemaphore), [`GMEM`](core.md#MemorySpace.GMEM), [`Semaphore`](../core.md#Semaphore)
- used by: [`all_gather`](../../../experimental/pallas/ops/gpu/all_gather_mgpu.md#all_gather), [`reduce_scatter`](../../../experimental/pallas/ops/gpu/reduce_scatter_mgpu.md#reduce_scatter), [`kernel_body`](../../../experimental/pallas/ops/gpu/collective_matmul_mgpu.md#all_gather_lhs_matmul.kernel_body)

### `SomeLayout`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1854`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1854)
- members:
  - `reduce(self, axes: int | Sequence[int])` — [`L1856`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1856)
  - `to_mgpu(self, *args, **kwargs)` — [`L1861`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1861)
- uses (calls/refs, reference-scoped): [`to_mgpu`](core.md#Layout.to_mgpu), [`FragmentedLayout`](../../../experimental/mosaic/gpu/fragmented_array.md#FragmentedLayout.FragmentedLayout), [`to_mgpu`](core.md#ReducedLayout.to_mgpu), [`to_mgpu`](core.md#ParameterizedLayout.to_mgpu), [`ParameterizedLayout`](core.md#ParameterizedLayout), [`ReducedLayout`](core.md#ReducedLayout)
- used by: [`_compute_wg`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_forward.kernel._compute_wg), [`_compute_thread`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_bwd.kernel_dq._compute_thread), [`Layout`](core.md#Layout), [`q_pipeline`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_bwd.kernel_dkv.q_pipeline), [`_custom_primitive_in_specs`](primitives.md#_custom_primitive_in_specs), [`layout_cast`](core.md#layout_cast), [`_custom_primitive_op_results`](primitives.md#_custom_primitive_op_results), [`to_mgpu`](core.md#ReducedLayout.to_mgpu), [`SomeLayout`](primitives.md#SomeLayout), [`ParameterizedLayout`](core.md#ParameterizedLayout), [`ReducedLayout`](core.md#ReducedLayout), [`layout`](core.md#ReducedLayout.layout)

### `SwizzleTransform`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1180`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1180)
- members:
  - `transform_type(self, x: jax_core.AbstractValue)` — [`L1190`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1190)
  - `undo(self, x: jax_core.AbstractValue)` — [`L1207`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1207)
  - `swizzle` — [`L1181`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1181)
- protocol/private: `__post_init__`[`L1183`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1183)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../../state/types.md#AbstractRef), [`ShapedArray`](../../core.md#ShapedArray), [`AbstractValue`](../../core.md#AbstractValue), [`shape`](../../core.md#ShapedArray.shape), [`dtype`](../../core.md#ShapedArray.dtype), [`Transform`](../../state/types.md#Transform), [`itemsize_bits`](../../dtypes.md#itemsize_bits), [`UnswizzleRef`](core.md#UnswizzleRef), [`update`](../../state/types.md#AbstractRef.update)
- used by: [`mixed_matmul_kernel`](../../../experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.md#mixed_matmul_kernel), [`matmul_kernel`](../../../experimental/pallas/ops/gpu/blackwell_matmul_mgpu.md#matmul_kernel), [`_attention_bwd`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_bwd), [`ragged_dot_kernel`](../../../experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.md#ragged_dot_kernel), [`transform_type`](../../state/types.md#Transform.transform_type), [`kernel`](../../../experimental/pallas/ops/gpu/hopper_matmul_mgpu.md#kernel), [`transposed_ragged_dot`](../../../experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.md#transposed_ragged_dot), [`undo`](../../state/types.md#Transform.undo), [`entry`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_forward.entry), [`batch_transform`](core.md#batch_transform), [`to_transform_attr`](core.md#to_transform_attr), [`undo`](core.md#UnswizzleRef.undo)

### `TMEMLayout`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1983`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1983)
- members:
  - `to_mgpu(self, *args, **kwargs)` — [`L1994`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1994)
  - `M64_COLLECTIVE_LAYOUT` — [`L1988`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1988)
  - `SCALES_LAYOUT` — [`L1986`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1986)
  - `SCALES_M64_COLLECTIVE_LAYOUT` — [`L1989`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1989)
  - `SPARSE_METADATA_LAYOUT` — [`L1987`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1987)
- protocol/private: `__call__`[`L1991`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1991)
- uses (calls/refs, reference-scoped): [`ParameterizedLayout`](core.md#ParameterizedLayout)
- used by: [`__call__`](core.md#MemorySpace.__call__), [`layout_cls`](core.md#ParameterizedLayout.layout_cls)

### `TilingTransform`  ·  implements/extends Transform
- def: [`jax/_src/pallas/mosaic_gpu/core.py:711`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L711)
- doc: Represents a tiling transformation for memory refs.
- signature: `class TilingTransform(state_types.Transform):`
- members:
  - `transform_type(self, x)` — [`L720`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L720)
  - `undo(self, x: jax_core.AbstractValue)` — [`L737`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L737)
  - `tiling` — [`L718`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L718)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../../state/types.md#AbstractRef), [`ShapedArray`](../../core.md#ShapedArray), [`AbstractValue`](../../core.md#AbstractValue), [`shape`](../../core.md#ShapedArray.shape), [`Transform`](../../state/types.md#Transform), [`update`](../../core.md#ShapedArray.update), [`UntilingTransform`](core.md#UntilingTransform), [`update`](../../state/types.md#AbstractRef.update)
- used by: [`Transform`](../../state/types.md#Transform), [`mixed_matmul_kernel`](../../../experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.md#mixed_matmul_kernel), [`matmul_kernel`](../../../experimental/pallas/ops/gpu/blackwell_matmul_mgpu.md#matmul_kernel), [`_attention_bwd`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_bwd), [`ragged_dot_kernel`](../../../experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.md#ragged_dot_kernel), [`transform_type`](../../state/types.md#Transform.transform_type), [`kernel`](../../../experimental/pallas/ops/gpu/hopper_matmul_mgpu.md#kernel), [`transposed_ragged_dot`](../../../experimental/pallas/ops/gpu/transposed_ragged_dot_mgpu.md#transposed_ragged_dot), [`undo`](../../state/types.md#Transform.undo), [`entry`](../../../experimental/pallas/ops/gpu/attention_mgpu.md#_attention_forward.entry), [`to_gpu_transform`](core.md#to_gpu_transform), [`batch_transform`](core.md#batch_transform), [`undo`](core.md#UntilingTransform.undo)

### `TraceScope`  ·  implements/extends Enum
- def: [`jax/_src/pallas/mosaic_gpu/core.py:84`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L84)
- signature: `class TraceScope(enum.Enum):`
- members:
  - `WARP` — [`L85`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L85)
  - `WARPGROUP` — [`L86`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L86)
- used by: [`lower_jaxpr_to_module`](lowering.md#lower_jaxpr_to_module), [`profile_trace_scope`](core.md#CompilerParams.profile_trace_scope)

### `UnswizzleRef`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1213`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1213)
- members:
  - `commute_ndindexer(self, aval: jax_core.AbstractValue, indexer: indexing.NDIndexer)` — [`L1247`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1247)
  - `commute_reshape(self, aval: jax_core.ShapedArray, transform: state_types.ReshapeTransform)` — [`L1236`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1236)
  - `commute_transpose(self, _: jax_core.AbstractValue, transpose: state_types.TransposeTransform)` — [`L1228`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1228)
  - `pretty_print(self, context: jax_core.JaxprPpContext)` — [`L1278`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1278)
  - `swizzle_elems(self, dtype: jax.typing.DTypeLike | ir.Type)` — [`L1223`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1223)
  - `transform_type(self, x: jax_core.AbstractValue)` — [`L1216`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1216)
  - `undo(self, x: jax_core.AbstractValue)` — [`L1220`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1220)
  - `swizzle` — [`L1214`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1214)
- uses (calls/refs, reference-scoped): [`ShapedArray`](../../core.md#ShapedArray), [`DTypeLike`](../../typing.md#DTypeLike), [`AbstractValue`](../../core.md#AbstractValue), [`shape`](../../core.md#ShapedArray.shape), [`dtype`](../../core.md#ShapedArray.dtype), [`text`](../../pretty_printer.md#text), [`Transform`](../../state/types.md#Transform), [`bitwidth`](../../../experimental/mosaic/gpu/utils.md#bitwidth), [`NDIndexer`](../../state/indexing.md#NDIndexer), [`Slice`](../../indexing.md#Slice), [`indices`](../../state/indexing.md#NDIndexer.indices), [`JaxprPpContext`](../../core.md#JaxprPpContext), [`Doc`](../../pretty_printer.md#Doc), [`start`](../../indexing.md#Slice.start), [`size`](../../indexing.md#Slice.size), [`dtype_to_ir_type`](../../../experimental/mosaic/gpu/utils.md#dtype_to_ir_type), [`register_dataclass`](../../tree_util.md#register_dataclass), [`TransposeTransform`](../../state/types.md#TransposeTransform), [`ReshapeTransform`](../../state/types.md#ReshapeTransform), [`SwizzleTransform`](core.md#SwizzleTransform), [`permutation`](../../state/types.md#TransposeTransform.permutation)
- used by: [`_handle_transforms`](lowering.md#_handle_transforms), [`_swap_lowering_rule`](lowering.md#_swap_lowering_rule), [`_get_lowering_rule`](lowering.md#_get_lowering_rule), [`_async_store_smem_lowering`](primitives.md#_async_store_smem_lowering), [`_tcgen05_mma_lowering`](primitives.md#_tcgen05_mma_lowering), [`_jaxpr_call_lowering_rule`](lowering.md#_jaxpr_call_lowering_rule), [`_commute_transform`](lowering.md#_commute_transform), [`transform_type`](../../state/types.md#Transform.transform_type), [`_wgmma_lowering`](primitives.md#_wgmma_lowering), [`_async_copy_smem_to_tmem_lowering_rule`](primitives.md#_async_copy_smem_to_tmem_lowering_rule), [`undo`](../../state/types.md#Transform.undo), [`_atomic_store_lowering_rule`](primitives.md#_atomic_store_lowering_rule), [`pretty_print`](../../state/types.md#Transform.pretty_print), [`apply_unswizzle_and_untile`](interpret/jaxpr_interpret.md#apply_unswizzle_and_untile), [`undo`](core.md#SwizzleTransform.undo), [`_extract_smem_copy_params`](primitives.md#_extract_smem_copy_params), [`NOOP_TRANSFORMS`](interpret/gpu_callbacks.md#NOOP_TRANSFORMS)

### `UntilingTransform`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:742`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L742)
- members:
  - `commute_ndindexer(self, aval: jax_core.AbstractValue, indexer: indexing.NDIndexer)` — [`L788`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L788)
  - `commute_reshape(self, aval: jax_core.ShapedArray, transform: state_types.ReshapeTransform)` — [`L844`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L844)
  - `commute_transpose(self, _: jax_core.AbstractValue, transpose: state_types.TransposeTransform)` — [`L768`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L768)
  - `pretty_print(self, context: jax_core.JaxprPpContext)` — [`L933`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L933)
  - `transform_type(self, x)` — [`L745`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L745)
  - `undo(self, x: jax_core.AbstractValue)` — [`L765`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L765)
  - `tiling` — [`L743`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L743)
- protocol/private: `_maybe_cdiv_with_cast`[`L820`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L820)
- uses (calls/refs, reference-scoped): [`Array`](../../basearray.md#Array), [`AbstractRef`](../../state/types.md#AbstractRef), [`ShapedArray`](../../core.md#ShapedArray), [`dtype`](../../basearray.md#Array.dtype), [`AbstractValue`](../../core.md#AbstractValue), [`shape`](../../core.md#ShapedArray.shape), [`text`](../../pretty_printer.md#text), [`Transform`](../../state/types.md#Transform), [`NDIndexer`](../../state/indexing.md#NDIndexer), [`Slice`](../../indexing.md#Slice), [`indices`](../../state/indexing.md#NDIndexer.indices), [`JaxprPpContext`](../../core.md#JaxprPpContext), [`array`](../../../numpy/__init__.pyi.md#array), [`Doc`](../../pretty_printer.md#Doc), [`start`](../../indexing.md#Slice.start), [`cdiv`](../utils.md#cdiv), [`update`](../../core.md#ShapedArray.update), [`size`](../../indexing.md#Slice.size), [`register_dataclass`](../../tree_util.md#register_dataclass), [`from_indices_shape`](../../state/indexing.md#NDIndexer.from_indices_shape), [`stride`](../../indexing.md#Slice.stride), [`update`](../../state/types.md#AbstractRef.update), [`TransposeTransform`](../../state/types.md#TransposeTransform), [`shape`](../../state/indexing.md#NDIndexer.shape), [`ReshapeTransform`](../../state/types.md#ReshapeTransform), [`TilingTransform`](core.md#TilingTransform), [`shape`](../../state/types.md#ReshapeTransform.shape), [`permutation`](../../state/types.md#TransposeTransform.permutation), [`from_slice`](../../indexing.md#Slice.from_slice)
- used by: [`_handle_transforms`](lowering.md#_handle_transforms), [`_swap_lowering_rule`](lowering.md#_swap_lowering_rule), [`_get_lowering_rule`](lowering.md#_get_lowering_rule), [`_async_store_smem_lowering`](primitives.md#_async_store_smem_lowering), [`_tcgen05_mma_lowering`](primitives.md#_tcgen05_mma_lowering), [`_jaxpr_call_lowering_rule`](lowering.md#_jaxpr_call_lowering_rule), [`_commute_transform`](lowering.md#_commute_transform), [`transform_type`](../../state/types.md#Transform.transform_type), [`_wgmma_lowering`](primitives.md#_wgmma_lowering), [`_async_copy_smem_to_tmem_lowering_rule`](primitives.md#_async_copy_smem_to_tmem_lowering_rule), [`undo`](../../state/types.md#Transform.undo), [`_atomic_store_lowering_rule`](primitives.md#_atomic_store_lowering_rule), [`pretty_print`](../../state/types.md#Transform.pretty_print), [`apply_unswizzle_and_untile`](interpret/jaxpr_interpret.md#apply_unswizzle_and_untile), [`undo`](core.md#TilingTransform.undo), [`NOOP_TRANSFORMS`](interpret/gpu_callbacks.md#NOOP_TRANSFORMS)

### `WGMMAAbstractAccumulatorRef`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1585`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1585)
- members:
  - `update(self, inner_aval=None, memory_space=None, kind=None)` — [`L1591`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1591)
- protocol/private: `__repr__`[`L1588`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1588), `__slots__`[`L1586`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1586), `_getitem`[`L1598`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1598), `_setitem`[`L1606`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1606)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../../state/types.md#AbstractRef), [`update`](../../state/types.md#AbstractRef.update), [`wgmma_accumulator_load`](primitives.md#wgmma_accumulator_load), [`wgmma_accumulator_store`](primitives.md#wgmma_accumulator_store), [`is_trivial_index`](core.md#is_trivial_index)
- used by: [`_run_scoped_lowering_rule`](lowering.md#_run_scoped_lowering_rule), [`_run_state_lowering_rule`](lowering.md#_run_state_lowering_rule), [`ref_for_aval`](lowering.md#lower_pipelined_jaxpr_to_module.ref_for_aval), [`wgmma`](primitives.md#wgmma), [`update`](../../state/types.md#AbstractRef.update), [`_wgmma_accumulator_store_abstract_eval`](primitives.md#_wgmma_accumulator_store_abstract_eval), [`get_ref_aval`](core.md#WGMMAAccumulatorRef.get_ref_aval), [`_wgmma_ref_type_mapping`](core.md#_wgmma_ref_type_mapping), [`_wgmma_ref_effectful_abstract_eval`](primitives.md#_wgmma_ref_effectful_abstract_eval), [`wgmma_accumulator_load`](primitives.md#wgmma_accumulator_load), [`_getitem`](../../state/types.md#AbstractRef._getitem), [`wgmma_accumulator_store`](primitives.md#wgmma_accumulator_store), [`_setitem`](../../state/types.md#AbstractRef._setitem)

### `WGMMAAccumulatorRef`
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1558`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1558)
- signature: `class WGMMAAccumulatorRef:`
- members:
  - `get_ref_aval(self)` — [`L1563`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1563)
  - `init(array)` — [`L1573`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1573)
  - `dtype` — [`L1560`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1560)
  - `shape` — [`L1559`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1559)
- protocol/private: `_init`[`L1561`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1561)
- uses (calls/refs, reference-scoped): [`AbstractRef`](../../state/types.md#AbstractRef), [`ShapedArray`](../../core.md#ShapedArray), [`float32`](../../../numpy/__init__.pyi.md#float32.float32), [`dtype`](../../../numpy/__init__.pyi.md#dtype), [`MemorySpace`](core.md#MemorySpace), [`WGMMAAbstractAccumulatorRef`](core.md#WGMMAAbstractAccumulatorRef), [`REGS`](core.md#MemorySpace.REGS), [`uninitialized`](../../state/types.md#uninitialized)
- used by: [`ref_for_aval`](lowering.md#lower_pipelined_jaxpr_to_module.ref_for_aval), [`_wgmma_ref_type_mapping`](core.md#_wgmma_ref_type_mapping)

### `WarpMesh`  ·  implements/extends Mesh
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1718`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1718)
- doc: Represents a mesh over individual warps within a warpgroup.
- signature: `class WarpMesh(pallas_core.Mesh):`
- members:
  - `check_is_compatible_with(self, other_mesh)` — [`L1743`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1743)
  - `core_type(self)` — [`L1747`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1747)
  - `default_memory_space(self)` — [`L1736`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1736)
  - `discharges_effect(self, effect: jax_core.Effect)` — [`L1739`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1739)
  - `shape(self)` — [`L1730`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1730)
  - `supported_memory_spaces(self)` — [`L1751`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1751)
  - `tracing_context(self)` — [`L1755`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1755)
  - `axis_name` — [`L1727`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1727)
- protocol/private: `_NUM_WARPS_PER_WARPGROUP`[`L1726`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1726)
- uses (calls/refs, reference-scoped): [`MemorySpace`](core.md#MemorySpace), [`Mesh`](../core.md#Mesh), [`SMEM`](core.md#SMEM), [`AxisName`](../../core.md#AxisName), [`Effect`](../../core.md#Effect), [`GMEM`](core.md#GMEM), [`TMEM`](core.md#TMEM)
- used by: [`_core_map_lowering_rule`](lowering.md#_core_map_lowering_rule), [`_mpmd_map_lowering_rule`](lowering.md#_mpmd_map_lowering_rule), [`Mesh`](../core.md#Mesh), [`core_type`](../core.md#Mesh.core_type), [`shape`](../core.md#Mesh.shape), [`warp_map`](helpers.md#warp_map), [`check_is_compatible_with`](../core.md#Mesh.check_is_compatible_with), [`default_memory_space`](../core.md#Mesh.default_memory_space), [`discharges_effect`](../core.md#Mesh.discharges_effect), [`supported_memory_spaces`](../core.md#Mesh.supported_memory_spaces), [`tracing_context`](../core.md#Mesh.tracing_context), [`_per_warp`](../../../experimental/pallas/ops/gpu/blackwell_matmul_mgpu.md#matmul_kernel.kernel.mn_loop._._per_warp), [`_per_warp`](../../../experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.md#do_matmul._._per_warp)

### `_WGMMAPipelineEffect`  ·  implements/extends Effect
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1813`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1813)
- signature: `class _WGMMAPipelineEffect(effects.Effect):`
- used by: [`_wgmma_pipeline_effect`](core.md#_wgmma_pipeline_effect)

### `barrier_dtype`  ·  implements/extends extended
- def: [`jax/_src/pallas/mosaic_gpu/core.py:1463`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1463)
- signature: `class barrier_dtype(dtypes.extended):`
- uses (calls/refs, reference-scoped): [`extended`](../../dtypes.md#extended)
- used by: [`extended`](../../dtypes.md#extended), [`BarrierType`](core.md#BarrierType), [`ClusterBarrierType`](core.md#ClusterBarrierType)

## Functions
- `TryClusterCancelResult(num_buffers: int | None = None)` — [`L2007`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L2007) — Helper function to create Refs for cluster launch control results.
- `_gpu_mesh_discharge_rule(in_avals, out_avals, *args, mesh, jaxpr, compiler_params, interpret, debug, cost_estimate, name, metadata)` — [`L1759`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1759)
- `_layout_cast_abstract_eval(x, new_layout)` — [`L1844`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1844)
- `_ref_group_size(refs: _GPUMemoryRefTree)` — [`L480`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L480)
- `_ref_group_tmem_col_size(refs: _GPUMemoryRefTree)` — [`L501`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L501) — Returns the total number of TMEM columns used by a group of aliased Refs.
- `_slices(d)` — [`L74`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L74)
- `_vmap_rule(axis_size, in_batched, *args)` — [`L400`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L400)
- `_wgmma_ref_type_mapping(ref: WGMMAAccumulatorRef)` — [`L1577`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1577)
- `align_to(x: int, alignment: int)` — [`L470`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L470)
- `batch_transform(transform: state_types.Transform, leading_rank: int)` — [`L937`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L937)
- `batched_body(*refs, **scratch_ref_kwargs)` — [`L403`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L403)
- `cluster_ref(ref: _Ref, block_id: dict[jax_core.AxisName, Any])` — [`L1082`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1082) — Translate memref to a peer memref in the cluster.
- `commute_transpose_indexer(_: jax_core.AbstractValue, transpose: state_types.TransposeTransform, indexer: indexing.NDIndexer)` — [`L974`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L974)
- `flatten_ref_union(ref_union: AbstractRefUnion)` — [`L526`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L526) — Flattens a union of trees of references into a tuple of references.
- `infer_tmem_layout(shape: tuple[int, ...], dtype: jax.typing.DTypeLike, *, packed: bool, collective: bool)` — [`L512`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L512) — Infers the number of columns used and layout for allocating TMEM Refs.
- `is_trivial_index(idx, shape)` — [`L70`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L70) — Checks if the index selects the entire shape.
- `kernel(body: Callable[..., None] | api.NotSpecified = api.NotSpecified(), out_shape: object | api.NotSpecified = api.NotSpecified(), *, out_type: object | api.NotSpecified = api.NotSpecified(), scratch_types: ScratchShapeTree | api.NotSpecified = api.NotSpecified(), scratch_shapes: ScratchShapeTree | api.NotSpecified = api.NotSpecified(), compiler_params: pallas_core.CompilerParams | None = None, grid: tuple[int, ...] = (), grid_names: tuple[str, ...] = (), cluster: tuple[int, ...] = (), cluster_names: tuple[str, ...] = (), num_threads: int | None = None, thread_name: str | None = None, interpret: Any = None, debug: bool = False, **mesh_kwargs: Any)` — [`L248`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L248) — Entry point for defining a Mosaic GPU kernel. — documented in [jax-_src-pallas-mosaic_gpu-core](../../../../../concepts/jax-_src-pallas-mosaic_gpu-core.md)
- `kernel_body(*refs)` — [`L370`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L370)
- `layout_cast(x: Any, new_layout: SomeLayout)` — [`L1849`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1849) — Casts the layout of the given array.
- `multicast_ref(ref: _Ref, collective_axes: Hashable | tuple[Hashable, ...])` — [`L1098`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1098) — Return a multicast reference for cross-device operations.
- `remote_ref(ref: _Ref, device_id: jax.typing.ArrayLike, device_id_type: pallas_primitives.DeviceIdType = pallas_primitives.DeviceIdType.MESH)` — [`L1041`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1041) — Translate memref to a symmetric memref on a peer device.
- `to_gpu_transform(transform: state_types.Transform)` — [`L951`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L951)
- `to_transform_attr(transform: state_types.Transform)` — [`L963`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L963)
- `transform_ref(ref: pallas_core.TransformedRef, transform: state_types.Transform)` — [`L1121`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1121)
- `transpose_ref(ref: pallas_core.TransformedRef | Any, permutation: tuple[int, ...])` — [`L1133`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1133)
- `unflatten(ref)` — [`L536`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L536)
- `wrapper(*operands)` — [`L367`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L367)

## Module values
- `DimensionSemantics` — [`L58`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L58)
- `GMEM` — [`L1457`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1457)
- `Index` — [`L707`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L707)
- `LANExWARP_SEMANTICS` — [`L240`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L240)
- `LANExWG_SEMANTICS` — [`L238`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L238)
- `REGS` — [`L1460`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1460)
- `SMEM` — [`L1458`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1458)
- `SMEM_ALIGNMENT` — [`L66`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L66)
- `ScratchShapeTree` — [`L61`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L61)
- `TMEM` — [`L1459`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1459)
- `TMEM_COL_ALIGNMENT` — [`L67`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L67)
- `TransposeTransform` — [`L59`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L59)
- `WGxWARP_SEMANTICS` — [`L244`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L244)
- `WGxWG_SEMANTICS` — [`L242`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L242)
- `_GPUMemoryRefTree` — [`L477`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L477)
- `_Ref` — [`L56`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L56)
- `_WARPGROUP_AXIS_NAME` — [`L1634`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1634)
- `_memory_effect` — [`L1810`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1810)
- `_pdl_effect` — [`L1835`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1835)
- `_wgmma_pipeline_effect` — [`L1819`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1819)
- `layout_cast_p` — [`L1840`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic_gpu/core.py#L1840)

