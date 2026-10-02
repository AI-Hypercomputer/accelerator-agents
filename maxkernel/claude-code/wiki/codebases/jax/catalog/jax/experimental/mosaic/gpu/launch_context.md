---
title: 'Module: jax/experimental/mosaic/gpu/launch_context.py'
type: catalog
provenance: extracted
module: jax/experimental/mosaic/gpu/launch_context.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.mosaic.gpu.launch_context`/
symbols:
  LaunchContext.async_copy: LaunchContext#async_copy().
  LaunchContext._prepare_async_copy: LaunchContext#_prepare_async_copy().
  LaunchContext.to_remote_multicast: LaunchContext#to_remote_multicast().
  LaunchContext._prepare_tma: LaunchContext#_prepare_tma().
  LaunchContext.to_remote: LaunchContext#to_remote().
  MemRefTransform: MemRefTransform#
  TileTransform.tiling: TileTransform#tiling.
  LaunchContext.async_prefetch: LaunchContext#async_prefetch().
  LaunchContext.init_tma_desc: LaunchContext#init_tma_desc().
  LaunchContext._get_tma_desc: LaunchContext#_get_tma_desc().
  c: c.
  TileTransform: TileTransform#
  CopyPartition: CopyPartition#
  TileTransform.apply: TileTransform#apply().
  LaunchContext.host_collective_metadata: LaunchContext#host_collective_metadata().
  LaunchContext.device_id: LaunchContext#device_id().
  TileTransform.transform_gmem_shape: TileTransform#transform_gmem_shape().
  Scratch._create_ops: Scratch#_create_ops().
  AsyncCopyImplementation: AsyncCopyImplementation#
  Scratch._find_alloc_load_and_device_ptr: Scratch#_find_alloc_load_and_device_ptr().
  MemRefTransform.batch: MemRefTransform#batch().
  LaunchContext._alloc_scratch: LaunchContext#_alloc_scratch().
  LaunchContext.await_async_copy: LaunchContext#await_async_copy().
  OOBFillMode: OOBFillMode#
  LaunchContext._mark_parameters_if_multiprocess: LaunchContext#_mark_parameters_if_multiprocess().
  MemRefTransform.apply: MemRefTransform#apply().
  TileTransform.transform_shape: TileTransform#transform_shape().
  LaunchContext._recompute_peer_id: LaunchContext#_recompute_peer_id().
  LaunchContext._ensure_nvshmem_decls: LaunchContext#_ensure_nvshmem_decls().
  MemRefTransform.transform_gmem_shape: MemRefTransform#transform_gmem_shape().
  LaunchContext._get_parameter_address_on_peer: LaunchContext#_get_parameter_address_on_peer().
  MemRefTransform.transform_index: MemRefTransform#transform_index().
  MemRefTransform.transform_shape: MemRefTransform#transform_shape().
  CollapseLeadingIndicesTransform.strides: CollapseLeadingIndicesTransform#strides.
  LaunchContext.cluster_size: LaunchContext#cluster_size.
  CopyPartition.PARTITIONED: CopyPartition#PARTITIONED.
  TileTransform.rounding: TileTransform#rounding.
  LaunchContext.partition_dim: LaunchContext#partition_dim().
  TransposeTransform: TransposeTransform#
  LaunchContext: LaunchContext#
  LaunchContext._get_collective_metadata: LaunchContext#_get_collective_metadata().
  CollapseLeadingIndicesTransform.transform_index: CollapseLeadingIndicesTransform#transform_index().
  Scratch._find_first_op: Scratch#_find_first_op().
  LaunchContext.scratch: LaunchContext#scratch.
  TMAReductionOp: TMAReductionOp.
  _DefaultPredicate: _DefaultPredicate#
  AsyncCopyImplementation.TMA: AsyncCopyImplementation#TMA.
  TransposeTransform.batch: TransposeTransform#batch().
  Scratch.finalize_size: Scratch#finalize_size().
  MemRefTransform.transform_strides: MemRefTransform#transform_strides().
  TransposeTransform.permutation: TransposeTransform#permutation.
  LaunchContext.module: LaunchContext#module.
  GLOBAL_BROADCAST: GLOBAL_BROADCAST.
  _Partitioned: _Partitioned#
  _Replicated: _Replicated#
  MULTIMEM_ARGS_ATTR: MULTIMEM_ARGS_ATTR.
  GlobalBroadcast: GlobalBroadcast#
  Rounding: Rounding#
  DropUnitDimsTransform.apply: DropUnitDimsTransform#apply().
  _tma_dma_type: _tma_dma_type().
  LaunchContext.num_params: LaunchContext#num_params.
  LaunchContext._flag_multimem_usage: LaunchContext#_flag_multimem_usage().
  TransposeTransform.transform_shape: TransposeTransform#transform_shape().
  CollapseLeadingIndicesTransform.common_stride: CollapseLeadingIndicesTransform#common_stride().
  DropUnitDimsTransform.transform_shape: DropUnitDimsTransform#transform_shape().
  KERNEL_ARG_ID_ATTR: KERNEL_ARG_ID_ATTR.
  TileTransform.transform_index: TileTransform#transform_index().
  TransposeTransform.apply: TransposeTransform#apply().
  CollapseLeadingIndicesTransform.apply: CollapseLeadingIndicesTransform#apply().
  Scratch.next_offset: Scratch#next_offset.
  uses_collective_metadata: uses_collective_metadata().
  DropUnitDimsTransform: DropUnitDimsTransform#
  Scratch.device_ptr: Scratch#device_ptr().
  _find_kernel_argument_for_gmem_ref: _find_kernel_argument_for_gmem_ref().
  get_collective_metadata_size: get_collective_metadata_size().
  LaunchContext.profiler: LaunchContext#profiler.
  LaunchContext.tma_descriptors: LaunchContext#tma_descriptors.
  LaunchContext.named_region: LaunchContext#named_region().
  LaunchContext._find_kernel_argument_index: LaunchContext#_find_kernel_argument_index().
  DropUnitDimsTransform.unit_dims: DropUnitDimsTransform#unit_dims.
  AsyncCopyImplementation.CP_ASYNC: AsyncCopyImplementation#CP_ASYNC.
  LaunchContext.device_collective_metadata: LaunchContext#device_collective_metadata.
  LaunchContext.num_peers: LaunchContext#num_peers.
  ReplicationError: ReplicationError#
  _reduction_op_to_ptx: _reduction_op_to_ptx().
  MemRefTransform.to_attr: MemRefTransform#to_attr().
  TransposeTransform.transform_index: TransposeTransform#transform_index().
  CollapseLeadingIndicesTransform: CollapseLeadingIndicesTransform#
  CollapseLeadingIndicesTransform.transform_shape: CollapseLeadingIndicesTransform#transform_shape().
  DropUnitDimsTransform.transform_index: DropUnitDimsTransform#transform_index().
  _is_tma_reduction_op_supported: _is_tma_reduction_op_supported().
  COLLECTIVE_METADATA_SIZE: COLLECTIVE_METADATA_SIZE.
  COLLECTIVE_ATTR: COLLECTIVE_ATTR.
  CopyPartition.REPLICATED: CopyPartition#REPLICATED.
  Rounding.DOWN: Rounding#DOWN.
  Scratch._ops_created: Scratch#_ops_created.
  OOBFillMode.ZEROS: OOBFillMode#ZEROS.
  LaunchContext.num_processes: LaunchContext#num_processes.
  TileTransform.transform_strides: TileTransform#transform_strides().
  TileTransform.batch: TileTransform#batch().
  TileTransform.to_attr: TileTransform#to_attr().
  TransposeTransform.__post_init__: TransposeTransform#__post_init__().
  TransposeTransform.transform_gmem_shape: TransposeTransform#transform_gmem_shape().
  TransposeTransform.transform_strides: TransposeTransform#transform_strides().
  CollapseLeadingIndicesTransform.batch: CollapseLeadingIndicesTransform#batch().
  DropUnitDimsTransform.transform_gmem_shape: DropUnitDimsTransform#transform_gmem_shape().
  DropUnitDimsTransform.batch: DropUnitDimsTransform#batch().
  LaunchContext.await_cp_async_copy: LaunchContext#await_cp_async_copy().
  TMA_DESCRIPTOR_ALIGNMENT: TMA_DESCRIPTOR_ALIGNMENT.
  ORIGINAL_KERNEL_ARG_ATTR: ORIGINAL_KERNEL_ARG_ATTR.
  DEVICE_ID_ATTR: DEVICE_ID_ATTR.
  Rounding.UP: Rounding#UP.
  MOSAIC_GPU_SMEM_ALLOC_ATTR: MOSAIC_GPU_SMEM_ALLOC_ATTR.
  Scratch: Scratch#
  Scratch.host_init: Scratch#host_init.
  Scratch._module_op: Scratch#_module_op.
  OOBFillMode.UNDEFINED: OOBFillMode#UNDEFINED.
  OOBFillMode.PROMISE_IN_BOUNDS: OOBFillMode#PROMISE_IN_BOUNDS.
  LaunchContext.is_device_collective: LaunchContext#is_device_collective.
  LaunchContext._get_offset_to_parameter: LaunchContext#_get_offset_to_parameter().
  TMA_DESCRIPTOR_BYTES: TMA_DESCRIPTOR_BYTES.
  USES_MULTIMEM_ATTR: USES_MULTIMEM_ATTR.
  _Partitioned.axis: _Partitioned#axis.
  OnDeviceProfiler: OnDeviceProfiler.
  LaunchContext.buffers: LaunchContext#buffers.
  LaunchContext.host_init_wrapped: LaunchContext#host_init_wrapped().
  LaunchContext.cast_tma_desc: LaunchContext#cast_tma_desc().
  LaunchContext.check_contiguous_slice: LaunchContext#check_contiguous_slice().
  CollapseLeadingIndicesTransform.transform_gmem_shape: CollapseLeadingIndicesTransform#transform_gmem_shape().
  Scratch.__init__: Scratch#__init__().
---
# Module: [`jax/experimental/mosaic/gpu/launch_context.py`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py)

## Classes
### `AsyncCopyImplementation`  ·  implements/extends Enum
- def: [`jax/experimental/mosaic/gpu/launch_context.py:667`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L667)
- signature: `class AsyncCopyImplementation(enum.Enum):`
- members:
  - `CP_ASYNC` — [`L669`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L669)
  - `TMA` — [`L668`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L668)
- used by: [`async_copy`](launch_context.md#LaunchContext.async_copy), [`_prepare_async_copy`](launch_context.md#LaunchContext._prepare_async_copy), [`async_prefetch`](launch_context.md#LaunchContext.async_prefetch)

### `CollapseLeadingIndicesTransform`  ·  implements/extends MemRefTransform
- def: [`jax/experimental/mosaic/gpu/launch_context.py:313`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L313)
- doc: Collapses leading indices into one.
- signature: `class CollapseLeadingIndicesTransform(MemRefTransform):`
- members:
  - `apply(self, ref: ir.Value)` — [`L321`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L321)
  - `batch(self, leading_rank: int)` — [`L362`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L362)
  - `common_stride(self)` — [`L318`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L318)
  - `transform_gmem_shape(self, shape: Sequence[int])` — [`L345`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L345)
  - `transform_index(self, idx: Sequence[ir.Value])` — [`L348`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L348)
  - `transform_shape(self, shape: Sequence[int])` — [`L357`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L357)
  - `strides` — [`L315`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L315)
- uses (calls/refs, reference-scoped): [`MemRefTransform`](launch_context.md#MemRefTransform), [`c`](launch_context.md#c)
- used by: [`_prepare_tma`](launch_context.md#LaunchContext._prepare_tma), [`MemRefTransform`](launch_context.md#MemRefTransform), [`batch`](launch_context.md#MemRefTransform.batch), [`apply`](launch_context.md#MemRefTransform.apply), [`transform_gmem_shape`](launch_context.md#MemRefTransform.transform_gmem_shape), [`transform_index`](launch_context.md#MemRefTransform.transform_index), [`transform_shape`](launch_context.md#MemRefTransform.transform_shape)

### `CopyPartition`
- def: [`jax/experimental/mosaic/gpu/launch_context.py:96`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L96)
- members:
  - `PARTITIONED` — [`L97`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L97)
  - `REPLICATED` — [`L98`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L98)
- uses (calls/refs, reference-scoped): [`_Partitioned`](launch_context.md#_Partitioned), [`_Replicated`](launch_context.md#_Replicated)
- used by: [`async_copy`](launch_context.md#LaunchContext.async_copy), [`_copy_gmem_to_smem_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_copy_gmem_to_smem_lowering), [`_mgpu_async_load_op_lowering_rule`](dialect_lowering.md#_mgpu_async_load_op_lowering_rule), [`_prepare_async_copy`](launch_context.md#LaunchContext._prepare_async_copy), [`_prepare_tma`](launch_context.md#LaunchContext._prepare_tma), [`copy_gmem_to_smem`](../../../_src/pallas/mosaic_gpu/primitives.md#copy_gmem_to_smem), [`async_prefetch`](launch_context.md#LaunchContext.async_prefetch), [`_loop_body`](../../pallas/ops/gpu/blackwell_matmul_mgpu.md#matmul_kernel.kernel.mn_loop._._per_warp._memory._loop_body), [`_loop_body`](../../pallas/ops/gpu/blackwell_ragged_dot_mgpu.md#do_matmul._._per_warp._memory._loop_body), [`async_prefetch`](../../../_src/pallas/mosaic_gpu/primitives.md#async_prefetch), [`_Partitioned`](launch_context.md#_Partitioned), [`_Replicated`](launch_context.md#_Replicated)

### `DropUnitDimsTransform`  ·  implements/extends MemRefTransform
- def: [`jax/experimental/mosaic/gpu/launch_context.py:367`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L367)
- doc: Drops unit dimensions at the given positions.
- signature: `class DropUnitDimsTransform(MemRefTransform):`
- members:
  - `apply(self, ref: ir.Value)` — [`L371`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L371)
  - `batch(self, leading_rank: int)` — [`L410`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L410)
  - `transform_gmem_shape(self, shape: Sequence[int])` — [`L395`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L395)
  - `transform_index(self, idx: Sequence[ir.Value])` — [`L398`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L398)
  - `transform_shape(self, shape: Sequence[int])` — [`L401`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L401)
  - `unit_dims` — [`L369`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L369)
- uses (calls/refs, reference-scoped): [`MemRefTransform`](launch_context.md#MemRefTransform)
- used by: [`_prepare_tma`](launch_context.md#LaunchContext._prepare_tma), [`MemRefTransform`](launch_context.md#MemRefTransform), [`batch`](launch_context.md#MemRefTransform.batch), [`apply`](launch_context.md#MemRefTransform.apply), [`transform_gmem_shape`](launch_context.md#MemRefTransform.transform_gmem_shape), [`transform_index`](launch_context.md#MemRefTransform.transform_index), [`transform_shape`](launch_context.md#MemRefTransform.transform_shape)

### `GlobalBroadcast`
- def: [`jax/experimental/mosaic/gpu/launch_context.py:90`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L90)
- signature: `class GlobalBroadcast:`
- used by: [`async_copy`](launch_context.md#LaunchContext.async_copy), [`_mgpu_async_store_op_lowering_rule`](dialect_lowering.md#_mgpu_async_store_op_lowering_rule), [`init_tma_desc`](launch_context.md#LaunchContext.init_tma_desc), [`_get_tma_desc`](launch_context.md#LaunchContext._get_tma_desc), [`GLOBAL_BROADCAST`](launch_context.md#GLOBAL_BROADCAST)

### `LaunchContext`
- def: [`jax/experimental/mosaic/gpu/launch_context.py:691`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L691)
- signature: `class LaunchContext:`
- members:
  - `_alloc_scratch(self, size: int, alignment: int | None = None, host_init: Callable[[ir.Value], None] = lambda _: None, device_init: Callable[[ir.Value], Any] = lambda x: x)` — [`L730`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L730) — Allocates a GMEM scratch buffer.
  - `_find_kernel_argument_index(self, ref: ir.Value)` — [`L1964`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L1964) — Finds the index of the kernel argument used to derive the given reference.
  - `_get_parameter_address_on_peer(self, parameter_id: int, peer_id: ir.Value, on_host: bool)` — [`L1999`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L1999) — Returns the address of the given parameter on the given peer.
  - `_prepare_async_copy(self, gmem_ref: ir.Value, gmem_slice: Any, gmem_transform: tuple[MemRefTransform, ...], collective: Sequence[gpu.Dimension] | None, leader_tracked: CopyPartition | None, implementation: AsyncCopyImplementation)` — [`L917`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L917) — Performs setup common to TMA and CP_ASYNC implementations.
  - `_prepare_tma(self, gmem_ref: ir.Value, smem_ref: ir.Value | None, swizzle: int | None, slice_shape: list[int], dyn_base_indices: tuple[ir.Value, ...], gather_indices, squeezed_dims: tuple[int, ...], gmem_transform: tuple[MemRefTransform, ...], collective: Sequence[gpu.Dimension], leader_tracked: CopyPartition | None = None)` — [`L1040`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L1040) — Finalizes setup specific to the TMA implementation of async_copy.
  - `async_copy(self, *, src_ref: ir.Value, dst_ref: ir.Value, gmem_slice: Any = (), gmem_transform: MemRefTransform | tuple[MemRefTransform, ...] = (), gmem_peer_id: int | ir.Value | GlobalBroadcast | None = None, barrier: utils.BarrierRef | None = None, swizzle: int | None = None, arrive: bool | None = None, collective: Sequence[gpu.Dimension] | gpu.Dimension | None = None, leader_tracked: CopyPartition | None = None, predicate: ir.Value | None | _DefaultPredicate = _DefaultPredicate(), reduction_op: TMAReductionOp | None = None, implementation: AsyncCopyImplementation = AsyncCopyImplementation.TMA, oob_mode: OOBFillMode = OOBFillMode.ZEROS)` — [`L1191`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L1191) — Initiates an async copy between GMEM and SMEM.
  - `async_prefetch(self, *, gmem_ref: ir.Value, gmem_slice: Any = (), gmem_transform: MemRefTransform | tuple[MemRefTransform, ...] = (), gmem_peer_id: int | ir.Value | None = None, swizzle: int | None = None, collective: Sequence[gpu.Dimension] | gpu.Dimension | None = None, leader_tracked: CopyPartition | None = None, predicate: ir.Value | None | _DefaultPredicate = _DefaultPredicate())` — [`L1851`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L1851)
  - `await_async_copy(self, allow_groups: int, await_read_only: bool = False, scope: utils.ThreadSubset = utils.ThreadSubset.WARPGROUP)` — [`L1928`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L1928)
  - `await_cp_async_copy(self, allow_groups: int)` — [`L1940`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L1940)
  - `cast_tma_desc(device_ptr)` — [`L904`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L904)
  - `check_contiguous_slice(slice_shape, strides)` — [`L1655`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L1655)
  - `device_id(self, on_host: bool = False)` — [`L2182`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L2182)
  - `host_collective_metadata(self)` — [`L716`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L716)
  - `host_init_wrapped(host_ptr)` — [`L750`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L750)
  - `init_tma_desc(host_ptr: ir.Value)` — [`L831`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L831)
  - `named_region(self, *args, **kwargs)` — [`L708`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L708)
  - `partition_dim(dim: int, idx: ir.Value, num_chunks: int)` — [`L1129`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L1129)
  - `to_remote(self, ref: ir.Value, peer: ir.Value, *, _kernel_arg_idx: int | None = None, on_host: bool = False)` — [`L2030`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L2030)
  - `to_remote_multicast(self, ref: ir.Value, on_host: bool = False)` — [`L2114`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L2114)
  - `buffers` — [`L695`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L695)
  - `cluster_size` — [`L694`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L694)
  - `device_collective_metadata` — [`L697`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L697)
  - `is_device_collective` — [`L705`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L705)
  - `module` — [`L692`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L692)
  - `num_params` — [`L699`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L699)
  - `num_peers` — [`L698`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L698)
  - `num_processes` — [`L700`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L700)
  - `profiler` — [`L696`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L696)
  - `scratch` — [`L693`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L693)
  - `tma_descriptors` — [`L701`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L701)
- protocol/private: `_ensure_nvshmem_decls`[`L1944`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L1944), `_flag_multimem_usage`[`L809`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L809), `_get_collective_metadata`[`L1992`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L1992), `_get_offset_to_parameter`[`L2015`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L2015), `_get_tma_desc`[`L812`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L812), `_mark_parameters_if_multiprocess`[`L2021`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L2021), `_recompute_peer_id`[`L766`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L766)
- uses (calls/refs, reference-scoped): [`FragmentedArray`](fragmented_array.md#FragmentedArray), [`mlir_dtype`](fragmented_array.md#FragmentedArray.mlir_dtype), [`layout`](fragmented_array.md#FragmentedArray.layout), [`bitwidth`](utils.md#bitwidth), [`registers`](fragmented_array.md#FragmentedArray.registers), [`shape`](fragmented_array.md#FragmentedArray.shape), [`transfer_tiled`](fragmented_array.md#FragmentedArray.transfer_tiled), [`is_signed`](fragmented_array.md#FragmentedArray.is_signed), [`ThreadSubset`](utils.md#ThreadSubset), [`WGStridedFragLayout`](fragmented_array.md#WGStridedFragLayout), [`memref_ptr`](utils.md#memref_ptr), [`warpgroup_barrier`](utils.md#warpgroup_barrier), [`vec_size`](fragmented_array.md#WGStridedFragLayout.vec_size), [`MemRefTransform`](launch_context.md#MemRefTransform), [`vector_length`](fragmented_array.md#TiledLayout.vector_length), [`c`](launch_context.md#c), [`single_thread_predicate`](utils.md#single_thread_predicate), [`WARPGROUP`](utils.md#ThreadSubset.WARPGROUP), [`MultimemRef`](utils.md#MultimemRef), [`get_ptr`](utils.md#BarrierRef.get_ptr), [`TileTransform`](launch_context.md#TileTransform), [`WARPGROUP_SIZE`](fragmented_array.md#WARPGROUP_SIZE), [`CopyPartition`](launch_context.md#CopyPartition), [`tiled_copy_smem_gmem_layout`](fragmented_array.md#tiled_copy_smem_gmem_layout), [`TMA_INDICES_4_LAYOUT`](fragmented_array.md#TMA_INDICES_4_LAYOUT), [`is_smem_ref`](utils.md#is_smem_ref), [`BarrierRef`](utils.md#BarrierRef), [`TMA_INDICES_LAYOUT`](fragmented_array.md#TMA_INDICES_LAYOUT), [`getelementptr`](utils.md#getelementptr), [`WARP`](utils.md#ThreadSubset.WARP), [`memref_slice`](utils.md#memref_slice), [`AsyncCopyImplementation`](launch_context.md#AsyncCopyImplementation), [`ref`](utils.md#MultimemRef.ref), [`batch`](launch_context.md#MemRefTransform.batch), [`cluster_idx`](utils.md#cluster_idx), [`warp_idx`](utils.md#warp_idx), [`OOBFillMode`](launch_context.md#OOBFillMode), [`linear_thread_idxs`](fragmented_array.md#WGStridedFragLayout.linear_thread_idxs), [`apply`](launch_context.md#MemRefTransform.apply), [`get_contiguous_strides`](utils.md#get_contiguous_strides)  (+53 more)
- used by: [`_handle_transforms`](../../../_src/pallas/mosaic_gpu/lowering.md#_handle_transforms), [`lower_jaxpr_to_mosaic_gpu`](../../../_src/pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_mosaic_gpu), [`body`](../../../_src/pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_module.body), [`_copy_gmem_to_smem_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_copy_gmem_to_smem_lowering), [`_semaphore_signal_multicast_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_semaphore_signal_multicast_lowering), [`_copy_smem_to_gmem_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_copy_smem_to_gmem_lowering), [`_launch`](core.md#_launch), [`_semaphore_signal_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_semaphore_signal_lowering_rule), [`_axis_index_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_axis_index_rule), [`_multimem_store_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_multimem_store_lowering_rule), [`_mgpu_async_load_op_lowering_rule`](dialect_lowering.md#_mgpu_async_load_op_lowering_rule), [`_cond_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_cond_lowering_rule), [`_async_prefetch_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_async_prefetch_lowering), [`_multimem_load_reduce_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_multimem_load_reduce_lowering_rule), [`query_cluster_cancel_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#query_cluster_cancel_lowering), [`_wait_smem_to_gmem_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_wait_smem_to_gmem_lowering), [`_multimem_load_reduce_lowering_rule_wg`](../../../_src/pallas/mosaic_gpu/primitives.md#_multimem_load_reduce_lowering_rule_wg), [`launch_ctx`](../../../_src/pallas/mosaic_gpu/lowering.md#LoweringRuleContext.launch_ctx), [`_mgpu_async_store_op_lowering_rule`](dialect_lowering.md#_mgpu_async_store_op_lowering_rule), [`_eval_index_map`](../../../_src/pallas/mosaic_gpu/lowering.md#_eval_index_map), [`_lowering_context`](dialect_lowering.md#_lowering_context), [`_mgpu_async_prefetch_op_lowering_rule`](dialect_lowering.md#_mgpu_async_prefetch_op_lowering_rule), [`_block_id`](../../../_src/pallas/mosaic_gpu/lowering.md#_block_id), [`launch_context`](dialect_lowering.md#LoweringContext.launch_context), [`lower_mgpu_dialect`](dialect_lowering.md#lower_mgpu_dialect)

### `MemRefTransform`
- def: [`jax/experimental/mosaic/gpu/launch_context.py:116`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L116)
- signature: `class MemRefTransform:`
- members:
  - `apply(self, ref: ir.Value)` — [`L117`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L117)
  - `batch(self, leading_rank: int)` — [`L137`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L137) — Returns a transform that accepts a ref with the extra `leading_rank` dims.
  - `to_attr(self)` — [`L145`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L145)
  - `transform_gmem_shape(self, shape: Sequence[int])` — [`L126`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L126) — Applies the shape transformation to the given GMEM shape.
  - `transform_index(self, idx: Sequence[ir.Value])` — [`L120`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L120)
  - `transform_shape(self, shape: Sequence[int])` — [`L123`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L123)
  - `transform_strides(self, strides: Sequence[int])` — [`L134`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L134)
- uses (calls/refs, reference-scoped): [`TileTransform`](launch_context.md#TileTransform), [`apply`](launch_context.md#TileTransform.apply), [`transform_gmem_shape`](launch_context.md#TileTransform.transform_gmem_shape), [`transform_shape`](launch_context.md#TileTransform.transform_shape), [`transform_index`](launch_context.md#CollapseLeadingIndicesTransform.transform_index), [`batch`](launch_context.md#TransposeTransform.batch), [`apply`](launch_context.md#DropUnitDimsTransform.apply), [`transform_shape`](launch_context.md#DropUnitDimsTransform.transform_shape), [`transform_shape`](launch_context.md#TransposeTransform.transform_shape), [`apply`](launch_context.md#CollapseLeadingIndicesTransform.apply), [`apply`](launch_context.md#TransposeTransform.apply), [`transform_index`](launch_context.md#TileTransform.transform_index), [`DropUnitDimsTransform`](launch_context.md#DropUnitDimsTransform), [`CollapseLeadingIndicesTransform`](launch_context.md#CollapseLeadingIndicesTransform), [`transform_index`](launch_context.md#DropUnitDimsTransform.transform_index), [`transform_index`](launch_context.md#TransposeTransform.transform_index), [`transform_shape`](launch_context.md#CollapseLeadingIndicesTransform.transform_shape), [`batch`](launch_context.md#CollapseLeadingIndicesTransform.batch), [`batch`](launch_context.md#DropUnitDimsTransform.batch), [`batch`](launch_context.md#TileTransform.batch), [`to_attr`](launch_context.md#TileTransform.to_attr), [`transform_gmem_shape`](launch_context.md#DropUnitDimsTransform.transform_gmem_shape), [`transform_gmem_shape`](launch_context.md#TransposeTransform.transform_gmem_shape), [`transform_strides`](launch_context.md#TileTransform.transform_strides), [`transform_strides`](launch_context.md#TransposeTransform.transform_strides), [`transform_gmem_shape`](launch_context.md#CollapseLeadingIndicesTransform.transform_gmem_shape)
- used by: [`async_copy`](launch_context.md#LaunchContext.async_copy), [`_prepare_async_copy`](launch_context.md#LaunchContext._prepare_async_copy), [`_prepare_tma`](launch_context.md#LaunchContext._prepare_tma), [`async_prefetch`](launch_context.md#LaunchContext.async_prefetch), [`init_tma_desc`](launch_context.md#LaunchContext.init_tma_desc), [`_get_tma_desc`](launch_context.md#LaunchContext._get_tma_desc), [`transform_type`](dialect_lowering.md#transform_type), [`TileTransform`](launch_context.md#TileTransform), [`memref_transforms_from_transforms_attr`](dialect_lowering.md#memref_transforms_from_transforms_attr), [`to_gpu_transform`](../../../_src/pallas/mosaic_gpu/core.md#to_gpu_transform), [`to_transform_attr`](../../../_src/pallas/mosaic_gpu/core.md#to_transform_attr), [`_check_collapse_shape`](dialect_lowering.md#_check_collapse_shape), [`TransposeTransform`](launch_context.md#TransposeTransform), [`to_transform_attr`](layouts.md#to_transform_attr), [`batch`](launch_context.md#TransposeTransform.batch), [`from_transform_attr`](layouts.md#from_transform_attr), [`DropUnitDimsTransform`](launch_context.md#DropUnitDimsTransform), [`tma_descriptors`](launch_context.md#LaunchContext.tma_descriptors), [`CollapseLeadingIndicesTransform`](launch_context.md#CollapseLeadingIndicesTransform), [`batch`](launch_context.md#CollapseLeadingIndicesTransform.batch), [`batch`](launch_context.md#DropUnitDimsTransform.batch), [`batch`](launch_context.md#TileTransform.batch)

### `OOBFillMode`  ·  implements/extends IntEnum
- def: [`jax/experimental/mosaic/gpu/launch_context.py:684`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L684)
- signature: `class OOBFillMode(enum.IntEnum):`
- members:
  - `PROMISE_IN_BOUNDS` — [`L686`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L686)
  - `UNDEFINED` — [`L685`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L685)
  - `ZEROS` — [`L687`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L687)
- used by: [`async_copy`](launch_context.md#LaunchContext.async_copy), [`_copy_smem_to_gmem_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_copy_smem_to_gmem_lowering), [`_mgpu_async_load_op_lowering_rule`](dialect_lowering.md#_mgpu_async_load_op_lowering_rule), [`copy_gmem_to_smem`](../../../_src/pallas/mosaic_gpu/primitives.md#copy_gmem_to_smem), [`_mgpu_async_store_op_lowering_rule`](dialect_lowering.md#_mgpu_async_store_op_lowering_rule)

### `ReplicationError`
- def: [`jax/experimental/mosaic/gpu/launch_context.py:2200`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L2200)
- used by: [`init_tma_desc`](launch_context.md#LaunchContext.init_tma_desc), [`_recompute_peer_id`](launch_context.md#LaunchContext._recompute_peer_id)

### `Rounding`  ·  implements/extends Enum
- def: [`jax/experimental/mosaic/gpu/launch_context.py:151`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L151)
- signature: `class Rounding(enum.Enum):`
- members:
  - `DOWN` — [`L153`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L153)
  - `UP` — [`L152`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L152)
- used by: [`apply`](launch_context.md#TileTransform.apply), [`transform_gmem_shape`](launch_context.md#TileTransform.transform_gmem_shape), [`transform_shape`](launch_context.md#TileTransform.transform_shape), [`rounding`](launch_context.md#TileTransform.rounding)

### `Scratch`
- def: [`jax/experimental/mosaic/gpu/launch_context.py:418`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L418)
- doc: Manages ops handling the GMEM scratch that contains the TMA descriptors.
- signature: `class Scratch:`
- members:
  - `device_ptr(self)` — [`L544`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L544)
  - `finalize_size(self)` — [`L548`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L548) — Allocates and initializes the host buffer. This needs to be done after
  - `host_init` — [`L472`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L472)
  - `next_offset` — [`L471`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L471)
- protocol/private: `__init__`[`L470`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L470), `_create_ops`[`L501`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L501), `_find_alloc_load_and_device_ptr`[`L526`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L526), `_find_first_op`[`L485`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L485), `_module_op`[`L483`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L483), `_ops_created`[`L473`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L473)
- uses (calls/refs, reference-scoped): [`c`](launch_context.md#c), [`MOSAIC_GPU_SMEM_ALLOC_ATTR`](launch_context.md#MOSAIC_GPU_SMEM_ALLOC_ATTR), [`TMA_DESCRIPTOR_ALIGNMENT`](launch_context.md#TMA_DESCRIPTOR_ALIGNMENT)
- used by: [`_launch`](core.md#_launch), [`_alloc_scratch`](launch_context.md#LaunchContext._alloc_scratch), [`scratch`](launch_context.md#LaunchContext.scratch)

### `TileTransform`  ·  implements/extends MemRefTransform
- def: [`jax/experimental/mosaic/gpu/launch_context.py:157`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L157)
- doc: Tiles a suffix of memref dimensions.
- signature: `class TileTransform(MemRefTransform):`
- members:
  - `apply(self, ref: ir.Value)` — [`L200`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L200)
  - `batch(self, leading_rank: int)` — [`L277`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L277)
  - `to_attr(self)` — [`L280`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L280)
  - `transform_gmem_shape(self, shape: Sequence[int])` — [`L169`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L169)
  - `transform_index(self, idx: Sequence[ir.Value])` — [`L233`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L233)
  - `transform_shape(self, shape: Sequence[int])` — [`L248`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L248)
  - `transform_strides(self, strides: Sequence[int])` — [`L269`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L269)
  - `rounding` — [`L167`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L167)
  - `tiling` — [`L166`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L166)
- uses (calls/refs, reference-scoped): [`MemRefTransform`](launch_context.md#MemRefTransform), [`c`](launch_context.md#c), [`memref_slice`](utils.md#memref_slice), [`memref_transpose`](utils.md#memref_transpose), [`TransposeTransform`](launch_context.md#TransposeTransform), [`Rounding`](launch_context.md#Rounding), [`memref_unfold`](utils.md#memref_unfold), [`transform_shape`](launch_context.md#TransposeTransform.transform_shape), [`DOWN`](launch_context.md#Rounding.DOWN), [`UP`](launch_context.md#Rounding.UP)
- used by: [`async_copy`](launch_context.md#LaunchContext.async_copy), [`_async_store_smem_op_lowering_rule`](dialect_lowering.md#_async_store_smem_op_lowering_rule), [`_vector_load_op_lowering_rule`](dialect_lowering.md#_vector_load_op_lowering_rule), [`_memref_subview_op_lowering_rule`](dialect_lowering.md#_memref_subview_op_lowering_rule), [`_vector_store_op_lowering_rule`](dialect_lowering.md#_vector_store_op_lowering_rule), [`MemRefTransform`](launch_context.md#MemRefTransform), [`transform_type`](dialect_lowering.md#transform_type), [`_memref_collapse_shape_op_lowering_rule`](dialect_lowering.md#_memref_collapse_shape_op_lowering_rule), [`_memref_transpose_op_lowering_rule`](dialect_lowering.md#_memref_transpose_op_lowering_rule), [`memref_transforms_from_transforms_attr`](dialect_lowering.md#memref_transforms_from_transforms_attr), [`to_gpu_transform`](../../../_src/pallas/mosaic_gpu/core.md#to_gpu_transform), [`batch`](launch_context.md#MemRefTransform.batch), [`apply`](launch_context.md#MemRefTransform.apply), [`_check_collapse_shape`](dialect_lowering.md#_check_collapse_shape), [`_extract_layout_candidates_from_smem_registers_transfer`](layout_inference.md#_extract_layout_candidates_from_smem_registers_transfer), [`transform_gmem_shape`](launch_context.md#MemRefTransform.transform_gmem_shape), [`transform_index`](launch_context.md#MemRefTransform.transform_index), [`transform_shape`](launch_context.md#MemRefTransform.transform_shape), [`to_transform_attr`](layouts.md#to_transform_attr), [`load_tiled`](dialect_lowering.md#_vector_load_op_lowering_rule.load_tiled), [`_compute_swizzle`](layout_inference.md#_compute_swizzle), [`_extract_smem_transforms_from_custom_transform_attrs`](layout_inference.md#_extract_smem_transforms_from_custom_transform_attrs), [`transform_strides`](launch_context.md#MemRefTransform.transform_strides), [`from_transform_attr`](layouts.md#from_transform_attr), [`store_tiled`](dialect_lowering.md#_vector_store_op_lowering_rule.store_tiled), [`store_tiled_async`](dialect_lowering.md#_async_store_smem_op_lowering_rule.store_tiled_async), [`_extract_layout_candidates_from_mma_tiling`](layout_inference.md#_extract_layout_candidates_from_mma_tiling), [`to_attr`](launch_context.md#MemRefTransform.to_attr)

### `TransposeTransform`
- def: [`jax/experimental/mosaic/gpu/launch_context.py:284`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L284)
- members:
  - `apply(self, ref: ir.Value)` — [`L292`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L292)
  - `batch(self, leading_rank: int)` — [`L307`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L307)
  - `transform_gmem_shape(self, shape: Sequence[int])` — [`L295`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L295)
  - `transform_index(self, idx: Sequence[ir.Value])` — [`L298`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L298)
  - `transform_shape(self, shape: Sequence[int])` — [`L301`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L301)
  - `transform_strides(self, strides: Sequence[int])` — [`L304`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L304)
  - `permutation` — [`L286`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L286)
- protocol/private: `__post_init__`[`L288`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L288)
- uses (calls/refs, reference-scoped): [`MemRefTransform`](launch_context.md#MemRefTransform), [`memref_transpose`](utils.md#memref_transpose)
- used by: [`_mgpu_async_load_op_lowering_rule`](dialect_lowering.md#_mgpu_async_load_op_lowering_rule), [`_prepare_async_copy`](launch_context.md#LaunchContext._prepare_async_copy), [`_mgpu_async_store_op_lowering_rule`](dialect_lowering.md#_mgpu_async_store_op_lowering_rule), [`transform_gmem_shape`](launch_context.md#TileTransform.transform_gmem_shape), [`to_gpu_transform`](../../../_src/pallas/mosaic_gpu/core.md#to_gpu_transform), [`batch`](launch_context.md#MemRefTransform.batch), [`apply`](launch_context.md#MemRefTransform.apply), [`transform_gmem_shape`](launch_context.md#MemRefTransform.transform_gmem_shape), [`transform_index`](launch_context.md#MemRefTransform.transform_index), [`transform_shape`](launch_context.md#MemRefTransform.transform_shape), [`transform_strides`](launch_context.md#MemRefTransform.transform_strides)

### `_DefaultPredicate`
- def: [`jax/experimental/mosaic/gpu/launch_context.py:569`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L569)
- signature: `class _DefaultPredicate:`
- used by: [`async_copy`](launch_context.md#LaunchContext.async_copy), [`async_prefetch`](launch_context.md#LaunchContext.async_prefetch)

### `_Partitioned`
- def: [`jax/experimental/mosaic/gpu/launch_context.py:102`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L102)
- members:
  - `axis` — [`L103`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L103)
- uses (calls/refs, reference-scoped): [`CopyPartition`](launch_context.md#CopyPartition)
- used by: [`_prepare_async_copy`](launch_context.md#LaunchContext._prepare_async_copy), [`_prepare_tma`](launch_context.md#LaunchContext._prepare_tma), [`PARTITIONED`](launch_context.md#CopyPartition.PARTITIONED)

### `_Replicated`
- def: [`jax/experimental/mosaic/gpu/launch_context.py:107`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L107)
- uses (calls/refs, reference-scoped): [`CopyPartition`](launch_context.md#CopyPartition)
- used by: [`async_copy`](launch_context.md#LaunchContext.async_copy), [`CopyPartition`](launch_context.md#CopyPartition)

## Functions
- `_find_kernel_argument_for_gmem_ref(gmem_ref: ir.Value)` — [`L573`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L573) — Returns the kernel argument value for a given gmem_ref.
- `_is_tma_reduction_op_supported(reduction_op: TMAReductionOp | None, dtype: ir.Type)` — [`L592`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L592) — Returns whether the given TMA reduction op supports the given dtype.
- `_reduction_op_to_ptx(reduction_op: TMAReductionOp)` — [`L84`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L84)
- `_tma_dma_type(element_type: ir.Type, reduction_op: TMAReductionOp | None)` — [`L626`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L626) — Returns the TMA DMA type for the given element type and signedness.
- `get_collective_metadata_size(num_params: int, num_peers: int)` — [`L672`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L672) — Returns the size of the collective metadata buffer for the given number of parameters and peers.
- `uses_collective_metadata(module)` — [`L80`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L80)

## Module values
- `COLLECTIVE_ATTR` — [`L64`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L64)
- `COLLECTIVE_METADATA_SIZE` — [`L61`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L61)
- `DEVICE_ID_ATTR` — [`L72`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L72)
- `GLOBAL_BROADCAST` — [`L93`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L93)
- `KERNEL_ARG_ID_ATTR` — [`L67`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L67)
- `MOSAIC_GPU_SMEM_ALLOC_ATTR` — [`L416`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L416)
- `MULTIMEM_ARGS_ATTR` — [`L77`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L77)
- `ORIGINAL_KERNEL_ARG_ATTR` — [`L69`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L69)
- `OnDeviceProfiler` — [`L414`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L414)
- `TMAReductionOp` — [`L43`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L43)
- `TMA_DESCRIPTOR_ALIGNMENT` — [`L42`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L42)
- `TMA_DESCRIPTOR_BYTES` — [`L41`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L41)
- `USES_MULTIMEM_ATTR` — [`L74`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L74)
- `c` — [`L88`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/launch_context.py#L88)

