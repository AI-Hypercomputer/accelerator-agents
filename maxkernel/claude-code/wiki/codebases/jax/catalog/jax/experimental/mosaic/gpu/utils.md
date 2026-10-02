---
title: 'Module: jax/experimental/mosaic/gpu/utils.py'
type: catalog
provenance: extracted
module: jax/experimental/mosaic/gpu/utils.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.mosaic.gpu.utils`/
symbols:
  bitwidth: bitwidth().
  c: c().
  ThreadSubset: ThreadSubset#
  bitcast: bitcast().
  dtype_to_ir_type: dtype_to_ir_type().
  memref_ptr: memref_ptr().
  warpgroup_barrier: warpgroup_barrier().
  CollectiveBarrierRef.arrive: CollectiveBarrierRef#arrive().
  is_signed: is_signed().
  single_thread_predicate: single_thread_predicate().
  ThreadSubset.WARPGROUP: ThreadSubset#WARPGROUP.
  MultimemRef: MultimemRef#
  BarrierRef.get_ptr: BarrierRef#get_ptr().
  DialectBarrierRef.from_barrier_memref: DialectBarrierRef#from_barrier_memref().
  is_known_divisible: is_known_divisible().
  BarrierRef.arrive: BarrierRef#arrive().
  DialectBarrierRef.barrier_ref: DialectBarrierRef#barrier_ref.
  BarrierRef.remap_to_cluster: BarrierRef#remap_to_cluster().
  SemaphoreRef.wait: SemaphoreRef#wait().
  BarrierRef.initialize: BarrierRef#initialize().
  shfl_bfly: shfl_bfly().
  is_smem_ref: is_smem_ref().
  BarrierRef: BarrierRef#
  DialectBarrierRef: DialectBarrierRef#
  DialectBarrierRef.as_barrier_memref: DialectBarrierRef#as_barrier_memref().
  getelementptr: getelementptr().
  vector_concat: vector_concat().
  ThreadSubset.WARP: ThreadSubset#WARP.
  multimem_load_reduce: multimem_load_reduce().
  try_cluster_cancel: try_cluster_cancel().
  redux: redux().
  DialectBarrierRef.initialize: DialectBarrierRef#initialize().
  vector_slice: vector_slice().
  smem: smem().
  memref_slice: memref_slice().
  get_cluster_ref: get_cluster_ref().
  MultimemRef.ref: MultimemRef#ref.
  BarrierRef.__getitem__: BarrierRef#__getitem__().
  thread_idx: thread_idx.
  SemaphoreRef.signal: SemaphoreRef#signal().
  memref_transpose: memref_transpose().
  warp_idx: warp_idx().
  CollectiveBarrierRef.initialize: CollectiveBarrierRef#initialize().
  cluster_idx: cluster_idx().
  debug_print: debug_print().
  when: when().
  single_thread: single_thread().
  get_contiguous_strides: get_contiguous_strides().
  MultimemRef.store: MultimemRef#store().
  BarrierRef.arrive_expect_tx: BarrierRef#arrive_expect_tx().
  warpgroup_idx: warpgroup_idx().
  parse_indices: parse_indices().
  DynamicSlice: DynamicSlice#
  BarrierRef.num_barriers: BarrierRef#num_barriers.
  BarrierRef.test: BarrierRef#test().
  SemaphoreRef.signal_multimem: SemaphoreRef#signal_multimem().
  Partition.get_base: Partition#get_base().
  WARPGROUP_SIZE.WARPGROUP_SIZE: WARPGROUP_SIZE.WARPGROUP_SIZE.
  ptr_as_memref: ptr_as_memref().
  bytewidth: bytewidth().
  memref_reshape: memref_reshape().
  DialectBarrierRef.test: DialectBarrierRef#test().
  DialectBarrierRef.wait: DialectBarrierRef#wait().
  CollectiveBarrierRef.__iter__: CollectiveBarrierRef#__iter__().
  CollectiveBarrierRef.__getitem__: CollectiveBarrierRef#__getitem__().
  Partition.__init__: Partition#__init__().
  Partition1D.refine: Partition1D#refine().
  BarrierRef.wait: BarrierRef#wait().
  CollectiveBarrierRef.barrier: CollectiveBarrierRef#barrier.
  Partition.target_block_shape: Partition#target_block_shape().
  memref_fold: memref_fold().
  BarrierRef.phases: BarrierRef#phases.
  CollectiveBarrierRef: CollectiveBarrierRef#
  _reshape: _reshape().
  warp_barrier: warp_barrier().
  BarrierRef.complete_tx: BarrierRef#complete_tx().
  DialectBarrierRef.wait_parity: DialectBarrierRef#wait_parity().
  get_arch: get_arch().
  _ONCE_PER._ONCE_PER: _ONCE_PER._ONCE_PER.
  pack_array: pack_array().
  multimem_store: multimem_store().
  is_memref_transposed: is_memref_transposed().
  BarrierRef.wait_parity: BarrierRef#wait_parity().
  BarrierRef.update_parities: BarrierRef#update_parities().
  DialectBarrierRef.orders_tensor_core: DialectBarrierRef#orders_tensor_core.
  get_memref_llvm_address_space: get_memref_llvm_address_space().
  cluster_collective_mask: cluster_collective_mask().
  dyn_dot: dyn_dot().
  ds: ds.
  DialectBarrierRef.__iter__: DialectBarrierRef#__iter__().
  DialectBarrierRef.__getitem__: DialectBarrierRef#__getitem__().
  DialectBarrierRef.test_parity: DialectBarrierRef#test_parity().
  Partition1D.partition: Partition1D#partition.
  warp_tree_reduce: warp_tree_reduce().
  is_cluster_smem_ref: is_cluster_smem_ref().
  _warp_bcast: _warp_bcast().
  bitwidth_impl: bitwidth_impl().
  DynamicSlice.base: DynamicSlice#base.
  BarrierRef.test_parity: BarrierRef#test_parity().
  DialectBarrierRef.get_ptr: DialectBarrierRef#get_ptr().
  Partition.source_bounds: Partition#source_bounds.
  tmem: tmem().
  WARPS_IN_WARPGROUP.WARPS_IN_WARPGROUP: WARPS_IN_WARPGROUP.WARPS_IN_WARPGROUP.
  BarrierRef._ptx_scope: BarrierRef#_ptx_scope().
  prmt: prmt().
  _vector_concat_rec: _vector_concat_rec().
  ThreadSubset.BLOCK: ThreadSubset#BLOCK.
  memref_unfold: memref_unfold().
  Partition.partition: Partition#partition.
  Partition1D.get_base: Partition1D#get_base().
  Arch.major: Arch#major.
  fori: fori().
  is_tmem_ref: is_tmem_ref().
  query_cluster_cancel: query_cluster_cancel().
  MBARRIER_BYTES: MBARRIER_BYTES.
  DynamicSlice.length: DynamicSlice#length.
  DynamicSlice.__post_init__: DynamicSlice#__post_init__().
  BarrierRef.base_address: BarrierRef#base_address.
  BarrierRef.offset: BarrierRef#offset.
  BarrierRef.__iter__: BarrierRef#__iter__().
  DialectBarrierRef.update_parities: DialectBarrierRef#update_parities().
  DialectBarrierRef.arrive: DialectBarrierRef#arrive().
  CollectiveBarrierRef.wait: CollectiveBarrierRef#wait().
  CollectiveBarrierRef.wait_parity: CollectiveBarrierRef#wait_parity().
  SemaphoreRef: SemaphoreRef#
  Partition.target_bounds: Partition#target_bounds.
  Partition1D.base_offset: Partition1D#base_offset.
  Partition1D.num_chunks: Partition1D#num_chunks().
  nvvm_mbarrier_arrive_expect_tx: nvvm_mbarrier_arrive_expect_tx().
  commit_shared: commit_shared().
  MultimemReductionOp: MultimemReductionOp.
  ForResult.results: ForResult#results.
  CollectiveBarrierRef.cluster_mask: CollectiveBarrierRef#cluster_mask.
  Partition: Partition#
  WORKGROUP_NVPTX_ADDRESS_SPACE: WORKGROUP_NVPTX_ADDRESS_SPACE.
  fori.wrapper: fori().wrapper().
  block_idx: block_idx.
  memref_unsqueeze: memref_unsqueeze().
  BarrierRef._nvvm_scope: BarrierRef#_nvvm_scope().
  elements_to_bytes: elements_to_bytes().
  DYNAMIC: DYNAMIC.
  nvvm_shfl_sync: nvvm_shfl_sync().
  SemaphoreRef.ptr: SemaphoreRef#ptr.
  Partition.base_offset: Partition#base_offset.
  tile_shape: tile_shape().
  ReductionKind: ReductionKind.
  get_cluster_ptr: get_cluster_ptr().
  MultimemRef.type: MultimemRef#type().
  ForResult.result: ForResult#result().
  DialectBarrierRef.arrive_expect_tx: DialectBarrierRef#arrive_expect_tx().
  Partition.num_chunks: Partition#num_chunks().
  gpu_address_space_to_nvptx: gpu_address_space_to_nvptx().
  _debug_scalar_ty_format: _debug_scalar_ty_format().
  _3d_to_1d_idx: _3d_to_1d_idx().
  _reshape.fold_until: _reshape().fold_until().
  CollectiveBarrierRef.leader_tracked: CollectiveBarrierRef#leader_tracked.
  Partition1D: Partition1D#
  nanosleep: nanosleep().
  WARP_SIZE.WARP_SIZE: WARP_SIZE.WARP_SIZE.
  Arch: Arch#
  logger: logger.
  DYNAMIC32: DYNAMIC32.
  ForResult: ForResult#
  _is_contiguous_shape_slice: _is_contiguous_shape_slice().
  fence_release_sys: fence_release_sys().
  _MEMORY_SPACES: _MEMORY_SPACES.
  smem_cluster: smem_cluster().
  ForResult.op: ForResult#op.
  clock: clock().
  smid: smid().
  globaltimer: globaltimer().
  Partition1D.__init__: Partition1D#__init__().
  ceil_div: ceil_div().
  Arch.minor: Arch#minor.
  reduce_shape: reduce_shape().
---
# Module: [`jax/experimental/mosaic/gpu/utils.py`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py)

## Classes
### `Arch`
- def: [`jax/experimental/mosaic/gpu/utils.py:2356`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2356)
- members:
  - `major` — [`L2357`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2357)
  - `minor` — [`L2358`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2358)
- used by: [`reduce`](fragmented_array.md#FragmentedArray.reduce), [`astype`](fragmented_array.md#FragmentedArray.astype), [`redux`](utils.md#redux), [`upcast_i4_to_bf16`](fragmented_array.md#FragmentedArray.upcast_i4_to_bf16), [`get_arch`](utils.md#get_arch)

### `BarrierRef`
- def: [`jax/experimental/mosaic/gpu/utils.py:1002`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1002)
- signature: `class BarrierRef:`
- members:
  - `arrive(self, arrival_count: int = 1, can_complete: bool = True, orders_tensor_core: bool = False, predicate: ir.Value | None = None, scope: ThreadSubset = ThreadSubset.WARPGROUP)` — [`L1107`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1107)
  - `arrive_expect_tx(self, bytes: int | ir.Value, predicate: ir.Value | None = None)` — [`L1149`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1149)
  - `complete_tx(self, bytes: int | ir.Value, predicate: ir.Value | None = None)` — [`L1161`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1161)
  - `get_ptr(self)` — [`L1184`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1184)
  - `initialize(barrier_memref: ir.Value, arrival_count: int = 1)` — [`L1009`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1009)
  - `remap_to_cluster(self, dim: gpu.Dimension, idx: ir.Value)` — [`L1194`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1194)
  - `test(self, orders_tensor_core: bool = False)` — [`L1075`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1075)
  - `test_parity(self, parity, orders_tensor_core=False)` — [`L1066`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1066)
  - `update_parities(self, parities: ir.Value)` — [`L1099`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1099)
  - `wait(self, orders_tensor_core: bool = False)` — [`L1093`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1093)
  - `wait_parity(self, parity, orders_tensor_core=False)` — [`L1083`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1083)
  - `base_address` — [`L1003`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1003)
  - `num_barriers` — [`L1006`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1006)
  - `offset` — [`L1004`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1004)
  - `phases` — [`L1005`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1005)
- protocol/private: `__getitem__`[`L1037`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1037), `__iter__`[`L1030`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1030), `_nvvm_scope`[`L1061`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1061), `_ptx_scope`[`L1055`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1055)
- uses (calls/refs, reference-scoped): [`c`](utils.md#c), [`ThreadSubset`](utils.md#ThreadSubset), [`memref_ptr`](utils.md#memref_ptr), [`warpgroup_barrier`](utils.md#warpgroup_barrier), [`single_thread_predicate`](utils.md#single_thread_predicate), [`WARPGROUP`](utils.md#ThreadSubset.WARPGROUP), [`getelementptr`](utils.md#getelementptr), [`WARP`](utils.md#ThreadSubset.WARP), [`cluster_idx`](utils.md#cluster_idx), [`when`](utils.md#when), [`warp_barrier`](utils.md#warp_barrier), [`BLOCK`](utils.md#ThreadSubset.BLOCK), [`nvvm_mbarrier_arrive_expect_tx`](utils.md#nvvm_mbarrier_arrive_expect_tx), [`get_cluster_ptr`](utils.md#get_cluster_ptr)
- used by: [`async_copy`](launch_context.md#LaunchContext.async_copy), [`_construct_smem_reftree`](core.md#_construct_smem_reftree), [`_async_store_smem_op_lowering_rule`](dialect_lowering.md#_async_store_smem_op_lowering_rule), [`store_tiled_async`](fragmented_array.md#FragmentedArray.store_tiled_async), [`_mgpu_arrive_expect_tx_op_lowering_rule`](dialect_lowering.md#_mgpu_arrive_expect_tx_op_lowering_rule), [`_tcgen05_commit_arrive_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_tcgen05_commit_arrive_lowering), [`arrive`](utils.md#CollectiveBarrierRef.arrive), [`_mgpu_arrive_op_lowering_rule`](dialect_lowering.md#_mgpu_arrive_op_lowering_rule), [`from_barrier_memref`](utils.md#DialectBarrierRef.from_barrier_memref), [`barrier_ref`](utils.md#DialectBarrierRef.barrier_ref), [`as_barrier_memref`](utils.md#DialectBarrierRef.as_barrier_memref), [`try_cluster_cancel`](utils.md#try_cluster_cancel), [`initialize`](utils.md#DialectBarrierRef.initialize), [`reserve_barrier`](../../../_src/pallas/mosaic_gpu/lowering.md#ModuleContext.reserve_barrier), [`initialize`](utils.md#CollectiveBarrierRef.initialize), [`test`](utils.md#DialectBarrierRef.test), [`wait`](utils.md#DialectBarrierRef.wait), [`AnyBarrierRef`](../../../_src/pallas/mosaic_gpu/lowering.md#AnyBarrierRef), [`barrier`](utils.md#CollectiveBarrierRef.barrier), [`wait_parity`](utils.md#DialectBarrierRef.wait_parity), [`__iter__`](utils.md#DialectBarrierRef.__iter__), [`test_parity`](utils.md#DialectBarrierRef.test_parity), [`get_ptr`](utils.md#DialectBarrierRef.get_ptr), [`update_parities`](utils.md#DialectBarrierRef.update_parities), [`wait`](utils.md#CollectiveBarrierRef.wait), [`wait_parity`](utils.md#CollectiveBarrierRef.wait_parity)

### `CollectiveBarrierRef`
- def: [`jax/experimental/mosaic/gpu/utils.py:1315`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1315)
- signature: `class CollectiveBarrierRef:`
- members:
  - `arrive(self, orders_tensor_core: bool = False)` — [`L1379`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1379) — Arrives on a barrier in one or several blocks in a cluster.
  - `initialize(barrier_memref: ir.Value, arrival_count: int, dims: Sequence[gpu.Dimension | Sequence[gpu.Dimension]], cluster_shape: tuple[int, int, int], leader_tracked: bool = False)` — [`L1321`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1321)
  - `wait(self, *args, **kwargs)` — [`L1435`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1435)
  - `wait_parity(self, *args, **kwargs)` — [`L1438`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1438)
  - `barrier` — [`L1316`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1316)
  - `cluster_mask` — [`L1317`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1317)
  - `leader_tracked` — [`L1318`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1318)
- protocol/private: `__getitem__`[`L1374`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1374), `__iter__`[`L1370`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1370)
- uses (calls/refs, reference-scoped): [`c`](utils.md#c), [`ThreadSubset`](utils.md#ThreadSubset), [`warpgroup_barrier`](utils.md#warpgroup_barrier), [`single_thread_predicate`](utils.md#single_thread_predicate), [`WARPGROUP`](utils.md#ThreadSubset.WARPGROUP), [`get_ptr`](utils.md#BarrierRef.get_ptr), [`arrive`](utils.md#BarrierRef.arrive), [`initialize`](utils.md#BarrierRef.initialize), [`BarrierRef`](utils.md#BarrierRef), [`thread_idx`](utils.md#thread_idx), [`WARPGROUP_SIZE`](utils.md#WARPGROUP_SIZE.WARPGROUP_SIZE), [`num_barriers`](utils.md#BarrierRef.num_barriers), [`wait`](utils.md#BarrierRef.wait), [`cluster_collective_mask`](utils.md#cluster_collective_mask), [`wait_parity`](utils.md#BarrierRef.wait_parity)
- used by: [`_construct_smem_reftree`](core.md#_construct_smem_reftree), [`_barrier_arrive_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_barrier_arrive_lowering), [`reserve_barrier`](../../../_src/pallas/mosaic_gpu/lowering.md#ModuleContext.reserve_barrier), [`AnyBarrierRef`](../../../_src/pallas/mosaic_gpu/lowering.md#AnyBarrierRef)

### `DialectBarrierRef`
- def: [`jax/experimental/mosaic/gpu/utils.py:1207`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1207)
- signature: `class DialectBarrierRef:`
- members:
  - `arrive(self, orders_tensor_core: bool = False)` — [`L1265`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1265)
  - `arrive_expect_tx(self, bytes: int | ir.Value)` — [`L1269`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1269)
  - `as_barrier_memref(self)` — [`L1276`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1276)
  - `from_barrier_memref(cls, barrier: ir.Value)` — [`L1288`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1288) — Creates a DialectBarrierRef from a memref of a dialect barrier.
  - `get_ptr(self)` — [`L1273`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1273)
  - `initialize(barrier_memref: ir.Value, arrival_count: int, orders_tensor_core: bool)` — [`L1212`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1212)
  - `test(self, orders_tensor_core: bool = False)` — [`L1248`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1248)
  - `test_parity(self, parity, orders_tensor_core=False)` — [`L1244`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1244)
  - `update_parities(self, parities: ir.Value)` — [`L1262`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1262)
  - `wait(self, orders_tensor_core: bool = False)` — [`L1257`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1257)
  - `wait_parity(self, parity, orders_tensor_core=False)` — [`L1253`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1253)
  - `barrier_ref` — [`L1208`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1208)
  - `orders_tensor_core` — [`L1209`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1209)
- protocol/private: `__getitem__`[`L1241`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1241), `__iter__`[`L1234`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1234)
- uses (calls/refs, reference-scoped): [`c`](utils.md#c), [`memref_ptr`](utils.md#memref_ptr), [`get_ptr`](utils.md#BarrierRef.get_ptr), [`BarrierRef`](utils.md#BarrierRef), [`smem`](utils.md#smem), [`num_barriers`](utils.md#BarrierRef.num_barriers), [`test`](utils.md#BarrierRef.test), [`wait`](utils.md#BarrierRef.wait), [`phases`](utils.md#BarrierRef.phases), [`get_memref_llvm_address_space`](utils.md#get_memref_llvm_address_space), [`update_parities`](utils.md#BarrierRef.update_parities), [`wait_parity`](utils.md#BarrierRef.wait_parity), [`test_parity`](utils.md#BarrierRef.test_parity), [`base_address`](utils.md#BarrierRef.base_address), [`offset`](utils.md#BarrierRef.offset), [`WORKGROUP_NVPTX_ADDRESS_SPACE`](utils.md#WORKGROUP_NVPTX_ADDRESS_SPACE)
- used by: [`_construct_smem_reftree`](core.md#_construct_smem_reftree), [`_async_store_smem_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_async_store_smem_lowering), [`_mgpu_async_load_op_lowering_rule`](dialect_lowering.md#_mgpu_async_load_op_lowering_rule), [`_async_store_smem_op_lowering_rule`](dialect_lowering.md#_async_store_smem_op_lowering_rule), [`_mgpu_arrive_expect_tx_op_lowering_rule`](dialect_lowering.md#_mgpu_arrive_expect_tx_op_lowering_rule), [`_tcgen05_mma_lowering_wg`](../../../_src/pallas/mosaic_gpu/primitives.md#_tcgen05_mma_lowering_wg), [`_tcgen05_commit_arrive_lowering_wg`](../../../_src/pallas/mosaic_gpu/primitives.md#_tcgen05_commit_arrive_lowering_wg), [`_mgpu_arrive_op_lowering_rule`](dialect_lowering.md#_mgpu_arrive_op_lowering_rule), [`_tcgen05_commit_arrive_op_lowering_rule`](dialect_lowering.md#_tcgen05_commit_arrive_op_lowering_rule), [`_try_cluster_cancel_op_lowering_rule`](dialect_lowering.md#_try_cluster_cancel_op_lowering_rule), [`reserve_barrier`](../../../_src/pallas/mosaic_gpu/lowering.md#ModuleContext.reserve_barrier), [`_mgpu_wait_op_lowering_rule`](dialect_lowering.md#_mgpu_wait_op_lowering_rule), [`AnyBarrierRef`](../../../_src/pallas/mosaic_gpu/lowering.md#AnyBarrierRef)

### `DynamicSlice`
- def: [`jax/experimental/mosaic/gpu/utils.py:562`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L562)
- signature: `class DynamicSlice:`
- members:
  - `base` — [`L563`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L563)
  - `length` — [`L564`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L564)
- protocol/private: `__post_init__`[`L566`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L566)
- used by: [`_memref_subview_op_lowering_rule`](dialect_lowering.md#_memref_subview_op_lowering_rule), [`_ndindexer_indices`](../../../_src/pallas/mosaic_gpu/lowering.md#_ndindexer_indices), [`_gmem_slice_and_predicate`](dialect_lowering.md#_gmem_slice_and_predicate), [`_vector_extract_strided_slice_op_lowering_rule`](dialect_lowering.md#_vector_extract_strided_slice_op_lowering_rule), [`_split_gmem_slice`](../../../_src/pallas/mosaic_gpu/primitives.md#_split_gmem_slice), [`parse_indices`](utils.md#parse_indices), [`ds`](utils.md#ds), [`Index`](../../../_src/pallas/mosaic_gpu/core.md#Index)

### `ForResult`
- def: [`jax/experimental/mosaic/gpu/utils.py:353`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L353)
- signature: `class ForResult:`
- members:
  - `result(self)` — [`L358`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L358)
  - `op` — [`L354`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L354)
  - `results` — [`L355`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L355)
- used by: [`_lower_jaxpr_to_for_loop`](../../../_src/pallas/mosaic_gpu/lowering.md#_lower_jaxpr_to_for_loop), [`wrapper`](utils.md#fori.wrapper)

### `MultimemRef`
- def: [`jax/experimental/mosaic/gpu/utils.py:211`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L211)
- signature: `class MultimemRef:`
- members:
  - `store(self, value: ir.Value, indices: Sequence[ir.Value])` — [`L218`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L218)
  - `type(self)` — [`L215`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L215)
  - `ref` — [`L212`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L212)
- uses (calls/refs, reference-scoped): [`memref_ptr`](utils.md#memref_ptr), [`memref_slice`](utils.md#memref_slice), [`multimem_store`](utils.md#multimem_store)
- used by: [`async_copy`](launch_context.md#LaunchContext.async_copy), [`_semaphore_signal_multicast_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_semaphore_signal_multicast_lowering), [`_multimem_store_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_multimem_store_lowering_rule), [`store_untiled`](fragmented_array.md#FragmentedArray.store_untiled), [`to_remote_multicast`](launch_context.md#LaunchContext.to_remote_multicast), [`store_tiled_async`](fragmented_array.md#FragmentedArray.store_tiled_async), [`_multimem_load_reduce_lowering_rule_wg`](../../../_src/pallas/mosaic_gpu/primitives.md#_multimem_load_reduce_lowering_rule_wg), [`store_tiled`](fragmented_array.md#FragmentedArray.store_tiled), [`_vector_store_op_lowering_rule`](dialect_lowering.md#_vector_store_op_lowering_rule), [`load_reduce_untiled`](fragmented_array.md#FragmentedArray.load_reduce_untiled), [`init_tma_desc`](launch_context.md#LaunchContext.init_tma_desc), [`_multimem_load_reduce_op_lowering_rule`](dialect_lowering.md#_multimem_load_reduce_op_lowering_rule)

### `Partition`
- def: [`jax/experimental/mosaic/gpu/utils.py:1568`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1568)
- signature: `class Partition:`
- members:
  - `get_base(self, *source_coords: ir.Value | int)` — [`L1629`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1629)
  - `num_chunks(self)` — [`L1619`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1619)
  - `target_block_shape(self)` — [`L1623`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1623)
  - `base_offset` — [`L1572`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1572)
  - `partition` — [`L1571`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1571)
  - `source_bounds` — [`L1569`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1569)
  - `target_bounds` — [`L1570`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1570)
- protocol/private: `__init__`[`L1574`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1574)
- uses (calls/refs, reference-scoped): [`c`](utils.md#c)
- used by: [`refine`](utils.md#Partition1D.refine), [`partition`](utils.md#Partition1D.partition), [`get_base`](utils.md#Partition1D.get_base), [`base_offset`](utils.md#Partition1D.base_offset), [`num_chunks`](utils.md#Partition1D.num_chunks)

### `Partition1D`
- def: [`jax/experimental/mosaic/gpu/utils.py:1644`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1644)
- signature: `class Partition1D:`
- members:
  - `get_base(self, source_coords: ir.Value)` — [`L1673`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1673)
  - `num_chunks(self)` — [`L1670`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1670)
  - `refine(self, *, chunk: ir.Value | None = None, num_chunks: int | None = None, chunk_size: int | None = None)` — [`L1676`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1676)
  - `base_offset` — [`L1655`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1655)
  - `partition` — [`L1645`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1645)
- protocol/private: `__init__`[`L1647`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1647)
- uses (calls/refs, reference-scoped): [`get_base`](utils.md#Partition.get_base), [`target_block_shape`](utils.md#Partition.target_block_shape), [`source_bounds`](utils.md#Partition.source_bounds), [`Partition`](utils.md#Partition)

### `SemaphoreRef`
- def: [`jax/experimental/mosaic/gpu/utils.py:1443`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1443)
- signature: `class SemaphoreRef:`
- members:
  - `signal(self, value: ir.Value | int, predicate: ir.Value | None = None, relaxed: bool = False, memory_scope: Literal["sys", "gpu"] = "sys")` — [`L1446`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1446)
  - `signal_multimem(ptr, value, predicate: ir.Value | None = None)` — [`L1473`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1473)
  - `wait(self, value: ir.Value | int = 1, *, decrement: bool = True, scope: ThreadSubset = ThreadSubset.WARPGROUP, memory_scope: Literal["sys", "gpu"] = "sys")` — [`L1493`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1493)
  - `ptr` — [`L1444`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1444)
- uses (calls/refs, reference-scoped): [`c`](utils.md#c), [`ThreadSubset`](utils.md#ThreadSubset), [`warpgroup_barrier`](utils.md#warpgroup_barrier), [`single_thread_predicate`](utils.md#single_thread_predicate), [`WARPGROUP`](utils.md#ThreadSubset.WARPGROUP), [`WARP`](utils.md#ThreadSubset.WARP), [`single_thread`](utils.md#single_thread), [`warp_barrier`](utils.md#warp_barrier)
- used by: [`_semaphore_signal_multicast_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_semaphore_signal_multicast_lowering), [`_semaphore_signal_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_semaphore_signal_lowering_rule), [`_semaphore_wait_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_semaphore_wait_lowering_rule)

### `ThreadSubset`
- def: [`jax/experimental/mosaic/gpu/utils.py:441`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L441) — documented in [jax-experimental-mosaic-gpu-utils](../../../../../concepts/jax-experimental-mosaic-gpu-utils.md)
- members:
  - `BLOCK` — [`L444`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L444)
  - `WARP` — [`L442`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L442)
  - `WARPGROUP` — [`L443`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L443)
- used by: [`async_copy`](launch_context.md#LaunchContext.async_copy), [`lower_jaxpr_to_module`](../../../_src/pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_module), [`body`](../../../_src/pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_module.body), [`_semaphore_wait_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_semaphore_wait_lowering_rule), [`_barrier_arrive_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_barrier_arrive_lowering), [`_debug_print_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_debug_print_lowering_rule), [`_mgpu_async_load_op_lowering_rule`](dialect_lowering.md#_mgpu_async_load_op_lowering_rule), [`_wait_smem_to_gmem_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_wait_smem_to_gmem_lowering), [`_mgpu_arrive_expect_tx_op_lowering_rule`](dialect_lowering.md#_mgpu_arrive_expect_tx_op_lowering_rule), [`async_prefetch`](launch_context.md#LaunchContext.async_prefetch), [`arrive`](utils.md#CollectiveBarrierRef.arrive), [`single_thread_predicate`](utils.md#single_thread_predicate), [`_lowering_context`](dialect_lowering.md#_lowering_context), [`_mgpu_arrive_op_lowering_rule`](dialect_lowering.md#_mgpu_arrive_op_lowering_rule), [`single_lane_predicate`](dialect_lowering.md#LoweringContext.single_lane_predicate), [`arrive`](utils.md#BarrierRef.arrive), [`_mgpu_warp_map_op_lowering_rule`](dialect_lowering.md#_mgpu_warp_map_op_lowering_rule), [`lower_op`](dialect_lowering.md#LoweringContext.lower_op), [`wait`](utils.md#SemaphoreRef.wait), [`initialize`](utils.md#BarrierRef.initialize), [`try_cluster_cancel`](utils.md#try_cluster_cancel), [`thread_semantics`](dialect_lowering.md#LoweringContext.thread_semantics), [`signal`](utils.md#SemaphoreRef.signal), [`await_async_copy`](launch_context.md#LaunchContext.await_async_copy), [`debug_print`](utils.md#debug_print), [`single_thread`](utils.md#single_thread), [`signal_multimem`](utils.md#SemaphoreRef.signal_multimem), [`_ONCE_PER`](utils.md#_ONCE_PER._ONCE_PER)

## Functions
- `_3d_to_1d_idx(dim_idx_fn, dim_size_fn)` — [`L402`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L402)
- `_debug_scalar_ty_format(arg)` — [`L149`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L149)
- `_is_contiguous_shape_slice(ref_ty: ir.MemRefType, dim_slice: slice = slice(None))` — [`L610`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L610)
- `_reshape(ref: ir.Value, sh0: list[int], sh1: list[int])` — [`L629`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L629) — Reshapes using only "parallel" folds/unfolds.
- `_vector_concat_rec(vectors: Sequence[ir.Value[ir.VectorType]])` — [`L2052`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2052)
- `_warp_bcast(val, lane_idx=0)` — [`L417`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L417)
- `bitcast(x: ir.Value, new_type: ir.Type)` — [`L1973`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1973) — documented in [jax-experimental-mosaic-gpu-utils](../../../../../concepts/jax-experimental-mosaic-gpu-utils.md)
- `bitwidth(ty: ir.Type)` — [`L554`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L554) — documented in [jax-experimental-mosaic-gpu-utils](../../../../../concepts/jax-experimental-mosaic-gpu-utils.md)
- `bitwidth_impl(ty: ir.Type)` — [`L535`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L535)
- `bytewidth(ty: ir.Type)` — [`L529`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L529)
- `c(val: int | float, ty)` — [`L135`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L135) — documented in [jax-experimental-mosaic-gpu-utils](../../../../../concepts/jax-experimental-mosaic-gpu-utils.md)
- `ceil_div(x: int, y: int)` — [`L2014`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2014)
- `clock()` — [`L495`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L495)
- `cluster_collective_mask(cluster_shape: tuple[int, int, int], collective: Sequence[gpu.Dimension] | gpu.Dimension, leader_tracked: bool = False)` — [`L1798`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1798)
- `cluster_idx(dim: gpu.Dimension | Sequence[gpu.Dimension] | None = None, dim_idx: ir.Value | Sequence[ir.Value] | None = None)` — [`L2273`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2273) — Returns the linear index of a block within a subset of the cluster spanned by the given dimensions.
- `commit_shared()` — [`L977`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L977)
- `debug_print(fmt, *args, uniform=True, scope=None)` — [`L169`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L169)
- `dtype_to_ir_type(dtype: jax.typing.DTypeLike)` — [`L1835`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1835)
- `dyn_dot(x, y)` — [`L1866`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1866)
- `elements_to_bytes(offset: ir.Value, element_bitwidth: int)` — [`L2343`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2343) — Convert an element-based linear offset to a byte-based offset.
- `fence_release_sys()` — [`L1558`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1558)
- `fold_until(shape, off, target)` — [`L639`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L639)
- `fori(bound, carrys)` — [`L364`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L364)
- `get_arch()` — [`L2361`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2361)
- `get_cluster_ptr(ptr: ir.Value, cluster_block: ir.Value, generic: bool = True)` — [`L2303`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2303)
- `get_cluster_ref(ref: ir.Value, dim: gpu.Dimension, idx: ir.Value, generic: bool = True)` — [`L2315`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2315)
- `get_contiguous_strides(xs)` — [`L126`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L126)
- `get_memref_llvm_address_space(memref_ty: ir.MemRefType)` — [`L1740`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1740)
- `getelementptr(ptr: ir.Value, indices: Sequence[ir.Value | int], dtype: ir.Type)` — [`L1851`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1851)
- `globaltimer(kind: Literal["low", "high"] | None = None)` — [`L507`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L507)
- `gpu_address_space_to_nvptx(address_space: gpu.AddressSpace)` — [`L61`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L61)
- `is_cluster_smem_ref(ref: ir.Value | ir.Type)` — [`L2181`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2181) — Returns true if the input mem ref or memref type points to cluster SMEM.
- `is_known_divisible(value: ir.Value, divisor: int, max_depth=10)` — [`L2070`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2070) — Returns True if the value is statically known to be divisible by the divisor.
- `is_memref_transposed(ref: ir.MemRefType)` — [`L892`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L892)
- `is_signed(dtype: jax.typing.DTypeLike)` — [`L1843`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1843)
- `is_smem_ref(ref: ir.Value | ir.Type)` — [`L2155`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2155) — Returns true if the input mem ref or memref type points to SMEM.
- `is_tmem_ref(ref: ir.Value | ir.Type)` — [`L2168`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2168) — Returns true if the input mem ref or memref type points to TMEM.
- `memref_fold(ref: ir.Value, dim, fold_rank)` — [`L771`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L771)
- `memref_ptr(memref_arg)` — [`L1748`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1748)
- `memref_reshape(ref: ir.Value, shape: tuple[int, ...])` — [`L692`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L692)
- `memref_slice(ref: ir.Value[ir.MemRefType], index)` — [`L576`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L576)
- `memref_transpose(ref: ir.Value, permutation: Sequence[int])` — [`L902`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L902)
- `memref_unfold(ref: ir.Value, dim, factors)` — [`L825`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L825) — Unfolds dim into two dimensions, the size of leading one given be major_factor.
- `memref_unsqueeze(ref: ir.Value, dim)` — [`L867`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L867) — Inserts a singleton dimension.
- `multimem_load_reduce(ty: ir.Type, ptr: ir.Value, reduction: MultimemReductionOp, is_signed: bool | None = None)` — [`L253`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L253)
- `multimem_store(ptr: ir.Value, value: ir.Value)` — [`L223`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L223)
- `nanosleep(nanos: ir.Value)` — [`L2245`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2245) — Sleeps the current thread for the given number of nanoseconds.
- `nvvm_mbarrier_arrive_expect_tx(barrier: ir.Value, expect_tx: ir.Value, predicate: ir.Value | None = None, scope: nvvm.MemScopeKind | None = None)` — [`L2256`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2256)
- `nvvm_shfl_sync(ty, *args)` — [`L53`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L53)
- `pack_array(values)` — [`L113`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L113)
- `parse_indices(index, shape: Sequence[int], *, check_oob: bool = True)` — [`L916`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L916)
- `prmt(high: ir.Value, low: ir.Value, permutation: ir.Value)` — [`L1956`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1956)
- `ptr_as_memref(ptr, memref_ty: ir.MemRefType)` — [`L76`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L76)
- `query_cluster_cancel(result_ref)` — [`L2213`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2213) — Decodes the response of `try_cluster_cancel`.
- `reduce_shape(shape: Sequence[int], axes: Sequence[int], keep_dims: bool = False)` — [`L2380`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2380)
- `redux(x: ir.Value, mask: ir.Value, kind: ReductionKind)` — [`L1928`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1928)
- `shfl_bfly(x: ir.Value, distance: int | ir.Value)` — [`L1871`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1871)
- `single_thread(scope: ThreadSubset = ThreadSubset.BLOCK)` — [`L470`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L470) — Runs the context only from a single thread.
- `single_thread_predicate(scope: ThreadSubset = ThreadSubset.BLOCK)` — [`L451`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L451) — Returns a predicate that selects a single thread.
- `smem()` — [`L2140`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2140) — Returns the attribute for the SMEM memory space.
- `smem_cluster()` — [`L2150`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2150) — Returns the attribute for the cluster SMEM memory space.
- `smid()` — [`L502`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L502)
- `tile_shape(shape, tiling)` — [`L1691`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1691)
- `tmem()` — [`L2145`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2145) — Returns the attribute for the TMEM memory space.
- `try_cluster_cancel(result_ref, barrier: BarrierRef, predicate: ir.Value | None = None)` — [`L2191`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2191) — Atomically cancels a pending cluster launch.
- `vector_concat(vectors: Sequence[ir.Value[ir.VectorType]])` — [`L2033`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2033)
- `vector_slice(v: ir.Value, s: slice)` — [`L2018`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L2018)
- `warp_barrier()` — [`L997`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L997)
- `warp_idx(sync=True)` — [`L425`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L425)
- `warp_tree_reduce(value, op, group_size)` — [`L1711`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1711) — Reduce a value across the warpgroup.
- `warpgroup_barrier()` — [`L984`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L984)
- `warpgroup_idx(sync=True)` — [`L433`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L433)
- `when(cond)` — [`L396`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L396)
- `wrapper(f)` — [`L371`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L371)

## Module values
- `DYNAMIC` — [`L47`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L47)
- `DYNAMIC32` — [`L48`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L48)
- `MBARRIER_BYTES` — [`L49`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L49)
- `MultimemReductionOp` — [`L250`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L250)
- `ReductionKind` — [`L1925`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1925)
- `WARPGROUP_SIZE` — [`L45`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L45)
- `WARPS_IN_WARPGROUP` — [`L46`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L46)
- `WARP_SIZE` — [`L44`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L44)
- `WORKGROUP_NVPTX_ADDRESS_SPACE` — [`L71`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L71)
- `_MEMORY_SPACES` — [`L1737`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L1737)
- `_ONCE_PER` — [`L448`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L448)
- `block_idx` — [`L414`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L414)
- `ds` — [`L573`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L573)
- `logger` — [`L42`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L42)
- `thread_idx` — [`L413`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/utils.py#L413)

