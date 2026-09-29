---
title: 'Module: jax/experimental/mosaic/gpu/core.py'
type: catalog
provenance: extracted
module: jax/experimental/mosaic/gpu/core.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.mosaic.gpu.core`/
symbols:
  LoweringSemantics: LoweringSemantics#
  LoweringSemantics.Warpgroup: LoweringSemantics#Warpgroup.
  LoweringSemantics.Lane: LoweringSemantics#Lane.
  _construct_smem_reftree: _construct_smem_reftree().
  _launch: _launch().
  _kernel_to_module: _kernel_to_module().
  _mosaic_gpu_lowering_rule: _mosaic_gpu_lowering_rule().
  _smem_tree_size: _smem_tree_size().
  _lower_as_gpu_kernel.main: _lower_as_gpu_kernel().main().
  _lower_as_gpu_kernel: _lower_as_gpu_kernel().
  as_gpu_kernel: as_gpu_kernel().
  as_torch_gpu_kernel: as_torch_gpu_kernel().
  _TMEMDialectAlloc.alloc: _TMEMDialectAlloc#alloc().
  _construct_smem_reftree.barrier_memref: _construct_smem_reftree().barrier_memref().
  TMEM.__post_init__: TMEM#__post_init__().
  _count_buffer_bytes: _count_buffer_bytes().
  mosaic_gpu_p: mosaic_gpu_p.
  as_gpu_kernel.prof_kernel: as_gpu_kernel().prof_kernel().
  _lower_as_gpu_kernel._shape_to_ref_ty: _lower_as_gpu_kernel()._shape_to_ref_ty().
  ShapeTree: ShapeTree.
  Union: Union#
  RUNTIME_PATH: RUNTIME_PATH.
  files: files.
  _slice_smem: _slice_smem().
  _TMEMAlloc.alloc: _TMEMAlloc#alloc().
  _TMEMAlloc.dealloc: _TMEMAlloc#dealloc().
  as_gpu_kernel.kernel: as_gpu_kernel().kernel().
  _infer_arch: _infer_arch().
  c: c.
  libdevice_path: libdevice_path.
  is_single_process_multi_device_topology: is_single_process_multi_device_topology().
  Barrier: Barrier#
  ClusterBarrier: ClusterBarrier#
  as_gpu_kernel.bind: as_gpu_kernel().bind().
  solib_path: solib_path.
  cuda_root: cuda_root.
  _SMEM_SIZE_BOUND: _SMEM_SIZE_BOUND.
  _TMEMDialectAlloc.tmem_ref: _TMEMDialectAlloc#tmem_ref.
  _as_torch_gpu_kernel: _as_torch_gpu_kernel().
  _as_torch_gpu_kernel.apply: _as_torch_gpu_kernel().apply().
  _compile_as_torch_gpu_kernel: _compile_as_torch_gpu_kernel().
  PYTHON_RUNFILES: PYTHON_RUNFILES.
  TMEM: TMEM#
  _TMEMAlloc.collective: _TMEMAlloc#collective.
  _has_communication: _has_communication().
  Union.members: Union#members.
  as_gpu_kernel.prof_kernel.dump_profile: as_gpu_kernel().prof_kernel().dump_profile().
  is_nvshmem_available: is_nvshmem_available().
  TMABarrier: TMABarrier#
  _TMEMAlloc: _TMEMAlloc#
  _TMEMDialectAlloc: _TMEMDialectAlloc#
  _as_torch_gpu_kernel.as_torch_dtype: _as_torch_gpu_kernel().as_torch_dtype().
  artificial_shared_memory_limit: artificial_shared_memory_limit().
  _mosaic_gpu_abstract_eval: _mosaic_gpu_abstract_eval().
  Union.__iter__: Union#__iter__().
  Barrier.__post_init__: Barrier#__post_init__().
  _TMEMDialectAlloc.dealloc: _TMEMDialectAlloc#dealloc().
  Barrier.arrival_count: Barrier#arrival_count.
  ClusterBarrier.leader_tracked: ClusterBarrier#leader_tracked.
  TMEM.shape: TMEM#shape.
  TMEM.dtype: TMEM#dtype.
  TMEM.layout: TMEM#layout.
  _TMEMAlloc.addr_ref: _TMEMAlloc#addr_ref.
  _TMEMAlloc.num_cols: _TMEMAlloc#num_cols.
  _TMEMDialectAlloc.shape: _TMEMDialectAlloc#shape.
  _TMEMDialectAlloc.packing: _TMEMDialectAlloc#packing.
  _run_serde_pass: _run_serde_pass().
  as_gpu_kernel._check_args: as_gpu_kernel()._check_args().
  KNOWN_KERNELS.KNOWN_KERNELS: KNOWN_KERNELS.KNOWN_KERNELS.
  BAZEL_TEST: BAZEL_TEST.
  FWD_COMPAT_IR_VERSION: FWD_COMPAT_IR_VERSION.
  RefTree: RefTree.
  ClusterBarrier.orders_tensor_core: ClusterBarrier#orders_tensor_core.
  TMEM.collective: TMEM#collective.
  TMEM.packing: TMEM#packing.
  _TMEMDialectAlloc.addr_ref: _TMEMDialectAlloc#addr_ref.
  _TMEMDialectAlloc.dtype: _TMEMDialectAlloc#dtype.
  _TMEMDialectAlloc.collective: _TMEMDialectAlloc#collective.
  _construct_smem_reftree.ref_tree_thunk: _construct_smem_reftree().ref_tree_thunk().
  _declare_runtime_functions: _declare_runtime_functions().
  _compile_as_torch_gpu_kernel.launch: _compile_as_torch_gpu_kernel().launch().
  root: root.
  TMABarrier.num_barriers: TMABarrier#num_barriers.
  Barrier.num_barriers: Barrier#num_barriers.
  Barrier.orders_tensor_core: Barrier#orders_tensor_core.
  ClusterBarrier.collective_dims: ClusterBarrier#collective_dims.
  ClusterBarrier.arrival_count: ClusterBarrier#arrival_count.
  ClusterBarrier.num_barriers: ClusterBarrier#num_barriers.
---
# Module: [`jax/experimental/mosaic/gpu/core.py`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py)

## Classes
### `Barrier`
- def: [`jax/experimental/mosaic/gpu/core.py:321`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L321)
- signature: `class Barrier:`
- members:
  - `arrival_count` — [`L322`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L322)
  - `num_barriers` — [`L323`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L323)
  - `orders_tensor_core` — [`L324`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L324)
- protocol/private: `__post_init__`[`L326`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L326)
- used by: [`_construct_smem_reftree`](core.md#_construct_smem_reftree), [`_smem_tree_size`](core.md#_smem_tree_size), [`_get_barrier`](../../../_src/pallas/mosaic_gpu/lowering.md#_get_barrier), [`reserve_barrier`](../../../_src/pallas/mosaic_gpu/lowering.md#ModuleContext.reserve_barrier), [`AnyBarrier`](../../../_src/pallas/mosaic_gpu/lowering.md#AnyBarrier)

### `ClusterBarrier`
- def: [`jax/experimental/mosaic/gpu/core.py:333`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L333)
- signature: `class ClusterBarrier:`
- members:
  - `arrival_count` — [`L335`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L335)
  - `collective_dims` — [`L334`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L334)
  - `leader_tracked` — [`L339`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L339)
  - `num_barriers` — [`L336`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L336)
  - `orders_tensor_core` — [`L338`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L338)
- used by: [`_construct_smem_reftree`](core.md#_construct_smem_reftree), [`_get_cluster_barrier`](../../../_src/pallas/mosaic_gpu/lowering.md#_get_cluster_barrier), [`_smem_tree_size`](core.md#_smem_tree_size), [`reserve_barrier`](../../../_src/pallas/mosaic_gpu/lowering.md#ModuleContext.reserve_barrier), [`AnyBarrier`](../../../_src/pallas/mosaic_gpu/lowering.md#AnyBarrier)

### `LoweringSemantics`
- def: [`jax/experimental/mosaic/gpu/core.py:363`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L363) — documented in [jax-experimental-mosaic-gpu-core](../../../../../concepts/jax-experimental-mosaic-gpu-core.md)
- members:
  - `Lane` — [`L366`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L366) — documented in [jax-experimental-mosaic-gpu-core](../../../../../concepts/jax-experimental-mosaic-gpu-core.md)
  - `Warpgroup` — [`L367`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L367) — documented in [jax-experimental-mosaic-gpu-core](../../../../../concepts/jax-experimental-mosaic-gpu-core.md)
- used by: [`register_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#register_lowering_rule), [`_handle_transforms`](../../../_src/pallas/mosaic_gpu/lowering.md#_handle_transforms), [`lower_jaxpr_to_module`](../../../_src/pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_module), [`_run_scoped_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_run_scoped_lowering_rule), [`body`](../../../_src/pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_module.body), [`_copy_gmem_to_smem_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_copy_gmem_to_smem_lowering), [`_swap_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_swap_lowering_rule), [`_get_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_get_lowering_rule), [`mixed_matmul_kernel`](../../pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.md#mixed_matmul_kernel), [`_semaphore_signal_multicast_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_semaphore_signal_multicast_lowering), [`_construct_smem_reftree`](core.md#_construct_smem_reftree), [`matmul_kernel`](../../pallas/ops/gpu/blackwell_matmul_mgpu.md#matmul_kernel), [`_copy_smem_to_gmem_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_copy_smem_to_gmem_lowering), [`all_gather_lhs_matmul`](../../pallas/ops/gpu/collective_matmul_mgpu.md#all_gather_lhs_matmul), [`_async_store_smem_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_async_store_smem_lowering), [`_launch`](core.md#_launch), [`_semaphore_signal_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_semaphore_signal_lowering_rule), [`_broadcast_in_dim_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_broadcast_in_dim_lowering_rule), [`_semaphore_wait_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_semaphore_wait_lowering_rule), [`_axis_index_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_axis_index_rule), [`_core_map_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_core_map_lowering_rule), [`_mpmd_map_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_mpmd_map_lowering_rule), [`_multimem_store_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_multimem_store_lowering_rule), [`_run_state_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_run_state_lowering_rule), [`lowering_semantics`](../../../_src/pallas/mosaic_gpu/lowering.md#ModuleContext.lowering_semantics), [`f_pred`](../../../_src/pallas/mosaic_gpu/lowering.md#f_pred), [`_barrier_arrive_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_barrier_arrive_lowering), [`_debug_print_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_debug_print_lowering_rule), [`attention_with_pipeline_emitter`](../../pallas/ops/gpu/attention_mgpu.md#attention_with_pipeline_emitter), [`_cond_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_cond_lowering_rule), [`_jaxpr_call_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_jaxpr_call_lowering_rule), [`_inline_mgpu_flat_transformed_args`](../../../_src/pallas/mosaic_gpu/primitives.md#_inline_mgpu_flat_transformed_args), [`_async_prefetch_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_async_prefetch_lowering), [`_multimem_load_reduce_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_multimem_load_reduce_lowering_rule), [`_convert_element_type_lowering_rule_wg`](../../../_src/pallas/mosaic_gpu/lowering.md#_convert_element_type_lowering_rule_wg), [`_select_n_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_select_n_lowering_rule), [`_while_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_while_lowering_rule), [`query_cluster_cancel_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#query_cluster_cancel_lowering), [`ragged_dot_kernel`](../../pallas/ops/gpu/blackwell_ragged_dot_mgpu.md#ragged_dot_kernel), [`_scan_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_scan_lowering_rule)  (+117 more)

### `TMABarrier`
- def: [`jax/experimental/mosaic/gpu/core.py:317`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L317)
- signature: `class TMABarrier:`
- members:
  - `num_barriers` — [`L318`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L318)
- used by: [`_construct_smem_reftree`](core.md#_construct_smem_reftree), [`_smem_tree_size`](core.md#_smem_tree_size)

### `TMEM`
- def: [`jax/experimental/mosaic/gpu/core.py:342`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L342)
- signature: `class TMEM:`
- members:
  - `collective` — [`L347`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L347)
  - `dtype` — [`L344`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L344)
  - `layout` — [`L346`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L346)
  - `packing` — [`L348`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L348)
  - `shape` — [`L343`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L343)
- protocol/private: `__post_init__`[`L350`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L350)
- uses (calls/refs, reference-scoped): [`bitwidth`](utils.md#bitwidth), [`dtype_to_ir_type`](utils.md#dtype_to_ir_type)
- used by: [`lower_jaxpr_to_module`](../../../_src/pallas/mosaic_gpu/lowering.md#lower_jaxpr_to_module), [`_construct_smem_reftree`](core.md#_construct_smem_reftree), [`_smem_tree_size`](core.md#_smem_tree_size)

### `Union`
- def: [`jax/experimental/mosaic/gpu/core.py:310`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L310)
- signature: `class Union:`
- members:
  - `members` — [`L311`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L311)
- protocol/private: `__iter__`[`L313`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L313)
- used by: [`_construct_smem_reftree`](core.md#_construct_smem_reftree), [`_launch`](core.md#_launch), [`_kernel_to_module`](core.md#_kernel_to_module), [`_smem_tree_size`](core.md#_smem_tree_size), [`_lower_as_gpu_kernel`](core.md#_lower_as_gpu_kernel), [`as_gpu_kernel`](core.md#as_gpu_kernel), [`as_torch_gpu_kernel`](core.md#as_torch_gpu_kernel)

### `_TMEMAlloc`
- def: [`jax/experimental/mosaic/gpu/core.py:371`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L371)
- signature: `class _TMEMAlloc:`
- members:
  - `alloc(self)` — [`L376`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L376) — Allocates TMEM and returns the number of columns allocated.
  - `dealloc(self)` — [`L383`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L383)
  - `addr_ref` — [`L372`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L372)
  - `collective` — [`L374`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L374)
  - `num_cols` — [`L373`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L373)
- used by: [`_construct_smem_reftree`](core.md#_construct_smem_reftree), [`_launch`](core.md#_launch)

### `_TMEMDialectAlloc`
- def: [`jax/experimental/mosaic/gpu/core.py:391`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L391)
- signature: `class _TMEMDialectAlloc:`
- members:
  - `alloc(self)` — [`L399`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L399) — Allocates TMEM and returns the number of columns allocated.
  - `dealloc(self)` — [`L415`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L415)
  - `addr_ref` — [`L392`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L392)
  - `collective` — [`L396`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L396)
  - `dtype` — [`L394`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L394)
  - `packing` — [`L395`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L395)
  - `shape` — [`L393`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L393)
  - `tmem_ref` — [`L397`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L397)
- uses (calls/refs, reference-scoped): [`tmem`](utils.md#tmem)
- used by: [`_construct_smem_reftree`](core.md#_construct_smem_reftree), [`_launch`](core.md#_launch)

## Functions
- `_as_torch_gpu_kernel(module_asm: bytes, in_shape: Iterable[object], out_shape: Iterable[object], inout_shape: Iterable[object] = (), *, unwrap_output_tuple: bool = False, _prepare_args=None, _prepare_results=None)` — [`L1210`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L1210)
- `_check_args(*args)` — [`L1060`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L1060)
- `_compile_as_torch_gpu_kernel(module_asm: bytes)` — [`L1161`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L1161)
- `_construct_smem_reftree(cluster_shape: tuple[int, int, int], dynamic_smem: ir.Value, smem_buffers: ShapeTree, tmem_allocs: list[_TMEMAlloc | _TMEMDialectAlloc], lowering_semantics: LoweringSemantics, dynamic_smem_offset: int = 0)` — [`L433`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L433) — documented in [jax-experimental-mosaic-gpu-core](../../../../../concepts/jax-experimental-mosaic-gpu-core.md)
- `_count_buffer_bytes(shape_dtype: jax.ShapeDtypeStruct)` — [`L359`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L359)
- `_declare_runtime_functions()` — [`L966`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L966) — Declares the runtime functions that can be used by the generated code.
- `_has_communication(module, **_)` — [`L177`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L177)
- `_infer_arch()` — [`L787`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L787)
- `_kernel_to_module(body, grid: tuple[int, int, int], block: tuple[int, int, int], in_shape, out_shape, smem_scratch_shape: ShapeTree | Union[ShapeTree], prof_spec: profiler.ProfilerSpec | None = None, cluster: tuple[int, int, int] = (1, 1, 1), module_name: str = "unknown", kernel_name: str | None = None, thread_semantics: LoweringSemantics = LoweringSemantics.Lane, inout_shape=())` — [`L977`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L977)
- `_launch(token, grid: tuple[int, int, int], cluster: tuple[int, int, int], block: tuple[int, int, int], smem_buffers: ShapeTree | Union[ShapeTree], lowering_semantics: LoweringSemantics, module: ir.Module, inout_buffers_ptr: ir.Value, profiler_spec: profiler.ProfilerSpec | None = None, maybe_prof_buffer: ir.Value | None = None, device_collective_metadata: ir.Value | None = None, num_peers: int = 0, num_params: int = 0)` — [`L601`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L601)
- `_lower_as_gpu_kernel(body, grid: tuple[int, int, int], cluster: tuple[int, int, int], block: tuple[int, int, int], in_shapes: tuple[Any, ...], out_shape, inout_shape, smem_scratch_shape: ShapeTree | Union[ShapeTree], lowering_semantics: LoweringSemantics, module_name: str, kernel_name: str, prof_spec: profiler.ProfilerSpec | None = None, jax_mesh: mesh_lib.Mesh | None = None, base_loc: ir.Location | None = None, uses_pdl: bool = False)` — [`L809`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L809)
- `_mosaic_gpu_abstract_eval(*_, module, out_types, inout_types)` — [`L169`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L169)
- `_mosaic_gpu_lowering_rule(ctx, *args, module, out_types, inout_types, input_output_aliases: tuple[tuple[int, int], ...] = (), use_custom_barrier: bool = False)` — [`L192`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L192)
- `_run_serde_pass(module: ir.Module, *, serialize: bool, ir_version: int | None = None)` — [`L943`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L943)
- `_shape_to_ref_ty(shape: jax.ShapeDtypeStruct)` — [`L831`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L831)
- `_slice_smem(result: ir.Type, smem_base: ir.Value, offset: int, lowering_semantics: LoweringSemantics)` — [`L420`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L420)
- `_smem_tree_size(smem_buffers: ShapeTree)` — [`L569`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L569)
- `apply(*args)` — [`L1232`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L1232)
- `artificial_shared_memory_limit(limit)` — [`L71`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L71)
- `as_gpu_kernel(body, grid: tuple[int, int, int], block: tuple[int, int, int], in_shape, out_shape, smem_scratch_shape: ShapeTree | Union[ShapeTree], prof_spec: profiler.ProfilerSpec | None = None, cluster: tuple[int, int, int] = (1, 1, 1), module_name: str = "unknown", kernel_name: str | None = None, ir_version: int | None = None, thread_semantics: LoweringSemantics = LoweringSemantics.Lane, inout_shape=())` — [`L1039`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L1039)
- `as_torch_dtype(dtype)` — [`L1228`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L1228)
- `as_torch_gpu_kernel(body, grid: tuple[int, int, int], block: tuple[int, int, int], in_shape, out_shape, smem_scratch_shape: ShapeTree | Union[ShapeTree], prof_spec: profiler.ProfilerSpec | None = None, cluster: tuple[int, int, int] = (1, 1, 1), module_name: str = "unknown", kernel_name: str | None = None, thread_semantics: LoweringSemantics = LoweringSemantics.Lane, inout_shape=())` — [`L1114`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L1114)
- `barrier_memref(num_barriers: int)` — [`L451`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L451)
- `bind(*args)` — [`L1082`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L1082)
- `dump_profile(prof_buffer)` — [`L1095`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L1095)
- `is_nvshmem_available()` — [`L140`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L140)
- `is_single_process_multi_device_topology()` — [`L159`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L159)
- `kernel(*args)` — [`L1107`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L1107)
- `launch(arg_ptrs, device)` — [`L1193`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L1193)
- `main(token_ptr, buffers)` — [`L869`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L869)
- `prof_kernel(*args)` — [`L1092`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L1092)
- `ref_tree_thunk()` — [`L559`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L559)

## Module values
- `BAZEL_TEST` — [`L66`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L66)
- `FWD_COMPAT_IR_VERSION` — [`L81`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L81)
- `KNOWN_KERNELS` — [`L189`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L189)
- `PYTHON_RUNFILES` — [`L65`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L65)
- `RUNTIME_PATH` — [`L86`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L86)
- `RefTree` — [`L306`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L306)
- `ShapeTree` — [`L305`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L305)
- `_SMEM_SIZE_BOUND` — [`L68`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L68)
- `c` — [`L83`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L83)
- `cuda_root` — [`L63`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L63)
- `files` — [`L117`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L117)
- `libdevice_path` — [`L107`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L107)
- `mosaic_gpu_p` — [`L164`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L164)
- `root` — [`L117`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L117)
- `solib_path` — [`L112`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/core.py#L112)

