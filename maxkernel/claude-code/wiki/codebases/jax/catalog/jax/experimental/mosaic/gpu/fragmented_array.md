---
title: 'Module: jax/experimental/mosaic/gpu/fragmented_array.py'
type: catalog
provenance: extracted
module: jax/experimental/mosaic/gpu/fragmented_array.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.mosaic.gpu.fragmented_array`/
symbols:
  FragmentedArray: FragmentedArray#
  FragmentedArray.mlir_dtype: FragmentedArray#mlir_dtype().
  FragmentedArray.layout: FragmentedArray#layout.
  FragmentedArray.reduce: FragmentedArray#reduce().
  FragmentedArray.registers: FragmentedArray#registers.
  FragmentedArray.shape: FragmentedArray#shape().
  FragmentedArray.astype: FragmentedArray#astype().
  FragmentedArray._pointwise: FragmentedArray#_pointwise().
  FragmentedArray.to_layout: FragmentedArray#to_layout().
  TiledLayout: TiledLayout#
  FragmentedArray.transfer_tiled: FragmentedArray#transfer_tiled().
  FragmentedArray.is_signed: FragmentedArray#is_signed.
  FragmentedArray.store_untiled: FragmentedArray#store_untiled().
  WGSplatFragLayout: WGSplatFragLayout#
  WGStridedFragLayout: WGStridedFragLayout#
  FragmentedArray.broadcast_in_dim: FragmentedArray#broadcast_in_dim().
  c: c.
  plan_tiled_transfer: plan_tiled_transfer().
  FragmentedArray.store_tiled_async: FragmentedArray#store_tiled_async().
  Replicated: Replicated#
  FragmentedArray.store_tiled: FragmentedArray#store_tiled().
  TiledLayout.warp_dims: TiledLayout#warp_dims.
  TiledLayout.lane_dims: TiledLayout#lane_dims.
  WGMMA_LAYOUT: WGMMA_LAYOUT.
  TiledLayout.vector_dim: TiledLayout#vector_dim.
  FragmentedArray.splat: FragmentedArray#splat().
  FragmentedArray.load_reduce_untiled: FragmentedArray#load_reduce_untiled().
  WGStridedFragLayout.vec_size: WGStridedFragLayout#vec_size.
  FragmentedArray.load_strided: FragmentedArray#load_strided().
  TiledLayout.remove_dimension: TiledLayout#remove_dimension().
  FragmentedArray._store_untiled_splat: FragmentedArray#_store_untiled_splat().
  FragmentedArray.load_tiled: FragmentedArray#load_tiled().
  TiledLayout.canonicalize: TiledLayout#canonicalize().
  TiledLayout.__post_init__: TiledLayout#__post_init__().
  TiledLayout.vector_length: TiledLayout#vector_length().
  concatenate: concatenate().
  Tiling: Tiling#
  FragmentedLayout.FragmentedLayout: FragmentedLayout.FragmentedLayout.
  Tiling.tiles: Tiling#tiles.
  FragmentedArray.reshape: FragmentedArray#reshape().
  TiledLayout.tiling: TiledLayout#tiling.
  FragmentedArray.pairwise_convert: FragmentedArray#pairwise_convert().
  FragmentedArray.__init__: FragmentedArray#__init__().
  FragmentedArray.__getitem__: FragmentedArray#__getitem__().
  WARPGROUP_SIZE: WARPGROUP_SIZE.
  FragmentedArray.broadcast: FragmentedArray#broadcast().
  FragmentedArray.transfer_strided: FragmentedArray#transfer_strided().
  copy_tiled: copy_tiled().
  FragmentedArray.abs: FragmentedArray#abs().
  tiled_copy_smem_gmem_layout: tiled_copy_smem_gmem_layout().
  TiledLayout.thread_idxs: TiledLayout#thread_idxs().
  WGMMA_TRANSPOSED_LAYOUT: WGMMA_TRANSPOSED_LAYOUT.
  TCGEN05_LAYOUT: TCGEN05_LAYOUT.
  WGStridedFragLayout.shape: WGStridedFragLayout#shape.
  TCGEN05_ROW_LAYOUT: TCGEN05_ROW_LAYOUT.
  TCGEN05_TRANSPOSED_LAYOUT: TCGEN05_TRANSPOSED_LAYOUT.
  FragmentedArray.max: FragmentedArray#max().
  FragmentedArray.min: FragmentedArray#min().
  TCGEN05_COL_LAYOUT: TCGEN05_COL_LAYOUT.
  TMA_INDICES_4_LAYOUT: TMA_INDICES_4_LAYOUT.
  FragmentedArray.reduce_within_warp: FragmentedArray#reduce_within_warp().
  FragmentedArray.foreach: FragmentedArray#foreach().
  TiledLayout.base_tile_shape: TiledLayout#base_tile_shape().
  TMA_INDICES_LAYOUT: TMA_INDICES_LAYOUT.
  FragmentedArray.__setitem__: FragmentedArray#__setitem__().
  FragmentedArray.broadcast_minor: FragmentedArray#broadcast_minor().
  TiledLayout.registers_shape: TiledLayout#registers_shape().
  FragmentedArray.reduce_stored: FragmentedArray#reduce_stored().
  WGMMA_ROW_LAYOUT: WGMMA_ROW_LAYOUT.
  WGMMA_LAYOUT_ACC_32BIT: WGMMA_LAYOUT_ACC_32BIT.
  WGMMA_LAYOUT_8BIT: WGMMA_LAYOUT_8BIT.
  WGMMA_LAYOUT_UPCAST_2X: WGMMA_LAYOUT_UPCAST_2X.
  tmem_native_layout: tmem_native_layout().
  WARP_SIZE: WARP_SIZE.
  TiledLayout.shape_from_registers_shape: TiledLayout#shape_from_registers_shape().
  TiledLayout.warp_indices: TiledLayout#warp_indices().
  FragmentedArray.get_fp8_rounding: FragmentedArray#get_fp8_rounding().
  WGMMA_LAYOUT_UPCAST_4X: WGMMA_LAYOUT_UPCAST_4X.
  FragmentedArray.fast_instr: FragmentedArray#fast_instr().
  FragmentedArray.bitcast: FragmentedArray#bitcast().
  FragmentedArray._store_register_atomic: FragmentedArray#_store_register_atomic().
  Rounding: Rounding#
  WGMMA_COL_LAYOUT: WGMMA_COL_LAYOUT.
  FragmentedArray.load_untiled: FragmentedArray#load_untiled().
  TiledLayout.tiled_tiling_shape: TiledLayout#tiled_tiling_shape().
  TiledLayout._delinearize_index: TiledLayout#_delinearize_index().
  TiledLayout.lane_indices: TiledLayout#lane_indices().
  FragmentedArray.broadcasted_iota: FragmentedArray#broadcasted_iota().
  FragmentedArray._is_e8m0_constant_one: FragmentedArray#_is_e8m0_constant_one().
  FragmentedArray._lift_fast_packed_instr: FragmentedArray#_lift_fast_packed_instr().
  WARPS_IN_WARPGROUP: WARPS_IN_WARPGROUP.
  TiledLayout.partitioned_warp_dims: TiledLayout#partitioned_warp_dims().
  FragmentedArray._compare: FragmentedArray#_compare().
  FragmentedArray._lift_fast_instr: FragmentedArray#_lift_fast_instr().
  FragmentedArray.swizzle_warp_idx_fn: FragmentedArray#swizzle_warp_idx_fn().
  Rounding.ptx: Rounding#ptx().
  Replicated.times: Replicated#times.
  TiledLayout.partitioned_lane_dims: TiledLayout#partitioned_lane_dims().
  WGStridedFragLayout.linear_thread_idxs: WGStridedFragLayout#linear_thread_idxs().
  FragmentedArray.exp2: FragmentedArray#exp2().
  FragmentedArray.__truediv__: FragmentedArray#__truediv__().
  FragmentedArray.__rtruediv__: FragmentedArray#__rtruediv__().
  FragmentedArray.exp: FragmentedArray#exp().
  FragmentedArray.log: FragmentedArray#log().
  FragmentedArray.sin: FragmentedArray#sin().
  FragmentedArray.cos: FragmentedArray#cos().
  FragmentedArray.tanh: FragmentedArray#tanh().
  FragmentedArray.rsqrt: FragmentedArray#rsqrt().
  FragmentedArray.upcast_i4_to_bf16: FragmentedArray#upcast_i4_to_bf16().
  WGStridedFragLayout.from_shaped_type: WGStridedFragLayout#from_shaped_type().
  Rounding.arith: Rounding#arith().
  TiledLayout.reduce: TiledLayout#reduce().
  FragmentedArray.log2: FragmentedArray#log2().
  FragmentedArray.copy: FragmentedArray#copy().
  FragmentedArray.do_convert: FragmentedArray#do_convert().
  FragmentedArray.store_swizzled: FragmentedArray#store_swizzled().
  Tiling.tile_dimension: Tiling#tile_dimension().
  TiledLayout.replication_factor: TiledLayout#replication_factor().
  FragmentedArray.build: FragmentedArray#build().
  WGStridedFragLayout.registers_shape: WGStridedFragLayout#registers_shape().
  FragmentedArray.copysign: FragmentedArray#copysign().
  FragmentedArray.round: FragmentedArray#round().
  FragmentedArray.round_even: FragmentedArray#round_even().
  FragmentedArray.erf: FragmentedArray#erf().
  FragmentedArray.atan2: FragmentedArray#atan2().
  StaggeredTransferPlan.group_pred: StaggeredTransferPlan#group_pred().
  is_supported_strided_layout_broadcast: is_supported_strided_layout_broadcast().
  Tiling.tile_shape: Tiling#tile_shape().
  WGSplatFragLayout.registers_shape: WGSplatFragLayout#registers_shape().
  WGStridedFragLayout.thread_idxs: WGStridedFragLayout#thread_idxs().
  FragmentedArray.debug_print: FragmentedArray#debug_print().
  TransferPlan.tile_index_transforms: TransferPlan#tile_index_transforms().
  optimization_barrier: optimization_barrier().
  Tiling.untile_shape: Tiling#untile_shape().
  WGSplatFragLayout.shape: WGSplatFragLayout#shape.
  FragmentedArray.packed_registers: FragmentedArray#packed_registers().
  FragmentedArray.process_ptx_result: FragmentedArray#process_ptx_result().
  TransferPlan.select: TransferPlan#select().
  TiledLayout.tiled_tiling_rank: TiledLayout#tiled_tiling_rank().
  TiledLayout.replace_tiled_dim: TiledLayout#replace_tiled_dim().
  WGStridedFragLayout.__post_init__: WGStridedFragLayout#__post_init__().
  FragmentedArray.__add__: FragmentedArray#__add__().
  FragmentedArray.__mul__: FragmentedArray#__mul__().
  FragmentedArray.__sub__: FragmentedArray#__sub__().
  FragmentedArray.__rsub__: FragmentedArray#__rsub__().
  FragmentedArray.__floordiv__: FragmentedArray#__floordiv__().
  FragmentedArray.trunc_div: FragmentedArray#trunc_div().
  FragmentedArray.__rfloordiv__: FragmentedArray#__rfloordiv__().
  FragmentedArray.__mod__: FragmentedArray#__mod__().
  FragmentedArray.__rmod__: FragmentedArray#__rmod__().
  FragmentedArray.__rshift__: FragmentedArray#__rshift__().
  FragmentedArray.tree_flatten: FragmentedArray#tree_flatten().
  Tiling.tile_nested_shape_strides: Tiling#tile_nested_shape_strides().
  FragmentedArray._e8m0_reciprocal: FragmentedArray#_e8m0_reciprocal().
  FragmentedArray.upcast_i4_to_i8: FragmentedArray#upcast_i4_to_i8().
  FragmentedArray.expand_nested_dims: FragmentedArray#expand_nested_dims().
  Tiling.fail: Tiling#fail().
  Tiling.fail_if: Tiling#fail_if().
  TMEM_NATIVE_LAYOUT: TMEM_NATIVE_LAYOUT.
  SMEM_BANK_BYTES: SMEM_BANK_BYTES.
  Tiling.canonicalize: Tiling#canonicalize().
  Tiling.remove_dimension: Tiling#remove_dimension().
  FragmentedArray.upcast_to_f8e4m3fn: FragmentedArray#upcast_to_f8e4m3fn().
  FragmentedArray.get_base_index: FragmentedArray#get_base_index().
  TransferPlan.select_if_group: TransferPlan#select_if_group().
  StaggeredTransferPlan.rotate: StaggeredTransferPlan#rotate().
  enumerate_negative: enumerate_negative().
  TransferPlan: TransferPlan#
  TrivialTransferPlan: TrivialTransferPlan#
  ReductionKind: ReductionKind.
  Rounding.TO_NEAREST_EVEN: Rounding#TO_NEAREST_EVEN.
  FragmentedArray.__neg__: FragmentedArray#__neg__().
  FragmentedArray.__or__: FragmentedArray#__or__().
  FragmentedArray.__and__: FragmentedArray#__and__().
  FragmentedArray.__xor__: FragmentedArray#__xor__().
  FragmentedArray.__lshift__: FragmentedArray#__lshift__().
  StaggeredTransferPlan.tile_index_transforms: StaggeredTransferPlan#tile_index_transforms().
  TransferPlanDerivationError: TransferPlanDerivationError#
  Tiling.tile_strides: Tiling#tile_strides().
  TiledLayout.is_nontrivial: TiledLayout#is_nontrivial().
  FragmentedArray.select: FragmentedArray#select().
  FragmentedArray.wrapped_fn: FragmentedArray#wrapped_fn().
  plan_tiled_transfer.has_bank_conflicts: plan_tiled_transfer().has_bank_conflicts().
  Rounding.TO_POSITIVE_INFINITY: Rounding#TO_POSITIVE_INFINITY.
  Rounding.TO_ZERO: Rounding#TO_ZERO.
  Tiling.untile_indices: Tiling#untile_indices().
  WGSplatFragLayout.can_broadcast_to: WGSplatFragLayout#can_broadcast_to().
  WGSplatFragLayout.thread_idxs: WGSplatFragLayout#thread_idxs().
  WGStridedFragLayout.registers_element_type: WGStridedFragLayout#registers_element_type().
  FragmentedArray.e8m0_inv: FragmentedArray#e8m0_inv().
  FragmentedArray.int_pow: FragmentedArray#int_pow().
  FragmentedArray.mem_idx_to_reg_idx: FragmentedArray#mem_idx_to_reg_idx().
  FragmentedArray.get_register: FragmentedArray#get_register().
  FragmentedArray.update_registers: FragmentedArray#update_registers().
  StaggeredTransferPlan: StaggeredTransferPlan#
  optimization_barrier._repack: optimization_barrier()._repack().
  TiledLayout._check_canonical: TiledLayout#_check_canonical.
  WGSplatFragLayout.registers_element_type: WGSplatFragLayout#registers_element_type().
  FragmentedArray.upcast_i8_to_bf16: FragmentedArray#upcast_i8_to_bf16().
  FragmentedArray.check_supported_rounding: FragmentedArray#check_supported_rounding().
  Tiling.__post_init__: Tiling#__post_init__().
  Tiling.__str__: Tiling#__str__().
  Tiling.tile_indices: Tiling#tile_indices().
  TiledLayout.registers_element_type: TiledLayout#registers_element_type().
  _tiled_wgmma_layout: _tiled_wgmma_layout().
  WGSplatFragLayout.shape_from_registers_shape: WGSplatFragLayout#shape_from_registers_shape().
  WGStridedFragLayout.shape_from_registers_shape: WGStridedFragLayout#shape_from_registers_shape().
  FragmentedArray.__invert__: FragmentedArray#__invert__().
  FragmentedArray.__eq__: FragmentedArray#__eq__().
  FragmentedArray.__ne__: FragmentedArray#__ne__().
  FragmentedArray.__lt__: FragmentedArray#__lt__().
  FragmentedArray.__le__: FragmentedArray#__le__().
  FragmentedArray.__gt__: FragmentedArray#__gt__().
  FragmentedArray.__ge__: FragmentedArray#__ge__().
  StaggeredTransferPlan.select: StaggeredTransferPlan#select().
  StaggeredTransferPlan.select_if_group: StaggeredTransferPlan#select_if_group().
  SMEM_BANKS: SMEM_BANKS.
  FragmentedArray.get_ptx_constraint: FragmentedArray#get_ptx_constraint().
  addf: addf().
  subf: subf().
  can_relayout_wgmma_4x_to_wgmma_2x: can_relayout_wgmma_4x_to_wgmma_2x().
  can_relayout_wgmma_2x_to_wgmma: can_relayout_wgmma_2x_to_wgmma().
  FragmentedArray.cast: FragmentedArray#cast().
  FragmentedArray.update: FragmentedArray#update().
  FragmentedArray.get: FragmentedArray#get().
  StaggeredTransferPlan.stagger: StaggeredTransferPlan#stagger.
  StaggeredTransferPlan.dim: StaggeredTransferPlan#dim.
  StaggeredTransferPlan.size: StaggeredTransferPlan#size.
  StaggeredTransferPlan.group_stride: StaggeredTransferPlan#group_stride.
  mulf: mulf().
  IndexTransform.IndexTransform: IndexTransform.IndexTransform.
  FragmentedArray.__pos__: FragmentedArray#__pos__().
  FragmentedArray.__radd__: FragmentedArray#__radd__().
  FragmentedArray.__rmul__: FragmentedArray#__rmul__().
  FragmentedArray.__ror__: FragmentedArray#__ror__().
  FragmentedArray.__rand__: FragmentedArray#__rand__().
  FragmentedArray.__rxor__: FragmentedArray#__rxor__().
  FragmentedArray.tree_unflatten: FragmentedArray#tree_unflatten().
  TrivialTransferPlan.tile_index_transforms: TrivialTransferPlan#tile_index_transforms().
  TrivialTransferPlan.select: TrivialTransferPlan#select().
  TrivialTransferPlan.select_if_group: TrivialTransferPlan#select_if_group().
---
# Module: [`jax/experimental/mosaic/gpu/fragmented_array.py`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py)

## Classes
### `FragmentedArray`
- def: [`jax/experimental/mosaic/gpu/fragmented_array.py:970`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L970) — documented in [jax-experimental-mosaic-gpu-fragmented_array](../../../../../concepts/jax-experimental-mosaic-gpu-fragmented_array.md)
- signature: `class FragmentedArray:`
- members:
  - `__init__(self, *, _registers: np.ndarray, _layout: FragmentedLayout, _is_signed: bool | None)` — [`L976`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L976) — Initializes a fragmented array.
  - `abs(self)` — [`L1857`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1857)
  - `astype(self, new_dtype: ir.Type, *, is_signed: bool | None = None, rounding: Rounding | None = None)` — [`L2103`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2103)
  - `atan2(self, other: FragmentedArray)` — [`L1899`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1899)
  - `bitcast(self, elt: ir.Type, *, output_is_signed: bool | None = None)` — [`L1988`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1988)
  - `broadcast(self, shape: tuple[int, ...])` — [`L3351`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3351)
  - `broadcast_in_dim(self, shape: tuple[int, ...], source_dimensions: tuple[int, ...], layout: FragmentedLayout)` — [`L3413`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3413)
  - `broadcast_minor(self, n: int)` — [`L3400`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3400)
  - `broadcasted_iota(dtype: ir.Type, shape: tuple[int, ...], dimension: int, layout: FragmentedLayout | None = None, *, is_signed: bool | None = None)` — [`L1108`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1108) — Creates a broadcasted iota array along the specified dimension.
  - `build(cls, shape: tuple[int, ...], layout: FragmentedLayout, fn: Callable[..., ir.Value], *, is_signed: bool | None = None)` — [`L3504`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3504)
  - `cast(idx: ir.Value)` — [`L1122`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1122)
  - `check_supported_rounding(allowed_roundings, hw_limitation: bool = True)` — [`L2683`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2683)
  - `copy(self)` — [`L2095`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2095)
  - `copysign(self, other: FragmentedArray)` — [`L1780`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1780)
  - `cos(self, *, approx: bool = False)` — [`L1830`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1830)
  - `debug_print(self, fmt: str)` — [`L3576`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3576)
  - `do_convert(vec, convert_vec_len)` — [`L2502`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2502)
  - `e8m0_inv(x)` — [`L1529`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1529)
  - `erf(self)` — [`L1894`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1894)
  - `exp(self, *, approx: bool = False)` — [`L1785`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1785)
  - `exp2(self, *, approx: bool = False)` — [`L1794`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1794)
  - `expand_nested_dims(idxs: Sequence[ir.Value])` — [`L4250`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4250)
  - `fast_instr(*args)` — [`L1908`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1908)
  - `foreach(self, fn: Callable[[ir.Value, tuple[ir.Value, ...]], ir.Value | None], *, create_array=False, is_signed=None)` — [`L3518`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3518) — Call a function for each value and index.
  - `get(registers, _i=i)` — [`L4103`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4103)
  - `get_base_index()` — [`L4322`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4322)
  - `get_fp8_rounding(fp8_type: ir.Type)` — [`L2693`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2693)
  - `get_ptx_constraint(bitwidth)` — [`L2513`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2513)
  - `get_register(regs, reg_idxs=reg_idxs)` — [`L4308`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4308)
  - `int_pow(x, n=reduced_elems)` — [`L2939`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2939)
  - `load_reduce_untiled(cls, ref: utils.MultimemRef, layout: TiledLayout | WGStridedFragLayout, reduction: utils.MultimemReductionOp, swizzle: int = 16, is_signed: bool | None = None)` — [`L3660`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3660)
  - `load_strided(cls, ref: ir.Value, *, is_signed: bool | None = None, vec_size: int | None = None)` — [`L1031`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1031)
  - `load_tiled(cls, ref, swizzle: int | None, *, is_signed: bool | None = None, layout: FragmentedLayout = WGMMA_LAYOUT, optimized: bool = True, tiling_rank: int | None = None, _load_fun: Callable[[ir.VectorType, ir.Value], ir.Value] = llvm.load, _narrow_float_as_int: bool = True)` — [`L3995`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3995)
  - `load_untiled(cls, ref: ir.Value, *, layout: TiledLayout, swizzle: int = 16, is_signed: bool | None = None, optimized: bool = True)` — [`L3696`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3696)
  - `log(self, *, approx: bool = False)` — [`L1803`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1803)
  - `log2(self, *, approx: bool = False)` — [`L1812`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1812)
  - `max(self, other)` — [`L1736`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1736)
  - `mem_idx_to_reg_idx(idx)` — [`L4298`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4298)
  - `min(self, other)` — [`L1758`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1758)
  - `mlir_dtype(self)` — [`L1152`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1152) — documented in [jax-experimental-mosaic-gpu-fragmented_array](../../../../../concepts/jax-experimental-mosaic-gpu-fragmented_array.md)
  - `packed_registers(dst_vector_len: int, *, if_not_sliced: bool)` — [`L2159`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2159) — Tries to pack registers up to destination vector length.
  - `pairwise_convert(ptx_instr: str)` — [`L2495`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2495)
  - `process_ptx_result(ptx_result)` — [`L2584`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2584)
  - `reduce(self, op: str | Callable[[ir.Value, ir.Value], ir.Value], axis: int | Sequence[int], scratch: ir.Value | None = None)` — [`L2864`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2864) — documented in [jax-experimental-mosaic-gpu-fragmented_array](../../../../../concepts/jax-experimental-mosaic-gpu-fragmented_array.md)
  - `reduce_stored(reg_ty: ir.VectorType, step_idx: int, lane_idx: ir.Value, swizzle_warp_idx: Callable[[ir.Value], ir.Value])` — [`L3183`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3183)
  - `reduce_within_warp(out_idx)` — [`L3017`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3017)
  - `reshape(self, shape: tuple[int, ...])` — [`L3362`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3362)
  - `round(self)` — [`L1882`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1882) — Same as `lax.round(..., AWAY_FROM_ZERO)`.
  - `round_even(self)` — [`L1888`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1888) — Same as `lax.round(..., TO_NEAREST_EVEN)`.
  - `rsqrt(self, *, approx: bool = False)` — [`L1848`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1848)
  - `select(self, on_true, on_false)` — [`L3491`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3491)
  - `shape(self)` — [`L1140`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1140)
  - `sin(self, *, approx: bool = False)` — [`L1821`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1821)
  - `splat(cls, value: ir.Value, shape: tuple[int, ...], layout: FragmentedLayout | None = None, *, is_signed: bool | None = None)` — [`L1082`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1082)
  - `store_swizzled(reg: ir.Value, step_idx: int, lane_idx: ir.Value, scratch: ir.Value, swizzle_warp_idx: Callable[[ir.Value], ir.Value])` — [`L3157`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3157)
  - `store_tiled(self, ref: ir.Value | utils.MultimemRef, swizzle: int | None, optimized: bool = True, tiling_rank: int | None = None, atomic: Literal["add", "max", "min", "and", "or", "xor"] | None = None)` — [`L3942`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3942)
  - `store_tiled_async(self, ref: ir.Value, barrier: utils.BarrierRef, cluster_dim: gpu.Dimension, cluster_idx: ir.Value, swizzle: int | None, optimized: bool = True, tiling_rank: int | None = None, atomic: Literal["add", "max", "min", "and", "or", "xor"] | None = None)` — [`L3737`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3737)
  - `store_untiled(self, ref: ir.Value | utils.MultimemRef, *, swizzle: int = 16, optimized: bool = True, atomic: Literal["add", "max", "min", "and", "or", "xor"] | None = None)` — [`L3583`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3583)
  - `swizzle_warp_idx_fn(lane_idx: ir.Value, vec_len: int)` — [`L3071`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3071)
  - `tanh(self, *, approx: bool = False)` — [`L1839`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1839)
  - `to_layout(self, new_layout: FragmentedLayout)` — [`L1162`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1162) — Converts the fragmented array to the given layout.
  - `transfer_strided(self, ref: ir.Value, vec_size: int)` — [`L4055`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4055)
  - `transfer_tiled(ref: ir.Value, swizzle: int | None, layout: TiledLayout, shape: tuple[int, ...], optimized: bool = True, ref_tiling_rank: int | None = None)` — [`L4108`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4108) — Generate a transfer schedule for a tiled layout.
  - `tree_flatten(self)` — [`L4332`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4332)
  - `tree_unflatten(cls, aux, flat_registers)` — [`L4337`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4337)
  - `trunc_div(self, other)` — [`L1589`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1589)
  - `upcast_i4_to_bf16(reg: ir.Value, part: int)` — [`L2279`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2279)
  - `upcast_i4_to_i8(reg: ir.Value, first_valid_nibble: int = 0)` — [`L2377`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2377)
  - `upcast_i8_to_bf16(reg, high)` — [`L2447`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2447)
  - `upcast_to_f8e4m3fn(reg: ir.Value, part: int)` — [`L2193`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2193)
  - `update(registers, reg, _i=i)` — [`L4101`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4101)
  - `update_registers(regs, new, reg_idxs=reg_idxs)` — [`L4312`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4312)
  - `wrapped_fn(*args)` — [`L3530`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3530)
  - `is_signed` — [`L974`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L974)
  - `layout` — [`L973`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L973) — documented in [jax-experimental-mosaic-gpu-fragmented_array](../../../../../concepts/jax-experimental-mosaic-gpu-fragmented_array.md)
  - `registers` — [`L972`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L972) — documented in [jax-experimental-mosaic-gpu-fragmented_array](../../../../../concepts/jax-experimental-mosaic-gpu-fragmented_array.md)
- protocol/private: `__add__`[`L1490`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1490), `__and__`[`L1641`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1641), `__eq__`[`L1670`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1670), `__floordiv__`[`L1576`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1576), `__ge__`[`L1712`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1712), `__getitem__`[`L2012`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2012), `__gt__`[`L1704`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1704), `__invert__`[`L1628`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1628), `__le__`[`L1696`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1696), `__lshift__`[`L1657`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1657), `__lt__`[`L1688`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1688), `__mod__`[`L1612`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1612), `__mul__`[`L1501`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1501), `__ne__`[`L1679`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1679), `__neg__`[`L1482`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1482), `__or__`[`L1633`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1633), `__pos__`[`L1479`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1479), `__radd__`[`L1498`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1498), `__rand__`[`L1646`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1646), `__rfloordiv__`[`L1599`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1599), `__rmod__`[`L1620`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1620), `__rmul__`[`L1509`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1509), `__ror__`[`L1638`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1638), `__rshift__`[`L1662`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1662), `__rsub__`[`L1520`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1520), `__rtruediv__`[`L1567`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1567), `__rxor__`[`L1654`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1654), `__setitem__`[`L2054`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L2054), `__sub__`[`L1512`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1512), `__truediv__`[`L1556`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1556), `__xor__`[`L1649`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1649), `_compare`[`L1720`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1720), `_e8m0_reciprocal`[`L1528`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1528), `_is_e8m0_constant_one`[`L1540`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1540), `_lift_fast_instr`[`L1905`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1905), `_lift_fast_packed_instr`[`L1945`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1945), `_pointwise`[`L1400`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L1400), `_store_register_atomic`[`L3812`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3812), `_store_untiled_splat`[`L3711`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L3711)
- uses (calls/refs, reference-scoped): [`bitwidth`](utils.md#bitwidth), [`c`](utils.md#c), [`TiledLayout`](fragmented_array.md#TiledLayout), [`WGSplatFragLayout`](fragmented_array.md#WGSplatFragLayout), [`WGStridedFragLayout`](fragmented_array.md#WGStridedFragLayout), [`bitcast`](utils.md#bitcast), [`c`](fragmented_array.md#c), [`plan_tiled_transfer`](fragmented_array.md#plan_tiled_transfer), [`Replicated`](fragmented_array.md#Replicated), [`WGMMA_LAYOUT`](fragmented_array.md#WGMMA_LAYOUT), [`lane_dims`](fragmented_array.md#TiledLayout.lane_dims), [`warp_dims`](fragmented_array.md#TiledLayout.warp_dims), [`memref_ptr`](utils.md#memref_ptr), [`vector_dim`](fragmented_array.md#TiledLayout.vector_dim), [`warpgroup_barrier`](utils.md#warpgroup_barrier), [`vec_size`](fragmented_array.md#WGStridedFragLayout.vec_size), [`vector_length`](fragmented_array.md#TiledLayout.vector_length), [`Tiling`](fragmented_array.md#Tiling), [`FragmentedLayout`](fragmented_array.md#FragmentedLayout.FragmentedLayout), [`tiling`](fragmented_array.md#TiledLayout.tiling), [`MultimemRef`](utils.md#MultimemRef), [`get_ptr`](utils.md#BarrierRef.get_ptr), [`WARPGROUP_SIZE`](fragmented_array.md#WARPGROUP_SIZE), [`TCGEN05_LAYOUT`](fragmented_array.md#TCGEN05_LAYOUT), [`WGMMA_TRANSPOSED_LAYOUT`](fragmented_array.md#WGMMA_TRANSPOSED_LAYOUT), [`TCGEN05_ROW_LAYOUT`](fragmented_array.md#TCGEN05_ROW_LAYOUT), [`shape`](fragmented_array.md#WGStridedFragLayout.shape), [`TCGEN05_TRANSPOSED_LAYOUT`](fragmented_array.md#TCGEN05_TRANSPOSED_LAYOUT), [`is_smem_ref`](utils.md#is_smem_ref), [`shfl_bfly`](utils.md#shfl_bfly), [`BarrierRef`](utils.md#BarrierRef), [`base_tile_shape`](fragmented_array.md#TiledLayout.base_tile_shape), [`getelementptr`](utils.md#getelementptr), [`registers_shape`](fragmented_array.md#TiledLayout.registers_shape), [`vector_concat`](utils.md#vector_concat), [`WGMMA_ROW_LAYOUT`](fragmented_array.md#WGMMA_ROW_LAYOUT), [`multimem_load_reduce`](utils.md#multimem_load_reduce), [`WGMMA_LAYOUT_ACC_32BIT`](fragmented_array.md#WGMMA_LAYOUT_ACC_32BIT), [`WGMMA_LAYOUT_UPCAST_2X`](fragmented_array.md#WGMMA_LAYOUT_UPCAST_2X), [`redux`](utils.md#redux)  (+62 more)
- used by: [`async_copy`](launch_context.md#LaunchContext.async_copy), [`_copy_gmem_to_smem_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_copy_gmem_to_smem_lowering), [`_swap_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_swap_lowering_rule), [`_get_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_get_lowering_rule), [`_copy_smem_to_gmem_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_copy_smem_to_gmem_lowering), [`_async_store_smem_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_async_store_smem_lowering), [`_broadcast_in_dim_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_broadcast_in_dim_lowering_rule), [`_ensure_ir_value`](../../../_src/pallas/mosaic_gpu/lowering.md#_ensure_ir_value), [`_tcgen05_mma_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_tcgen05_mma_lowering), [`_inline_mgpu_flat_transformed_args`](../../../_src/pallas/mosaic_gpu/primitives.md#_inline_mgpu_flat_transformed_args), [`_prepare_async_copy`](launch_context.md#LaunchContext._prepare_async_copy), [`_fragmented_array_from_ir`](dialect_lowering.md#_fragmented_array_from_ir), [`_async_prefetch_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_async_prefetch_lowering), [`_multimem_load_reduce_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_multimem_load_reduce_lowering_rule), [`_async_store_smem_op_lowering_rule`](dialect_lowering.md#_async_store_smem_op_lowering_rule), [`_select_n_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_select_n_lowering_rule), [`_vector_load_op_lowering_rule`](dialect_lowering.md#_vector_load_op_lowering_rule), [`_while_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_while_lowering_rule), [`_ensure_fa`](../../../_src/pallas/mosaic_gpu/lowering.md#_ensure_fa), [`_vector_multi_dim_reduction_op_lowering_rule`](dialect_lowering.md#_vector_multi_dim_reduction_op_lowering_rule), [`_populate_custom_primitive_op_block`](../../../_src/pallas/mosaic_gpu/primitives.md#_populate_custom_primitive_op_block), [`_is_signed`](dialect_lowering.md#_is_signed), [`_wgmma_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_wgmma_lowering), [`_bitcast_convert_type_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_bitcast_convert_type_lowering_rule), [`_convert_element_type_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_convert_element_type_lowering_rule), [`wgmma_m64`](wgmma.md#wgmma_m64), [`_barrier_test_lowering`](../../../_src/pallas/mosaic_gpu/primitives.md#_barrier_test_lowering), [`_inline_mgpu_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_inline_mgpu_lowering_rule), [`wgmma`](wgmma.md#wgmma), [`_vector_store_op_lowering_rule`](dialect_lowering.md#_vector_store_op_lowering_rule), [`fragmented_array_to_ir`](dialect_lowering.md#fragmented_array_to_ir), [`mma`](mma.md#mma), [`_atomic_store_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_atomic_store_lowering_rule), [`_cos_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_cos_lowering_rule), [`_exp2_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_exp2_lowering_rule), [`_exp_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_exp_lowering_rule), [`_log_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_log_lowering_rule), [`_rsqrt_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_rsqrt_lowering_rule), [`_sin_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_sin_lowering_rule), [`_tanh_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_tanh_lowering_rule)  (+56 more)

### `Replicated`
- def: [`jax/experimental/mosaic/gpu/fragmented_array.py:305`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L305)
- signature: `class Replicated:`
- members:
  - `times` — [`L306`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L306)
- used by: [`plan_tiled_transfer`](fragmented_array.md#plan_tiled_transfer), [`store_tiled_async`](fragmented_array.md#FragmentedArray.store_tiled_async), [`store_tiled`](fragmented_array.md#FragmentedArray.store_tiled), [`lane_dims`](fragmented_array.md#TiledLayout.lane_dims), [`warp_dims`](fragmented_array.md#TiledLayout.warp_dims), [`__post_init__`](fragmented_array.md#TiledLayout.__post_init__), [`TCGEN05_ROW_LAYOUT`](fragmented_array.md#TCGEN05_ROW_LAYOUT), [`TCGEN05_COL_LAYOUT`](fragmented_array.md#TCGEN05_COL_LAYOUT), [`TMA_INDICES_4_LAYOUT`](fragmented_array.md#TMA_INDICES_4_LAYOUT), [`reduce_within_warp`](fragmented_array.md#FragmentedArray.reduce_within_warp), [`TMA_INDICES_LAYOUT`](fragmented_array.md#TMA_INDICES_LAYOUT), [`WGMMA_ROW_LAYOUT`](fragmented_array.md#WGMMA_ROW_LAYOUT), [`rhs`](mma.md#MMALayouts.rhs), [`WGMMA_COL_LAYOUT`](fragmented_array.md#WGMMA_COL_LAYOUT), [`_delinearize_index`](fragmented_array.md#TiledLayout._delinearize_index), [`partitioned_warp_dims`](fragmented_array.md#TiledLayout.partitioned_warp_dims), [`partitioned_lane_dims`](fragmented_array.md#TiledLayout.partitioned_lane_dims), [`replication_factor`](fragmented_array.md#TiledLayout.replication_factor), [`replace_tiled_dim`](fragmented_array.md#TiledLayout.replace_tiled_dim), [`_from_int_or_replicated_attr`](layouts.md#_from_tiled_layout_attr._from_int_or_replicated_attr), [`_int_or_replicated`](layouts.md#_to_tiled_layout_attr._int_or_replicated), [`is_nontrivial`](fragmented_array.md#TiledLayout.is_nontrivial)

### `Rounding`  ·  implements/extends Enum
- def: [`jax/experimental/mosaic/gpu/fragmented_array.py:271`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L271)
- signature: `class Rounding(enum.Enum):`
- members:
  - `arith(self)` — [`L287`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L287)
  - `ptx(self)` — [`L277`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L277)
  - `TO_NEAREST_EVEN` — [`L272`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L272)
  - `TO_POSITIVE_INFINITY` — [`L273`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L273)
  - `TO_ZERO` — [`L274`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L274)
- used by: [`astype`](fragmented_array.md#FragmentedArray.astype), [`get_fp8_rounding`](fragmented_array.md#FragmentedArray.get_fp8_rounding)

### `StaggeredTransferPlan`  ·  implements/extends TransferPlan
- def: [`jax/experimental/mosaic/gpu/fragmented_array.py:4383`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4383)
- signature: `class StaggeredTransferPlan(TransferPlan):`
- members:
  - `group_pred(self)` — [`L4390`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4390)
  - `rotate(idx: tuple[int, ...])` — [`L4399`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4399)
  - `select(self, group_elems: Sequence[ir.Value])` — [`L4405`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4405)
  - `select_if_group(self, group_idx: int, old: ir.Value, new: ir.Value)` — [`L4409`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4409)
  - `tile_index_transforms(self)` — [`L4397`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4397)
  - `dim` — [`L4385`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4385)
  - `group_stride` — [`L4387`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4387)
  - `size` — [`L4386`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4386)
  - `stagger` — [`L4384`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4384)
- uses (calls/refs, reference-scoped): [`WARP_SIZE`](fragmented_array.md#WARP_SIZE), [`thread_idx`](utils.md#thread_idx), [`TransferPlan`](fragmented_array.md#TransferPlan)
- used by: [`plan_tiled_transfer`](fragmented_array.md#plan_tiled_transfer), [`tile_index_transforms`](fragmented_array.md#TransferPlan.tile_index_transforms), [`select`](fragmented_array.md#TransferPlan.select), [`select_if_group`](fragmented_array.md#TransferPlan.select_if_group), [`TransferPlan`](fragmented_array.md#TransferPlan)

### `TiledLayout`
- def: [`jax/experimental/mosaic/gpu/fragmented_array.py:310`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L310)
- doc: A FragmentedArray layout derived from a tiling expression.
- signature: `class TiledLayout:`
- members:
  - `base_tile_shape(self)` — [`L429`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L429) — The shape of the first tile in the tiling expression.
  - `canonicalize(self)` — [`L552`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L552) — Returns a version of this layout where tiling is canonical.
  - `is_nontrivial(d: int | Replicated)` — [`L596`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L596)
  - `lane_indices(self)` — [`L505`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L505)
  - `partitioned_lane_dims(self)` — [`L402`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L402)
  - `partitioned_warp_dims(self)` — [`L396`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L396)
  - `reduce(self, axes: Sequence[int])` — [`L546`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L546)
  - `registers_element_type(self, t: ir.Type)` — [`L457`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L457)
  - `registers_shape(self, shape: tuple[int, ...])` — [`L460`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L460) — Returns the shape of the register array needed to represent an array of the given logical shape.
  - `remove_dimension(self, dim: int)` — [`L518`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L518)
  - `replace_tiled_dim(d: int | Replicated)` — [`L531`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L531)
  - `replication_factor(self)` — [`L608`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L608)
  - `shape_from_registers_shape(self, shape: tuple[int, ...])` — [`L470`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L470) — Returns the logical shape of an array given its register array shape.
  - `thread_idxs(self, shape: tuple[int, ...])` — [`L407`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L407)
  - `tiled_tiling_rank(self)` — [`L450`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L450)
  - `tiled_tiling_shape(self)` — [`L438`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L438) — The shape of the suffix of the array after tiling.
  - `vector_length(self)` — [`L454`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L454)
  - `warp_indices(self)` — [`L510`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L510)
  - `lane_dims` — [`L352`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L352)
  - `tiling` — [`L350`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L350)
  - `vector_dim` — [`L353`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L353)
  - `warp_dims` — [`L351`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L351)
- protocol/private: `__post_init__`[`L360`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L360), `_check_canonical`[`L358`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L358), `_delinearize_index`[`L484`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L484)
- uses (calls/refs, reference-scoped): [`c`](fragmented_array.md#c), [`Replicated`](fragmented_array.md#Replicated), [`Tiling`](fragmented_array.md#Tiling), [`tiles`](fragmented_array.md#Tiling.tiles), [`WARP_SIZE`](fragmented_array.md#WARP_SIZE), [`thread_idx`](utils.md#thread_idx), [`WARPS_IN_WARPGROUP`](fragmented_array.md#WARPS_IN_WARPGROUP), [`times`](fragmented_array.md#Replicated.times), [`get_contiguous_strides`](utils.md#get_contiguous_strides), [`tile_dimension`](fragmented_array.md#Tiling.tile_dimension), [`tile_shape`](fragmented_array.md#Tiling.tile_shape), [`dyn_dot`](utils.md#dyn_dot), [`untile_shape`](fragmented_array.md#Tiling.untile_shape), [`canonicalize`](fragmented_array.md#Tiling.canonicalize), [`remove_dimension`](fragmented_array.md#Tiling.remove_dimension), [`tile_strides`](fragmented_array.md#Tiling.tile_strides)
- used by: [`mlir_dtype`](fragmented_array.md#FragmentedArray.mlir_dtype), [`async_copy`](launch_context.md#LaunchContext.async_copy), [`reduce`](fragmented_array.md#FragmentedArray.reduce), [`_swap_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_swap_lowering_rule), [`to_mgpu`](../../../_src/pallas/mosaic_gpu/core.md#Layout.to_mgpu), [`shape`](fragmented_array.md#FragmentedArray.shape), [`astype`](fragmented_array.md#FragmentedArray.astype), [`_get_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_get_lowering_rule), [`to_layout`](fragmented_array.md#FragmentedArray.to_layout), [`_construct_smem_reftree`](core.md#_construct_smem_reftree), [`transfer_tiled`](fragmented_array.md#FragmentedArray.transfer_tiled), [`_broadcast_in_dim_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_broadcast_in_dim_lowering_rule), [`store_untiled`](fragmented_array.md#FragmentedArray.store_untiled), [`_multimem_load_reduce_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_multimem_load_reduce_lowering_rule), [`broadcast_in_dim`](fragmented_array.md#FragmentedArray.broadcast_in_dim), [`_async_store_smem_op_lowering_rule`](dialect_lowering.md#_async_store_smem_op_lowering_rule), [`_vector_load_op_lowering_rule`](dialect_lowering.md#_vector_load_op_lowering_rule), [`plan_tiled_transfer`](fragmented_array.md#plan_tiled_transfer), [`_vector_multi_dim_reduction_op_lowering_rule`](dialect_lowering.md#_vector_multi_dim_reduction_op_lowering_rule), [`store_tiled_async`](fragmented_array.md#FragmentedArray.store_tiled_async), [`_reduce_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_reduce_lowering_rule), [`store_tiled`](fragmented_array.md#FragmentedArray.store_tiled), [`WGMMA_LAYOUT`](fragmented_array.md#WGMMA_LAYOUT), [`splat`](fragmented_array.md#FragmentedArray.splat), [`to_layout_attr`](layouts.md#to_layout_attr), [`mma`](mma.md#mma), [`_tmem_ref_from_ir`](dialect_lowering.md#_tmem_ref_from_ir), [`load_reduce_untiled`](fragmented_array.md#FragmentedArray.load_reduce_untiled), [`load_tiled`](fragmented_array.md#FragmentedArray.load_tiled), [`_async_load_tmem_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_async_load_tmem_lowering_rule), [`_mma_single_tile`](mma.md#_mma_single_tile), [`concatenate`](fragmented_array.md#concatenate), [`FragmentedLayout`](fragmented_array.md#FragmentedLayout.FragmentedLayout), [`reshape`](fragmented_array.md#FragmentedArray.reshape), [`extract_assignment_candidates_from_reduce_equation`](layout_inference.md#extract_assignment_candidates_from_reduce_equation), [`__getitem__`](fragmented_array.md#FragmentedArray.__getitem__), [`__init__`](fragmented_array.md#FragmentedArray.__init__), [`tiled_copy_smem_gmem_layout`](fragmented_array.md#tiled_copy_smem_gmem_layout), [`TCGEN05_LAYOUT`](fragmented_array.md#TCGEN05_LAYOUT), [`WGMMA_TRANSPOSED_LAYOUT`](fragmented_array.md#WGMMA_TRANSPOSED_LAYOUT)  (+29 more)

### `Tiling`
- def: [`jax/experimental/mosaic/gpu/fragmented_array.py:57`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L57)
- doc: A tiling expression describing a permutation of elements of an nd-array.
- signature: `class Tiling:`
- members:
  - `canonicalize(self)` — [`L121`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L121) — Returns a canonicalized version of the tiling.
  - `fail()` — [`L92`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L92)
  - `fail_if(cond, shape=shape)` — [`L211`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L211)
  - `remove_dimension(self, dim: int)` — [`L174`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L174) — Returns a tiling with the given dimension removed.
  - `tile_dimension(self, dim: int)` — [`L165`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L165) — Result is True whenever the tiled dim originated from the given input dim.
  - `tile_indices(self, indices: tuple[int, ...])` — [`L252`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L252)
  - `tile_nested_shape_strides(self, shape: tuple[tuple[int, ...], ...], strides: tuple[tuple[int, ...], ...])` — [`L193`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L193) — A fused version of `tile_shape` and `tile_strides` for nested shapes.
  - `tile_shape(self, shape: tuple[int, ...])` — [`L89`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L89) — Computes the shape of an array after tiling.
  - `tile_strides(self, strides: tuple[int, ...])` — [`L158`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L158) — Computes the strides of an array after tiling.
  - `untile_indices(self, indices: tuple[int, ...])` — [`L262`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L262)
  - `untile_shape(self, shape: tuple[int, ...])` — [`L103`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L103) — Computes the shape of an array before tiling from its tiled shape.
  - `tiles` — [`L71`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L71)
- protocol/private: `__post_init__`[`L73`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L73), `__str__`[`L86`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L86)
- used by: [`reduce`](fragmented_array.md#FragmentedArray.reduce), [`transfer_tiled`](fragmented_array.md#FragmentedArray.transfer_tiled), [`broadcast_in_dim`](fragmented_array.md#FragmentedArray.broadcast_in_dim), [`plan_tiled_transfer`](fragmented_array.md#plan_tiled_transfer), [`_vector_multi_dim_reduction_op_lowering_rule`](dialect_lowering.md#_vector_multi_dim_reduction_op_lowering_rule), [`_reduce_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_reduce_lowering_rule), [`WGMMA_LAYOUT`](fragmented_array.md#WGMMA_LAYOUT), [`load_tiled`](fragmented_array.md#FragmentedArray.load_tiled), [`remove_dimension`](fragmented_array.md#TiledLayout.remove_dimension), [`canonicalize`](fragmented_array.md#TiledLayout.canonicalize), [`__post_init__`](fragmented_array.md#TiledLayout.__post_init__), [`tiling`](fragmented_array.md#TiledLayout.tiling), [`tiled_copy_smem_gmem_layout`](fragmented_array.md#tiled_copy_smem_gmem_layout), [`TCGEN05_LAYOUT`](fragmented_array.md#TCGEN05_LAYOUT), [`WGMMA_TRANSPOSED_LAYOUT`](fragmented_array.md#WGMMA_TRANSPOSED_LAYOUT), [`thread_idxs`](fragmented_array.md#TiledLayout.thread_idxs), [`TCGEN05_ROW_LAYOUT`](fragmented_array.md#TCGEN05_ROW_LAYOUT), [`TCGEN05_TRANSPOSED_LAYOUT`](fragmented_array.md#TCGEN05_TRANSPOSED_LAYOUT), [`TCGEN05_COL_LAYOUT`](fragmented_array.md#TCGEN05_COL_LAYOUT), [`TMA_INDICES_4_LAYOUT`](fragmented_array.md#TMA_INDICES_4_LAYOUT), [`_to_tiled_layout_attr`](layouts.md#_to_tiled_layout_attr), [`TMA_INDICES_LAYOUT`](fragmented_array.md#TMA_INDICES_LAYOUT), [`base_tile_shape`](fragmented_array.md#TiledLayout.base_tile_shape), [`registers_shape`](fragmented_array.md#TiledLayout.registers_shape), [`_from_tiled_layout_attr`](layouts.md#_from_tiled_layout_attr), [`WGMMA_ROW_LAYOUT`](fragmented_array.md#WGMMA_ROW_LAYOUT), [`WGMMA_LAYOUT_8BIT`](fragmented_array.md#WGMMA_LAYOUT_8BIT), [`WGMMA_LAYOUT_ACC_32BIT`](fragmented_array.md#WGMMA_LAYOUT_ACC_32BIT), [`WGMMA_LAYOUT_UPCAST_2X`](fragmented_array.md#WGMMA_LAYOUT_UPCAST_2X), [`tmem_native_layout`](fragmented_array.md#tmem_native_layout), [`rhs`](mma.md#MMALayouts.rhs), [`shape_from_registers_shape`](fragmented_array.md#TiledLayout.shape_from_registers_shape), [`WGMMA_LAYOUT_UPCAST_4X`](fragmented_array.md#WGMMA_LAYOUT_UPCAST_4X), [`WGMMA_COL_LAYOUT`](fragmented_array.md#WGMMA_COL_LAYOUT), [`acc`](mma.md#MMALayouts.acc), [`lhs`](mma.md#MMALayouts.lhs), [`tiled_tiling_shape`](fragmented_array.md#TiledLayout.tiled_tiling_shape), [`get_base_index`](fragmented_array.md#FragmentedArray.get_base_index)

### `TransferPlan`
- def: [`jax/experimental/mosaic/gpu/fragmented_array.py:4347`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4347)
- members:
  - `select(self, group_elems: Sequence[ir.Value],)` — [`L4352`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4352) — Selects the value corresponding to the group of the current thread.
  - `select_if_group(self, group_idx: int, old: ir.Value, new: ir.Value,)` — [`L4359`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4359) — Returns `new` if the current thread belongs to the given group and `old` otherwise.
  - `tile_index_transforms(self)` — [`L4349`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4349)
- uses (calls/refs, reference-scoped): [`tile_index_transforms`](fragmented_array.md#StaggeredTransferPlan.tile_index_transforms), [`StaggeredTransferPlan`](fragmented_array.md#StaggeredTransferPlan), [`select`](fragmented_array.md#StaggeredTransferPlan.select), [`select_if_group`](fragmented_array.md#StaggeredTransferPlan.select_if_group), [`IndexTransform`](fragmented_array.md#IndexTransform.IndexTransform), [`select`](fragmented_array.md#TrivialTransferPlan.select), [`select_if_group`](fragmented_array.md#TrivialTransferPlan.select_if_group), [`tile_index_transforms`](fragmented_array.md#TrivialTransferPlan.tile_index_transforms)
- used by: [`transfer_tiled`](fragmented_array.md#FragmentedArray.transfer_tiled), [`plan_tiled_transfer`](fragmented_array.md#plan_tiled_transfer), [`TrivialTransferPlan`](fragmented_array.md#TrivialTransferPlan), [`StaggeredTransferPlan`](fragmented_array.md#StaggeredTransferPlan), [`get_register`](fragmented_array.md#FragmentedArray.get_register), [`update_registers`](fragmented_array.md#FragmentedArray.update_registers)

### `TransferPlanDerivationError`  ·  implements/extends Exception
- def: [`jax/experimental/mosaic/gpu/fragmented_array.py:4415`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4415)
- doc: Raised when a transfer plan cannot be derived due to a known limitation.
- signature: `class TransferPlanDerivationError(Exception):`
- used by: [`plan_tiled_transfer`](fragmented_array.md#plan_tiled_transfer), [`_retry_on_failure`](dialect_lowering.md#_retry_on_failure)

### `TrivialTransferPlan`
- def: [`jax/experimental/mosaic/gpu/fragmented_array.py:4368`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4368)
- members:
  - `select(self, group_elems: Sequence[ir.Value])` — [`L4373`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4373)
  - `select_if_group(self, group_idx: int, old: ir.Value, new: ir.Value)` — [`L4377`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4377)
  - `tile_index_transforms(self)` — [`L4370`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4370)
- uses (calls/refs, reference-scoped): [`TransferPlan`](fragmented_array.md#TransferPlan)
- used by: [`transfer_tiled`](fragmented_array.md#FragmentedArray.transfer_tiled), [`plan_tiled_transfer`](fragmented_array.md#plan_tiled_transfer), [`tile_index_transforms`](fragmented_array.md#TransferPlan.tile_index_transforms), [`select`](fragmented_array.md#TransferPlan.select), [`get_base_index`](fragmented_array.md#FragmentedArray.get_base_index), [`select_if_group`](fragmented_array.md#TransferPlan.select_if_group)

### `WGSplatFragLayout`
- def: [`jax/experimental/mosaic/gpu/fragmented_array.py:633`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L633)
- doc: A fragmented array where all the values are equal represented as a register per thread.
- signature: `class WGSplatFragLayout:`
- members:
  - `can_broadcast_to(self, shape)` — [`L655`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L655) — Check that the shape can be broadcast.
  - `registers_element_type(self, t: ir.Type)` — [`L667`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L667)
  - `registers_shape(self, shape: tuple[int, ...])` — [`L670`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L670) — Returns the shape of the register array needed to represent an array of the given logical shape.
  - `shape_from_registers_shape(self, shape: tuple[int, ...])` — [`L675`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L675)
  - `thread_idxs(self, shape)` — [`L681`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L681)
  - `shape` — [`L653`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L653)
- used by: [`mlir_dtype`](fragmented_array.md#FragmentedArray.mlir_dtype), [`reduce`](fragmented_array.md#FragmentedArray.reduce), [`to_mgpu`](../../../_src/pallas/mosaic_gpu/core.md#Layout.to_mgpu), [`shape`](fragmented_array.md#FragmentedArray.shape), [`astype`](fragmented_array.md#FragmentedArray.astype), [`_pointwise`](fragmented_array.md#FragmentedArray._pointwise), [`to_layout`](fragmented_array.md#FragmentedArray.to_layout), [`_broadcast_in_dim_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_broadcast_in_dim_lowering_rule), [`_ensure_ir_value`](../../../_src/pallas/mosaic_gpu/lowering.md#_ensure_ir_value), [`store_untiled`](fragmented_array.md#FragmentedArray.store_untiled), [`_fragmented_array_from_ir`](dialect_lowering.md#_fragmented_array_from_ir), [`broadcast_in_dim`](fragmented_array.md#FragmentedArray.broadcast_in_dim), [`_populate_custom_primitive_op_block`](../../../_src/pallas/mosaic_gpu/primitives.md#_populate_custom_primitive_op_block), [`splat`](fragmented_array.md#FragmentedArray.splat), [`to_layout_attr`](layouts.md#to_layout_attr), [`concatenate`](fragmented_array.md#concatenate), [`_vector_reduction_op_lowering_rule`](dialect_lowering.md#_vector_reduction_op_lowering_rule), [`FragmentedLayout`](fragmented_array.md#FragmentedLayout.FragmentedLayout), [`reshape`](fragmented_array.md#FragmentedArray.reshape), [`_vector_load_constraint_system`](layout_inference.md#_vector_load_constraint_system), [`extract_assignment_candidates_from_reduce_equation`](layout_inference.md#extract_assignment_candidates_from_reduce_equation), [`_as_index`](../../../_src/pallas/mosaic_gpu/lowering.md#_as_index), [`__getitem__`](fragmented_array.md#FragmentedArray.__getitem__), [`__init__`](fragmented_array.md#FragmentedArray.__init__), [`_vector_concat_constraint_system`](layout_inference.md#_vector_concat_constraint_system), [`broadcast`](fragmented_array.md#FragmentedArray.broadcast), [`_async_store_smem_constraint_system`](layout_inference.md#_async_store_smem_constraint_system), [`_extract_strided_slice_constraint_system`](layout_inference.md#_extract_strided_slice_constraint_system), [`_insert_strided_slice_constraint_system`](layout_inference.md#_insert_strided_slice_constraint_system), [`_layout_cast_lowering_wg`](../../../_src/pallas/mosaic_gpu/lowering.md#_layout_cast_lowering_wg), [`_vector_extract_constraint_system`](layout_inference.md#_vector_extract_constraint_system), [`foreach`](fragmented_array.md#FragmentedArray.foreach), [`_constant_constraint_system`](layout_inference.md#_constant_constraint_system), [`_vector_extract_op_lowering_rule`](dialect_lowering.md#_vector_extract_op_lowering_rule), [`_broadcasted_iota_constraint_system`](layout_inference.md#_broadcasted_iota_constraint_system), [`_multimem_load_reduce_constraint_system`](layout_inference.md#_multimem_load_reduce_constraint_system), [`_vector_broadcast_constraint_system`](layout_inference.md#_vector_broadcast_constraint_system), [`_divf_lowering_rule`](dialect_lowering.md#_divf_lowering_rule), [`_is_e8m0_constant_one`](fragmented_array.md#FragmentedArray._is_e8m0_constant_one), [`splat_is_compatible_with_tiled`](layouts.md#splat_is_compatible_with_tiled)  (+4 more)

### `WGStridedFragLayout`
- def: [`jax/experimental/mosaic/gpu/fragmented_array.py:687`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L687)
- doc: Convert the array to 1D and then shard across threads.
- signature: `class WGStridedFragLayout:`
- members:
  - `from_shaped_type(cls, shaped_ty: ir.Type)` — [`L698`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L698) — Returns a WGStridedFragLayout for the given shaped type.
  - `linear_thread_idxs(self)` — [`L746`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L746) — The indexes to be used for vector load/store WGStridedFragLayout.
  - `registers_element_type(self, t: ir.Type)` — [`L719`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L719)
  - `registers_shape(self, shape: tuple[int, ...])` — [`L722`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L722) — Returns the shape of the register array needed to represent an array of the given logical shape.
  - `shape_from_registers_shape(self, shape: tuple[int, ...])` — [`L728`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L728)
  - `thread_idxs(self, shape)` — [`L734`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L734)
  - `shape` — [`L690`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L690)
  - `vec_size` — [`L691`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L691)
- protocol/private: `__post_init__`[`L693`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L693)
- uses (calls/refs, reference-scoped): [`bitwidth`](utils.md#bitwidth), [`c`](fragmented_array.md#c), [`WARPGROUP_SIZE`](fragmented_array.md#WARPGROUP_SIZE)
- used by: [`mlir_dtype`](fragmented_array.md#FragmentedArray.mlir_dtype), [`async_copy`](launch_context.md#LaunchContext.async_copy), [`reduce`](fragmented_array.md#FragmentedArray.reduce), [`to_mgpu`](../../../_src/pallas/mosaic_gpu/core.md#Layout.to_mgpu), [`shape`](fragmented_array.md#FragmentedArray.shape), [`astype`](fragmented_array.md#FragmentedArray.astype), [`_get_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_get_lowering_rule), [`_broadcast_in_dim_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_broadcast_in_dim_lowering_rule), [`store_untiled`](fragmented_array.md#FragmentedArray.store_untiled), [`_multimem_load_reduce_lowering_rule`](../../../_src/pallas/mosaic_gpu/primitives.md#_multimem_load_reduce_lowering_rule), [`broadcast_in_dim`](fragmented_array.md#FragmentedArray.broadcast_in_dim), [`_vector_load_op_lowering_rule`](dialect_lowering.md#_vector_load_op_lowering_rule), [`_reduce_lowering_rule`](../../../_src/pallas/mosaic_gpu/lowering.md#_reduce_lowering_rule), [`splat`](fragmented_array.md#FragmentedArray.splat), [`to_layout_attr`](layouts.md#to_layout_attr), [`load_reduce_untiled`](fragmented_array.md#FragmentedArray.load_reduce_untiled), [`load_strided`](fragmented_array.md#FragmentedArray.load_strided), [`_store_untiled_splat`](fragmented_array.md#FragmentedArray._store_untiled_splat), [`concatenate`](fragmented_array.md#concatenate), [`FragmentedLayout`](fragmented_array.md#FragmentedLayout.FragmentedLayout), [`reshape`](fragmented_array.md#FragmentedArray.reshape), [`extract_assignment_candidates_from_reduce_equation`](layout_inference.md#extract_assignment_candidates_from_reduce_equation), [`__init__`](fragmented_array.md#FragmentedArray.__init__), [`_vector_concat_constraint_system`](layout_inference.md#_vector_concat_constraint_system), [`broadcast`](fragmented_array.md#FragmentedArray.broadcast), [`transfer_strided`](fragmented_array.md#FragmentedArray.transfer_strided), [`_async_store_smem_constraint_system`](layout_inference.md#_async_store_smem_constraint_system), [`_extract_strided_slice_constraint_system`](layout_inference.md#_extract_strided_slice_constraint_system), [`_insert_strided_slice_constraint_system`](layout_inference.md#_insert_strided_slice_constraint_system), [`_multi_dim_reduction_constraint_system`](layout_inference.md#_multi_dim_reduction_constraint_system), [`_vector_extract_constraint_system`](layout_inference.md#_vector_extract_constraint_system), [`_register_layouts_for_optimized_transfer_to_smem`](layout_inference.md#_register_layouts_for_optimized_transfer_to_smem), [`is_supported_strided_layout_broadcast`](fragmented_array.md#is_supported_strided_layout_broadcast), [`_extract_layout_candidates_from_broadcast`](layout_inference.md#_extract_layout_candidates_from_broadcast), [`_from_strided_fragmented_layout_attr`](layouts.md#_from_strided_fragmented_layout_attr), [`_to_strided_fragmented_layout_attr`](layouts.md#_to_strided_fragmented_layout_attr), [`_strided_layout_for_variable`](layout_inference.md#_strided_layout_for_variable)

## Functions
- `_repack(regs_it, reg_ty)` — [`L4616`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4616)
- `_tiled_wgmma_layout(shape: tuple[int, ...])` — [`L619`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L619) — Returns the tiled layout relevant for WGMMA operations.
- `addf(a: ir.Value, b: ir.Value)` — [`L4583`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4583)
- `can_relayout_wgmma_2x_to_wgmma(bitwidth: int)` — [`L964`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L964)
- `can_relayout_wgmma_4x_to_wgmma_2x(bitwidth: int)` — [`L960`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L960)
- `concatenate(arrays: Sequence[FragmentedArray], axis: int = 0)` — [`L4874`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4874) — Concatenates fragmented arrays along the specified axis.
- `copy_tiled(src: ir.Value, dst: ir.Value, swizzle: int = 16)` — [`L4788`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4788) — Copy the data from the src reference to the dst reference.
- `enumerate_negative(elems: Sequence[T])` — [`L297`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L297) — Like built-in enumerate, but returns negative indices into the sequence.
- `has_bank_conflicts(tile_idx_transform)` — [`L4519`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4519)
- `is_supported_strided_layout_broadcast(src: WGStridedFragLayout, dst: WGStridedFragLayout, dims: tuple[int, ...])` — [`L4847`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4847) — We only support broadcasting of leading dimensions.
- `mulf(a: ir.Value, b: ir.Value)` — [`L4589`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4589)
- `optimization_barrier(a: mgpu.FragmentedArray, b: mgpu.FragmentedArray,, *arrays: mgpu.FragmentedArray)` — [`L4594`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4594)
- `plan_tiled_transfer(nested_ref_shape: Sequence[Sequence[int]], nested_ref_strides: Sequence[Sequence[int]], layout: TiledLayout, element_bits: int, swizzle: int)` — [`L4420`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4420) — Plans the tiled transfer in a way that avoids SMEM bank conflicts.
- `subf(a: ir.Value, b: ir.Value)` — [`L4586`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4586)
- `tiled_copy_smem_gmem_layout(row_tiles: int, col_tiles: int, swizzle: int, bitwidth: int)` — [`L4725`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4725)
- `tmem_native_layout(vector_length: int)` — [`L922`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L922) — A layout resembling the logical organization of TMEM.

## Module values
- `FragmentedLayout` — [`L761`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L761)
- `IndexTransform` — [`L4343`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L4343)
- `ReductionKind` — [`L54`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L54)
- `SMEM_BANKS` — [`L49`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L49)
- `SMEM_BANK_BYTES` — [`L50`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L50)
- `TCGEN05_COL_LAYOUT` — [`L914`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L914)
- `TCGEN05_LAYOUT` — [`L891`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L891)
- `TCGEN05_ROW_LAYOUT` — [`L906`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L906)
- `TCGEN05_TRANSPOSED_LAYOUT` — [`L898`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L898)
- `TMA_INDICES_4_LAYOUT` — [`L953`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L953)
- `TMA_INDICES_LAYOUT` — [`L945`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L945)
- `TMEM_NATIVE_LAYOUT` — [`L940`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L940)
- `WARPGROUP_SIZE` — [`L46`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L46)
- `WARPS_IN_WARPGROUP` — [`L48`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L48)
- `WARP_SIZE` — [`L47`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L47)
- `WGMMA_COL_LAYOUT` — [`L766`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L766)
- `WGMMA_LAYOUT` — [`L792`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L792)
- `WGMMA_LAYOUT_8BIT` — [`L825`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L825)
- `WGMMA_LAYOUT_ACC_32BIT` — [`L806`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L806)
- `WGMMA_LAYOUT_UPCAST_2X` — [`L844`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L844)
- `WGMMA_LAYOUT_UPCAST_4X` — [`L856`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L856)
- `WGMMA_ROW_LAYOUT` — [`L772`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L772)
- `WGMMA_TRANSPOSED_LAYOUT` — [`L883`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L883)
- `c` — [`L51`](../../../../../../../../raw/code/jax/jax/experimental/mosaic/gpu/fragmented_array.py#L51)

