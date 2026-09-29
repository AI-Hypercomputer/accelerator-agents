---
title: 'Module: ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.grouped_matmulv3._pallas_impl`/
symbols:
  grouped_matmulv3_pallas_impl: grouped_matmulv3_pallas_impl().
  generate_block_specs: generate_block_specs().
  kernel_main: kernel_main().
  inner_kernel._matmul: inner_kernel()._matmul().
  make_gmm_configs: make_gmm_configs().
  get_cost_estimate: get_cost_estimate().
  calculate_tiling: calculate_tiling().
  GmmConfigs.dims: GmmConfigs#dims.
  get_scope_name: get_scope_name().
  GmmConfigs.tiles: GmmConfigs#tiles.
  Dimensions.size_lhs_sublane: Dimensions#size_lhs_sublane.
  GmmConfigs.rhs_cfgs: GmmConfigs#rhs_cfgs.
  IndexMaps.rhs_scale_index_map: IndexMaps#rhs_scale_index_map().
  inner_kernel: inner_kernel().
  validate_inputs: validate_inputs().
  fill_metadata: fill_metadata().
  fill_metadata.inner_tm_loop: fill_metadata().inner_tm_loop().
  zero_out_start: zero_out_start().
  IndexMaps.lhs_index_map: IndexMaps#lhs_index_map().
  IndexMaps.out_index_map: IndexMaps#out_index_map().
  fill_metadata.outer_group_loop: fill_metadata().outer_group_loop().
  TileSizes.tile_k: TileSizes#tile_k.
  Dimensions.size_k: Dimensions#size_k.
  GmmConfigs.num_quant_blocks_per_tile_k: GmmConfigs#num_quant_blocks_per_tile_k().
  GmmConfigs.out_size_n: GmmConfigs#out_size_n().
  MetadataRef.gm_id_to_m_offset: MetadataRef#gm_id_to_m_offset.
  TileSizes.tile_n: TileSizes#tile_n.
  TileSizes: TileSizes#
  TileSizes.tile_m: TileSizes#tile_m.
  Dimensions.size_n: Dimensions#size_n.
  InputConfigs.quant_block_size: InputConfigs#quant_block_size.
  WeightsRef: WeightsRef#
  TileFn: TileFn.
  Dimensions.size_m: Dimensions#size_m.
  InputConfigs.dtype: InputConfigs#dtype.
  GmmConfigs: GmmConfigs#
  GmmConfigs.lhs_cfgs: GmmConfigs#lhs_cfgs.
  RhsRef: RhsRef#
  InputConfigs: InputConfigs#
  IndexMaps.cfgs: IndexMaps#cfgs.
  FusedWeightsRef.gate: FusedWeightsRef#gate.
  FusedWeightsRef.up: FusedWeightsRef#up.
  FusedWeightsRef.get_weight: FusedWeightsRef#get_weight().
  FusedWeightsRef.get_scale: FusedWeightsRef#get_scale().
  FusedWeightsRef.get_bias: FusedWeightsRef#get_bias().
  RhsRef.get_scale: RhsRef#get_scale().
  MetadataRef: MetadataRef#
  Dimensions: Dimensions#
  Dimensions.size_group: Dimensions#size_group.
  InputConfigs.quant_dtype: InputConfigs#quant_dtype.
  IndexMaps.metadata_ref: IndexMaps#metadata_ref.
  align_to: align_to().
  RhsRef.get_weight: RhsRef#get_weight().
  RhsRef.get_bias: RhsRef#get_bias().
  GmmConfigs.out_dtype: GmmConfigs#out_dtype.
  GmmConfigs.fuse_act: GmmConfigs#fuse_act.
  IndexMaps.rhs_weight_index_map: IndexMaps#rhs_weight_index_map().
  IndexMaps.rhs_bias_index_map: IndexMaps#rhs_bias_index_map().
  zero_out_end: zero_out_end().
  InputConfigs.should_bitcast: InputConfigs#should_bitcast().
  WeightsRef.weight: WeightsRef#weight.
  WeightsRef.scale: WeightsRef#scale.
  WeightsRef.bias: WeightsRef#bias.
  MetadataRef.gm_id_to_group_id: MetadataRef#gm_id_to_group_id.
  InputConfigs.has_scale: InputConfigs#has_scale.
  IndexMaps.__init__: IndexMaps#__init__().
  WeightsRef.get_weight: WeightsRef#get_weight().
  WeightsRef.get_scale: WeightsRef#get_scale().
  WeightsRef.get_bias: WeightsRef#get_bias().
  FusedWeightsRef: FusedWeightsRef#
  InputConfigs.has_bias: InputConfigs#has_bias.
  GmmConfigs.zero_init: GmmConfigs#zero_init.
  apply_act_fn: apply_act_fn().
  inner_kernel.matmul_first_last: inner_kernel().matmul_first_last().
  inner_kernel.matmul_first: inner_kernel().matmul_first().
  inner_kernel.matmul: inner_kernel().matmul().
  inner_kernel.matmul_last: inner_kernel().matmul_last().
  zero_out_start.left_fill_zero: zero_out_start().left_fill_zero().
  zero_out_start.right_fill_zero: zero_out_start().right_fill_zero().
  calculate_tiling._is_tile_k_quant_block_compatible: calculate_tiling()._is_tile_k_quant_block_compatible().
  get_metadata: get_metadata().
  GmmConfigs.acc_dtype: GmmConfigs#acc_dtype.
  zero_out_start.fill_zero: zero_out_start().fill_zero().
  swigluoai: swigluoai().
  Dimensions.size_lhs_group: Dimensions#size_lhs_group.
  IndexMaps: IndexMaps#
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py)

## Classes
### `Dimensions`
- def: [`ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py:202`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L202)
- doc: Problem dimensions for the v3 grouped matmul kernel.
- signature: `class Dimensions:`
- members:
  - `size_group` — [`L220`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L220) — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)
  - `size_k` — [`L218`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L218) — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)
  - `size_lhs_group` — [`L221`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L221)
  - `size_lhs_sublane` — [`L222`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L222) — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)
  - `size_m` — [`L217`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L217) — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)
  - `size_n` — [`L219`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L219) — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)
- used by: [`grouped_matmulv3_pallas_impl`](_pallas_impl.md#grouped_matmulv3_pallas_impl), [`generate_block_specs`](_pallas_impl.md#generate_block_specs), [`kernel_main`](_pallas_impl.md#kernel_main), [`_matmul`](_pallas_impl.md#inner_kernel._matmul), [`make_gmm_configs`](_pallas_impl.md#make_gmm_configs), [`get_cost_estimate`](_pallas_impl.md#get_cost_estimate), [`calculate_tiling`](_pallas_impl.md#calculate_tiling), [`dims`](_pallas_impl.md#GmmConfigs.dims), [`get_scope_name`](_pallas_impl.md#get_scope_name), [`validate_inputs`](_pallas_impl.md#validate_inputs), [`fill_metadata`](_pallas_impl.md#fill_metadata), [`inner_tm_loop`](_pallas_impl.md#fill_metadata.inner_tm_loop), [`zero_out_start`](_pallas_impl.md#zero_out_start), [`lhs_index_map`](_pallas_impl.md#IndexMaps.lhs_index_map), [`out_index_map`](_pallas_impl.md#IndexMaps.out_index_map), [`outer_group_loop`](_pallas_impl.md#fill_metadata.outer_group_loop), [`out_size_n`](_pallas_impl.md#GmmConfigs.out_size_n), [`TileFn`](_pallas_impl.md#TileFn), [`zero_out_end`](_pallas_impl.md#zero_out_end)

### `FusedWeightsRef`  ·  implements/extends RhsRef
- def: [`ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py:133`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L133)
- doc: RHS tile reference for fused gate+up projections (e.g. SwiGLU / SiLU).
- signature: `class FusedWeightsRef(RhsRef):`
- members:
  - `get_bias(self)` — [`L159`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L159) — Concatenate gate and up bias tiles along axis -1. — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)
  - `get_scale(self)` — [`L153`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L153) — Concatenate gate and up scale tiles along axis -1. — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)
  - `get_weight(self)` — [`L147`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L147) — Concatenate gate and up weight tiles along axis -1. — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)
  - `gate` — [`L144`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L144) — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)
  - `up` — [`L145`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L145) — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)
- uses (calls/refs, reference-scoped): [`WeightsRef`](_pallas_impl.md#WeightsRef), [`RhsRef`](_pallas_impl.md#RhsRef), [`get_bias`](_pallas_impl.md#WeightsRef.get_bias), [`get_scale`](_pallas_impl.md#WeightsRef.get_scale), [`get_weight`](_pallas_impl.md#WeightsRef.get_weight)
- used by: [`kernel_main`](_pallas_impl.md#kernel_main), [`RhsRef`](_pallas_impl.md#RhsRef), [`get_scale`](_pallas_impl.md#RhsRef.get_scale), [`get_bias`](_pallas_impl.md#RhsRef.get_bias), [`get_weight`](_pallas_impl.md#RhsRef.get_weight)

### `GmmConfigs`
- def: [`ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py:254`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L254)
- doc: Full configuration bundle for one grouped matmul v3 kernel launch.
- signature: `class GmmConfigs:`
- members:
  - `num_quant_blocks_per_tile_k(self)` — [`L280`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L280) — Number of quantisation blocks per tile along the K dimension.
  - `out_size_n(self)` — [`L285`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L285) — Actual output N dimension (halved when a fused activation is active).
  - `acc_dtype` — [`L275`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L275)
  - `dims` — [`L271`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L271)
  - `fuse_act` — [`L277`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L277)
  - `lhs_cfgs` — [`L272`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L272)
  - `out_dtype` — [`L274`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L274)
  - `rhs_cfgs` — [`L273`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L273)
  - `tiles` — [`L270`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L270)
  - `zero_init` — [`L276`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L276)
- uses (calls/refs, reference-scoped): [`tile_k`](_pallas_impl.md#TileSizes.tile_k), [`TileSizes`](_pallas_impl.md#TileSizes), [`size_n`](_pallas_impl.md#Dimensions.size_n), [`quant_block_size`](_pallas_impl.md#InputConfigs.quant_block_size), [`InputConfigs`](_pallas_impl.md#InputConfigs), [`Dimensions`](_pallas_impl.md#Dimensions)
- used by: [`grouped_matmulv3_pallas_impl`](_pallas_impl.md#grouped_matmulv3_pallas_impl), [`generate_block_specs`](_pallas_impl.md#generate_block_specs), [`kernel_main`](_pallas_impl.md#kernel_main), [`_matmul`](_pallas_impl.md#inner_kernel._matmul), [`make_gmm_configs`](_pallas_impl.md#make_gmm_configs), [`get_cost_estimate`](_pallas_impl.md#get_cost_estimate), [`get_scope_name`](_pallas_impl.md#get_scope_name), [`rhs_scale_index_map`](_pallas_impl.md#IndexMaps.rhs_scale_index_map), [`inner_kernel`](_pallas_impl.md#inner_kernel), [`fill_metadata`](_pallas_impl.md#fill_metadata), [`inner_tm_loop`](_pallas_impl.md#fill_metadata.inner_tm_loop), [`lhs_index_map`](_pallas_impl.md#IndexMaps.lhs_index_map), [`out_index_map`](_pallas_impl.md#IndexMaps.out_index_map), [`outer_group_loop`](_pallas_impl.md#fill_metadata.outer_group_loop), [`__init__`](_pallas_impl.md#IndexMaps.__init__), [`get_metadata`](_pallas_impl.md#get_metadata)

### `IndexMaps`
- def: [`ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py:295`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L295)
- doc: Pallas `BlockSpec` index-map callables for the v3 GMM kernel.
- signature: `class IndexMaps:`
- members:
  - `lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array)` — [`L312`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L312) — Return the LHS ``BoundedSlice`` tile index for grid step ``gm_id``.
  - `out_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array)` — [`L345`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L345) — Return the output ``BoundedSlice`` tile index for grid step ``gm_id``.
  - `rhs_bias_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array)` — [`L332`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L332) — Return the RHS bias tile index ``(group_id, 0, n_id)``.
  - `rhs_scale_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array)` — [`L337`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L337) — Return the RHS scale tile index ``(group_id, b_tile_id, 0, n_id)``.
  - `rhs_weight_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array)` — [`L327`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L327) — Return the RHS weight tile index ``(group_id, k_id, n_id)``.
  - `cfgs` — [`L310`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L310)
  - `metadata_ref` — [`L309`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L309)
- protocol/private: `__init__`[`L308`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L308)
- uses (calls/refs, reference-scoped): [`dims`](_pallas_impl.md#GmmConfigs.dims), [`tiles`](_pallas_impl.md#GmmConfigs.tiles), [`size_lhs_sublane`](_pallas_impl.md#Dimensions.size_lhs_sublane), [`rhs_cfgs`](_pallas_impl.md#GmmConfigs.rhs_cfgs), [`num_quant_blocks_per_tile_k`](_pallas_impl.md#GmmConfigs.num_quant_blocks_per_tile_k), [`tile_k`](_pallas_impl.md#TileSizes.tile_k), [`gm_id_to_m_offset`](_pallas_impl.md#MetadataRef.gm_id_to_m_offset), [`quant_block_size`](_pallas_impl.md#InputConfigs.quant_block_size), [`GmmConfigs`](_pallas_impl.md#GmmConfigs), [`MetadataRef`](_pallas_impl.md#MetadataRef), [`gm_id_to_group_id`](_pallas_impl.md#MetadataRef.gm_id_to_group_id)
- used by: [`generate_block_specs`](_pallas_impl.md#generate_block_specs)

### `InputConfigs`
- def: [`ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py:226`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L226)
- doc: Per-operand (LHS or RHS) quantisation and dtype configuration.
- signature: `class InputConfigs:`
- members:
  - `should_bitcast(self)` — [`L247`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L247) — Return True when the dtype requires a bitcast (sub-byte elements).
  - `dtype` — [`L242`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L242)
  - `has_bias` — [`L243`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L243)
  - `has_scale` — [`L244`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L244)
  - `quant_block_size` — [`L241`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L241)
  - `quant_dtype` — [`L240`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L240)
- used by: [`generate_block_specs`](_pallas_impl.md#generate_block_specs), [`kernel_main`](_pallas_impl.md#kernel_main), [`_matmul`](_pallas_impl.md#inner_kernel._matmul), [`make_gmm_configs`](_pallas_impl.md#make_gmm_configs), [`get_cost_estimate`](_pallas_impl.md#get_cost_estimate), [`calculate_tiling`](_pallas_impl.md#calculate_tiling), [`rhs_cfgs`](_pallas_impl.md#GmmConfigs.rhs_cfgs), [`rhs_scale_index_map`](_pallas_impl.md#IndexMaps.rhs_scale_index_map), [`num_quant_blocks_per_tile_k`](_pallas_impl.md#GmmConfigs.num_quant_blocks_per_tile_k), [`TileFn`](_pallas_impl.md#TileFn), [`lhs_cfgs`](_pallas_impl.md#GmmConfigs.lhs_cfgs), [`_is_tile_k_quant_block_compatible`](_pallas_impl.md#calculate_tiling._is_tile_k_quant_block_compatible)

### `MetadataRef`
- def: [`ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py:168`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L168)
- doc: SMEM scratch holding the per-grid-step group and row-offset mappings.
- signature: `class MetadataRef:`
- members:
  - `gm_id_to_group_id` — [`L182`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L182)
  - `gm_id_to_m_offset` — [`L183`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L183)
- used by: [`grouped_matmulv3_pallas_impl`](_pallas_impl.md#grouped_matmulv3_pallas_impl), [`generate_block_specs`](_pallas_impl.md#generate_block_specs), [`kernel_main`](_pallas_impl.md#kernel_main), [`_matmul`](_pallas_impl.md#inner_kernel._matmul), [`rhs_scale_index_map`](_pallas_impl.md#IndexMaps.rhs_scale_index_map), [`inner_kernel`](_pallas_impl.md#inner_kernel), [`fill_metadata`](_pallas_impl.md#fill_metadata), [`inner_tm_loop`](_pallas_impl.md#fill_metadata.inner_tm_loop), [`zero_out_start`](_pallas_impl.md#zero_out_start), [`lhs_index_map`](_pallas_impl.md#IndexMaps.lhs_index_map), [`out_index_map`](_pallas_impl.md#IndexMaps.out_index_map), [`rhs_bias_index_map`](_pallas_impl.md#IndexMaps.rhs_bias_index_map), [`rhs_weight_index_map`](_pallas_impl.md#IndexMaps.rhs_weight_index_map), [`__init__`](_pallas_impl.md#IndexMaps.__init__)

### `RhsRef`  ·  implements/extends ABC
- def: [`ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py:74`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L74)
- doc: Abstract base for RHS tile references inside the Pallas kernel.
- signature: `class RhsRef(ABC):`
- members:
  - `get_bias(self)` — [`L93`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L93) — Return the per-group bias tile.
  - `get_scale(self)` — [`L88`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L88) — Return the per-block quantisation scale tile.
  - `get_weight(self)` — [`L83`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L83) — Return the raw weight tile (possibly bitcast from a packed dtype).
- uses (calls/refs, reference-scoped): [`WeightsRef`](_pallas_impl.md#WeightsRef), [`get_bias`](_pallas_impl.md#FusedWeightsRef.get_bias), [`get_scale`](_pallas_impl.md#FusedWeightsRef.get_scale), [`get_weight`](_pallas_impl.md#FusedWeightsRef.get_weight), [`FusedWeightsRef`](_pallas_impl.md#FusedWeightsRef), [`get_bias`](_pallas_impl.md#WeightsRef.get_bias), [`get_scale`](_pallas_impl.md#WeightsRef.get_scale), [`get_weight`](_pallas_impl.md#WeightsRef.get_weight)
- used by: [`_matmul`](_pallas_impl.md#inner_kernel._matmul), [`inner_kernel`](_pallas_impl.md#inner_kernel), [`WeightsRef`](_pallas_impl.md#WeightsRef), [`FusedWeightsRef`](_pallas_impl.md#FusedWeightsRef)

### `TileSizes`
- def: [`ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py:187`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L187)
- doc: Static tile dimensions for the v3 grouped matmul kernel.
- signature: `class TileSizes:`
- members:
  - `tile_k` — [`L197`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L197)
  - `tile_m` — [`L196`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L196)
  - `tile_n` — [`L198`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L198)
- used by: [`grouped_matmulv3_pallas_impl`](_pallas_impl.md#grouped_matmulv3_pallas_impl), [`generate_block_specs`](_pallas_impl.md#generate_block_specs), [`kernel_main`](_pallas_impl.md#kernel_main), [`_matmul`](_pallas_impl.md#inner_kernel._matmul), [`make_gmm_configs`](_pallas_impl.md#make_gmm_configs), [`calculate_tiling`](_pallas_impl.md#calculate_tiling), [`get_scope_name`](_pallas_impl.md#get_scope_name), [`tiles`](_pallas_impl.md#GmmConfigs.tiles), [`rhs_scale_index_map`](_pallas_impl.md#IndexMaps.rhs_scale_index_map), [`_normalize_tiling`](_interface.md#_normalize_tiling), [`inner_tm_loop`](_pallas_impl.md#fill_metadata.inner_tm_loop), [`_wrapped_tile_fn`](_interface.md#_normalize_tiling._wrapped_tile_fn), [`outer_group_loop`](_pallas_impl.md#fill_metadata.outer_group_loop), [`num_quant_blocks_per_tile_k`](_pallas_impl.md#GmmConfigs.num_quant_blocks_per_tile_k), [`TileFn`](_pallas_impl.md#TileFn)

### `WeightsRef`  ·  implements/extends RhsRef
- def: [`ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py:100`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L100)
- doc: Concrete RHS tile reference for a single (non-fused) weight.
- signature: `class WeightsRef(RhsRef):`
- members:
  - `get_bias(self)` — [`L125`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L125) — Load and return the bias tile; asserts bias is not None.
  - `get_scale(self)` — [`L120`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L120) — Load and return the scale tile; asserts scale is not None.
  - `get_weight(self)` — [`L116`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L116) — Load and return the weight tile from VMEM.
  - `bias` — [`L114`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L114)
  - `scale` — [`L113`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L113)
  - `weight` — [`L112`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L112)
- uses (calls/refs, reference-scoped): [`RhsRef`](_pallas_impl.md#RhsRef)
- used by: [`grouped_matmulv3_pallas_impl`](_pallas_impl.md#grouped_matmulv3_pallas_impl), [`generate_block_specs`](_pallas_impl.md#generate_block_specs), [`kernel_main`](_pallas_impl.md#kernel_main), [`RhsRef`](_pallas_impl.md#RhsRef), [`gate`](_pallas_impl.md#FusedWeightsRef.gate), [`get_bias`](_pallas_impl.md#FusedWeightsRef.get_bias), [`get_scale`](_pallas_impl.md#FusedWeightsRef.get_scale), [`get_weight`](_pallas_impl.md#FusedWeightsRef.get_weight), [`up`](_pallas_impl.md#FusedWeightsRef.up), [`get_scale`](_pallas_impl.md#RhsRef.get_scale), [`get_bias`](_pallas_impl.md#RhsRef.get_bias), [`get_weight`](_pallas_impl.md#RhsRef.get_weight)

## Functions
- `_is_tile_k_quant_block_compatible(tk: int)` — [`L939`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L939)
- `_matmul(is_first_k_step: bool, is_last_k_step: bool)` — [`L469`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L469) — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)
- `align_to(x, a)` — [`L69`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L69) — Round ``x`` up to the nearest multiple of ``a``.
- `apply_act_fn(acc: jax.Array, fuse_act: str | None)` — [`L52`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L52) — Apply an optional fused activation to the accumulator.
- `calculate_tiling(dims: Dimensions, lhs_cfgs: InputConfigs, rhs_cfgs: InputConfigs, vmem_limit_bytes: int, fuse_act: str | None = None)` — [`L880`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L880) — Automatically choose tile sizes for the v3 grouped matmul kernel.
- `fill_metadata(lhs_group_sizes_ref: jax.Array, group_offset_ref: jax.Array, metadata_ref: MetadataRef, *, cfgs: GmmConfigs)` — [`L613`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L613) — Populate SMEM metadata for the grouped matmul grid.
- `fill_zero(i, zero_size, *, start, end)` — [`L736`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L736)
- `generate_block_specs(metadata_ref: MetadataRef, cfgs: GmmConfigs)` — [`L364`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L364) — Build Pallas ``BlockSpec`` descriptors for LHS, RHS, and output tensors. — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)
- `get_cost_estimate(cfgs: GmmConfigs)` — [`L1040`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L1040) — Build a ``pl.CostEstimate`` for the given v3 kernel configuration.
- `get_metadata(cfgs: GmmConfigs)` — [`L1204`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L1204) — Flatten ``GmmConfigs`` into a dict of scalar values for XProf metadata.
- `get_scope_name(cfgs: GmmConfigs)` — [`L1082`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L1082) — Return a human-readable XProf scope name for the v3 kernel launch.
- `grouped_matmulv3_pallas_impl(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, rhs_scale: jax.Array | None = None, rhs_bias: jax.Array | None = None, group_offset: jax.Array | None = None, *, tile_info: TileSizes | TileFn = calculate_tiling, vmem_limit_bytes: int | None = None, precision: jax.lax.Precision = jax.lax.Precision.DEFAULT, preferred_element_type: jnp.dtype | None = None, acc_dtype: jnp.dtype | None = None, maybe_quantize_lhs: bool = True, zero_initialize: bool = True, fuse_act: str | None = None, interpret: bool = False)` — [`L1240`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L1240) — Core TPU Pallas grouped matmul v3 using ``pltpu.emit_pipeline``. — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)
- `inner_kernel(tiled_lhs_ref: jax.Array, tiled_rhs_ref: RhsRef, tiled_out_ref: jax.Array, partial_out_ref: jax.Array, acc_ref: jax.Array, metadata_ref: MetadataRef, *, cfgs: GmmConfigs)` — [`L431`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L431) — Inner pipeline body invoked by ``emit_pipeline`` for each tile. — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)
- `inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id)` — [`L647`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L647)
- `kernel_main(lhs_group_sizes_ref: jax.Array, group_offset_ref: jax.Array, lhs_ref: jax.Array, rhs_ref: WeightsRef, out_ref: jax.Array, partial_out_ref: jax.Array, acc_ref: jax.Array, metadata_ref: MetadataRef, zero_ref: jax.Array | None, semaphore_ref: jax.Array | None, *, cfgs: GmmConfigs)` — [`L791`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L791) — Top-level Pallas kernel body for grouped matmul v3. — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)
- `left_fill_zero(i, zero_size)` — [`L752`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L752)
- `make_gmm_configs(lhs: jax.Array, rhs: jax.Array, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None, group_sizes: jax.Array, group_offset: jax.Array, *, tile_info: TileSizes | TileFn, vmem_limit_bytes: int | None, out_dtype: jnp.dtype | None, acc_dtype: jnp.dtype | None, maybe_quantize_lhs: bool, zero_initialize: bool, fuse_act: str | None = None)` — [`L1092`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L1092) — Validate inputs and build the full ``GmmConfigs`` bundle.
- `matmul()` — [`L594`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L594)
- `matmul_first()` — [`L590`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L590)
- `matmul_first_last()` — [`L586`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L586)
- `matmul_last()` — [`L598`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L598)
- `outer_group_loop(lhs_group_id, carry)` — [`L658`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L658)
- `right_fill_zero(i, zero_size)` — [`L756`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L756)
- `swigluoai(gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7)` — [`L38`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L38) — Activation used in some models such as GPT-OSS.
- `validate_inputs(lhs: jax.Array, rhs: jax.Array, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None, group_sizes: jax.Array, group_offset: jax.Array, fuse_act: str | None = None)` — [`L969`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L969) — Validate inputs and return problem ``Dimensions`` for the v3 kernel.
- `zero_out_end(out_ref: jax.Array, semaphore_ref: jax.Array, zero_size: jax.Array, *, dims: Dimensions)` — [`L764`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L764) — Wait for the zero-fill DMA copies initiated by ``zero_out_start``. — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)
- `zero_out_start(out_ref: jax.Array, zero_ref: jax.Array, semaphore_ref: jax.Array, metadata_ref: MetadataRef, num_gm: jax.Array, *, dims: Dimensions)` — [`L681`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L681) — Asynchronously zero output rows outside the active compute range. — documented in [ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-grouped_matmulv3-_pallas_impl.md)

## Module values
- `TileFn` — [`L292`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L292)
- `__all__` — [`L1409`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py#L1409)

