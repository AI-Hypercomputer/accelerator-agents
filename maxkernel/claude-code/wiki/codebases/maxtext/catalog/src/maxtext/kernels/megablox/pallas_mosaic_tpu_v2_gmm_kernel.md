---
title: 'Module: src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py'
type: catalog
provenance: extracted
module: src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.kernels.megablox.pallas_mosaic_tpu_v2_gmm_kernel`/
symbols:
  gmm_v2: gmm_v2().
  generate_block_specs: generate_block_specs().
  inner_kernel._matmul: inner_kernel()._matmul().
  kernel_main: kernel_main().
  make_gmm_configs: make_gmm_configs().
  get_cost_estimate: get_cost_estimate().
  calculate_tiling: calculate_tiling().
  GmmConfigs.dims: GmmConfigs#dims.
  get_scope_name: get_scope_name().
  Dimensions.size_lhs_sublane: Dimensions#size_lhs_sublane.
  GmmConfigs.tiles: GmmConfigs#tiles.
  GmmConfigs.rhs_cfgs: GmmConfigs#rhs_cfgs.
  IndexMaps.rhs_scale_index_map: IndexMaps#rhs_scale_index_map().
  inner_kernel: inner_kernel().
  validate_inputs: validate_inputs().
  calculate_tiling._gmm_vmem_estimate: calculate_tiling()._gmm_vmem_estimate().
  fill_metadata: fill_metadata().
  fill_metadata.inner_tm_loop: fill_metadata().inner_tm_loop().
  zero_out_start: zero_out_start().
  InputConfigs.quant_block_size: InputConfigs#quant_block_size.
  MetadataRef.gm_id_to_m_offset: MetadataRef#gm_id_to_m_offset.
  GmmConfigs.num_quant_blocks_per_tile_k: GmmConfigs#num_quant_blocks_per_tile_k().
  IndexMaps.lhs_index_map: IndexMaps#lhs_index_map().
  IndexMaps.out_index_map: IndexMaps#out_index_map().
  IndexMaps.ps_index_map: IndexMaps#ps_index_map().
  fill_metadata.outer_group_loop: fill_metadata().outer_group_loop().
  GmmConfigs.out_size_n: GmmConfigs#out_size_n().
  TileSizes.tile_k: TileSizes#tile_k.
  Dimensions.size_k: Dimensions#size_k.
  TileSizes.tile_n: TileSizes#tile_n.
  WeightsRef: WeightsRef#
  Dimensions.size_n: Dimensions#size_n.
  InputConfigs.dtype: InputConfigs#dtype.
  IndexMaps.cfgs: IndexMaps#cfgs.
  TileFn: TileFn.
  TileSizes.tile_m: TileSizes#tile_m.
  Dimensions.size_m: Dimensions#size_m.
  GmmConfigs: GmmConfigs#
  IndexMaps.metadata_ref: IndexMaps#metadata_ref.
  GmmConfigs.lhs_cfgs: GmmConfigs#lhs_cfgs.
  RhsRef: RhsRef#
  RhsRef.get_scale: RhsRef#get_scale().
  InputConfigs: InputConfigs#
  InputConfigs.quant_dtype: InputConfigs#quant_dtype.
  InputConfigs.should_dequantize_after_matmul: InputConfigs#should_dequantize_after_matmul().
  FusedWeightsRef.gate: FusedWeightsRef#gate.
  FusedWeightsRef.up: FusedWeightsRef#up.
  FusedWeightsRef.get_weight: FusedWeightsRef#get_weight().
  FusedWeightsRef.get_scale: FusedWeightsRef#get_scale().
  FusedWeightsRef.get_bias: FusedWeightsRef#get_bias().
  MetadataRef: MetadataRef#
  TileSizes: TileSizes#
  Dimensions: Dimensions#
  Dimensions.size_group: Dimensions#size_group.
  InputConfigs.should_dequantize_before_matmul: InputConfigs#should_dequantize_before_matmul().
  align_to: align_to().
  RhsRef.get_weight: RhsRef#get_weight().
  RhsRef.get_bias: RhsRef#get_bias().
  WeightsRef.weight: WeightsRef#weight.
  WeightsRef.scale: WeightsRef#scale.
  WeightsRef.bias: WeightsRef#bias.
  InputConfigs.has_scale: InputConfigs#has_scale.
  GmmConfigs.out_dtype: GmmConfigs#out_dtype.
  GmmConfigs.fuse_act: GmmConfigs#fuse_act.
  IndexMaps.rhs_weight_index_map: IndexMaps#rhs_weight_index_map().
  IndexMaps.rhs_bias_index_map: IndexMaps#rhs_bias_index_map().
  zero_out_end: zero_out_end().
  InputConfigs.should_bitcast: InputConfigs#should_bitcast().
  MetadataRef.gm_id_to_group_id: MetadataRef#gm_id_to_group_id.
  InputConfigs.has_bias: InputConfigs#has_bias.
  IndexMaps.__init__: IndexMaps#__init__().
  WeightsRef.get_weight: WeightsRef#get_weight().
  WeightsRef.get_scale: WeightsRef#get_scale().
  WeightsRef.get_bias: WeightsRef#get_bias().
  FusedWeightsRef: FusedWeightsRef#
  GmmConfigs.has_partial_sum: GmmConfigs#has_partial_sum.
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
---
# Module: [`src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py)

## Classes
### `Dimensions`
- def: [`src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py:160`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L160)
- signature: `class Dimensions:`
- members:
  - `size_group` — [`L164`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L164) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `size_k` — [`L162`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L162) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `size_lhs_group` — [`L165`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L165)
  - `size_lhs_sublane` — [`L166`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L166) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `size_m` — [`L161`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L161) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `size_n` — [`L163`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L163) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- used by: [`gmm_v2`](pallas_mosaic_tpu_v2_gmm_kernel.md#gmm_v2), [`generate_block_specs`](pallas_mosaic_tpu_v2_gmm_kernel.md#generate_block_specs), [`_matmul`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel._matmul), [`kernel_main`](pallas_mosaic_tpu_v2_gmm_kernel.md#kernel_main), [`make_gmm_configs`](pallas_mosaic_tpu_v2_gmm_kernel.md#make_gmm_configs), [`get_cost_estimate`](pallas_mosaic_tpu_v2_gmm_kernel.md#get_cost_estimate), [`calculate_tiling`](pallas_mosaic_tpu_v2_gmm_kernel.md#calculate_tiling), [`dims`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.dims), [`get_scope_name`](pallas_mosaic_tpu_v2_gmm_kernel.md#get_scope_name), [`validate_inputs`](pallas_mosaic_tpu_v2_gmm_kernel.md#validate_inputs), [`_gmm_vmem_estimate`](pallas_mosaic_tpu_v2_gmm_kernel.md#calculate_tiling._gmm_vmem_estimate), [`fill_metadata`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata), [`inner_tm_loop`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata.inner_tm_loop), [`zero_out_start`](pallas_mosaic_tpu_v2_gmm_kernel.md#zero_out_start), [`lhs_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.lhs_index_map), [`out_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.out_index_map), [`outer_group_loop`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata.outer_group_loop), [`ps_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.ps_index_map), [`out_size_n`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.out_size_n), [`TileFn`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileFn), [`zero_out_end`](pallas_mosaic_tpu_v2_gmm_kernel.md#zero_out_end)

### `FusedWeightsRef`  ·  implements/extends RhsRef
- def: [`src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py:123`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L123)
- doc: Dataclass for gate and up weights used in fused activation.
- signature: `class FusedWeightsRef(RhsRef):`
- members:
  - `get_bias(self)` — [`L139`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L139)
  - `get_scale(self)` — [`L134`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L134)
  - `get_weight(self)` — [`L129`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L129)
  - `gate` — [`L126`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L126) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `up` — [`L127`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L127) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- uses (calls/refs, reference-scoped): [`WeightsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef), [`RhsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef), [`get_bias`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef.get_bias), [`get_scale`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef.get_scale), [`get_weight`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef.get_weight)
- used by: [`kernel_main`](pallas_mosaic_tpu_v2_gmm_kernel.md#kernel_main), [`RhsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef), [`get_scale`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef.get_scale), [`get_bias`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef.get_bias), [`get_weight`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef.get_weight)

### `GmmConfigs`
- def: [`src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py:198`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L198) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- doc: Full configuration details for GMM execution.
- signature: `class GmmConfigs:`
- members:
  - `num_quant_blocks_per_tile_k(self)` — [`L212`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L212) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `out_size_n(self)` — [`L216`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L216) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `acc_dtype` — [`L206`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L206)
  - `dims` — [`L202`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L202) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `fuse_act` — [`L209`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L209) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `has_partial_sum` — [`L207`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L207)
  - `lhs_cfgs` — [`L203`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L203) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `out_dtype` — [`L205`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L205) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `rhs_cfgs` — [`L204`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L204) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `tiles` — [`L201`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L201) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `zero_init` — [`L208`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L208)
- uses (calls/refs, reference-scoped): [`quant_block_size`](pallas_mosaic_tpu_v2_gmm_kernel.md#InputConfigs.quant_block_size), [`tile_k`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes.tile_k), [`size_n`](pallas_mosaic_tpu_v2_gmm_kernel.md#Dimensions.size_n), [`InputConfigs`](pallas_mosaic_tpu_v2_gmm_kernel.md#InputConfigs), [`Dimensions`](pallas_mosaic_tpu_v2_gmm_kernel.md#Dimensions), [`TileSizes`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes)
- used by: [`gmm_v2`](pallas_mosaic_tpu_v2_gmm_kernel.md#gmm_v2), [`generate_block_specs`](pallas_mosaic_tpu_v2_gmm_kernel.md#generate_block_specs), [`_matmul`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel._matmul), [`kernel_main`](pallas_mosaic_tpu_v2_gmm_kernel.md#kernel_main), [`make_gmm_configs`](pallas_mosaic_tpu_v2_gmm_kernel.md#make_gmm_configs), [`get_cost_estimate`](pallas_mosaic_tpu_v2_gmm_kernel.md#get_cost_estimate), [`get_scope_name`](pallas_mosaic_tpu_v2_gmm_kernel.md#get_scope_name), [`rhs_scale_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.rhs_scale_index_map), [`inner_kernel`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel), [`fill_metadata`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata), [`inner_tm_loop`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata.inner_tm_loop), [`lhs_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.lhs_index_map), [`out_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.out_index_map), [`outer_group_loop`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata.outer_group_loop), [`ps_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.ps_index_map), [`__init__`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.__init__), [`get_metadata`](pallas_mosaic_tpu_v2_gmm_kernel.md#get_metadata)

### `IndexMaps`
- def: [`src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py:226`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L226)
- doc: Index maps for GMM kernel.
- signature: `class IndexMaps:`
- members:
  - `lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array)` — [`L233`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L233) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `out_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array)` — [`L260`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L260) — Calculates index map for the output tensor. — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `ps_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array)` — [`L274`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L274) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `rhs_bias_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array)` — [`L247`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L247) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `rhs_scale_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array)` — [`L251`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L251) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `rhs_weight_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array)` — [`L243`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L243) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `cfgs` — [`L231`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L231)
  - `metadata_ref` — [`L230`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L230)
- protocol/private: `__init__`[`L229`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L229)
- uses (calls/refs, reference-scoped): [`dims`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.dims), [`size_lhs_sublane`](pallas_mosaic_tpu_v2_gmm_kernel.md#Dimensions.size_lhs_sublane), [`tiles`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.tiles), [`rhs_cfgs`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.rhs_cfgs), [`quant_block_size`](pallas_mosaic_tpu_v2_gmm_kernel.md#InputConfigs.quant_block_size), [`gm_id_to_m_offset`](pallas_mosaic_tpu_v2_gmm_kernel.md#MetadataRef.gm_id_to_m_offset), [`num_quant_blocks_per_tile_k`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.num_quant_blocks_per_tile_k), [`tile_k`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes.tile_k), [`GmmConfigs`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs), [`MetadataRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#MetadataRef), [`gm_id_to_group_id`](pallas_mosaic_tpu_v2_gmm_kernel.md#MetadataRef.gm_id_to_group_id)
- used by: [`generate_block_specs`](pallas_mosaic_tpu_v2_gmm_kernel.md#generate_block_specs)

### `InputConfigs`
- def: [`src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py:170`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L170)
- doc: Configuration parameters for input tensors.
- signature: `class InputConfigs:`
- members:
  - `should_bitcast(self)` — [`L180`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L180)
  - `should_dequantize_after_matmul(self)` — [`L193`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L193) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `should_dequantize_before_matmul(self)` — [`L185`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L185) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `dtype` — [`L175`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L175) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `has_bias` — [`L176`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L176)
  - `has_scale` — [`L177`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L177) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `quant_block_size` — [`L174`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L174) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `quant_dtype` — [`L173`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L173) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- used by: [`generate_block_specs`](pallas_mosaic_tpu_v2_gmm_kernel.md#generate_block_specs), [`_matmul`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel._matmul), [`kernel_main`](pallas_mosaic_tpu_v2_gmm_kernel.md#kernel_main), [`make_gmm_configs`](pallas_mosaic_tpu_v2_gmm_kernel.md#make_gmm_configs), [`get_cost_estimate`](pallas_mosaic_tpu_v2_gmm_kernel.md#get_cost_estimate), [`calculate_tiling`](pallas_mosaic_tpu_v2_gmm_kernel.md#calculate_tiling), [`rhs_cfgs`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.rhs_cfgs), [`rhs_scale_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.rhs_scale_index_map), [`_gmm_vmem_estimate`](pallas_mosaic_tpu_v2_gmm_kernel.md#calculate_tiling._gmm_vmem_estimate), [`num_quant_blocks_per_tile_k`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.num_quant_blocks_per_tile_k), [`TileFn`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileFn), [`lhs_cfgs`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.lhs_cfgs), [`_is_tile_k_quant_block_compatible`](pallas_mosaic_tpu_v2_gmm_kernel.md#calculate_tiling._is_tile_k_quant_block_compatible)

### `MetadataRef`
- def: [`src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py:147`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L147) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- signature: `class MetadataRef:`
- members:
  - `gm_id_to_group_id` — [`L148`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L148)
  - `gm_id_to_m_offset` — [`L149`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L149) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- used by: [`gmm_v2`](pallas_mosaic_tpu_v2_gmm_kernel.md#gmm_v2), [`generate_block_specs`](pallas_mosaic_tpu_v2_gmm_kernel.md#generate_block_specs), [`_matmul`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel._matmul), [`kernel_main`](pallas_mosaic_tpu_v2_gmm_kernel.md#kernel_main), [`rhs_scale_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.rhs_scale_index_map), [`inner_kernel`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel), [`fill_metadata`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata), [`inner_tm_loop`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata.inner_tm_loop), [`zero_out_start`](pallas_mosaic_tpu_v2_gmm_kernel.md#zero_out_start), [`lhs_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.lhs_index_map), [`out_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.out_index_map), [`ps_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.ps_index_map), [`rhs_bias_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.rhs_bias_index_map), [`rhs_weight_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.rhs_weight_index_map), [`__init__`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.__init__)

### `RhsRef`  ·  implements/extends ABC
- def: [`src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py:84`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L84)
- doc: Abstract class that defines interfaces for rhs values.
- signature: `class RhsRef(ABC):`
- members:
  - `get_bias(self)` — [`L96`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L96) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `get_scale(self)` — [`L92`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L92) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `get_weight(self)` — [`L88`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L88) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- uses (calls/refs, reference-scoped): [`WeightsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef), [`get_bias`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef.get_bias), [`get_scale`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef.get_scale), [`get_weight`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef.get_weight), [`FusedWeightsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef), [`get_bias`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef.get_bias), [`get_scale`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef.get_scale), [`get_weight`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef.get_weight)
- used by: [`_matmul`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel._matmul), [`inner_kernel`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel), [`WeightsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef), [`FusedWeightsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef)

### `TileSizes`
- def: [`src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py:153`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L153) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- signature: `class TileSizes:`
- members:
  - `tile_k` — [`L155`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L155) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `tile_m` — [`L154`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L154) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `tile_n` — [`L156`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L156) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- used by: [`gmm_v2`](pallas_mosaic_tpu_v2_gmm_kernel.md#gmm_v2), [`generate_block_specs`](pallas_mosaic_tpu_v2_gmm_kernel.md#generate_block_specs), [`_matmul`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel._matmul), [`kernel_main`](pallas_mosaic_tpu_v2_gmm_kernel.md#kernel_main), [`make_gmm_configs`](pallas_mosaic_tpu_v2_gmm_kernel.md#make_gmm_configs), [`calculate_tiling`](pallas_mosaic_tpu_v2_gmm_kernel.md#calculate_tiling), [`get_scope_name`](pallas_mosaic_tpu_v2_gmm_kernel.md#get_scope_name), [`tiles`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.tiles), [`rhs_scale_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.rhs_scale_index_map), [`inner_tm_loop`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata.inner_tm_loop), [`num_quant_blocks_per_tile_k`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.num_quant_blocks_per_tile_k), [`outer_group_loop`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata.outer_group_loop), [`TileFn`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileFn)

### `WeightsRef`  ·  implements/extends RhsRef
- def: [`src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py:102`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L102) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- doc: Dataclass for a single weights.
- signature: `class WeightsRef(RhsRef):`
- members:
  - `get_bias(self)` — [`L116`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L116)
  - `get_scale(self)` — [`L112`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L112)
  - `get_weight(self)` — [`L109`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L109)
  - `bias` — [`L107`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L107) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `scale` — [`L106`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L106) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `weight` — [`L105`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L105) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- uses (calls/refs, reference-scoped): [`RhsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef)
- used by: [`gmm_v2`](pallas_mosaic_tpu_v2_gmm_kernel.md#gmm_v2), [`generate_block_specs`](pallas_mosaic_tpu_v2_gmm_kernel.md#generate_block_specs), [`kernel_main`](pallas_mosaic_tpu_v2_gmm_kernel.md#kernel_main), [`RhsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef), [`get_scale`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef.get_scale), [`gate`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef.gate), [`get_bias`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef.get_bias), [`get_scale`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef.get_scale), [`get_weight`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef.get_weight), [`up`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef.up), [`get_bias`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef.get_bias), [`get_weight`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef.get_weight)

## Functions
- `_gmm_vmem_estimate(tn: int, tk: int)` — [`L922`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L922) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `_is_tile_k_quant_block_compatible(tk: int)` — [`L911`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L911)
- `_matmul(is_first_k_step: bool, is_last_k_step: bool)` — [`L381`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L381) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `align_to(x, a)` — [`L77`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L77) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `apply_act_fn(acc: jax.Array, fuse_act: str | None)` — [`L43`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L43) — Applies a fused activation function to the accumulator.
- `calculate_tiling(dims: Dimensions, lhs_cfgs: InputConfigs, rhs_cfgs: InputConfigs, vmem_limit_bytes: int, fuse_act: str | None = None, has_partial_sum: bool = False)` — [`L872`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L872) — Calculate optimal tile sizes for GMM kernel. — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `fill_metadata(lhs_group_sizes_ref: jax.Array, group_offset_ref: jax.Array, metadata_ref: MetadataRef, *, cfgs: GmmConfigs)` — [`L601`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L601) — Fills the metadata for the given lhs group sizes and group offset. — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `fill_zero(i, zero_size, *, start, end)` — [`L721`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L721)
- `generate_block_specs(metadata_ref: MetadataRef, cfgs: GmmConfigs)` — [`L285`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L285) — Generates block specs for the given lhs, rhs, and out refs. — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `get_cost_estimate(cfgs: GmmConfigs)` — [`L1030`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L1030) — Returns the cost estimate for the GMM kernel. — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `get_metadata(cfgs: GmmConfigs)` — [`L1170`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L1170)
- `get_scope_name(cfgs: GmmConfigs)` — [`L1065`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L1065) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `gmm_v2(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, rhs_scale: jax.Array | None = None, rhs_bias: jax.Array | None = None, partial_sum: jax.Array | None = None, group_offset: jax.Array | None = None, *, tile_info: TileSizes | TileFn = calculate_tiling, vmem_limit_bytes: int | None = None, precision: jax.lax.Precision = jax.lax.Precision.DEFAULT, preferred_element_type: jnp.dtype | None = None, acc_dtype: jnp.dtype | None = None, maybe_quantize_lhs: bool = True, zero_initialize: bool = True, fuse_act: str | None = None)` — [`L1193`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L1193) — GMM kernel implemented with emit_pipeline. — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `inner_kernel(tiled_lhs_ref: jax.Array, tiled_rhs_ref: RhsRef, tiled_ps_ref: jax.Array | None, tiled_out_ref: jax.Array, partial_out_ref: jax.Array, acc_ref: jax.Array, metadata_ref: MetadataRef, *, cfgs: GmmConfigs)` — [`L343`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L343) — Inner kernel invoked by emit_pipeline to perform matmul. — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id)` — [`L631`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L631) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `kernel_main(lhs_group_sizes_ref: jax.Array, group_offset_ref: jax.Array, lhs_ref: jax.Array, rhs_ref: WeightsRef, partial_sum_ref: jax.Array, out_ref: jax.Array, partial_out_ref: jax.Array, acc_ref: jax.Array, metadata_ref: MetadataRef, zero_ref: jax.Array | None, semaphore_ref: jax.Array | None, *, cfgs: GmmConfigs)` — [`L765`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L765) — Entry point for GMM kernel. — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `left_fill_zero(i, zero_size)` — [`L738`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L738)
- `make_gmm_configs(lhs: jax.Array, rhs: jax.Array, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None, partial_sum: jax.Array | None, group_sizes: jax.Array, group_offset: jax.Array, *, tile_info: TileSizes | TileFn, vmem_limit_bytes: int | None, out_dtype: jnp.dtype | None, acc_dtype: jnp.dtype | None, maybe_quantize_lhs: bool, zero_initialize: bool, fuse_act: str | None = None)` — [`L1074`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L1074) — Fills the GMM config for the GMM kernel. — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `matmul()` — [`L572`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L572)
- `matmul_first()` — [`L568`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L568)
- `matmul_first_last()` — [`L564`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L564)
- `matmul_last()` — [`L576`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L576)
- `outer_group_loop(lhs_group_id, carry)` — [`L644`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L644) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `right_fill_zero(i, zero_size)` — [`L742`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L742)
- `swigluoai(gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7)` — [`L34`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L34) — Activation used in some models such as GPT-OSS.
- `validate_inputs(lhs: jax.Array, rhs: jax.Array, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None, partial_sum: jax.Array | None, group_sizes: jax.Array, group_offset: jax.Array, fuse_act: str | None = None)` — [`L980`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L980) — Validates the inputs for the GMM kernel. — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `zero_out_end(out_ref: jax.Array, semaphore_ref: jax.Array, zero_size: jax.Array, *, dims: Dimensions)` — [`L750`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L750)
- `zero_out_start(out_ref: jax.Array, zero_ref: jax.Array, semaphore_ref: jax.Array, metadata_ref: MetadataRef, num_gm: jax.Array, *, dims: Dimensions)` — [`L689`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L689) — Zero out output rows that are not used in the computation. — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)

## Module values
- `TileFn` — [`L223`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_gmm_kernel.py#L223) — documented in [maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/maxtext-kernels-megablox-pallas_mosaic_tpu_v2_gmm_kernel.md)

