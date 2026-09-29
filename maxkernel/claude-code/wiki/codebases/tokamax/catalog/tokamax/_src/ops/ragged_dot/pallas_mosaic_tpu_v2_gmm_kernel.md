---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_tpu_v2_gmm_kernel`/
symbols:
  gmm_v2: gmm_v2().
  generate_block_specs: generate_block_specs().
  inner_kernel._matmul: inner_kernel()._matmul().
  kernel_main: kernel_main().
  make_gmm_configs: make_gmm_configs().
  GmmConfigs.dims: GmmConfigs#dims.
  get_cost_estimate: get_cost_estimate().
  calculate_tiling: calculate_tiling().
  GmmConfigs.tiles: GmmConfigs#tiles.
  Dimensions.size_lhs_sublane: Dimensions#size_lhs_sublane.
  get_scope_name: get_scope_name().
  GmmConfigs.rhs_cfgs: GmmConfigs#rhs_cfgs.
  TileSizes.tile_k: TileSizes#tile_k.
  TileSizes.tile_n: TileSizes#tile_n.
  Dimensions.size_k: Dimensions#size_k.
  Dimensions.size_n: Dimensions#size_n.
  TileSizes: TileSizes#
  TileSizes.tile_m: TileSizes#tile_m.
  IndexMaps.rhs_scale_index_map: IndexMaps#rhs_scale_index_map().
  MetadataRef.gm_id_to_m_offset: MetadataRef#gm_id_to_m_offset.
  inner_kernel: inner_kernel().
  validate_inputs: validate_inputs().
  InputConfigs.quant_block_size: InputConfigs#quant_block_size.
  fill_metadata: fill_metadata().
  Dimensions.size_m: Dimensions#size_m.
  InputConfigs.dtype: InputConfigs#dtype.
  fill_metadata.inner_tm_loop: fill_metadata().inner_tm_loop().
  zero_out_start: zero_out_start().
  calculate_tiling._gmm_vmem_estimate: calculate_tiling()._gmm_vmem_estimate().
  GmmConfigs: GmmConfigs#
  align_to: align_to().
  InputConfigs: InputConfigs#
  GmmConfigs.num_quant_blocks_per_tile_k: GmmConfigs#num_quant_blocks_per_tile_k().
  IndexMaps.lhs_index_map: IndexMaps#lhs_index_map().
  IndexMaps.out_index_map: IndexMaps#out_index_map().
  fill_metadata.outer_group_loop: fill_metadata().outer_group_loop().
  GmmConfigs.out_size_n: GmmConfigs#out_size_n().
  MetadataRef: MetadataRef#
  GmmConfigs.out_dtype: GmmConfigs#out_dtype.
  MetadataRef.gm_id_to_group_id: MetadataRef#gm_id_to_group_id.
  Dimensions.size_group: Dimensions#size_group.
  GmmConfigs.lhs_cfgs: GmmConfigs#lhs_cfgs.
  Dimensions: Dimensions#
  InputConfigs.quant_dtype: InputConfigs#quant_dtype.
  WeightsRef: WeightsRef#
  TileFn: TileFn.
  InputConfigs.has_scale: InputConfigs#has_scale.
  RhsRef: RhsRef#
  RhsRef.get_scale: RhsRef#get_scale().
  InputConfigs.should_dequantize_after_matmul: InputConfigs#should_dequantize_after_matmul().
  GmmConfigs.fuse_act: GmmConfigs#fuse_act.
  IndexMaps.cfgs: IndexMaps#cfgs.
  FusedWeightsRef.gate: FusedWeightsRef#gate.
  FusedWeightsRef.up: FusedWeightsRef#up.
  FusedWeightsRef.get_weight: FusedWeightsRef#get_weight().
  FusedWeightsRef.get_scale: FusedWeightsRef#get_scale().
  FusedWeightsRef.get_bias: FusedWeightsRef#get_bias().
  InputConfigs.should_dequantize_before_matmul: InputConfigs#should_dequantize_before_matmul().
  IndexMaps.metadata_ref: IndexMaps#metadata_ref.
  RhsRef.get_weight: RhsRef#get_weight().
  RhsRef.get_bias: RhsRef#get_bias().
  IndexMaps.rhs_weight_index_map: IndexMaps#rhs_weight_index_map().
  IndexMaps.rhs_bias_index_map: IndexMaps#rhs_bias_index_map().
  zero_out_end: zero_out_end().
  InputConfigs.should_bitcast: InputConfigs#should_bitcast().
  WeightsRef.weight: WeightsRef#weight.
  WeightsRef.scale: WeightsRef#scale.
  WeightsRef.bias: WeightsRef#bias.
  InputConfigs.has_bias: InputConfigs#has_bias.
  GmmConfigs.zero_init: GmmConfigs#zero_init.
  IndexMaps.__init__: IndexMaps#__init__().
  apply_act_fn: apply_act_fn().
  WeightsRef.get_weight: WeightsRef#get_weight().
  WeightsRef.get_scale: WeightsRef#get_scale().
  WeightsRef.get_bias: WeightsRef#get_bias().
  FusedWeightsRef: FusedWeightsRef#
  get_metadata: get_metadata().
  GmmConfigs.acc_dtype: GmmConfigs#acc_dtype.
  inner_kernel.matmul_first_last: inner_kernel().matmul_first_last().
  inner_kernel.matmul_first: inner_kernel().matmul_first().
  inner_kernel.matmul: inner_kernel().matmul().
  inner_kernel.matmul_last: inner_kernel().matmul_last().
  zero_out_start.left_fill_zero: zero_out_start().left_fill_zero().
  zero_out_start.right_fill_zero: zero_out_start().right_fill_zero().
  calculate_tiling._is_tile_k_quant_block_compatible: calculate_tiling()._is_tile_k_quant_block_compatible().
  Dimensions.size_lhs_group: Dimensions#size_lhs_group.
  zero_out_start.fill_zero: zero_out_start().fill_zero().
  swigluoai: swigluoai().
  IndexMaps: IndexMaps#
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py)

## Classes
### `Dimensions`
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py:157`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L157)
- signature: `class Dimensions:`
- members:
  - `size_group` — [`L161`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L161) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `size_k` — [`L159`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L159)
  - `size_lhs_group` — [`L162`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L162)
  - `size_lhs_sublane` — [`L163`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L163) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `size_m` — [`L158`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L158) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `size_n` — [`L160`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L160) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md)
- used by: [`gmm_v2`](pallas_mosaic_tpu_v2_gmm_kernel.md#gmm_v2), [`tgmm_v2`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_v2), [`generate_block_specs`](pallas_mosaic_tpu_v2_gmm_kernel.md#generate_block_specs), [`_matmul`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel._matmul), [`kernel_main`](pallas_mosaic_tpu_v2_gmm_kernel.md#kernel_main), [`make_tgmm_configs`](pallas_mosaic_tpu_v2_tgmm_kernel.md#make_tgmm_configs), [`make_gmm_configs`](pallas_mosaic_tpu_v2_gmm_kernel.md#make_gmm_configs), [`generate_tgmm_block_specs`](pallas_mosaic_tpu_v2_tgmm_kernel.md#generate_tgmm_block_specs), [`dims`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.dims), [`get_cost_estimate`](pallas_mosaic_tpu_v2_gmm_kernel.md#get_cost_estimate), [`calculate_tiling`](pallas_mosaic_tpu_v2_gmm_kernel.md#calculate_tiling), [`calculate_tgmm_tiling`](pallas_mosaic_tpu_v2_tgmm_kernel.md#calculate_tgmm_tiling), [`get_scope_name`](pallas_mosaic_tpu_v2_gmm_kernel.md#get_scope_name), [`get_scope_name`](pallas_mosaic_tpu_v2_tgmm_kernel.md#get_scope_name), [`get_cost_estimate`](pallas_mosaic_tpu_v2_tgmm_kernel.md#get_cost_estimate), [`validate_inputs`](pallas_mosaic_tpu_v2_gmm_kernel.md#validate_inputs), [`fill_metadata`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata), [`_matmul`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_inner_kernel._matmul), [`inner_tm_loop`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata.inner_tm_loop), [`zero_out_start`](pallas_mosaic_tpu_v2_gmm_kernel.md#zero_out_start), [`lhs_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.lhs_index_map), [`lhs_index_map`](pallas_mosaic_tpu_v2_tgmm_kernel.md#TgmmIndexMaps.lhs_index_map), [`out_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.out_index_map), [`outer_group_loop`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata.outer_group_loop), [`rhs_index_map`](pallas_mosaic_tpu_v2_tgmm_kernel.md#TgmmIndexMaps.rhs_index_map), [`out_size_n`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.out_size_n), [`TileFn`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileFn), [`TileTgmmFn`](pallas_mosaic_tpu_v2_tgmm_kernel.md#TileTgmmFn), [`zero_out_end`](pallas_mosaic_tpu_v2_gmm_kernel.md#zero_out_end)

### `FusedWeightsRef`  ·  implements/extends RhsRef
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py:120`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L120)
- doc: Dataclass for gate and up weights used in fused activation.
- signature: `class FusedWeightsRef(RhsRef):`
- members:
  - `get_bias(self)` — [`L136`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L136)
  - `get_scale(self)` — [`L131`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L131)
  - `get_weight(self)` — [`L126`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L126)
  - `gate` — [`L123`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L123)
  - `up` — [`L124`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L124)
- uses (calls/refs, reference-scoped): [`WeightsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef), [`RhsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef), [`get_bias`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef.get_bias), [`get_scale`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef.get_scale), [`get_weight`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef.get_weight)
- used by: [`kernel_main`](pallas_mosaic_tpu_v2_gmm_kernel.md#kernel_main), [`RhsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef), [`get_scale`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef.get_scale), [`get_bias`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef.get_bias), [`get_weight`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef.get_weight)

### `GmmConfigs`
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py:193`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L193) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md)
- signature: `class GmmConfigs:`
- members:
  - `num_quant_blocks_per_tile_k(self)` — [`L204`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L204) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `out_size_n(self)` — [`L208`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L208) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `acc_dtype` — [`L199`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L199)
  - `dims` — [`L195`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L195) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `fuse_act` — [`L201`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L201) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `lhs_cfgs` — [`L196`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L196) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `out_dtype` — [`L198`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L198)
  - `rhs_cfgs` — [`L197`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L197) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `tiles` — [`L194`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L194) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `zero_init` — [`L200`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L200)
- uses (calls/refs, reference-scoped): [`tile_k`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes.tile_k), [`size_n`](pallas_mosaic_tpu_v2_gmm_kernel.md#Dimensions.size_n), [`TileSizes`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes), [`quant_block_size`](pallas_mosaic_tpu_v2_gmm_kernel.md#InputConfigs.quant_block_size), [`InputConfigs`](pallas_mosaic_tpu_v2_gmm_kernel.md#InputConfigs), [`Dimensions`](pallas_mosaic_tpu_v2_gmm_kernel.md#Dimensions)
- used by: [`gmm_v2`](pallas_mosaic_tpu_v2_gmm_kernel.md#gmm_v2), [`tgmm_v2`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_v2), [`generate_block_specs`](pallas_mosaic_tpu_v2_gmm_kernel.md#generate_block_specs), [`_matmul`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel._matmul), [`kernel_main`](pallas_mosaic_tpu_v2_gmm_kernel.md#kernel_main), [`make_tgmm_configs`](pallas_mosaic_tpu_v2_tgmm_kernel.md#make_tgmm_configs), [`make_gmm_configs`](pallas_mosaic_tpu_v2_gmm_kernel.md#make_gmm_configs), [`generate_tgmm_block_specs`](pallas_mosaic_tpu_v2_tgmm_kernel.md#generate_tgmm_block_specs), [`get_cost_estimate`](pallas_mosaic_tpu_v2_gmm_kernel.md#get_cost_estimate), [`get_scope_name`](pallas_mosaic_tpu_v2_gmm_kernel.md#get_scope_name), [`get_scope_name`](pallas_mosaic_tpu_v2_tgmm_kernel.md#get_scope_name), [`get_cost_estimate`](pallas_mosaic_tpu_v2_tgmm_kernel.md#get_cost_estimate), [`tgmm_inner_kernel`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_inner_kernel), [`rhs_scale_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.rhs_scale_index_map), [`inner_kernel`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel), [`fill_metadata`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata), [`_matmul`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_inner_kernel._matmul), [`inner_tm_loop`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata.inner_tm_loop), [`lhs_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.lhs_index_map), [`lhs_index_map`](pallas_mosaic_tpu_v2_tgmm_kernel.md#TgmmIndexMaps.lhs_index_map), [`out_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.out_index_map), [`outer_group_loop`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata.outer_group_loop), [`rhs_index_map`](pallas_mosaic_tpu_v2_tgmm_kernel.md#TgmmIndexMaps.rhs_index_map), [`__init__`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.__init__), [`__init__`](pallas_mosaic_tpu_v2_tgmm_kernel.md#TgmmIndexMaps.__init__), [`get_metadata`](pallas_mosaic_tpu_v2_gmm_kernel.md#get_metadata)

### `IndexMaps`
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py:220`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L220)
- doc: Index maps for GMM kernel.
- signature: `class IndexMaps:`
- members:
  - `lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array)` — [`L227`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L227)
  - `out_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array)` — [`L258`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L258)
  - `rhs_bias_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array)` — [`L243`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L243)
  - `rhs_scale_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array)` — [`L247`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L247)
  - `rhs_weight_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array)` — [`L237`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L237)
  - `cfgs` — [`L225`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L225)
  - `metadata_ref` — [`L224`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L224)
- protocol/private: `__init__`[`L223`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L223)
- uses (calls/refs, reference-scoped): [`dims`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.dims), [`tiles`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.tiles), [`size_lhs_sublane`](pallas_mosaic_tpu_v2_gmm_kernel.md#Dimensions.size_lhs_sublane), [`rhs_cfgs`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.rhs_cfgs), [`tile_k`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileSizes.tile_k), [`gm_id_to_m_offset`](pallas_mosaic_tpu_v2_gmm_kernel.md#MetadataRef.gm_id_to_m_offset), [`quant_block_size`](pallas_mosaic_tpu_v2_gmm_kernel.md#InputConfigs.quant_block_size), [`GmmConfigs`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs), [`num_quant_blocks_per_tile_k`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.num_quant_blocks_per_tile_k), [`MetadataRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#MetadataRef), [`gm_id_to_group_id`](pallas_mosaic_tpu_v2_gmm_kernel.md#MetadataRef.gm_id_to_group_id)
- used by: [`generate_block_specs`](pallas_mosaic_tpu_v2_gmm_kernel.md#generate_block_specs)

### `InputConfigs`
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py:167`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L167)
- signature: `class InputConfigs:`
- members:
  - `should_bitcast(self)` — [`L175`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L175)
  - `should_dequantize_after_matmul(self)` — [`L188`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L188) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md)
  - `should_dequantize_before_matmul(self)` — [`L180`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L180)
  - `dtype` — [`L170`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L170)
  - `has_bias` — [`L171`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L171)
  - `has_scale` — [`L172`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L172)
  - `quant_block_size` — [`L169`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L169)
  - `quant_dtype` — [`L168`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L168)
- used by: [`generate_block_specs`](pallas_mosaic_tpu_v2_gmm_kernel.md#generate_block_specs), [`_matmul`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel._matmul), [`kernel_main`](pallas_mosaic_tpu_v2_gmm_kernel.md#kernel_main), [`make_tgmm_configs`](pallas_mosaic_tpu_v2_tgmm_kernel.md#make_tgmm_configs), [`make_gmm_configs`](pallas_mosaic_tpu_v2_gmm_kernel.md#make_gmm_configs), [`generate_tgmm_block_specs`](pallas_mosaic_tpu_v2_tgmm_kernel.md#generate_tgmm_block_specs), [`get_cost_estimate`](pallas_mosaic_tpu_v2_gmm_kernel.md#get_cost_estimate), [`calculate_tiling`](pallas_mosaic_tpu_v2_gmm_kernel.md#calculate_tiling), [`calculate_tgmm_tiling`](pallas_mosaic_tpu_v2_tgmm_kernel.md#calculate_tgmm_tiling), [`rhs_cfgs`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.rhs_cfgs), [`get_cost_estimate`](pallas_mosaic_tpu_v2_tgmm_kernel.md#get_cost_estimate), [`rhs_scale_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.rhs_scale_index_map), [`_matmul`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_inner_kernel._matmul), [`_gmm_vmem_estimate`](pallas_mosaic_tpu_v2_gmm_kernel.md#calculate_tiling._gmm_vmem_estimate), [`num_quant_blocks_per_tile_k`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.num_quant_blocks_per_tile_k), [`lhs_cfgs`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.lhs_cfgs), [`TileFn`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileFn), [`TileTgmmFn`](pallas_mosaic_tpu_v2_tgmm_kernel.md#TileTgmmFn), [`within_vmem_limit`](pallas_mosaic_tpu_v2_tgmm_kernel.md#calculate_tgmm_tiling.within_vmem_limit), [`_is_tile_k_quant_block_compatible`](pallas_mosaic_tpu_v2_gmm_kernel.md#calculate_tiling._is_tile_k_quant_block_compatible)

### `MetadataRef`
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py:144`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L144)
- signature: `class MetadataRef:`
- members:
  - `gm_id_to_group_id` — [`L145`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L145)
  - `gm_id_to_m_offset` — [`L146`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L146)
- used by: [`gmm_v2`](pallas_mosaic_tpu_v2_gmm_kernel.md#gmm_v2), [`tgmm_v2`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_v2), [`generate_block_specs`](pallas_mosaic_tpu_v2_gmm_kernel.md#generate_block_specs), [`_matmul`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel._matmul), [`kernel_main`](pallas_mosaic_tpu_v2_gmm_kernel.md#kernel_main), [`generate_tgmm_block_specs`](pallas_mosaic_tpu_v2_tgmm_kernel.md#generate_tgmm_block_specs), [`tgmm_inner_kernel`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_inner_kernel), [`tgmm_kernel_main`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_kernel_main), [`rhs_scale_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.rhs_scale_index_map), [`inner_kernel`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel), [`fill_metadata`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata), [`_matmul`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_inner_kernel._matmul), [`inner_tm_loop`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata.inner_tm_loop), [`zero_out_start`](pallas_mosaic_tpu_v2_gmm_kernel.md#zero_out_start), [`lhs_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.lhs_index_map), [`lhs_index_map`](pallas_mosaic_tpu_v2_tgmm_kernel.md#TgmmIndexMaps.lhs_index_map), [`out_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.out_index_map), [`rhs_index_map`](pallas_mosaic_tpu_v2_tgmm_kernel.md#TgmmIndexMaps.rhs_index_map), [`out_index_map`](pallas_mosaic_tpu_v2_tgmm_kernel.md#TgmmIndexMaps.out_index_map), [`rhs_bias_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.rhs_bias_index_map), [`rhs_weight_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.rhs_weight_index_map), [`__init__`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.__init__), [`__init__`](pallas_mosaic_tpu_v2_tgmm_kernel.md#TgmmIndexMaps.__init__)

### `RhsRef`  ·  implements/extends ABC
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py:81`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L81)
- doc: Abstract class that defines interfaces for rhs values.
- signature: `class RhsRef(ABC):`
- members:
  - `get_bias(self)` — [`L93`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L93)
  - `get_scale(self)` — [`L89`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L89)
  - `get_weight(self)` — [`L85`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L85)
- uses (calls/refs, reference-scoped): [`WeightsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef), [`get_bias`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef.get_bias), [`get_scale`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef.get_scale), [`get_weight`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef.get_weight), [`FusedWeightsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef), [`get_bias`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef.get_bias), [`get_scale`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef.get_scale), [`get_weight`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef.get_weight)
- used by: [`_matmul`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel._matmul), [`inner_kernel`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel), [`WeightsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#WeightsRef), [`FusedWeightsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef)

### `TileSizes`
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py:150`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L150)
- signature: `class TileSizes:`
- members:
  - `tile_k` — [`L152`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L152)
  - `tile_m` — [`L151`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L151)
  - `tile_n` — [`L153`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L153)
- used by: [`gmm_v2`](pallas_mosaic_tpu_v2_gmm_kernel.md#gmm_v2), [`tgmm_v2`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_v2), [`generate_block_specs`](pallas_mosaic_tpu_v2_gmm_kernel.md#generate_block_specs), [`_fwd`](pallas_mosaic_tpu_v2.md#PallasMosaicTpuV2RaggedDot._fwd), [`_matmul`](pallas_mosaic_tpu_v2_gmm_kernel.md#inner_kernel._matmul), [`kernel_main`](pallas_mosaic_tpu_v2_gmm_kernel.md#kernel_main), [`make_tgmm_configs`](pallas_mosaic_tpu_v2_tgmm_kernel.md#make_tgmm_configs), [`make_gmm_configs`](pallas_mosaic_tpu_v2_gmm_kernel.md#make_gmm_configs), [`generate_tgmm_block_specs`](pallas_mosaic_tpu_v2_tgmm_kernel.md#generate_tgmm_block_specs), [`calculate_tiling`](pallas_mosaic_tpu_v2_gmm_kernel.md#calculate_tiling), [`tiles`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.tiles), [`test_tgmm_drhs_with_tile_info_pipes`](pallas_mosaic_tpu_v2_test.md#PallasMosaicTpuV2OpParameterPipingTest.test_tgmm_drhs_with_tile_info_pipes), [`calculate_tgmm_tiling`](pallas_mosaic_tpu_v2_tgmm_kernel.md#calculate_tgmm_tiling), [`get_scope_name`](pallas_mosaic_tpu_v2_gmm_kernel.md#get_scope_name), [`get_scope_name`](pallas_mosaic_tpu_v2_tgmm_kernel.md#get_scope_name), [`test_tgmm_with_rhs_scale_n_padding`](gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.md#GmmTest.test_tgmm_with_rhs_scale_n_padding), [`test_tgmm_with_tile_info`](gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.md#GmmTest.test_tgmm_with_tile_info), [`rhs_scale_index_map`](pallas_mosaic_tpu_v2_gmm_kernel.md#IndexMaps.rhs_scale_index_map), [`test_gmm_activation_weight_quantized_block_larger_than_tile_k`](gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.md#GmmTest.test_gmm_activation_weight_quantized_block_larger_than_tile_k), [`test_gmm_weight_quantized_block_larger_than_tile_k`](gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.md#GmmTest.test_gmm_weight_quantized_block_larger_than_tile_k), [`test_tgmm_explicitly_exercises_all_branches`](gmm_v2_kernel_tests/pallas_mosaic_tpu_v2_kernel_test.md#GmmTest.test_tgmm_explicitly_exercises_all_branches), [`inner_tm_loop`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata.inner_tm_loop), [`num_quant_blocks_per_tile_k`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.num_quant_blocks_per_tile_k), [`outer_group_loop`](pallas_mosaic_tpu_v2_gmm_kernel.md#fill_metadata.outer_group_loop), [`TileFn`](pallas_mosaic_tpu_v2_gmm_kernel.md#TileFn), [`TileTgmmFn`](pallas_mosaic_tpu_v2_tgmm_kernel.md#TileTgmmFn)

### `WeightsRef`  ·  implements/extends RhsRef
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py:99`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L99)
- doc: Dataclass for a single weights.
- signature: `class WeightsRef(RhsRef):`
- members:
  - `get_bias(self)` — [`L113`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L113)
  - `get_scale(self)` — [`L109`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L109)
  - `get_weight(self)` — [`L106`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L106)
  - `bias` — [`L104`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L104)
  - `scale` — [`L103`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L103)
  - `weight` — [`L102`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L102)
- uses (calls/refs, reference-scoped): [`RhsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef)
- used by: [`gmm_v2`](pallas_mosaic_tpu_v2_gmm_kernel.md#gmm_v2), [`generate_block_specs`](pallas_mosaic_tpu_v2_gmm_kernel.md#generate_block_specs), [`kernel_main`](pallas_mosaic_tpu_v2_gmm_kernel.md#kernel_main), [`RhsRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef), [`get_scale`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef.get_scale), [`gate`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef.gate), [`get_bias`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef.get_bias), [`get_scale`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef.get_scale), [`get_weight`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef.get_weight), [`up`](pallas_mosaic_tpu_v2_gmm_kernel.md#FusedWeightsRef.up), [`get_bias`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef.get_bias), [`get_weight`](pallas_mosaic_tpu_v2_gmm_kernel.md#RhsRef.get_weight)

## Functions
- `_gmm_vmem_estimate(tn: int, tk: int)` — [`L907`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L907)
- `_is_tile_k_quant_block_compatible(tk: int)` — [`L893`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L893)
- `_matmul(is_first_k_step: bool, is_last_k_step: bool)` — [`L361`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L361)
- `align_to(x, a)` — [`L74`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L74)
- `apply_act_fn(acc: jax.Array, fuse_act: str | None)` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L40) — Applies a fused activation function to the accumulator.
- `calculate_tiling(dims: Dimensions, lhs_cfgs: InputConfigs, rhs_cfgs: InputConfigs, vmem_limit_bytes: int, fuse_act: str | None = None)` — [`L855`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L855) — Calculate optimal tile sizes for GMM kernel.
- `fill_metadata(lhs_group_sizes_ref: jax.Array, group_offset_ref: jax.Array, metadata_ref: MetadataRef, *, cfgs: GmmConfigs)` — [`L586`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L586) — Fills the metadata for the given lhs group sizes and group offset.
- `fill_zero(i, zero_size, *, start, end)` — [`L708`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L708)
- `generate_block_specs(metadata_ref: MetadataRef, cfgs: GmmConfigs)` — [`L272`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L272) — Generates block specs for the given lhs, rhs, and out refs. — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `get_cost_estimate(cfgs: GmmConfigs)` — [`L1018`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L1018) — Returns the cost estimate for the GMM kernel.
- `get_metadata(cfgs: GmmConfigs)` — [`L1158`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L1158)
- `get_scope_name(cfgs: GmmConfigs)` — [`L1053`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L1053)
- `gmm_v2(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, rhs_scale: jax.Array | None = None, rhs_bias: jax.Array | None = None, group_offset: jax.Array | None = None, *, tile_info: TileSizes | TileFn = calculate_tiling, vmem_limit_bytes: int | None = None, precision: jax.lax.Precision = jax.lax.Precision.DEFAULT, preferred_element_type: jnp.dtype | None = None, acc_dtype: jnp.dtype | None = None, maybe_quantize_lhs: bool = True, zero_initialize: bool = True, fuse_act: str | None = None)` — [`L1181`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L1181) — GMM kernel implemented with emit_pipeline. — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `inner_kernel(tiled_lhs_ref: jax.Array, tiled_rhs_ref: RhsRef, tiled_out_ref: jax.Array, partial_out_ref: jax.Array, acc_ref: jax.Array, metadata_ref: MetadataRef, *, cfgs: GmmConfigs)` — [`L326`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L326) — Inner kernel invoked by emit_pipeline to perform matmul.
- `inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id)` — [`L616`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L616)
- `kernel_main(lhs_group_sizes_ref: jax.Array, group_offset_ref: jax.Array, lhs_ref: jax.Array, rhs_ref: WeightsRef, out_ref: jax.Array, partial_out_ref: jax.Array, acc_ref: jax.Array, metadata_ref: MetadataRef, zero_ref: jax.Array | None, semaphore_ref: jax.Array | None, *, cfgs: GmmConfigs)` — [`L752`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L752) — Entry point for GMM kernel. — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_gmm_kernel.md)
- `left_fill_zero(i, zero_size)` — [`L725`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L725)
- `make_gmm_configs(lhs: jax.Array, rhs: jax.Array, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None, group_sizes: jax.Array, group_offset: jax.Array, *, tile_info: TileSizes | TileFn, vmem_limit_bytes: int | None, out_dtype: jnp.dtype | None, acc_dtype: jnp.dtype | None, maybe_quantize_lhs: bool, zero_initialize: bool, fuse_act: str | None = None)` — [`L1062`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L1062) — Fills the GMM config for the GMM kernel.
- `matmul()` — [`L557`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L557)
- `matmul_first()` — [`L553`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L553)
- `matmul_first_last()` — [`L549`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L549)
- `matmul_last()` — [`L561`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L561)
- `outer_group_loop(lhs_group_id, carry)` — [`L631`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L631)
- `right_fill_zero(i, zero_size)` — [`L729`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L729)
- `swigluoai(gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7)` — [`L29`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L29) — Activation used in some models such as GPT-OSS.
- `validate_inputs(lhs: jax.Array, rhs: jax.Array, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None, group_sizes: jax.Array, group_offset: jax.Array, fuse_act: str | None = None)` — [`L971`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L971) — Validates the inputs for the GMM kernel.
- `zero_out_end(out_ref: jax.Array, semaphore_ref: jax.Array, zero_size: jax.Array, *, dims: Dimensions)` — [`L737`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L737)
- `zero_out_start(out_ref: jax.Array, zero_ref: jax.Array, semaphore_ref: jax.Array, metadata_ref: MetadataRef, num_gm: jax.Array, *, dims: Dimensions)` — [`L676`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L676) — Zero out output rows that are not used in the computation.

## Module values
- `TileFn` — [`L215`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_gmm_kernel.py#L215)

