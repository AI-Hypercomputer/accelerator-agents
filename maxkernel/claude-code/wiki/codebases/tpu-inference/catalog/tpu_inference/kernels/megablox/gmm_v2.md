---
title: 'Module: tpu_inference/kernels/megablox/gmm_v2.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/megablox/gmm_v2.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.megablox.gmm_v2`/
symbols:
  gmm_v2: gmm_v2().
  generate_block_specs: generate_block_specs().
  inner_kernel._matmul: inner_kernel()._matmul().
  kernel_main: kernel_main().
  make_gmm_configs: make_gmm_configs().
  get_cost_estimate: get_cost_estimate().
  calculate_tiling: calculate_tiling().
  GmmConfigs.tiles: GmmConfigs#tiles.
  GmmConfigs.dims: GmmConfigs#dims.
  get_scope_name: get_scope_name().
  Dimensions.size_lhs_sublane: Dimensions#size_lhs_sublane.
  GmmConfigs.rhs_cfgs: GmmConfigs#rhs_cfgs.
  IndexMaps.rhs_scale_index_map: IndexMaps#rhs_scale_index_map().
  inner_kernel: inner_kernel().
  validate_inputs: validate_inputs().
  fill_metadata: fill_metadata().
  fill_metadata.inner_tm_loop: fill_metadata().inner_tm_loop().
  zero_out_start: zero_out_start().
  calculate_tiling._gmm_vmem_estimate: calculate_tiling()._gmm_vmem_estimate().
  TileSizes.tile_k: TileSizes#tile_k.
  IndexMaps.lhs_index_map: IndexMaps#lhs_index_map().
  IndexMaps.out_index_map: IndexMaps#out_index_map().
  fill_metadata.outer_group_loop: fill_metadata().outer_group_loop().
  InputConfigs.quant_block_size: InputConfigs#quant_block_size.
  GmmConfigs.num_quant_blocks_per_tile_k: GmmConfigs#num_quant_blocks_per_tile_k().
  GmmConfigs.out_size_n: GmmConfigs#out_size_n().
  MetadataRef.gm_id_to_m_offset: MetadataRef#gm_id_to_m_offset.
  Dimensions.size_k: Dimensions#size_k.
  InputConfigs.dtype: InputConfigs#dtype.
  GmmConfigs.lhs_cfgs: GmmConfigs#lhs_cfgs.
  TileSizes.tile_n: TileSizes#tile_n.
  Dimensions.size_n: Dimensions#size_n.
  WeightsRef: WeightsRef#
  TileFn: TileFn.
  TileSizes.tile_m: TileSizes#tile_m.
  Dimensions.size_m: Dimensions#size_m.
  GmmConfigs: GmmConfigs#
  RhsRef: RhsRef#
  RhsRef.get_scale: RhsRef#get_scale().
  InputConfigs: InputConfigs#
  InputConfigs.quant_dtype: InputConfigs#quant_dtype.
  InputConfigs.should_dequantize_after_matmul: InputConfigs#should_dequantize_after_matmul().
  IndexMaps.cfgs: IndexMaps#cfgs.
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
  IndexMaps.metadata_ref: IndexMaps#metadata_ref.
  apply_act_fn: apply_act_fn().
  align_to: align_to().
  RhsRef.get_weight: RhsRef#get_weight().
  RhsRef.get_bias: RhsRef#get_bias().
  InputConfigs.has_scale: InputConfigs#has_scale.
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
  InputConfigs.has_bias: InputConfigs#has_bias.
  IndexMaps.__init__: IndexMaps#__init__().
  WeightsRef.get_weight: WeightsRef#get_weight().
  WeightsRef.get_scale: WeightsRef#get_scale().
  WeightsRef.get_bias: WeightsRef#get_bias().
  FusedWeightsRef: FusedWeightsRef#
  GmmConfigs.zero_init: GmmConfigs#zero_init.
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
  silu_and_mul_with_clamp: silu_and_mul_with_clamp().
  Dimensions.size_lhs_group: Dimensions#size_lhs_group.
  IndexMaps: IndexMaps#
---
# Module: [`tpu_inference/kernels/megablox/gmm_v2.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py)

## Classes
### `Dimensions`
- def: [`tpu_inference/kernels/megablox/gmm_v2.py:174`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L174)
- signature: `class Dimensions:`
- members:
  - `size_group` — [`L178`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L178)
  - `size_k` — [`L176`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L176)
  - `size_lhs_group` — [`L179`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L179)
  - `size_lhs_sublane` — [`L180`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L180)
  - `size_m` — [`L175`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L175)
  - `size_n` — [`L177`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L177)
- used by: [`gmm_v2`](gmm_v2.md#gmm_v2), [`generate_block_specs`](gmm_v2.md#generate_block_specs), [`_matmul`](gmm_v2.md#inner_kernel._matmul), [`kernel_main`](gmm_v2.md#kernel_main), [`make_gmm_configs`](gmm_v2.md#make_gmm_configs), [`get_cost_estimate`](gmm_v2.md#get_cost_estimate), [`calculate_tiling`](gmm_v2.md#calculate_tiling), [`dims`](gmm_v2.md#GmmConfigs.dims), [`get_scope_name`](gmm_v2.md#get_scope_name), [`validate_inputs`](gmm_v2.md#validate_inputs), [`fill_metadata`](gmm_v2.md#fill_metadata), [`inner_tm_loop`](gmm_v2.md#fill_metadata.inner_tm_loop), [`zero_out_start`](gmm_v2.md#zero_out_start), [`lhs_index_map`](gmm_v2.md#IndexMaps.lhs_index_map), [`out_index_map`](gmm_v2.md#IndexMaps.out_index_map), [`outer_group_loop`](gmm_v2.md#fill_metadata.outer_group_loop), [`out_size_n`](gmm_v2.md#GmmConfigs.out_size_n), [`TileFn`](gmm_v2.md#TileFn), [`zero_out_end`](gmm_v2.md#zero_out_end)

### `FusedWeightsRef`  ·  implements/extends RhsRef
- def: [`tpu_inference/kernels/megablox/gmm_v2.py:137`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L137)
- doc: Dataclass for gate and up weights used in fused activation.
- signature: `class FusedWeightsRef(RhsRef):`
- members:
  - `get_bias(self)` — [`L153`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L153)
  - `get_scale(self)` — [`L148`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L148)
  - `get_weight(self)` — [`L143`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L143)
  - `gate` — [`L140`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L140)
  - `up` — [`L141`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L141)
- uses (calls/refs, reference-scoped): [`WeightsRef`](gmm_v2.md#WeightsRef), [`RhsRef`](gmm_v2.md#RhsRef), [`get_bias`](gmm_v2.md#WeightsRef.get_bias), [`get_scale`](gmm_v2.md#WeightsRef.get_scale), [`get_weight`](gmm_v2.md#WeightsRef.get_weight)
- used by: [`kernel_main`](gmm_v2.md#kernel_main), [`RhsRef`](gmm_v2.md#RhsRef), [`get_scale`](gmm_v2.md#RhsRef.get_scale), [`get_bias`](gmm_v2.md#RhsRef.get_bias), [`get_weight`](gmm_v2.md#RhsRef.get_weight)

### `GmmConfigs`
- def: [`tpu_inference/kernels/megablox/gmm_v2.py:211`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L211)
- signature: `class GmmConfigs:`
- members:
  - `num_quant_blocks_per_tile_k(self)` — [`L222`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L222)
  - `out_size_n(self)` — [`L226`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L226)
  - `acc_dtype` — [`L217`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L217)
  - `dims` — [`L213`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L213)
  - `fuse_act` — [`L219`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L219)
  - `lhs_cfgs` — [`L214`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L214)
  - `out_dtype` — [`L216`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L216)
  - `rhs_cfgs` — [`L215`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L215)
  - `tiles` — [`L212`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L212)
  - `zero_init` — [`L218`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L218)
- uses (calls/refs, reference-scoped): [`tile_k`](gmm_v2.md#TileSizes.tile_k), [`quant_block_size`](gmm_v2.md#InputConfigs.quant_block_size), [`size_n`](gmm_v2.md#Dimensions.size_n), [`InputConfigs`](gmm_v2.md#InputConfigs), [`Dimensions`](gmm_v2.md#Dimensions), [`TileSizes`](gmm_v2.md#TileSizes)
- used by: [`gmm_v2`](gmm_v2.md#gmm_v2), [`generate_block_specs`](gmm_v2.md#generate_block_specs), [`_matmul`](gmm_v2.md#inner_kernel._matmul), [`kernel_main`](gmm_v2.md#kernel_main), [`make_gmm_configs`](gmm_v2.md#make_gmm_configs), [`get_cost_estimate`](gmm_v2.md#get_cost_estimate), [`get_scope_name`](gmm_v2.md#get_scope_name), [`rhs_scale_index_map`](gmm_v2.md#IndexMaps.rhs_scale_index_map), [`inner_kernel`](gmm_v2.md#inner_kernel), [`fill_metadata`](gmm_v2.md#fill_metadata), [`inner_tm_loop`](gmm_v2.md#fill_metadata.inner_tm_loop), [`lhs_index_map`](gmm_v2.md#IndexMaps.lhs_index_map), [`out_index_map`](gmm_v2.md#IndexMaps.out_index_map), [`outer_group_loop`](gmm_v2.md#fill_metadata.outer_group_loop), [`__init__`](gmm_v2.md#IndexMaps.__init__), [`get_metadata`](gmm_v2.md#get_metadata)

### `IndexMaps`
- def: [`tpu_inference/kernels/megablox/gmm_v2.py:237`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L237)
- doc: Index maps for GMM kernel.
- signature: `class IndexMaps:`
- members:
  - `lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array)` — [`L244`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L244)
  - `out_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array)` — [`L274`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L274)
  - `rhs_bias_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array)` — [`L259`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L259)
  - `rhs_scale_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array)` — [`L264`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L264)
  - `rhs_weight_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array)` — [`L254`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L254)
  - `cfgs` — [`L242`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L242)
  - `metadata_ref` — [`L241`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L241)
- protocol/private: `__init__`[`L240`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L240)
- uses (calls/refs, reference-scoped): [`dims`](gmm_v2.md#GmmConfigs.dims), [`tiles`](gmm_v2.md#GmmConfigs.tiles), [`size_lhs_sublane`](gmm_v2.md#Dimensions.size_lhs_sublane), [`rhs_cfgs`](gmm_v2.md#GmmConfigs.rhs_cfgs), [`tile_k`](gmm_v2.md#TileSizes.tile_k), [`num_quant_blocks_per_tile_k`](gmm_v2.md#GmmConfigs.num_quant_blocks_per_tile_k), [`quant_block_size`](gmm_v2.md#InputConfigs.quant_block_size), [`gm_id_to_m_offset`](gmm_v2.md#MetadataRef.gm_id_to_m_offset), [`GmmConfigs`](gmm_v2.md#GmmConfigs), [`MetadataRef`](gmm_v2.md#MetadataRef), [`gm_id_to_group_id`](gmm_v2.md#MetadataRef.gm_id_to_group_id)
- used by: [`generate_block_specs`](gmm_v2.md#generate_block_specs)

### `InputConfigs`
- def: [`tpu_inference/kernels/megablox/gmm_v2.py:184`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L184)
- signature: `class InputConfigs:`
- members:
  - `should_bitcast(self)` — [`L192`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L192)
  - `should_dequantize_after_matmul(self)` — [`L206`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L206)
  - `should_dequantize_before_matmul(self)` — [`L197`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L197) — Dequantize rhs before matmul if block size limits MXU utilization.
  - `dtype` — [`L187`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L187)
  - `has_bias` — [`L188`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L188)
  - `has_scale` — [`L189`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L189)
  - `quant_block_size` — [`L186`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L186)
  - `quant_dtype` — [`L185`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L185)
- used by: [`generate_block_specs`](gmm_v2.md#generate_block_specs), [`_matmul`](gmm_v2.md#inner_kernel._matmul), [`kernel_main`](gmm_v2.md#kernel_main), [`make_gmm_configs`](gmm_v2.md#make_gmm_configs), [`get_cost_estimate`](gmm_v2.md#get_cost_estimate), [`calculate_tiling`](gmm_v2.md#calculate_tiling), [`rhs_cfgs`](gmm_v2.md#GmmConfigs.rhs_cfgs), [`rhs_scale_index_map`](gmm_v2.md#IndexMaps.rhs_scale_index_map), [`_gmm_vmem_estimate`](gmm_v2.md#calculate_tiling._gmm_vmem_estimate), [`num_quant_blocks_per_tile_k`](gmm_v2.md#GmmConfigs.num_quant_blocks_per_tile_k), [`lhs_cfgs`](gmm_v2.md#GmmConfigs.lhs_cfgs), [`TileFn`](gmm_v2.md#TileFn), [`_is_tile_k_quant_block_compatible`](gmm_v2.md#calculate_tiling._is_tile_k_quant_block_compatible)

### `MetadataRef`
- def: [`tpu_inference/kernels/megablox/gmm_v2.py:161`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L161)
- signature: `class MetadataRef:`
- members:
  - `gm_id_to_group_id` — [`L162`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L162)
  - `gm_id_to_m_offset` — [`L163`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L163)
- used by: [`gmm_v2`](gmm_v2.md#gmm_v2), [`generate_block_specs`](gmm_v2.md#generate_block_specs), [`_matmul`](gmm_v2.md#inner_kernel._matmul), [`kernel_main`](gmm_v2.md#kernel_main), [`rhs_scale_index_map`](gmm_v2.md#IndexMaps.rhs_scale_index_map), [`inner_kernel`](gmm_v2.md#inner_kernel), [`fill_metadata`](gmm_v2.md#fill_metadata), [`inner_tm_loop`](gmm_v2.md#fill_metadata.inner_tm_loop), [`zero_out_start`](gmm_v2.md#zero_out_start), [`lhs_index_map`](gmm_v2.md#IndexMaps.lhs_index_map), [`out_index_map`](gmm_v2.md#IndexMaps.out_index_map), [`rhs_bias_index_map`](gmm_v2.md#IndexMaps.rhs_bias_index_map), [`rhs_weight_index_map`](gmm_v2.md#IndexMaps.rhs_weight_index_map), [`__init__`](gmm_v2.md#IndexMaps.__init__)

### `RhsRef`  ·  implements/extends ABC
- def: [`tpu_inference/kernels/megablox/gmm_v2.py:98`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L98)
- doc: Abstract class that defines interfaces for rhs values.
- signature: `class RhsRef(ABC):`
- members:
  - `get_bias(self)` — [`L110`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L110)
  - `get_scale(self)` — [`L106`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L106)
  - `get_weight(self)` — [`L102`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L102)
- uses (calls/refs, reference-scoped): [`WeightsRef`](gmm_v2.md#WeightsRef), [`get_bias`](gmm_v2.md#FusedWeightsRef.get_bias), [`get_scale`](gmm_v2.md#FusedWeightsRef.get_scale), [`get_weight`](gmm_v2.md#FusedWeightsRef.get_weight), [`FusedWeightsRef`](gmm_v2.md#FusedWeightsRef), [`get_bias`](gmm_v2.md#WeightsRef.get_bias), [`get_scale`](gmm_v2.md#WeightsRef.get_scale), [`get_weight`](gmm_v2.md#WeightsRef.get_weight)
- used by: [`_matmul`](gmm_v2.md#inner_kernel._matmul), [`inner_kernel`](gmm_v2.md#inner_kernel), [`WeightsRef`](gmm_v2.md#WeightsRef), [`FusedWeightsRef`](gmm_v2.md#FusedWeightsRef)

### `TileSizes`
- def: [`tpu_inference/kernels/megablox/gmm_v2.py:167`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L167)
- signature: `class TileSizes:`
- members:
  - `tile_k` — [`L169`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L169)
  - `tile_m` — [`L168`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L168)
  - `tile_n` — [`L170`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L170)
- used by: [`gmm_v2`](gmm_v2.md#gmm_v2), [`generate_block_specs`](gmm_v2.md#generate_block_specs), [`_matmul`](gmm_v2.md#inner_kernel._matmul), [`kernel_main`](gmm_v2.md#kernel_main), [`make_gmm_configs`](gmm_v2.md#make_gmm_configs), [`calculate_tiling`](gmm_v2.md#calculate_tiling), [`get_scope_name`](gmm_v2.md#get_scope_name), [`tiles`](gmm_v2.md#GmmConfigs.tiles), [`rhs_scale_index_map`](gmm_v2.md#IndexMaps.rhs_scale_index_map), [`inner_tm_loop`](gmm_v2.md#fill_metadata.inner_tm_loop), [`outer_group_loop`](gmm_v2.md#fill_metadata.outer_group_loop), [`num_quant_blocks_per_tile_k`](gmm_v2.md#GmmConfigs.num_quant_blocks_per_tile_k), [`TileFn`](gmm_v2.md#TileFn)

### `WeightsRef`  ·  implements/extends RhsRef
- def: [`tpu_inference/kernels/megablox/gmm_v2.py:116`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L116)
- doc: Dataclass for a single weights.
- signature: `class WeightsRef(RhsRef):`
- members:
  - `get_bias(self)` — [`L130`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L130)
  - `get_scale(self)` — [`L126`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L126)
  - `get_weight(self)` — [`L123`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L123)
  - `bias` — [`L121`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L121)
  - `scale` — [`L120`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L120)
  - `weight` — [`L119`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L119)
- uses (calls/refs, reference-scoped): [`RhsRef`](gmm_v2.md#RhsRef)
- used by: [`gmm_v2`](gmm_v2.md#gmm_v2), [`generate_block_specs`](gmm_v2.md#generate_block_specs), [`kernel_main`](gmm_v2.md#kernel_main), [`RhsRef`](gmm_v2.md#RhsRef), [`get_scale`](gmm_v2.md#RhsRef.get_scale), [`gate`](gmm_v2.md#FusedWeightsRef.gate), [`get_bias`](gmm_v2.md#FusedWeightsRef.get_bias), [`get_scale`](gmm_v2.md#FusedWeightsRef.get_scale), [`get_weight`](gmm_v2.md#FusedWeightsRef.get_weight), [`up`](gmm_v2.md#FusedWeightsRef.up), [`get_bias`](gmm_v2.md#RhsRef.get_bias), [`get_weight`](gmm_v2.md#RhsRef.get_weight)

## Functions
- `_gmm_vmem_estimate(tn: int, tk: int)` — [`L934`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L934)
- `_is_tile_k_quant_block_compatible(tk: int)` — [`L922`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L922)
- `_matmul(is_first_k_step: bool, is_last_k_step: bool)` — [`L376`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L376)
- `align_to(x, a)` — [`L91`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L91)
- `apply_act_fn(acc: jax.Array, fuse_act: str | None)` — [`L53`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L53) — Applies a fused activation function to the accumulator.
- `calculate_tiling(dims: Dimensions, lhs_cfgs: InputConfigs, rhs_cfgs: InputConfigs, vmem_limit_bytes: int, fuse_act: str | None = None)` — [`L884`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L884) — Calculate optimal tile sizes for GMM kernel.
- `fill_metadata(lhs_group_sizes_ref: jax.Array, group_offset_ref: jax.Array, metadata_ref: MetadataRef, *, cfgs: GmmConfigs)` — [`L608`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L608) — Fills the metadata for the given lhs group sizes and group offset.
- `fill_zero(i, zero_size, *, start, end)` — [`L729`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L729)
- `generate_block_specs(metadata_ref: MetadataRef, cfgs: GmmConfigs)` — [`L288`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L288) — Generates block specs for the given lhs, rhs, and out refs.
- `get_cost_estimate(cfgs: GmmConfigs)` — [`L1043`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L1043) — Returns the cost estimate for the GMM kernel.
- `get_metadata(cfgs: GmmConfigs)` — [`L1182`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L1182)
- `get_scope_name(cfgs: GmmConfigs)` — [`L1078`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L1078)
- `gmm_v2(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, rhs_scale: jax.Array | None = None, rhs_bias: jax.Array | None = None, group_offset: jax.Array | None = None, *, tile_info: TileSizes | TileFn = calculate_tiling, vmem_limit_bytes: int | None = None, precision: jax.lax.Precision = jax.lax.Precision.DEFAULT, preferred_element_type: jnp.dtype | None = None, acc_dtype: jnp.dtype | None = None, maybe_quantize_lhs: bool = True, zero_initialize: bool = True, fuse_act: str | None = None)` — [`L1203`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L1203) — GMM kernel implemented with emit_pipeline.
- `inner_kernel(tiled_lhs_ref: jax.Array, tiled_rhs_ref: RhsRef, tiled_out_ref: jax.Array, partial_out_ref: jax.Array, acc_ref: jax.Array, metadata_ref: MetadataRef, *, cfgs: GmmConfigs)` — [`L341`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L341) — Inner kernel invoked by emit_pipeline to perform matmul.
- `inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id)` — [`L638`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L638)
- `kernel_main(lhs_group_sizes_ref: jax.Array, group_offset_ref: jax.Array, lhs_ref: jax.Array, rhs_ref: WeightsRef, out_ref: jax.Array, partial_out_ref: jax.Array, acc_ref: jax.Array, metadata_ref: MetadataRef, zero_ref: jax.Array | None, semaphore_ref: jax.Array | None, *, cfgs: GmmConfigs)` — [`L780`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L780) — Entry point for GMM kernel.
- `left_fill_zero(i, zero_size)` — [`L747`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L747)
- `make_gmm_configs(lhs: jax.Array, rhs: jax.Array, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None, group_sizes: jax.Array, group_offset: jax.Array, *, tile_info: TileSizes | TileFn, vmem_limit_bytes: int | None, out_dtype: jnp.dtype | None, acc_dtype: jnp.dtype | None, maybe_quantize_lhs: bool, zero_initialize: bool, fuse_act: str | None = None)` — [`L1087`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L1087) — Fills the GMM config for the GMM kernel.
- `matmul()` — [`L579`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L579)
- `matmul_first()` — [`L575`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L575)
- `matmul_first_last()` — [`L571`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L571)
- `matmul_last()` — [`L583`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L583)
- `outer_group_loop(lhs_group_id, carry)` — [`L652`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L652)
- `right_fill_zero(i, zero_size)` — [`L754`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L754)
- `silu_and_mul_with_clamp(gate: jax.Array, up: jax.Array, limit: float = 10)` — [`L42`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L42) — Activation used in some models DeepSeek V4.
- `swigluoai(gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7)` — [`L29`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L29) — Activation used in some models such as GPT-OSS.
- `validate_inputs(lhs: jax.Array, rhs: jax.Array, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None, group_sizes: jax.Array, group_offset: jax.Array, fuse_act: str | None = None)` — [`L997`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L997) — Validates the inputs for the GMM kernel.
- `zero_out_end(out_ref: jax.Array, semaphore_ref: jax.Array, zero_size: jax.Array, *, dims: Dimensions)` — [`L765`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L765)
- `zero_out_start(out_ref: jax.Array, zero_ref: jax.Array, semaphore_ref: jax.Array, metadata_ref: MetadataRef, num_gm: jax.Array, *, dims: Dimensions)` — [`L697`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L697) — Zero out output rows that are not used in the computation.

## Module values
- `TileFn` — [`L233`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py#L233)

