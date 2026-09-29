---
title: 'Module: python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2`/
symbols:
  gmm_v2: gmm_v2().
  generate_block_specs: generate_block_specs().
  inner_kernel._matmul: inner_kernel()._matmul().
  make_gmm_configs: make_gmm_configs().
  kernel_main: kernel_main().
  GmmConfigs.tiles: GmmConfigs#tiles.
  GmmConfigs.dims: GmmConfigs#dims.
  calculate_tiling: calculate_tiling().
  get_cost_estimate: get_cost_estimate().
  get_scope_name: get_scope_name().
  Dimensions.size_lhs_sublane: Dimensions#size_lhs_sublane.
  inner_kernel: inner_kernel().
  validate_inputs: validate_inputs().
  TileSizes.tile_k: TileSizes#tile_k.
  Dimensions.size_n: Dimensions#size_n.
  fill_metadata: fill_metadata().
  fill_metadata.inner_tm_loop: fill_metadata().inner_tm_loop().
  zero_out_start: zero_out_start().
  Dimensions.size_k: Dimensions#size_k.
  IndexMaps.lhs_index_map: IndexMaps#lhs_index_map().
  IndexMaps.out_index_map: IndexMaps#out_index_map().
  fill_metadata.outer_group_loop: fill_metadata().outer_group_loop().
  TileSizes.tile_n: TileSizes#tile_n.
  MetadataRef.gm_id_to_m_offset: MetadataRef#gm_id_to_m_offset.
  GmmConfigs.rhs_cfgs: GmmConfigs#rhs_cfgs.
  TileSizes.tile_m: TileSizes#tile_m.
  Dimensions.size_m: Dimensions#size_m.
  TileSizes: TileSizes#
  Dimensions: Dimensions#
  IndexMaps.metadata_ref: IndexMaps#metadata_ref.
  MetadataRef: MetadataRef#
  WeightsRef: WeightsRef#
  GmmConfigs: GmmConfigs#
  TileFn: TileFn.
  GmmConfigs.lhs_cfgs: GmmConfigs#lhs_cfgs.
  align_to: align_to().
  MetadataRef.gm_id_to_group_id: MetadataRef#gm_id_to_group_id.
  WeightsRef.weight: WeightsRef#weight.
  WeightsRef.scale: WeightsRef#scale.
  InputConfigs.quant_block_size: InputConfigs#quant_block_size.
  IndexMaps.rhs_weight_index_map: IndexMaps#rhs_weight_index_map().
  IndexMaps.rhs_bias_index_map: IndexMaps#rhs_bias_index_map().
  IndexMaps.rhs_scale_index_map: IndexMaps#rhs_scale_index_map().
  IndexMaps.rhs_scale_block_index_map: IndexMaps#rhs_scale_block_index_map().
  zero_out_end: zero_out_end().
  WeightsRef.bias: WeightsRef#bias.
  Dimensions.size_group: Dimensions#size_group.
  InputConfigs.quant_dtype: InputConfigs#quant_dtype.
  GmmConfigs.out_dtype: GmmConfigs#out_dtype.
  IndexMaps.__init__: IndexMaps#__init__().
  IndexMaps.cfgs: IndexMaps#cfgs.
  zero_out_start.fill_zero: zero_out_start().fill_zero().
  InputConfigs: InputConfigs#
  InputConfigs.has_scale: InputConfigs#has_scale.
  GmmConfigs.zero_init: GmmConfigs#zero_init.
  inner_kernel.matmul_first_last: inner_kernel().matmul_first_last().
  inner_kernel.matmul_first: inner_kernel().matmul_first().
  inner_kernel.matmul: inner_kernel().matmul().
  inner_kernel.matmul_last: inner_kernel().matmul_last().
  zero_out_start.left_fill_zero: zero_out_start().left_fill_zero().
  zero_out_start.right_fill_zero: zero_out_start().right_fill_zero().
  get_metadata: get_metadata().
  InputConfigs.has_bias: InputConfigs#has_bias.
  GmmConfigs.acc_dtype: GmmConfigs#acc_dtype.
  is_supported_by_gmm_v2: is_supported_by_gmm_v2().
  Dimensions.size_lhs_group: Dimensions#size_lhs_group.
  IndexMaps: IndexMaps#
---
# Module: [`python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py)

## Classes
### `Dimensions`
- def: [`python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py:46`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L46)
- signature: `class Dimensions:`
- members:
  - `size_group` — [`L50`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L50)
  - `size_k` — [`L48`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L48) — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
  - `size_lhs_group` — [`L51`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L51)
  - `size_lhs_sublane` — [`L52`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L52) — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
  - `size_m` — [`L47`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L47) — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
  - `size_n` — [`L49`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L49) — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
- used by: [`gmm_v2`](gmm_v2.md#gmm_v2), [`generate_block_specs`](gmm_v2.md#generate_block_specs), [`_matmul`](gmm_v2.md#inner_kernel._matmul), [`make_gmm_configs`](gmm_v2.md#make_gmm_configs), [`kernel_main`](gmm_v2.md#kernel_main), [`dims`](gmm_v2.md#GmmConfigs.dims), [`calculate_tiling`](gmm_v2.md#calculate_tiling), [`get_cost_estimate`](gmm_v2.md#get_cost_estimate), [`get_scope_name`](gmm_v2.md#get_scope_name), [`validate_inputs`](gmm_v2.md#validate_inputs), [`fill_metadata`](gmm_v2.md#fill_metadata), [`inner_tm_loop`](gmm_v2.md#fill_metadata.inner_tm_loop), [`zero_out_start`](gmm_v2.md#zero_out_start), [`lhs_index_map`](gmm_v2.md#IndexMaps.lhs_index_map), [`out_index_map`](gmm_v2.md#IndexMaps.out_index_map), [`outer_group_loop`](gmm_v2.md#fill_metadata.outer_group_loop), [`TileFn`](gmm_v2.md#TileFn), [`zero_out_end`](gmm_v2.md#zero_out_end), [`fill_zero`](gmm_v2.md#zero_out_start.fill_zero)

### `GmmConfigs`
- def: [`python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py:64`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L64)
- signature: `class GmmConfigs:`
- members:
  - `acc_dtype` — [`L70`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L70)
  - `dims` — [`L66`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L66) — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
  - `lhs_cfgs` — [`L67`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L67)
  - `out_dtype` — [`L69`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L69)
  - `rhs_cfgs` — [`L68`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L68) — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
  - `tiles` — [`L65`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L65) — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
  - `zero_init` — [`L71`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L71)
- uses (calls/refs, reference-scoped): [`Dimensions`](gmm_v2.md#Dimensions), [`TileSizes`](gmm_v2.md#TileSizes), [`InputConfigs`](gmm_v2.md#InputConfigs)
- used by: [`gmm_v2`](gmm_v2.md#gmm_v2), [`generate_block_specs`](gmm_v2.md#generate_block_specs), [`_matmul`](gmm_v2.md#inner_kernel._matmul), [`make_gmm_configs`](gmm_v2.md#make_gmm_configs), [`kernel_main`](gmm_v2.md#kernel_main), [`inner_kernel`](gmm_v2.md#inner_kernel), [`fill_metadata`](gmm_v2.md#fill_metadata), [`inner_tm_loop`](gmm_v2.md#fill_metadata.inner_tm_loop), [`lhs_index_map`](gmm_v2.md#IndexMaps.lhs_index_map), [`out_index_map`](gmm_v2.md#IndexMaps.out_index_map), [`outer_group_loop`](gmm_v2.md#fill_metadata.outer_group_loop), [`__init__`](gmm_v2.md#IndexMaps.__init__), [`get_metadata`](gmm_v2.md#get_metadata)

### `IndexMaps`
- def: [`python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py:77`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L77)
- doc: Index maps for GMM kernel.
- signature: `class IndexMaps:`
- members:
  - `lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array)` — [`L84`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L84) — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
  - `out_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array)` — [`L110`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L110) — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
  - `rhs_bias_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array)` — [`L98`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L98)
  - `rhs_scale_block_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array)` — [`L106`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L106)
  - `rhs_scale_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array)` — [`L102`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L102)
  - `rhs_weight_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array)` — [`L94`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L94)
  - `cfgs` — [`L82`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L82)
  - `metadata_ref` — [`L81`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L81)
- protocol/private: `__init__`[`L80`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L80)
- uses (calls/refs, reference-scoped): [`dims`](gmm_v2.md#GmmConfigs.dims), [`size_lhs_sublane`](gmm_v2.md#Dimensions.size_lhs_sublane), [`gm_id_to_m_offset`](gmm_v2.md#MetadataRef.gm_id_to_m_offset), [`GmmConfigs`](gmm_v2.md#GmmConfigs), [`MetadataRef`](gmm_v2.md#MetadataRef), [`gm_id_to_group_id`](gmm_v2.md#MetadataRef.gm_id_to_group_id)
- used by: [`generate_block_specs`](gmm_v2.md#generate_block_specs)

### `InputConfigs`
- def: [`python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py:56`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L56)
- signature: `class InputConfigs:`
- members:
  - `has_bias` — [`L59`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L59)
  - `has_scale` — [`L60`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L60)
  - `quant_block_size` — [`L58`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L58)
  - `quant_dtype` — [`L57`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L57)
- used by: [`gmm_v2`](gmm_v2.md#gmm_v2), [`generate_block_specs`](gmm_v2.md#generate_block_specs), [`_matmul`](gmm_v2.md#inner_kernel._matmul), [`make_gmm_configs`](gmm_v2.md#make_gmm_configs), [`rhs_cfgs`](gmm_v2.md#GmmConfigs.rhs_cfgs), [`lhs_cfgs`](gmm_v2.md#GmmConfigs.lhs_cfgs)

### `MetadataRef`
- def: [`python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py:25`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L25)
- signature: `class MetadataRef:`
- members:
  - `gm_id_to_group_id` — [`L26`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L26)
  - `gm_id_to_m_offset` — [`L27`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L27) — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
- used by: [`gmm_v2`](gmm_v2.md#gmm_v2), [`generate_block_specs`](gmm_v2.md#generate_block_specs), [`_matmul`](gmm_v2.md#inner_kernel._matmul), [`kernel_main`](gmm_v2.md#kernel_main), [`inner_kernel`](gmm_v2.md#inner_kernel), [`fill_metadata`](gmm_v2.md#fill_metadata), [`inner_tm_loop`](gmm_v2.md#fill_metadata.inner_tm_loop), [`zero_out_start`](gmm_v2.md#zero_out_start), [`lhs_index_map`](gmm_v2.md#IndexMaps.lhs_index_map), [`out_index_map`](gmm_v2.md#IndexMaps.out_index_map), [`rhs_bias_index_map`](gmm_v2.md#IndexMaps.rhs_bias_index_map), [`rhs_scale_block_index_map`](gmm_v2.md#IndexMaps.rhs_scale_block_index_map), [`rhs_scale_index_map`](gmm_v2.md#IndexMaps.rhs_scale_index_map), [`rhs_weight_index_map`](gmm_v2.md#IndexMaps.rhs_weight_index_map), [`__init__`](gmm_v2.md#IndexMaps.__init__)

### `TileSizes`
- def: [`python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py:39`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L39) — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
- signature: `class TileSizes:`
- members:
  - `tile_k` — [`L41`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L41) — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
  - `tile_m` — [`L40`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L40) — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
  - `tile_n` — [`L42`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L42) — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
- used by: [`gmm_v2`](gmm_v2.md#gmm_v2), [`generate_block_specs`](gmm_v2.md#generate_block_specs), [`_matmul`](gmm_v2.md#inner_kernel._matmul), [`make_gmm_configs`](gmm_v2.md#make_gmm_configs), [`kernel_main`](gmm_v2.md#kernel_main), [`tiles`](gmm_v2.md#GmmConfigs.tiles), [`calculate_tiling`](gmm_v2.md#calculate_tiling), [`get_scope_name`](gmm_v2.md#get_scope_name), [`inner_tm_loop`](gmm_v2.md#fill_metadata.inner_tm_loop), [`outer_group_loop`](gmm_v2.md#fill_metadata.outer_group_loop), [`TileFn`](gmm_v2.md#TileFn)

### `WeightsRef`
- def: [`python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py:32`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L32)
- signature: `class WeightsRef:`
- members:
  - `bias` — [`L35`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L35)
  - `scale` — [`L34`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L34)
  - `weight` — [`L33`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L33)
- used by: [`gmm_v2`](gmm_v2.md#gmm_v2), [`generate_block_specs`](gmm_v2.md#generate_block_specs), [`_matmul`](gmm_v2.md#inner_kernel._matmul), [`kernel_main`](gmm_v2.md#kernel_main), [`get_cost_estimate`](gmm_v2.md#get_cost_estimate), [`inner_kernel`](gmm_v2.md#inner_kernel)

## Functions
- `_matmul(is_first_k_step: bool, is_last_k_step: bool)` — [`L217`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L217) — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
- `align_to(x, a)` — [`L16`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L16)
- `calculate_tiling(lhs_dtype: jnp.dtype, rhs_dtype: jnp.dtype, dims: Dimensions, vmem_limit_bytes: int)` — [`L651`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L651) — Calculate optimal tile sizes for GMM kernel. — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
- `fill_metadata(lhs_group_sizes_ref: jax.Array, group_offset_ref: jax.Array, metadata_ref: MetadataRef, *, cfgs: GmmConfigs)` — [`L400`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L400) — Fills the metadata for the given lhs group sizes and group offset. — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
- `fill_zero(i, zero_size, *, start, end)` — [`L520`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L520)
- `generate_block_specs(metadata_ref: MetadataRef, cfgs: GmmConfigs)` — [`L124`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L124) — Generates block specs for the given lhs, rhs, and out refs. — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
- `get_cost_estimate(lhs: jax.Array, rhs: WeightsRef, out_dtype: jnp.dtype, dims: Dimensions)` — [`L760`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L760) — Returns the cost estimate for the GMM kernel.
- `get_metadata(cfgs: GmmConfigs)` — [`L887`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L887)
- `get_scope_name(dims: Dimensions, tiles: TileSizes)` — [`L792`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L792)
- `gmm_v2(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, rhs_scale: jax.Array | None = None, rhs_bias: jax.Array | None = None, group_offset: jax.Array | None = None, *, tile_info: TileSizes | TileFn = calculate_tiling, vmem_limit_bytes: int | None = None, precision: jax.lax.Precision = jax.lax.Precision.DEFAULT, preferred_element_type: jnp.dtype | None = None, acc_dtype: jnp.dtype | None = None, maybe_quantize_lhs: bool = True, zero_initialize: bool = True)` — [`L911`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L911) — GMM kernel implemented with emit_pipeline. — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
- `inner_kernel(tiled_lhs_ref: jax.Array, tiled_rhs_ref: WeightsRef, tiled_out_ref: jax.Array, partial_out_ref: jax.Array, acc_ref: jax.Array, metadata_ref: MetadataRef, *, cfgs: GmmConfigs)` — [`L182`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L182) — Inner kernel invoked by emit_pipeline to perform matmul. — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
- `inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id)` — [`L430`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L430)
- `is_supported_by_gmm_v2(rhs_scale: jax.Array | None, *, maybe_quantize_lhs: bool = False)` — [`L1067`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L1067)
- `kernel_main(lhs_group_sizes_ref: jax.Array, group_offset_ref: jax.Array, lhs_ref: jax.Array, rhs_ref: WeightsRef, out_ref: jax.Array, partial_out_ref: jax.Array, acc_ref: jax.Array, metadata_ref: MetadataRef, zero_ref: jax.Array | None, semaphore_ref: jax.Array | None, *, cfgs: GmmConfigs)` — [`L564`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L564) — Entry point for GMM kernel. — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
- `left_fill_zero(i, zero_size)` — [`L537`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L537)
- `make_gmm_configs(lhs: jax.Array, rhs: jax.Array, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None, group_sizes: jax.Array, group_offset: jax.Array, *, tile_info: TileSizes | TileFn, vmem_limit_bytes: int | None, out_dtype: jnp.dtype | None, acc_dtype: jnp.dtype | None, maybe_quantize_lhs: bool, zero_initialize: bool)` — [`L799`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L799) — Fills the GMM config for the GMM kernel. — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
- `matmul()` — [`L371`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L371)
- `matmul_first()` — [`L367`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L367)
- `matmul_first_last()` — [`L363`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L363)
- `matmul_last()` — [`L375`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L375)
- `outer_group_loop(lhs_group_id, carry)` — [`L443`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L443)
- `right_fill_zero(i, zero_size)` — [`L541`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L541)
- `validate_inputs(lhs: jax.Array, rhs: jax.Array, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None, group_sizes: jax.Array, group_offset: jax.Array)` — [`L713`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L713) — Validates the inputs for the GMM kernel. — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)
- `zero_out_end(out_ref: jax.Array, semaphore_ref: jax.Array, zero_size: jax.Array, *, dims: Dimensions)` — [`L549`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L549)
- `zero_out_start(out_ref: jax.Array, zero_ref: jax.Array, semaphore_ref: jax.Array, metadata_ref: MetadataRef, num_gm: jax.Array, *, dims: Dimensions)` — [`L488`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L488) — Zero out output rows that are not used in the computation. — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)

## Module values
- `TileFn` — [`L74`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py#L74)

