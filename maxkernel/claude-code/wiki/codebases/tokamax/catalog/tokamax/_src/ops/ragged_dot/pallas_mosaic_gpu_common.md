---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_gpu_common`/
symbols:
  Config.block_k: Config#block_k.
  Config.block_m: Config#block_m.
  Config.block_n: Config#block_n.
  Config.num_stages: Config#num_stages.
  Config.grid_minor_dim: Config#grid_minor_dim.
  GroupInfo.create_aligned: GroupInfo#create_aligned().
  Config.grid_tile_width: Config#grid_tile_width.
  Config.split_k: Config#split_k.
  GroupInfo: GroupInfo#
  MatmulDimension: MatmulDimension#
  Config.collective: Config#collective.
  Config: Config#
  Config.persistent: Config#persistent.
  GroupInfo.group_id: GroupInfo#group_id.
  GroupInfo.block_start: GroupInfo#block_start.
  GroupInfo.start_within_block: GroupInfo#start_within_block.
  GroupInfo.actual_size: GroupInfo#actual_size.
  MatmulDimension.M: MatmulDimension#M.
  Config.split_m: Config#split_m.
  Config.post_scale: Config#post_scale.
  get_smem_capacity: get_smem_capacity().
  MatmulDimension.N: MatmulDimension#N.
  check_bf16xbf16_or_f16xf16: check_bf16xbf16_or_f16xf16().
  SMEM_CAPACITY_MAP: SMEM_CAPACITY_MAP.
  GroupInfo.actual_start: GroupInfo#actual_start.
  GroupInfo.actual_end: GroupInfo#actual_end.
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py)

## Classes
### `Config`
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py:46`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L46)
- doc: Configuration for the ragged dot kernel.
- signature: `class Config:`
- members:
  - `block_k` — [`L51`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L51) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_gpu_common.md)
  - `block_m` — [`L49`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L49) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_gpu_common.md)
  - `block_n` — [`L50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L50) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_gpu_common.md)
  - `collective` — [`L58`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L58)
  - `grid_minor_dim` — [`L60`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L60)
  - `grid_tile_width` — [`L62`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L62)
  - `num_stages` — [`L52`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L52) — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_gpu_common.md)
  - `persistent` — [`L55`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L55)
  - `post_scale` — [`L56`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L56)
  - `split_k` — [`L53`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L53)
  - `split_m` — [`L54`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L54)
- uses (calls/refs, reference-scoped): [`MatmulDimension`](pallas_mosaic_gpu_common.md#MatmulDimension), [`N`](pallas_mosaic_gpu_common.md#MatmulDimension.N)
- used by: [`ragged_dot_gpu_fp8_quant_blackwell_kernel`](pallas_mosaic_gpu_kernel_sm100_fp8_quant.md#ragged_dot_gpu_fp8_quant_blackwell_kernel), [`_fwd`](pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot._fwd), [`ragged_dot_gpu_i8_quant_blackwell_kernel`](pallas_mosaic_gpu_kernel_sm100_i8_quant.md#ragged_dot_gpu_i8_quant_blackwell_kernel), [`ragged_dot_gpu_quant_blackwell_kernel`](pallas_mosaic_gpu_kernel_sm100_quant.md#ragged_dot_gpu_quant_blackwell_kernel), [`_get_sm90_autotuning_configs`](pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot._get_sm90_autotuning_configs), [`ragged_dot_gpu_quant_post_scale_blackwell_kernel`](pallas_mosaic_gpu_kernel_sm100_quant_post_scale.md#ragged_dot_gpu_quant_post_scale_blackwell_kernel), [`ragged_dot_quantized_kernel`](pallas_mosaic_gpu_kernel_sm90_quant.md#ragged_dot_quantized_kernel), [`ragged_dot_gpu_non_quant_blackwell_kernel`](pallas_mosaic_gpu_kernel_sm100.md#ragged_dot_gpu_non_quant_blackwell_kernel), [`_get_heuristics_config`](pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot._get_heuristics_config), [`ragged_dot_kernel`](pallas_mosaic_gpu_kernel_sm90.md#ragged_dot_kernel), [`_generate_configs`](pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot._generate_configs), [`_CONFIG`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#_CONFIG), [`_CONFIG`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#_CONFIG), [`Config`](pallas_mosaic_gpu.md#Config), [`fn`](pallas_mosaic_gpu_test.md#PallasMosaicGpuRaggedDotTest.fn), [`_ragged_contracting_dim_dot_kernel_body`](pallas_mosaic_gpu_kernel_sm90.md#_ragged_contracting_dim_dot_kernel_body), [`ragged_contracting_dim_dot_kernel`](pallas_mosaic_gpu_kernel_sm90.md#ragged_contracting_dim_dot_kernel), [`_CONFIG`](pallas_mosaic_gpu_test.md#_CONFIG), [`fn`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest.fn), [`acc_scope`](pallas_mosaic_gpu_kernel_sm90.md#_ragged_contracting_dim_dot_kernel_body.acc_scope), [`compute_acc`](pallas_mosaic_gpu_kernel_sm90.md#ragged_dot_kernel.kernel.mn_loop_body.body.compute_acc), [`fn`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest.fn), [`kernel`](pallas_mosaic_gpu_kernel_sm90_quant.md#ragged_dot_quantized_kernel.kernel), [`kernel`](pallas_mosaic_gpu_kernel_sm90.md#ragged_dot_kernel.kernel), [`mn_loop`](pallas_mosaic_gpu_kernel_sm100_i8_quant.md#ragged_dot_gpu_i8_quant_blackwell_kernel.kernel.mn_loop), [`mn_loop_body`](pallas_mosaic_gpu_kernel_sm90_quant.md#ragged_dot_quantized_kernel.kernel.mn_loop_body), [`mn_loop_body`](pallas_mosaic_gpu_kernel_sm90.md#ragged_dot_kernel.kernel.mn_loop_body), [`mn_loop`](pallas_mosaic_gpu_kernel_sm100_fp8_quant.md#ragged_dot_gpu_fp8_quant_blackwell_kernel.kernel.mn_loop), [`mn_loop`](pallas_mosaic_gpu_kernel_sm100_quant_post_scale.md#ragged_dot_gpu_quant_post_scale_blackwell_kernel.kernel.mn_loop), [`mn_loop`](pallas_mosaic_gpu_kernel_sm100_quant.md#ragged_dot_gpu_quant_blackwell_kernel.kernel.mn_loop), [`mn_loop`](pallas_mosaic_gpu_kernel_sm100.md#ragged_dot_gpu_non_quant_blackwell_kernel.kernel.mn_loop)

### `GroupInfo`
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py:66`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L66)
- doc: Information regarding the group being processed in a block.
- signature: `class GroupInfo:`
- members:
  - `create_aligned(cls, group_sizes: Sequence[jax.Array], tile: int, tid_size: int, align_tile: int = 8)` — [`L77`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L77) — Creates a GroupInfo instance with block-aligned task assignments.
  - `actual_end` — [`L72`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L72)
  - `actual_size` — [`L74`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L74)
  - `actual_start` — [`L71`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L71)
  - `block_start` — [`L70`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L70)
  - `group_id` — [`L69`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L69)
  - `start_within_block` — [`L73`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L73)
- used by: [`ragged_dot_gpu_fp8_quant_blackwell_kernel`](pallas_mosaic_gpu_kernel_sm100_fp8_quant.md#ragged_dot_gpu_fp8_quant_blackwell_kernel), [`ragged_dot_gpu_i8_quant_blackwell_kernel`](pallas_mosaic_gpu_kernel_sm100_i8_quant.md#ragged_dot_gpu_i8_quant_blackwell_kernel), [`ragged_dot_gpu_quant_blackwell_kernel`](pallas_mosaic_gpu_kernel_sm100_quant.md#ragged_dot_gpu_quant_blackwell_kernel), [`ragged_dot_gpu_quant_post_scale_blackwell_kernel`](pallas_mosaic_gpu_kernel_sm100_quant_post_scale.md#ragged_dot_gpu_quant_post_scale_blackwell_kernel), [`ragged_dot_quantized_kernel`](pallas_mosaic_gpu_kernel_sm90_quant.md#ragged_dot_quantized_kernel), [`ragged_dot_gpu_non_quant_blackwell_kernel`](pallas_mosaic_gpu_kernel_sm100.md#ragged_dot_gpu_non_quant_blackwell_kernel), [`ragged_dot_kernel`](pallas_mosaic_gpu_kernel_sm90.md#ragged_dot_kernel), [`test_create_aligned`](pallas_mosaic_gpu_common_test.md#GroupInfoTest.test_create_aligned), [`test_create_aligned_empty_groups`](pallas_mosaic_gpu_common_test.md#GroupInfoTest.test_create_aligned_empty_groups), [`test_create_aligned_multiple_blocks_per_group`](pallas_mosaic_gpu_common_test.md#GroupInfoTest.test_create_aligned_multiple_blocks_per_group), [`test_create_aligned_multiple_blocks_per_group_aligned`](pallas_mosaic_gpu_common_test.md#GroupInfoTest.test_create_aligned_multiple_blocks_per_group_aligned)

### `MatmulDimension`  ·  implements/extends IntEnum
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py:40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L40)
- signature: `class MatmulDimension(enum.IntEnum):`
- members:
  - `M` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L41)
  - `N` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L42)
- used by: [`_get_sm90_autotuning_configs`](pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot._get_sm90_autotuning_configs), [`_get_heuristics_config`](pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot._get_heuristics_config), [`_generate_configs`](pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot._generate_configs), [`_CONFIG`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#_CONFIG), [`_CONFIG`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#_CONFIG), [`_get_sm100_autotuning_configs`](pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot._get_sm100_autotuning_configs), [`grid_minor_dim`](pallas_mosaic_gpu_common.md#Config.grid_minor_dim)

## Functions
- `check_bf16xbf16_or_f16xf16(lhs: jax.Array, rhs: jax.Array)` — [`L162`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L162)
- `get_smem_capacity()` — [`L145`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L145) — Returns the shared memory capacity of the device.

## Module values
- `SMEM_CAPACITY_MAP` — [`L29`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L29)

