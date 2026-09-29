---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_tpu_v2_tgmm_kernel`/
symbols:
  tgmm_v2: tgmm_v2().
  make_tgmm_configs: make_tgmm_configs().
  generate_tgmm_block_specs: generate_tgmm_block_specs().
  calculate_tgmm_tiling: calculate_tgmm_tiling().
  get_scope_name: get_scope_name().
  get_cost_estimate: get_cost_estimate().
  tgmm_inner_kernel: tgmm_inner_kernel().
  tgmm_kernel_main: tgmm_kernel_main().
  tgmm_inner_kernel._matmul: tgmm_inner_kernel()._matmul().
  TgmmIndexMaps.lhs_index_map: TgmmIndexMaps#lhs_index_map().
  TgmmIndexMaps.rhs_index_map: TgmmIndexMaps#rhs_index_map().
  TileTgmmFn: TileTgmmFn.
  validate_tgmm_inputs: validate_tgmm_inputs().
  calculate_tgmm_tiling.within_vmem_limit: calculate_tgmm_tiling().within_vmem_limit().
  TgmmIndexMaps.out_index_map: TgmmIndexMaps#out_index_map().
  OperandRef: OperandRef#
  OperandRef.value: OperandRef#value.
  TgmmIndexMaps.__init__: TgmmIndexMaps#__init__().
  TgmmIndexMaps.metadata_ref: TgmmIndexMaps#metadata_ref.
  OperandRef.scale: OperandRef#scale.
  TgmmIndexMaps.cfgs: TgmmIndexMaps#cfgs.
  tgmm_inner_kernel.matmul_new_group_and_changing: tgmm_inner_kernel().matmul_new_group_and_changing().
  tgmm_inner_kernel.matmul_new_group: tgmm_inner_kernel().matmul_new_group().
  tgmm_inner_kernel.matmul: tgmm_inner_kernel().matmul().
  tgmm_inner_kernel.matmul_group_changing: tgmm_inner_kernel().matmul_group_changing().
  zero_out_start: zero_out_start().
  TgmmIndexMaps: TgmmIndexMaps#
  TgmmIndexMaps.rhs_scale_index_map: TgmmIndexMaps#rhs_scale_index_map().
  zero_out_start.fill_zero: zero_out_start().fill_zero().
  zero_out_end: zero_out_end().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py)

## Classes
### `OperandRef`
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py:31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L31)
- doc: Bundles a kernel operand with its optional per-N scale.
- signature: `class OperandRef:`
- members:
  - `scale` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L41)
  - `value` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L40)
- used by: [`tgmm_v2`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_v2), [`generate_tgmm_block_specs`](pallas_mosaic_tpu_v2_tgmm_kernel.md#generate_tgmm_block_specs), [`tgmm_inner_kernel`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_inner_kernel), [`tgmm_kernel_main`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_kernel_main)

### `TgmmIndexMaps`
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py:408`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L408)
- doc: Index maps for TGMM kernel.
- signature: `class TgmmIndexMaps:`
- members:
  - `lhs_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array)` — [`L415`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L415)
  - `out_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array)` — [`L438`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L438)
  - `rhs_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array)` — [`L424`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L424)
  - `rhs_scale_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array)` — [`L433`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L433)
  - `cfgs` — [`L413`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L413)
  - `metadata_ref` — [`L412`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L412)
- protocol/private: `__init__`[`L411`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L411)
- uses (calls/refs, reference-scoped): [`dims`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs.dims), [`size_lhs_sublane`](pallas_mosaic_tpu_v2_gmm_kernel.md#Dimensions.size_lhs_sublane), [`gm_id_to_m_offset`](pallas_mosaic_tpu_v2_gmm_kernel.md#MetadataRef.gm_id_to_m_offset), [`GmmConfigs`](pallas_mosaic_tpu_v2_gmm_kernel.md#GmmConfigs), [`MetadataRef`](pallas_mosaic_tpu_v2_gmm_kernel.md#MetadataRef), [`gm_id_to_group_id`](pallas_mosaic_tpu_v2_gmm_kernel.md#MetadataRef.gm_id_to_group_id)
- used by: [`generate_tgmm_block_specs`](pallas_mosaic_tpu_v2_tgmm_kernel.md#generate_tgmm_block_specs)

## Functions
- `_matmul(is_new_group: bool, is_group_changing: bool)` — [`L314`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L314)
- `calculate_tgmm_tiling(dims: gmm_v2.Dimensions, lhs_cfgs: gmm_v2.InputConfigs, rhs_cfgs: gmm_v2.InputConfigs, vmem_limit_bytes: int, out_dtype: jnp.dtype, acc_dtype: jnp.dtype, target_zero_ref_bytes: int)` — [`L82`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L82) — Calculate optimal tile sizes for TGMM kernel. — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_tgmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_tgmm_kernel.md)
- `fill_zero(local_group_id, should_copy)` — [`L494`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L494)
- `generate_tgmm_block_specs(metadata_ref: gmm_v2.MetadataRef, cfgs: gmm_v2.GmmConfigs)` — [`L443`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L443) — Generates block specs for the given lhs, rhs, and out refs. — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_tgmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_tgmm_kernel.md)
- `get_cost_estimate(cfgs: gmm_v2.GmmConfigs)` — [`L67`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L67)
- `get_scope_name(cfgs: gmm_v2.GmmConfigs)` — [`L58`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L58)
- `make_tgmm_configs(lhs: jax.Array, rhs: jax.Array, rhs_scale: jax.Array, group_sizes: jax.Array, num_actual_groups: int, *, tile_info: gmm_v2.TileSizes | TileTgmmFn, vmem_limit_bytes: int | None, out_dtype: jnp.dtype, acc_dtype: jnp.dtype | None, target_zero_ref_bytes: int)` — [`L184`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L184) — Fills the GMM config for the TGMM kernel. — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_tgmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_tgmm_kernel.md)
- `matmul()` — [`L365`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L365)
- `matmul_group_changing()` — [`L369`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L369)
- `matmul_new_group()` — [`L361`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L361)
- `matmul_new_group_and_changing()` — [`L357`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L357)
- `tgmm_inner_kernel(tiled_lhs_ref: jax.Array, tiled_rhs_ref: OperandRef, tiled_out_ref: jax.Array, acc_ref: jax.Array, metadata_ref: gmm_v2.MetadataRef, *, cfgs: gmm_v2.GmmConfigs)` — [`L280`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L280) — Inner kernel for TGMM computation.
- `tgmm_kernel_main(lhs_group_sizes_ref, group_offset_ref, lhs_ref, rhs_ref, out_ref, acc_ref: jax.Array, metadata_ref: gmm_v2.MetadataRef, zero_ref: jax.Array, semaphore_ref: jax.Array, *, cfgs)` — [`L540`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L540) — Main kernel function for TGMM computation.
- `tgmm_v2(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, num_actual_groups: int, rhs_scale: jax.Array | None = None, group_offset: jax.Array | None = None, *, tile_info: gmm_v2.TileSizes | TileTgmmFn = calculate_tgmm_tiling, vmem_limit_bytes: int | None = None, precision: jax.lax.Precision = jax.lax.Precision.DEFAULT, preferred_element_type: jnp.dtype | None = None, acc_dtype: jnp.dtype | None = None)` — [`L647`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L647) — Computes a transposed grouped matrix multiplication. — documented in [tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_tgmm_kernel](../../../../../concepts/tokamax-_src-ops-ragged_dot-pallas_mosaic_tpu_v2_tgmm_kernel.md)
- `validate_tgmm_inputs(group_sizes: jax.Array, num_actual_groups: int, group_offset: jax.Array | None = None)` — [`L609`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L609) — Validates inputs to 'tgmm_v2'.
- `within_vmem_limit(tile_m, tile_k, tile_n)` — [`L111`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L111)
- `zero_out_end(num_groups_to_zero, out_ref, semaphore_ref)` — [`L523`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L523) — Drain the DMAs started by zero_out_start.
- `zero_out_start(lhs_group_sizes_ref, group_offset_ref, out_ref, zero_ref, semaphore_ref)` — [`L478`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L478) — If group_sizes[i]==0, kick off async DMAs to zero out drhs[i].

## Module values
- `TileTgmmFn` — [`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L44)

