---
title: 'Module: src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py'
type: catalog
provenance: extracted
module: src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.kernels.megablox.pallas_mosaic_tpu_v2_tgmm_kernel`/
symbols:
  tgmm_v2: tgmm_v2().
  generate_tgmm_block_specs: generate_tgmm_block_specs().
  tgmm_inner_kernel: tgmm_inner_kernel().
  tgmm_kernel_main: tgmm_kernel_main().
  TgmmIndexMaps.lhs_index_map: TgmmIndexMaps#lhs_index_map().
  TgmmIndexMaps.rhs_index_map: TgmmIndexMaps#rhs_index_map().
  OperandRef: OperandRef#
  OperandRef.value: OperandRef#value.
  TgmmIndexMaps.metadata_ref: TgmmIndexMaps#metadata_ref.
  OperandRef.scale: OperandRef#scale.
  calculate_tgmm_tiling.within_vmem_limit: calculate_tgmm_tiling().within_vmem_limit().
  tgmm_inner_kernel._matmul: tgmm_inner_kernel()._matmul().
  TgmmIndexMaps.cfgs: TgmmIndexMaps#cfgs.
  calculate_tgmm_tiling: calculate_tgmm_tiling().
  make_tgmm_configs: make_tgmm_configs().
  tgmm_inner_kernel.matmul_new_group_and_changing: tgmm_inner_kernel().matmul_new_group_and_changing().
  tgmm_inner_kernel.matmul_new_group: tgmm_inner_kernel().matmul_new_group().
  tgmm_inner_kernel.matmul: tgmm_inner_kernel().matmul().
  tgmm_inner_kernel.matmul_group_changing: tgmm_inner_kernel().matmul_group_changing().
  TgmmIndexMaps.out_index_map: TgmmIndexMaps#out_index_map().
  zero_out_start: zero_out_start().
  TileTgmmFn: TileTgmmFn.
  get_scope_name: get_scope_name().
  get_cost_estimate: get_cost_estimate().
  TgmmIndexMaps: TgmmIndexMaps#
  TgmmIndexMaps.rhs_scale_index_map: TgmmIndexMaps#rhs_scale_index_map().
  zero_out_start.fill_zero: zero_out_start().fill_zero().
  zero_out_end: zero_out_end().
  TgmmIndexMaps.__init__: TgmmIndexMaps#__init__().
  validate_tgmm_inputs: validate_tgmm_inputs().
---
# Module: [`src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py)

## Classes
### `OperandRef`
- def: [`src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py:35`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L35)
- doc: Bundles a kernel operand with its optional per-N scale.
- signature: `class OperandRef:`
- members:
  - `scale` — [`L45`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L45)
  - `value` — [`L44`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L44)
- used by: [`tgmm_v2`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_v2), [`generate_tgmm_block_specs`](pallas_mosaic_tpu_v2_tgmm_kernel.md#generate_tgmm_block_specs), [`tgmm_inner_kernel`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_inner_kernel), [`tgmm_kernel_main`](pallas_mosaic_tpu_v2_tgmm_kernel.md#tgmm_kernel_main)

### `TgmmIndexMaps`
- def: [`src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py:398`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L398)
- doc: Index maps for TGMM kernel.
- signature: `class TgmmIndexMaps:`
- members:
  - `lhs_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array)` — [`L405`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L405)
  - `out_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array)` — [`L426`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L426)
  - `rhs_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array)` — [`L414`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L414)
  - `rhs_scale_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array)` — [`L423`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L423)
  - `cfgs` — [`L403`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L403)
  - `metadata_ref` — [`L402`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L402)
- protocol/private: `__init__`[`L401`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L401)
- used by: [`generate_tgmm_block_specs`](pallas_mosaic_tpu_v2_tgmm_kernel.md#generate_tgmm_block_specs)

## Functions
- `_matmul(is_new_group: bool, is_group_changing: bool)` — [`L311`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L311)
- `calculate_tgmm_tiling(dims: gmm_v2.Dimensions, lhs_cfgs: gmm_v2.InputConfigs, rhs_cfgs: gmm_v2.InputConfigs, vmem_limit_bytes: int, out_dtype: jnp.dtype, acc_dtype: jnp.dtype, target_zero_ref_bytes: int)` — [`L84`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L84) — Calculate optimal tile sizes for TGMM kernel.
- `fill_zero(local_group_id, should_copy)` — [`L480`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L480)
- `generate_tgmm_block_specs(metadata_ref: gmm_v2.MetadataRef, cfgs: gmm_v2.GmmConfigs)` — [`L431`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L431) — Generates block specs for the given lhs, rhs, and out refs.
- `get_cost_estimate(cfgs: gmm_v2.GmmConfigs)` — [`L71`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L71)
- `get_scope_name(cfgs: gmm_v2.GmmConfigs)` — [`L62`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L62)
- `make_tgmm_configs(lhs: jax.Array, rhs: jax.Array, rhs_scale: jax.Array, group_sizes: jax.Array, num_actual_groups: int, *, tile_info: gmm_v2.TileSizes | TileTgmmFn, vmem_limit_bytes: int | None, out_dtype: jnp.dtype, acc_dtype: jnp.dtype | None, target_zero_ref_bytes: int)` — [`L178`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L178) — Fills the GMM config for the TGMM kernel.
- `matmul()` — [`L358`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L358)
- `matmul_group_changing()` — [`L362`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L362)
- `matmul_new_group()` — [`L354`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L354)
- `matmul_new_group_and_changing()` — [`L350`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L350)
- `tgmm_inner_kernel(tiled_lhs_ref: jax.Array, tiled_rhs_ref: OperandRef, tiled_out_ref: jax.Array, acc_ref: jax.Array, metadata_ref: gmm_v2.MetadataRef, *, cfgs: gmm_v2.GmmConfigs)` — [`L277`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L277) — Inner kernel for TGMM computation.
- `tgmm_kernel_main(lhs_group_sizes_ref, group_offset_ref, lhs_ref, rhs_ref, out_ref, acc_ref: jax.Array, metadata_ref: gmm_v2.MetadataRef, zero_ref: jax.Array, semaphore_ref: jax.Array, *, cfgs)` — [`L524`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L524) — Main kernel function for TGMM computation.
- `tgmm_v2(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, num_actual_groups: int, rhs_scale: jax.Array | None = None, group_offset: jax.Array | None = None, *, tile_info: gmm_v2.TileSizes | TileTgmmFn = calculate_tgmm_tiling, vmem_limit_bytes: int | None = None, precision: jax.lax.Precision = jax.lax.Precision.DEFAULT, preferred_element_type: jnp.dtype | None = None, acc_dtype: jnp.dtype | None = None)` — [`L629`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L629) — Computes a transposed grouped matrix multiplication.
- `validate_tgmm_inputs(group_sizes: jax.Array, num_actual_groups: int, group_offset: jax.Array | None = None)` — [`L591`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L591) — Validates inputs to 'tgmm_v2'.
- `within_vmem_limit(tile_m, tile_k, tile_n)` — [`L113`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L113)
- `zero_out_end(num_groups_to_zero, out_ref, semaphore_ref)` — [`L509`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L509) — Drain the DMAs started by zero_out_start.
- `zero_out_start(lhs_group_sizes_ref, group_offset_ref, out_ref, zero_ref, semaphore_ref)` — [`L464`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L464) — If group_sizes[i]==0, kick off async DMAs to zero out drhs[i].

## Module values
- `TileTgmmFn` — [`L48`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/pallas_mosaic_tpu_v2_tgmm_kernel.py#L48)

