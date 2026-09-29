---
title: 'Module: python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm`/
symbols:
  gmm: gmm().
  gmm.kernel._accum: gmm().kernel()._accum().
  _get_store_mask: _get_store_mask().
  make_group_metadata: make_group_metadata().
  _zero_uninitialized_memory: _zero_uninitialized_memory().
  gmm.kernel: gmm().kernel().
  partial: partial.
  GroupMetadata: GroupMetadata#
  GroupMetadata.group_offsets: GroupMetadata#group_offsets.
  _validate_args: _validate_args().
  _calculate_irregular_num_tiles: _calculate_irregular_num_tiles().
  GroupMetadata.group_ids: GroupMetadata#group_ids.
  gmm.kernel.mask_k_rem: gmm().kernel().mask_k_rem().
  gmm.kernel._accum._wrapper: gmm().kernel()._accum()._wrapper().
  _calculate_num_tiles: _calculate_num_tiles().
  GroupMetadata.m_tile_ids: GroupMetadata#m_tile_ids.
  LutFn: LutFn.
  gmm.lhs_transform_indices: gmm().lhs_transform_indices().
  gmm.rhs_transform_indices: gmm().rhs_transform_indices().
  gmm.rhs_scale_transform_indices: gmm().rhs_scale_transform_indices().
  gmm.rhs_bias_transform_indices: gmm().rhs_bias_transform_indices().
  gmm.out_transform_indices: gmm().out_transform_indices().
  gmm.kernel._zero_acc: gmm().kernel()._zero_acc().
  gmm.kernel._zero_acc._init_out: gmm().kernel()._zero_acc()._init_out().
---
# Module: [`python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py)

## Classes
### `GroupMetadata`
- def: [`python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py:93`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L93)
- signature: `class GroupMetadata:`
- members:
  - `group_ids` — [`L95`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L95)
  - `group_offsets` — [`L94`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L94)
  - `m_tile_ids` — [`L96`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L96)
- used by: [`gmm`](gmm.md#gmm), [`_get_store_mask`](gmm.md#_get_store_mask), [`_zero_uninitialized_memory`](gmm.md#_zero_uninitialized_memory), [`make_group_metadata`](gmm.md#make_group_metadata)

## Functions
- `_accum(is_last_k_tile)` — [`L456`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L456)
- `_calculate_irregular_num_tiles(x: int, tx: int)` — [`L84`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L84)
- `_calculate_num_tiles(x: int, tx: int)` — [`L77`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L77)
- `_get_store_mask(*, grid_id: jnp.ndarray, group_metadata: GroupMetadata, tm: int, tn: int)` — [`L269`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L269) — Mask for rows that belong to the current group in the current tile.
- `_init_out()` — [`L444`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L444)
- `_validate_args(*, lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, rhs_scale: jnp.ndarray | None = None, rhs_bias: jnp.ndarray | None = None)` — [`L23`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L23) — Validates the arguments for the gmm function.
- `_wrapper(x)` — [`L462`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L462)
- `_zero_acc()` — [`L434`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L434)
- `_zero_uninitialized_memory(out: jnp.ndarray, *, start_group: jnp.ndarray, num_nonzero_groups: int, group_metadata: GroupMetadata)` — [`L285`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L285) — Zero out uninitialized memory from output.
- `gmm(lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, preferred_element_type: jnp.dtype = jnp.float32, rhs_scale: jnp.ndarray | None = None, rhs_bias: jnp.ndarray | None = None, tiling: tuple[int, int, int] | LutFn | None = None, group_offset: jnp.ndarray | None = None, existing_out: jnp.ndarray | None = None, interpret: bool = False)` — [`L311`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L311) — Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.
- `kernel(group_metadata, group_offset, lhs, rhs, rhs_scale, rhs_bias, existing_out, out, acc_scratch)` — [`L416`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L416)
- `lhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L505`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L505)
- `make_group_metadata(*, group_sizes: jnp.ndarray, m: int, tm: int, start_group: jnp.ndarray, num_nonzero_groups: int, visit_empty_groups: bool = True)` — [`L99`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L99) — Create the metadata needed for grouped matmul computation.
- `mask_k_rem(x, *, dim)` — [`L447`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L447)
- `out_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L532`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L532)
- `rhs_bias_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L527`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L527)
- `rhs_scale_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L521`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L521)
- `rhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L511`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L511)

## Module values
- `LutFn` — [`L300`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L300)
- `partial` — [`L20`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm.py#L20)

