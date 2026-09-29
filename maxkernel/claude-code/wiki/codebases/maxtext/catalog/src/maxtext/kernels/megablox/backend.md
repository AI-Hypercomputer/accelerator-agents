---
title: 'Module: src/maxtext/kernels/megablox/backend.py'
type: catalog
provenance: extracted
module: src/maxtext/kernels/megablox/backend.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.kernels.megablox.backend`/
symbols:
  gmm: gmm().
  tgmm: tgmm().
  make_group_metadata: make_group_metadata().
  gmm.kernel._accum: gmm().kernel()._accum().
  _get_store_mask: _get_store_mask().
  _calculate_irregular_num_tiles: _calculate_irregular_num_tiles().
  GroupMetadata: GroupMetadata.
  _get_group_size: _get_group_size().
  _zero_uninitialized_memory: _zero_uninitialized_memory().
  gmm.kernel: gmm().kernel().
  gmm.kernel._store_accum: gmm().kernel()._store_accum().
  tgmm.kernel: tgmm().kernel().
  _calculate_bytes: _calculate_bytes().
  tgmm.kernel._do: tgmm().kernel()._do().
  _validate_args: _validate_args().
  LutFn: LutFn.
  gmm.kernel.mask_k_rem: gmm().kernel().mask_k_rem().
  gmm.rhs_transform_indices: gmm().rhs_transform_indices().
  _calculate_num_tiles: _calculate_num_tiles().
  gmm.lhs_transform_indices: gmm().lhs_transform_indices().
  gmm.out_transform_indices: gmm().out_transform_indices().
  tgmm.lhs_transform_indices: tgmm().lhs_transform_indices().
  tgmm.rhs_transform_indices: tgmm().rhs_transform_indices().
  tgmm.out_transform_indices: tgmm().out_transform_indices().
  gmm.kernel._zero_acc: gmm().kernel()._zero_acc().
  gmm.kernel._zero_acc._init_out: gmm().kernel()._zero_acc()._init_out().
  tgmm.kernel._zero_acc: tgmm().kernel()._zero_acc().
  tgmm.kernel._store_accum: tgmm().kernel()._store_accum().
---
# Module: [`src/maxtext/kernels/megablox/backend.py`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py)

## Functions
- `_accum(is_last_k_tile)` — [`L424`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L424)
- `_calculate_bytes(x: jax.Array | qpl.QArray)` — [`L284`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L284)
- `_calculate_irregular_num_tiles(x: int, tx: int)` — [`L62`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L62)
- `_calculate_num_tiles(x: int, tx: int)` — [`L55`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L55)
- `_do()` — [`L668`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L668)
- `_get_group_size(*, grid_id: jnp.ndarray, group_metadata: GroupMetadata)` — [`L242`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L242) — Calculate the number of rows in the current group.
- `_get_store_mask(*, grid_id: jnp.ndarray, group_metadata: GroupMetadata, tm: int, tn: int)` — [`L251`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L251) — Mask for rows that belong to the current group in the current tile.
- `_init_out()` — [`L404`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L404)
- `_store_accum()` — [`L414`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L414)
- `_store_accum()` — [`L729`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L729)
- `_validate_args(*, lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, expected_rhs_dims: int = 3)` — [`L33`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L33) — Validates the arguments for the gmm function.
- `_zero_acc()` — [`L394`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L394)
- `_zero_acc()` — [`L661`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L661)
- `_zero_uninitialized_memory(out: jnp.ndarray, *, start_group: jnp.ndarray, num_nonzero_groups: int, group_metadata: GroupMetadata)` — [`L268`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L268) — Zero out uninitialized memory from output.
- `gmm(lhs: jnp.ndarray | qpl.QArray, rhs: jnp.ndarray | qpl.QArray, group_sizes: jnp.ndarray, preferred_element_type: jnp.dtype = jnp.float32, tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128), group_offset: jnp.ndarray | None = None, existing_out: jnp.ndarray | None = None, transpose_rhs: bool = False, interpret: bool = False, varying_axes: tuple[str, ...] = ())` — [`L304`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L304) — Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.
- `kernel(group_metadata, group_offset, lhs: jax.Array | qpl.QArray, rhs: jax.Array | qpl.QArray, existing_out, out, acc_scratch)` — [`L378`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L378)
- `kernel(group_metadata, group_offset, lhs, rhs, existing_out, out, acc_scratch)` — [`L641`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L641)
- `lhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L464`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L464)
- `lhs_transform_indices(n_i, k_i, grid_id, group_metadata, group_offset)` — [`L735`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L735)
- `make_group_metadata(*, group_sizes: jnp.ndarray, m: int, tm: int, start_group: jnp.ndarray, num_nonzero_groups: int, visit_empty_groups: bool = True)` — [`L72`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L72) — Create the metadata needed for grouped matmul computation.
- `mask_k_rem(x: jax.Array, *, dim: int)` — [`L407`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L407)
- `out_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L483`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L483)
- `out_transform_indices(n_i, k_i, grid_id, group_metadata, group_offset)` — [`L747`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L747)
- `rhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L470`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L470)
- `rhs_transform_indices(n_i, k_i, grid_id, group_metadata, group_offset)` — [`L741`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L741)
- `tgmm(lhs: jnp.ndarray | qpl.QArray, rhs: jnp.ndarray | qpl.QArray, group_sizes: jnp.ndarray, preferred_element_type: jnp.dtype = jnp.float32, tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128), group_offset: jnp.ndarray | None = None, num_actual_groups: int | None = None, existing_out: jnp.ndarray | None = None, interpret: bool = False, varying_axes: tuple[str, ...] = ())` — [`L575`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L575) — Compute lhs[:, sizes[i-1]:sizes[i]] @ rhs[sizes[i-1]:sizes[i], :].

## Module values
- `GroupMetadata` — [`L69`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L69)
- `LutFn` — [`L291`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/backend.py#L291)

