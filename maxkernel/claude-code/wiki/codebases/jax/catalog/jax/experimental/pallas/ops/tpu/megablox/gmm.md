---
title: 'Module: jax/experimental/pallas/ops/tpu/megablox/gmm.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/tpu/megablox/gmm.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.tpu.megablox.gmm`/
symbols:
  gmm: gmm().
  tgmm: tgmm().
  make_group_metadata: make_group_metadata().
  tgmm.kernel._do: tgmm().kernel()._do().
  _validate_args: _validate_args().
  gmm.kernel._accum: gmm().kernel()._accum().
  _zero_uninitialized_memory: _zero_uninitialized_memory().
  _get_store_mask: _get_store_mask().
  gmm.kernel.mask_k_rem: gmm().kernel().mask_k_rem().
  tgmm.kernel: tgmm().kernel().
  gmm.kernel: gmm().kernel().
  gmm.kernel._store_accum: gmm().kernel()._store_accum().
  gmm.kernel._zero_acc: gmm().kernel()._zero_acc().
  _get_group_size: _get_group_size().
  tgmm.kernel._zero_acc: tgmm().kernel()._zero_acc().
  tgmm.kernel._store_accum: tgmm().kernel()._store_accum().
  LutFn: LutFn.
  partial: partial.
  _calculate_irregular_num_tiles: _calculate_irregular_num_tiles().
  GroupMetadata: GroupMetadata.
  gmm.kernel._zero_acc._init_out: gmm().kernel()._zero_acc()._init_out().
  gmm.rhs_transform_indices: gmm().rhs_transform_indices().
  _calculate_num_tiles: _calculate_num_tiles().
  gmm.lhs_transform_indices: gmm().lhs_transform_indices().
  gmm.out_transform_indices: gmm().out_transform_indices().
  tgmm.lhs_transform_indices: tgmm().lhs_transform_indices().
  tgmm.rhs_transform_indices: tgmm().rhs_transform_indices().
  tgmm.out_transform_indices: tgmm().out_transform_indices().
---
# Module: [`jax/experimental/pallas/ops/tpu/megablox/gmm.py`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py)

## Functions
- `_accum(is_last_k_tile)` — [`L445`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L445)
- `_calculate_irregular_num_tiles(x: int, tx: int)` — [`L69`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L69)
- `_calculate_num_tiles(x: int, tx: int)` — [`L62`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L62)
- `_do()` — [`L671`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L671)
- `_get_group_size(*, grid_id: jnp.ndarray, group_metadata: GroupMetadata)` — [`L258`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L258) — Calculate the number of rows in the current group.
- `_get_store_mask(*, grid_id: jnp.ndarray, group_metadata: GroupMetadata, tm: int, tn: int)` — [`L269`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L269) — Mask for rows that belong to the current group in the current tile.
- `_init_out()` — [`L421`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L421)
- `_store_accum()` — [`L433`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L433)
- `_store_accum()` — [`L711`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L711)
- `_validate_args(*, lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, expected_rhs_dims: int = 3)` — [`L32`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L32) — Validates the arguments for the gmm function.
- `_zero_acc()` — [`L409`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L409)
- `_zero_acc()` — [`L662`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L662)
- `_zero_uninitialized_memory(out: jnp.ndarray, *, start_group: jnp.ndarray, num_nonzero_groups: int, group_metadata: GroupMetadata)` — [`L286`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L286) — Zero out uninitialized memory from output.
- `gmm(lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, preferred_element_type: jnp.dtype = jnp.float32, tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128), group_offset: jnp.ndarray | None = None, existing_out: jnp.ndarray | None = None, transpose_rhs: bool = False, interpret: bool = False)` — [`L314`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L314) — Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.
- `kernel(group_metadata, group_offset, lhs, rhs, existing_out, out, acc_scratch)` — [`L393`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L393)
- `kernel(group_metadata, group_offset, lhs, rhs, existing_out, out, acc_scratch)` — [`L642`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L642)
- `lhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L476`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L476)
- `lhs_transform_indices(n_i, k_i, grid_id, group_metadata, group_offset)` — [`L717`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L717)
- `make_group_metadata(*, group_sizes: jnp.ndarray, m: int, tm: int, start_group: jnp.ndarray, num_nonzero_groups: int, visit_empty_groups: bool = True)` — [`L79`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L79) — Create the metadata needed for grouped matmul computation.
- `mask_k_rem(x, *, dim)` — [`L424`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L424)
- `out_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L495`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L495)
- `out_transform_indices(n_i, k_i, grid_id, group_metadata, group_offset)` — [`L729`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L729)
- `rhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L482`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L482)
- `rhs_transform_indices(n_i, k_i, grid_id, group_metadata, group_offset)` — [`L723`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L723)
- `tgmm(lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, preferred_element_type: jnp.dtype = jnp.float32, tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128), group_offset: jnp.ndarray | None = None, num_actual_groups: int | None = None, existing_out: jnp.ndarray | None = None, interpret: bool = False)` — [`L573`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L573) — Compute lhs[:, sizes[i-1]:sizes[i]] @ rhs[sizes[i-1]:sizes[i], :].

## Module values
- `GroupMetadata` — [`L76`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L76)
- `LutFn` — [`L302`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L302)
- `partial` — [`L29`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L29)

