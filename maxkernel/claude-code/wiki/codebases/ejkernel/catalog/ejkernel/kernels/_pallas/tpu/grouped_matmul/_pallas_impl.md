---
title: 'Module: ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.grouped_matmul._pallas_impl`/
symbols:
  grouped_matmul: grouped_matmul().
  LutFn: LutFn.
  transposed_grouped_matmul: transposed_grouped_matmul().
  _validate_args: _validate_args().
  make_group_metadata: make_group_metadata().
  grouped_matmul.kernel._accum: grouped_matmul().kernel()._accum().
  _get_store_mask: _get_store_mask().
  _calculate_irregular_num_tiles: _calculate_irregular_num_tiles().
  GroupMetadata: GroupMetadata.
  _get_group_size: _get_group_size().
  _zero_uninitialized_memory: _zero_uninitialized_memory().
  grouped_matmul.kernel: grouped_matmul().kernel().
  grouped_matmul.kernel._store_accum: grouped_matmul().kernel()._store_accum().
  transposed_grouped_matmul.kernel: transposed_grouped_matmul().kernel().
  transposed_grouped_matmul.kernel._do: transposed_grouped_matmul().kernel()._do().
  grouped_matmul.kernel.mask_k_rem: grouped_matmul().kernel().mask_k_rem().
  grouped_matmul.rhs_transform_indices: grouped_matmul().rhs_transform_indices().
  _calculate_num_tiles: _calculate_num_tiles().
  grouped_matmul.lhs_transform_indices: grouped_matmul().lhs_transform_indices().
  grouped_matmul.out_transform_indices: grouped_matmul().out_transform_indices().
  transposed_grouped_matmul.lhs_transform_indices: transposed_grouped_matmul().lhs_transform_indices().
  transposed_grouped_matmul.rhs_transform_indices: transposed_grouped_matmul().rhs_transform_indices().
  transposed_grouped_matmul.out_transform_indices: transposed_grouped_matmul().out_transform_indices().
  grouped_matmul.kernel._zero_acc: grouped_matmul().kernel()._zero_acc().
  grouped_matmul.kernel._zero_acc._init_out: grouped_matmul().kernel()._zero_acc()._init_out().
  transposed_grouped_matmul.kernel._zero_acc: transposed_grouped_matmul().kernel()._zero_acc().
  transposed_grouped_matmul.kernel._store_accum: transposed_grouped_matmul().kernel()._store_accum().
---
# Module: [`ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py)

## Functions
- `_accum(is_last_k_tile)` — [`L603`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L603)
- `_calculate_irregular_num_tiles(x: int, tx: int)` — [`L137`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L137) — Calculate the number of tiles needed for a dimension allowing partial tiles.
- `_calculate_num_tiles(x: int, tx: int)` — [`L105`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L105) — Calculate the number of tiles needed for a dimension requiring exact divisibility.
- `_do()` — [`L883`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L883)
- `_get_group_size(*, grid_id: jnp.ndarray, group_metadata: GroupMetadata)` — [`L291`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L291) — Calculate the number of rows in the group being processed by a grid index.
- `_get_store_mask(*, grid_id: jnp.ndarray, group_metadata: GroupMetadata, tm: int, tn: int)` — [`L321`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L321) — Generate a mask for valid elements within a tile for the current group.
- `_init_out()` — [`L581`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L581)
- `_store_accum()` — [`L593`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L593)
- `_store_accum()` — [`L927`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L927)
- `_validate_args(*, lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, expected_rhs_dims: int = 3)` — [`L48`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L48) — Validate input arguments for grouped matrix multiplication operations.
- `_zero_acc()` — [`L571`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L571)
- `_zero_acc()` — [`L877`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L877)
- `_zero_uninitialized_memory(out: jnp.ndarray, *, start_group: jnp.ndarray, num_nonzero_groups: int, group_metadata: GroupMetadata)` — [`L372`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L372) — Zero out memory regions in output that weren't written by the kernel.
- `grouped_matmul(lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, preferred_element_type: DTypeLike = jnp.float32, tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128), group_offset: jnp.ndarray | None = None, existing_out: jnp.ndarray | None = None, transpose_rhs: bool = False, interpret: bool = False)` — [`L428`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L428) — Grouped Matrix Multiplication: Compute separate matrix products for each group.
- `kernel(group_metadata, group_offset, lhs: jax.Array, rhs: jax.Array, existing_out, out, acc_scratch)` — [`L555`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L555)
- `kernel(group_metadata, group_offset, lhs, rhs, existing_out, out, acc_scratch)` — [`L857`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L857)
- `lhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L633`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L633)
- `lhs_transform_indices(n_i, k_i, grid_id, group_metadata, group_offset)` — [`L933`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L933)
- `make_group_metadata(*, group_sizes: jnp.ndarray, m: int, tm: int, start_group: jnp.ndarray, num_nonzero_groups: int, visit_empty_groups: bool = True)` — [`L172`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L172) — Create metadata for efficient grouped matrix multiplication on TPU.
- `mask_k_rem(x: jax.Array, *, dim: int)` — [`L584`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L584)
- `out_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L646`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L646)
- `out_transform_indices(n_i, k_i, grid_id, group_metadata, group_offset)` — [`L943`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L943)
- `rhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L638`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L638)
- `rhs_transform_indices(n_i, k_i, grid_id, group_metadata, group_offset)` — [`L938`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L938)
- `transposed_grouped_matmul(lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, preferred_element_type: DTypeLike = jnp.float32, tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128), group_offset: jnp.ndarray | None = None, num_actual_groups: int | None = None, existing_out: jnp.ndarray | None = None, interpret: bool = False)` — [`L727`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L727) — Transposed Grouped Matrix Multiplication: Compute grouped products with transposed access pattern.

## Module values
- `GroupMetadata` — [`L169`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L169)
- `LutFn` — [`L421`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_pallas_impl.py#L421)

