---
title: 'Module: ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.grouped_matmulv2._pallas_impl`/
symbols:
  grouped_matmul: grouped_matmul().
  transposed_grouped_matmul: transposed_grouped_matmul().
  grouped_matmul.kernel.accum: grouped_matmul().kernel().accum().
  transposed_grouped_matmul.kernel._do: transposed_grouped_matmul().kernel()._do().
  _get_store_mask: _get_store_mask().
  make_group_metadata: make_group_metadata().
  grouped_matmul.kernel: grouped_matmul().kernel().
  _validate_args: _validate_args().
  GroupMetadata: GroupMetadata.
  _TilingFn: _TilingFn.
  grouped_matmul.kernel.accum.iota: grouped_matmul().kernel().accum().iota().
  grouped_matmul.kernel.accum.is_int: grouped_matmul().kernel().accum().is_int().
  grouped_matmul.rhs_index_map: grouped_matmul().rhs_index_map().
  transposed_grouped_matmul.kernel._do.is_int: transposed_grouped_matmul().kernel()._do().is_int().
  grouped_matmul.kernel.dot_general: grouped_matmul().kernel().dot_general().
  grouped_matmul.lhs_index_map: grouped_matmul().lhs_index_map().
  grouped_matmul.out_index_map: grouped_matmul().out_index_map().
  transposed_grouped_matmul.kernel: transposed_grouped_matmul().kernel().
  transposed_grouped_matmul.kernel._do.dot: transposed_grouped_matmul().kernel()._do().dot().
  transposed_grouped_matmul.lhs_index_map: transposed_grouped_matmul().lhs_index_map().
  transposed_grouped_matmul.rhs_index_map: transposed_grouped_matmul().rhs_index_map().
  transposed_grouped_matmul.out_index_map: transposed_grouped_matmul().out_index_map().
  grouped_matmul.kernel._zero_acc: grouped_matmul().kernel()._zero_acc().
  transposed_grouped_matmul.kernel._zero_acc: transposed_grouped_matmul().kernel()._zero_acc().
  transposed_grouped_matmul.kernel._store_accum: transposed_grouped_matmul().kernel()._store_accum().
---
# Module: [`ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py)

## Functions
- `_do()` — [`L545`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L545)
- `_get_store_mask(*, grid_id: jax.Array, group_metadata: GroupMetadata, tm: int, tn: int)` — [`L189`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L189) — Generate a boolean mask for rows belonging to the current group in the current tile.
- `_store_accum()` — [`L572`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L572)
- `_validate_args(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, expected_rhs_dims: int = 3)` — [`L55`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L55) — Validate input arguments for grouped matrix multiplication.
- `_zero_acc()` — [`L335`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L335)
- `_zero_acc()` — [`L539`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L539)
- `accum(is_last_k_tile)` — [`L338`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L338)
- `dot(x, y, preferred_element_type)` — [`L546`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L546)
- `dot_general(x, y, preferred_element_type)` — [`L323`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L323)
- `grouped_matmul(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, preferred_element_type: jnp.dtype, tiling: tuple[int, int, int] | _TilingFn | None = (128, 128, 128), input_buffer_count: int = 2, group_offset: jax.Array | None = None, transpose_rhs: bool = False, interpret: bool = False)` — [`L236`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L236) — Compute grouped matrix multiplication: lhs[start_i:end_i, :] @ rhs[i] for each group i.
- `iota(x, d)` — [`L346`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L346)
- `is_int(x)` — [`L352`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L352)
- `is_int(x)` — [`L557`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L557)
- `kernel(group_metadata, _, lhs_ref, rhs_ref, out_ref, acc_scratch)` — [`L317`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L317)
- `kernel(group_metadata, _, lhs_ref, rhs_ref, out_ref, acc_scratch)` — [`L532`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L532)
- `lhs_index_map(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L371`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L371)
- `lhs_index_map(n_i, k_i, grid_id, group_metadata, group_offset)` — [`L575`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L575)
- `make_group_metadata(*, group_sizes: jax.Array, m: int, tm: int, start_group: jax.Array, num_nonzero_groups: int, visit_empty_groups: bool)` — [`L97`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L97) — Create the metadata needed for grouped matmul computation.
- `out_index_map(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L384`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L384)
- `out_index_map(n_i, k_i, grid_id, group_metadata, group_offset)` — [`L587`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L587)
- `rhs_index_map(n_i, grid_id, k_i, group_metadata, group_offset)` — [`L377`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L377)
- `rhs_index_map(n_i, k_i, grid_id, group_metadata, group_offset)` — [`L581`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L581)
- `transposed_grouped_matmul(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, preferred_element_type: jnp.dtype, tiling: tuple[int, int, int] | _TilingFn | None = (128, 128, 128), input_buffer_count: int = 2, group_offset: jax.Array | None = None, num_actual_groups: int | None = None, interpret: bool = False)` — [`L452`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L452) — Compute transposed grouped matrix multiplication for gradient computation.

## Module values
- `GroupMetadata` — [`L94`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L94)
- `_TilingFn` — [`L223`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_pallas_impl.py#L223)

