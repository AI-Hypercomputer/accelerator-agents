---
title: 'Module: axlearn/common/megablock/gmm_gpu.py'
type: catalog
provenance: extracted
module: axlearn/common/megablock/gmm_gpu.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.megablock.gmm_gpu`/
symbols:
  gmm: gmm().
  tgmm: tgmm().
  gmm.call_gmm: gmm().call_gmm().
  tgmm.call_tgmm: tgmm().call_tgmm().
  make_group_metadata: make_group_metadata().
  _validate_args: _validate_args().
  gmm.matmul_kernel: gmm().matmul_kernel().
  _select_input_dtype: _select_input_dtype().
  _zero_uninitialized_memory: _zero_uninitialized_memory().
  tgmm.kernel: tgmm().kernel().
  tgmm.kernel.loop_tiles_m: tgmm().kernel().loop_tiles_m().
  _calculate_num_tiles: _calculate_num_tiles().
  _generate_group_mask_cond: _generate_group_mask_cond().
  _get_tiling: _get_tiling().
  gmm.matmul_kernel.body: gmm().matmul_kernel().body().
  DEFAULT_TILING: DEFAULT_TILING.
  GroupMetadata: GroupMetadata.
  NUM_STAGES: NUM_STAGES.
  NUM_WARPS: NUM_WARPS.
  is_gpu: is_gpu().
  _assert_is_supported_dtype: _assert_is_supported_dtype().
  get_gpu_dot_precision: get_gpu_dot_precision().
  partial: partial.
---
# Module: [`axlearn/common/megablock/gmm_gpu.py`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py)

## Functions
- `_assert_is_supported_dtype(dtype: jnp.dtype)` — [`L35`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L35)
- `_calculate_num_tiles(x: int, tx: int)` — [`L89`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L89)
- `_generate_group_mask_cond(m_i: int, tm: int, group_mask_start: int, group_mask_end: int)` — [`L330`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L330) — Create mask condition for grouped matrix multiplication between matrix A and B.
- `_get_tiling(tiling: tuple[int, int, int])` — [`L350`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L350) — Validate and return the appropriate tiling dimensions for GPU.
- `_select_input_dtype(lhs: Tensor, rhs: Tensor)` — [`L40`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L40) — A type to which both input should be adapted to before dot product.
- `_validate_args(*, lhs: Tensor, rhs: Tensor, group_sizes: Tensor, expected_rhs_dims: int = 3)` — [`L62`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L62) — Validates the arguments for the gmm function.
- `_zero_uninitialized_memory(out: Tensor, *, start_group: Tensor, num_nonzero_groups: int, group_metadata: GroupMetadata)` — [`L314`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L314) — Zero out uninitialized memory from output.
- `body(k_i, o)` — [`L508`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L508)
- `call_gmm(lhs: Tensor, rhs: Tensor, g_offsets: Tensor, g_ids: Tensor, m_t_ids: Tensor)` — [`L526`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L526)
- `call_tgmm(lhs: Tensor, rhs: Tensor, g_offsets, m_t_ids, g_changing_idx)` — [`L704`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L704)
- `get_gpu_dot_precision(dtype)` — [`L51`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L51) — Get the suitable DotAlgorithmPreset for the given dtype.
- `gmm(lhs: Tensor, rhs: Tensor, group_sizes: Tensor, preferred_element_type: jnp.dtype = jnp.float32, tiling: Optional[tuple[int, int, int]] = DEFAULT_TILING, group_offset: Optional[Tensor] = None, transpose_rhs: Optional[bool] = False, interpret: Optional[bool] = False)` — [`L374`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L374) — Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.
- `is_gpu()` — [`L31`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L31)
- `kernel(lhs: Tensor, rhs: Tensor, group_offsets_ref: Tensor, m_tile_ids_ref: Tensor, group_changing_index_ref: Tensor, o_ref: Tensor)` — [`L633`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L633) — Kernel function to compute the lhs @ rhs with mask.
- `loop_tiles_m(i, o)` — [`L674`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L674)
- `make_group_metadata(*, group_sizes: Tensor, m: int, tm: int, start_group: Tensor, num_nonzero_groups: int, visit_empty_groups: bool = True)` — [`L96`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L96) — Create the metadata needed for grouped matmul computation.
- `matmul_kernel(lhs: Tensor, rhs: Tensor, group_offsets_ref: Tensor, group_ids_ref: Tensor, m_tile_ids_ref: Tensor, o_ref: Tensor)` — [`L439`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L439) — Kernel function to compute lhs @ rhs with mask.
- `tgmm(lhs: Tensor, rhs: Tensor, group_sizes: Tensor, preferred_element_type: jnp.dtype = jnp.float32, tiling: Optional[tuple[int, int, int]] = DEFAULT_TILING, group_offset: Optional[Tensor] = None, num_actual_groups: Optional[int] = None, interpret: bool = False)` — [`L566`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L566) — Compute lhs[:, sizes[i-1]:sizes[i]] @ rhs[sizes[i-1]:sizes[i], :].

## Module values
- `DEFAULT_TILING` — [`L28`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L28)
- `GroupMetadata` — [`L23`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L23)
- `NUM_STAGES` — [`L25`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L25)
- `NUM_WARPS` — [`L26`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L26)
- `partial` — [`L22`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/gmm_gpu.py#L22)

