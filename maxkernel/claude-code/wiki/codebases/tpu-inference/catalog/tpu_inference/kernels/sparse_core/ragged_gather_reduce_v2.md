---
title: 'Module: tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.sparse_core.ragged_gather_reduce_v2`/
symbols:
  ragged_gather_reduce: ragged_gather_reduce().
  main_kernel: main_kernel().
  main_kernel.row_pipeline: main_kernel().row_pipeline().
  _calculate_num_column_partitions: _calculate_num_column_partitions().
  _Config.row_chunk_size: _Config#row_chunk_size().
  _calculate_col_chunk_size: _calculate_col_chunk_size().
  main_kernel.row_pipeline.col_pipeline: main_kernel().row_pipeline().col_pipeline().
  main_kernel.row_pipeline.col_pipeline.col_loop: main_kernel().row_pipeline().col_pipeline().col_loop().
  _Scratch.prev_dst_row_smem: _Scratch#prev_dst_row_smem.
  _Scratch.sorted_by_validity_vmem: _Scratch#sorted_by_validity_vmem.
  _Config.row_shift: _Config#row_shift().
  _preprocess: _preprocess().
  _Config.num_row_partitions: _Config#num_row_partitions.
  _Config.num_column_partitions: _Config#num_column_partitions.
  _Config.col_size: _Config#col_size.
  _Config.num_row_subchunks: _Config#num_row_subchunks.
  _Config.num_simd_lanes: _Config#num_simd_lanes.
  _Config.in_dtype: _Config#in_dtype.
  _Inputs.sorted_by_validity: _Inputs#sorted_by_validity.
  _Scratch.num_rows_per_row_partition_vmem: _Scratch#num_rows_per_row_partition_vmem.
  _Scratch.prev_iter_last_row_vmem: _Scratch#prev_iter_last_row_vmem.
  _Scratch.sem: _Scratch#sem.
  _align_to: _align_to().
  _Config: _Config#
  _Config.reduce_group_size: _Config#reduce_group_size.
  _Config.col_chunk_size: _Config#col_chunk_size.
  _Config.topk_dtype: _Config#topk_dtype.
  _Config.core_axis_name: _Config#core_axis_name.
  _Config.subcore_axis_name: _Config#subcore_axis_name.
  _Inputs: _Inputs#
  _Inputs.num_src_rows_per_row_partition: _Inputs#num_src_rows_per_row_partition.
  _Inputs.x: _Inputs#x.
  _Inputs.indices: _Inputs#indices.
  _Inputs.topk_weights: _Inputs#topk_weights.
  _Scratch: _Scratch#
  _Scratch.src_indices_vmem: _Scratch#src_indices_vmem.
  _Scratch.dst_indices_vmem: _Scratch#dst_indices_vmem.
  _Scratch.tw_f32_vmem: _Scratch#tw_f32_vmem.
  _Scratch.dma_src_row_vmem: _Scratch#dma_src_row_vmem.
  _Scratch.dma_dst_row_vmem: _Scratch#dma_dst_row_vmem.
  _Scratch.prev_dst_val_vmem: _Scratch#prev_dst_val_vmem.
  _Scratch.out_vmem: _Scratch#out_vmem.
  _CostModelConstants: _CostModelConstants#
  _fallback_implementation: _fallback_implementation().
  _calculate_row_tiling: _calculate_row_tiling().
  _pack_scalars_to_vector: _pack_scalars_to_vector().
  main_kernel.row_pipeline.get_dst_idx: main_kernel().row_pipeline().get_dst_idx().
  _CostModelConstants.MAX_ITERATIONS: _CostModelConstants#MAX_ITERATIONS.
  _CostModelConstants.MAX_COL_CHUNK_SIZE: _CostModelConstants#MAX_COL_CHUNK_SIZE.
  _row_gather_spec: _row_gather_spec().
  _Scratch.__len__: _Scratch#__len__().
  _Scratch.__getitem__: _Scratch#__getitem__().
---
# Module: [`tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py)

## Classes
### `_Config`
- def: [`tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py:29`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L29)
- signature: `class _Config:`
- members:
  - `row_chunk_size(self)` — [`L43`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L43) — Number of rows handled per row-pipeline block.
  - `row_shift(self)` — [`L48`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L48) — log2 of how many source rows pack into one uint32 gather element.
  - `col_chunk_size` — [`L34`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L34)
  - `col_size` — [`L33`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L33)
  - `core_axis_name` — [`L39`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L39)
  - `in_dtype` — [`L38`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L38)
  - `num_column_partitions` — [`L31`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L31)
  - `num_row_partitions` — [`L30`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L30)
  - `num_row_subchunks` — [`L35`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L35)
  - `num_simd_lanes` — [`L36`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L36)
  - `reduce_group_size` — [`L32`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L32)
  - `subcore_axis_name` — [`L40`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L40)
  - `topk_dtype` — [`L37`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L37)
- used by: [`ragged_gather_reduce`](ragged_gather_reduce_v2.md#ragged_gather_reduce), [`main_kernel`](ragged_gather_reduce_v2.md#main_kernel), [`row_pipeline`](ragged_gather_reduce_v2.md#main_kernel.row_pipeline), [`col_loop`](ragged_gather_reduce_v2.md#main_kernel.row_pipeline.col_pipeline.col_loop), [`col_pipeline`](ragged_gather_reduce_v2.md#main_kernel.row_pipeline.col_pipeline)

### `_CostModelConstants`
- def: [`tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py:91`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L91)
- signature: `class _CostModelConstants:`
- members:
  - `MAX_COL_CHUNK_SIZE` — [`L102`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L102)
  - `MAX_ITERATIONS` — [`L96`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L96)
- used by: [`_calculate_num_column_partitions`](ragged_gather_reduce_v2.md#_calculate_num_column_partitions), [`_calculate_col_chunk_size`](ragged_gather_reduce_v2.md#_calculate_col_chunk_size)

### `_Inputs`
- def: [`tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py:60`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L60)
- signature: `class _Inputs:`
- members:
  - `indices` — [`L63`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L63)
  - `num_src_rows_per_row_partition` — [`L61`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L61)
  - `sorted_by_validity` — [`L65`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L65)
  - `topk_weights` — [`L64`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L64)
  - `x` — [`L62`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L62)
- used by: [`ragged_gather_reduce`](ragged_gather_reduce_v2.md#ragged_gather_reduce), [`main_kernel`](ragged_gather_reduce_v2.md#main_kernel)

### `_Scratch`
- def: [`tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py:70`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L70)
- signature: `class _Scratch:`
- members:
  - `dma_dst_row_vmem` — [`L79`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L79)
  - `dma_src_row_vmem` — [`L78`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L78)
  - `dst_indices_vmem` — [`L76`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L76)
  - `num_rows_per_row_partition_vmem` — [`L71`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L71)
  - `out_vmem` — [`L81`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L81)
  - `prev_dst_row_smem` — [`L73`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L73)
  - `prev_dst_val_vmem` — [`L80`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L80)
  - `prev_iter_last_row_vmem` — [`L72`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L72)
  - `sem` — [`L82`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L82)
  - `sorted_by_validity_vmem` — [`L74`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L74)
  - `src_indices_vmem` — [`L75`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L75)
  - `tw_f32_vmem` — [`L77`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L77)
- protocol/private: `__getitem__`[`L87`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L87), `__len__`[`L84`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L84)
- used by: [`ragged_gather_reduce`](ragged_gather_reduce_v2.md#ragged_gather_reduce), [`main_kernel`](ragged_gather_reduce_v2.md#main_kernel), [`row_pipeline`](ragged_gather_reduce_v2.md#main_kernel.row_pipeline), [`col_loop`](ragged_gather_reduce_v2.md#main_kernel.row_pipeline.col_pipeline.col_loop)

## Functions
- `_align_to(a, b)` — [`L106`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L106)
- `_calculate_col_chunk_size(col_size: int, num_simd_lanes: int)` — [`L176`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L176) — Picks the column chunk size the inner pipeline gathers at a time.
- `_calculate_num_column_partitions(hidden_size: int, input_size: int, num_cores: int, num_lanes: int, num_simd_lanes: int)` — [`L124`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L124) — Calculates the number of row partitions.
- `_calculate_row_tiling(input_size: int, num_simd_lanes: int, num_row_partitions: int)` — [`L164`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L164) — Calculates the number of row subchunks and row chunk size.
- `_fallback_implementation(x: jax.Array, indices: jax.Array, topk_weights: jax.Array, valid_rows_mask: jax.Array, reduce_group_size: int)` — [`L110`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L110)
- `_pack_scalars_to_vector(scalar_list: list[jax.Array], num_simd_lanes: int)` — [`L254`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L254) — Builds a lane vector from per-lane scalars.
- `_preprocess(valid_rows_mask: jax.Array, reduce_group_size: int, num_row_partitions: int, num_simd_lanes: int, row_chunk_size: int)` — [`L205`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L205) — Sorts valid source rows to the front of each row partition.
- `_row_gather_spec(sorted_by_validity_vmem: jax.Ref, sub: int, *, num_simd_lanes: int, row_chunk_size: int)` — [`L268`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L268) — Indirect BlockSpec gathering sub-chunk ``sub``'s rows of a 1-D input.
- `col_loop(col_compute_offset)` — [`L498`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L498)
- `col_pipeline(gather_ref, sem_inner)` — [`L486`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L486)
- `get_dst_idx(global_idx)` — [`L410`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L410)
- `main_kernel(inputs: _Inputs, out_hbm_ref: jax.Ref, scratch: _Scratch, *, cfg: _Config)` — [`L283`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L283)
- `ragged_gather_reduce(x: jax.Array, indices: jax.Array, topk_weights: jax.Array, valid_rows_mask: jax.Array, reduce_group_size: int)` — [`L587`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L587) — Gathers ``x`` by ``indices``, weights and masks, then reduces by group.
- `row_pipeline(*args)` — [`L361`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py#L361)

