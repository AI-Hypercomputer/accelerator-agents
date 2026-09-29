---
title: 'Module: ejkernel/utils.py'
type: catalog
provenance: extracted
module: ejkernel/utils.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.utils`/
symbols:
  DEBUG_GLOBAL_RNG: DEBUG_GLOBAL_RNG.
  F: F.
  arch_supports_fp8: arch_supports_fp8().
  safe_autotune.decorator: safe_autotune().decorator().
  safe_autotune: safe_autotune().
  is_cdna: is_cdna().
  is_rdna: is_rdna().
  calculate_blocksize_and_wraps: calculate_blocksize_and_wraps().
  assert_close: assert_close().
  make_dummy_rpa_inputs: make_dummy_rpa_inputs().
  barrier_sync: barrier_sync().
  cdiv: cdiv().
  is_hip: is_hip().
  get_stride: get_stride().
  get_strides: get_strides().
  _sync_counter: _sync_counter.
  kw_strides: kw_strides().
  numeric_gen: numeric_gen().
  random_dense: random_dense().
  is_fp8: is_fp8().
  get_err_ratio: get_err_ratio().
  _align_to: _align_to().
  CDNA_ARCHS: CDNA_ARCHS.
  RDNA_ARCHS: RDNA_ARCHS.
  next_power_of_2: next_power_of_2().
  get_abs_err: get_abs_err().
  get_gpu_arch: get_gpu_arch().
  _dtype_packing: _dtype_packing().
  strides_from_shape: strides_from_shape().
  dtype_index: dtype_index().
  get_sharding: get_sharding().
  get_padded_headsize: get_padded_headsize().
  narrow: narrow().
  get_input_shapes: get_input_shapes().
  generate_block_indices: generate_block_indices().
  get_tpu_generation: get_tpu_generation().
  make_mesh: make_mesh().
  get_qkv_shardings: get_qkv_shardings().
  get_segments_shardings: get_segments_shardings().
  Layouts.Layouts: Layouts.Layouts.
---
# Module: [`ejkernel/utils.py`](../../../../../raw/code/ejkernel/ejkernel/utils.py)

## Functions
- `_align_to(x: int, multiple: int)` — [`L850`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L850) — Round up a value to the nearest multiple.
- `_dtype_packing(dtype: jnp.dtype)` — [`L863`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L863) — Calculate lanes per 32-bit slot for dtype packing.
- `arch_supports_fp8()` — [`L720`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L720) — Check if the current GPU architecture supports FP8 arithmetic.
- `assert_close(prefix, ref, tri, ratio, warning=False, err_atol=0.000001)` — [`L627`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L627) — Assert that two arrays are numerically close within tolerance.
- `barrier_sync(timeout: float = 200)` — [`L788`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L788) — Synchronize all JAX processes at a barrier point.
- `calculate_blocksize_and_wraps(n)` — [`L483`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L483) — Calculate optimal block size and number of warps for Triton kernels.
- `cdiv(a: int, b: int)` — [`L98`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L98)
- `decorator(fn: F)` — [`L236`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L236) — Wrap ``fn`` with Triton Autotuner, falling back to ``fn`` on failure.
- `dtype_index(x: jnp.array)` — [`L266`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L266) — Get numeric index for array dtype.
- `generate_block_indices(batch: int, num_query_blocks: int, heads: int, selected_blocks: int, block_size: int, seed: int = 42)` — [`L733`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L733) — Generate random block indices for sparse attention benchmarks.
- `get_abs_err(x, y)` — [`L582`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L582) — Calculate maximum absolute error between two arrays.
- `get_err_ratio(x, y)` — [`L602`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L602) — Calculate relative error ratio between two arrays.
- `get_gpu_arch()` — [`L702`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L702) — Get the architecture identifier string of the current GPU.
- `get_input_shapes()` — [`L416`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L416) — Generate test input shapes for benchmarking and testing.
- `get_padded_headsize(size)` — [`L337`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L337) — Calculate padded head size for optimal memory alignment.
- `get_qkv_shardings(layout: Literal["bhsd", "bshd", "thd"])` — [`L1135`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L1135) — Get sharding specifications for attention QKV tensors based on layout.
- `get_segments_shardings()` — [`L1196`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L1196) — Get sharding specifications for segment ID tensors.
- `get_sharding(arr: jax.Array)` — [`L293`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L293) — Retrieve the sharding specification of a JAX array.
- `get_stride(shape: tuple[int, ...] | jax.Array, index=0)` — [`L155`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L155) — Get the C-contiguous stride at a specific dimension index.
- `get_strides(shape: tuple[int, ...] | jax.Array)` — [`L309`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L309) — Calculate row-major (C-contiguous) strides for a given shape.
- `get_tpu_generation()` — [`L1084`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L1084) — Detect and return the current TPU generation.
- `is_cdna()` — [`L450`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L450) — Check if running on AMD CDNA architecture.
- `is_fp8(x)` — [`L680`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L680) — Check if an array uses FP8 dtype and if hardware supports it.
- `is_hip()` — [`L435`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L435) — Check if running on AMD HIP backend.
- `is_rdna()` — [`L467`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L467) — Check if running on AMD RDNA architecture.
- `kw_strides(x: Array | None, *stride_names: str)` — [`L366`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L366) — Generate stride keyword arguments for kernel calls.
- `make_dummy_rpa_inputs(*, rng_seed: int = 0, num_seqs: int = 4, pages_per_seq: int = 3, page_size: int = 16, num_q_heads: int = 8, num_kv_heads: int = 2, head_dim: int = 80, kv_dtype: jnp.dtype = jnp.float32, q_dtype: jnp.dtype | None = None, kv_len_max: int | None = None, total_q: int | None = None, total_num_pages: int | None = None, decode_prefill_mixed: tuple[int, int, int] | None = None)` — [`L884`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L884) — Generate dummy inputs for Ragged Paged Attention testing and benchmarking.
- `make_mesh(mesh_axis: tuple[int, int, int, int])` — [`L1116`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L1116) — Create a JAX mesh with standard sharding axes.
- `narrow(x, dim: int, start: int, length: int)` — [`L391`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L391) — Narrow a tensor along a specific dimension.
- `next_power_of_2(x: int)` — [`L177`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L177) — Returns the next power of two greater than or equal to `x`.
- `numeric_gen(*shape, dtype: str | jnp.dtype = jnp.float16, method: str = "normal")` — [`L518`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L518) — Generate random numeric arrays for testing and debugging.
- `random_dense(*shape, dtype: str | jnp.dtype = jnp.float16, limit: int | None = 1)` — [`L549`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L549) — Generate a random dense array with uniform distribution.
- `safe_autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None, warmup=None, rep=None)` — [`L194`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L194) — Safely apply Triton autotuning with fallback on failure.
- `strides_from_shape(shape: tuple[int, ...])` — [`L130`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L130) — Calculate row-major (C-contiguous) strides for an array with the given shape.

## Module values
- `CDNA_ARCHS` — [`L87`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L87)
- `DEBUG_GLOBAL_RNG` — [`L84`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L84)
- `F` — [`L81`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L81)
- `Layouts` — [`L93`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L93)
- `RDNA_ARCHS` — [`L90`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L90)
- `_sync_counter` — [`L784`](../../../../../raw/code/ejkernel/ejkernel/utils.py#L784)

