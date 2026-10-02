---
title: 'Module: tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.pallas_mosaic_gpu_common`/
symbols:
  ConfigBase.block_q: ConfigBase#block_q.
  ConfigBase.block_kv: ConfigBase#block_kv.
  ConfigBase: ConfigBase#
  ConfigBase.num_stages: ConfigBase#num_stages.
  cast_qkv.cast: cast_qkv().cast().
  QArray: QArray.
  decompose_mask: decompose_mask().
  cast_qkv: cast_qkv().
  load_bcast: load_bcast().
  pad_head_dim_to_next_multiple_of_min_swizzle: pad_head_dim_to_next_multiple_of_min_swizzle().
  ConfigBase.split_k: ConfigBase#split_k.
  safe_downcast: safe_downcast().
  unpack_bool_bits_tmem_native: unpack_bool_bits_tmem_native().
  MIN_SWIZZLE: MIN_SWIZZLE.
  ConfigBase.__post_init__: ConfigBase#__post_init__().
  ConfigBase.fold_q_sequence_heads: ConfigBase#fold_q_sequence_heads.
  unpack_bool_bits_tmem_native.unpack_booleans: unpack_bool_bits_tmem_native().unpack_booleans().
---
# Module: [`tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py)

## Classes
### `ConfigBase`
- def: [`tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py:41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L41) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_common.md)
- doc: Common configuration parameters for Pallas-Mosaic-GPU kernels.
- signature: `class ConfigBase:`
- members:
  - `block_kv` — [`L54`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L54) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_common.md)
  - `block_q` — [`L53`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L53) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_common.md)
  - `fold_q_sequence_heads` — [`L56`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L56)
  - `num_stages` — [`L55`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L55) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_common](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_common.md)
  - `split_k` — [`L57`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L57)
- protocol/private: `__post_init__`[`L59`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L59)
- uses (calls/refs, reference-scoped): [`Config`](pallas_mosaic_gpu_kernel_sm100.md#Config), [`Config`](pallas_mosaic_gpu_kernel_sm90.md#Config)
- used by: [`_fwd`](pallas_mosaic_gpu.md#PallasMosaicGpuFlashAttention._fwd), [`flash_attention_kernel`](pallas_mosaic_gpu_kernel_sm100.md#flash_attention_kernel), [`get_autotuning_configs`](pallas_mosaic_gpu_kernel_sm100.md#get_autotuning_configs), [`get_heuristics_config`](pallas_mosaic_gpu_kernel_sm100.md#get_heuristics_config), [`flash_attention_kernel`](pallas_mosaic_gpu_kernel_sm90.md#flash_attention_kernel), [`_estimate_smem_bytes`](pallas_mosaic_gpu_kernel_sm100.md#_estimate_smem_bytes), [`get_autotuning_configs`](pallas_mosaic_gpu_kernel_sm90.md#get_autotuning_configs), [`get_heuristics_config`](pallas_mosaic_gpu_kernel_sm90.md#get_heuristics_config), [`Config`](pallas_mosaic_gpu_kernel_sm100.md#Config), [`Config`](pallas_mosaic_gpu_kernel_sm90.md#Config), [`__post_init__`](pallas_mosaic_gpu_kernel_sm100.md#Config.__post_init__), [`pad_seq_k`](pallas_mosaic_gpu.md#PallasMosaicGpuFlashAttention.pad_seq_k)

## Functions
- `cast(x, precision)` — [`L114`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L114)
- `cast_qkv(q: jax.Array | QArray, k: jax.Array | QArray, v: jax.Array | QArray, precision: tuple[jax.lax.DotAlgorithmPreset, jax.lax.DotAlgorithmPreset])` — [`L106`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L106) — Casts Q, K, and V to the given precision.
- `decompose_mask(mask, q, k, q_indices, k_indices)` — [`L74`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L74) — Decomposes `mask` into a mask array, `is_causal`, `k_start` and `k_end`.
- `load_bcast(ref: Any, idx: tuple[int | jax.Array | pl.Slice, ...], *, layout: Any, optimized: bool = False)` — [`L140`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L140) — Loads from a reference, with given index, broadcasting if needed.
- `pad_head_dim_to_next_multiple_of_min_swizzle(x)` — [`L69`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L69)
- `safe_downcast(arr: jax.Array, target_dtype: jax.typing.DTypeLike)` — [`L231`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L231) — Clips the array to the target dtype's range before casting to prevent infinities.
- `unpack_bool_bits_tmem_native(a)` — [`L172`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L172) — Unpacks boolean bits from an int packed array in TMEM_NATIVE layout.
- `unpack_booleans(_, fa: mgpu.FragmentedArray)` — [`L189`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L189)

## Module values
- `MIN_SWIZZLE` — [`L64`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L64)
- `QArray` — [`L35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_common.py#L35)

