---
title: 'Module: tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.pallas_mosaic_gpu_kernel_sm100`/
symbols:
  flash_attention_kernel: flash_attention_kernel().
  get_autotuning_configs: get_autotuning_configs().
  get_heuristics_config: get_heuristics_config().
  flash_attention_kernel.kernel.scale_wg.kv_loop: flash_attention_kernel().kernel().scale_wg().kv_loop().
  flash_attention_kernel.kernel.softmax_wg.compute_mask: flash_attention_kernel().kernel().softmax_wg().compute_mask().
  _estimate_smem_bytes: _estimate_smem_bytes().
  flash_attention_kernel.kernel.softmax_wg: flash_attention_kernel().kernel().softmax_wg().
  flash_attention_kernel.kernel.softmax_wg.kv_loop: flash_attention_kernel().kernel().softmax_wg().kv_loop().
  flash_attention_kernel.kernel.scale_wg.write_acc: flash_attention_kernel().kernel().scale_wg().write_acc().
  _TMEM: _TMEM.
  Config: Config#
  Config.collective: Config#collective.
  _TMEM_ROW: _TMEM_ROW.
  _MASK_PACKED_BITS: _MASK_PACKED_BITS.
  flash_attention_kernel.kernel.mma_tma_wg.per_warp.mma_warp: flash_attention_kernel().kernel().mma_tma_wg().per_warp().mma_warp().
  flash_attention_kernel.kernel.scale_wg: flash_attention_kernel().kernel().scale_wg().
  Config.__post_init__: Config#__post_init__().
  flash_attention_kernel.kernel.mma_tma_wg.per_warp.tma_load_qk_warp: flash_attention_kernel().kernel().mma_tma_wg().per_warp().tma_load_qk_warp().
  flash_attention_kernel.kernel.mma_tma_wg.per_warp.tma_load_v_warp: flash_attention_kernel().kernel().mma_tma_wg().per_warp().tma_load_v_warp().
  flash_attention_kernel.kernel.mma_tma_wg.per_warp.mma_warp.kv_loop: flash_attention_kernel().kernel().mma_tma_wg().per_warp().mma_warp().kv_loop().
  flash_attention_kernel.kernel.mma_tma_wg.per_warp.tma_load_kv_warp: flash_attention_kernel().kernel().mma_tma_wg().per_warp().tma_load_kv_warp().
  flash_attention_kernel.pre_reduce_k_range_per_qtile: flash_attention_kernel().pre_reduce_k_range_per_qtile().
  Residuals: Residuals.
  _TCGEN05_ROW: _TCGEN05_ROW.
  _load_bcast: _load_bcast.
  flash_attention_kernel.kernel: flash_attention_kernel().kernel().
  flash_attention_kernel.kernel.scale_wg.kv_loop.load_acc_tiles: flash_attention_kernel().kernel().scale_wg().kv_loop().load_acc_tiles().
  flash_attention_kernel.kernel.scale_wg.kv_loop.warp_any: flash_attention_kernel().kernel().scale_wg().kv_loop().warp_any().
  flash_attention_kernel.kernel.scale_wg.write_acc.load_acc_tiles: flash_attention_kernel().kernel().scale_wg().write_acc().load_acc_tiles().
  flash_attention_kernel.kernel.scale_wg.write_zeros: flash_attention_kernel().kernel().scale_wg().write_zeros().
  _TCGEN05: _TCGEN05.
  _MASK_PACK_DTYPE: _MASK_PACK_DTYPE.
  Config.num_tma_splits: Config#num_tma_splits.
  flash_attention_kernel.tmem: flash_attention_kernel().tmem().
  flash_attention_kernel.maybe_cluster_barrier: flash_attention_kernel().maybe_cluster_barrier().
  PagingInfo: PagingInfo.
  QArray: QArray.
  _TMEM_COL: _TMEM_COL.
  flash_attention_kernel.kernel.mma_tma_wg: flash_attention_kernel().kernel().mma_tma_wg().
  flash_attention_kernel.kernel.mma_tma_wg.per_warp.tma_load_bias_mask_warp: flash_attention_kernel().kernel().mma_tma_wg().per_warp().tma_load_bias_mask_warp().
  flash_attention_kernel.kernel.softmax_wg.kv_loop.write_alpha_to_smem: flash_attention_kernel().kernel().softmax_wg().kv_loop().write_alpha_to_smem().
  _L_BARRIER_ID: _L_BARRIER_ID.
  _ALPHA_BARRIER_OFFSET: _ALPHA_BARRIER_OFFSET.
  flash_attention_kernel.kernel.load_k_bound: flash_attention_kernel().kernel().load_k_bound().
  flash_attention_kernel.kernel.mma_tma_wg.per_warp.mma_warp.qk_mma: flash_attention_kernel().kernel().mma_tma_wg().per_warp().mma_warp().qk_mma().
  flash_attention_kernel.kernel.mma_tma_wg.per_warp.mma_warp.pv_mma: flash_attention_kernel().kernel().mma_tma_wg().per_warp().mma_warp().pv_mma().
  flash_attention_kernel.kernel.softmax_wg.needs_k_range_mask: flash_attention_kernel().kernel().softmax_wg().needs_k_range_mask().
  flash_attention_kernel.kernel.scale_wg.two_in_flight: flash_attention_kernel().kernel().scale_wg().two_in_flight().
  flash_attention_kernel.pre_reduce_k_range_per_qtile.pad_reduce: flash_attention_kernel().pre_reduce_k_range_per_qtile().pad_reduce().
  _WG_SPLAT: _WG_SPLAT.
  _DEFAULT_MASK_VALUE: _DEFAULT_MASK_VALUE.
  _MMA_TMA_WG: _MMA_TMA_WG.
  _SOFTMAX_WG: _SOFTMAX_WG.
  _SCALE_WG: _SCALE_WG.
  _MMA_WARP: _MMA_WARP.
  _TMA_LOAD_QK_WARP: _TMA_LOAD_QK_WARP.
  _TMA_LOAD_V_WARP: _TMA_LOAD_V_WARP.
  _TMA_LOAD_MASK_WARP: _TMA_LOAD_MASK_WARP.
  flash_attention_kernel.kernel.mma_tma_wg.per_warp.tma_load_kv: flash_attention_kernel().kernel().mma_tma_wg().per_warp().tma_load_kv().
  flash_attention_kernel.kernel.softmax_wg.compute_mask.k_range_mask: flash_attention_kernel().kernel().softmax_wg().compute_mask().k_range_mask().
  flash_attention_kernel.kernel.scale_wg.kv_loop.rescale_acc: flash_attention_kernel().kernel().scale_wg().kv_loop().rescale_acc().
  flash_attention_kernel.kernel.scale_wg.kv_loop.no_rescale: flash_attention_kernel().kernel().scale_wg().kv_loop().no_rescale().
  DotPrecisionLike: DotPrecisionLike.
  flash_attention_kernel.kernel.mma_tma_wg.per_warp: flash_attention_kernel().kernel().mma_tma_wg().per_warp().
  flash_attention_kernel.kernel.mma_tma_wg.per_warp.tma_load_kv_warp.prologue: flash_attention_kernel().kernel().mma_tma_wg().per_warp().tma_load_kv_warp().prologue().
  flash_attention_kernel.kernel.mma_tma_wg.per_warp.tma_load_kv_warp.kv_loop: flash_attention_kernel().kernel().mma_tma_wg().per_warp().tma_load_kv_warp().kv_loop().
  flash_attention_kernel.kernel.mma_tma_wg.per_warp.tma_load_kv_warp.kv_loop.tma_loop: flash_attention_kernel().kernel().mma_tma_wg().per_warp().tma_load_kv_warp().kv_loop().tma_loop().
  flash_attention_kernel.kernel.mma_tma_wg.per_warp.tma_load_bias_mask_warp.kv_loop: flash_attention_kernel().kernel().mma_tma_wg().per_warp().tma_load_bias_mask_warp().kv_loop().
  flash_attention_kernel.kernel.mma_tma_wg.per_warp.mma_warp.qk_mma.tma_loop: flash_attention_kernel().kernel().mma_tma_wg().per_warp().mma_warp().qk_mma().tma_loop().
  flash_attention_kernel.kernel.mma_tma_wg.per_warp.mma_warp.pv_mma.tma_loop: flash_attention_kernel().kernel().mma_tma_wg().per_warp().mma_warp().pv_mma().tma_loop().
  flash_attention_kernel.kernel.scale_wg.release_out_scaled_barriers: flash_attention_kernel().kernel().scale_wg().release_out_scaled_barriers().
---
# Module: [`tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py)

## Classes
### `Config`  ·  implements/extends ConfigBase
- def: [`tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py:74`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L74) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_kernel_sm100.md)
- doc: Configuration parameters for Pallas-Mosaic-GPU kernels on SM100 GPUs.
- signature: `class Config(common.ConfigBase):`
- members:
  - `collective` — [`L85`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L85) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_kernel_sm100.md)
  - `num_tma_splits` — [`L84`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L84) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_kernel_sm100.md)
- protocol/private: `__post_init__`[`L87`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L87)
- uses (calls/refs, reference-scoped): [`block_q`](pallas_mosaic_gpu_common.md#ConfigBase.block_q), [`ConfigBase`](pallas_mosaic_gpu_common.md#ConfigBase)
- used by: [`flash_attention_kernel`](pallas_mosaic_gpu_kernel_sm100.md#flash_attention_kernel), [`get_autotuning_configs`](pallas_mosaic_gpu_kernel_sm100.md#get_autotuning_configs), [`get_heuristics_config`](pallas_mosaic_gpu_kernel_sm100.md#get_heuristics_config), [`_estimate_smem_bytes`](pallas_mosaic_gpu_kernel_sm100.md#_estimate_smem_bytes), [`ConfigBase`](pallas_mosaic_gpu_common.md#ConfigBase), [`ConfigSM100`](pallas_mosaic_gpu.md#ConfigSM100)

## Functions
- `_estimate_smem_bytes(config: Config, head_dim: int, head_dim_out: int, q_dtype: jax.typing.DTypeLike, k_dtype: jax.typing.DTypeLike, v_dtype: jax.typing.DTypeLike, out_dtype: jax.typing.DTypeLike, bias: jax.ShapeDtypeStruct | jax.Array | None, mask: jax.ShapeDtypeStruct | jax.Array | None)` — [`L147`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L147)
- `compute_mask(ki, do_causal)` — [`L591`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L591)
- `flash_attention_kernel(q: Float[Array, "T H D"], k: Float[Array, "t h D"], v: Float[Array, "t h d"], bias: Float[Array, "#H #T #t"] | None, mask: Bool[Array, "#H #T #t"] | None, k_start: Int[Array, "#H #T"] | None, k_end: Int[Array, "#H #T"] | None, *, is_causal: bool, logits_soft_cap: float | None, logits_scale: float, out_dtype: jnp.dtype, normalize_output: bool, return_residuals: bool, use_stable_softmax: bool, rescale_threshold: float, config: Config)` — [`L269`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L269) — SM100 Pallas Mosaic GPU Flash Attention. — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_kernel_sm100.md)
- `get_autotuning_configs(ba: op.BoundArguments)` — [`L212`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L212) — Returns a set of configs for autotuning flash attention on SM100 GPUs.
- `get_heuristics_config(ba: op.BoundArguments)` — [`L97`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L97) — Returns a heuristic configuration for flash attention on SM100 GPUs. — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_kernel_sm100.md)
- `k_range_mask(mask)` — [`L617`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L617)
- `kernel(q_gmem, k_gmem, v_gmem, bias_gmem, mask_gmem, k_start_gmem, k_end_gmem, k_start_minmax_gmems, k_end_minmax_gmems, o_gmem, *residual_gmems, qo_smem_union, k_smem, v_smem, mask_smem, alpha_smem, li_smem, qk_acc_tmem, p_tmem, acc_tmem, bias_smem, q_barrier, k_barrier, v_barrier, bias_barrier, bias_consumed_barrier, mask_barrier, mask_consumed_barrier, qk_mma_barrier, k_consumed_barrier, qk_consumed_barrier, pv_mma_barrier, v_consumed_barrier, p_produced_barrier, out_scaled_barrier)` — [`L348`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L348)
- `kv_loop(ki)` — [`L458`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L458)
- `kv_loop(ki)` — [`L491`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L491)
- `kv_loop(ki)` — [`L564`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L564)
- `kv_loop(ki, carry, *, do_causal=False)` — [`L630`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L630)
- `kv_loop(ki)` — [`L734`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L734)
- `load_acc_tiles()` — [`L740`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L740)
- `load_acc_tiles()` — [`L810`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L810)
- `load_k_bound(k_range_ref)` — [`L402`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L402)
- `maybe_cluster_barrier(**kwargs)` — [`L910`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L910)
- `mma_tma_wg()` — [`L422`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L422)
- `mma_warp()` — [`L511`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L511)
- `needs_k_range_mask(ki)` — [`L583`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L583)
- `no_rescale()` — [`L783`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L783)
- `pad_reduce(pad_value: int)` — [`L849`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L849)
- `per_warp()` — [`L426`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L426)
- `pre_reduce_k_range_per_qtile(range_ref)` — [`L845`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L845)
- `prologue(ki)` — [`L454`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L454)
- `pv_mma(ki)` — [`L535`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L535)
- `qk_mma(ki)` — [`L513`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L513)
- `release_out_scaled_barriers()` — [`L725`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L725)
- `rescale_acc(ds=ds, acc=acc)` — [`L766`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L766)
- `scale_wg()` — [`L721`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L721)
- `softmax_wg()` — [`L571`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L571)
- `tma_load_bias_mask_warp()` — [`L489`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L489)
- `tma_load_kv(gmem, smem, barrier, partition_axis, ki, split_idx)` — [`L429`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L429)
- `tma_load_kv_warp(gmem, smem, barrier, consumed_barrier, partition_axis)` — [`L446`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L446)
- `tma_load_qk_warp()` — [`L468`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L468)
- `tma_load_v_warp()` — [`L483`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L483)
- `tma_loop(split_idx)` — [`L462`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L462)
- `tma_loop(split_idx)` — [`L520`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L520)
- `tma_loop(split_idx)` — [`L543`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L543)
- `tmem(shape, dtype=jnp.float32, **kwargs)` — [`L907`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L907)
- `two_in_flight(iterable)` — [`L729`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L729)
- `warp_any(_, needs_rescale)` — [`L758`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L758)
- `write_acc()` — [`L800`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L800)
- `write_alpha_to_smem()` — [`L671`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L671)
- `write_zeros()` — [`L826`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L826)

## Module values
- `DotPrecisionLike` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L41)
- `PagingInfo` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L42)
- `QArray` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L43)
- `Residuals` — [`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L44)
- `_ALPHA_BARRIER_OFFSET` — [`L63`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L63)
- `_DEFAULT_MASK_VALUE` — [`L53`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L53)
- `_L_BARRIER_ID` — [`L62`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L62)
- `_MASK_PACKED_BITS` — [`L68`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L68)
- `_MASK_PACK_DTYPE` — [`L67`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L67)
- `_MMA_TMA_WG` — [`L55`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L55)
- `_MMA_WARP` — [`L58`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L58)
- `_SCALE_WG` — [`L57`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L57)
- `_SOFTMAX_WG` — [`L56`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L56)
- `_TCGEN05` — [`L50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L50)
- `_TCGEN05_ROW` — [`L51`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L51)
- `_TMA_LOAD_MASK_WARP` — [`L61`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L61)
- `_TMA_LOAD_QK_WARP` — [`L59`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L59)
- `_TMA_LOAD_V_WARP` — [`L60`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L60)
- `_TMEM` — [`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L47)
- `_TMEM_COL` — [`L48`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L48)
- `_TMEM_ROW` — [`L49`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L49)
- `_WG_SPLAT` — [`L52`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L52)
- `_load_bcast` — [`L65`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L65)

