---
title: 'Module: tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.pallas_mosaic_gpu_vjp_kernel_sm100`/
symbols:
  get_autotuning_configs: get_autotuning_configs().
  flash_attention_vjp_kernel: flash_attention_vjp_kernel().
  _kernel_dq.sfu_wg.sfu_loop: _kernel_dq().sfu_wg().sfu_loop().
  _get_dq_scratch_shapes: _get_dq_scratch_shapes().
  _get_dkv_scratch_shapes: _get_dkv_scratch_shapes().
  _TMEM: _TMEM.
  Config: Config#
  Config.chunk_size: Config#chunk_size.
  _pad: _pad().
  get_autotuning_configs._get: get_autotuning_configs()._get().
  _smem_transforms: _smem_transforms().
  get_heuristics_config._score: get_heuristics_config()._score().
  _kernel_dq: _kernel_dq().
  _load_bcast_smem: _load_bcast_smem().
  get_heuristics_config: get_heuristics_config().
  _TMEM_ROW: _TMEM_ROW.
  _kernel_dkv.sfu_wg.sfu_loop: _kernel_dkv().sfu_wg().sfu_loop().
  Config.eltwise_stages: Config#eltwise_stages.
  _kernel_dq.mma_tma_wg.per_warp.tma_kv.kv_loop: _kernel_dq().mma_tma_wg().per_warp().tma_kv().kv_loop().
  _kernel_dq.mma_tma_wg.per_warp.mma.mma_loop: _kernel_dq().mma_tma_wg().per_warp().mma().mma_loop().
  _TMEM_COL: _TMEM_COL.
  _add_eltwise_scratch_shapes: _add_eltwise_scratch_shapes().
  _kernel: _kernel().
  _kernel.wrapped: _kernel().wrapped().
  Config.double_buffer: Config#double_buffer.
  Config.residual_stages: Config#residual_stages.
  Config.load_residuals_in_regs: Config#load_residuals_in_regs.
  _kernel._wrapped_vmap: _kernel()._wrapped_vmap().
  _kernel_dq.mma_tma_wg.per_warp.tma_q: _kernel_dq().mma_tma_wg().per_warp().tma_q().
  _kernel_dq.mma_tma_wg.per_warp.tma_kv.prologue: _kernel_dq().mma_tma_wg().per_warp().tma_kv().prologue().
  _kernel_dq.mma_tma_wg.per_warp.tma_eltwise.eltwise_loop: _kernel_dq().mma_tma_wg().per_warp().tma_eltwise().eltwise_loop().
  _kernel_dq.sfu_wg: _kernel_dq().sfu_wg().
  _kernel_dq.sfu_wg.sfu_loop.iota: _kernel_dq().sfu_wg().sfu_loop().iota().
  _kernel_dkv.sfu_wg.sfu_loop.iota: _kernel_dkv().sfu_wg().sfu_loop().iota().
  get_autotuning_configs._downcast_if_needed: get_autotuning_configs()._downcast_if_needed().
  flash_attention_vjp_kernel._reshape_4d: flash_attention_vjp_kernel()._reshape_4d().
  _kernel.wrapped.stateful: _kernel().wrapped().stateful().
  _kernel._wrapped_vmap.loop_body: _kernel()._wrapped_vmap().loop_body().
  _kernel.unwrap: _kernel().unwrap().
  flash_attention_vjp_kernel.dq_body: flash_attention_vjp_kernel().dq_body().
  flash_attention_vjp_kernel.dq_body_bias: flash_attention_vjp_kernel().dq_body_bias().
  flash_attention_vjp_kernel.dkv_body: flash_attention_vjp_kernel().dkv_body().
  _SMEM_SIZE_LIMIT: _SMEM_SIZE_LIMIT.
  _kernel_dq.mma_tma_wg: _kernel_dq().mma_tma_wg().
  _kernel_dq.mma_tma_wg.per_warp.tma_kv.kv_epilogue: _kernel_dq().mma_tma_wg().per_warp().tma_kv().kv_epilogue().
  _kernel_dq.mma_tma_wg.per_warp.tma_eltwise.eltwise_loop.wait_bias: _kernel_dq().mma_tma_wg().per_warp().tma_eltwise().eltwise_loop().wait_bias().
  _kernel_dq.mma_tma_wg.per_warp.tma_eltwise.eltwise_loop.wait_mask: _kernel_dq().mma_tma_wg().per_warp().tma_eltwise().eltwise_loop().wait_mask().
  _kernel_dq.mma_tma_wg.per_warp.tma_eltwise.eltwise_epilogue: _kernel_dq().mma_tma_wg().per_warp().tma_eltwise().eltwise_epilogue().
  _kernel_dkv.mma_tma_wg: _kernel_dkv().mma_tma_wg().
  _kernel.wrapped.stateful._body: _kernel().wrapped().stateful()._body().
  _get_input_metadata: _get_input_metadata().
  _estimate_smem_bytes: _estimate_smem_bytes().
  flash_attention_vjp_kernel._squeeze_trailing_1s: flash_attention_vjp_kernel()._squeeze_trailing_1s().
  _kernel_dkv: _kernel_dkv().
  _kernel_dq.mma_tma_wg.per_warp: _kernel_dq().mma_tma_wg().per_warp().
  _kernel_dq.mma_tma_wg.per_warp.tma_kv: _kernel_dq().mma_tma_wg().per_warp().tma_kv().
  _kernel_dq.mma_tma_wg.per_warp.tma_eltwise: _kernel_dq().mma_tma_wg().per_warp().tma_eltwise().
  _kernel_dq.mma_tma_wg.per_warp.mma: _kernel_dq().mma_tma_wg().per_warp().mma().
  _kernel_dq.sfu_wg.ds_prologue: _kernel_dq().sfu_wg().ds_prologue().
  _kernel_dq.sfu_wg.ds_cleanup: _kernel_dq().sfu_wg().ds_cleanup().
  _kernel_dkv.mma_tma_wg.per_warp: _kernel_dkv().mma_tma_wg().per_warp().
  _kernel_dkv.mma_tma_wg.per_warp.tma_kv: _kernel_dkv().mma_tma_wg().per_warp().tma_kv().
  _kernel_dkv.mma_tma_wg.per_warp.tma_kv.residual_loop: _kernel_dkv().mma_tma_wg().per_warp().tma_kv().residual_loop().
  _kernel_dkv.mma_tma_wg.per_warp.tma_kv.residual_loop.wait_res: _kernel_dkv().mma_tma_wg().per_warp().tma_kv().residual_loop().wait_res().
  _kernel_dkv.mma_tma_wg.per_warp.tma_kv.residual_epilogue: _kernel_dkv().mma_tma_wg().per_warp().tma_kv().residual_epilogue().
  _kernel_dkv.mma_tma_wg.per_warp.tma_q: _kernel_dkv().mma_tma_wg().per_warp().tma_q().
  _kernel_dkv.mma_tma_wg.per_warp.tma_q.q_loop: _kernel_dkv().mma_tma_wg().per_warp().tma_q().q_loop().
  _kernel_dkv.mma_tma_wg.per_warp.tma_q.q_loop.wait_q: _kernel_dkv().mma_tma_wg().per_warp().tma_q().q_loop().wait_q().
  _kernel_dkv.mma_tma_wg.per_warp.tma_q.q_epilogue: _kernel_dkv().mma_tma_wg().per_warp().tma_q().q_epilogue().
  _kernel_dkv.mma_tma_wg.per_warp.tma_eltwise: _kernel_dkv().mma_tma_wg().per_warp().tma_eltwise().
  _kernel_dkv.mma_tma_wg.per_warp.tma_eltwise.eltwise_loop: _kernel_dkv().mma_tma_wg().per_warp().tma_eltwise().eltwise_loop().
  _kernel_dkv.mma_tma_wg.per_warp.tma_eltwise.eltwise_loop.wait_bias: _kernel_dkv().mma_tma_wg().per_warp().tma_eltwise().eltwise_loop().wait_bias().
  _kernel_dkv.mma_tma_wg.per_warp.tma_eltwise.eltwise_loop.wait_mask: _kernel_dkv().mma_tma_wg().per_warp().tma_eltwise().eltwise_loop().wait_mask().
  _kernel_dkv.mma_tma_wg.per_warp.tma_eltwise.eltwise_epilogue: _kernel_dkv().mma_tma_wg().per_warp().tma_eltwise().eltwise_epilogue().
  _kernel_dkv.mma_tma_wg.per_warp.mma: _kernel_dkv().mma_tma_wg().per_warp().mma().
  _kernel_dkv.mma_tma_wg.per_warp.mma.mma_loop: _kernel_dkv().mma_tma_wg().per_warp().mma().mma_loop().
  _kernel_dkv.sfu_wg: _kernel_dkv().sfu_wg().
  _kernel_dkv.sfu_wg.ds_prologue: _kernel_dkv().sfu_wg().ds_prologue().
  _kernel_dkv.sfu_wg.ds_cleanup: _kernel_dkv().sfu_wg().ds_cleanup().
---
# Module: [`tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py)

## Classes
### `Config`  ·  implements/extends Config
- def: [`tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py:43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L43) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100.md)
- doc: Configuration for the VJP.
- signature: `class Config(vjp_common.Config):`
- members:
  - `chunk_size` — [`L59`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L59) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100.md)
  - `double_buffer` — [`L57`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L57) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100.md)
  - `eltwise_stages` — [`L56`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L56) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100.md)
  - `load_residuals_in_regs` — [`L60`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L60) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100.md)
  - `residual_stages` — [`L58`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L58) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100.md)
- uses (calls/refs, reference-scoped): [`Config`](pallas_mosaic_gpu_vjp_common.md#Config)
- used by: [`get_autotuning_configs`](pallas_mosaic_gpu_vjp_kernel_sm100.md#get_autotuning_configs), [`_fwd`](pallas_mosaic_gpu_vjp.md#PallasMosaicGpuFlashAttentionVjp._fwd), [`flash_attention_vjp_kernel`](pallas_mosaic_gpu_vjp_kernel_sm100.md#flash_attention_vjp_kernel), [`sfu_loop`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq.sfu_wg.sfu_loop), [`_get_dkv_scratch_shapes`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_get_dkv_scratch_shapes), [`_get_dq_scratch_shapes`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_get_dq_scratch_shapes), [`_score`](pallas_mosaic_gpu_vjp_kernel_sm100.md#get_heuristics_config._score), [`_get_autotuning_configs`](pallas_mosaic_gpu_vjp.md#PallasMosaicGpuFlashAttentionVjp._get_autotuning_configs), [`_get_heuristics_config`](pallas_mosaic_gpu_vjp.md#PallasMosaicGpuFlashAttentionVjp._get_heuristics_config), [`_kernel_dq`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq), [`get_heuristics_config`](pallas_mosaic_gpu_vjp_kernel_sm100.md#get_heuristics_config), [`kv_loop`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq.mma_tma_wg.per_warp.tma_kv.kv_loop), [`mma_loop`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq.mma_tma_wg.per_warp.mma.mma_loop), [`Config`](pallas_mosaic_gpu_vjp_common.md#Config), [`config_cls`](pallas_mosaic_gpu_vjp.md#PallasMosaicGpuFlashAttentionVjp.config_cls), [`eltwise_loop`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq.mma_tma_wg.per_warp.tma_eltwise.eltwise_loop), [`sfu_wg`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq.sfu_wg), [`tma_q`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq.mma_tma_wg.per_warp.tma_q), [`eltwise_epilogue`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq.mma_tma_wg.per_warp.tma_eltwise.eltwise_epilogue), [`wait_bias`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq.mma_tma_wg.per_warp.tma_eltwise.eltwise_loop.wait_bias), [`wait_mask`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq.mma_tma_wg.per_warp.tma_eltwise.eltwise_loop.wait_mask)

## Functions
- `_add_eltwise_scratch_shapes(name, shape, dtype, config, block_q, block_kv, swizzle_limit)` — [`L595`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L595)
- `_body(*args, **kwargs)` — [`L431`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L431)
- `_downcast_if_needed(dtype, prec)` — [`L294`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L294)
- `_estimate_smem_bytes(scratch_shapes: dict)` — [`L566`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L566) — Estimates the total SMEM usage in bytes for a given scratch shapes dict.
- `_get(name, pos)` — [`L280`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L280)
- `_get_dkv_scratch_shapes(config: Config, head_dim: int, head_dim_out: int, chunk_size: int, q_dtype, dout_dtype, k_dtype, v_dtype, ds_dtype, bias_shape, bias_dtype, mask_shape, mask_dtype)` — [`L173`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L173)
- `_get_dq_scratch_shapes(config: Config, head_dim: int, head_dim_out: int, chunk_size: int, q_dtype, dout_dtype, k_dtype, v_dtype, ds_dtype, bias_shape, bias_dtype, mask_shape, mask_dtype, orig_ds_dtype=None)` — [`L63`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L63)
- `_get_input_metadata(q, v)` — [`L552`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L552) — Normalizes and returns head dimensions and datatypes.
- `_kernel(body, out_type, **kernel_kwargs)` — [`L408`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L408) — Interface for SM100 attention VJP kernel.
- `_kernel_dkv(q_ref, k_ref, v_ref, dout_ref, m_ref, l_ref, delta_ref, bias_ref, k_start_ref, k_end_ref, mask_ref, dk_ref, dv_ref, bias_4d_shape=None, mask_4d_shape=None, *, k_smem=None, v_smem=None, q_smem=None, do_smem=None, residuals_smem=None, ds_smem=None, p_smem=None, s_tmem=None, dp_tmem=None, dk_tmem=None, dv_tmem=None, kv_produced, q_do_produced, q_do_consumed, residual_produced, residual_consumed, s_produced, s_consumed, dp_produced, dp_consumed, ds_produced, ds_consumed, kv_mma_finished, bias_smem=None, mask_smem=None, bias_produced=None, bias_consumed=None, mask_produced=None, mask_consumed=None, config, is_causal, logits_scale, logits_soft_cap)` — [`L1199`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1199) — Computes dkv. — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100.md)
- `_kernel_dq(q_ref, k_ref, v_ref, dout_ref, m_ref, l_ref, delta_ref, bias_ref, k_start_ref, k_end_ref, mask_ref, dq_ref, ds_ref=None, bias_4d_shape=None, mask_4d_shape=None, *, q_smem=None, do_smem=None, residuals_smem=None, k_smem=None, v_smem=None, ds_smem=None, s_tmem=None, dp_tmem=None, dq_tmem=None, q_do_produced, k_produced, v_produced, k_consumed, v_consumed, s_produced, s_consumed, dp_produced, dp_consumed, ds_produced, ds_consumed, dq_mma_finished, bias_smem=None, mask_smem=None, bias_produced=None, bias_consumed=None, mask_produced=None, mask_consumed=None, config: Config, is_causal, logits_scale, logits_soft_cap, orig_ds_dtype=None, reduce_ds: bool = False)` — [`L621`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L621) — Computes dq. — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100.md)
- `_load_bcast_smem(ref, smem, b, hi, elt_bi, s_shape, chunk_slice, is_dq, ref_4d_shape)` — [`L519`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L519)
- `_pad(x, axis, block, constant_values=0, broadcastable=False)` — [`L484`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L484) — Pads an array to a multiple of `block` along `axis`.
- `_reshape_4d(arr, core_ndim)` — [`L1732`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1732)
- `_score(c: Config)` — [`L402`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L402)
- `_smem_transforms(dtype, swizzle=128)` — [`L512`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L512)
- `_squeeze_trailing_1s(arr)` — [`L1737`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1737)
- `_wrapped_vmap(axis_size, in_batched, *kernel_args)` — [`L452`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L452)
- `dkv_body(q_ref, k_ref, v_ref, dout_ref, m_ref, l_ref, delta_ref, bias_ref, k_start_ref, k_end_ref, mask_ref, dk_ref, dv_ref, **scratches)` — [`L2009`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L2009)
- `dq_body(q_ref, k_ref, v_ref, dout_ref, m_ref, l_ref, delta_ref, bias_ref, k_start_ref, k_end_ref, mask_ref, dq_ref, **scratches)` — [`L1841`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1841)
- `dq_body_bias(q_ref, k_ref, v_ref, dout_ref, m_ref, l_ref, delta_ref, bias_ref, k_start_ref, k_end_ref, mask_ref, dq_ref, ds_ref, **scratches)` — [`L1914`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1914)
- `ds_cleanup(i)` — [`L1195`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1195)
- `ds_cleanup(i)` — [`L1693`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1693)
- `ds_prologue(i)` — [`L915`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L915)
- `ds_prologue(i)` — [`L1495`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1495)
- `eltwise_epilogue(ki)` — [`L843`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L843)
- `eltwise_epilogue(step)` — [`L1420`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1420)
- `eltwise_loop(ki)` — [`L796`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L796)
- `eltwise_loop(step)` — [`L1370`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1370)
- `flash_attention_vjp_kernel(q: Float[Array, "*B T H D"], k: Float[Array, "*B t h D"], v: Float[Array, "*B t h d"], residuals: base.Residuals, out: Float[Array, "*B T H d"], dout: Float[Array, "*B T H d"], bias: Float[Array, "*#B #H #T #t"] | None, mask: Bool[Array, "*#B #H #T #t"] | None, k_start: Int[Array, "*#B #H #T"] | None, k_end: Int[Array, "*#B #H #T"] | None, *, logits_scale: float, logits_soft_cap: float | None, is_causal: bool, ds_dtype: jax.typing.DTypeLike | None, config: Config)` — [`L1698`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1698) — SM100 Pallas Mosaic GPU Flash Attention VJP.
- `get_autotuning_configs(ba: op.BoundArguments)` — [`L277`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L277)
- `get_heuristics_config(ba: op.BoundArguments)` — [`L396`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L396) — Returns a heuristic configuration for flash attention VJP on SM100 GPUs.
- `iota(shape, d)` — [`L1041`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1041)
- `iota(shape, d)` — [`L1569`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1569)
- `kv_epilogue(ki)` — [`L786`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L786)
- `kv_loop(ki)` — [`L763`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L763)
- `loop_body(arrays_tuple)` — [`L468`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L468)
- `mma()` — [`L851`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L851)
- `mma()` — [`L1428`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1428)
- `mma_loop(ki)` — [`L855`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L855)
- `mma_loop(step)` — [`L1432`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1432)
- `mma_tma_wg()` — [`L699`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L699)
- `mma_tma_wg()` — [`L1278`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1278)
- `per_warp()` — [`L703`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L703)
- `per_warp()` — [`L1282`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1282)
- `prologue(ki)` — [`L744`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L744)
- `q_epilogue(step)` — [`L1361`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1361)
- `q_loop(step)` — [`L1336`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1336)
- `residual_epilogue(step)` — [`L1329`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1329)
- `residual_loop(step)` — [`L1299`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1299)
- `sfu_loop(ki)` — [`L959`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L959)
- `sfu_loop(step)` — [`L1502`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1502)
- `sfu_wg()` — [`L910`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L910)
- `sfu_wg()` — [`L1491`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1491)
- `stateful(out_refs)` — [`L430`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L430)
- `tma_eltwise()` — [`L792`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L792)
- `tma_eltwise()` — [`L1366`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1366)
- `tma_kv()` — [`L742`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L742)
- `tma_kv()` — [`L1286`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1286)
- `tma_q()` — [`L707`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L707)
- `tma_q()` — [`L1334`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1334)
- `unwrap(*args)` — [`L477`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L477)
- `wait_bias()` — [`L814`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L814)
- `wait_bias()` — [`L1391`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1391)
- `wait_mask()` — [`L835`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L835)
- `wait_mask()` — [`L1412`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1412)
- `wait_q()` — [`L1344`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1344)
- `wait_res()` — [`L1307`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L1307)
- `wrapped(*kernel_args)` — [`L428`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L428)

## Module values
- `_SMEM_SIZE_LIMIT` — [`L34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L34)
- `_TMEM` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L37)
- `_TMEM_COL` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L38)
- `_TMEM_ROW` — [`L39`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm100.py#L39)

