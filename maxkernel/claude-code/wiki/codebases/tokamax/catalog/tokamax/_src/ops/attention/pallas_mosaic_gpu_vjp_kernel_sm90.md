---
title: 'Module: tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.pallas_mosaic_gpu_vjp_kernel_sm90`/
symbols:
  flash_attention_vjp_kernel: flash_attention_vjp_kernel().
  get_heuristics_config: get_heuristics_config().
  get_autotuning_configs: get_autotuning_configs().
  flash_attention_vjp_kernel.kernel_dkv.q_pipeline: flash_attention_vjp_kernel().kernel_dkv().q_pipeline().
  flash_attention_vjp_kernel.kernel_dq.kv_pipeline: flash_attention_vjp_kernel().kernel_dq().kv_pipeline().
  flash_attention_vjp_kernel.kernel_dq: flash_attention_vjp_kernel().kernel_dq().
  flash_attention_vjp_kernel.kernel_dkv: flash_attention_vjp_kernel().kernel_dkv().
  flash_attention_vjp_kernel.kernel_dkv.q_pipeline.compute_sT: flash_attention_vjp_kernel().kernel_dkv().q_pipeline().compute_sT().
  Config: Config.
  flash_attention_vjp_kernel.kernel_dq.compute_thread: flash_attention_vjp_kernel().kernel_dq().compute_thread().
  _tiled_spec: _tiled_spec.
  _WGMMA: _WGMMA.
  _load_bcast: _load_bcast.
  flash_attention_vjp_kernel.kernel_dq.kv_pipeline.iota: flash_attention_vjp_kernel().kernel_dq().kv_pipeline().iota().
  flash_attention_vjp_kernel.kernel_dkv.q_pipeline.iota: flash_attention_vjp_kernel().kernel_dkv().q_pipeline().iota().
  _estimate_dq_smem_bytes: _estimate_dq_smem_bytes().
  _estimate_dkv_smem_bytes: _estimate_dkv_smem_bytes().
  flash_attention_vjp_kernel.kernel_dq.kv_pipeline.compute_s: flash_attention_vjp_kernel().kernel_dq().kv_pipeline().compute_s().
  flash_attention_vjp_kernel.kernel_dkv.compute_thread: flash_attention_vjp_kernel().kernel_dkv().compute_thread().
  _WGMMA_COL: _WGMMA_COL.
  _WGMMA_ROW: _WGMMA_ROW.
  flash_attention_vjp_kernel.kernel_dq.compute_thread.load_k_range: flash_attention_vjp_kernel().kernel_dq().compute_thread().load_k_range().
  flash_attention_vjp_kernel.kernel_dkv.q_pipeline.load_k_range: flash_attention_vjp_kernel().kernel_dkv().q_pipeline().load_k_range().
  flash_attention_vjp_kernel.tiled_wgs_smem: flash_attention_vjp_kernel().tiled_wgs_smem().
  Residuals: Residuals.
  flash_attention_vjp_kernel.kernel_dq.kv_pipeline.apply_causal_mask: flash_attention_vjp_kernel().kernel_dq().kv_pipeline().apply_causal_mask().
  flash_attention_vjp_kernel.kernel_dkv.q_pipeline.apply_causal_mask: flash_attention_vjp_kernel().kernel_dkv().q_pipeline().apply_causal_mask().
  _SMEM_SIZE_BYTES: _SMEM_SIZE_BYTES.
  _F32_BYTES: _F32_BYTES.
  _WGMMA_TRANSPOSED: _WGMMA_TRANSPOSED.
  flash_attention_vjp_kernel.kernel_dq.compute_thread.compute_dq: flash_attention_vjp_kernel().kernel_dq().compute_thread().compute_dq().
  flash_attention_vjp_kernel.kernel_dq.kv_pipeline.compute_dp: flash_attention_vjp_kernel().kernel_dq().kv_pipeline().compute_dp().
  flash_attention_vjp_kernel.kernel_dkv.compute_thread.compute_dk_dv: flash_attention_vjp_kernel().kernel_dkv().compute_thread().compute_dk_dv().
  flash_attention_vjp_kernel.kernel_dkv.q_pipeline.compute_dpT: flash_attention_vjp_kernel().kernel_dkv().q_pipeline().compute_dpT().
  flash_attention_vjp_kernel.kernel_dq.compute_thread.zero_ds: flash_attention_vjp_kernel().kernel_dq().compute_thread().zero_ds().
---
# Module: [`tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py)

## Functions
- `_estimate_dkv_smem_bytes(ba, block_q, block_kv, num_stages)` — [`L79`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L79) — Estimates the dkv kernel smem usage in bytes for a given configuration.
- `_estimate_dq_smem_bytes(ba, block_q, block_kv, num_stages)` — [`L49`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L49) — Estimates the dq kernel smem usage in bytes for a given configuration.
- `apply_causal_mask()` — [`L351`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L351)
- `apply_causal_mask()` — [`L587`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L587)
- `compute_dk_dv(dk_acc, dv_acc)` — [`L503`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L503)
- `compute_dp(acc)` — [`L381`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L381)
- `compute_dpT(acc)` — [`L624`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L624)
- `compute_dq(dq_acc)` — [`L276`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L276)
- `compute_s(acc)` — [`L315`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L315)
- `compute_sT(acc)` — [`L545`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L545)
- `compute_thread(pipeline_callback)` — [`L254`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L254)
- `compute_thread(pipeline_callback)` — [`L493`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L493)
- `flash_attention_vjp_kernel(q: Float[Array, "T H D"], k: Float[Array, "t h D"], v: Float[Array, "t h d"], residuals: Residuals, out: Float[Array, "T H d"], dout: Float[Array, "T H d"], bias: Float[Array, "#H #T #t"] | None, mask: Bool[Array, "#H #T #t"] | None, k_start: Int[Array, "#H #T"] | None, k_end: Int[Array, "#H #T"] | None, *, logits_scale: float, logits_soft_cap: float | None, is_causal: bool, ds_dtype: jax.typing.DTypeLike | None, config: Config)` — [`L146`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L146)
- `get_autotuning_configs(ba: op.BoundArguments)` — [`L124`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L124)
- `get_heuristics_config(ba: op.BoundArguments)` — [`L108`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L108) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm90](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm90.md)
- `iota(d)` — [`L346`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L346)
- `iota(d)` — [`L582`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L582)
- `kernel_dkv(q_gmem, k_gmem, v_gmem, dout_gmem, m_gmem, l_gmem, delta_gmem, bias_gmem, mask_gmem, k_start_gmem, k_end_gmem, dk_gmem, dv_gmem, k_smems, v_smems, barriers, block_q: int, block_kv: int)` — [`L446`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L446)
- `kernel_dq(q_gmem, k_gmem, v_gmem, dout_gmem, m_gmem, l_gmem, delta_gmem, bias_gmem, mask_gmem, k_start_gmem, k_end_gmem, dq_gmem, ds_gmem, q_smems, dout_smems, barriers, block_q: int, block_kv: int)` — [`L209`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L209)
- `kv_pipeline(index, bias_smem, mask_smem, v_smem, k_smem, bias_consumed_barrier, mask_consumed_barrier, v_consumed_barrier, k_consumed_barrier, carry)` — [`L297`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L297)
- `load_k_range(ref)` — [`L264`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L264)
- `load_k_range(ref)` — [`L594`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L594)
- `q_pipeline(index, bias_smem, m_smem, l_smem, mask_smem, dout_smem, delta_smem, q_smem, bias_consumed_barrier, m_consumed_barrier, l_consumed_barrier, mask_consumed_barrier, dout_consumed_barrier, delta_consumed_barrier, q_consumed_barrier, carry)` — [`L521`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L521)
- `tiled_wgs_smem(shape, dtype, what="")` — [`L691`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L691)
- `zero_ds(ki)` — [`L288`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L288)

## Module values
- `Config` — [`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L36)
- `Residuals` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L37)
- `_F32_BYTES` — [`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L44)
- `_SMEM_SIZE_BYTES` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L43)
- `_WGMMA` — [`L39`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L39)
- `_WGMMA_COL` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L40)
- `_WGMMA_ROW` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L41)
- `_WGMMA_TRANSPOSED` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L42)
- `_load_bcast` — [`L45`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L45)
- `_tiled_spec` — [`L46`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py#L46)

