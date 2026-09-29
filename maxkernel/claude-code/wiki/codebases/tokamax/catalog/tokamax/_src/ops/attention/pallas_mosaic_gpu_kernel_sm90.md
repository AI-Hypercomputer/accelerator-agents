---
title: 'Module: tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.pallas_mosaic_gpu_kernel_sm90`/
symbols:
  flash_attention_kernel.kernel.compute_wg.loop_body: flash_attention_kernel().kernel().compute_wg().loop_body().
  flash_attention_kernel: flash_attention_kernel().
  get_heuristics_config: get_heuristics_config().
  get_autotuning_configs: get_autotuning_configs().
  flash_attention_kernel.kernel.compute_wg: flash_attention_kernel().kernel().compute_wg().
  flash_attention_kernel.kernel.memory_wg.cp: flash_attention_kernel().kernel().memory_wg().cp().
  Config: Config#
  flash_attention_kernel.kernel.compute_wg.loop_body.compute_qk: flash_attention_kernel().kernel().compute_wg().loop_body().compute_qk().
  _load_bcast: _load_bcast.
  flash_attention_kernel.kernel.compute_wg.loop_body.iota: flash_attention_kernel().kernel().compute_wg().loop_body().iota().
  _estimate_shared_mem_usage_bytes: _estimate_shared_mem_usage_bytes().
  flash_attention_kernel.kernel.memory_wg: flash_attention_kernel().kernel().memory_wg().
  flash_attention_kernel.kernel.memory_wg.prologue: flash_attention_kernel().kernel().memory_wg().prologue().
  flash_attention_kernel.kernel.memory_wg.kv_loop: flash_attention_kernel().kernel().memory_wg().kv_loop().
  flash_attention_kernel.kernel.get_kv_ranges: flash_attention_kernel().kernel().get_kv_ranges().
  _WGMMA: _WGMMA.
  Residuals: Residuals.
  flash_attention_kernel.kernel.compute_wg.loop_body.apply_k_start: flash_attention_kernel().kernel().compute_wg().loop_body().apply_k_start().
  flash_attention_kernel.kernel.compute_wg.loop_body.apply_k_end: flash_attention_kernel().kernel().compute_wg().loop_body().apply_k_end().
  _WGMMA_ROW: _WGMMA_ROW.
  _WGMMA_COL: _WGMMA_COL.
  flash_attention_kernel.kernel: flash_attention_kernel().kernel().
  flash_attention_kernel.kernel.compute_wg.loop_body.compute_pv: flash_attention_kernel().kernel().compute_wg().loop_body().compute_pv().
---
# Module: [`tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py)

## Classes
### `Config`  ·  implements/extends ConfigBase
- def: [`tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py:49`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L49)
- doc: Configuration parameters for Pallas-Mosaic-GPU kernels on SM90 GPUs.
- signature: `class Config(common.ConfigBase):`
- uses (calls/refs, reference-scoped): [`ConfigBase`](pallas_mosaic_gpu_common.md#ConfigBase)
- used by: [`flash_attention_kernel`](pallas_mosaic_gpu_kernel_sm90.md#flash_attention_kernel), [`get_autotuning_configs`](pallas_mosaic_gpu_kernel_sm90.md#get_autotuning_configs), [`get_heuristics_config`](pallas_mosaic_gpu_kernel_sm90.md#get_heuristics_config), [`ConfigBase`](pallas_mosaic_gpu_common.md#ConfigBase), [`ConfigSM90`](pallas_mosaic_gpu.md#ConfigSM90)

## Functions
- `_estimate_shared_mem_usage_bytes(ba, block_q, block_kv, num_stages)` — [`L54`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L54) — Estimates the shared memory usage in bytes for a given configuration.
- `apply_k_end(k_end=k_end)` — [`L323`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L323)
- `apply_k_start(k_start=k_start)` — [`L312`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L312)
- `compute_pv(refs)` — [`L372`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L372)
- `compute_qk(acc)` — [`L277`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L277)
- `compute_wg()` — [`L234`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L234)
- `cp(gmem, smems, barriers, si)` — [`L458`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L458)
- `flash_attention_kernel(q: Float[Array, "T H D"], k: Float[Array, "t h D"], v: Float[Array, "t h d"], bias: Float[Array, "#H #T #t"] | None, mask: Bool[Array, "#H #T #t"] | None, k_start: Int[Array, "#H #T"] | None, k_end: Int[Array, "#H #T"] | None, *, is_causal: bool, logits_soft_cap: float | None, logits_scale: float, out_dtype: jnp.dtype, normalize_output: bool, return_residuals: bool, use_stable_softmax: bool, rescale_threshold: float, config: Config)` — [`L120`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L120) — Flash attention with Mosaic GPU.
- `get_autotuning_configs(ba: op.BoundArguments)` — [`L96`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L96) — Returns a set of configs for autotuning flash attention on SM90 GPUs.
- `get_heuristics_config(ba: op.BoundArguments)` — [`L88`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L88)
- `get_kv_ranges()` — [`L209`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L209)
- `iota(d)` — [`L274`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L274)
- `kernel(q_gmem, k_gmem, v_gmem, bias_gmem, mask_gmem, k_start_gmem, k_end_gmem, k_start_minmax_gmems, k_end_minmax_gmems, o_gmem, *residual_gmems, qko_smem_union, v_smem, bias_smem, mask_smem, k_barrier, q_barrier, k_consumed_barrier, v_barrier, v_consumed_barrier, bias_barrier, mask_barrier, mask_consumed_barrier)` — [`L176`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L176)
- `kv_loop(ki)` — [`L476`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L476)
- `loop_body(ki, carry, *, do_causal=False)` — [`L268`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L268)
- `memory_wg()` — [`L436`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L436)
- `prologue(ki)` — [`L464`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L464)

## Module values
- `Residuals` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L38)
- `_WGMMA` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L40)
- `_WGMMA_COL` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L42)
- `_WGMMA_ROW` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L41)
- `_load_bcast` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm90.py#L43)

