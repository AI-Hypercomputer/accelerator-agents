---
title: 'Module: tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_common.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_common.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.pallas_mosaic_gpu_vjp_common`/Config#
symbols:
  Config.block_kv_dq: block_kv_dq.
  Config.block_q_dq: block_q_dq.
  Config.num_stages: num_stages.
  Config.block_kv_dkv: block_kv_dkv.
  Config.block_q_dkv: block_q_dkv.
  Config: ''
  Config.compute_wgs: compute_wgs.
---
# Module: [`tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_common.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_common.py)

## Classes
### `Config`
- def: [`tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_common.py:22`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_common.py#L22) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm90](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm90.md)
- signature: `class Config:`
- members:
  - `block_kv_dkv` — [`L24`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_common.py#L24) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100.md)
  - `block_kv_dq` — [`L26`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_common.py#L26) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100.md)
  - `block_q_dkv` — [`L23`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_common.py#L23) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100.md)
  - `block_q_dq` — [`L25`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_common.py#L25) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm100.md)
  - `compute_wgs` — [`L28`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_common.py#L28) — documented in [tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm90](../../../../../concepts/tokamax-_src-ops-attention-pallas_mosaic_gpu_vjp_kernel_sm90.md)
  - `num_stages` — [`L27`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_common.py#L27)
- uses (calls/refs, reference-scoped): [`Config`](pallas_mosaic_gpu_vjp_kernel_sm100.md#Config)
- used by: [`get_autotuning_configs`](pallas_mosaic_gpu_vjp_kernel_sm100.md#get_autotuning_configs), [`flash_attention_vjp_kernel`](pallas_mosaic_gpu_vjp_kernel_sm100.md#flash_attention_vjp_kernel), [`flash_attention_vjp_kernel`](pallas_mosaic_gpu_vjp_kernel_sm90.md#flash_attention_vjp_kernel), [`get_heuristics_config`](pallas_mosaic_gpu_vjp_kernel_sm90.md#get_heuristics_config), [`sfu_loop`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq.sfu_wg.sfu_loop), [`_get_dkv_scratch_shapes`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_get_dkv_scratch_shapes), [`_get_dq_scratch_shapes`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_get_dq_scratch_shapes), [`get_autotuning_configs`](pallas_mosaic_gpu_vjp_kernel_sm90.md#get_autotuning_configs), [`Config`](pallas_mosaic_gpu_vjp_kernel_sm100.md#Config), [`_score`](pallas_mosaic_gpu_vjp_kernel_sm100.md#get_heuristics_config._score), [`_kernel_dq`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq), [`kernel_dkv`](pallas_mosaic_gpu_vjp_kernel_sm90.md#flash_attention_vjp_kernel.kernel_dkv), [`kernel_dq`](pallas_mosaic_gpu_vjp_kernel_sm90.md#flash_attention_vjp_kernel.kernel_dq), [`Config`](pallas_mosaic_gpu_vjp_kernel_sm90.md#Config), [`kv_loop`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq.mma_tma_wg.per_warp.tma_kv.kv_loop), [`mma_loop`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq.mma_tma_wg.per_warp.mma.mma_loop), [`Config`](pallas_mosaic_gpu_vjp.md#Config), [`eltwise_loop`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq.mma_tma_wg.per_warp.tma_eltwise.eltwise_loop), [`prologue`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq.mma_tma_wg.per_warp.tma_kv.prologue), [`tma_q`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq.mma_tma_wg.per_warp.tma_q), [`kv_epilogue`](pallas_mosaic_gpu_vjp_kernel_sm100.md#_kernel_dq.mma_tma_wg.per_warp.tma_kv.kv_epilogue)

