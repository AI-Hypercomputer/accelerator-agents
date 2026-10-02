---
title: 'Module: python/sgl_jax/srt/kernels/fused_moe/v1/tuned_block_configs.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/fused_moe/v1/tuned_block_configs.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.fused_moe.v1.tuned_block_configs`/
symbols:
  get_tuned_fused_moe_block_config: get_tuned_fused_moe_block_config().
  DEFAULT_FUSED_MOE_BLOCK_CONFIG: DEFAULT_FUSED_MOE_BLOCK_CONFIG.
  get_simplified_key: get_simplified_key().
  TUNED_BLOCK_CONFIGS.TUNED_BLOCK_CONFIGS: TUNED_BLOCK_CONFIGS.TUNED_BLOCK_CONFIGS.
  logger: logger.
---
# Module: [`python/sgl_jax/srt/kernels/fused_moe/v1/tuned_block_configs.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/tuned_block_configs.py)

## Functions
- `get_simplified_key(*, dtype: jnp.dtype, weight_dtype: jnp.dtype, num_tokens: int, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int, ep_size: int, use_shared_expert: bool, use_grouped_topk: bool)` — [`L251`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/tuned_block_configs.py#L251) — Get a simplified key to reduce the number of tuned combinations.
- `get_tuned_fused_moe_block_config(*, num_tokens: int, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int, dtype: jnp.dtype, weight_dtype: jnp.dtype, ep_size: int, use_shared_expert: bool, use_grouped_topk: bool)` — [`L288`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/tuned_block_configs.py#L288) — Look up the best block config from the tuned table. — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)

## Module values
- `DEFAULT_FUSED_MOE_BLOCK_CONFIG` — [`L238`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/tuned_block_configs.py#L238) — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
- `TUNED_BLOCK_CONFIGS` — [`L42`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/tuned_block_configs.py#L42)
- `logger` — [`L25`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v1/tuned_block_configs.py#L25)

