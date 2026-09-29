---
title: 'Module: python/sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.fused_moe.v2.tuned_block_configs`/
symbols:
  get_tuned_fused_moe_v2_block_config: get_tuned_fused_moe_v2_block_config().
  DEFAULT_V2_BLOCK_CONFIG: DEFAULT_V2_BLOCK_CONFIG.
  get_tuned_fused_moe_v2_block_config._lookup: get_tuned_fused_moe_v2_block_config()._lookup().
  get_simplified_key: get_simplified_key().
  TUNED_BLOCK_CONFIGS.TUNED_BLOCK_CONFIGS: TUNED_BLOCK_CONFIGS.TUNED_BLOCK_CONFIGS.
  logger: logger.
---
# Module: [`python/sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py)

## Functions
- `_lookup(k)` — [`L151`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py#L151)
- `get_simplified_key(*, dtype: jnp.dtype, weight_dtype: jnp.dtype, num_tokens: int, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int, ep_size: int, use_shared_expert: bool, use_grouped_topk: bool, enable_act_quant: bool = False)` — [`L80`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py#L80)
- `get_tuned_fused_moe_v2_block_config(*, num_tokens: int, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int, dtype: jnp.dtype, weight_dtype: jnp.dtype, ep_size: int, use_shared_expert: bool = False, use_grouped_topk: bool = False, enable_act_quant: bool = False)` — [`L118`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py#L118)

## Module values
- `DEFAULT_V2_BLOCK_CONFIG` — [`L72`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py#L72)
- `TUNED_BLOCK_CONFIGS` — [`L31`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py#L31)
- `logger` — [`L19`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py#L19)

