---
title: 'Module: recurrentgemma/jax/griffin.py'
type: catalog
provenance: extracted
module: recurrentgemma/jax/griffin.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.jax.griffin`/
symbols:
  Griffin.blocks: Griffin#blocks.
  Griffin: Griffin#
  Griffin.embedder: Griffin#embedder.
  Griffin.init_cache: Griffin#init_cache().
  Griffin.config: Griffin#config.
  Griffin.final_norm: Griffin#final_norm.
  Cache: Cache.
  Griffin.__call__: Griffin#__call__().
  Griffin.param_dtype: Griffin#param_dtype.
  Griffin.dtype: Griffin#dtype.
  Griffin.scan_sharding_spec: Griffin#scan_sharding_spec.
  Griffin.gradient_checkpointing: Griffin#gradient_checkpointing.
  Griffin.setup: Griffin#setup().
---
# Module: [`recurrentgemma/jax/griffin.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin.py)

## Classes
### `Griffin`  ·  implements/extends Module
- def: [`recurrentgemma/jax/griffin.py:31`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin.py#L31) — documented in [recurrentgemma-jax-griffin](../../../concepts/recurrentgemma-jax-griffin.md)
- doc: Griffin model - https://arxiv.org/abs/2402.19427.
- signature: `class Griffin(nn.Module):`
- members:
  - `init_cache(self, batch_size: int, dtype: at.dtype)` — [`L183`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin.py#L183) — Initializes an empty cache for the model. — documented in [recurrentgemma-jax-griffin](../../../concepts/recurrentgemma-jax-griffin.md)
  - `setup(self)` — [`L49`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin.py#L49)
  - `blocks` — [`L62`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin.py#L62) — documented in [recurrentgemma-jax-griffin](../../../concepts/recurrentgemma-jax-griffin.md)
  - `config` — [`L43`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin.py#L43) — documented in [recurrentgemma-jax-griffin](../../../concepts/recurrentgemma-jax-griffin.md)
  - `dtype` — [`L46`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin.py#L46)
  - `embedder` — [`L50`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin.py#L50)
  - `final_norm` — [`L79`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin.py#L79) — documented in [recurrentgemma-jax-griffin](../../../concepts/recurrentgemma-jax-griffin.md)
  - `gradient_checkpointing` — [`L45`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin.py#L45)
  - `param_dtype` — [`L47`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin.py#L47) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `scan_sharding_spec` — [`L44`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin.py#L44)
- protocol/private: `__call__`[`L86`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin.py#L86)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`ResidualBlock`](modules.md#ResidualBlock), [`dtype`](array_typing.md#dtype), [`SegmentPos`](array_typing.md#SegmentPos), [`init_cache`](modules.md#ResidualBlock.init_cache), [`vocab_size`](../common.md#GriffinConfig.vocab_size), [`width`](../common.md#GriffinConfig.width), [`block_types`](../common.md#GriffinConfig.block_types), [`attention_window_size`](../common.md#GriffinConfig.attention_window_size), [`encode`](modules.md#Embedder.encode), [`GriffinConfig`](../common.md#GriffinConfig), [`lru_width`](../common.md#GriffinConfig.lru_width), [`num_heads`](../common.md#GriffinConfig.num_heads), [`embeddings_scale_by_sqrt_dim`](../common.md#GriffinConfig.embeddings_scale_by_sqrt_dim), [`logits_soft_cap`](../common.md#GriffinConfig.logits_soft_cap), [`mlp_expanded_width`](../common.md#GriffinConfig.mlp_expanded_width), [`decode`](modules.md#Embedder.decode), [`Cache`](griffin.md#Cache), [`param_dtype`](modules.md#ResidualBlock.param_dtype), [`Tokens`](array_typing.md#Tokens), [`dtype`](modules.md#ResidualBlock.dtype), [`scan_type`](modules.md#ResidualBlock.scan_type), [`ShardingSpec`](scan.md#ShardingSpec), [`param_dtype`](layers.md#RMSNorm.param_dtype), [`scan_type`](../common.md#GriffinConfig.scan_type), [`width`](modules.md#ResidualBlock.width), [`dtype`](layers.md#RMSNorm.dtype), [`temporal_block_type`](modules.md#ResidualBlock.temporal_block_type), [`width`](layers.md#RMSNorm.width), [`dtype`](modules.md#Embedder.dtype), [`RMSNorm`](layers.md#RMSNorm), [`TokenLogits`](array_typing.md#TokenLogits), [`num_layers`](../common.md#GriffinConfig.num_layers), [`param_dtype`](modules.md#Embedder.param_dtype), [`scan_sharding_spec`](modules.md#ResidualBlock.scan_sharding_spec), [`final_w_init_variance_scale`](modules.md#ResidualBlock.final_w_init_variance_scale), [`num_heads`](modules.md#ResidualBlock.num_heads), [`attention_window_size`](modules.md#ResidualBlock.attention_window_size), [`embed_dim`](modules.md#Embedder.embed_dim), [`lru_width`](modules.md#ResidualBlock.lru_width)  (+4 more)
- used by: [`test_conversion_back_and_forth`](../conversion_test.md#ConversionTest.test_conversion_back_and_forth), [`test_griffin_output_shape`](griffin_test.md#GriffinTest.test_griffin_output_shape), [`test_numerically_to_jax`](../torch/griffin_test.md#GriffinTest.test_numerically_to_jax), [`test_numerically_to_jax`](../torch/modules_test.md#EmbedderTest.test_numerically_to_jax), [`Griffin`](__init__.md#Griffin)

## Module values
- `Cache` — [`L28`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/griffin.py#L28)

