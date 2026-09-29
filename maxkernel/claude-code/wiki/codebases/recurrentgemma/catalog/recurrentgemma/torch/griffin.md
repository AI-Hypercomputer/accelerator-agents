---
title: 'Module: recurrentgemma/torch/griffin.py'
type: catalog
provenance: extracted
module: recurrentgemma/torch/griffin.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.torch.griffin`/
symbols:
  Griffin: Griffin#
  Griffin.blocks: Griffin#blocks.
  Griffin.init_cache: Griffin#init_cache().
  Griffin.config: Griffin#config.
  Griffin.embedder: Griffin#embedder.
  Griffin.forward: Griffin#forward().
  Griffin.reset_parameters: Griffin#reset_parameters().
  Cache: Cache.
  Griffin.final_norm: Griffin#final_norm.
  Griffin.__init__: Griffin#__init__().
  Griffin.gradient_checkpointing: Griffin#gradient_checkpointing.
---
# Module: [`recurrentgemma/torch/griffin.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/griffin.py)

## Classes
### `Griffin`  ·  implements/extends Module
- def: [`recurrentgemma/torch/griffin.py:30`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/griffin.py#L30)
- doc: Griffin model - https://arxiv.org/abs/2402.19427.
- signature: `class Griffin(nn.Module):`
- members:
  - `__init__(self, config: common.GriffinConfig, gradient_checkpointing: bool = True, device: str | torch.device | None = None, dtype: torch.dtype | None = None)` — [`L33`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/griffin.py#L33) — Initializes the Griffin model.
  - `forward(self, tokens: at.Tokens, segment_pos: at.SegmentPos, cache: Cache | None = None, return_logits: Literal[False] = False, return_cache: Literal[False] = False)` — [`L88`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/griffin.py#L88) — documented in [recurrentgemma-torch-array_typing](../../../concepts/recurrentgemma-torch-array_typing.md)
  - `init_cache(self, batch_size: int, dtype: torch.dtype)` — [`L192`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/griffin.py#L192) — Initializes an empty cache for the model.
  - `reset_parameters(self)` — [`L80`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/griffin.py#L80) — Resets the parameters of the module.
  - `blocks` — [`L62`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/griffin.py#L62) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `config` — [`L51`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/griffin.py#L51)
  - `embedder` — [`L54`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/griffin.py#L54)
  - `final_norm` — [`L76`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/griffin.py#L76)
  - `gradient_checkpointing` — [`L52`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/griffin.py#L52)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`ResidualBlock`](modules.md#ResidualBlock), [`SegmentPos`](array_typing.md#SegmentPos), [`init_cache`](modules.md#ResidualBlock.init_cache), [`vocab_size`](../common.md#GriffinConfig.vocab_size), [`width`](../common.md#GriffinConfig.width), [`block_types`](../common.md#GriffinConfig.block_types), [`attention_window_size`](../common.md#GriffinConfig.attention_window_size), [`GriffinConfig`](../common.md#GriffinConfig), [`encode`](modules.md#Embedder.encode), [`lru_width`](../common.md#GriffinConfig.lru_width), [`num_heads`](../common.md#GriffinConfig.num_heads), [`embeddings_scale_by_sqrt_dim`](../common.md#GriffinConfig.embeddings_scale_by_sqrt_dim), [`logits_soft_cap`](../common.md#GriffinConfig.logits_soft_cap), [`mlp_expanded_width`](../common.md#GriffinConfig.mlp_expanded_width), [`Cache`](griffin.md#Cache), [`decode`](modules.md#Embedder.decode), [`Tokens`](array_typing.md#Tokens), [`reset_parameters`](modules.md#Embedder.reset_parameters), [`reset_parameters`](layers.md#RMSNorm.reset_parameters), [`TokenLogits`](array_typing.md#TokenLogits), [`num_layers`](../common.md#GriffinConfig.num_layers), [`RMSNorm`](layers.md#RMSNorm), [`Embedder`](modules.md#Embedder)
- used by: [`test_forward_equivalence`](sampler_test.md#SamplerTest.test_forward_equivalence), [`test_numerically_to_jax`](griffin_test.md#GriffinTest.test_numerically_to_jax), [`test_numerically_to_jax`](modules_test.md#EmbedderTest.test_numerically_to_jax), [`vocab_size`](sampler.md#Sampler.vocab_size), [`Griffin`](__init__.md#Griffin), [`__init__`](sampler.md#Sampler.__init__)  (1 test-only)

## Module values
- `Cache` — [`L27`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/griffin.py#L27)

