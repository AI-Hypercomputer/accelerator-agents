---
title: 'Module: recurrentgemma/torch/modules.py'
type: catalog
provenance: extracted
module: recurrentgemma/torch/modules.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.torch.modules`/
symbols:
  LocalAttentionBlock: LocalAttentionBlock#
  RecurrentBlock: RecurrentBlock#
  ResidualBlock: ResidualBlock#
  RecurrentBlock.reset_parameters: RecurrentBlock#reset_parameters().
  _attention_cache_from_prompt: _attention_cache_from_prompt().
  _update_attention_cache: _update_attention_cache().
  ResidualBlock.recurrent_block: ResidualBlock#recurrent_block.
  ResidualBlock.init_cache: ResidualBlock#init_cache().
  LocalAttentionBlock.reset_parameters: LocalAttentionBlock#reset_parameters().
  RecurrentBlock.init_cache: RecurrentBlock#init_cache().
  ResidualBlock.temporal_block: ResidualBlock#temporal_block().
  Embedder.encode: Embedder#encode().
  ResidualBlockCache: ResidualBlockCache.
  ResidualBlock.reset_parameters: ResidualBlock#reset_parameters().
  MLPBlock.reset_parameters: MLPBlock#reset_parameters().
  ResidualBlock.temporal_pre_norm: ResidualBlock#temporal_pre_norm.
  ResidualBlock.attention_block: ResidualBlock#attention_block.
  AttentionBlockCache: AttentionBlockCache#
  MLPBlock.forward: MLPBlock#forward().
  _apply_rope: _apply_rope().
  _compute_causal_mask: _compute_causal_mask().
  RecurrentBlock.rg_lru: RecurrentBlock#rg_lru.
  ResidualBlock.mlp_block: ResidualBlock#mlp_block.
  _compute_forward_pass_mask: _compute_forward_pass_mask().
  _compute_cache_mask: _compute_cache_mask().
  LocalAttentionBlock.init_cache: LocalAttentionBlock#init_cache().
  Embedder.decode: Embedder#decode().
  MLPBlock.ffw_down: MLPBlock#ffw_down.
  Embedder.input_embedding: Embedder#input_embedding.
  RecurrentBlockCache: RecurrentBlockCache#
  LocalAttentionBlock.forward: LocalAttentionBlock#forward().
  RecurrentBlock.conv_1d: RecurrentBlock#conv_1d.
  RecurrentBlock.forward: RecurrentBlock#forward().
  MLPBlock.ffw_up: MLPBlock#ffw_up.
  ResidualBlock.forward: ResidualBlock#forward().
  LocalAttentionBlock.width: LocalAttentionBlock#width.
  AttentionBlockCache.num_tokens: AttentionBlockCache#num_tokens.
  RecurrentBlockCache.conv1d_state: RecurrentBlockCache#conv1d_state.
  LocalAttentionBlock.proj_final: LocalAttentionBlock#proj_final.
  LocalAttentionBlock.head_dim: LocalAttentionBlock#head_dim().
  RecurrentBlock.linear_y: RecurrentBlock#linear_y.
  RecurrentBlock.linear_x: RecurrentBlock#linear_x.
  RecurrentBlock.linear_out: RecurrentBlock#linear_out.
  AttentionBlockCache.keys: AttentionBlockCache#keys.
  LocalAttentionBlock.proj_k: LocalAttentionBlock#proj_k.
  LocalAttentionBlock.proj_v: LocalAttentionBlock#proj_v.
  Embedder.reset_parameters: Embedder#reset_parameters().
  AttentionBlockCache.values: AttentionBlockCache#values.
  LocalAttentionBlock.out_w_init_: LocalAttentionBlock#out_w_init_().
  RecurrentBlock.lru_width: RecurrentBlock#lru_width.
  RecurrentBlock.out_w_init_: RecurrentBlock#out_w_init_().
  MLPBlock.out_w_init_: MLPBlock#out_w_init_().
  RecurrentBlockCache.rg_lru_state: RecurrentBlockCache#rg_lru_state.
  LocalAttentionBlock.w_init_: LocalAttentionBlock#w_init_().
  LocalAttentionBlock.proj_q: LocalAttentionBlock#proj_q.
  RecurrentBlock.w_init_: RecurrentBlock#w_init_().
  ResidualBlock.channel_pre_norm: ResidualBlock#channel_pre_norm.
  RecurrentBlock.width: RecurrentBlock#width.
  MLPBlock.expanded_width: MLPBlock#expanded_width.
  ResidualBlock.width: ResidualBlock#width.
  _MAX_WAVELENGTH: _MAX_WAVELENGTH.
  LocalAttentionBlock.num_heads: LocalAttentionBlock#num_heads.
  LocalAttentionBlock.window_size: LocalAttentionBlock#window_size.
  ResidualBlock.final_w_init_variance_scale: ResidualBlock#final_w_init_variance_scale.
  Embedder.embed_dim: Embedder#embed_dim.
  ResidualBlock.__init__: ResidualBlock#__init__().
  _roll_tensor: _roll_tensor().
  _right_pad_tensor: _right_pad_tensor().
  gelu: gelu().
  MLPBlock: MLPBlock#
  MLPBlock.width: MLPBlock#width.
  ResidualBlock.num_heads: ResidualBlock#num_heads.
  ResidualBlock.temporal_block_type: ResidualBlock#temporal_block_type.
  _MIN_LOGITS_VALUE: _MIN_LOGITS_VALUE.
  LocalAttentionBlock.final_w_init_variance_scale: LocalAttentionBlock#final_w_init_variance_scale.
  RecurrentBlock.num_heads: RecurrentBlock#num_heads.
  RecurrentBlock.conv1d_temporal_width: RecurrentBlock#conv1d_temporal_width.
  RecurrentBlock.final_w_init_variance_scale: RecurrentBlock#final_w_init_variance_scale.
  MLPBlock.final_w_init_variance_scale: MLPBlock#final_w_init_variance_scale.
  ResidualBlock.mlp_expanded_width: ResidualBlock#mlp_expanded_width.
  ResidualBlock.attention_window_size: ResidualBlock#attention_window_size.
  ResidualBlock.lru_width: ResidualBlock#lru_width.
  ResidualBlock.conv1d_temporal_width: ResidualBlock#conv1d_temporal_width.
  Embedder: Embedder#
  Embedder.vocab_size: Embedder#vocab_size.
  Embedder.scale_by_sqrt_dim: Embedder#scale_by_sqrt_dim.
  LocalAttentionBlock.__init__: LocalAttentionBlock#__init__().
  RecurrentBlock.__init__: RecurrentBlock#__init__().
  MLPBlock.__init__: MLPBlock#__init__().
  Embedder.__init__: Embedder#__init__().
---
# Module: [`recurrentgemma/torch/modules.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py)

## Classes
### `AttentionBlockCache`  ·  implements/extends NamedTuple
- def: [`recurrentgemma/torch/modules.py:41`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L41) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
- doc: The cache for an attention block.
- signature: `class AttentionBlockCache(NamedTuple):`
- members:
  - `keys` — [`L44`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L44) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `num_tokens` — [`L46`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L46) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `values` — [`L45`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L45) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
- uses (calls/refs, reference-scoped): [`NumTokens`](array_typing.md#NumTokens), [`CachedKeys`](array_typing.md#CachedKeys), [`CachedValues`](array_typing.md#CachedValues)
- used by: [`LocalAttentionBlock`](modules.md#LocalAttentionBlock), [`_attention_cache_from_prompt`](modules.md#_attention_cache_from_prompt), [`_update_attention_cache`](modules.md#_update_attention_cache), [`ResidualBlockCache`](modules.md#ResidualBlockCache), [`init_cache`](modules.md#LocalAttentionBlock.init_cache), [`forward`](modules.md#LocalAttentionBlock.forward), [`AttentionBlockCache`](__init__.md#AttentionBlockCache)

### `Embedder`  ·  implements/extends Module
- def: [`recurrentgemma/torch/modules.py:946`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L946)
- doc: Embedder module.
- signature: `class Embedder(nn.Module):`
- members:
  - `__init__(self, vocab_size: int, embed_dim: int, scale_by_sqrt_dim: bool, device: str | torch.device | None = None, dtype: torch.dtype | None = None)` — [`L949`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L949) — Initializes the embedder.
  - `decode(self, x: at.Activations)` — [`L1001`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L1001) — Decodes an input sequence of activations.
  - `encode(self, x: at.Tokens)` — [`L992`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L992) — Encodes an input sequence of tokens.
  - `reset_parameters(self)` — [`L983`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L983) — Resets the parameters of the module.
  - `embed_dim` — [`L970`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L970)
  - `input_embedding` — [`L974`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L974)
  - `scale_by_sqrt_dim` — [`L971`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L971)
  - `vocab_size` — [`L969`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L969)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`Activations`](array_typing.md#Activations), [`Tokens`](array_typing.md#Tokens), [`TokenLogits`](array_typing.md#TokenLogits)
- used by: [`Griffin`](griffin.md#Griffin), [`embedder`](griffin.md#Griffin.embedder), [`reset_parameters`](griffin.md#Griffin.reset_parameters)

### `LocalAttentionBlock`  ·  implements/extends Module
- def: [`recurrentgemma/torch/modules.py:294`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L294) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
- doc: Local Multi-Head Attention (MHA) block.
- signature: `class LocalAttentionBlock(nn.Module):`
- members:
  - `__init__(self, width: int, num_heads: int, window_size: int, final_w_init_variance_scale: float = 1, device: str | torch.device | None = None, dtype: torch.dtype | None = None)` — [`L297`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L297) — Initializes the local attention block.
  - `forward(self, x: at.Activations, segment_pos: at.SegmentPos, cache: AttentionBlockCache | None = None, return_cache: Literal[True] = True)` — [`L379`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L379) — documented in [recurrentgemma-torch-array_typing](../../../concepts/recurrentgemma-torch-array_typing.md)
  - `head_dim(self)` — [`L366`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L366) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `init_cache(cls, batch_size: int, window_size: int, heads_dim: int, dtype: torch.dtype, device: str | torch.device | None = None)` — [`L479`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L479) — Initializes an empty KV-cache for the block.
  - `out_w_init_(self, w: torch.Tensor)` — [`L373`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L373) — Initializes the weights of the final projection.
  - `reset_parameters(self)` — [`L357`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L357) — Resets the parameters of the module.
  - `w_init_(self, w: torch.Tensor)` — [`L369`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L369) — Initializes the weights of the queries, keys and values projections.
  - `final_w_init_variance_scale` — [`L322`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L322)
  - `num_heads` — [`L320`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L320)
  - `proj_final` — [`L346`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L346) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `proj_k` — [`L332`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L332) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `proj_q` — [`L325`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L325)
  - `proj_v` — [`L339`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L339) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `width` — [`L319`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L319) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `window_size` — [`L321`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L321)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`SegmentPos`](array_typing.md#SegmentPos), [`_attention_cache_from_prompt`](modules.md#_attention_cache_from_prompt), [`_update_attention_cache`](modules.md#_update_attention_cache), [`Activations`](array_typing.md#Activations), [`AttentionBlockCache`](modules.md#AttentionBlockCache), [`_apply_rope`](modules.md#_apply_rope), [`_compute_cache_mask`](modules.md#_compute_cache_mask), [`_compute_forward_pass_mask`](modules.md#_compute_forward_pass_mask), [`num_tokens`](modules.md#AttentionBlockCache.num_tokens), [`keys`](modules.md#AttentionBlockCache.keys), [`values`](modules.md#AttentionBlockCache.values), `bias`, [`_MIN_LOGITS_VALUE`](modules.md#_MIN_LOGITS_VALUE)
- used by: [`init_cache`](modules.md#ResidualBlock.init_cache), [`test_numerically_to_jax`](modules_test.md#LocalAttentionTest.test_numerically_to_jax), [`attention_block`](modules.md#ResidualBlock.attention_block), [`LocalAttentionBlock`](__init__.md#LocalAttentionBlock)

### `MLPBlock`  ·  implements/extends Module
- def: [`recurrentgemma/torch/modules.py:683`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L683)
- doc: MLP block.
- signature: `class MLPBlock(nn.Module):`
- members:
  - `__init__(self, width: int, expanded_width: int, final_w_init_variance_scale: float = 1, device: str | torch.device | None = None, dtype: torch.dtype | None = None)` — [`L686`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L686) — Initializes the MLP block.
  - `forward(self, x: at.Activations)` — [`L740`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L740) — Calls the MLP block.
  - `out_w_init_(self, w: torch.Tensor)` — [`L734`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L734) — Initializes the weights of the last layer of the block.
  - `reset_parameters(self)` — [`L728`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L728) — Resets the parameters of the module.
  - `expanded_width` — [`L707`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L707)
  - `ffw_down` — [`L718`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L718)
  - `ffw_up` — [`L711`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L711)
  - `final_w_init_variance_scale` — [`L708`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L708)
  - `width` — [`L706`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L706)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`Activations`](array_typing.md#Activations), [`reset_parameters`](layers.md#Einsum.reset_parameters), `bias`, [`Einsum`](layers.md#Einsum), [`gelu`](modules.md#gelu)
- used by: [`reset_parameters`](modules.md#ResidualBlock.reset_parameters), [`test_numerically_to_jax`](modules_test.md#MLPBlockTest.test_numerically_to_jax), [`mlp_block`](modules.md#ResidualBlock.mlp_block)

### `RecurrentBlock`  ·  implements/extends Module
- def: [`recurrentgemma/torch/modules.py:496`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L496) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
- doc: Griffin and Hawk's recurrent block.
- signature: `class RecurrentBlock(nn.Module):`
- members:
  - `__init__(self, width: int, num_heads: int, lru_width: int | None = None, conv1d_temporal_width: int = 4, final_w_init_variance_scale: float = 1, device: str | torch.device | None = None, dtype: torch.dtype | None = None)` — [`L499`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L499) — Initializes the recurrent block.
  - `forward(self, x: at.Activations, segment_pos: at.SegmentPos, cache: RecurrentBlockCache | None = None, return_cache: Literal[True] = True)` — [`L586`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L586) — documented in [recurrentgemma-torch-array_typing](../../../concepts/recurrentgemma-torch-array_typing.md)
  - `init_cache(cls, batch_size: int, lru_width: int, dtype: torch.dtype, conv1d_temporal_width: int = 4, device: str | torch.device | None = None)` — [`L658`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L658) — Initializes an empty RG-LRU and Conv1D cache for the block.
  - `out_w_init_(self, w: torch.Tensor)` — [`L580`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L580) — Initializes the weights of the last layer of the block.
  - `reset_parameters(self)` — [`L565`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L565) — Resets the parameters of the module. — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `w_init_(self, w: torch.Tensor)` — [`L576`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L576) — Initializes the weights of the linear x and y layers of the block.
  - `conv1d_temporal_width` — [`L527`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L527)
  - `conv_1d` — [`L549`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L549) — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `final_w_init_variance_scale` — [`L528`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L528)
  - `linear_out` — [`L543`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L543) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `linear_x` — [`L537`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L537) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `linear_y` — [`L531`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L531) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `lru_width` — [`L526`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L526)
  - `num_heads` — [`L525`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L525)
  - `rg_lru` — [`L555`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L555) — documented in [recurrentgemma-torch-layers](../../../concepts/recurrentgemma-torch-layers.md)
  - `width` — [`L524`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L524)
- uses (calls/refs, reference-scoped): [`Conv1D`](layers.md#Conv1D), [`typed`](array_typing.md#typed), [`RGLRU`](layers.md#RGLRU), [`SegmentPos`](array_typing.md#SegmentPos), [`Activations`](array_typing.md#Activations), [`reset_parameters`](layers.md#RGLRU.reset_parameters), [`RecurrentBlockCache`](modules.md#RecurrentBlockCache), [`reset_parameters`](layers.md#Conv1D.reset_parameters), [`conv1d_state`](modules.md#RecurrentBlockCache.conv1d_state), [`rg_lru_state`](modules.md#RecurrentBlockCache.rg_lru_state), `bias`, [`init_cache`](layers.md#Conv1D.init_cache), [`init_cache`](layers.md#RGLRU.init_cache), [`gelu`](modules.md#gelu)
- used by: [`test_numerically_to_jax`](modules_test.md#RecurrentBlockTest.test_numerically_to_jax), [`init_cache`](modules.md#ResidualBlock.init_cache), [`recurrent_block`](modules.md#ResidualBlock.recurrent_block), [`RecurrentBlock`](__init__.md#RecurrentBlock)

### `RecurrentBlockCache`  ·  implements/extends NamedTuple
- def: [`recurrentgemma/torch/modules.py:33`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L33) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
- doc: The cache for a recurrent block.
- signature: `class RecurrentBlockCache(NamedTuple):`
- members:
  - `conv1d_state` — [`L37`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L37)
  - `rg_lru_state` — [`L36`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L36)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`Conv1DState`](array_typing.md#Conv1DState), [`RNNState`](array_typing.md#RNNState)
- used by: [`RecurrentBlock`](modules.md#RecurrentBlock), [`init_cache`](modules.md#RecurrentBlock.init_cache), [`ResidualBlockCache`](modules.md#ResidualBlockCache), [`forward`](modules.md#RecurrentBlock.forward), [`RecurrentBlockCache`](__init__.md#RecurrentBlockCache)

### `ResidualBlock`  ·  implements/extends Module
- def: [`recurrentgemma/torch/modules.py:755`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L755)
- doc: Griffin and Hawk's residual block.
- signature: `class ResidualBlock(nn.Module):`
- members:
  - `__init__(self, width: int, mlp_expanded_width: int, num_heads: int, attention_window_size: int, temporal_block_type: common.TemporalBlockType, lru_width: int | None = None, conv1d_temporal_width: int = 4, final_w_init_variance_scale: float = 1, device: str | torch.device | None = None, dtype: torch.dtype | None = None)` — [`L758`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L758) — Initializes the residual block.
  - `forward(self, x: at.Activations, segment_pos: at.SegmentPos, cache: ResidualBlockCache | None = None, return_cache: Literal[True] = True)` — [`L857`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L857) — documented in [recurrentgemma-torch-array_typing](../../../concepts/recurrentgemma-torch-array_typing.md)
  - `init_cache(cls, batch_size: int, width: int, num_heads: int, attention_window_size: int, temporal_block_type: common.TemporalBlockType, dtype: torch.dtype, lru_width: int | None = None, conv1d_temporal_width: int = 4, device: str | torch.device | None = None)` — [`L914`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L914) — Initializes an empty cache for the block. — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `reset_parameters(self)` — [`L836`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L836) — Resets the parameters of the module.
  - `temporal_block(self)` — [`L844`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L844) — Alias for the temporal block. — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `attention_block` — [`L816`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L816) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `attention_window_size` — [`L792`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L792)
  - `channel_pre_norm` — [`L825`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L825)
  - `conv1d_temporal_width` — [`L795`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L795)
  - `final_w_init_variance_scale` — [`L796`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L796)
  - `lru_width` — [`L794`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L794)
  - `mlp_block` — [`L828`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L828) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `mlp_expanded_width` — [`L790`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L790)
  - `num_heads` — [`L791`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L791)
  - `recurrent_block` — [`L805`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L805) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `temporal_block_type` — [`L793`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L793)
  - `temporal_pre_norm` — [`L799`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L799) — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
  - `width` — [`L789`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L789)
- uses (calls/refs, reference-scoped): [`TemporalBlockType`](../common.md#TemporalBlockType), [`LocalAttentionBlock`](modules.md#LocalAttentionBlock), [`RecurrentBlock`](modules.md#RecurrentBlock), [`typed`](array_typing.md#typed), [`RECURRENT`](../common.md#TemporalBlockType.RECURRENT), [`SegmentPos`](array_typing.md#SegmentPos), [`Activations`](array_typing.md#Activations), [`ATTENTION`](../common.md#TemporalBlockType.ATTENTION), [`init_cache`](modules.md#RecurrentBlock.init_cache), [`ResidualBlockCache`](modules.md#ResidualBlockCache), [`reset_parameters`](modules.md#MLPBlock.reset_parameters), [`init_cache`](modules.md#LocalAttentionBlock.init_cache), [`reset_parameters`](layers.md#RMSNorm.reset_parameters), [`RMSNorm`](layers.md#RMSNorm), [`MLPBlock`](modules.md#MLPBlock)
- used by: [`test_numerically_to_jax`](modules_test.md#ResidualBlockTest.test_numerically_to_jax), [`blocks`](griffin.md#Griffin.blocks), [`init_cache`](griffin.md#Griffin.init_cache), [`ResidualBlock`](__init__.md#ResidualBlock)

## Functions
- `_apply_rope(inputs: at.Keys | at.Queries, positions: at.SegmentPos, max_wavelength: int = _MAX_WAVELENGTH)` — [`L53`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L53) — Applies RoPE to the first half of inputs. — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
- `_attention_cache_from_prompt(keys: at.Keys, values: at.Values, segment_pos: torch.Tensor, window_size: int)` — [`L257`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L257) — Creates a new cache from a prompt. — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
- `_compute_cache_mask(seq_len: int, cache_num_tokens: at.NumTokens, window_size: int)` — [`L154`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L154) — Computes the mask when there a KV-cache is present. — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
- `_compute_causal_mask(q_positions: torch.Tensor, k_positions: torch.Tensor, window_size: int, q_segment_ids: at.QuerySegmentIds | None, k_segment_ids: at.KeySegmentIds | None)` — [`L89`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L89) — Computes the causal mask for local attention. — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
- `_compute_forward_pass_mask(segment_pos: at.SegmentPos, window_size: int)` — [`L129`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L129) — Compute the forward pass mask. — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
- `_right_pad_tensor(x: torch.Tensor, size: int, dim: int)` — [`L246`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L246)
- `_roll_tensor(x: torch.Tensor, shifts: torch.Tensor, dim: int)` — [`L226`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L226) — Rolls the tensor along the given axis by shifts.
- `_update_attention_cache(keys: at.Keys, values: at.Values, segment_pos: at.SegmentPos, cache: AttentionBlockCache)` — [`L187`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L187) — Updates the cache with the new keys and values. — documented in [recurrentgemma-torch-modules](../../../concepts/recurrentgemma-torch-modules.md)
- `gelu(x: torch.Tensor)` — [`L289`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L289) — Returns the GELU activation function with the same approximation as JAX.

## Module values
- `ResidualBlockCache` — [`L49`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L49)
- `_MAX_WAVELENGTH` — [`L29`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L29)
- `_MIN_LOGITS_VALUE` — [`L28`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/modules.py#L28)

