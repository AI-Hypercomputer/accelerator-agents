---
title: 'Module: recurrentgemma/jax/modules.py'
type: catalog
provenance: extracted
module: recurrentgemma/jax/modules.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.jax.modules`/
symbols:
  ResidualBlock.recurrent_block: ResidualBlock#recurrent_block.
  LocalAttentionBlock: LocalAttentionBlock#
  RecurrentBlock.lru: RecurrentBlock#lru.
  ResidualBlock.attention_block: ResidualBlock#attention_block.
  RecurrentBlock: RecurrentBlock#
  ResidualBlock: ResidualBlock#
  ResidualBlock.mlp: ResidualBlock#mlp.
  MLPBlock.ffw_up: MLPBlock#ffw_up.
  ResidualBlock.temporal_pre_norm: ResidualBlock#temporal_pre_norm.
  _attention_cache_from_prompt: _attention_cache_from_prompt().
  ResidualBlock.init_cache: ResidualBlock#init_cache().
  _update_attention_cache: _update_attention_cache().
  RecurrentBlock.conv_1d: RecurrentBlock#conv_1d.
  RecurrentBlock.init_cache: RecurrentBlock#init_cache().
  ResidualBlock.channel_pre_norm: ResidualBlock#channel_pre_norm.
  Embedder.encode: Embedder#encode().
  ResidualBlock.temporal_block: ResidualBlock#temporal_block().
  ResidualBlockCache: ResidualBlockCache.
  LocalAttentionBlock.init_cache: LocalAttentionBlock#init_cache().
  AttentionBlockCache: AttentionBlockCache#
  MLPBlock.ffw_down: MLPBlock#ffw_down.
  Embedder.decode: Embedder#decode().
  _apply_rope: _apply_rope().
  _compute_causal_mask: _compute_causal_mask().
  RecurrentBlockCache: RecurrentBlockCache#
  _compute_forward_pass_mask: _compute_forward_pass_mask().
  _compute_cache_mask: _compute_cache_mask().
  LocalAttentionBlock.q: LocalAttentionBlock#q.
  LocalAttentionBlock.k: LocalAttentionBlock#k.
  LocalAttentionBlock.v: LocalAttentionBlock#v.
  LocalAttentionBlock.out: LocalAttentionBlock#out.
  RecurrentBlock.linear_out: RecurrentBlock#linear_out.
  AttentionBlockCache.keys: AttentionBlockCache#keys.
  AttentionBlockCache.num_tokens: AttentionBlockCache#num_tokens.
  LocalAttentionBlock.__call__: LocalAttentionBlock#__call__().
  RecurrentBlock.param_dtype: RecurrentBlock#param_dtype.
  RecurrentBlock.__call__: RecurrentBlock#__call__().
  ResidualBlock.param_dtype: ResidualBlock#param_dtype.
  ResidualBlock.__call__: ResidualBlock#__call__().
  RecurrentBlockCache.conv1d_state: RecurrentBlockCache#conv1d_state.
  AttentionBlockCache.values: AttentionBlockCache#values.
  LocalAttentionBlock.param_dtype: LocalAttentionBlock#param_dtype.
  RecurrentBlock.dtype: RecurrentBlock#dtype.
  RecurrentBlock.linear_y: RecurrentBlock#linear_y.
  RecurrentBlock.linear_x: RecurrentBlock#linear_x.
  ResidualBlock.dtype: ResidualBlock#dtype.
  LocalAttentionBlock.head_dim: LocalAttentionBlock#head_dim().
  RecurrentBlock.scan_type: RecurrentBlock#scan_type.
  ResidualBlock.scan_type: ResidualBlock#scan_type.
  LocalAttentionBlock.dtype: LocalAttentionBlock#dtype.
  MLPBlock.__call__: MLPBlock#__call__().
  Embedder.setup: Embedder#setup().
  LocalAttentionBlock.width: LocalAttentionBlock#width.
  LocalAttentionBlock.num_heads: LocalAttentionBlock#num_heads.
  LocalAttentionBlock.window_size: LocalAttentionBlock#window_size.
  ResidualBlock.width: ResidualBlock#width.
  RecurrentBlockCache.rg_lru_state: RecurrentBlockCache#rg_lru_state.
  MLPBlock.param_dtype: MLPBlock#param_dtype.
  ResidualBlock.temporal_block_type: ResidualBlock#temporal_block_type.
  MLPBlock.dtype: MLPBlock#dtype.
  Embedder.dtype: Embedder#dtype.
  RecurrentBlock.width: RecurrentBlock#width.
  RecurrentBlock.setup: RecurrentBlock#setup().
  _vmap_cache_roll: _vmap_cache_roll.
  RecurrentBlock.scan_sharding_spec: RecurrentBlock#scan_sharding_spec.
  ResidualBlock.scan_sharding_spec: ResidualBlock#scan_sharding_spec.
  Embedder.param_dtype: Embedder#param_dtype.
  RecurrentBlock.num_heads: RecurrentBlock#num_heads.
  RecurrentBlock.lru_width: RecurrentBlock#lru_width.
  MLPBlock.width: MLPBlock#width.
  MLPBlock.expanded_width: MLPBlock#expanded_width.
  ResidualBlock.num_heads: ResidualBlock#num_heads.
  ResidualBlock.final_w_init_variance_scale: ResidualBlock#final_w_init_variance_scale.
  LocalAttentionBlock.out_kernel_init: LocalAttentionBlock#out_kernel_init().
  RecurrentBlock.out_kernel_init: RecurrentBlock#out_kernel_init().
  MLPBlock.out_kernel_init: MLPBlock#out_kernel_init().
  LocalAttentionBlock.kernel_init: LocalAttentionBlock#kernel_init().
  RecurrentBlock.conv1d_temporal_width: RecurrentBlock#conv1d_temporal_width.
  ResidualBlock.mlp_expanded_width: ResidualBlock#mlp_expanded_width.
  ResidualBlock.attention_window_size: ResidualBlock#attention_window_size.
  ResidualBlock.lru_width: ResidualBlock#lru_width.
  Embedder.embed_dim: Embedder#embed_dim.
  LocalAttentionBlock.final_w_init_variance_scale: LocalAttentionBlock#final_w_init_variance_scale.
  RecurrentBlock.final_w_init_variance_scale: RecurrentBlock#final_w_init_variance_scale.
  RecurrentBlock.lru_only_real: RecurrentBlock#lru_only_real.
  RecurrentBlock.min_rad: RecurrentBlock#min_rad.
  RecurrentBlock.kernel_init: RecurrentBlock#kernel_init().
  MLPBlock: MLPBlock#
  MLPBlock.final_w_init_variance_scale: MLPBlock#final_w_init_variance_scale.
  ResidualBlock.use_mlp: ResidualBlock#use_mlp.
  Embedder.vocab_size: Embedder#vocab_size.
  Embedder.scale_by_sqrt_dim: Embedder#scale_by_sqrt_dim.
  Embedder.input_embedding_table: Embedder#input_embedding_table.
  _MIN_LOGITS_VALUE: _MIN_LOGITS_VALUE.
  _MAX_WAVELENGTH: _MAX_WAVELENGTH.
  ResidualBlock.conv1d_temporal_width: ResidualBlock#conv1d_temporal_width.
  ResidualBlock.lru_only_real: ResidualBlock#lru_only_real.
  ResidualBlock.min_rad: ResidualBlock#min_rad.
  Embedder: Embedder#
  LocalAttentionBlock.setup: LocalAttentionBlock#setup().
  MLPBlock.setup: MLPBlock#setup().
  ResidualBlock.setup: ResidualBlock#setup().
---
# Module: [`recurrentgemma/jax/modules.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py)

## Classes
### `AttentionBlockCache`  ·  implements/extends NamedTuple
- def: [`recurrentgemma/jax/modules.py:42`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L42) — documented in [recurrentgemma-jax-modules](../../../concepts/recurrentgemma-jax-modules.md)
- doc: The cache for an attention block.
- signature: `class AttentionBlockCache(NamedTuple):`
- members:
  - `keys` — [`L45`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L45) — documented in [recurrentgemma-jax-modules](../../../concepts/recurrentgemma-jax-modules.md)
  - `num_tokens` — [`L47`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L47) — documented in [recurrentgemma-jax-modules](../../../concepts/recurrentgemma-jax-modules.md)
  - `values` — [`L46`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L46) — documented in [recurrentgemma-jax-modules](../../../concepts/recurrentgemma-jax-modules.md)
- uses (calls/refs, reference-scoped): [`NumTokens`](array_typing.md#NumTokens), [`CachedKeys`](array_typing.md#CachedKeys), [`CachedValues`](array_typing.md#CachedValues)
- used by: [`LocalAttentionBlock`](modules.md#LocalAttentionBlock), [`test_griffin_output_shape`](griffin_test.md#GriffinTest.test_griffin_output_shape), [`_attention_cache_from_prompt`](modules.md#_attention_cache_from_prompt), [`_update_attention_cache`](modules.md#_update_attention_cache), [`ResidualBlockCache`](modules.md#ResidualBlockCache), [`init_cache`](modules.md#LocalAttentionBlock.init_cache), [`__call__`](modules.md#LocalAttentionBlock.__call__), [`AttentionBlockCache`](__init__.md#AttentionBlockCache)

### `Embedder`  ·  implements/extends Module
- def: [`recurrentgemma/jax/modules.py:890`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L890)
- doc: Embedder module.
- signature: `class Embedder(nn.Module):`
- members:
  - `decode(self, x: at.Activations)` — [`L935`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L935) — Decodes an input sequence of activations. — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `encode(self, x: at.Tokens)` — [`L924`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L924) — Encodes an input sequence of tokens. — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `setup(self)` — [`L908`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L908)
  - `dtype` — [`L905`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L905)
  - `embed_dim` — [`L903`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L903)
  - `input_embedding_table` — [`L910`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L910)
  - `param_dtype` — [`L906`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L906)
  - `scale_by_sqrt_dim` — [`L904`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L904)
  - `vocab_size` — [`L902`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L902)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`dtype`](array_typing.md#dtype), [`Activations`](array_typing.md#Activations), [`Tokens`](array_typing.md#Tokens), [`TokenLogits`](array_typing.md#TokenLogits)
- used by: [`Griffin`](griffin.md#Griffin), [`embedder`](griffin.md#Griffin.embedder)

### `LocalAttentionBlock`  ·  implements/extends Module
- def: [`recurrentgemma/jax/modules.py:251`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L251) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
- doc: Local Multi-Head Attention (MHA) block.
- signature: `class LocalAttentionBlock(nn.Module):`
- members:
  - `head_dim(self)` — [`L272`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L272) — The dimension of each head. — documented in [recurrentgemma-jax-modules](../../../concepts/recurrentgemma-jax-modules.md)
  - `init_cache(cls, batch_size: int, window_size: int, heads_dim: int, dtype: at.dtype)` — [`L430`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L430) — Initializes an empty KV-cache for the block.
  - `kernel_init(self)` — [`L277`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L277) — Initialization of the kernel for the queries, keys and values projections.
  - `out_kernel_init(self)` — [`L286`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L286) — Initialization of the kernel for the final projection.
  - `setup(self)` — [`L294`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L294)
  - `dtype` — [`L268`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L268)
  - `final_w_init_variance_scale` — [`L267`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L267)
  - `k` — [`L304`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L304) — documented in [recurrentgemma-jax-modules](../../../concepts/recurrentgemma-jax-modules.md)
  - `num_heads` — [`L265`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L265)
  - `out` — [`L320`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L320)
  - `param_dtype` — [`L269`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L269) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `q` — [`L296`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L296) — documented in [recurrentgemma-jax-modules](../../../concepts/recurrentgemma-jax-modules.md)
  - `v` — [`L312`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L312) — documented in [recurrentgemma-jax-modules](../../../concepts/recurrentgemma-jax-modules.md)
  - `width` — [`L264`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L264)
  - `window_size` — [`L266`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L266)
- protocol/private: `__call__`[`L330`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L330)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`dtype`](array_typing.md#dtype), [`SegmentPos`](array_typing.md#SegmentPos), [`_attention_cache_from_prompt`](modules.md#_attention_cache_from_prompt), [`_update_attention_cache`](modules.md#_update_attention_cache), [`Activations`](array_typing.md#Activations), [`AttentionBlockCache`](modules.md#AttentionBlockCache), [`_apply_rope`](modules.md#_apply_rope), [`_compute_cache_mask`](modules.md#_compute_cache_mask), [`_compute_forward_pass_mask`](modules.md#_compute_forward_pass_mask), [`keys`](modules.md#AttentionBlockCache.keys), [`num_tokens`](modules.md#AttentionBlockCache.num_tokens), [`values`](modules.md#AttentionBlockCache.values), [`_MIN_LOGITS_VALUE`](modules.md#_MIN_LOGITS_VALUE)
- used by: [`attention_block`](modules.md#ResidualBlock.attention_block), [`init_cache`](modules.md#ResidualBlock.init_cache), [`test_numerically_to_jax`](../torch/modules_test.md#LocalAttentionTest.test_numerically_to_jax), [`test_local_attention_output_shapes`](modules_test.md#LocalAttentionTest.test_local_attention_output_shapes), [`test_local_attention_updates_cache_correctly`](modules_test.md#LocalAttentionTest.test_local_attention_updates_cache_correctly), [`LocalAttentionBlock`](__init__.md#LocalAttentionBlock)

### `MLPBlock`  ·  implements/extends Module
- def: [`recurrentgemma/jax/modules.py:633`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L633)
- doc: MLP block.
- signature: `class MLPBlock(nn.Module):`
- members:
  - `__call__(self, x: at.Activations)` — [`L680`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L680) — Calls the MLP block. — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `out_kernel_init(self)` — [`L652`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L652) — Initialization of the kernel for the last layer of the block.
  - `setup(self)` — [`L660`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L660)
  - `dtype` — [`L648`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L648)
  - `expanded_width` — [`L646`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L646)
  - `ffw_down` — [`L670`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L670)
  - `ffw_up` — [`L662`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L662)
  - `final_w_init_variance_scale` — [`L647`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L647)
  - `param_dtype` — [`L649`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L649) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `width` — [`L645`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L645)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`dtype`](array_typing.md#dtype), [`Activations`](array_typing.md#Activations), [`param_dtype`](layers.md#Einsum.param_dtype), [`dtype`](layers.md#Einsum.dtype), [`b_shape`](layers.md#Einsum.b_shape), [`eqn`](layers.md#Einsum.eqn), [`w_shape`](layers.md#Einsum.w_shape), [`Einsum`](layers.md#Einsum)
- used by: [`mlp`](modules.md#ResidualBlock.mlp), [`test_numerically_to_jax`](../torch/modules_test.md#MLPBlockTest.test_numerically_to_jax)

### `RecurrentBlock`  ·  implements/extends Module
- def: [`recurrentgemma/jax/modules.py:445`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L445) — documented in [recurrentgemma-jax-modules](../../../concepts/recurrentgemma-jax-modules.md)
- doc: Griffin and Hawk's recurrent block.
- signature: `class RecurrentBlock(nn.Module):`
- members:
  - `init_cache(cls, batch_size: int, lru_width: int, dtype: at.dtype, conv1d_temporal_width: int = 4)` — [`L611`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L611) — Initializes an empty RG-LRU and Conv1D cache for the block. — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `kernel_init(self)` — [`L476`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L476) — Initialization of the kernel for the linear x and y layers of the block.
  - `out_kernel_init(self)` — [`L485`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L485) — Initialization of the kernel for the last layer of the block.
  - `setup(self)` — [`L493`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L493)
  - `conv1d_temporal_width` — [`L467`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L467)
  - `conv_1d` — [`L518`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L518) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `dtype` — [`L470`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L470) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `final_w_init_variance_scale` — [`L468`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L468)
  - `linear_out` — [`L511`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L511)
  - `linear_x` — [`L504`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L504)
  - `linear_y` — [`L497`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L497)
  - `lru` — [`L525`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L525) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
  - `lru_only_real` — [`L472`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L472)
  - `lru_width` — [`L465`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L465)
  - `min_rad` — [`L473`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L473)
  - `num_heads` — [`L464`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L464)
  - `param_dtype` — [`L471`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L471) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `scan_sharding_spec` — [`L469`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L469)
  - `scan_type` — [`L466`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L466) — documented in [recurrentgemma-common](../../../concepts/recurrentgemma-common.md)
  - `width` — [`L463`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L463)
- protocol/private: `__call__`[`L538`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L538)
- uses (calls/refs, reference-scoped): [`RGLRU`](layers.md#RGLRU), [`Conv1D`](layers.md#Conv1D), [`ScanType`](../common.md#ScanType), [`typed`](array_typing.md#typed), [`dtype`](array_typing.md#dtype), [`SegmentPos`](array_typing.md#SegmentPos), [`Activations`](array_typing.md#Activations), [`AUTO`](../common.md#ScanType.AUTO), [`RecurrentBlockCache`](modules.md#RecurrentBlockCache), [`conv1d_state`](modules.md#RecurrentBlockCache.conv1d_state), [`only_real`](layers.md#RGLRU.only_real), [`scan_type`](layers.md#RGLRU.scan_type), [`temporal_width`](layers.md#Conv1D.temporal_width), [`dtype`](layers.md#RGLRU.dtype), [`param_dtype`](layers.md#RGLRU.param_dtype), [`width`](layers.md#RGLRU.width), [`ShardingSpec`](scan.md#ShardingSpec), [`param_dtype`](layers.md#Conv1D.param_dtype), [`rg_lru_state`](modules.md#RecurrentBlockCache.rg_lru_state), [`init_cache`](layers.md#Conv1D.init_cache), [`min_rad`](layers.md#RGLRU.min_rad), [`num_heads`](layers.md#RGLRU.num_heads), [`dtype`](layers.md#Conv1D.dtype), [`scan_sharding_spec`](layers.md#RGLRU.scan_sharding_spec), [`width`](layers.md#Conv1D.width), [`init_cache`](layers.md#RGLRU.init_cache)
- used by: [`recurrent_block`](modules.md#ResidualBlock.recurrent_block), [`test_numerically_to_jax`](../torch/modules_test.md#RecurrentBlockTest.test_numerically_to_jax), [`init_cache`](modules.md#ResidualBlock.init_cache), [`test_recurrent_block_output_shapes`](modules_test.md#RecurrentBlockTest.test_recurrent_block_output_shapes), [`RecurrentBlock`](__init__.md#RecurrentBlock)

### `RecurrentBlockCache`  ·  implements/extends NamedTuple
- def: [`recurrentgemma/jax/modules.py:34`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L34)
- doc: The cache for a recurrent block.
- signature: `class RecurrentBlockCache(NamedTuple):`
- members:
  - `conv1d_state` — [`L38`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L38) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `rg_lru_state` — [`L37`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L37)
- uses (calls/refs, reference-scoped): [`typed`](array_typing.md#typed), [`RNNState`](array_typing.md#RNNState), [`Conv1DState`](array_typing.md#Conv1DState)
- used by: [`test_griffin_output_shape`](griffin_test.md#GriffinTest.test_griffin_output_shape), [`RecurrentBlock`](modules.md#RecurrentBlock), [`init_cache`](modules.md#RecurrentBlock.init_cache), [`ResidualBlockCache`](modules.md#ResidualBlockCache), [`__call__`](modules.md#RecurrentBlock.__call__), [`RecurrentBlockCache`](__init__.md#RecurrentBlockCache)

### `ResidualBlock`  ·  implements/extends Module
- def: [`recurrentgemma/jax/modules.py:696`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L696) — documented in [recurrentgemma-jax-griffin](../../../concepts/recurrentgemma-jax-griffin.md)
- doc: Griffin and Hawk's residual block.
- signature: `class ResidualBlock(nn.Module):`
- members:
  - `init_cache(cls, batch_size: int, width: int, num_heads: int, attention_window_size: int, temporal_block_type: common.TemporalBlockType, dtype: at.dtype, lru_width: int | None = None, conv1d_temporal_width: int = 4)` — [`L858`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L858) — Initializes an empty cache for the block. — documented in [recurrentgemma-jax-griffin](../../../concepts/recurrentgemma-jax-griffin.md)
  - `setup(self)` — [`L735`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L735)
  - `temporal_block(self)` — [`L787`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L787) — Alias for the temporal block. — documented in [recurrentgemma-jax-modules](../../../concepts/recurrentgemma-jax-modules.md)
  - `attention_block` — [`L761`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L761) — documented in [recurrentgemma-jax-modules](../../../concepts/recurrentgemma-jax-modules.md)
  - `attention_window_size` — [`L722`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L722)
  - `channel_pre_norm` — [`L771`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L771)
  - `conv1d_temporal_width` — [`L726`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L726)
  - `dtype` — [`L729`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L729) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `final_w_init_variance_scale` — [`L727`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L727)
  - `lru_only_real` — [`L731`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L731)
  - `lru_width` — [`L724`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L724)
  - `min_rad` — [`L732`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L732)
  - `mlp` — [`L777`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L777)
  - `mlp_expanded_width` — [`L720`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L720)
  - `num_heads` — [`L721`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L721)
  - `param_dtype` — [`L730`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L730) — documented in [recurrentgemma-jax-array_typing](../../../concepts/recurrentgemma-jax-array_typing.md)
  - `recurrent_block` — [`L745`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L745) — documented in [recurrentgemma-common](../../../concepts/recurrentgemma-common.md)
  - `scan_sharding_spec` — [`L728`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L728)
  - `scan_type` — [`L725`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L725)
  - `temporal_block_type` — [`L723`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L723) — documented in [recurrentgemma-common](../../../concepts/recurrentgemma-common.md)
  - `temporal_pre_norm` — [`L737`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L737)
  - `use_mlp` — [`L733`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L733)
  - `width` — [`L719`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L719)
- protocol/private: `__call__`[`L800`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L800)
- uses (calls/refs, reference-scoped): [`LocalAttentionBlock`](modules.md#LocalAttentionBlock), [`TemporalBlockType`](../common.md#TemporalBlockType), [`ScanType`](../common.md#ScanType), [`typed`](array_typing.md#typed), [`RecurrentBlock`](modules.md#RecurrentBlock), [`RECURRENT`](../common.md#TemporalBlockType.RECURRENT), [`dtype`](array_typing.md#dtype), [`SegmentPos`](array_typing.md#SegmentPos), [`ATTENTION`](../common.md#TemporalBlockType.ATTENTION), [`Activations`](array_typing.md#Activations), [`init_cache`](modules.md#RecurrentBlock.init_cache), [`ResidualBlockCache`](modules.md#ResidualBlockCache), [`init_cache`](modules.md#LocalAttentionBlock.init_cache), [`AUTO`](../common.md#ScanType.AUTO), [`param_dtype`](modules.md#RecurrentBlock.param_dtype), [`dtype`](modules.md#RecurrentBlock.dtype), [`param_dtype`](modules.md#LocalAttentionBlock.param_dtype), [`scan_type`](modules.md#RecurrentBlock.scan_type), [`ShardingSpec`](scan.md#ShardingSpec), [`dtype`](modules.md#LocalAttentionBlock.dtype), [`param_dtype`](layers.md#RMSNorm.param_dtype), [`num_heads`](modules.md#LocalAttentionBlock.num_heads), [`width`](modules.md#LocalAttentionBlock.width), [`window_size`](modules.md#LocalAttentionBlock.window_size), [`dtype`](layers.md#RMSNorm.dtype), [`param_dtype`](modules.md#MLPBlock.param_dtype), [`width`](layers.md#RMSNorm.width), [`dtype`](modules.md#MLPBlock.dtype), [`RMSNorm`](layers.md#RMSNorm), [`width`](modules.md#RecurrentBlock.width), [`scan_sharding_spec`](modules.md#RecurrentBlock.scan_sharding_spec), [`expanded_width`](modules.md#MLPBlock.expanded_width), [`lru_width`](modules.md#RecurrentBlock.lru_width), [`num_heads`](modules.md#RecurrentBlock.num_heads), [`width`](modules.md#MLPBlock.width), [`conv1d_temporal_width`](modules.md#RecurrentBlock.conv1d_temporal_width), [`MLPBlock`](modules.md#MLPBlock), [`final_w_init_variance_scale`](modules.md#LocalAttentionBlock.final_w_init_variance_scale), [`final_w_init_variance_scale`](modules.md#MLPBlock.final_w_init_variance_scale), [`final_w_init_variance_scale`](modules.md#RecurrentBlock.final_w_init_variance_scale)  (+2 more)
- used by: [`blocks`](griffin.md#Griffin.blocks), [`test_numerically_to_jax`](../torch/modules_test.md#ResidualBlockTest.test_numerically_to_jax), [`embedder`](griffin.md#Griffin.embedder), [`init_cache`](griffin.md#Griffin.init_cache), [`test_recurrent_block_output_shapes`](modules_test.md#RecurrentBlockTest.test_recurrent_block_output_shapes), [`ResidualBlock`](__init__.md#ResidualBlock)

## Functions
- `_apply_rope(inputs: at.Keys | at.Queries, positions: at.SegmentPos, max_wavelength: int = _MAX_WAVELENGTH)` — [`L54`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L54) — Applies RoPE to the first half of inputs. — documented in [recurrentgemma-jax-modules](../../../concepts/recurrentgemma-jax-modules.md)
- `_attention_cache_from_prompt(keys: at.Keys, values: at.Values, segment_pos: at.SegmentPos, window_size: int)` — [`L219`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L219) — Creates a new cache from a prompt. — documented in [recurrentgemma-jax-modules](../../../concepts/recurrentgemma-jax-modules.md)
- `_compute_cache_mask(seq_len: int, cache_num_tokens: at.NumTokens, window_size: int)` — [`L153`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L153) — Computes the mask when there a KV-cache is present. — documented in [recurrentgemma-jax-modules](../../../concepts/recurrentgemma-jax-modules.md)
- `_compute_causal_mask(q_positions: jax.Array, k_positions: jax.Array, window_size: int, q_segment_ids: at.QuerySegmentIds | None, k_segment_ids: at.KeySegmentIds | None)` — [`L90`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L90) — Computes the causal mask for local attention.
- `_compute_forward_pass_mask(segment_pos: at.SegmentPos, window_size: int)` — [`L130`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L130) — Compute the forward pass mask. — documented in [recurrentgemma-jax-modules](../../../concepts/recurrentgemma-jax-modules.md)
- `_update_attention_cache(keys: at.Keys, values: at.Values, segment_pos: at.SegmentPos, cache: AttentionBlockCache)` — [`L183`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L183) — Updates the cache with the new keys and values. — documented in [recurrentgemma-jax-modules](../../../concepts/recurrentgemma-jax-modules.md)

## Module values
- `ResidualBlockCache` — [`L50`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L50)
- `_MAX_WAVELENGTH` — [`L29`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L29)
- `_MIN_LOGITS_VALUE` — [`L28`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L28)
- `_vmap_cache_roll` — [`L30`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/modules.py#L30)

