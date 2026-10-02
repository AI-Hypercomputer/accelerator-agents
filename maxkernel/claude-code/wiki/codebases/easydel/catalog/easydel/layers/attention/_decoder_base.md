---
title: 'Module: easydel/layers/attention/_decoder_base.py'
type: catalog
provenance: extracted
module: easydel/layers/attention/_decoder_base.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.attention._decoder_base`/
symbols:
  BaseDecoderLayer.standard_decoder_layer_call: BaseDecoderLayer#standard_decoder_layer_call().
  BaseDecoderLayer.pre_norm_residual_attn: BaseDecoderLayer#pre_norm_residual_attn().
  BaseDecoderLayer: BaseDecoderLayer#
  BaseDecoderLayer.pre_norm_residual_mlp: BaseDecoderLayer#pre_norm_residual_mlp().
  block_wise_ffn: block_wise_ffn().
  BaseDecoderLayer.apply_output_sharding: BaseDecoderLayer#apply_output_sharding().
---
# Module: [`easydel/layers/attention/_decoder_base.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_decoder_base.py)

## Classes
### `BaseDecoderLayer`
- def: [`easydel/layers/attention/_decoder_base.py:126`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_decoder_base.py#L126)
- doc: Utility class providing common decoder layer patterns.
- signature: `class BaseDecoderLayer:`
- members:
  - `apply_output_sharding(hidden_states: Float[Array, "batch seq_len hidden_dim"], partition_manager)` — [`L280`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_decoder_base.py#L280) — Apply sharding to decoder layer output.
  - `pre_norm_residual_attn(hidden_states: Float[Array, "batch seq_len hidden_dim"], attention_module: Callable, norm_module: Callable, mask_info: MaskInfo | None, position_ids: Int[Array, "batch seq_len"], mode: common_types.RUNTIME_MODE_TYPES, cache_view: TransformerCacheView | RaggedPagesCacheView | UnifiedAttentionCacheView | None = None, cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None, output_attentions: bool = False, frequencies: Float[Array, "seq_len head_dim"] | None = None, checkpoint_names: tuple[str, str] = ("norm", "residual"))` — [`L173`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_decoder_base.py#L173) — Apply attention with pre-norm residual connection.
  - `pre_norm_residual_mlp(hidden_states: Float[Array, "batch seq_len hidden_dim"], mlp_module: Callable, norm_module: Callable, use_scan: bool = False, scan_chunk_size: int = 1024, checkpoint_names: tuple[str, str] = ("norm", "residual"))` — [`L234`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_decoder_base.py#L234) — Apply MLP with pre-norm residual connection.
  - `standard_decoder_layer_call(hidden_states: Float[Array, "batch seq_len hidden_dim"], attention_module: Callable, mlp_module: Callable, input_norm: Callable, post_attn_norm: Callable, mask_info: MaskInfo | None, position_ids: Int[Array, "batch seq_len"], mode: common_types.RUNTIME_MODE_TYPES, partition_manager, cache_view: TransformerCacheView | RaggedPagesCacheView | UnifiedAttentionCacheView | None = None, cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None, output_attentions: bool = False, frequencies: Float[Array, "seq_len head_dim"] | None = None, use_scan_mlp: bool = False, scan_mlp_chunk_size: int = 1024)` — [`L299`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_decoder_base.py#L299) — Complete standard decoder layer forward pass.
- uses (calls/refs, reference-scoped): [`OperationsMetadata`](../../caching/_abstracts.md#OperationsMetadata), [`RaggedPagesMetadata`](../../caching/ragged_page/cache.md#RaggedPagesMetadata), [`TransformerMetadata`](../../caching/transformer/cache.md#TransformerMetadata), [`RaggedPagesCacheView`](../../caching/ragged_page/cache.md#RaggedPagesCacheView), [`TransformerCacheView`](../../caching/transformer/cache.md#TransformerCacheView), [`DecoderLayerOutput`](../../infra/modeling_outputs.md#DecoderLayerOutput), [`AttentionLayerOutput`](../../infra/modeling_outputs.md#AttentionLayerOutput), [`attention_output`](../../infra/modeling_outputs.md#AttentionLayerOutput.attention_output), [`cache_view`](../../infra/modeling_outputs.md#AttentionLayerOutput.cache_view), [`attention_weight`](../../infra/modeling_outputs.md#AttentionLayerOutput.attention_weight), [`UnifiedAttentionCacheView`](../../caching/unified_attention/cache.md#UnifiedAttentionCacheView), [`block_wise_ffn`](_decoder_base.md#block_wise_ffn)

## Functions
- `block_wise_ffn(mlp_module: Callable, inputs: Float[Array, "batch_size seq_len hidden_dim"], chunk_size: int = 1024)` — [`L382`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_decoder_base.py#L382) — Apply MLP block-wise for memory efficiency.

