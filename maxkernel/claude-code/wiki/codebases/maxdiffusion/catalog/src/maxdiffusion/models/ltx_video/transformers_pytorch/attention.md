---
title: 'Module: src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.transformers_pytorch.attention`/
symbols:
  AttnProcessor2_0.__call__: AttnProcessor2_0#__call__().
  AttnProcessor.__call__: AttnProcessor#__call__().
  BasicTransformerBlock.forward: BasicTransformerBlock#forward().
  Attention.to_out: Attention#to_out.
  Attention.get_processor: Attention#get_processor().
  Attention.to_v: Attention#to_v.
  Attention.norm_cross: Attention#norm_cross.
  Attention.to_q: Attention#to_q.
  Attention.to_k: Attention#to_k.
  BasicTransformerBlock.set_use_tpu_flash_attention: BasicTransformerBlock#set_use_tpu_flash_attention().
  Attention.k_norm: Attention#k_norm.
  Attention.processor: Attention#processor.
  Attention.forward: Attention#forward().
  BasicTransformerBlock.adaptive_norm: BasicTransformerBlock#adaptive_norm.
  Attention.heads: Attention#heads.
  Attention.set_processor: Attention#set_processor().
  Attention.get_attention_scores: Attention#get_attention_scores().
  Attention.group_norm: Attention#group_norm.
  Attention.spatial_norm: Attention#spatial_norm.
  FeedForward.net: FeedForward#net.
  BasicTransformerBlock.attn2_norm: BasicTransformerBlock#attn2_norm.
  Attention.inner_dim: Attention#inner_dim.
  Attention.only_cross_attention: Attention#only_cross_attention.
  BasicTransformerBlock.attn2: BasicTransformerBlock#attn2.
  Attention.add_k_proj: Attention#add_k_proj.
  Attention.head_to_batch_dim: Attention#head_to_batch_dim().
  BasicTransformerBlock.set_chunk_feed_forward: BasicTransformerBlock#set_chunk_feed_forward().
  Attention.cross_attention_dim: Attention#cross_attention_dim.
  BasicTransformerBlock.attn1: BasicTransformerBlock#attn1.
  BasicTransformerBlock.ff: BasicTransformerBlock#ff.
  Attention.scale: Attention#scale.
  Attention.set_use_tpu_flash_attention: Attention#set_use_tpu_flash_attention().
  Attention.prepare_attention_mask: Attention#prepare_attention_mask().
  Attention.norm_encoder_hidden_states: Attention#norm_encoder_hidden_states().
  Attention.q_norm: Attention#q_norm.
  Attention: Attention#
  Attention.add_v_proj: Attention#add_v_proj.
  Attention.batch_to_head_dim: Attention#batch_to_head_dim().
  logger: logger.
  BasicTransformerBlock._chunk_size: BasicTransformerBlock#_chunk_size.
  Attention.use_tpu_flash_attention: Attention#use_tpu_flash_attention.
  Attention.added_kv_proj_dim: Attention#added_kv_proj_dim.
  Attention.__init__: Attention#__init__().
  FeedForward.forward: FeedForward#forward().
  BasicTransformerBlock.scale_shift_table: BasicTransformerBlock#scale_shift_table.
  BasicTransformerBlock._chunk_dim: BasicTransformerBlock#_chunk_dim.
  Attention.rescale_output_factor: Attention#rescale_output_factor.
  Attention.residual_connection: Attention#residual_connection.
  Attention.apply_rotary_emb: Attention#apply_rotary_emb().
  AttnProcessor: AttnProcessor#
  BasicTransformerBlock.only_cross_attention: BasicTransformerBlock#only_cross_attention.
  BasicTransformerBlock.use_tpu_flash_attention: BasicTransformerBlock#use_tpu_flash_attention.
  BasicTransformerBlock.norm1: BasicTransformerBlock#norm1.
  BasicTransformerBlock.norm2: BasicTransformerBlock#norm2.
  Attention.upcast_attention: Attention#upcast_attention.
  Attention.upcast_softmax: Attention#upcast_softmax.
  Attention.out_dim: Attention#out_dim.
  Attention.use_rope: Attention#use_rope.
  Attention.scale_qk: Attention#scale_qk.
  AttnProcessor2_0: AttnProcessor2_0#
  FeedForward: FeedForward#
  BasicTransformerBlock: BasicTransformerBlock#
  BasicTransformerBlock.__init__: BasicTransformerBlock#__init__().
  Attention.query_dim: Attention#query_dim.
  Attention.use_bias: Attention#use_bias.
  Attention.is_cross_attention: Attention#is_cross_attention.
  Attention.dropout: Attention#dropout.
  Attention.fused_projections: Attention#fused_projections.
  Attention._from_deprecated_attn_block: Attention#_from_deprecated_attn_block.
  Attention.sliceable_head_dim: Attention#sliceable_head_dim.
  Attention.linear_cls: Attention#linear_cls.
  AttnProcessor2_0.__init__: AttnProcessor2_0#__init__().
  FeedForward.__init__: FeedForward#__init__().
---
# Module: [`src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py)

## Classes
### `Attention`
- def: [`src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py:312`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L312)
- members:
  - `apply_rotary_emb(input_tensor: torch.Tensor, freqs_cis: Tuple[torch.FloatTensor, torch.FloatTensor])` — [`L863`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L863)
  - `batch_to_head_dim(self, tensor: torch.Tensor)` — [`L675`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L675) — Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
  - `forward(self, hidden_states: torch.FloatTensor, freqs_cis: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None, encoder_hidden_states: Optional[torch.FloatTensor] = None, attention_mask: Optional[torch.FloatTensor] = None, skip_layer_mask: Optional[torch.Tensor] = None, skip_layer_strategy: Optional[SkipLayerStrategy] = None, **cross_attention_kwargs)` — [`L621`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L621) — The forward method of the `Attention` class.
  - `get_attention_scores(self, query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor = None)` — [`L720`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L720) — Compute the attention scores.
  - `get_processor(self, return_deprecated_lora: bool = False)` — [`L531`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L531) — Get the attention processor in use.
  - `head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3)` — [`L692`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L692) — Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
  - `norm_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor)` — [`L833`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L833) — Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
  - `prepare_attention_mask(self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3)` — [`L774`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L774) — Prepare the attention mask for the attention computation.
  - `set_processor(self, processor: AttnProcessor)` — [`L511`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L511) — Set the attention processor to use.
  - `set_use_tpu_flash_attention(self)` — [`L505`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L505) — Function sets the flag in this object. The flag will enforce the usage of TPU attention kernel.
  - `add_k_proj` — [`L490`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L490)
  - `add_v_proj` — [`L491`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L491)
  - `added_kv_proj_dim` — [`L434`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L434)
  - `cross_attention_dim` — [`L398`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L398)
  - `dropout` — [`L403`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L403)
  - `fused_projections` — [`L404`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L404)
  - `group_norm` — [`L443`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L443)
  - `heads` — [`L428`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L428)
  - `inner_dim` — [`L394`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L394)
  - `is_cross_attention` — [`L397`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L397)
  - `k_norm` — [`L418`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L418)
  - `linear_cls` — [`L478`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L478) — ---
  - `norm_cross` — [`L453`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L453)
  - `only_cross_attention` — [`L435`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L435)
  - `out_dim` — [`L405`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L405)
  - `processor` — [`L529`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L529)
  - `q_norm` — [`L417`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L417)
  - `query_dim` — [`L395`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L395)
  - `rescale_output_factor` — [`L401`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L401)
  - `residual_connection` — [`L402`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L402)
  - `scale` — [`L414`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L414)
  - `scale_qk` — [`L413`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L413)
  - `sliceable_head_dim` — [`L432`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L432)
  - `spatial_norm` — [`L448`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L448)
  - `to_k` — [`L483`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L483)
  - `to_out` — [`L493`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L493)
  - `to_q` — [`L479`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L479)
  - `to_v` — [`L484`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L484)
  - `upcast_attention` — [`L399`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L399)
  - `upcast_softmax` — [`L400`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L400)
  - `use_bias` — [`L396`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L396)
  - `use_rope` — [`L407`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L407)
  - `use_tpu_flash_attention` — [`L406`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L406)
- protocol/private: `__init__`[`L365`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L365), `_from_deprecated_attn_block`[`L411`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L411)
- uses (calls/refs, reference-scoped): [`__call__`](attention.md#AttnProcessor.__call__), [`SkipLayerStrategy`](../utils/skip_layer_strategy.md#SkipLayerStrategy), [`logger`](attention.md#logger), [`AttnProcessor`](attention.md#AttnProcessor), [`AttnProcessor2_0`](attention.md#AttnProcessor2_0)
- used by: [`__call__`](attention.md#AttnProcessor2_0.__call__), [`__call__`](attention.md#AttnProcessor.__call__), [`set_use_tpu_flash_attention`](attention.md#BasicTransformerBlock.set_use_tpu_flash_attention), [`attn2`](attention.md#BasicTransformerBlock.attn2), [`attn1`](attention.md#BasicTransformerBlock.attn1)

### `AttnProcessor`
- def: [`src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py:1020`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L1020)
- protocol/private: `__call__`[`L1025`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L1025)
- uses (calls/refs, reference-scoped): [`to_out`](attention.md#Attention.to_out), [`to_v`](attention.md#Attention.to_v), [`norm_cross`](attention.md#Attention.norm_cross), [`to_q`](attention.md#Attention.to_q), [`to_k`](attention.md#Attention.to_k), [`k_norm`](attention.md#Attention.k_norm), [`get_attention_scores`](attention.md#Attention.get_attention_scores), [`group_norm`](attention.md#Attention.group_norm), [`spatial_norm`](attention.md#Attention.spatial_norm), [`head_to_batch_dim`](attention.md#Attention.head_to_batch_dim), [`norm_encoder_hidden_states`](attention.md#Attention.norm_encoder_hidden_states), [`prepare_attention_mask`](attention.md#Attention.prepare_attention_mask), [`Attention`](attention.md#Attention), [`q_norm`](attention.md#Attention.q_norm), [`batch_to_head_dim`](attention.md#Attention.batch_to_head_dim), [`rescale_output_factor`](attention.md#Attention.rescale_output_factor), [`residual_connection`](attention.md#Attention.residual_connection)
- used by: [`forward`](attention.md#Attention.forward), [`set_processor`](attention.md#Attention.set_processor), [`__init__`](attention.md#Attention.__init__)

### `AttnProcessor2_0`
- def: [`src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py:880`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L880)
- protocol/private: `__call__`[`L888`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L888), `__init__`[`L885`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L885)
- uses (calls/refs, reference-scoped): [`to_out`](attention.md#Attention.to_out), [`to_v`](attention.md#Attention.to_v), [`norm_cross`](attention.md#Attention.norm_cross), [`to_q`](attention.md#Attention.to_q), [`to_k`](attention.md#Attention.to_k), [`SkipLayerStrategy`](../utils/skip_layer_strategy.md#SkipLayerStrategy), [`k_norm`](attention.md#Attention.k_norm), [`heads`](attention.md#Attention.heads), [`group_norm`](attention.md#Attention.group_norm), [`spatial_norm`](attention.md#Attention.spatial_norm), [`norm_encoder_hidden_states`](attention.md#Attention.norm_encoder_hidden_states), [`prepare_attention_mask`](attention.md#Attention.prepare_attention_mask), [`scale`](attention.md#Attention.scale), [`Attention`](attention.md#Attention), [`q_norm`](attention.md#Attention.q_norm), [`use_tpu_flash_attention`](attention.md#Attention.use_tpu_flash_attention), [`Residual`](../utils/skip_layer_strategy.md#SkipLayerStrategy.Residual), [`apply_rotary_emb`](attention.md#Attention.apply_rotary_emb), [`rescale_output_factor`](attention.md#Attention.rescale_output_factor), [`residual_connection`](attention.md#Attention.residual_connection), [`AttentionSkip`](../utils/skip_layer_strategy.md#SkipLayerStrategy.AttentionSkip), [`AttentionValues`](../utils/skip_layer_strategy.md#SkipLayerStrategy.AttentionValues), [`use_rope`](attention.md#Attention.use_rope)
- used by: [`to_out`](attention.md#Attention.to_out)

### `BasicTransformerBlock`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py:54`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L54)
- doc: A basic Transformer block.
- signature: `class BasicTransformerBlock(nn.Module):`
- members:
  - `forward(self, hidden_states: torch.FloatTensor, freqs_cis: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None, attention_mask: Optional[torch.FloatTensor] = None, encoder_hidden_states: Optional[torch.FloatTensor] = None, encoder_attention_mask: Optional[torch.FloatTensor] = None, timestep: Optional[torch.LongTensor] = None, cross_attention_kwargs: Dict[str, Any] = None, class_labels: Optional[torch.LongTensor] = None, skip_layer_mask: Optional[torch.Tensor] = None, skip_layer_strategy: Optional[SkipLayerStrategy] = None)` — [`L204`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L204) — documented in [maxdiffusion-models-ltx_video-transformers_pytorch-attention](../../../../../../concepts/maxdiffusion-models-ltx_video-transformers_pytorch-attention.md)
  - `set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0)` — [`L199`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L199)
  - `set_use_tpu_flash_attention(self)` — [`L190`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L190) — Function sets the flag in this object and propagates down the children. The flag will enforce the usage of TPU
  - `adaptive_norm` — [`L122`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L122)
  - `attn1` — [`L133`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L133)
  - `attn2` — [`L149`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L149)
  - `attn2_norm` — [`L164`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L164)
  - `ff` — [`L172`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L172)
  - `norm1` — [`L131`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L131)
  - `norm2` — [`L169`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L169)
  - `only_cross_attention` — [`L120`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L120)
  - `scale_shift_table` — [`L184`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L184)
  - `use_tpu_flash_attention` — [`L121`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L121)
- protocol/private: `__init__`[`L93`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L93), `_chunk_dim`[`L188`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L188), `_chunk_size`[`L187`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L187)
- uses (calls/refs, reference-scoped): [`SkipLayerStrategy`](../utils/skip_layer_strategy.md#SkipLayerStrategy), [`set_use_tpu_flash_attention`](attention.md#Attention.set_use_tpu_flash_attention), [`Attention`](attention.md#Attention), [`logger`](attention.md#logger), [`FeedForward`](attention.md#FeedForward), [`TransformerBlock`](../utils/skip_layer_strategy.md#SkipLayerStrategy.TransformerBlock)

### `FeedForward`
- def: [`src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py:1093`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L1093)
- members:
  - `forward(self, hidden_states: torch.Tensor, scale: float = 1)` — [`L1146`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L1146)
  - `net` — [`L1135`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L1135)
- protocol/private: `__init__`[`L1107`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L1107)
- used by: [`ff`](attention.md#BasicTransformerBlock.ff)

## Module values
- `logger` — [`L50`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/attention.py#L50)

