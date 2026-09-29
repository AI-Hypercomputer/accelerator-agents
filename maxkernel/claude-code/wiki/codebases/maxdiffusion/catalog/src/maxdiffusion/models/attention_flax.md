---
title: 'Module: src/maxdiffusion/models/attention_flax.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/attention_flax.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.attention_flax`/
symbols:
  FlaxTransformer2DModel.transformer_blocks: FlaxTransformer2DModel#transformer_blocks.
  FlaxWanAttention.__call__: FlaxWanAttention#__call__().
  FlaxBasicTransformerBlock.attn1: FlaxBasicTransformerBlock#attn1.
  NNXAttentionOp.apply_attention: NNXAttentionOp#apply_attention().
  FlaxAttention.attention_op: FlaxAttention#attention_op.
  FlaxBasicTransformerBlock.attn2: FlaxBasicTransformerBlock#attn2.
  FlaxFluxAttention.attention_op: FlaxFluxAttention#attention_op.
  AttentionOp.apply_attention: AttentionOp#apply_attention().
  FlaxFluxAttention.__call__: FlaxFluxAttention#__call__().
  Array: Array.
  FlaxAttention.__call__: FlaxAttention#__call__().
  _ulysses_ring_attention: _ulysses_ring_attention().
  AxisNames: AxisNames.
  BATCH: BATCH.
  _ulysses_ring_custom_attention: _ulysses_ring_custom_attention().
  FlaxWanAttention.compute_kv: FlaxWanAttention#compute_kv().
  FlaxBasicTransformerBlock.ff: FlaxBasicTransformerBlock#ff.
  NNXAttentionOp.__init__: NNXAttentionOp#__init__().
  _tpu_flash_attention: _tpu_flash_attention().
  LENGTH: LENGTH.
  FlaxWanAttention.conditional_named_scope: FlaxWanAttention#conditional_named_scope().
  HEAD: HEAD.
  _ulysses_attention: _ulysses_attention().
  _tpu_flash_attention.wrap_flash_attention: _tpu_flash_attention().wrap_flash_attention().
  _cudnn_flash_attention: _cudnn_flash_attention().
  FlaxWanAttention.enable_jax_named_scopes: FlaxWanAttention#enable_jax_named_scopes.
  _pad_data_for_flash: _pad_data_for_flash().
  FlaxBasicTransformerBlock.__call__: FlaxBasicTransformerBlock#__call__().
  FlaxFeedForward.net_0: FlaxFeedForward#net_0.
  BlockSizes: BlockSizes.
  FlaxWanAttention.__init__: FlaxWanAttention#__init__().
  FlaxTransformer2DModel.__call__: FlaxTransformer2DModel#__call__().
  _apply_attention: _apply_attention().
  _ulysses_attention.wrap_ulysses_attention: _ulysses_attention().wrap_ulysses_attention().
  FlaxTransformer2DModel.norm: FlaxTransformer2DModel#norm.
  _reshape_data_for_flash: _reshape_data_for_flash().
  FlaxWanAttention.norm_added_k: FlaxWanAttention#norm_added_k.
  AttentionOp.setup: AttentionOp#setup().
  Mesh: Mesh.
  register_kernel: register_kernel().
  AttentionOp.axis_names_q: AttentionOp#axis_names_q.
  AttentionOp.axis_names_kv: AttentionOp#axis_names_kv.
  _select_flash_block_sizes: _select_flash_block_sizes().
  FlaxWanAttention.inner_dim: FlaxWanAttention#inner_dim.
  FlaxTransformer2DModel.proj_in: FlaxTransformer2DModel#proj_in.
  FlaxTransformer2DModel.proj_out: FlaxTransformer2DModel#proj_out.
  CONTEXT: CONTEXT.
  D_KV: D_KV.
  FlaxWanAttention.norm_k: FlaxWanAttention#norm_k.
  _create_internal_ulysses_ring_mesh: _create_internal_ulysses_ring_mesh().
  _reshape_heads_to_head_dim: _reshape_heads_to_head_dim().
  _ulysses_ring_custom_attention.wrap_ulysses_ring_attention: _ulysses_ring_custom_attention().wrap_ulysses_ring_attention().
  _apply_attention_dot: _apply_attention_dot().
  FlaxFluxAttention.dtype: FlaxFluxAttention#dtype.
  FlaxFluxAttention.query_axis_names: FlaxFluxAttention#query_axis_names.
  FlaxFluxAttention.key_axis_names: FlaxFluxAttention#key_axis_names.
  FlaxFluxAttention.value_axis_names: FlaxFluxAttention#value_axis_names.
  FlaxFluxAttention.qkv: FlaxFluxAttention#qkv.
  FlaxFluxAttention.encoder_qkv: FlaxFluxAttention#encoder_qkv.
  FlaxFluxAttention.proj_attn: FlaxFluxAttention#proj_attn.
  FlaxFluxAttention.encoder_proj_attn: FlaxFluxAttention#encoder_proj_attn.
  FlaxAttention.query_axis_names: FlaxAttention#query_axis_names.
  FlaxAttention.key_axis_names: FlaxAttention#key_axis_names.
  FlaxAttention.value_axis_names: FlaxAttention#value_axis_names.
  FlaxAttention.proj_attn: FlaxAttention#proj_attn.
  FlaxTransformer2DModel.hidden_state_axis_names: FlaxTransformer2DModel#hidden_state_axis_names.
  FlaxFeedForward.net_2: FlaxFeedForward#net_2.
  _reshape_heads_to_batch_dim: _reshape_heads_to_batch_dim().
  NNXSimpleFeedForward.__call__: NNXSimpleFeedForward#__call__().
  FlaxFluxAttention.weights_dtype: FlaxFluxAttention#weights_dtype.
  FlaxFluxAttention.out_axis_names: FlaxFluxAttention#out_axis_names.
  FlaxFluxAttention.setup: FlaxFluxAttention#setup().
  FlaxAttention.out_axis_names: FlaxAttention#out_axis_names.
  FlaxAttention.setup: FlaxAttention#setup().
  Quant: Quant.
  FlaxTransformer2DModel.dtype: FlaxTransformer2DModel#dtype.
  FlaxTransformer2DModel.weights_dtype: FlaxTransformer2DModel#weights_dtype.
  _reshape_batch_dim_to_heads: _reshape_batch_dim_to_heads().
  _ulysses_ring_attention.wrap_ulysses_ring_attention: _ulysses_ring_attention().wrap_ulysses_ring_attention().
  FlaxAttention.query: FlaxAttention#query.
  FlaxAttention.key: FlaxAttention#key.
  FlaxAttention.value: FlaxAttention#value.
  FlaxGEGLU.proj: FlaxGEGLU#proj.
  FlaxFluxAttention.heads: FlaxFluxAttention#heads.
  FlaxTransformer2DModel.precision: FlaxTransformer2DModel#precision.
  FlaxAttention.quant: FlaxAttention#quant.
  FlaxFluxAttention.query_norm: FlaxFluxAttention#query_norm.
  FlaxFluxAttention.key_norm: FlaxFluxAttention#key_norm.
  FlaxAttention.dtype: FlaxAttention#dtype.
  FlaxBasicTransformerBlock.dtype: FlaxBasicTransformerBlock#dtype.
  FlaxBasicTransformerBlock.weights_dtype: FlaxBasicTransformerBlock#weights_dtype.
  FlaxBasicTransformerBlock.norm1: FlaxBasicTransformerBlock#norm1.
  FlaxTransformer2DModel.use_linear_projection: FlaxTransformer2DModel#use_linear_projection.
  _replace_mesh_axis_names: _replace_mesh_axis_names().
  AttentionOp.dtype: AttentionOp#dtype.
  FlaxWanAttention.attention_op: FlaxWanAttention#attention_op.
  FlaxWanAttention.key: FlaxWanAttention#key.
  FlaxWanAttention.value: FlaxWanAttention#value.
  FlaxTransformer2DModel.flash_block_sizes: FlaxTransformer2DModel#flash_block_sizes.
  FlaxTransformer2DModel.quant: FlaxTransformer2DModel#quant.
  FlaxWanAttention: FlaxWanAttention#
  FlaxWanAttention.qk_norm: FlaxWanAttention#qk_norm.
  FlaxWanAttention.added_kv_proj_dim: FlaxWanAttention#added_kv_proj_dim.
  FlaxFluxAttention.encoder_query_norm: FlaxFluxAttention#encoder_query_norm.
  FlaxFluxAttention.encoder_key_norm: FlaxFluxAttention#encoder_key_norm.
  FlaxAttention.weights_dtype: FlaxAttention#weights_dtype.
  FlaxAttention.precision: FlaxAttention#precision.
  FlaxBasicTransformerBlock.norm2: FlaxBasicTransformerBlock#norm2.
  FlaxBasicTransformerBlock.norm3: FlaxBasicTransformerBlock#norm3.
  convert_to_tokamax_splash_config: convert_to_tokamax_splash_config().
  apply_rope: apply_rope().
  AttentionOp.mesh: AttentionOp#mesh.
  AttentionOp.flash_block_sizes: AttentionOp#flash_block_sizes.
  FlaxFluxAttention.flash_block_sizes: FlaxFluxAttention#flash_block_sizes.
  FlaxAttention.flash_block_sizes: FlaxAttention#flash_block_sizes.
  FlaxBasicTransformerBlock.flash_block_sizes: FlaxBasicTransformerBlock#flash_block_sizes.
  FlaxBasicTransformerBlock.quant: FlaxBasicTransformerBlock#quant.
  _maybe_aqt_einsum: _maybe_aqt_einsum().
  dot_product_kernel: dot_product_kernel().
  ulysses_custom_kernel: ulysses_custom_kernel().
  ulysses_ring_custom_kernel: ulysses_ring_custom_kernel().
  ulysses_ring_custom_bidir_kernel: ulysses_ring_custom_bidir_kernel().
  ulysses_kernel: ulysses_kernel().
  ulysses_ring_kernel: ulysses_ring_kernel().
  flash_kernel: flash_kernel().
  tokamax_flash_kernel: tokamax_flash_kernel().
  tokamax_ring_kernel: tokamax_ring_kernel().
  tokamax_ring_custom_kernel: tokamax_ring_custom_kernel().
  cudnn_flash_te_kernel: cudnn_flash_te_kernel().
  NNXAttentionOp: NNXAttentionOp#
  AttentionOp.heads: AttentionOp#heads.
  FlaxFluxAttention.dim_head: FlaxFluxAttention#dim_head.
  FlaxFluxAttention.mesh: FlaxFluxAttention#mesh.
  FlaxAttention.mesh: FlaxAttention#mesh.
  FlaxBasicTransformerBlock.dropout: FlaxBasicTransformerBlock#dropout.
  FlaxTransformer2DModel.n_heads: FlaxTransformer2DModel#n_heads.
  FlaxTransformer2DModel.d_head: FlaxTransformer2DModel#d_head.
  FlaxFeedForward.__call__: FlaxFeedForward#__call__().
  FlaxGEGLU.__call__: FlaxGEGLU#__call__().
  DType: DType.
  KV_LENGTH: KV_LENGTH.
  EMBED: EMBED.
  SELF_ATTN_HEAD: SELF_ATTN_HEAD.
  CROSS_ATTN_HEAD: CROSS_ATTN_HEAD.
  _flash_sequence_length: _flash_sequence_length().
  FlaxWanAttention.query: FlaxWanAttention#query.
  _unflatten_heads: _unflatten_heads().
  KERNEL_REGISTRY: KERNEL_REGISTRY.
  AttentionOp.attention_kernel: AttentionOp#attention_kernel.
  AttentionOp.scale: AttentionOp#scale.
  AttentionOp.dim_head: AttentionOp#dim_head.
  FlaxWanAttention.image_seq_len: FlaxWanAttention#image_seq_len.
  FlaxFluxAttention.query_dim: FlaxFluxAttention#query_dim.
  FlaxFluxAttention.attention_kernel: FlaxFluxAttention#attention_kernel.
  FlaxFluxAttention.precision: FlaxFluxAttention#precision.
  FlaxAttention.attention_kernel: FlaxAttention#attention_kernel.
  FlaxBasicTransformerBlock.precision: FlaxBasicTransformerBlock#precision.
  FlaxTransformer2DModel: FlaxTransformer2DModel#
  FlaxTransformer2DModel.depth: FlaxTransformer2DModel#depth.
  FlaxTransformer2DModel.use_memory_efficient_attention: FlaxTransformer2DModel#use_memory_efficient_attention.
  FlaxTransformer2DModel.split_head_dim: FlaxTransformer2DModel#split_head_dim.
  FlaxTransformer2DModel.attention_kernel: FlaxTransformer2DModel#attention_kernel.
  FlaxTransformer2DModel.flash_min_seq_length: FlaxTransformer2DModel#flash_min_seq_length.
  FlaxTransformer2DModel.mesh: FlaxTransformer2DModel#mesh.
  FlaxTransformer2DModel.norm_num_groups: FlaxTransformer2DModel#norm_num_groups.
  SELF_ATTN_Q_LENGTH: SELF_ATTN_Q_LENGTH.
  SELF_ATTN_KV_LENGTH: SELF_ATTN_KV_LENGTH.
  CROSS_ATTN_Q_LENGTH: CROSS_ATTN_Q_LENGTH.
  CROSS_ATTN_KV_LENGTH: CROSS_ATTN_KV_LENGTH.
  _check_attention_inputs: _check_attention_inputs().
  _tpu_flash_attention.wrap_flash_attention.ring_scan_body: _tpu_flash_attention().wrap_flash_attention().ring_scan_body().
  register_kernel.decorator: register_kernel().decorator().
  _query_chunk_attention: _query_chunk_attention().
  _query_chunk_attention.chunk_scanner: _query_chunk_attention().chunk_scanner().
  jax_memory_efficient_attention: jax_memory_efficient_attention().
  jax_memory_efficient_attention.chunk_scanner: jax_memory_efficient_attention().chunk_scanner().
  NNXSimpleFeedForward.act: NNXSimpleFeedForward#act.
  NNXAttentionOp.ulysses_shards: NNXAttentionOp#ulysses_shards.
  AttentionOp.quant: AttentionOp#quant.
  FlaxWanAttention.proj_attn: FlaxWanAttention#proj_attn.
  FlaxAttention.dropout_layer: FlaxAttention#dropout_layer.
  FlaxBasicTransformerBlock.dropout_layer: FlaxBasicTransformerBlock#dropout_layer.
  FlaxTransformer2DModel.dropout_layer: FlaxTransformer2DModel#dropout_layer.
  FlaxGEGLU.dropout_layer: FlaxGEGLU#dropout_layer.
  LOG2E: LOG2E.
  INTERNAL_RING_AXIS: INTERNAL_RING_AXIS.
  INTERNAL_ULYSSES_AXIS: INTERNAL_ULYSSES_AXIS.
  _reshape_data_for_cudnn_flash: _reshape_data_for_cudnn_flash().
  _extract_custom_block_sizes: _extract_custom_block_sizes().
  _build_padding_segment_ids: _build_padding_segment_ids().
  AttentionOp.use_memory_efficient_attention: AttentionOp#use_memory_efficient_attention.
  AttentionOp.split_head_dim: AttentionOp#split_head_dim.
  AttentionOp.flash_min_seq_length: AttentionOp#flash_min_seq_length.
  FlaxWanAttention.heads: FlaxWanAttention#heads.
  FlaxWanAttention.norm_q: FlaxWanAttention#norm_q.
  FlaxWanAttention.add_k_proj: FlaxWanAttention#add_k_proj.
  FlaxWanAttention.add_v_proj: FlaxWanAttention#add_v_proj.
  FlaxFluxAttention: FlaxFluxAttention#
  FlaxFluxAttention.qkv_bias: FlaxFluxAttention#qkv_bias.
  FlaxAttention.dim_head: FlaxAttention#dim_head.
  FlaxAttention.flash_min_seq_length: FlaxAttention#flash_min_seq_length.
  FlaxBasicTransformerBlock.dim: FlaxBasicTransformerBlock#dim.
  FlaxBasicTransformerBlock.use_memory_efficient_attention: FlaxBasicTransformerBlock#use_memory_efficient_attention.
  FlaxBasicTransformerBlock.split_head_dim: FlaxBasicTransformerBlock#split_head_dim.
  FlaxBasicTransformerBlock.attention_kernel: FlaxBasicTransformerBlock#attention_kernel.
  FlaxBasicTransformerBlock.flash_min_seq_length: FlaxBasicTransformerBlock#flash_min_seq_length.
  FlaxBasicTransformerBlock.mesh: FlaxBasicTransformerBlock#mesh.
  FlaxTransformer2DModel.in_channels: FlaxTransformer2DModel#in_channels.
  FlaxTransformer2DModel.only_cross_attention: FlaxTransformer2DModel#only_cross_attention.
  FlaxFeedForward.dim: FlaxFeedForward#dim.
  FlaxFeedForward.dtype: FlaxFeedForward#dtype.
  FlaxFeedForward.weights_dtype: FlaxFeedForward#weights_dtype.
  FlaxFeedForward.precision: FlaxFeedForward#precision.
  FlaxGEGLU.setup: FlaxGEGLU#setup().
  NNXSimpleFeedForward: NNXSimpleFeedForward#
  NNXAttentionOp.dpa_layer: NNXAttentionOp#dpa_layer.
  AttentionOp: AttentionOp#
  AttentionOp.float32_qk_product: AttentionOp#float32_qk_product.
  AttentionOp.dpa_layer: AttentionOp#dpa_layer.
  FlaxWanAttention.alignment: FlaxWanAttention#alignment.
  FlaxWanAttention.drop_out: FlaxWanAttention#drop_out.
  FlaxAttention: FlaxAttention#
  FlaxAttention.heads: FlaxAttention#heads.
  FlaxBasicTransformerBlock.n_heads: FlaxBasicTransformerBlock#n_heads.
  FlaxBasicTransformerBlock.d_head: FlaxBasicTransformerBlock#d_head.
  FlaxBasicTransformerBlock.only_cross_attention: FlaxBasicTransformerBlock#only_cross_attention.
  FlaxTransformer2DModel.dropout: FlaxTransformer2DModel#dropout.
  FlaxFeedForward.dropout: FlaxFeedForward#dropout.
  FlaxGEGLU.precision: FlaxGEGLU#precision.
  _coerce_tokamax_block_sizes: _coerce_tokamax_block_sizes().
  _reshape_data_from_cudnn_flash: _reshape_data_from_cudnn_flash().
  _replace_mesh_axis: _replace_mesh_axis().
  _query_chunk_attention.summarize_chunk: _query_chunk_attention().summarize_chunk().
  NNXSimpleFeedForward.net_0: NNXSimpleFeedForward#net_0.
  NNXSimpleFeedForward.net_2: NNXSimpleFeedForward#net_2.
  NNXAttentionOp.use_base2_exp: NNXAttentionOp#use_base2_exp.
  NNXAttentionOp.use_experimental_scheduler: NNXAttentionOp#use_experimental_scheduler.
  NNXAttentionOp.mesh: NNXAttentionOp#mesh.
  NNXAttentionOp.scale: NNXAttentionOp#scale.
  NNXAttentionOp.heads: NNXAttentionOp#heads.
  NNXAttentionOp.dim_head: NNXAttentionOp#dim_head.
  NNXAttentionOp.attention_kernel: NNXAttentionOp#attention_kernel.
  NNXAttentionOp.use_memory_efficient_attention: NNXAttentionOp#use_memory_efficient_attention.
  NNXAttentionOp.split_head_dim: NNXAttentionOp#split_head_dim.
  NNXAttentionOp.float32_qk_product: NNXAttentionOp#float32_qk_product.
  NNXAttentionOp.axis_names_q: NNXAttentionOp#axis_names_q.
  NNXAttentionOp.axis_names_kv: NNXAttentionOp#axis_names_kv.
  NNXAttentionOp.flash_min_seq_length: NNXAttentionOp#flash_min_seq_length.
  NNXAttentionOp.flash_block_sizes: NNXAttentionOp#flash_block_sizes.
  NNXAttentionOp.dtype: NNXAttentionOp#dtype.
  NNXAttentionOp.mask_padding_tokens: NNXAttentionOp#mask_padding_tokens.
  NNXAttentionOp.residual_checkpoint_name: NNXAttentionOp#residual_checkpoint_name.
  AttentionOp.use_base2_exp: AttentionOp#use_base2_exp.
  AttentionOp.use_experimental_scheduler: AttentionOp#use_experimental_scheduler.
  AttentionOp.ulysses_shards: AttentionOp#ulysses_shards.
  FlaxWanAttention._apply_rope: FlaxWanAttention#_apply_rope().
  FlaxFluxAttention.use_memory_efficient_attention: FlaxFluxAttention#use_memory_efficient_attention.
  FlaxFluxAttention.split_head_dim: FlaxFluxAttention#split_head_dim.
  FlaxFluxAttention.flash_min_seq_length: FlaxFluxAttention#flash_min_seq_length.
  FlaxAttention.query_dim: FlaxAttention#query_dim.
  FlaxAttention.dropout: FlaxAttention#dropout.
  FlaxAttention.use_memory_efficient_attention: FlaxAttention#use_memory_efficient_attention.
  FlaxAttention.split_head_dim: FlaxAttention#split_head_dim.
  FlaxBasicTransformerBlock: FlaxBasicTransformerBlock#
  FlaxGEGLU.dim: FlaxGEGLU#dim.
  FlaxGEGLU.dropout: FlaxGEGLU#dropout.
  FlaxGEGLU.dtype: FlaxGEGLU#dtype.
  FlaxGEGLU.weights_dtype: FlaxGEGLU#weights_dtype.
  FlaxFeedForward: FlaxFeedForward#
  FlaxGEGLU: FlaxGEGLU#
  NNXSimpleFeedForward.__init__: NNXSimpleFeedForward#__init__().
  NNXAttentionOp.quant: NNXAttentionOp#quant.
  FlaxWanAttention.dim_head: FlaxWanAttention#dim_head.
  FlaxWanAttention.query_axis_names: FlaxWanAttention#query_axis_names.
  FlaxWanAttention.key_axis_names: FlaxWanAttention#key_axis_names.
  FlaxWanAttention.value_axis_names: FlaxWanAttention#value_axis_names.
  FlaxWanAttention.out_axis_names: FlaxWanAttention#out_axis_names.
  FlaxFluxAttention.dropout: FlaxFluxAttention#dropout.
  FlaxBasicTransformerBlock.setup: FlaxBasicTransformerBlock#setup().
  FlaxTransformer2DModel.setup: FlaxTransformer2DModel#setup().
  FlaxFeedForward.setup: FlaxFeedForward#setup().
---
# Module: [`src/maxdiffusion/models/attention_flax.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py)

## Classes
### `AttentionOp`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/attention_flax.py:1722`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1722)
- signature: `class AttentionOp(nn.Module):`
- members:
  - `apply_attention(self, query: Array, key: Array, value: Array, attention_mask: Array = None)` — [`L1765`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1765)
  - `setup(self)` — [`L1741`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1741)
  - `attention_kernel` — [`L1724`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1724)
  - `axis_names_kv` — [`L1732`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1732)
  - `axis_names_q` — [`L1731`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1731)
  - `dim_head` — [`L1727`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1727)
  - `dpa_layer` — [`L1742`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1742)
  - `dtype` — [`L1735`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1735)
  - `flash_block_sizes` — [`L1734`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1734)
  - `flash_min_seq_length` — [`L1733`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1733)
  - `float32_qk_product` — [`L1730`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1730)
  - `heads` — [`L1726`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1726)
  - `mesh` — [`L1723`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1723)
  - `quant` — [`L1736`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1736)
  - `scale` — [`L1725`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1725)
  - `split_head_dim` — [`L1729`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1729)
  - `ulysses_shards` — [`L1739`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1739)
  - `use_base2_exp` — [`L1737`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1737)
  - `use_experimental_scheduler` — [`L1738`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1738)
  - `use_memory_efficient_attention` — [`L1728`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1728)
- uses (calls/refs, reference-scoped): [`Array`](attention_flax.md#Array), [`AxisNames`](attention_flax.md#AxisNames), [`BATCH`](attention_flax.md#BATCH), [`LENGTH`](attention_flax.md#LENGTH), [`HEAD`](attention_flax.md#HEAD), [`BlockSizes`](attention_flax.md#BlockSizes), [`_apply_attention`](attention_flax.md#_apply_attention), [`Mesh`](attention_flax.md#Mesh), [`D_KV`](attention_flax.md#D_KV), [`Quant`](attention_flax.md#Quant), [`DType`](attention_flax.md#DType), [`KV_LENGTH`](attention_flax.md#KV_LENGTH)
- used by: [`attention_op`](attention_flax.md#FlaxAttention.attention_op), [`attention_op`](attention_flax.md#FlaxFluxAttention.attention_op), [`__call__`](attention_flax.md#FlaxFluxAttention.__call__), [`__call__`](attention_flax.md#FlaxAttention.__call__), [`__call__`](flux/transformers/transformer_flux_flax.md#FluxSingleTransformerBlock.__call__)

### `FlaxAttention`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/attention_flax.py:2437`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2437)
- doc: A Flax multi-head attention module as described in: https://arxiv.org/abs/1706.03762
- signature: `class FlaxAttention(nn.Module):`
- members:
  - `setup(self)` — [`L2488`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2488)
  - `attention_kernel` — [`L2475`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2475)
  - `attention_op` — [`L2494`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2494)
  - `dim_head` — [`L2471`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2471)
  - `dropout` — [`L2472`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2472)
  - `dropout_layer` — [`L2554`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2554)
  - `dtype` — [`L2479`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2479)
  - `flash_block_sizes` — [`L2477`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2477)
  - `flash_min_seq_length` — [`L2476`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2476)
  - `heads` — [`L2470`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2470)
  - `key` — [`L2523`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2523)
  - `key_axis_names` — [`L2482`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2482)
  - `mesh` — [`L2478`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2478)
  - `out_axis_names` — [`L2484`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2484)
  - `precision` — [`L2485`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2485)
  - `proj_attn` — [`L2545`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2545)
  - `quant` — [`L2486`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2486)
  - `query` — [`L2512`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2512)
  - `query_axis_names` — [`L2481`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2481)
  - `query_dim` — [`L2469`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2469)
  - `split_head_dim` — [`L2474`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2474)
  - `use_memory_efficient_attention` — [`L2473`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2473)
  - `value` — [`L2534`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2534)
  - `value_axis_names` — [`L2483`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2483)
  - `weights_dtype` — [`L2480`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2480)
- protocol/private: `__call__`[`L2556`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2556)
- uses (calls/refs, reference-scoped): [`apply_attention`](attention_flax.md#AttentionOp.apply_attention), [`AxisNames`](attention_flax.md#AxisNames), [`BATCH`](attention_flax.md#BATCH), [`LENGTH`](attention_flax.md#LENGTH), [`HEAD`](attention_flax.md#HEAD), [`BlockSizes`](attention_flax.md#BlockSizes), [`Quant`](attention_flax.md#Quant), [`dtype`](attention_flax.md#AttentionOp.dtype), [`flash_block_sizes`](attention_flax.md#AttentionOp.flash_block_sizes), [`mesh`](attention_flax.md#AttentionOp.mesh), [`heads`](attention_flax.md#AttentionOp.heads), [`attention_kernel`](attention_flax.md#AttentionOp.attention_kernel), [`dim_head`](attention_flax.md#AttentionOp.dim_head), [`scale`](attention_flax.md#AttentionOp.scale), [`quant`](attention_flax.md#AttentionOp.quant), [`flash_min_seq_length`](attention_flax.md#AttentionOp.flash_min_seq_length), [`split_head_dim`](attention_flax.md#AttentionOp.split_head_dim), [`use_memory_efficient_attention`](attention_flax.md#AttentionOp.use_memory_efficient_attention), [`AttentionOp`](attention_flax.md#AttentionOp), [`dot_general_cls`](quantizations.md#AqtQuantization.dot_general_cls)
- used by: [`attn1`](attention_flax.md#FlaxBasicTransformerBlock.attn1), [`attn2`](attention_flax.md#FlaxBasicTransformerBlock.attn2)

### `FlaxBasicTransformerBlock`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/attention_flax.py:2579`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2579)
- doc: A Flax transformer block layer with `GLU` (Gated Linear Unit) activation function as described in:
- signature: `class FlaxBasicTransformerBlock(nn.Module):`
- members:
  - `setup(self)` — [`L2630`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2630)
  - `attention_kernel` — [`L2623`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2623)
  - `attn1` — [`L2632`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2632) — documented in [maxdiffusion-models-attention_flax](../../../../concepts/maxdiffusion-models-attention_flax.md)
  - `attn2` — [`L2649`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2649)
  - `d_head` — [`L2616`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2616)
  - `dim` — [`L2614`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2614)
  - `dropout` — [`L2617`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2617)
  - `dropout_layer` — [`L2675`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2675)
  - `dtype` — [`L2619`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2619)
  - `ff` — [`L2665`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2665)
  - `flash_block_sizes` — [`L2625`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2625)
  - `flash_min_seq_length` — [`L2624`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2624)
  - `mesh` — [`L2626`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2626)
  - `n_heads` — [`L2615`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2615)
  - `norm1` — [`L2672`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2672)
  - `norm2` — [`L2673`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2673)
  - `norm3` — [`L2674`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2674)
  - `only_cross_attention` — [`L2618`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2618)
  - `precision` — [`L2627`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2627)
  - `quant` — [`L2628`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2628)
  - `split_head_dim` — [`L2622`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2622)
  - `use_memory_efficient_attention` — [`L2621`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2621)
  - `weights_dtype` — [`L2620`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2620)
- protocol/private: `__call__`[`L2677`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2677)
- uses (calls/refs, reference-scoped): [`BlockSizes`](attention_flax.md#BlockSizes), [`Quant`](attention_flax.md#Quant), [`quant`](attention_flax.md#FlaxAttention.quant), [`dtype`](attention_flax.md#FlaxAttention.dtype), [`precision`](attention_flax.md#FlaxAttention.precision), [`weights_dtype`](attention_flax.md#FlaxAttention.weights_dtype), [`flash_block_sizes`](attention_flax.md#FlaxAttention.flash_block_sizes), [`mesh`](attention_flax.md#FlaxAttention.mesh), [`attention_kernel`](attention_flax.md#FlaxAttention.attention_kernel), [`dim`](attention_flax.md#FlaxFeedForward.dim), [`dtype`](attention_flax.md#FlaxFeedForward.dtype), [`flash_min_seq_length`](attention_flax.md#FlaxAttention.flash_min_seq_length), [`precision`](attention_flax.md#FlaxFeedForward.precision), [`weights_dtype`](attention_flax.md#FlaxFeedForward.weights_dtype), [`FlaxAttention`](attention_flax.md#FlaxAttention), [`dropout`](attention_flax.md#FlaxFeedForward.dropout), [`FlaxFeedForward`](attention_flax.md#FlaxFeedForward)
- used by: [`transformer_blocks`](attention_flax.md#FlaxTransformer2DModel.transformer_blocks)

### `FlaxFeedForward`
- def: [`src/maxdiffusion/models/attention_flax.py:2880`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2880)
- members:
  - `setup(self)` — [`L2904`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2904)
  - `dim` — [`L2898`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2898)
  - `dropout` — [`L2899`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2899)
  - `dtype` — [`L2900`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2900)
  - `net_0` — [`L2907`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2907)
  - `net_2` — [`L2914`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2914)
  - `precision` — [`L2902`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2902)
  - `weights_dtype` — [`L2901`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2901)
- protocol/private: `__call__`[`L2921`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2921)
- uses (calls/refs, reference-scoped): [`precision`](attention_flax.md#FlaxGEGLU.precision), [`FlaxGEGLU`](attention_flax.md#FlaxGEGLU)
- used by: [`ff`](attention_flax.md#FlaxBasicTransformerBlock.ff)

### `FlaxFluxAttention`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/attention_flax.py:2267`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2267)
- signature: `class FlaxFluxAttention(nn.Module):`
- members:
  - `setup(self)` — [`L2287`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2287)
  - `attention_kernel` — [`L2274`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2274)
  - `attention_op` — [`L2293`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2293)
  - `dim_head` — [`L2270`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2270)
  - `dropout` — [`L2271`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2271)
  - `dtype` — [`L2278`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2278)
  - `encoder_key_norm` — [`L2372`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2372)
  - `encoder_proj_attn` — [`L2345`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2345)
  - `encoder_qkv` — [`L2321`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2321)
  - `encoder_query_norm` — [`L2367`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2367)
  - `flash_block_sizes` — [`L2276`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2276)
  - `flash_min_seq_length` — [`L2275`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2275)
  - `heads` — [`L2269`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2269)
  - `key_axis_names` — [`L2281`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2281)
  - `key_norm` — [`L2361`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2361)
  - `mesh` — [`L2277`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2277)
  - `out_axis_names` — [`L2283`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2283)
  - `precision` — [`L2284`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2284)
  - `proj_attn` — [`L2334`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2334)
  - `qkv` — [`L2310`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2310)
  - `qkv_bias` — [`L2285`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2285)
  - `query_axis_names` — [`L2280`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2280)
  - `query_dim` — [`L2268`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2268)
  - `query_norm` — [`L2356`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2356)
  - `split_head_dim` — [`L2273`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2273)
  - `use_memory_efficient_attention` — [`L2272`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2272)
  - `value_axis_names` — [`L2282`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2282)
  - `weights_dtype` — [`L2279`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2279)
- protocol/private: `__call__`[`L2378`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2378)
- uses (calls/refs, reference-scoped): [`apply_attention`](attention_flax.md#AttentionOp.apply_attention), [`AxisNames`](attention_flax.md#AxisNames), [`BATCH`](attention_flax.md#BATCH), [`LENGTH`](attention_flax.md#LENGTH), [`HEAD`](attention_flax.md#HEAD), [`BlockSizes`](attention_flax.md#BlockSizes), [`dtype`](attention_flax.md#AttentionOp.dtype), [`apply_rope`](attention_flax.md#apply_rope), [`flash_block_sizes`](attention_flax.md#AttentionOp.flash_block_sizes), [`mesh`](attention_flax.md#AttentionOp.mesh), [`heads`](attention_flax.md#AttentionOp.heads), [`EMBED`](attention_flax.md#EMBED), [`attention_kernel`](attention_flax.md#AttentionOp.attention_kernel), [`dim_head`](attention_flax.md#AttentionOp.dim_head), [`scale`](attention_flax.md#AttentionOp.scale), [`flash_min_seq_length`](attention_flax.md#AttentionOp.flash_min_seq_length), [`split_head_dim`](attention_flax.md#AttentionOp.split_head_dim), [`use_memory_efficient_attention`](attention_flax.md#AttentionOp.use_memory_efficient_attention), [`AttentionOp`](attention_flax.md#AttentionOp), [`float32_qk_product`](attention_flax.md#AttentionOp.float32_qk_product)
- used by: [`attn`](flux/transformers/transformer_flux_flax.md#FluxTransformerBlock.attn), [`attn`](flux/transformers/transformer_flux_flax.md#FluxSingleTransformerBlock.attn), [`__call__`](flux/transformers/transformer_flux_flax.md#FluxSingleTransformerBlock.__call__)

### `FlaxGEGLU`
- def: [`src/maxdiffusion/models/attention_flax.py:2927`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2927)
- members:
  - `setup(self)` — [`L2947`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2947)
  - `dim` — [`L2941`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2941)
  - `dropout` — [`L2942`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2942)
  - `dropout_layer` — [`L2955`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2955)
  - `dtype` — [`L2943`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2943)
  - `precision` — [`L2945`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2945)
  - `proj` — [`L2949`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2949)
  - `weights_dtype` — [`L2944`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2944)
- protocol/private: `__call__`[`L2957`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2957)
- used by: [`net_0`](attention_flax.md#FlaxFeedForward.net_0)

### `FlaxTransformer2DModel`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/attention_flax.py:2714`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2714)
- doc: A Spatial Transformer layer with Gated Linear Unit (GLU) activation function as described in:
- signature: `class FlaxTransformer2DModel(nn.Module):`
- members:
  - `setup(self)` — [`L2772`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2772)
  - `attention_kernel` — [`L2763`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2763)
  - `d_head` — [`L2754`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2754)
  - `depth` — [`L2755`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2755)
  - `dropout` — [`L2756`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2756)
  - `dropout_layer` — [`L2846`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2846)
  - `dtype` — [`L2759`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2759)
  - `flash_block_sizes` — [`L2765`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2765)
  - `flash_min_seq_length` — [`L2764`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2764)
  - `hidden_state_axis_names` — [`L2769`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2769)
  - `in_channels` — [`L2752`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2752)
  - `mesh` — [`L2766`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2766)
  - `n_heads` — [`L2753`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2753)
  - `norm` — [`L2773`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2773)
  - `norm_num_groups` — [`L2767`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2767)
  - `only_cross_attention` — [`L2758`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2758)
  - `precision` — [`L2768`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2768)
  - `proj_in` — [`L2786`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2786)
  - `proj_out` — [`L2827`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2827)
  - `quant` — [`L2770`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2770)
  - `split_head_dim` — [`L2762`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2762)
  - `transformer_blocks` — [`L2805`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2805) — documented in [maxdiffusion-models-attention_flax](../../../../concepts/maxdiffusion-models-attention_flax.md)
  - `use_linear_projection` — [`L2757`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2757)
  - `use_memory_efficient_attention` — [`L2761`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2761)
  - `weights_dtype` — [`L2760`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2760)
- protocol/private: `__call__`[`L2848`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2848)
- uses (calls/refs, reference-scoped): [`AxisNames`](attention_flax.md#AxisNames), [`BATCH`](attention_flax.md#BATCH), [`LENGTH`](attention_flax.md#LENGTH), [`BlockSizes`](attention_flax.md#BlockSizes), [`D_KV`](attention_flax.md#D_KV), [`Quant`](attention_flax.md#Quant), [`dtype`](attention_flax.md#FlaxBasicTransformerBlock.dtype), [`weights_dtype`](attention_flax.md#FlaxBasicTransformerBlock.weights_dtype), [`flash_block_sizes`](attention_flax.md#FlaxBasicTransformerBlock.flash_block_sizes), [`quant`](attention_flax.md#FlaxBasicTransformerBlock.quant), [`dropout`](attention_flax.md#FlaxBasicTransformerBlock.dropout), [`precision`](attention_flax.md#FlaxBasicTransformerBlock.precision), [`attention_kernel`](attention_flax.md#FlaxBasicTransformerBlock.attention_kernel), [`flash_min_seq_length`](attention_flax.md#FlaxBasicTransformerBlock.flash_min_seq_length), [`mesh`](attention_flax.md#FlaxBasicTransformerBlock.mesh), [`split_head_dim`](attention_flax.md#FlaxBasicTransformerBlock.split_head_dim), [`use_memory_efficient_attention`](attention_flax.md#FlaxBasicTransformerBlock.use_memory_efficient_attention), [`only_cross_attention`](attention_flax.md#FlaxBasicTransformerBlock.only_cross_attention), [`FlaxBasicTransformerBlock`](attention_flax.md#FlaxBasicTransformerBlock)
- used by: [`setup`](unet_2d_blocks_flax.md#FlaxCrossAttnUpBlock2D.setup), [`setup`](unet_2d_blocks_flax.md#FlaxCrossAttnDownBlock2D.setup), [`setup`](unet_2d_blocks_flax.md#FlaxUNetMidBlock2DCrossAttn.setup)

### `FlaxWanAttention`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/attention_flax.py:1791`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1791)
- signature: `class FlaxWanAttention(nnx.Module):`
- members:
  - `compute_kv(self, encoder_hidden_states: jax.Array, encoder_attention_mask: Optional[jax.Array] = None)` — [`L2208`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2208)
  - `conditional_named_scope(self, name: str)` — [`L2039`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2039) — Return a JAX named scope if enabled, otherwise a null context. — documented in [maxdiffusion-models-attention_flax](../../../../concepts/maxdiffusion-models-attention_flax.md)
  - `add_k_proj` — [`L1973`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1973)
  - `add_v_proj` — [`L1974`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1974)
  - `added_kv_proj_dim` — [`L1856`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1856)
  - `alignment` — [`L1859`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1859)
  - `attention_op` — [`L1861`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1861)
  - `dim_head` — [`L1835`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1835)
  - `drop_out` — [`L1944`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1944)
  - `enable_jax_named_scopes` — [`L1844`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1844)
  - `heads` — [`L1836`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1836)
  - `image_seq_len` — [`L1857`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1857)
  - `inner_dim` — [`L1837`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1837)
  - `key` — [`L1902`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1902)
  - `key_axis_names` — [`L1841`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1841)
  - `norm_added_k` — [`L1975`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1975) — documented in [maxdiffusion-models-attention_flax](../../../../concepts/maxdiffusion-models-attention_flax.md)
  - `norm_k` — [`L1947`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1947) — documented in [maxdiffusion-models-attention_flax](../../../../concepts/maxdiffusion-models-attention_flax.md)
  - `norm_q` — [`L1946`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1946)
  - `out_axis_names` — [`L1843`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1843)
  - `proj_attn` — [`L1930`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1930)
  - `qk_norm` — [`L1839`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1839) — documented in [maxdiffusion-models-attention_flax](../../../../concepts/maxdiffusion-models-attention_flax.md)
  - `query` — [`L1888`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1888) — documented in [maxdiffusion-models-attention_flax](../../../../concepts/maxdiffusion-models-attention_flax.md)
  - `query_axis_names` — [`L1840`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1840)
  - `value` — [`L1916`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1916)
  - `value_axis_names` — [`L1842`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1842)
- protocol/private: `__call__`[`L2043`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2043), `__init__`[`L1793`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1793), `_apply_rope`[`L2013`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L2013)
- uses (calls/refs, reference-scoped): [`apply_attention`](attention_flax.md#NNXAttentionOp.apply_attention), [`AxisNames`](attention_flax.md#AxisNames), [`BATCH`](attention_flax.md#BATCH), [`LENGTH`](attention_flax.md#LENGTH), [`HEAD`](attention_flax.md#HEAD), [`BlockSizes`](attention_flax.md#BlockSizes), [`D_KV`](attention_flax.md#D_KV), [`Quant`](attention_flax.md#Quant), [`NNXAttentionOp`](attention_flax.md#NNXAttentionOp), [`CROSS_ATTN_HEAD`](attention_flax.md#CROSS_ATTN_HEAD), [`EMBED`](attention_flax.md#EMBED), [`SELF_ATTN_HEAD`](attention_flax.md#SELF_ATTN_HEAD), [`_unflatten_heads`](attention_flax.md#_unflatten_heads), [`CROSS_ATTN_KV_LENGTH`](attention_flax.md#CROSS_ATTN_KV_LENGTH), [`CROSS_ATTN_Q_LENGTH`](attention_flax.md#CROSS_ATTN_Q_LENGTH), [`SELF_ATTN_KV_LENGTH`](attention_flax.md#SELF_ATTN_KV_LENGTH), [`SELF_ATTN_Q_LENGTH`](attention_flax.md#SELF_ATTN_Q_LENGTH), `rate`
- used by: [`compute_kv`](wan/transformers/transformer_wan_vace.md#WanVACETransformerBlock.compute_kv), [`compute_kv`](wan/transformers/transformer_wan.md#WanTransformerBlock.compute_kv), [`attn2`](wan/transformers/transformer_wan_vace.md#WanVACETransformerBlock.attn2), [`attn2`](wan/transformers/transformer_wan.md#WanTransformerBlock.attn2), [`attn1`](wan/transformers/transformer_wan_vace.md#WanVACETransformerBlock.attn1), [`attn1`](wan/transformers/transformer_wan.md#WanTransformerBlock.attn1)

### `NNXAttentionOp`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/attention_flax.py:1627`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1627)
- signature: `class NNXAttentionOp(nnx.Module):`
- members:
  - `apply_attention(self, query: Array, key: Array, value: Array, attention_mask: Array = None)` — [`L1694`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1694) — documented in [maxdiffusion-models-attention_flax](../../../../concepts/maxdiffusion-models-attention_flax.md)
  - `attention_kernel` — [`L1681`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1681)
  - `axis_names_kv` — [`L1686`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1686)
  - `axis_names_q` — [`L1685`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1685)
  - `dim_head` — [`L1680`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1680)
  - `dpa_layer` — [`L1652`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1652)
  - `dtype` — [`L1689`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1689)
  - `flash_block_sizes` — [`L1688`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1688)
  - `flash_min_seq_length` — [`L1687`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1687)
  - `float32_qk_product` — [`L1684`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1684)
  - `heads` — [`L1679`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1679)
  - `mask_padding_tokens` — [`L1691`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1691)
  - `mesh` — [`L1677`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1677)
  - `quant` — [`L1690`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1690)
  - `residual_checkpoint_name` — [`L1692`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1692)
  - `scale` — [`L1678`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1678)
  - `split_head_dim` — [`L1683`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1683)
  - `ulysses_shards` — [`L1655`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1655)
  - `use_base2_exp` — [`L1653`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1653)
  - `use_experimental_scheduler` — [`L1654`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1654)
  - `use_memory_efficient_attention` — [`L1682`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1682)
- protocol/private: `__init__`[`L1629`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1629)
- uses (calls/refs, reference-scoped): [`Array`](attention_flax.md#Array), [`AxisNames`](attention_flax.md#AxisNames), [`BATCH`](attention_flax.md#BATCH), [`LENGTH`](attention_flax.md#LENGTH), [`HEAD`](attention_flax.md#HEAD), [`BlockSizes`](attention_flax.md#BlockSizes), [`_apply_attention`](attention_flax.md#_apply_attention), [`Mesh`](attention_flax.md#Mesh), [`D_KV`](attention_flax.md#D_KV), [`Quant`](attention_flax.md#Quant), [`DType`](attention_flax.md#DType), [`KV_LENGTH`](attention_flax.md#KV_LENGTH)
- used by: [`__call__`](attention_flax.md#FlaxWanAttention.__call__), [`__call__`](wan/transformers/transformer_wan_animate.md#WanAnimateFaceBlockCrossAttention.__call__), [`__call__`](ltx2/attention_ltx2.md#LTX2Attention.__call__), [`attention_op`](ltx2/attention_ltx2.md#LTX2Attention.attention_op), [`attention_op`](wan/transformers/transformer_wan_animate.md#WanAnimateFaceBlockCrossAttention.attention_op), [`attention_op`](attention_flax.md#FlaxWanAttention.attention_op)

### `NNXSimpleFeedForward`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/attention_flax.py:1574`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1574) — documented in [maxdiffusion-models-embeddings_flax](../../../../concepts/maxdiffusion-models-embeddings_flax.md)
- signature: `class NNXSimpleFeedForward(nnx.Module):`
- members:
  - `act` — [`L1607`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1607)
  - `net_0` — [`L1596`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1596)
  - `net_2` — [`L1608`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1608)
- protocol/private: `__call__`[`L1620`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1620), `__init__`[`L1576`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1576)
- uses (calls/refs, reference-scoped): [`Array`](attention_flax.md#Array), [`get_activation`](modeling_flax_utils.md#get_activation)
- used by: [`ff`](embeddings_flax.md#NNXWanImageEmbedding.ff)

## Functions
- `_apply_attention(query: Array, key: Array, value: Array, heads: int, dim_head: int, split_head_dim: bool, float32_qk_product: bool, attention_kernel: str, flash_min_seq_length: int, use_memory_efficient_attention: bool, scale: float, dtype: jnp.dtype, mesh: Mesh, axis_names_q: AxisNames, axis_names_kv: AxisNames, flash_block_sizes: BlockSizes, dpa_layer: Callable, mask_padding_tokens: bool = True, residual_checkpoint_name: str | None = None, attention_mask: Array = None, use_base2_exp: bool = False, use_experimental_scheduler: bool = False, ulysses_shards: int = -1)` — [`L1383`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1383) — Routes to different attention kernels using a module-level registry.
- `_apply_attention_dot(query: Array, key: Array, value: Array, dtype: jnp.dtype, heads: int, dim_head: int, scale: float, split_head_dim: bool, float32_qk_product: bool, use_memory_efficient_attention: bool)` — [`L1067`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1067) — Apply Attention.
- `_build_padding_segment_ids(query_seq_len: int, q_padded_len: int, key_seq_len: int, kv_padded_len: int, attention_mask: jax.Array | None, segment_ids_cls=splash_attention_kernel.SegmentIds)` — [`L381`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L381) — Build splash segment ids that mask q/kv padding and the attention mask.
- `_check_attention_inputs(query: Array, key: Array, value: Array)` — [`L99`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L99) — Check attention inputs.
- `_coerce_tokamax_block_sizes(block_sizes)` — [`L70`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L70)
- `_create_internal_ulysses_ring_mesh(mesh: Mesh, ring_shards: int, ulysses_shards: int, ring_axis: str = INTERNAL_RING_AXIS, ulysses_axis: str = INTERNAL_ULYSSES_AXIS)` — [`L185`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L185) — Split the public context mesh axis into private ring and Ulysses axes.
- `_cudnn_flash_attention(query: Array, key: Array, value: Array, heads: int, mesh: Mesh, dpa_layer: Callable)` — [`L1144`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1144) — CUDNN Flash Attention with Transformer Engine.
- `_extract_custom_block_sizes(flash_block_sizes)` — [`L350`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L350) — Pulls custom-kernel block sizes out of the (dict or BlockSizes-like) config.
- `_flash_sequence_length(tensor: Array)` — [`L265`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L265)
- `_maybe_aqt_einsum(quant: Quant)` — [`L95`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L95)
- `_pad_data_for_flash(tensor, heads, flash_block_size, num_shards: int = 1)` — [`L226`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L226) — Reshapes tensors for pallas flash attention adding padding to both seq_len and head_dim.
- `_query_chunk_attention(query, key, value, precision, key_chunk_size: int = 4096)` — [`L1455`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1455) — Multi-head dot product attention with a limited number of queries.
- `_replace_mesh_axis(axis_spec, old_axis: str, new_axes: tuple[str, ...])` — [`L167`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L167)
- `_replace_mesh_axis_names(axis_names, old_axis: str, new_axes: tuple[str, ...])` — [`L181`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L181)
- `_reshape_batch_dim_to_heads(tensor, heads)` — [`L124`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L124)
- `_reshape_data_for_cudnn_flash(tensor, heads)` — [`L114`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L114)
- `_reshape_data_for_flash(tensor, heads, num_context_shards=1)` — [`L203`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L203) — Reshapes tensors for pallas flash attention adding padding to both seq_len and head_dim.
- `_reshape_data_from_cudnn_flash(tensor)` — [`L109`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L109)
- `_reshape_heads_to_batch_dim(tensor, heads)` — [`L134`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L134)
- `_reshape_heads_to_head_dim(tensor)` — [`L148`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L148)
- `_select_flash_block_sizes(query: Array, key: Array, flash_block_sizes: BlockSizes, dtype: jnp.dtype, attention_kernel: str)` — [`L273`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L273)
- `_tpu_flash_attention(query: jax.Array, key: jax.Array, value: jax.Array, heads: int, mesh: Mesh, axis_names_q: AxisNames, axis_names_kv: AxisNames, flash_block_sizes: BlockSizes, dtype: jnp.dtype = jnp.float32, attention_kernel: str = "flash", mask_padding_tokens: bool = True, residual_checkpoint_name: str | None = None, attention_mask: jax.Array = None, use_base2_exp: bool = False, use_experimental_scheduler: bool = False)` — [`L416`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L416) — TPU Flash Attention
- `_ulysses_attention(query: jax.Array, key: jax.Array, value: jax.Array, heads: int, mesh: Mesh, axis_names_q: AxisNames, axis_names_kv: AxisNames, flash_block_sizes: BlockSizes, dtype: jnp.dtype = jnp.float32, mask_padding_tokens: bool = True, residual_checkpoint_name: str | None = None, attention_mask: jax.Array = None, use_custom_kernel: bool = False, use_base2_exp: bool = True, use_experimental_scheduler: bool = False)` — [`L627`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L627) — Ulysses sequence-parallel attention.
- `_ulysses_ring_attention(query: jax.Array, key: jax.Array, value: jax.Array, heads: int, mesh: Mesh, axis_names_q: AxisNames, axis_names_kv: AxisNames, flash_block_sizes: BlockSizes, dtype: jnp.dtype = jnp.float32, mask_padding_tokens: bool = True, residual_checkpoint_name: str | None = None, attention_mask: jax.Array = None, ulysses_axis: str = INTERNAL_ULYSSES_AXIS, ring_axis: str = INTERNAL_RING_AXIS, use_base2_exp: bool = False, use_experimental_scheduler: bool = False, ulysses_shards: int = -1)` — [`L772`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L772) — 2D context-parallel attention using a private Ulysses x ring mesh.
- `_ulysses_ring_custom_attention(query: jax.Array, key: jax.Array, value: jax.Array, heads: int, mesh: Mesh, axis_names_q: AxisNames, axis_names_kv: AxisNames, flash_block_sizes: BlockSizes, dtype: jnp.dtype = jnp.float32, mask_padding_tokens: bool = True, residual_checkpoint_name: str | None = None, attention_mask: jax.Array = None, ulysses_shards: int = -1, use_base2_exp: bool = True, use_experimental_scheduler: bool = False, bidirectional: bool = False)` — [`L927`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L927) — Hybrid Ulysses + Ring (USP) with the CUSTOM splash kernel on main's mesh.
- `_unflatten_heads(tensor, heads)` — [`L158`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L158)
- `apply_rope(xq: Array, xk: Array, freqs_cis: Array)` — [`L1563`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1563)
- `chunk_scanner(chunk_idx)` — [`L1475`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1475)
- `chunk_scanner(chunk_idx, _)` — [`L1534`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1534)
- `convert_to_tokamax_splash_config(block_sizes: BlockSizes, q_layout: tokamax_splash_attention_kernel.QKVLayout = tokamax_splash_attention_kernel.QKVLayout.HEAD_DIM_MINOR, k_layout: tokamax_splash_attention_kernel.QKVLayout = tokamax_splash_attention_kernel.QKVLayout.HEAD_DIM_MINOR, v_layout: tokamax_splash_attention_kernel.QKVLayout = tokamax_splash_attention_kernel.QKVLayout.HEAD_DIM_MINOR, residual_checkpoint_name: str | None = None, attn_logits_soft_cap: float | None = None, fuse_reciprocal: bool = True, use_base2_exp: bool = False, use_experimental_scheduler: bool = False, max_logit_const: float | None = None, interpret: bool = False, dq_reduction_steps: int | None = None)` — [`L311`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L311)
- `cudnn_flash_te_kernel(q, k, v, context)` — [`L1379`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1379)
- `decorator(func)` — [`L1168`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1168)
- `dot_product_kernel(q, k, v, context)` — [`L1177`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1177)
- `flash_kernel(q, k, v, context)` — [`L1299`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1299)
- `jax_memory_efficient_attention(query, key, value, precision=jax.lax.Precision.HIGHEST, query_chunk_size: int = 1024, key_chunk_size: int = 4096)` — [`L1506`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1506) — Flax Memory-efficient multi-head dot product attention. https://arxiv.org/abs/2112.05682v2
- `register_kernel(name: str)` — [`L1167`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1167)
- `ring_scan_body(carry, _)` — [`L573`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L573)
- `summarize_chunk(query, key, value)` — [`L1463`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1463)
- `tokamax_flash_kernel(q, k, v, context)` — [`L1320`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1320)
- `tokamax_ring_custom_kernel(q, k, v, context)` — [`L1359`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1359)
- `tokamax_ring_kernel(q, k, v, context)` — [`L1341`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1341)
- `ulysses_custom_kernel(q, k, v, context)` — [`L1193`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1193)
- `ulysses_kernel(q, k, v, context)` — [`L1260`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1260)
- `ulysses_ring_custom_bidir_kernel(q, k, v, context)` — [`L1235`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1235) — Wrap-free (bidirectional) variant of ulysses_ring_custom: the ring streams
- `ulysses_ring_custom_kernel(q, k, v, context)` — [`L1214`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1214)
- `ulysses_ring_kernel(q, k, v, context)` — [`L1278`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1278)
- `wrap_flash_attention(query, key, value)` — [`L451`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L451)
- `wrap_ulysses_attention(query, key, value)` — [`L678`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L678)
- `wrap_ulysses_ring_attention(query, key, value)` — [`L846`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L846)
- `wrap_ulysses_ring_attention(query, key, value)` — [`L1005`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1005)

## Module values
- `Array` — [`L43`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L43)
- `AxisNames` — [`L49`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L49)
- `BATCH` — [`L51`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L51)
- `BlockSizes` — [`L46`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L46) — documented in [maxdiffusion-models-flux-transformers-transformer_flux_flax](../../../../concepts/maxdiffusion-models-flux-transformers-transformer_flux_flax.md)
- `CONTEXT` — [`L50`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L50)
- `CROSS_ATTN_HEAD` — [`L62`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L62)
- `CROSS_ATTN_KV_LENGTH` — [`L64`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L64)
- `CROSS_ATTN_Q_LENGTH` — [`L63`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L63)
- `DType` — [`L45`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L45)
- `D_KV` — [`L55`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L55)
- `EMBED` — [`L56`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L56)
- `HEAD` — [`L54`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L54)
- `INTERNAL_RING_AXIS` — [`L66`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L66)
- `INTERNAL_ULYSSES_AXIS` — [`L67`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L67)
- `KERNEL_REGISTRY` — [`L1164`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L1164)
- `KV_LENGTH` — [`L53`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L53)
- `LENGTH` — [`L52`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L52)
- `LOG2E` — [`L41`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L41)
- `Mesh` — [`L44`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L44)
- `Quant` — [`L57`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L57)
- `SELF_ATTN_HEAD` — [`L59`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L59)
- `SELF_ATTN_KV_LENGTH` — [`L61`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L61)
- `SELF_ATTN_Q_LENGTH` — [`L60`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/attention_flax.py#L60)

