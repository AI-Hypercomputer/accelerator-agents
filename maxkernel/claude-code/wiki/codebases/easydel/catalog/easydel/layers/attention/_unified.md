---
title: 'Module: easydel/layers/attention/_unified.py'
type: catalog
provenance: extracted
module: easydel/layers/attention/_unified.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.attention._unified`/
symbols:
  UnifiedAttention: UnifiedAttention#
  UnifiedAttention.head_dim: UnifiedAttention#head_dim.
  UnifiedAttention.forward: UnifiedAttention#forward().
  UnifiedAttention._create_attention_performer: UnifiedAttention#_create_attention_performer().
  UnifiedAttention.forward_mla: UnifiedAttention#forward_mla().
  UnifiedAttention._create_rotary: UnifiedAttention#_create_rotary().
  UnifiedAttention.forward_alibi: UnifiedAttention#forward_alibi().
  UnifiedAttention.define_network: UnifiedAttention#define_network().
  UnifiedAttention.__init__: UnifiedAttention#__init__().
  UnifiedAttention.attention_performer: UnifiedAttention#attention_performer.
  UnifiedAttention.num_heads: UnifiedAttention#num_heads.
  UnifiedAttention._postprocess_qkv: UnifiedAttention#_postprocess_qkv().
  UnifiedAttention._create_o_proj: UnifiedAttention#_create_o_proj().
  UnifiedAttention._create_q_norm: UnifiedAttention#_create_q_norm().
  UnifiedAttention._create_k_norm: UnifiedAttention#_create_k_norm().
  UnifiedAttention.__call__: UnifiedAttention#__call__().
  UnifiedAttention._create_q_proj: UnifiedAttention#_create_q_proj().
  UnifiedAttention._merge_heads: UnifiedAttention#_merge_heads().
  UnifiedAttention._create_k_proj: UnifiedAttention#_create_k_proj().
  UnifiedAttention._create_v_proj: UnifiedAttention#_create_v_proj().
  UnifiedAttention.projection_mapping: UnifiedAttention#projection_mapping.
  UnifiedAttention.query_normalization: UnifiedAttention#query_normalization().
  UnifiedAttention.key_normalization: UnifiedAttention#key_normalization().
  UnifiedAttention._create_fused_qkv_proj: UnifiedAttention#_create_fused_qkv_proj().
  Cfg: Cfg.
  UnifiedAttention.num_key_value_heads: UnifiedAttention#num_key_value_heads.
  UnifiedAttention.output_projection: UnifiedAttention#output_projection().
  UnifiedAttention._preprocess_qkv: UnifiedAttention#_preprocess_qkv().
  UnifiedAttention.sliding_window: UnifiedAttention#sliding_window.
  UnifiedAttention._apply_rotary: UnifiedAttention#_apply_rotary().
  UnifiedAttention.mla_kv_b_proj: UnifiedAttention#mla_kv_b_proj().
  UnifiedAttention.query_projection: UnifiedAttention#query_projection().
  UnifiedAttention.mla_kv_a_layernorm: UnifiedAttention#mla_kv_a_layernorm().
  UnifiedAttention.operation_requirements: UnifiedAttention#operation_requirements().
  UnifiedAttention.num_key_value_groups: UnifiedAttention#num_key_value_groups.
  UnifiedAttention.causal: UnifiedAttention#causal.
  UnifiedAttention.use_mla_lora: UnifiedAttention#use_mla_lora.
  UnifiedAttention.key_projection: UnifiedAttention#key_projection().
  UnifiedAttention.value_projection: UnifiedAttention#value_projection().
  UnifiedAttention.mla_q_proj: UnifiedAttention#mla_q_proj().
  UnifiedAttention.mla_q_a_proj: UnifiedAttention#mla_q_a_proj().
  UnifiedAttention.mla_q_a_layernorm: UnifiedAttention#mla_q_a_layernorm().
  UnifiedAttention.mla_q_b_proj: UnifiedAttention#mla_q_b_proj().
  UnifiedAttention.mla_kv_a_proj_with_mqa: UnifiedAttention#mla_kv_a_proj_with_mqa().
  UnifiedAttention._create_v_norm: UnifiedAttention#_create_v_norm().
  UnifiedAttention._create_o_norm: UnifiedAttention#_create_o_norm().
  UnifiedAttention.rotary: UnifiedAttention#rotary.
  UnifiedAttention._create_alibi_slopes: UnifiedAttention#_create_alibi_slopes().
  UnifiedAttention.query_key_value_projection: UnifiedAttention#query_key_value_projection().
  UnifiedAttention.norms_mapping: UnifiedAttention#norms_mapping.
  UnifiedAttention._compute_alibi_bias: UnifiedAttention#_compute_alibi_bias().
  UnifiedAttention.dtype: UnifiedAttention#dtype.
  UnifiedAttention.value_normalization: UnifiedAttention#value_normalization().
  UnifiedAttention.output_normalization: UnifiedAttention#output_normalization().
  UnifiedAttention.operation_executor: UnifiedAttention#operation_executor().
  UnifiedAttention.operation: UnifiedAttention#operation().
  UnifiedAttention.requires_cache: UnifiedAttention#requires_cache().
  UnifiedAttention.attention_type: UnifiedAttention#attention_type.
  UnifiedAttention.resid_dropout: UnifiedAttention#resid_dropout.
  UnifiedAttention.use_qk_norm: UnifiedAttention#use_qk_norm.
  apply_rotary_pos_emb: apply_rotary_pos_emb().
  apply_rotary_pos_emb.rotate_half: apply_rotary_pos_emb().rotate_half().
  UnifiedAttention.param_dtype: UnifiedAttention#param_dtype.
  UnifiedAttention.use_fused_qkv: UnifiedAttention#use_fused_qkv.
  UnifiedAttention.hidden_size: UnifiedAttention#hidden_size.
  UnifiedAttention.use_gqa: UnifiedAttention#use_gqa.
  UnifiedAttention.alibi_slopes: UnifiedAttention#alibi_slopes.
  UnifiedAttention.precision: UnifiedAttention#precision.
  UnifiedAttention.layer_idx: UnifiedAttention#layer_idx.
---
# Module: [`easydel/layers/attention/_unified.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py)

## Classes
### `UnifiedAttention`  ·  implements/extends AttentionModule, Generic
- def: [`easydel/layers/attention/_unified.py:209`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L209) — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
- doc: Unified attention module supporting multiple attention mechanisms.
- signature: `class UnifiedAttention(AttentionModule, Generic[Cfg]):`
- members:
  - `__call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"], mask_info: MaskInfo | None, position_ids: Int[Array, "batch seq_len"], mode: common_types.RUNTIME_MODE_TYPES, cache_view: TransformerCacheView | RaggedPagesCacheView | UnifiedAttentionCacheView | None = None, cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None, output_attentions: bool = False, frequencies: Float[Array, "seq_len head_dim"] | None = None, alibi: Float[Array, "batch_or_1 heads qseq_len_or_1 kvseq_len_or_1"] | None = None)` — [`L1525`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L1525) — Main entry point that routes to the appropriate forward method. — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
  - `__init__(self, config: Cfg, dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.bfloat16, precision: str | jax.lax.Precision | None = None, *, rngs: nn.Rngs, layer_idx: int, attention_type: Literal["standard", "mla", "alibi"] = "standard", causal: bool = True, sliding_window: int | tuple[int, int] | None = None, use_qk_norm: bool = False, use_fused_qkv: bool = False, use_gqa: bool = False, use_mla_lora: bool = False)` — [`L290`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L290) — Initialize unified attention module. — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
  - `_apply_rotary(self, query_states: Float[Array, "batch_size seq_len num_heads head_dim"], key_states: Float[Array, "batch_size seq_len num_kv_heads head_dim"], position_ids: Int[Array, "batch_size seq_len"], frequencies: Float[Array, "max_seq_len head_dim"] | None = None)` — [`L608`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L608) — Apply rotary position embeddings to query and key tensors.
  - `_compute_alibi_bias(self, sequence_length: int)` — [`L660`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L660) — Compute ALiBi positional bias matrix.
  - `_create_alibi_slopes(self, config: Cfg)` — [`L651`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L651) — Create ALiBi slope values for positional bias.
  - `_create_attention_performer(self, config: Cfg, rngs: nn.Rngs)` — [`L679`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L679) — Create attention performer module. — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
  - `_create_fused_qkv_proj(self, config: Cfg, dtype: DTypeLike, param_dtype: DTypeLike, precision: jax.lax.Precision, rngs: PRNGKeyArray)` — [`L569`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L569) — Create fused QKV projection (Phi3, DBRX, MPT style). — documented in [easydel-layers-linears-_linear](../../../../concepts/easydel-layers-linears-_linear.md)
  - `_create_k_norm(self, config: Cfg, dtype: DTypeLike, param_dtype: DTypeLike, rngs: nn.Rngs)` — [`L731`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L731) — Create key normalization layer. — documented in [easydel-layers-norms-_norms](../../../../concepts/easydel-layers-norms-_norms.md)
  - `_create_k_proj(self, config: Cfg, dtype: DTypeLike, param_dtype: DTypeLike, precision: jax.lax.Precision, rngs: PRNGKeyArray)` — [`L453`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L453) — Create key projection layer. — documented in [easydel-layers-linears-_linear](../../../../concepts/easydel-layers-linears-_linear.md)
  - `_create_o_norm(self, config: Cfg, dtype: DTypeLike, param_dtype: DTypeLike, rngs: nn.Rngs)` — [`L763`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L763) — Create output normalization layer.
  - `_create_o_proj(self, config: Cfg, dtype: DTypeLike, param_dtype: DTypeLike, precision: jax.lax.Precision, rngs: PRNGKeyArray)` — [`L531`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L531) — Create output projection layer. — documented in [easydel-layers-linears-_linear](../../../../concepts/easydel-layers-linears-_linear.md)
  - `_create_q_norm(self, config: Cfg, dtype: DTypeLike, param_dtype: DTypeLike, rngs: nn.Rngs)` — [`L715`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L715) — Create query normalization layer. — documented in [easydel-layers-norms-_norms](../../../../concepts/easydel-layers-norms-_norms.md)
  - `_create_q_proj(self, config: Cfg, dtype: DTypeLike, param_dtype: DTypeLike, precision: jax.lax.Precision, rngs: PRNGKeyArray)` — [`L415`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L415) — Create query projection layer. — documented in [easydel-layers-linears-_linear](../../../../concepts/easydel-layers-linears-_linear.md)
  - `_create_rotary(self, config: Cfg, dtype: DTypeLike)` — [`L593`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L593) — Create rotary position embedding layer. — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
  - `_create_v_norm(self, config: Cfg, dtype: DTypeLike, param_dtype: DTypeLike, rngs: nn.Rngs)` — [`L747`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L747) — Create value normalization layer.
  - `_create_v_proj(self, config: Cfg, dtype: DTypeLike, param_dtype: DTypeLike, precision: jax.lax.Precision, rngs: PRNGKeyArray)` — [`L492`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L492) — Create value projection layer. — documented in [easydel-layers-linears-_linear](../../../../concepts/easydel-layers-linears-_linear.md)
  - `_merge_heads(self, hidden_states: Float[Array, "batch_size seq_len num_heads head_dim"])` — [`L1032`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L1032) — Merge attention heads back to hidden dimension.
  - `_postprocess_qkv(self, query_states: Float[Array, "batch_size seq_len num_heads head_dim"], key_states: Float[Array, "batch_size seq_len num_kv_heads head_dim"], value_states: Float[Array, "batch_size seq_len num_kv_heads head_dim"])` — [`L1000`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L1000) — Post-process Q/K/V after projection and reshape, before RoPE/sharding. — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
  - `_preprocess_qkv(self, query_states: Float[Array, "batch_size seq_len num_heads_times_head_dim"], key_states: Float[Array, "batch_size seq_len num_kv_heads_times_head_dim"], value_states: Float[Array, "batch_size seq_len num_kv_heads_times_head_dim"])` — [`L968`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L968) — Pre-process Q/K/V after projection before reshape/RoPE/sharding.
  - `define_network(self, config: Cfg, dtype: DTypeLike, param_dtype: DTypeLike, precision: jax.lax.Precision, rngs: PRNGKeyArray)` — [`L342`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L342) — Define network structure. — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
  - `forward(self, hidden_states: Float[Array, "batch seq_len hidden_dim"], mask_info: MaskInfo | None, position_ids: Int[Array, "batch seq_len"], mode: common_types.RUNTIME_MODE_TYPES, cache_view: TransformerCacheView | RaggedPagesCacheView | UnifiedAttentionCacheView | None = None, cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None, output_attentions: bool = False, frequencies: Float[Array, "seq_len head_dim"] | None = None, alibi: Float[Array, "batch_or_1 heads qseq_len_or_1 kvseq_len_or_1"] | None = None)` — [`L1048`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L1048) — Standard RoPE-based attention (default path). — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
  - `forward_alibi(self, hidden_states: Float[Array, "batch seq_len hidden_dim"], mask_info: MaskInfo | None, position_ids: Int[Array, "batch seq_len"], mode: common_types.RUNTIME_MODE_TYPES, cache_view: TransformerCacheView | RaggedPagesCacheView | UnifiedAttentionCacheView | None = None, cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None, output_attentions: bool = False, alibi: Float[Array, "batch_or_1 heads qseq_len_or_1 kvseq_len_or_1"] | None = None)` — [`L1438`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L1438) — ALiBi positional bias attention forward path (Falcon, MPT). — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
  - `forward_mla(self, hidden_states: Float[Array, "batch seq_len hidden_dim"], mask_info: MaskInfo | None, position_ids: Int[Array, "batch seq_len"], mode: common_types.RUNTIME_MODE_TYPES, cache_view: TransformerCacheView | RaggedPagesCacheView | UnifiedAttentionCacheView | None = None, cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None, output_attentions: bool = False, frequencies: Float[Array, "seq_len head_dim"] | None = None, alibi: Float[Array, "batch_or_1 heads qseq_len_or_1 kvseq_len_or_1"] | None = None)` — [`L1194`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L1194) — Multi-head Latent Attention forward path (DeepSeek V2/V3). — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
  - `key_normalization(self)` — [`L845`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L845) — Get key normalization layer.
  - `key_projection(self)` — [`L790`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L790) — Get key projection layer.
  - `mla_kv_a_layernorm(self)` — [`L944`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L944) — Get MLA KV A layer normalization.
  - `mla_kv_a_proj_with_mqa(self)` — [`L931`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L931) — Get MLA KV A projection with multi-query attention.
  - `mla_kv_b_proj(self)` — [`L956`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L956) — Get MLA KV B projection.
  - `mla_q_a_layernorm(self)` — [`L906`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L906) — Get MLA query A layer normalization.
  - `mla_q_a_proj(self)` — [`L893`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L893) — Get MLA query A projection (LoRA down-projection).
  - `mla_q_b_proj(self)` — [`L918`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L918) — Get MLA query B projection (LoRA up-projection).
  - `mla_q_proj(self)` — [`L881`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L881) — Get MLA query projection (non-LoRA variant).
  - `operation(self)` — [`L1637`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L1637) — Get the primary operation from the attention performer.
  - `operation_executor(self)` — [`L1622`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L1622) — Get the OperationExecutor from the underlying attention performer.
  - `operation_requirements(self)` — [`L1649`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L1649) — Get combined requirements from the attention performer.
  - `output_normalization(self)` — [`L869`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L869) — Get output normalization layer.
  - `output_projection(self)` — [`L810`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L810) — Get output projection layer. — documented in [easydel-layers-linears-_linear](../../../../concepts/easydel-layers-linears-_linear.md)
  - `query_key_value_projection(self)` — [`L820`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L820) — Get fused QKV projection layer.
  - `query_normalization(self)` — [`L833`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L833) — Get query normalization layer.
  - `query_projection(self)` — [`L780`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L780) — Get query projection layer.
  - `requires_cache(self)` — [`L1666`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L1666) — Check whether this attention layer requires a KV cache.
  - `value_normalization(self)` — [`L857`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L857) — Get value normalization layer.
  - `value_projection(self)` — [`L800`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L800) — Get value projection layer.
  - `alibi_slopes` — [`L658`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L658)
  - `attention_performer` — [`L397`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L397) — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
  - `attention_type` — [`L326`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L326)
  - `causal` — [`L327`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L327)
  - `dtype` — [`L320`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L320)
  - `head_dim` — [`L338`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L338) — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
  - `hidden_size` — [`L334`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L334)
  - `layer_idx` — [`L324`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L324)
  - `norms_mapping` — [`L268`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L268)
  - `num_heads` — [`L335`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L335) — documented in [easydel-layers-attention-_unified](../../../../concepts/easydel-layers-attention-_unified.md)
  - `num_key_value_groups` — [`L339`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L339)
  - `num_key_value_heads` — [`L337`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L337)
  - `param_dtype` — [`L321`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L321)
  - `precision` — [`L322`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L322)
  - `projection_mapping` — [`L274`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L274)
  - `resid_dropout` — [`L413`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L413)
  - `rotary` — [`L393`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L393)
  - `sliding_window` — [`L328`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L328)
  - `use_fused_qkv` — [`L330`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L330)
  - `use_gqa` — [`L331`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L331)
  - `use_mla_lora` — [`L332`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L332)
  - `use_qk_norm` — [`L329`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L329)
- uses (calls/refs, reference-scoped): [`OperationsMetadata`](../../caching/_abstracts.md#OperationsMetadata), [`RaggedPagesMetadata`](../../caching/ragged_page/cache.md#RaggedPagesMetadata), [`TransformerMetadata`](../../caching/transformer/cache.md#TransformerMetadata), [`ColumnParallelLinear`](../linears/_linear.md#ColumnParallelLinear), [`RMSNorm`](../norms/_norms.md#RMSNorm), [`RaggedPagesCacheView`](../../caching/ragged_page/cache.md#RaggedPagesCacheView), [`TransformerCacheView`](../../caching/transformer/cache.md#TransformerCacheView), [`RowParallelLinear`](../linears/_linear.md#RowParallelLinear), [`forward_mla`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.forward_mla), [`OperationRequirements`](../../operations/requirements/requirements.md#OperationRequirements), [`forward_mla`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.forward_mla), [`FlexibleAttentionModule`](_flexible.md#FlexibleAttentionModule), [`forward`](_flexible.md#FlexibleAttentionModule.forward), [`AttentionLayerOutput`](../../infra/modeling_outputs.md#AttentionLayerOutput), [`concatenate`](_flexible.md#AttentionModule.concatenate), [`cache_view`](../../infra/modeling_outputs.md#AttentionLayerOutput.cache_view), [`forward`](../../modules/dbrx/modeling_dbrx.md#DbrxAttention.forward), [`attention_outputs`](../../operations/_attention_outputs.md#AttentionOutput.attention_outputs), [`forward`](../../modules/openelm/modeling_openelm.md#OpenELMMultiHeadCausalAttention.forward), [`UnifiedAttentionCacheView`](../../caching/unified_attention/cache.md#UnifiedAttentionCacheView), [`forward`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextFullAttention.forward), [`forward_alibi`](../../modules/mosaic_mpt/modeling_mosaic.md#MptAttention.forward_alibi), [`AttentionModule`](_flexible.md#AttentionModule), [`define_network`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2Attention.define_network), [`define_network`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3Attention.define_network), [`define_network`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.define_network), [`define_network`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.define_network), [`define_network`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiMLAAttention.define_network), [`define_network`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2Attention.define_network), [`MLARaggedPagesCacheView`](../../caching/mla_ragged_page/cache.md#MLARaggedPagesCacheView), [`get_basic_rope`](../../infra/base_config.md#EasyDeLBaseConfig.get_basic_rope), [`shard_attention_prod`](_flexible.md#AttentionModule.shard_attention_prod), [`default`](../../operations/requirements/requirements.md#OperationRequirements.default), [`apply_qkv_shardings`](_flexible.md#AttentionModule.apply_qkv_shardings), [`attention_weights`](../../operations/_attention_outputs.md#AttentionOutput.attention_weights), [`operation_executor`](_flexible.md#FlexibleAttentionModule.operation_executor), [`forward`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiMLAAttention.forward), [`kernel`](../linears/_linear.md#ParallelLinear.kernel), [`Cfg`](_unified.md#Cfg), [`_create_fused_qkv_proj`](../../modules/falcon/modeling_falcon.md#FalconAttention._create_fused_qkv_proj)  (+193 more)
- used by: [`forward_mla`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.forward_mla), [`forward_mla`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.forward_mla), [`__call__`](../../modules/stablelm/modeling_stablelm.md#StableLmAttention.__call__), [`forward`](../../modules/dbrx/modeling_dbrx.md#DbrxAttention.forward), [`_forward_with_kv_capture`](../../modules/gemma4/modeling_gemma4.md#Gemma4Attention._forward_with_kv_capture), [`forward`](../../modules/openelm/modeling_openelm.md#OpenELMMultiHeadCausalAttention.forward), [`__call__`](../../modules/phi3/modeling_phi3.md#Phi3Attention.__call__), [`forward`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextFullAttention.forward), [`forward_alibi`](../../modules/mosaic_mpt/modeling_mosaic.md#MptAttention.forward_alibi), [`AttentionModule`](_flexible.md#AttentionModule), [`__call__`](../../modules/gemma4/modeling_gemma4.md#Gemma4Attention.__call__), [`__call__`](../../modules/gpt2/modeling_gpt2.md#GPT2Attention.__call__), [`define_network`](../../modules/deepseek_v2/modeling_deepseek.md#DeepseekV2Attention.define_network), [`define_network`](../../modules/deepseek_v3/modeling_deepseek.md#DeepseekV3Attention.define_network), [`define_network`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteAttention.define_network), [`define_network`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaAttention.define_network), [`define_network`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiMLAAttention.define_network), [`define_network`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2Attention.define_network), [`__call__`](../../modules/glm4v/modeling_glm4v.md#Glm4vVisionAttention.__call__), [`__call__`](../../modules/qwen3_vl/modeling_qwen3_vl.md#Qwen3VLVisionAttention.__call__), [`__call__`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeVisionAttention.__call__), [`_merge_heads`](_flexible.md#AttentionModule._merge_heads), [`forward`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiMLAAttention.forward), [`__call__`](../../modules/qwen2_vl/modeling_qwen2_vl.md#Qwen2VLAttention.__call__), [`_create_fused_qkv_proj`](../../modules/falcon/modeling_falcon.md#FalconAttention._create_fused_qkv_proj), [`_create_o_proj`](../../modules/seed_oss/modeling_seed_oss.md#SeedOssAttention._create_o_proj), [`_create_attention_performer`](../../modules/mosaic_mpt/modeling_mosaic.md#MptAttention._create_attention_performer), [`k_norm`](../../modules/cohere/modeling_cohere.md#CohereAttention.k_norm), [`kv_shared_layer_index`](../../modules/gemma4/modeling_gemma4.md#Gemma4Attention.kv_shared_layer_index), [`q_norm`](../../modules/cohere/modeling_cohere.md#CohereAttention.q_norm), [`_create_o_proj`](../../modules/falcon/modeling_falcon.md#FalconAttention._create_o_proj), [`head_dim`](../../modules/openelm/modeling_openelm.md#OpenELMMultiHeadCausalAttention.head_dim), [`resid_dropout`](../../modules/mosaic_mpt/modeling_mosaic.md#MptAttention.resid_dropout), [`__init__`](../../modules/falcon/modeling_falcon.md#FalconAttention.__init__), [`__init__`](../../modules/phimoe/modeling_phimoe.md#PhiMoEAttention.__init__), [`__init__`](../../modules/qwen3_moe/modeling_qwen3_moe.md#Qwen3MoeAttention.__init__), [`__init__`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeTextAttention.__init__), [`_create_k_norm`](../../modules/olmo2/modeling_olmo2.md#Olmo2Attention._create_k_norm), [`_create_k_norm`](../../modules/olmo3/modeling_olmo3.md#Olmo3Attention._create_k_norm), [`_create_o_proj`](../../modules/glm4v/modeling_glm4v.md#Glm4vTextAttention._create_o_proj)  (+212 more)

## Functions
- `apply_rotary_pos_emb(q: Float[Array, "batch_size num_heads seq_len head_dim"], k: Float[Array, "batch_size num_kv_heads seq_len head_dim"], cos: Float[Array, "max_seq_len head_dim"], sin: Float[Array, "max_seq_len head_dim"], position_ids: Int[Array, "batch_size seq_len"])` — [`L143`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L143) — Apply rotary position embeddings to query and key tensors.
- `rotate_half(x: Float[Array, "batch_size num_heads_any seq_len head_dim"])` — [`L191`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L191) — Rotate half the hidden dims of the input.

## Module values
- `Cfg` — [`L140`](../../../../../../../raw/code/EasyDeL/easydel/layers/attention/_unified.py#L140)

