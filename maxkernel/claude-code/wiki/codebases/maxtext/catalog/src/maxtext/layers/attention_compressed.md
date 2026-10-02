---
title: 'Module: src/maxtext/layers/attention_compressed.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/attention_compressed.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.attention_compressed`/
symbols:
  DeepseekV4Indexer.__call__: DeepseekV4Indexer#__call__().
  DeepseekV4CSACompressor.__call__: DeepseekV4CSACompressor#__call__().
  CompressedAttention.__call__: CompressedAttention#__call__().
  DeepseekV4HCACompressor.__call__: DeepseekV4HCACompressor#__call__().
  CompressedAttention.compressed_query_projection: CompressedAttention#compressed_query_projection().
  DeepseekV4Indexer.q_proj: DeepseekV4Indexer#q_proj.
  DeepseekV4Indexer.kv_norm: DeepseekV4Indexer#kv_norm.
  CompressedAttention.compress_ratio: CompressedAttention#compress_ratio.
  DeepseekV4Indexer.kv_proj: DeepseekV4Indexer#kv_proj.
  DeepseekV4Indexer.gate_proj: DeepseekV4Indexer#gate_proj.
  DeepseekV4Indexer.weights_proj: DeepseekV4Indexer#weights_proj.
  BaseDeepseekCompressor.compress_rate: BaseDeepseekCompressor#compress_rate.
  BaseDeepseekCompressor.kv_proj: BaseDeepseekCompressor#kv_proj.
  BaseDeepseekCompressor.gate_proj: BaseDeepseekCompressor#gate_proj.
  BaseDeepseekCompressor.kv_norm: BaseDeepseekCompressor#kv_norm.
  DeepseekV4Indexer.index_head_dim: DeepseekV4Indexer#index_head_dim.
  DeepseekV4Indexer.position_bias: DeepseekV4Indexer#position_bias.
  CompressedAttention.hca_compressor: CompressedAttention#hca_compressor.
  CompressedAttention.csa_compressor: CompressedAttention#csa_compressor.
  CompressedAttention.compressed_kv_projection: CompressedAttention#compressed_kv_projection().
  CompressedAttention.rotary_embedding: CompressedAttention#rotary_embedding.
  BaseDeepseekCompressor: BaseDeepseekCompressor#
  BaseDeepseekCompressor.dtype: BaseDeepseekCompressor#dtype.
  BaseDeepseekCompressor.position_bias: BaseDeepseekCompressor#position_bias.
  DeepseekV4Indexer.weight_dtype: DeepseekV4Indexer#weight_dtype.
  BaseDeepseekCompressor.rngs: BaseDeepseekCompressor#rngs.
  DeepseekV4Indexer.index_n_heads: DeepseekV4Indexer#index_n_heads.
  DeepseekV4Indexer.dtype: DeepseekV4Indexer#dtype.
  DeepseekV4Indexer.rngs: DeepseekV4Indexer#rngs.
  BaseDeepseekCompressor.head_dim: BaseDeepseekCompressor#head_dim.
  DeepseekV4Indexer.compress_rate: DeepseekV4Indexer#compress_rate.
  DeepseekV4HCACompressor: DeepseekV4HCACompressor#
  DeepseekV4Indexer.softmax_scale: DeepseekV4Indexer#softmax_scale.
  DeepseekV4Indexer.weights_scaling: DeepseekV4Indexer#weights_scaling.
  DeepseekV4CSACompressor: DeepseekV4CSACompressor#
  DeepseekV4CSACompressor.indexer: DeepseekV4CSACompressor#indexer.
  CompressedAttention.wq_a: CompressedAttention#wq_a.
  CompressedAttention.q_norm: CompressedAttention#q_norm.
  CompressedAttention.wq_b: CompressedAttention#wq_b.
  CompressedAttention.kv_norm: CompressedAttention#kv_norm.
  BaseDeepseekCompressor.config: BaseDeepseekCompressor#config.
  BaseDeepseekCompressor.weight_dtype: BaseDeepseekCompressor#weight_dtype.
  CompressedAttention.q_lora_rank: CompressedAttention#q_lora_rank.
  DeepseekV4HCACompressor.__init__: DeepseekV4HCACompressor#__init__().
  DeepseekV4CSACompressor.__init__: DeepseekV4CSACompressor#__init__().
  compressed_attention: compressed_attention().
  csa_overlap_pooling: csa_overlap_pooling().
  BaseDeepseekCompressor.__init__: BaseDeepseekCompressor#__init__().
  BaseDeepseekCompressor.rotary_emb: BaseDeepseekCompressor#rotary_emb.
  DeepseekV4Indexer.index_topk: DeepseekV4Indexer#index_topk.
  DeepseekV4Indexer.rotary_emb: DeepseekV4Indexer#rotary_emb.
  DeepseekV4Indexer: DeepseekV4Indexer#
  DeepseekV4Indexer.config: DeepseekV4Indexer#config.
  CompressedAttention: CompressedAttention#
  CompressedAttention.sinks: CompressedAttention#sinks.
  CompressedAttention.q_up_norm: CompressedAttention#q_up_norm.
  CompressedAttention.wkv: CompressedAttention#wkv.
  CompressedAttention.softmax_scale: CompressedAttention#softmax_scale.
  CompressedAttention.o_a_proj: CompressedAttention#o_a_proj.
  CompressedAttention.o_b_proj: CompressedAttention#o_b_proj.
  BaseDeepseekCompressor.model_mode: BaseDeepseekCompressor#model_mode.
  DeepseekV4Indexer.__init__: DeepseekV4Indexer#__init__().
  CompressedAttention.__init__: CompressedAttention#__init__().
  CompressedAttention._init_projections: CompressedAttention#_init_projections().
  CompressedAttention.out_head_dim: CompressedAttention#out_head_dim().
---
# Module: [`src/maxtext/layers/attention_compressed.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py)

## Classes
### `BaseDeepseekCompressor`  ·  implements/extends Module
- def: [`src/maxtext/layers/attention_compressed.py:129`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L129) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
- doc: Shared base class for DeepSeek-V4 long-range attention compressors.
- signature: `class BaseDeepseekCompressor(nnx.Module):`
- members:
  - `compress_rate` — [`L154`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L154) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `config` — [`L153`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L153)
  - `dtype` — [`L156`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L156)
  - `gate_proj` — [`L177`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L177) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `head_dim` — [`L155`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L155) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `kv_norm` — [`L193`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L193) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `kv_proj` — [`L163`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L163) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `model_mode` — [`L158`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L158)
  - `position_bias` — [`L191`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L191) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `rngs` — [`L159`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L159)
  - `rotary_emb` — [`L202`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L202)
  - `weight_dtype` — [`L157`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L157)
- protocol/private: `__init__`[`L142`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L142)
- uses (calls/refs, reference-scoped): [`DeepseekV4CSACompressor`](attention_compressed.md#DeepseekV4CSACompressor), [`DeepseekV4HCACompressor`](attention_compressed.md#DeepseekV4HCACompressor)
- used by: [`__call__`](attention_compressed.md#DeepseekV4CSACompressor.__call__), [`__call__`](attention_compressed.md#DeepseekV4HCACompressor.__call__), [`DeepseekV4CSACompressor`](attention_compressed.md#DeepseekV4CSACompressor), [`DeepseekV4HCACompressor`](attention_compressed.md#DeepseekV4HCACompressor), [`__init__`](attention_compressed.md#DeepseekV4CSACompressor.__init__), [`__init__`](attention_compressed.md#DeepseekV4HCACompressor.__init__)

### `CompressedAttention`  ·  implements/extends Attention
- def: [`src/maxtext/layers/attention_compressed.py:639`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L639)
- doc: Compressed Attention layer (DeepSeek-V4).
- signature: `class CompressedAttention(Attention):`
- members:
  - `__call__(self, inputs_q: Array, inputs_kv: Array, decoder_segment_ids: Array, inputs_positions: Array, deterministic: bool, model_mode: str = MODEL_MODE_TRAIN, **kwargs)` — [`L951`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L951) — Forward pass for Compressed Attention. — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `__init__(self, config: Config, num_query_heads: int, num_kv_heads: int, head_dim: int, max_target_length: int, mesh: Mesh, attention_kernel: str, inputs_q_shape: Tuple, inputs_kv_shape: Tuple, dtype: DType = jnp.float32, weight_dtype: DType = jnp.float32, max_prefill_predict_length: int = -1, dropout_rate: float = 0, kernel_init: NdInitializer = nd_dense_init(1, "fan_in", "normal"), float32_qk_product: bool = False, float32_logits: bool = False, quant: Optional[Quant] = None, kv_quant: Optional[KVQuant] = None, attention_type: AttentionType = AttentionType.COMPRESSED, attn_logits_soft_cap: float | None = None, sliding_window_size: int | None = None, use_ragged_attention: bool = False, ragged_block_size: int = 256, use_qk_norm: bool = False, query_pre_attn_scalar: float | None = None, use_bias_in_projections: bool = False, q_lora_rank: int = 1536, compress_ratio: int = 0, name: str | None = None, rngs: Optional[nnx.Rngs] = None, **kwargs)` — [`L648`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L648) — Inherits all standard Attention hyperparameters and selectively instantiates
  - `_init_projections(self, inputs_q_shape: Tuple, inputs_kv_shape: Tuple)` — [`L741`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L741) — Initializes the compressed projections and Unweighted RMSNorms.
  - `compressed_kv_projection(self, inputs_kv: Array, inputs_positions: Array, model_mode)` — [`L923`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L923) — KV projection for Compressed Attention. — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `compressed_query_projection(self, inputs_q: Array, inputs_positions: Array, model_mode)` — [`L885`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L885) — Query projection for Compressed Attention. — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `out_head_dim(self)` — [`L881`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L881) — Returns the head dimension used prior to the output projection.
  - `compress_ratio` — [`L698`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L698) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `csa_compressor` — [`L837`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L837) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `hca_compressor` — [`L827`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L827) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `kv_norm` — [`L806`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L806) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `o_a_proj` — [`L854`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L854) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `o_b_proj` — [`L866`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L866) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `q_lora_rank` — [`L697`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L697) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `q_norm` — [`L758`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L758) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `q_up_norm` — [`L781`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L781) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `rotary_embedding` — [`L819`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L819) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `sinks` — [`L739`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L739) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `softmax_scale` — [`L848`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L848) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `wkv` — [`L792`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L792) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `wq_a` — [`L744`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L744) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `wq_b` — [`L767`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L767) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
- uses (calls/refs, reference-scoped): [`DeepseekV4CSACompressor`](attention_compressed.md#DeepseekV4CSACompressor), [`DeepseekV4HCACompressor`](attention_compressed.md#DeepseekV4HCACompressor)
- used by: [`compressed_attention`](attention_compressed.md#compressed_attention)

### `DeepseekV4CSACompressor`  ·  implements/extends BaseDeepseekCompressor
- def: [`src/maxtext/layers/attention_compressed.py:518`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L518) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
- doc: Compressed Sparse Attention compressor (DeepSeek-V4 paper §2.3.1).
- signature: `class DeepseekV4CSACompressor(BaseDeepseekCompressor):`
- members:
  - `__call__(self, hidden_states: Array, q_latent: Array, position_ids: Array, attention_mask: Optional[Array] = None)` — [`L565`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L565) — Forward pass for the CSA compressor. — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `__init__(self, config: Any, compress_ratio: int, rotary_embedding: Any, kernel_init: Any = nnx.initializers.normal(stddev=0.02), quant: Optional[Quant] = None, model_mode: str = MODEL_MODE_TRAIN, rngs: Optional[nnx.Rngs] = None)` — [`L531`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L531) — Initializes the CSA Compressor.
  - `indexer` — [`L556`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L556) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
- uses (calls/refs, reference-scoped): [`compress_rate`](attention_compressed.md#BaseDeepseekCompressor.compress_rate), [`gate_proj`](attention_compressed.md#BaseDeepseekCompressor.gate_proj), [`kv_norm`](attention_compressed.md#BaseDeepseekCompressor.kv_norm), [`kv_proj`](attention_compressed.md#BaseDeepseekCompressor.kv_proj), [`BaseDeepseekCompressor`](attention_compressed.md#BaseDeepseekCompressor), [`dtype`](attention_compressed.md#BaseDeepseekCompressor.dtype), [`position_bias`](attention_compressed.md#BaseDeepseekCompressor.position_bias), [`head_dim`](attention_compressed.md#BaseDeepseekCompressor.head_dim), [`__init__`](attention_compressed.md#BaseDeepseekCompressor.__init__), [`csa_overlap_pooling`](attention_compressed.md#csa_overlap_pooling), [`rotary_emb`](attention_compressed.md#BaseDeepseekCompressor.rotary_emb), [`DeepseekV4Indexer`](attention_compressed.md#DeepseekV4Indexer)
- used by: [`csa_compressor`](attention_compressed.md#CompressedAttention.csa_compressor), [`BaseDeepseekCompressor`](attention_compressed.md#BaseDeepseekCompressor)

### `DeepseekV4HCACompressor`  ·  implements/extends BaseDeepseekCompressor
- def: [`src/maxtext/layers/attention_compressed.py:205`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L205) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
- doc: Heavily Compressed Attention compressor.
- signature: `class DeepseekV4HCACompressor(BaseDeepseekCompressor):`
- members:
  - `__call__(self, hidden_states: Array, q_normed: Array, position_ids: Array)` — [`L244`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L244) — Forward pass for the HCA compressor. — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `__init__(self, config: Any, compress_ratio: int, rotary_embedding: Any, kernel_init: Any = nnx.initializers.normal(stddev=0.02), quant: Optional[Quant] = None, model_mode: str = MODEL_MODE_TRAIN, rngs: Optional[nnx.Rngs] = None)` — [`L219`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L219) — Initializes the HCA Compressor.
- uses (calls/refs, reference-scoped): [`compress_rate`](attention_compressed.md#BaseDeepseekCompressor.compress_rate), [`gate_proj`](attention_compressed.md#BaseDeepseekCompressor.gate_proj), [`kv_norm`](attention_compressed.md#BaseDeepseekCompressor.kv_norm), [`kv_proj`](attention_compressed.md#BaseDeepseekCompressor.kv_proj), [`BaseDeepseekCompressor`](attention_compressed.md#BaseDeepseekCompressor), [`dtype`](attention_compressed.md#BaseDeepseekCompressor.dtype), [`position_bias`](attention_compressed.md#BaseDeepseekCompressor.position_bias), [`head_dim`](attention_compressed.md#BaseDeepseekCompressor.head_dim), [`__init__`](attention_compressed.md#BaseDeepseekCompressor.__init__), [`rotary_emb`](attention_compressed.md#BaseDeepseekCompressor.rotary_emb)
- used by: [`hca_compressor`](attention_compressed.md#CompressedAttention.hca_compressor), [`BaseDeepseekCompressor`](attention_compressed.md#BaseDeepseekCompressor)

### `DeepseekV4Indexer`  ·  implements/extends Module
- def: [`src/maxtext/layers/attention_compressed.py:319`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L319) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
- doc: Indexer module for Compressed Sparse Attention (DeepSeek-V4 paper §2.3.1).
- signature: `class DeepseekV4Indexer(nnx.Module):`
- members:
  - `__init__(self, config: Any, compress_ratio: int, rotary_embedding: Any, kernel_init: Any = nnx.initializers.normal(stddev=0.02), quant: Optional[Quant] = None, rngs: Optional[nnx.Rngs] = None)` — [`L334`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L334) — Initializes the Indexer for CSA.
  - `compress_rate` — [`L354`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L354)
  - `config` — [`L353`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L353)
  - `dtype` — [`L360`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L360)
  - `gate_proj` — [`L391`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L391) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `index_head_dim` — [`L356`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L356) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `index_n_heads` — [`L355`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L355) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `index_topk` — [`L357`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L357) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `kv_norm` — [`L407`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L407) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `kv_proj` — [`L378`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L378) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `position_bias` — [`L405`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L405) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `q_proj` — [`L364`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L364) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `rngs` — [`L362`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L362)
  - `rotary_emb` — [`L430`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L430)
  - `softmax_scale` — [`L358`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L358) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `weight_dtype` — [`L361`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L361)
  - `weights_proj` — [`L416`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L416) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
  - `weights_scaling` — [`L359`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L359) — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)
- protocol/private: `__call__`[`L432`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L432)
- uses (calls/refs, reference-scoped): [`csa_overlap_pooling`](attention_compressed.md#csa_overlap_pooling)
- used by: [`indexer`](attention_compressed.md#DeepseekV4CSACompressor.indexer)

## Functions
- `compressed_attention(*, config: Config, num_query_heads: int, num_kv_heads: int, head_dim: int, max_target_length: int, mesh: Mesh, attention_kernel: str, inputs_q_shape: Tuple, inputs_kv_shape: Tuple, dtype: DType = jnp.float32, weight_dtype: DType = jnp.float32, max_prefill_predict_length: int = -1, dropout_rate: float = 0, kernel_init: NdInitializer = nd_dense_init(1, "fan_in", "normal"), float32_qk_product: bool = False, float32_logits: bool = False, quant: Optional[Quant] = None, kv_quant: Optional[KVQuant] = None, attention_type: AttentionType = AttentionType.COMPRESSED, attn_logits_soft_cap: float | None = None, sliding_window_size: int | None = None, use_ragged_attention: bool = False, ragged_block_size: int = 256, use_qk_norm: bool = False, query_pre_attn_scalar: float | None = None, use_bias_in_projections: bool = False, q_lora_rank: int = 1536, name: str | None = None)` — [`L1058`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L1058) — Wrapper to create the CompressedAttention linen module.
- `csa_overlap_pooling(hidden_states: Array, kv_proj: Any, gate_proj: Any, position_bias: Array, kv_norm: Any, compress_rate: int, head_dim: int)` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/layers/attention_compressed.py#L45) — Shared utility for Compressed Sparse Attention (CSA) overlap pooling. — documented in [maxtext-layers-attention_compressed](../../../../concepts/maxtext-layers-attention_compressed.md)

