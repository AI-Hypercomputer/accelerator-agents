---
title: 'Module: tpu_inference/models/jax/gemma4_mtp.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/gemma4_mtp.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.gemma4_mtp`/
symbols:
  Gemma4MTPAttention.__call__: Gemma4MTPAttention#__call__().
  Gemma4MTPMaskedEmbedder._select_and_score: Gemma4MTPMaskedEmbedder#_select_and_score().
  Gemma4MTPDecoderLayer.__call__: Gemma4MTPDecoderLayer#__call__().
  Gemma4MTPForCausalLM.compute_logits: Gemma4MTPForCausalLM#compute_logits().
  Gemma4MultiTokenPredictor.__call__: Gemma4MultiTokenPredictor#__call__().
  Gemma4MTPForCausalLM._get_full_lm_head_weight: Gemma4MTPForCausalLM#_get_full_lm_head_weight().
  Gemma4MTPAttention.q_proj: Gemma4MTPAttention#q_proj.
  Gemma4MTPAttention.o_proj: Gemma4MTPAttention#o_proj.
  Gemma4MultiTokenPredictor.embed_tokens: Gemma4MultiTokenPredictor#embed_tokens.
  Gemma4MTPForCausalLM.load_weights: Gemma4MTPForCausalLM#load_weights().
  Gemma4MTPAttention.rope_proportion: Gemma4MTPAttention#rope_proportion.
  Gemma4MultiTokenPredictor.layers: Gemma4MultiTokenPredictor#layers.
  init_fn: init_fn.
  Gemma4MTPAttention.q_norm: Gemma4MTPAttention#q_norm.
  Gemma4MTPForCausalLM.masked_embedding: Gemma4MTPForCausalLM#masked_embedding.
  Gemma4MTPForCausalLM.get_top_tokens: Gemma4MTPForCausalLM#get_top_tokens().
  Gemma4MTPAttention.head_dim_original: Gemma4MTPAttention#head_dim_original.
  Gemma4MTPAttention.head_dim: Gemma4MTPAttention#head_dim.
  Gemma4MultiTokenPredictor.pre_projection: Gemma4MultiTokenPredictor#pre_projection.
  Gemma4MultiTokenPredictor.post_projection: Gemma4MultiTokenPredictor#post_projection.
  Gemma4MultiTokenPredictor.norm: Gemma4MultiTokenPredictor#norm.
  Gemma4MTPAttention.is_sliding: Gemma4MTPAttention#is_sliding.
  Gemma4MTPForCausalLM: Gemma4MTPForCausalLM#
  Gemma4MTPAttention.layer_type: Gemma4MTPAttention#layer_type.
  Gemma4MTPForCausalLM.model: Gemma4MTPForCausalLM#model.
  Gemma4MTPAttention.sliding_window: Gemma4MTPAttention#sliding_window.
  Gemma4MTPDecoderLayer.input_layernorm: Gemma4MTPDecoderLayer#input_layernorm.
  Gemma4MTPDecoderLayer.post_attention_layernorm: Gemma4MTPDecoderLayer#post_attention_layernorm.
  Gemma4MTPDecoderLayer.pre_feedforward_layernorm: Gemma4MTPDecoderLayer#pre_feedforward_layernorm.
  Gemma4MTPDecoderLayer.post_feedforward_layernorm: Gemma4MTPDecoderLayer#post_feedforward_layernorm.
  Gemma4MultiTokenPredictor.embed_input_ids: Gemma4MultiTokenPredictor#embed_input_ids().
  Gemma4MTPAttention.num_kv_heads: Gemma4MTPAttention#num_kv_heads.
  Gemma4MultiTokenPredictor.backbone_hidden_size: Gemma4MultiTokenPredictor#backbone_hidden_size.
  Gemma4MTPMaskedEmbedder.vocab_size: Gemma4MTPMaskedEmbedder#vocab_size.
  Gemma4MTPMaskedEmbedder.__call__: Gemma4MTPMaskedEmbedder#__call__().
  Gemma4MultiTokenPredictor.hidden_size: Gemma4MultiTokenPredictor#hidden_size.
  Gemma4MTPForCausalLM.packed_modules_mapping: Gemma4MTPForCausalLM#packed_modules_mapping.
  Gemma4MTPForCausalLM.__call__: Gemma4MTPForCausalLM#__call__().
  Gemma4MTPAttention.kv_cache_quantized_dtype: Gemma4MTPAttention#kv_cache_quantized_dtype.
  Gemma4MTPDecoderLayer.layer_type: Gemma4MTPDecoderLayer#layer_type.
  Gemma4MTPForCausalLM.lm_head: Gemma4MTPForCausalLM#lm_head.
  Gemma4MTPMaskedEmbedder.hidden_size: Gemma4MTPMaskedEmbedder#hidden_size.
  Gemma4MTPMaskedEmbedder: Gemma4MTPMaskedEmbedder#
  Gemma4MTPMaskedEmbedder.num_selected: Gemma4MTPMaskedEmbedder#num_selected.
  Gemma4MTPMaskedEmbedder.centroids: Gemma4MTPMaskedEmbedder#centroids.
  Gemma4MTPMaskedEmbedder.get_top_tokens: Gemma4MTPMaskedEmbedder#get_top_tokens().
  Gemma4MTPAttention: Gemma4MTPAttention#
  Gemma4MTPDecoderLayer: Gemma4MTPDecoderLayer#
  Gemma4MTPDecoderLayer.self_attn: Gemma4MTPDecoderLayer#self_attn.
  Gemma4MTPDecoderLayer.mlp: Gemma4MTPDecoderLayer#mlp.
  Gemma4MultiTokenPredictor: Gemma4MultiTokenPredictor#
  Gemma4MultiTokenPredictor.normalizer: Gemma4MultiTokenPredictor#normalizer.
  Gemma4MTPForCausalLM.clean_and_map: Gemma4MTPForCausalLM#clean_and_map().
  Gemma4MTPAttention.hidden_size: Gemma4MTPAttention#hidden_size.
  Gemma4MTPAttention.num_heads: Gemma4MTPAttention#num_heads.
  Gemma4MTPForCausalLM.final_logit_softcapping: Gemma4MTPForCausalLM#final_logit_softcapping.
  logger: logger.
  Gemma4MTPAttention.__init__: Gemma4MTPAttention#__init__().
  Gemma4MTPDecoderLayer.__init__: Gemma4MTPDecoderLayer#__init__().
  Gemma4MTPDecoderLayer.is_sliding: Gemma4MTPDecoderLayer#is_sliding.
  Gemma4MTPForCausalLM.WeightLoader: Gemma4MTPForCausalLM#WeightLoader.
  Gemma4MTPMaskedEmbedder.vocab_size_per_centroid: Gemma4MTPMaskedEmbedder#vocab_size_per_centroid.
  Gemma4MTPAttention.rope_theta: Gemma4MTPAttention#rope_theta.
  Gemma4MTPAttention.rope_scaling: Gemma4MTPAttention#rope_scaling.
  Gemma4MTPMaskedEmbedder.num_centroids: Gemma4MTPMaskedEmbedder#num_centroids.
  Gemma4MTPMaskedEmbedder.centroid_intermediate_top_k: Gemma4MTPMaskedEmbedder#centroid_intermediate_top_k.
  Gemma4MTPMaskedEmbedder.token_ordering: Gemma4MTPMaskedEmbedder#token_ordering.
  Gemma4MTPAttention.rms_norm_eps: Gemma4MTPAttention#rms_norm_eps.
  Gemma4MTPAttention.scaling: Gemma4MTPAttention#scaling.
  Gemma4MTPAttention.mesh: Gemma4MTPAttention#mesh.
  Gemma4MTPDecoderLayer.layer_scalar: Gemma4MTPDecoderLayer#layer_scalar.
  Gemma4MultiTokenPredictor.config: Gemma4MultiTokenPredictor#config.
  Gemma4MultiTokenPredictor.vocab_size: Gemma4MultiTokenPredictor#vocab_size.
  Gemma4MultiTokenPredictor.num_mtp_layers: Gemma4MultiTokenPredictor#num_mtp_layers.
  Gemma4MTPForCausalLM.vllm_config: Gemma4MTPForCausalLM#vllm_config.
  Gemma4MTPMaskedEmbedder.__init__: Gemma4MTPMaskedEmbedder#__init__().
  Gemma4MTPAttention.is_kv_shared_layer: Gemma4MTPAttention#is_kv_shared_layer.
  Gemma4MultiTokenPredictor.__init__: Gemma4MultiTokenPredictor#__init__().
  Gemma4MultiTokenPredictor.start_layer: Gemma4MultiTokenPredictor#start_layer.
  Gemma4MultiTokenPredictor.end_layer: Gemma4MultiTokenPredictor#end_layer.
  Gemma4MTPForCausalLM.__init__: Gemma4MTPForCausalLM#__init__().
  Gemma4MTPForCausalLM.mesh: Gemma4MTPForCausalLM#mesh.
---
# Module: [`tpu_inference/models/jax/gemma4_mtp.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py)

## Classes
### `Gemma4MTPAttention`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/gemma4_mtp.py:148`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L148)
- doc: Q-only attention for Gemma4 MTP layers.
- signature: `class Gemma4MTPAttention(JaxModule):`
- members:
  - `head_dim` — [`L209`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L209)
  - `head_dim_original` — [`L197`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L197)
  - `hidden_size` — [`L167`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L167)
  - `is_kv_shared_layer` — [`L248`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L248)
  - `is_sliding` — [`L178`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L178)
  - `kv_cache_quantized_dtype` — [`L249`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L249)
  - `layer_type` — [`L173`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L173)
  - `mesh` — [`L210`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L210)
  - `num_heads` — [`L168`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L168)
  - `num_kv_heads` — [`L204`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L204)
  - `o_proj` — [`L235`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L235)
  - `q_norm` — [`L225`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L225)
  - `q_proj` — [`L212`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L212)
  - `rms_norm_eps` — [`L169`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L169)
  - `rope_proportion` — [`L188`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L188)
  - `rope_scaling` — [`L186`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L186)
  - `rope_theta` — [`L184`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L184)
  - `scaling` — [`L171`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L171)
  - `sliding_window` — [`L179`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L179)
- protocol/private: `__call__`[`L254`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L254), `__init__`[`L155`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L155)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxEinsum`](../../layers/jax/linear.md#JaxEinsum), [`VllmQuantConfig`](../../layers/vllm/quantization/configs.md#VllmQuantConfig), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`input_positions`](../../layers/common/attention_metadata.md#AttentionMetadata.input_positions), [`apply_rope`](../../layers/jax/rope_interface.md#apply_rope), [`attention`](../../layers/common/attention_interface.md#attention), [`init_fn`](gemma4_mtp.md#init_fn)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`self_attn`](gemma4_mtp.md#Gemma4MTPDecoderLayer.self_attn)

### `Gemma4MTPDecoderLayer`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/gemma4_mtp.py:296`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L296)
- signature: `class Gemma4MTPDecoderLayer(JaxModule):`
- members:
  - `input_layernorm` — [`L321`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L321)
  - `is_sliding` — [`L318`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L318)
  - `layer_scalar` — [`L319`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L319)
  - `layer_type` — [`L313`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L313)
  - `mlp` — [`L360`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L360)
  - `post_attention_layernorm` — [`L340`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L340)
  - `post_feedforward_layernorm` — [`L368`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L368)
  - `pre_feedforward_layernorm` — [`L350`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L350)
  - `self_attn` — [`L330`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L330)
- protocol/private: `__call__`[`L379`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L379), `__init__`[`L298`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L298)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`VllmQuantConfig`](../../layers/vllm/quantization/configs.md#VllmQuantConfig), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`init_fn`](gemma4_mtp.md#init_fn), [`Gemma4MLP`](gemma4.md#Gemma4MLP), [`Gemma4MTPAttention`](gemma4_mtp.md#Gemma4MTPAttention)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`layers`](gemma4_mtp.md#Gemma4MultiTokenPredictor.layers)

### `Gemma4MTPForCausalLM`  ·  implements/extends JaxModule, LoadableWithIterator
- def: [`tpu_inference/models/jax/gemma4_mtp.py:523`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L523)
- signature: `class Gemma4MTPForCausalLM(JaxModule, LoadableWithIterator):`
- members:
  - `clean_and_map(w_iter)` — [`L579`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L579)
  - `compute_logits(self, hidden_states: jax.Array)` — [`L667`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L667)
  - `get_top_tokens(self, hidden_states: jax.Array)` — [`L684`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L684)
  - `load_weights(self, weights: Iterable[Tuple[str, Any]])` — [`L575`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L575)
  - `WeightLoader` — [`L525`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L525)
  - `final_logit_softcapping` — [`L549`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L549)
  - `lm_head` — [`L552`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L552)
  - `masked_embedding` — [`L563`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L563)
  - `mesh` — [`L536`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L536)
  - `model` — [`L538`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L538)
  - `packed_modules_mapping` — [`L524`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L524)
  - `vllm_config` — [`L534`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L534)
- protocol/private: `__call__`[`L637`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L637), `__init__`[`L527`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L527), `_get_full_lm_head_weight`[`L661`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L661)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`LoadableWithIterator`](utils/weight_utils.md#LoadableWithIterator), [`JaxLmHead`](../../layers/jax/linear.md#JaxLmHead), [`StandardWeightLoader`](utils/weight_utils.md#StandardWeightLoader), [`embed_tokens`](gemma4_mtp.md#Gemma4MultiTokenPredictor.embed_tokens), [`layers`](gemma4_mtp.md#Gemma4MultiTokenPredictor.layers), [`weight`](../../layers/jax/linear.md#JaxLmHead.weight), [`decode`](../../layers/jax/embed.md#JaxEmbed.decode), [`Gemma4ForCausalLM`](gemma4.md#Gemma4ForCausalLM), [`load_weights`](utils/weight_utils.md#JaxAutoWeightsLoader.load_weights), `embedding`, [`JaxAutoWeightsLoader`](utils/weight_utils.md#JaxAutoWeightsLoader), [`Gemma4MTPMaskedEmbedder`](gemma4_mtp.md#Gemma4MTPMaskedEmbedder), [`Gemma4MultiTokenPredictor`](gemma4_mtp.md#Gemma4MultiTokenPredictor), [`get_top_tokens`](gemma4_mtp.md#Gemma4MTPMaskedEmbedder.get_top_tokens), [`packed_modules_mapping`](gemma4.md#Gemma4ForCausalLM.packed_modules_mapping)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`_get_model_architecture`](../common/model_loader.md#_get_model_architecture), [`LoadableWithIterator`](utils/weight_utils.md#LoadableWithIterator), [`load_weights`](utils/weight_utils.md#LoadableWithIterator.load_weights)

### `Gemma4MTPMaskedEmbedder`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/gemma4_mtp.py:44`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L44)
- doc: Sparse logit computation via centroid-based vocabulary masking in JAX.
- signature: `class Gemma4MTPMaskedEmbedder(JaxModule):`
- members:
  - `__call__(self, hidden_states: jax.Array, lm_head_weight: jax.Array)` — [`L120`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L120) — Full-vocab logits with non-selected positions masked to -inf.
  - `_select_and_score(self, hidden_states: jax.Array, lm_head_weight: jax.Array)` — [`L83`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L83) — Centroid selection + sparse dot product.
  - `get_top_tokens(self, hidden_states: jax.Array, lm_head_weight: jax.Array)` — [`L137`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L137) — Sparse argmax — returns vocab token IDs directly.
  - `centroid_intermediate_top_k` — [`L65`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L65)
  - `centroids` — [`L69`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L69)
  - `hidden_size` — [`L62`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L62)
  - `num_centroids` — [`L64`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L64)
  - `num_selected` — [`L67`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L67)
  - `token_ordering` — [`L78`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L78)
  - `vocab_size` — [`L63`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L63)
  - `vocab_size_per_centroid` — [`L66`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L66)
- protocol/private: `__init__`[`L51`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L51)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`JaxLinear`](../../layers/jax/linear.md#JaxLinear)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`get_top_tokens`](gemma4_mtp.md#Gemma4MTPForCausalLM.get_top_tokens), [`masked_embedding`](gemma4_mtp.md#Gemma4MTPForCausalLM.masked_embedding)

### `Gemma4MultiTokenPredictor`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/gemma4_mtp.py:407`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L407)
- signature: `class Gemma4MultiTokenPredictor(JaxModule):`
- members:
  - `embed_input_ids(self, input_ids: jax.Array)` — [`L486`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L486)
  - `backbone_hidden_size` — [`L424`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L424)
  - `config` — [`L420`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L420)
  - `embed_tokens` — [`L430`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L430)
  - `end_layer` — [`L460`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L460)
  - `hidden_size` — [`L423`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L423)
  - `layers` — [`L460`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L460)
  - `norm` — [`L474`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L474)
  - `normalizer` — [`L484`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L484)
  - `num_mtp_layers` — [`L428`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L428)
  - `post_projection` — [`L450`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L450)
  - `pre_projection` — [`L440`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L440)
  - `start_layer` — [`L460`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L460)
  - `vocab_size` — [`L427`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L427)
- protocol/private: `__call__`[`L489`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L489), `__init__`[`L409`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L409)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`make_layers`](../../layers/jax/pp_utils.md#make_layers), [`JaxLinear`](../../layers/jax/linear.md#JaxLinear), [`JaxEmbed`](../../layers/jax/embed.md#JaxEmbed), [`init_fn`](gemma4_mtp.md#init_fn), [`Gemma4MTPDecoderLayer`](gemma4_mtp.md#Gemma4MTPDecoderLayer)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`compute_logits`](gemma4_mtp.md#Gemma4MTPForCausalLM.compute_logits), [`_get_full_lm_head_weight`](gemma4_mtp.md#Gemma4MTPForCausalLM._get_full_lm_head_weight), [`load_weights`](gemma4_mtp.md#Gemma4MTPForCausalLM.load_weights), [`model`](gemma4_mtp.md#Gemma4MTPForCausalLM.model)

## Module values
- `init_fn` — [`L41`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L41)
- `logger` — [`L39`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mtp.py#L39)

