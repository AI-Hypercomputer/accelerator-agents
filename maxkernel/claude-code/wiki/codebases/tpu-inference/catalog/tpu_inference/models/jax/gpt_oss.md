---
title: 'Module: tpu_inference/models/jax/gpt_oss.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/gpt_oss.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.gpt_oss`/
symbols:
  GptOss.layers: GptOss#layers.
  GptOss.hf_config: GptOss#hf_config.
  GptOss.load_weights: GptOss#load_weights().
  GptOss.lm_head: GptOss#lm_head.
  GptOss.embedder: GptOss#embedder.
  GptOss.final_norm: GptOss#final_norm.
  GptOss.__call__: GptOss#__call__().
  GptOss.rng: GptOss#rng.
  GptOss.random_init: GptOss#random_init.
  GptOss._load_mxfp4: GptOss#_load_mxfp4().
  GptOss._load_regular_param: GptOss#_load_regular_param().
  GptOss.vllm_config: GptOss#vllm_config.
  GptOss.compute_logits: GptOss#compute_logits().
  GptOss.enable_return_routed_experts: GptOss#enable_return_routed_experts.
  GptOss.mesh: GptOss#mesh.
  GptOss.sliding_window: GptOss#sliding_window.
  logger: logger.
  GptOss.apply: GptOss#apply().
  GptOss: GptOss#
  DTYPE_VIEW_MAP: DTYPE_VIEW_MAP.
  GptOss._build_mxfp4_pool: GptOss#_build_mxfp4_pool().
  GptOss.get_q_slice: GptOss#get_q_slice().
  GptOss.get_s_slice: GptOss#get_s_slice().
  GptOss.get_slice: GptOss#get_slice().
  GptOss.__init__: GptOss#__init__().
---
# Module: [`tpu_inference/models/jax/gpt_oss.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py)

## Classes
### `GptOss`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/gpt_oss.py:52`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L52)
- doc: JAX implementation of the GPT-OSS model architecture.
- signature: `class GptOss(nnx.Module):`
- members:
  - `_build_mxfp4_pool(self, names_and_weights_generator, mappings)` — [`L404`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L404) — Collect MXFP4 weights into a pool keeping tuples (blocks_u8, scales_u8).
  - `_load_mxfp4(self, model_weight, codes_fp32_t, scales_fp32_t, transform_fn=None)` — [`L447`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L447) — Assign decoded MXFP4 codes/scales into a QArray (qvalue/scale).
  - `_load_regular_param(self, model_weight, loaded_weight: torch.Tensor, cast_type, transform_fn, target_shape, jax_path_template: str)` — [`L490`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L490) — Assign a regular tensor (non-MXFP4) into the model param with transform applied.
  - `apply(self, variables, *args, **kwargs)` — [`L205`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L205)
  - `compute_logits(self, hidden_states: jax.Array)` — [`L552`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L552)
  - `get_q_slice(index)` — [`L474`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L474)
  - `get_s_slice(index)` — [`L477`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L477)
  - `get_slice(index)` — [`L515`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L515)
  - `load_weights(self, rng: PRNGKey, cache_dir: Optional[str] = None)` — [`L208`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L208) — Loads and transforms all weights from a checkpoint
  - `embedder` — [`L96`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L96)
  - `enable_return_routed_experts` — [`L93`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L93)
  - `final_norm` — [`L184`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L184)
  - `hf_config` — [`L65`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L65)
  - `layers` — [`L105`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L105)
  - `lm_head` — [`L193`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L193)
  - `mesh` — [`L94`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L94)
  - `random_init` — [`L91`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L91)
  - `rng` — [`L66`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L66)
  - `sliding_window` — [`L89`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L89)
  - `vllm_config` — [`L64`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L64)
- protocol/private: `__call__`[`L523`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L523), `__init__`[`L57`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L57)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`encode`](../../layers/jax/layers.md#Embedder.encode), [`get_param`](utils/weight_utils.md#get_param), [`RMSNorm`](../../layers/jax/layers.md#RMSNorm), [`dtype`](../../layers/jax/layers.md#RMSNorm.dtype), [`activation_ffw_td`](../../layers/jax/layers.md#RMSNorm.activation_ffw_td), [`dtype`](../../layers/jax/attention/gpt_oss_attention.md#GptOssAttention.dtype), [`dtype`](../../layers/jax/layers.md#Embedder.dtype), [`head_dim`](../../layers/jax/attention/gpt_oss_attention.md#GptOssAttention.head_dim), [`print_param_info`](utils/weight_utils.md#print_param_info), [`dims`](../../layers/jax/layers.md#RMSNorm.dims), [`epsilon`](../../layers/jax/layers.md#RMSNorm.epsilon), [`random_init`](../../layers/jax/attention/gpt_oss_attention.md#GptOssAttention.random_init), [`random_init`](../../layers/jax/layers.md#RMSNorm.random_init), [`Embedder`](../../layers/jax/layers.md#Embedder), [`decode`](../../layers/jax/layers.md#LMhead.decode), [`dtype`](../../layers/jax/moe/gpt_oss_moe.md#GptOssMoE.dtype), [`dtype`](../../layers/jax/moe/moe.md#Router.dtype), [`hidden_size`](../../layers/jax/layers.md#Embedder.hidden_size), [`rngs`](../../layers/jax/layers.md#RMSNorm.rngs), [`u8_unpack_e2m1`](../../layers/common/__init__.md#u8_unpack_e2m1), [`LMhead`](../../layers/jax/layers.md#LMhead), [`model_weights_generator`](utils/weight_utils.md#model_weights_generator), [`random_init`](../../layers/jax/layers.md#Embedder.random_init), [`vocab_size`](../../layers/jax/layers.md#Embedder.vocab_size), [`TransformerBlock`](../../layers/jax/transformer_block.md#TransformerBlock), [`MXFP4`](../../layers/common/quant_methods.md#MXFP4), [`custom_module`](../../layers/jax/transformer_block.md#TransformerBlock.custom_module), [`random_init`](../../layers/jax/moe/gpt_oss_moe.md#GptOssMoE.random_init), [`KVCacheType`](../../layers/jax/constants.md#KVCacheType.KVCacheType), [`hidden_size`](../../layers/jax/attention/gpt_oss_attention.md#GptOssAttention.hidden_size), [`rngs`](../../layers/jax/layers.md#Embedder.rngs), [`vd_sharding`](../../layers/jax/layers.md#Embedder.vd_sharding), [`GptOssRouter`](../../layers/jax/moe/gpt_oss_moe.md#GptOssRouter), [`attn`](../../layers/jax/transformer_block.md#TransformerBlock.attn), [`e8m0_to_fp32`](../../layers/common/__init__.md#e8m0_to_fp32), [`num_attention_heads`](../../layers/jax/attention/gpt_oss_attention.md#GptOssAttention.num_attention_heads), [`num_key_value_heads`](../../layers/jax/attention/gpt_oss_attention.md#GptOssAttention.num_key_value_heads), [`pre_attention_norm`](../../layers/jax/transformer_block.md#TransformerBlock.pre_attention_norm), [`pre_mlp_norm`](../../layers/jax/transformer_block.md#TransformerBlock.pre_mlp_norm)  (+40 more)
- used by: [`_get_model_architecture`](../common/model_loader.md#_get_model_architecture)

## Module values
- `DTYPE_VIEW_MAP` — [`L44`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L44)
- `logger` — [`L41`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gpt_oss.py#L41)

