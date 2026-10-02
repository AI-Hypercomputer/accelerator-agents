---
title: 'Module: tpu_inference/layers/jax/layers.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/layers.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.layers`/
symbols:
  DenseFFW.__call__: DenseFFW#__call__().
  FlaxUtils: FlaxUtils#
  Embedder.__post_init__: Embedder#__post_init__().
  Embedder.encode: Embedder#encode().
  LMhead.__post_init__: LMhead#__post_init__().
  RMSNorm: RMSNorm#
  RMSNorm.dtype: RMSNorm#dtype.
  RMSNorm.activation_ffw_td: RMSNorm#activation_ffw_td.
  DenseFFW.kernel_gating_DF: DenseFFW#kernel_gating_DF.
  DenseFFW.kernel_up_proj_DF: DenseFFW#kernel_up_proj_DF.
  DenseFFW.kernel_down_proj_FD: DenseFFW#kernel_down_proj_FD.
  Embedder.dtype: Embedder#dtype.
  Embedder.decode: Embedder#decode().
  FlaxUtils.ACT2FN: FlaxUtils#ACT2FN.
  RMSNorm.dims: RMSNorm#dims.
  RMSNorm.random_init: RMSNorm#random_init.
  RMSNorm.epsilon: RMSNorm#epsilon.
  RMSNorm.__call__: RMSNorm#__call__().
  RMSNorm.__post_init__: RMSNorm#__post_init__().
  Embedder: Embedder#
  LMhead.decode: LMhead#decode().
  RMSNorm.rngs: RMSNorm#rngs.
  DenseFFW: DenseFFW#
  Embedder.hidden_size: Embedder#hidden_size.
  LMhead: LMhead#
  Embedder.vocab_size: Embedder#vocab_size.
  Embedder.random_init: Embedder#random_init.
  DenseFFW.dtype: DenseFFW#dtype.
  RMSNorm.with_scale: RMSNorm#with_scale.
  DenseFFW.random_init: DenseFFW#random_init.
  Embedder.vd_sharding: Embedder#vd_sharding.
  Embedder.rngs: Embedder#rngs.
  DenseFFW.df_sharding: DenseFFW#df_sharding.
  DenseFFW.__post_init__: DenseFFW#__post_init__().
  Embedder.__call__: Embedder#__call__().
  DenseFFW.hidden_act: DenseFFW#hidden_act.
  DenseFFW.hidden_size: DenseFFW#hidden_size.
  DenseFFW.intermediate_size: DenseFFW#intermediate_size.
  DenseFFW.fd_sharding: DenseFFW#fd_sharding.
  DenseFFW.activation_ffw_td: DenseFFW#activation_ffw_td.
  DenseFFW.mesh: DenseFFW#mesh.
  LMhead.dv_sharding: LMhead#dv_sharding.
  modeling_flax_utils: modeling_flax_utils.
  DenseFFW.rngs: DenseFFW#rngs.
  Embedder.prelogit_td: Embedder#prelogit_td.
  LMhead.input_embedding_table_DV: LMhead#input_embedding_table_DV.
  LMhead.__call__: LMhead#__call__().
  Embedder.input_embedding_table_VD: Embedder#input_embedding_table_VD.
  RMSNorm.scale: RMSNorm#scale.
  Embedder.normalize_embeddings: Embedder#normalize_embeddings.
  RuntimeParams: RuntimeParams#
  RuntimeParams.kv_cache: RuntimeParams#kv_cache.
  RuntimeParams.sharding_cfg: RuntimeParams#sharding_cfg.
  RuntimeParams.quantization: RuntimeParams#quantization.
---
# Module: [`tpu_inference/layers/jax/layers.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py)

## Classes
### `DenseFFW`  ·  implements/extends Module
- def: [`tpu_inference/layers/jax/layers.py:115`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L115)
- doc: A Gated Feed-Forward Network (FFN) layer.
- signature: `class DenseFFW(nnx.Module):`
- members:
  - `__call__(self, x_TD)` — [`L137`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L137) — Performs the forward pass of the FFW layer.
  - `activation_ffw_td` — [`L131`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L131)
  - `df_sharding` — [`L129`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L129)
  - `dtype` — [`L125`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L125)
  - `fd_sharding` — [`L130`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L130)
  - `hidden_act` — [`L126`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L126)
  - `hidden_size` — [`L127`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L127)
  - `intermediate_size` — [`L128`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L128)
  - `kernel_down_proj_FD` — [`L179`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L179)
  - `kernel_gating_DF` — [`L169`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L169)
  - `kernel_up_proj_DF` — [`L174`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L174)
  - `mesh` — [`L133`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L133)
  - `random_init` — [`L132`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L132)
  - `rngs` — [`L135`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L135)
- protocol/private: `__post_init__`[`L165`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L165)
- uses (calls/refs, reference-scoped): [`create_param`](base.md#create_param), [`ACT2FN`](layers.md#FlaxUtils.ACT2FN), [`modeling_flax_utils`](layers.md#modeling_flax_utils)
- used by: [`end_layer`](../../models/jax/llama4.md#Llama4ForCausalLM.end_layer), [`end_layer`](../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.end_layer), [`__call__`](transformer_block.md#SharedExpertsTransformerBlock.__call__), [`dense_ffw`](transformer_block.md#SharedExpertsTransformerBlock.dense_ffw), [`shared_experts`](transformer_block.md#SharedExpertsTransformerBlock.shared_experts)

### `Embedder`  ·  implements/extends Module
- def: [`tpu_inference/layers/jax/layers.py:188`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L188)
- doc: A module for token embedding and, optionally, decoding (tied embeddings).
- signature: `class Embedder(nnx.Module):`
- members:
  - `__call__(self, x, decode=False)` — [`L214`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L214) — Dispatches to either the encode or decode method.
  - `decode(self, x_TD: Float)` — [`L231`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L231) — Projects hidden states to vocabulary logits.
  - `encode(self, x_T: Int)` — [`L250`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L250) — Converts integer token IDs to dense embedding vectors.
  - `dtype` — [`L198`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L198)
  - `hidden_size` — [`L197`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L197)
  - `input_embedding_table_VD` — [`L207`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L207)
  - `normalize_embeddings` — [`L202`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L202)
  - `prelogit_td` — [`L199`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L199)
  - `random_init` — [`L201`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L201)
  - `rngs` — [`L204`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L204)
  - `vd_sharding` — [`L200`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L200)
  - `vocab_size` — [`L196`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L196)
- protocol/private: `__post_init__`[`L206`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L206)
- uses (calls/refs, reference-scoped): [`create_param`](base.md#create_param), [`decode`](layers.md#LMhead.decode), [`LMhead`](layers.md#LMhead)
- used by: [`lm_head`](../../models/jax/llama4.md#Llama4ForCausalLM.lm_head), [`lm_head`](../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.lm_head), [`embedder`](../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.embedder), [`embedder`](../../models/jax/llama4.md#Llama4ForCausalLM.embedder), [`lm_head`](../../models/jax/gpt_oss.md#GptOss.lm_head), [`__call__`](../../models/jax/llama4.md#Llama4ForCausalLM.__call__), [`__call__`](../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.__call__), [`embedder`](../../models/jax/gpt_oss.md#GptOss.embedder), [`__call__`](../../models/jax/gpt_oss.md#GptOss.__call__), [`__post_init__`](layers.md#LMhead.__post_init__), [`embed_input_ids`](../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.embed_input_ids), [`decode`](layers.md#LMhead.decode), [`LMhead`](layers.md#LMhead)

### `FlaxUtils`
- def: [`tpu_inference/layers/jax/layers.py:30`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L30)
- doc: A dummy class to namespace activation functions, mimicking external utilities.
- signature: `class FlaxUtils:`
- members:
  - `ACT2FN` — [`L32`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L32)
- used by: [`__call__`](moe/moe.md#Router.__call__), [`__call__`](layers.md#DenseFFW.__call__), [`sparse_moe_distributed_fwd`](moe/sparse_moe.md#sparse_moe_distributed_fwd), [`__call__`](../../models/jax/deepseek_v3.md#DeepseekV3MLP.__call__), [`dense_moe_fwd`](moe/dense_moe.md#dense_moe_fwd), [`dense_moe_fwd_preapply_router_weights`](moe/dense_moe.md#dense_moe_fwd_preapply_router_weights), [`__init__`](../../models/jax/qwen2_5_vl.md#Qwen2_5_VisionMLP.__init__), [`modeling_flax_utils`](moe/utils.md#modeling_flax_utils), [`act_fn`](../../models/jax/llama3.md#LlamaMLP.act_fn), [`act_fn`](../../models/jax/qwen2.md#Qwen2MLP.act_fn), [`mlp_act`](../../models/jax/qwen2_5_vl.md#Qwen2_5_VisionPatchMerger.mlp_act), [`modeling_flax_utils`](../../models/jax/qwen2_5_vl.md#modeling_flax_utils), [`modeling_flax_utils`](layers.md#modeling_flax_utils), [`modeling_flax_utils`](moe/moe.md#modeling_flax_utils), [`modeling_flax_utils`](../../models/jax/deepseek_v3.md#modeling_flax_utils), [`modeling_flax_utils`](../../models/jax/llama3.md#modeling_flax_utils), [`modeling_flax_utils`](../../models/jax/qwen2.md#modeling_flax_utils), [`modeling_flax_utils`](moe/gpt_oss_moe.md#modeling_flax_utils)

### `LMhead`  ·  implements/extends Embedder
- def: [`tpu_inference/layers/jax/layers.py:272`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L272)
- doc: An Embedder that uses a (D, V) shaped embedding table, inheriting from
- signature: `class LMhead(Embedder):`
- members:
  - `__call__(self, x)` — [`L290`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L290) — Dispatches to decode method.
  - `decode(self, x_TD: Float)` — [`L304`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L304) — Projects hidden states to vocabulary logits.
  - `dv_sharding` — [`L280`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L280)
  - `input_embedding_table_DV` — [`L283`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L283)
- protocol/private: `__post_init__`[`L282`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L282)
- uses (calls/refs, reference-scoped): [`create_param`](base.md#create_param), [`dtype`](layers.md#Embedder.dtype), [`Embedder`](layers.md#Embedder), [`hidden_size`](layers.md#Embedder.hidden_size), [`random_init`](layers.md#Embedder.random_init), [`vocab_size`](layers.md#Embedder.vocab_size), [`prelogit_td`](layers.md#Embedder.prelogit_td)
- used by: [`lm_head`](../../models/jax/llama4.md#Llama4ForCausalLM.lm_head), [`lm_head`](../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.lm_head), [`lm_head`](../../models/jax/gpt_oss.md#GptOss.lm_head), [`decode`](layers.md#Embedder.decode), [`Embedder`](layers.md#Embedder), [`compute_logits`](../../models/jax/gpt_oss.md#GptOss.compute_logits), [`compute_logits`](../../models/jax/llama4.md#Llama4ForCausalLM.compute_logits), [`compute_logits`](../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.compute_logits)

### `RMSNorm`  ·  implements/extends Module
- def: [`tpu_inference/layers/jax/layers.py:65`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L65)
- doc: An implementation of Root Mean Square Layer Normalization.
- signature: `class RMSNorm(nnx.Module):`
- members:
  - `__call__(self, x_TD: Float, op_mode='generate')` — [`L83`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L83) — Applies RMS Normalization to the input tensor.
  - `activation_ffw_td` — [`L75`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L75)
  - `dims` — [`L74`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L74)
  - `dtype` — [`L79`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L79)
  - `epsilon` — [`L77`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L77)
  - `random_init` — [`L76`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L76)
  - `rngs` — [`L81`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L81)
  - `scale` — [`L107`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L107)
  - `with_scale` — [`L78`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L78)
- protocol/private: `__post_init__`[`L106`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L106)
- uses (calls/refs, reference-scoped): [`create_param`](base.md#create_param)
- used by: [`end_layer`](../../models/jax/llama4.md#Llama4ForCausalLM.end_layer), [`end_layer`](../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.end_layer), [`layers`](../../models/jax/gpt_oss.md#GptOss.layers), [`final_norm`](../../models/jax/llama4.md#Llama4ForCausalLM.final_norm), [`final_norm`](../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.final_norm), [`final_norm`](../../models/jax/gpt_oss.md#GptOss.final_norm)

### `RuntimeParams`
- def: [`tpu_inference/layers/jax/layers.py:45`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L45)
- doc: A container for runtime parameters needed by neural network blocks.
- signature: `class RuntimeParams:`
- members:
  - `kv_cache` — [`L59`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L59)
  - `quantization` — [`L61`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L61)
  - `sharding_cfg` — [`L60`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L60)

## Module values
- `modeling_flax_utils` — [`L41`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/layers.py#L41)

