---
title: 'Module: tpu_inference/layers/jax/transformer_block.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/transformer_block.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.transformer_block`/
symbols:
  SharedExpertsTransformerBlock.__call__: SharedExpertsTransformerBlock#__call__().
  TransformerBlock.__call__: TransformerBlock#__call__().
  TransformerBlock: TransformerBlock#
  TransformerBlock.custom_module: TransformerBlock#custom_module.
  TransformerBlock.pre_attention_norm: TransformerBlock#pre_attention_norm.
  TransformerBlock.pre_mlp_norm: TransformerBlock#pre_mlp_norm.
  TransformerBlock.attn: TransformerBlock#attn.
  SharedExpertsTransformerBlock: SharedExpertsTransformerBlock#
  SharedExpertsTransformerBlock.moe_ffw: SharedExpertsTransformerBlock#moe_ffw.
  SharedExpertsTransformerBlock.dense_ffw: SharedExpertsTransformerBlock#dense_ffw.
  SharedExpertsTransformerBlock.shared_experts: SharedExpertsTransformerBlock#shared_experts.
  TransformerBlock.use_attention_rope: TransformerBlock#use_attention_rope.
  TransformerBlock.enable_return_routed_experts: TransformerBlock#enable_return_routed_experts.
  SharedExpertsTransformerBlock.enable_return_routed_experts: SharedExpertsTransformerBlock#enable_return_routed_experts.
  TransformerBlock.quant: TransformerBlock#quant.
---
# Module: [`tpu_inference/layers/jax/transformer_block.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/transformer_block.py)

## Classes
### `SharedExpertsTransformerBlock`  ·  implements/extends TransformerBlock
- def: [`tpu_inference/layers/jax/transformer_block.py:72`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/transformer_block.py#L72)
- doc: Create a modified TransformerBlock that sums MoE layer output with shared expert output.
- signature: `class SharedExpertsTransformerBlock(TransformerBlock):`
- members:
  - `dense_ffw` — [`L91`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/transformer_block.py#L91)
  - `enable_return_routed_experts` — [`L93`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/transformer_block.py#L93)
  - `moe_ffw` — [`L90`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/transformer_block.py#L90)
  - `shared_experts` — [`L92`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/transformer_block.py#L92)
- protocol/private: `__call__`[`L95`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/transformer_block.py#L95)
- uses (calls/refs, reference-scoped): [`JaxMoE`](moe/moe.md#JaxMoE), [`DenseFFW`](layers.md#DenseFFW), [`TransformerBlock`](transformer_block.md#TransformerBlock), [`custom_module`](transformer_block.md#TransformerBlock.custom_module), [`attn`](transformer_block.md#TransformerBlock.attn), [`pre_attention_norm`](transformer_block.md#TransformerBlock.pre_attention_norm), [`pre_mlp_norm`](transformer_block.md#TransformerBlock.pre_mlp_norm), [`use_attention_rope`](transformer_block.md#TransformerBlock.use_attention_rope)
- used by: [`end_layer`](../../models/jax/llama4.md#Llama4ForCausalLM.end_layer), [`TransformerBlock`](transformer_block.md#TransformerBlock)

### `TransformerBlock`  ·  implements/extends Module
- def: [`tpu_inference/layers/jax/transformer_block.py:29`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/transformer_block.py#L29)
- doc: A heavy weight module which serves as the stateful live blocks in serving
- signature: `class TransformerBlock(nnx.Module):`
- members:
  - `attn` — [`L38`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/transformer_block.py#L38)
  - `custom_module` — [`L37`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/transformer_block.py#L37)
  - `enable_return_routed_experts` — [`L41`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/transformer_block.py#L41)
  - `pre_attention_norm` — [`L35`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/transformer_block.py#L35)
  - `pre_mlp_norm` — [`L36`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/transformer_block.py#L36)
  - `quant` — [`L40`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/transformer_block.py#L40)
  - `use_attention_rope` — [`L39`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/transformer_block.py#L39)
- protocol/private: `__call__`[`L43`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/transformer_block.py#L43)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../common/attention_metadata.md#AttentionMetadata), [`KVCache`](attention/attention.md#KVCache), [`SharedExpertsTransformerBlock`](transformer_block.md#SharedExpertsTransformerBlock)
- used by: [`end_layer`](../../models/jax/llama4.md#Llama4ForCausalLM.end_layer), [`end_layer`](../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.end_layer), [`layers`](../../models/jax/gpt_oss.md#GptOss.layers), [`__call__`](transformer_block.md#SharedExpertsTransformerBlock.__call__), [`SharedExpertsTransformerBlock`](transformer_block.md#SharedExpertsTransformerBlock)

