---
title: 'Module: src/maxtext/models/gpt3.py'
type: catalog
provenance: extracted
module: src/maxtext/models/gpt3.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.gpt3`/
symbols:
  Gpt3MultiHeadAttention.__call__: Gpt3MultiHeadAttention#__call__().
  Gpt3MultiHeadAttention.init_kv_caches: Gpt3MultiHeadAttention#init_kv_caches().
  Gpt3MultiHeadAttention.attention_op: Gpt3MultiHeadAttention#attention_op.
  Gpt3MultiHeadAttention.create_projection_layer: Gpt3MultiHeadAttention#create_projection_layer().
  Gpt3LayerNorm.__call__: Gpt3LayerNorm#__call__().
  Gpt3DecoderLayer.__call__: Gpt3DecoderLayer#__call__().
  Gpt3LayerNorm.scale: Gpt3LayerNorm#scale.
  Gpt3DecoderLayer.self_attention: Gpt3DecoderLayer#self_attention.
  Gpt3LayerNorm.bias: Gpt3LayerNorm#bias.
  Gpt3MultiHeadAttention.num_heads: Gpt3MultiHeadAttention#num_heads.
  Gpt3MultiHeadAttention.KVCache_0: Gpt3MultiHeadAttention#KVCache_0.
  Gpt3MultiHeadAttention.qkv_proj: Gpt3MultiHeadAttention#qkv_proj.
  Gpt3MultiHeadAttention.query: Gpt3MultiHeadAttention#query.
  Gpt3MultiHeadAttention.key: Gpt3MultiHeadAttention#key.
  Gpt3MultiHeadAttention.value: Gpt3MultiHeadAttention#value.
  Gpt3MultiHeadAttention.out: Gpt3MultiHeadAttention#out.
  Gpt3DecoderLayer.mlp: Gpt3DecoderLayer#mlp.
  Gpt3MultiHeadAttention.head_dim: Gpt3MultiHeadAttention#head_dim.
  Gpt3MultiHeadAttention.update_kv_caches: Gpt3MultiHeadAttention#update_kv_caches().
  Gpt3DecoderLayer.config: Gpt3DecoderLayer#config.
  Gpt3DecoderLayer.pre_self_attention_norm: Gpt3DecoderLayer#pre_self_attention_norm.
  Gpt3DecoderLayer.activation_axis_names: Gpt3DecoderLayer#activation_axis_names.
  Gpt3MultiHeadAttention.dtype: Gpt3MultiHeadAttention#dtype.
  Gpt3MultiHeadAttention.projection: Gpt3MultiHeadAttention#projection().
  Gpt3DecoderLayer.rngs: Gpt3DecoderLayer#rngs.
  Gpt3DecoderLayer.dropout: Gpt3DecoderLayer#dropout.
  Gpt3LayerNorm.dtype: Gpt3LayerNorm#dtype.
  gpt3_layer_norm: gpt3_layer_norm().
  Gpt3DecoderLayerToLinen: Gpt3DecoderLayerToLinen.
  Gpt3LayerNorm: Gpt3LayerNorm#
  Gpt3LayerNorm.weight_dtype: Gpt3LayerNorm#weight_dtype.
  Gpt3LayerNorm.kernel_axes: Gpt3LayerNorm#kernel_axes.
  Gpt3LayerNorm.reductions_in_fp32: Gpt3LayerNorm#reductions_in_fp32.
  Gpt3MultiHeadAttention.config: Gpt3MultiHeadAttention#config.
  Gpt3MultiHeadAttention.max_target_length: Gpt3MultiHeadAttention#max_target_length.
  Gpt3MultiHeadAttention.max_prefill_predict_length: Gpt3MultiHeadAttention#max_prefill_predict_length.
  Gpt3MultiHeadAttention.fused_qkv: Gpt3MultiHeadAttention#fused_qkv.
  Gpt3MultiHeadAttention.quant: Gpt3MultiHeadAttention#quant.
  Gpt3MultiHeadAttention.kv_quant: Gpt3MultiHeadAttention#kv_quant.
  Gpt3MultiHeadAttention.rngs: Gpt3MultiHeadAttention#rngs.
  Gpt3MultiHeadAttention.model_mode: Gpt3MultiHeadAttention#model_mode.
  Gpt3DecoderLayer.mesh: Gpt3DecoderLayer#mesh.
  Gpt3DecoderLayer.quant: Gpt3DecoderLayer#quant.
  Gpt3LayerNorm.epsilon: Gpt3LayerNorm#epsilon.
  Gpt3LayerNorm.scale_init: Gpt3LayerNorm#scale_init.
  Gpt3LayerNorm.use_bias: Gpt3LayerNorm#use_bias.
  Gpt3LayerNorm.parameter_memory_host_offload: Gpt3LayerNorm#parameter_memory_host_offload.
  Gpt3MultiHeadAttention: Gpt3MultiHeadAttention#
  Gpt3MultiHeadAttention.mesh: Gpt3MultiHeadAttention#mesh.
  Gpt3MultiHeadAttention.attention_kernel: Gpt3MultiHeadAttention#attention_kernel.
  Gpt3MultiHeadAttention.weight_dtype: Gpt3MultiHeadAttention#weight_dtype.
  Gpt3MultiHeadAttention.kernel_init: Gpt3MultiHeadAttention#kernel_init.
  Gpt3MultiHeadAttention.float32_qk_product: Gpt3MultiHeadAttention#float32_qk_product.
  Gpt3MultiHeadAttention.float32_logits: Gpt3MultiHeadAttention#float32_logits.
  Gpt3MultiHeadAttention.use_bias: Gpt3MultiHeadAttention#use_bias.
  Gpt3MultiHeadAttention.input_axis_names: Gpt3MultiHeadAttention#input_axis_names.
  Gpt3MultiHeadAttention.query_axis_names: Gpt3MultiHeadAttention#query_axis_names.
  Gpt3MultiHeadAttention.key_axis_names: Gpt3MultiHeadAttention#key_axis_names.
  Gpt3MultiHeadAttention.value_axis_names: Gpt3MultiHeadAttention#value_axis_names.
  Gpt3MultiHeadAttention.out_axis_names: Gpt3MultiHeadAttention#out_axis_names.
  Gpt3MultiHeadAttention.prefill_cache_axis_order: Gpt3MultiHeadAttention#prefill_cache_axis_order.
  Gpt3MultiHeadAttention.ar_cache_axis_order: Gpt3MultiHeadAttention#ar_cache_axis_order.
  Gpt3MultiHeadAttention.use_ragged_attention: Gpt3MultiHeadAttention#use_ragged_attention.
  Gpt3MultiHeadAttention.qkv_projection: Gpt3MultiHeadAttention#qkv_projection().
  Gpt3DecoderLayer: Gpt3DecoderLayer#
  Gpt3LayerNorm.__init__: Gpt3LayerNorm#__init__().
  Gpt3MultiHeadAttention.__init__: Gpt3MultiHeadAttention#__init__().
  Gpt3MultiHeadAttention.dropout_rate: Gpt3MultiHeadAttention#dropout_rate.
  Gpt3DecoderLayer.__init__: Gpt3DecoderLayer#__init__().
---
# Module: [`src/maxtext/models/gpt3.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py)

## Classes
### `Gpt3DecoderLayer`  ·  implements/extends Module
- def: [`src/maxtext/models/gpt3.py:392`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L392)
- doc: Transformer decoder layer that attends to the encoder.
- signature: `class Gpt3DecoderLayer(nnx.Module):`
- members:
  - `activation_axis_names` — [`L462`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L462)
  - `config` — [`L403`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L403)
  - `dropout` — [`L460`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L460)
  - `mesh` — [`L404`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L404)
  - `mlp` — [`L421`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L421)
  - `pre_self_attention_norm` — [`L411`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L411)
  - `quant` — [`L405`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L405)
  - `rngs` — [`L406`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L406)
  - `self_attention` — [`L437`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L437)
- protocol/private: `__call__`[`L464`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L464), `__init__`[`L395`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L395)
- uses (calls/refs, reference-scoped): [`Gpt3LayerNorm`](gpt3.md#Gpt3LayerNorm), [`Gpt3MultiHeadAttention`](gpt3.md#Gpt3MultiHeadAttention)
- used by: [`Gpt3DecoderLayerToLinen`](gpt3.md#Gpt3DecoderLayerToLinen)

### `Gpt3LayerNorm`  ·  implements/extends Module
- def: [`src/maxtext/models/gpt3.py:47`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L47)
- doc: GPT3 Layer normalization operating on the last axis of the input data.
- signature: `class Gpt3LayerNorm(nnx.Module):`
- members:
  - `__call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None)` — [`L84`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L84) — Applies layer normalization on the input.
  - `bias` — [`L78`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L78)
  - `dtype` — [`L65`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L65)
  - `epsilon` — [`L64`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L64)
  - `kernel_axes` — [`L67`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L67)
  - `parameter_memory_host_offload` — [`L71`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L71)
  - `reductions_in_fp32` — [`L70`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L70)
  - `scale` — [`L73`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L73)
  - `scale_init` — [`L68`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L68)
  - `use_bias` — [`L69`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L69)
  - `weight_dtype` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L66)
- protocol/private: `__init__`[`L50`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L50)
- used by: [`pre_self_attention_norm`](gpt3.md#Gpt3DecoderLayer.pre_self_attention_norm), [`gpt3_layer_norm`](gpt3.md#gpt3_layer_norm)

### `Gpt3MultiHeadAttention`  ·  implements/extends Module
- def: [`src/maxtext/models/gpt3.py:166`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L166)
- doc: Multi-head attention in gpt3.
- signature: `class Gpt3MultiHeadAttention(nnx.Module):`
- members:
  - `create_projection_layer(self, input_shape: tuple[int, ...], output_shape: tuple[int, ...] | int, kernel_axes: tuple[str, ...], axis: int | tuple[int, ...] = -1)` — [`L271`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L271) — Create projection layer for Key, Value, Query and Output — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `init_kv_caches(self, inputs_kv_shape: tuple[int, ...])` — [`L309`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L309) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `projection(self, projection_layer: Any, inputs: Array)` — [`L304`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L304) — individual projection for one of q, k and v. — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `qkv_projection(self, projection_layer: Any, inputs: Array)` — [`L296`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L296) — Fused QKV projection — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `update_kv_caches(self, key, value, decoder_segment_ids, model_mode, previous_chunk)` — [`L332`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L332) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `KVCache_0` — [`L244`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L244) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `ar_cache_axis_order` — [`L242`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L242) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `attention_kernel` — [`L223`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L223) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `attention_op` — [`L256`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L256) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `config` — [`L217`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L217) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `dropout_rate` — [`L226`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L226)
  - `dtype` — [`L224`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L224) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `float32_logits` — [`L229`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L229) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `float32_qk_product` — [`L228`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L228) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `fused_qkv` — [`L230`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L230) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `head_dim` — [`L219`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L219) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `input_axis_names` — [`L234`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L234) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `kernel_init` — [`L227`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L227) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `key` — [`L251`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L251) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `key_axis_names` — [`L236`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L236) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `kv_quant` — [`L232`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L232) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `max_prefill_predict_length` — [`L221`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L221) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `max_target_length` — [`L220`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L220) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `mesh` — [`L222`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L222) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `model_mode` — [`L240`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L240) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `num_heads` — [`L218`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L218) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `out` — [`L253`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L253) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `out_axis_names` — [`L238`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L238) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `prefill_cache_axis_order` — [`L241`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L241) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `qkv_proj` — [`L246`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L246) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `quant` — [`L231`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L231) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `query` — [`L250`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L250) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `query_axis_names` — [`L235`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L235) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `rngs` — [`L239`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L239) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `use_bias` — [`L233`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L233) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `use_ragged_attention` — [`L243`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L243) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `value` — [`L252`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L252) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `value_axis_names` — [`L237`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L237) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
  - `weight_dtype` — [`L225`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L225) — documented in [maxtext-models-gpt3](../../../../concepts/maxtext-models-gpt3.md)
- protocol/private: `__call__`[`L343`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L343), `__init__`[`L188`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L188)
- used by: [`self_attention`](gpt3.md#Gpt3DecoderLayer.self_attention)

## Functions
- `gpt3_layer_norm(*, num_features: int, epsilon: float = 0.000001, dtype: Any = jnp.float32, weight_dtype: Any = jnp.float32, kernel_axes: tuple[None | str, ...] = (), scale_init: Initializer = nn.initializers.zeros, use_bias: bool = True, reductions_in_fp32: bool = False, parameter_memory_host_offload: bool = False, name: None | str = None)` — [`L116`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L116) — Initializes the gpt3_layer_norm module.

## Module values
- `Gpt3DecoderLayerToLinen` — [`L527`](../../../../../../../raw/code/maxtext/src/maxtext/models/gpt3.py#L527)

