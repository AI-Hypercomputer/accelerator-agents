---
title: 'Module: tpu_inference/models/jax/llama3.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/llama3.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.llama3`/
symbols:
  LlamaAttention.__call__: LlamaAttention#__call__().
  LlamaModel.__call__: LlamaModel#__call__().
  LlamaModel.lm_head: LlamaModel#lm_head.
  LlamaAttention.head_dim: LlamaAttention#head_dim.
  LlamaAttention.q_proj: LlamaAttention#q_proj.
  LlamaAttention.k_proj: LlamaAttention#k_proj.
  LlamaAttention.v_proj: LlamaAttention#v_proj.
  LlamaDecoderLayer.self_attn: LlamaDecoderLayer#self_attn.
  LlamaModel.is_last_rank: LlamaModel#is_last_rank.
  LlamaAttention.o_proj: LlamaAttention#o_proj.
  LlamaModel.embed: LlamaModel#embed.
  LlamaDecoderLayer.__call__: LlamaDecoderLayer#__call__().
  LlamaForCausalLM.load_weights: LlamaForCausalLM#load_weights().
  init_fn: init_fn.
  LlamaModel.norm: LlamaModel#norm.
  LlamaMLP.__call__: LlamaMLP#__call__().
  LlamaModel.layers: LlamaModel#layers.
  LlamaAttention.head_dim_original: LlamaAttention#head_dim_original.
  LlamaForCausalLM.model: LlamaForCausalLM#model.
  LlamaAttention.hidden_size: LlamaAttention#hidden_size.
  LlamaAttention.num_heads: LlamaAttention#num_heads.
  LlamaAttention.num_kv_heads: LlamaAttention#num_kv_heads.
  LlamaForCausalLM.pp_missing_layers: LlamaForCausalLM#pp_missing_layers.
  LlamaMLP.__init__: LlamaMLP#__init__().
  LlamaForCausalLM.__call__: LlamaForCausalLM#__call__().
  LlamaForCausalLM.compute_logits: LlamaForCausalLM#compute_logits().
  LlamaModel.is_first_rank: LlamaModel#is_first_rank.
  LlamaModel.aux_hidden_state_layers: LlamaModel#aux_hidden_state_layers.
  LlamaMLP.gate_proj: LlamaMLP#gate_proj.
  LlamaMLP.up_proj: LlamaMLP#up_proj.
  LlamaMLP.down_proj: LlamaMLP#down_proj.
  LlamaMLP.act_fn: LlamaMLP#act_fn.
  LlamaDecoderLayer.__init__: LlamaDecoderLayer#__init__().
  LlamaAttention.kv_cache_quantized_dtype: LlamaAttention#kv_cache_quantized_dtype.
  LlamaDecoderLayer: LlamaDecoderLayer#
  LlamaAttention.rope_theta: LlamaAttention#rope_theta.
  LlamaAttention.rope_scaling: LlamaAttention#rope_scaling.
  LlamaDecoderLayer.post_attention_layernorm: LlamaDecoderLayer#post_attention_layernorm.
  LlamaDecoderLayer.mlp: LlamaDecoderLayer#mlp.
  modeling_flax_utils: modeling_flax_utils.
  LlamaDecoderLayer.input_layernorm: LlamaDecoderLayer#input_layernorm.
  LlamaModel.get_eagle3_aux_hidden_state_layers: LlamaModel#get_eagle3_aux_hidden_state_layers().
  LlamaForCausalLM.WeightLoader: LlamaForCausalLM#WeightLoader.
  LlamaForCausalLM.vllm_config: LlamaForCausalLM#vllm_config.
  logger: logger.
  LlamaForCausalLM: LlamaForCausalLM#
  LlamaForCausalLM.rng: LlamaForCausalLM#rng.
  LlamaMLP: LlamaMLP#
  LlamaAttention: LlamaAttention#
  LlamaAttention.mesh: LlamaAttention#mesh.
  LlamaAttention._k_scale: LlamaAttention#_k_scale.
  LlamaAttention._v_scale: LlamaAttention#_v_scale.
  LlamaModel: LlamaModel#
  LlamaModel.start_layer: LlamaModel#start_layer.
  LlamaModel.end_layer: LlamaModel#end_layer.
  LlamaForCausalLM.mesh: LlamaForCausalLM#mesh.
  LlamaAttention.__init__: LlamaAttention#__init__().
  LlamaAttention._q_scale: LlamaAttention#_q_scale.
  LlamaModel.__init__: LlamaModel#__init__().
  LlamaForCausalLM.__init__: LlamaForCausalLM#__init__().
---
# Module: [`tpu_inference/models/jax/llama3.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py)

## Classes
### `LlamaAttention`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/llama3.py:93`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L93)
- signature: `class LlamaAttention(nnx.Module):`
- members:
  - `head_dim` — [`L106`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L106)
  - `head_dim_original` — [`L104`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L104)
  - `hidden_size` — [`L98`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L98)
  - `k_proj` — [`L125`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L125)
  - `kv_cache_quantized_dtype` — [`L153`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L153)
  - `mesh` — [`L115`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L115)
  - `num_heads` — [`L99`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L99)
  - `num_kv_heads` — [`L100`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L100)
  - `o_proj` — [`L141`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L141)
  - `q_proj` — [`L117`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L117)
  - `rope_scaling` — [`L102`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L102)
  - `rope_theta` — [`L101`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L101)
  - `v_proj` — [`L133`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L133)
- protocol/private: `__call__`[`L158`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L158), `__init__`[`L95`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L95), `_k_scale`[`L151`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L151), `_q_scale`[`L150`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L150), `_v_scale`[`L152`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L152)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../layers/common/sharding.md#ShardingAxisName), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`input_positions`](../../layers/common/attention_metadata.md#AttentionMetadata.input_positions), [`apply_rope`](../../layers/jax/rope_interface.md#apply_rope), [`attention`](../../layers/common/attention_interface.md#attention), [`quantize_kv`](../../layers/common/__init__.md#quantize_kv), [`init_fn`](llama3.md#init_fn), [`get_rope_scaling`](../../layers/jax/rope_interface.md#get_rope_scaling), [`get_rope_theta`](../../layers/jax/rope_interface.md#get_rope_theta), `hidden_size`, `num_attention_heads`, `num_key_value_heads`
- used by: [`config`](llama_eagle3.md#Eagle3LlamaDecoderLayer.config), [`self_attn`](llama3.md#LlamaDecoderLayer.self_attn)

### `LlamaDecoderLayer`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/llama3.py:201`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L201)
- signature: `class LlamaDecoderLayer(nnx.Module):`
- members:
  - `input_layernorm` — [`L208`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L208)
  - `mlp` — [`L227`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L227)
  - `post_attention_layernorm` — [`L220`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L220)
  - `self_attn` — [`L215`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L215)
- protocol/private: `__call__`[`L233`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L233), `__init__`[`L203`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L203)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`init_fn`](llama3.md#init_fn), `hidden_size`, [`Eagle3LlamaDecoderLayer`](llama_eagle3.md#Eagle3LlamaDecoderLayer), `rms_norm_eps`, [`LlamaAttention`](llama3.md#LlamaAttention), [`LlamaMLP`](llama3.md#LlamaMLP)
- used by: [`config`](llama_eagle3.md#Eagle3LlamaDecoderLayer.config), [`__call__`](llama_eagle3.md#Eagle3LlamaDecoderLayer.__call__), [`layers`](llama3.md#LlamaModel.layers), [`Eagle3LlamaDecoderLayer`](llama_eagle3.md#Eagle3LlamaDecoderLayer), [`__init__`](llama_eagle3.md#Eagle3LlamaDecoderLayer.__init__)

### `LlamaForCausalLM`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/llama3.py:355`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L355)
- signature: `class LlamaForCausalLM(nnx.Module):`
- members:
  - `compute_logits(self, hidden_states: jax.Array)` — [`L399`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L399)
  - `load_weights(self, rng_key: jax.Array)` — [`L406`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L406)
  - `WeightLoader` — [`L356`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L356)
  - `mesh` — [`L362`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L362)
  - `model` — [`L364`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L364)
  - `pp_missing_layers` — [`L370`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L370)
  - `rng` — [`L361`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L361)
  - `vllm_config` — [`L360`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L360)
- protocol/private: `__call__`[`L376`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L376), `__init__`[`L358`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L358)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxIntermediateTensors`](jax_intermediate_tensor.md#JaxIntermediateTensors), [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`lm_head`](llama3.md#LlamaModel.lm_head), [`StandardWeightLoader`](utils/weight_utils.md#StandardWeightLoader), [`load_weights`](utils/weight_utils.md#StandardWeightLoader.load_weights), [`LlamaModel`](llama3.md#LlamaModel)
- used by: [`_get_model_architecture`](../common/model_loader.md#_get_model_architecture)

### `LlamaMLP`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/llama3.py:49`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L49)
- signature: `class LlamaMLP(nnx.Module):`
- members:
  - `act_fn` — [`L83`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L83)
  - `down_proj` — [`L74`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L74)
  - `gate_proj` — [`L56`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L56)
  - `up_proj` — [`L65`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L65)
- protocol/private: `__call__`[`L85`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L85), `__init__`[`L51`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L51)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../layers/common/sharding.md#ShardingAxisName), [`init_fn`](llama3.md#init_fn), [`ACT2FN`](../../layers/jax/layers.md#FlaxUtils.ACT2FN), `hidden_size`, [`modeling_flax_utils`](llama3.md#modeling_flax_utils), `hidden_act`, `intermediate_size`
- used by: [`mlp`](llama3.md#LlamaDecoderLayer.mlp)

### `LlamaModel`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/llama3.py:254`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L254)
- signature: `class LlamaModel(nnx.Module):`
- members:
  - `get_eagle3_aux_hidden_state_layers(self)` — [`L317`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L317)
  - `aux_hidden_state_layers` — [`L312`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L312)
  - `embed` — [`L270`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L270)
  - `end_layer` — [`L281`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L281)
  - `is_first_rank` — [`L265`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L265)
  - `is_last_rank` — [`L266`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L266)
  - `layers` — [`L281`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L281)
  - `lm_head` — [`L303`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L303)
  - `norm` — [`L291`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L291)
  - `start_layer` — [`L281`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L281)
- protocol/private: `__call__`[`L321`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L321), `__init__`[`L256`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L256)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../layers/common/sharding.md#ShardingAxisName), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxIntermediateTensors`](jax_intermediate_tensor.md#JaxIntermediateTensors), [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`get_pp_group`](../../distributed/jax_parallel_state.md#get_pp_group), [`make_layers`](../../layers/jax/pp_utils.md#make_layers), [`is_first_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_first_rank), [`is_last_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_last_rank), [`init_fn`](llama3.md#init_fn), `embedding`, [`LlamaDecoderLayer`](llama3.md#LlamaDecoderLayer)
- used by: [`model`](llama3.md#LlamaForCausalLM.model), [`compute_logits`](llama3.md#LlamaForCausalLM.compute_logits)

## Module values
- `init_fn` — [`L45`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L45)
- `logger` — [`L43`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L43)
- `modeling_flax_utils` — [`L46`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/llama3.py#L46)

