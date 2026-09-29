---
title: 'Module: tpu_inference/models/jax/qwen3_moe.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/qwen3_moe.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.qwen3_moe`/
symbols:
  Qwen3MoeSparseMoeBlock.experts: Qwen3MoeSparseMoeBlock#experts.
  Qwen3MoeModel.__call__: Qwen3MoeModel#__call__().
  Qwen3MoeModel.is_last_rank: Qwen3MoeModel#is_last_rank.
  Qwen3MoeModel.embed_tokens: Qwen3MoeModel#embed_tokens.
  Qwen3MoeDecoderLayer.__call__: Qwen3MoeDecoderLayer#__call__().
  Qwen3MoeForCausalLM.compute_logits: Qwen3MoeForCausalLM#compute_logits().
  Qwen3MoeModel.norm: Qwen3MoeModel#norm.
  Qwen3MoeForCausalLM.model: Qwen3MoeForCausalLM#model.
  Qwen3MoeForCausalLM.__call__: Qwen3MoeForCausalLM#__call__().
  Qwen3MoeForCausalLM.lm_head: Qwen3MoeForCausalLM#lm_head.
  Qwen3MoeModel.layers: Qwen3MoeModel#layers.
  Qwen3MoeDecoderLayer.__init__: Qwen3MoeDecoderLayer#__init__().
  Qwen3MoeSparseMoeBlock.gate: Qwen3MoeSparseMoeBlock#gate.
  Qwen3MoeModel.is_first_rank: Qwen3MoeModel#is_first_rank.
  Qwen3MoeForCausalLM: Qwen3MoeForCausalLM#
  Qwen3MoeDecoderLayer.input_layernorm: Qwen3MoeDecoderLayer#input_layernorm.
  Qwen3MoeDecoderLayer.post_attention_layernorm: Qwen3MoeDecoderLayer#post_attention_layernorm.
  Qwen3MoeSparseMoeBlock.__init__: Qwen3MoeSparseMoeBlock#__init__().
  Qwen3MoeSparseMoeBlock.__call__: Qwen3MoeSparseMoeBlock#__call__().
  init_fn: init_fn.
  Qwen3MoeSparseMoeBlock: Qwen3MoeSparseMoeBlock#
  Qwen3MoeDecoderLayer: Qwen3MoeDecoderLayer#
  Qwen3MoeDecoderLayer.self_attn: Qwen3MoeDecoderLayer#self_attn.
  Qwen3MoeDecoderLayer.mlp: Qwen3MoeDecoderLayer#mlp.
  Qwen3MoeModel: Qwen3MoeModel#
  logger: logger.
  Qwen3MoeSparseMoeBlock.shared_expert: Qwen3MoeSparseMoeBlock#shared_expert.
  Qwen3MoeSparseMoeBlock.enable_return_routed_experts: Qwen3MoeSparseMoeBlock#enable_return_routed_experts.
  Qwen3MoeModel.__init__: Qwen3MoeModel#__init__().
  Qwen3MoeModel.start_layer: Qwen3MoeModel#start_layer.
  Qwen3MoeModel.end_layer: Qwen3MoeModel#end_layer.
  Qwen3MoeForCausalLM.__init__: Qwen3MoeForCausalLM#__init__().
  Qwen3MoeForCausalLM.vllm_config: Qwen3MoeForCausalLM#vllm_config.
  Qwen3MoeForCausalLM.mesh: Qwen3MoeForCausalLM#mesh.
---
# Module: [`tpu_inference/models/jax/qwen3_moe.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py)

## Classes
### `Qwen3MoeDecoderLayer`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/qwen3_moe.py:136`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L136)
- signature: `class Qwen3MoeDecoderLayer(JaxModule):`
- members:
  - `input_layernorm` — [`L151`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L151)
  - `mlp` — [`L185`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L185)
  - `post_attention_layernorm` — [`L170`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L170)
  - `self_attn` — [`L161`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L161)
- protocol/private: `__call__`[`L194`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L194), `__init__`[`L138`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L138)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`QuantizationConfig`](../../layers/jax/quantization/configs.md#QuantizationConfig), [`Qwen3Attention`](qwen3.md#Qwen3Attention), [`init_fn`](qwen3_moe.md#init_fn), [`Qwen3MoeSparseMoeBlock`](qwen3_moe.md#Qwen3MoeSparseMoeBlock), `hidden_size`, `rms_norm_eps`
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`layers`](qwen3_moe.md#Qwen3MoeModel.layers)

### `Qwen3MoeForCausalLM`  ·  implements/extends JaxModule, LoadableWithIterator
- def: [`tpu_inference/models/jax/qwen3_moe.py:316`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L316)
- signature: `class Qwen3MoeForCausalLM(JaxModule, LoadableWithIterator):`
- members:
  - `compute_logits(self, hidden_states: jax.Array)` — [`L376`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L376)
  - `lm_head` — [`L335`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L335)
  - `mesh` — [`L322`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L322)
  - `model` — [`L324`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L324)
  - `vllm_config` — [`L320`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L320)
- protocol/private: `__call__`[`L346`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L346), `__init__`[`L318`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L318)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxIntermediateTensors`](jax_intermediate_tensor.md#JaxIntermediateTensors), [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`LoadableWithIterator`](utils/weight_utils.md#LoadableWithIterator), [`JaxLmHead`](../../layers/jax/linear.md#JaxLmHead), [`tensors`](jax_intermediate_tensor.md#JaxIntermediateTensors.tensors), [`JaxEmbed`](../../layers/jax/embed.md#JaxEmbed), [`embed_tokens`](qwen3_moe.md#Qwen3MoeModel.embed_tokens), [`is_last_rank`](qwen3_moe.md#Qwen3MoeModel.is_last_rank), [`decode`](../../layers/jax/embed.md#JaxEmbed.decode), [`Qwen3MoeModel`](qwen3_moe.md#Qwen3MoeModel)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`_get_model_architecture`](../common/model_loader.md#_get_model_architecture), [`LoadableWithIterator`](utils/weight_utils.md#LoadableWithIterator)

### `Qwen3MoeModel`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/qwen3_moe.py:221`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L221)
- signature: `class Qwen3MoeModel(JaxModule):`
- members:
  - `embed_tokens` — [`L240`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L240)
  - `end_layer` — [`L253`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L253)
  - `is_first_rank` — [`L235`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L235)
  - `is_last_rank` — [`L236`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L236)
  - `layers` — [`L253`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L253)
  - `norm` — [`L268`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L268)
  - `start_layer` — [`L253`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L253)
- protocol/private: `__call__`[`L281`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L281), `__init__`[`L223`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L223)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`get_pp_group`](../../distributed/jax_parallel_state.md#get_pp_group), [`make_layers`](../../layers/jax/pp_utils.md#make_layers), [`JaxEmbed`](../../layers/jax/embed.md#JaxEmbed), [`is_first_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_first_rank), [`is_last_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_last_rank), [`init_fn`](qwen3_moe.md#init_fn), [`Qwen3MoeDecoderLayer`](qwen3_moe.md#Qwen3MoeDecoderLayer)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`compute_logits`](qwen3_moe.md#Qwen3MoeForCausalLM.compute_logits), [`model`](qwen3_moe.md#Qwen3MoeForCausalLM.model)

### `Qwen3MoeSparseMoeBlock`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/qwen3_moe.py:61`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L61)
- signature: `class Qwen3MoeSparseMoeBlock(JaxModule):`
- members:
  - `enable_return_routed_experts` — [`L103`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L103)
  - `experts` — [`L104`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L104) — documented in [tpu_inference-layers-jax-moe](../../../../concepts/tpu_inference-layers-jax-moe.md)
  - `gate` — [`L80`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L80)
  - `shared_expert` — [`L100`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L100)
- protocol/private: `__call__`[`L126`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L126), `__init__`[`L63`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L63)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../layers/common/sharding.md#ShardingAxisName), [`JaxModule`](../../layers/__init__.md#JaxModule), [`select_moe_backend`](../../layers/jax/moe/utils.md#select_moe_backend), [`JaxMoE`](../../layers/jax/moe/moe.md#JaxMoE), [`JaxLinear`](../../layers/jax/linear.md#JaxLinear), [`moe_backend`](../../layers/jax/moe/moe.md#JaxMoE.moe_backend), [`edf_sharding`](../../layers/jax/moe/moe.md#JaxMoE.edf_sharding), [`quant_config`](../../layers/jax/moe/moe.md#JaxMoE.quant_config), [`activation_ffw_td`](../../layers/jax/moe/moe.md#JaxMoE.activation_ffw_td), [`dtype`](../../layers/jax/moe/moe.md#JaxMoE.dtype), [`efd_sharding`](../../layers/jax/moe/moe.md#JaxMoE.efd_sharding), [`prefix`](../../layers/jax/moe/moe.md#JaxMoE.prefix), [`mesh`](../../layers/jax/moe/moe.md#JaxMoE.mesh), [`expert_axis_name`](../../layers/jax/moe/moe.md#JaxMoE.expert_axis_name), [`router`](../../layers/jax/moe/moe.md#JaxMoE.router), [`activation_ffw_ted`](../../layers/jax/moe/moe.md#JaxMoE.activation_ffw_ted), [`apply_expert_weight_before_computation`](../../layers/jax/moe/moe.md#JaxMoE.apply_expert_weight_before_computation), [`enable_return_routed_experts`](../../layers/jax/moe/moe.md#JaxMoE.enable_return_routed_experts), [`get_expert_parallelism`](../../layers/jax/moe/utils.md#get_expert_parallelism), [`hidden_act`](../../layers/jax/moe/moe.md#JaxMoE.hidden_act), [`hidden_size`](../../layers/jax/moe/moe.md#JaxMoE.hidden_size), [`intermediate_size_moe`](../../layers/jax/moe/moe.md#JaxMoE.intermediate_size_moe), [`num_expert_parallelism`](../../layers/jax/moe/moe.md#JaxMoE.num_expert_parallelism), [`num_local_experts`](../../layers/jax/moe/moe.md#JaxMoE.num_local_experts), [`num_experts_per_tok`](../../layers/jax/moe/moe.md#JaxMoE.num_experts_per_tok), [`rngs`](../../layers/jax/moe/moe.md#JaxMoE.rngs)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`mlp`](qwen3_moe.md#Qwen3MoeDecoderLayer.mlp)

## Module values
- `init_fn` — [`L58`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L58)
- `logger` — [`L56`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen3_moe.py#L56)

