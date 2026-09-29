---
title: 'Module: tpu_inference/layers/jax/moe/moe.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/moe/moe.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.moe.moe`/
symbols:
  JaxMoE.kernel_down_proj_EFD: JaxMoE#kernel_down_proj_EFD.
  Router.__call__: Router#__call__().
  JaxMoE: JaxMoE#
  JaxMoE.quant_method: JaxMoE#quant_method.
  JaxMoE._load_weights: JaxMoE#_load_weights().
  JaxMoE.__call__: JaxMoE#__call__().
  JaxMoE.kernel_gating_EDF: JaxMoE#kernel_gating_EDF.
  JaxMoE.kernel_up_proj_EDF: JaxMoE#kernel_up_proj_EDF.
  JaxMoE.moe_backend: JaxMoE#moe_backend.
  JaxRoutedExperts: JaxRoutedExperts#
  Router.__post_init__: Router#__post_init__().
  JaxRoutedExperts.kernel_gating_EDF: JaxRoutedExperts#kernel_gating_EDF.
  JaxRoutedExperts.kernel_up_proj_EDF: JaxRoutedExperts#kernel_up_proj_EDF.
  JaxRoutedExperts.kernel_down_proj_EFD: JaxRoutedExperts#kernel_down_proj_EFD.
  JaxRoutedExperts.moe_backend: JaxRoutedExperts#moe_backend.
  JaxRoutedExperts.quant_method: JaxRoutedExperts#quant_method.
  JaxRoutedExperts.__call__: JaxRoutedExperts#__call__().
  Router.dtype: Router#dtype.
  JaxMoE.edf_sharding: JaxMoE#edf_sharding.
  JaxMoE.quant_config: JaxMoE#quant_config.
  JaxMoE.activation: JaxMoE#activation.
  JaxMoE.__post_init__: JaxMoE#__post_init__().
  JaxMoE.dtype: JaxMoE#dtype.
  JaxMoE.activation_ffw_td: JaxMoE#activation_ffw_td.
  JaxMoE.efd_sharding: JaxMoE#efd_sharding.
  JaxMoE.prefix: JaxMoE#prefix.
  Router: Router#
  Router.moe_backend: Router#moe_backend.
  JaxMoE.mesh: JaxMoE#mesh.
  JaxMoE.router: JaxMoE#router.
  JaxMoE.expert_axis_name: JaxMoE#expert_axis_name.
  JaxMoE.qwix_quantized_weight_dtype: JaxMoE#qwix_quantized_weight_dtype.
  JaxMoE.is_batch_sharded_by_expert: JaxMoE#is_batch_sharded_by_expert.
  JaxMoE.load_weights: JaxMoE#load_weights().
  JaxRoutedExperts.quant_config: JaxRoutedExperts#quant_config.
  JaxRoutedExperts.load_weights: JaxRoutedExperts#load_weights().
  JaxRoutedExperts.use_ep: JaxRoutedExperts#use_ep.
  JaxRoutedExperts.activation_ffw_td: JaxRoutedExperts#activation_ffw_td.
  JaxRoutedExperts.activation: JaxRoutedExperts#activation.
  Router.num_experts: Router#num_experts.
  Router.num_experts_per_tok: Router#num_experts_per_tok.
  Router.router_act: Router#router_act.
  Router.activation_ffw_td: Router#activation_ffw_td.
  Router.random_init: Router#random_init.
  JaxMoE.num_local_experts: JaxMoE#num_local_experts.
  JaxMoE.hidden_size: JaxMoE#hidden_size.
  JaxMoE.intermediate_size_moe: JaxMoE#intermediate_size_moe.
  JaxMoE.hidden_act: JaxMoE#hidden_act.
  JaxMoE.activation_ffw_ted: JaxMoE#activation_ffw_ted.
  JaxMoE.apply_expert_weight_before_computation: JaxMoE#apply_expert_weight_before_computation.
  JaxMoE.num_expert_parallelism: JaxMoE#num_expert_parallelism.
  JaxMoE.random_init: JaxMoE#random_init.
  JaxMoE.enable_return_routed_experts: JaxMoE#enable_return_routed_experts.
  modeling_flax_utils: modeling_flax_utils.
  logger: logger.
  JaxMoE.data_axis_name: JaxMoE#data_axis_name.
  JaxRoutedExperts.__init__: JaxRoutedExperts#__init__().
  JaxRoutedExperts.top_k: JaxRoutedExperts#top_k.
  Router.hidden_size: Router#hidden_size.
  Router.ed_sharding: Router#ed_sharding.
  JaxMoE.rngs: JaxMoE#rngs.
  JaxMoE.scoring_func: JaxMoE#scoring_func.
  JaxMoE.num_experts_per_tok: JaxMoE#num_experts_per_tok.
  CombineExperts.__call__: CombineExperts#__call__().
  JaxMoE.top_k: JaxMoE#top_k.
  JaxMoE.use_ep: JaxMoE#use_ep.
  JaxMoE.named_parameters: JaxMoE#named_parameters().
  JaxRoutedExperts.num_experts_per_tok: JaxRoutedExperts#num_experts_per_tok.
  JaxRoutedExperts.renormalize: JaxRoutedExperts#renormalize.
  JaxRoutedExperts.scoring_func: JaxRoutedExperts#scoring_func.
  JaxRoutedExperts.dtype: JaxRoutedExperts#dtype.
  Router.kernel_DE: Router#kernel_DE.
  Router.rngs: Router#rngs.
  Router.mesh: Router#mesh.
  JaxRoutedExperts.mesh: JaxRoutedExperts#mesh.
  JaxRoutedExperts.num_local_experts: JaxRoutedExperts#num_local_experts.
  JaxRoutedExperts.hidden_act: JaxRoutedExperts#hidden_act.
  CombineExperts.dtype: CombineExperts#dtype.
  JaxRoutedExperts.enable_return_routed_experts: JaxRoutedExperts#enable_return_routed_experts.
  JaxRoutedExperts._compute_use_ep: JaxRoutedExperts#_compute_use_ep().
  CombineExperts: CombineExperts#
  JaxMoE.e2df_sharding: JaxMoE#e2df_sharding.
  JaxMoE.tile_size: JaxMoE#tile_size.
  JaxMoE.renormalize: JaxMoE#renormalize.
  JaxRoutedExperts.hidden_size: JaxRoutedExperts#hidden_size.
  JaxRoutedExperts.intermediate_size_moe: JaxRoutedExperts#intermediate_size_moe.
  JaxRoutedExperts.random_init: JaxRoutedExperts#random_init.
  JaxRoutedExperts.qwix_quantized_weight_dtype: JaxRoutedExperts#qwix_quantized_weight_dtype.
  JaxRoutedExperts.prefix: JaxRoutedExperts#prefix.
---
# Module: [`tpu_inference/layers/jax/moe/moe.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py)

## Classes
### `CombineExperts`  ·  implements/extends Module
- def: [`tpu_inference/layers/jax/moe/moe.py:43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L43)
- doc: Combines expert outputs with router weights.
- signature: `class CombineExperts(nnx.Module):`
- members:
  - `dtype` — [`L51`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L51)
- protocol/private: `__call__`[`L53`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L53)

### `JaxMoE`  ·  implements/extends JaxModule
- def: [`tpu_inference/layers/jax/moe/moe.py:132`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L132) — documented in [tpu_inference-layers-jax-moe](../../../../../concepts/tpu_inference-layers-jax-moe.md)
- doc: Mixture-of-Experts (MoE) Routed MLP Layer.
- signature: `class JaxMoE(JaxModule):`
- members:
  - `__call__(self, x_TD: jax.Array, router_logits: Optional[jax.Array] = None)` — [`L177`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L177) — Performs the forward pass of the MoE layer. — documented in [tpu_inference-layers-jax-moe](../../../../../concepts/tpu_inference-layers-jax-moe.md)
  - `__post_init__(self, rngs: nnx.Rngs)` — [`L209`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L209) — Generates the kernels (weights) for the router and experts (gating, up-projection, and down-projection layers).
  - `_load_weights(self, weights: Iterable, *, mesh: jax.sharding.Mesh | None = None)` — [`L281`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L281) — Load HF weights into the layer.
  - `load_weights(self, weights: Iterable)` — [`L270`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L270) — Used by JaxAutoWeightLoader to load HF weights into the layer.
  - `named_parameters(self, *args, **kwargs)` — [`L260`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L260)
  - `activation` — [`L247`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L247)
  - `activation_ffw_td` — [`L149`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L149)
  - `activation_ffw_ted` — [`L150`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L150)
  - `apply_expert_weight_before_computation` — [`L156`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L156)
  - `data_axis_name` — [`L240`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L240)
  - `dtype` — [`L140`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L140)
  - `e2df_sharding` — [`L153`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L153)
  - `edf_sharding` — [`L151`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L151)
  - `efd_sharding` — [`L152`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L152)
  - `enable_return_routed_experts` — [`L175`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L175)
  - `expert_axis_name` — [`L157`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L157)
  - `hidden_act` — [`L144`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L144)
  - `hidden_size` — [`L142`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L142)
  - `intermediate_size_moe` — [`L143`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L143)
  - `is_batch_sharded_by_expert` — [`L241`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L241)
  - `kernel_down_proj_EFD` — [`L229`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L229) — documented in [tpu_inference-layers-jax-moe](../../../../../concepts/tpu_inference-layers-jax-moe.md)
  - `kernel_gating_EDF` — [`L215`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L215)
  - `kernel_up_proj_EDF` — [`L222`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L222)
  - `mesh` — [`L147`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L147)
  - `moe_backend` — [`L160`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L160)
  - `num_expert_parallelism` — [`L158`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L158)
  - `num_experts_per_tok` — [`L164`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L164)
  - `num_local_experts` — [`L141`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L141)
  - `prefix` — [`L174`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L174)
  - `quant_config` — [`L173`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L173)
  - `quant_method` — [`L251`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L251) — documented in [tpu_inference-layers-jax-moe](../../../../../concepts/tpu_inference-layers-jax-moe.md)
  - `qwix_quantized_weight_dtype` — [`L167`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L167)
  - `random_init` — [`L159`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L159)
  - `renormalize` — [`L170`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L170)
  - `rngs` — [`L145`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L145)
  - `router` — [`L146`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L146)
  - `scoring_func` — [`L161`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L161)
  - `tile_size` — [`L165`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L165)
  - `top_k` — [`L245`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L245)
  - `use_ep` — [`L246`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L246)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../__init__.md#JaxModule), [`MoEBackend`](../../common/moe.md#MoEBackend), [`create_param`](../base.md#create_param), [`QuantizeMethodBase`](../quantization/__init__.md#QuantizeMethodBase), [`QuantizationConfig`](../quantization/configs.md#QuantizationConfig), [`apply_jax`](../quantization/__init__.md#QuantizeMethodBase.apply_jax), [`cpu_mesh_context`](../../common/utils.md#cpu_mesh_context), [`shard_put`](../../../models/jax/utils/weight_utils.md#shard_put), [`create_weights_jax`](../quantization/__init__.md#QuantizeMethodBase.create_weights_jax), [`fused_moe_backends`](../../common/moe.md#MoEBackend.fused_moe_backends), [`named_parameters`](../../__init__.md#JaxModule.named_parameters), [`get_quant_method`](../quantization/configs.md#QuantizationConfig.get_quant_method), [`jax_array_from_reshaped_torch`](../../../models/jax/utils/weight_utils.md#jax_array_from_reshaped_torch), [`DENSE_MAT`](../../common/moe.md#MoEBackend.DENSE_MAT), [`SharedFusedMoe`](../../../models/jax/deepseek_v3.md#SharedFusedMoe), [`logger`](moe.md#logger)
- used by: [`end_layer`](../../../models/jax/llama4.md#Llama4ForCausalLM.end_layer), [`JaxModule`](../../__init__.md#JaxModule), [`moe_apply`](../../common/moe.md#moe_apply), [`experts`](../../../models/jax/deepseek_v3.md#DeepseekV2Moe.experts), [`experts`](../../../models/jax/qwen3_moe.md#Qwen3MoeSparseMoeBlock.experts), [`apply_jax`](../quantization/fp8.md#Fp8FusedMoEMethod.apply_jax), [`process_weights_after_loading`](../quantization/fp8.md#Fp8FusedMoEMethod.process_weights_after_loading), [`sparse_moe_func`](sparse_moe.md#sparse_moe_func), [`get_quant_method`](../quantization/unquantized.md#UnquantizedConfig.get_quant_method), [`get_quant_method`](../quantization/fp8.md#Fp8Config.get_quant_method), [`__call__`](../transformer_block.md#SharedExpertsTransformerBlock.__call__), [`create_weights_jax`](../quantization/fp8.md#Fp8FusedMoEMethod.create_weights_jax), [`load_weights`](../quantization/fp8.md#Fp8FusedMoEMethod.load_weights), [`named_parameters`](../../__init__.md#JaxModule.named_parameters), [`__call__`](../../../models/jax/deepseek_v3.md#SharedFusedMoe.__call__), [`SharedFusedMoe`](../../../models/jax/deepseek_v3.md#SharedFusedMoe), [`moe_ffw`](../transformer_block.md#SharedExpertsTransformerBlock.moe_ffw)

### `JaxRoutedExperts`  ·  implements/extends JaxModule
- def: [`tpu_inference/layers/jax/moe/moe.py:345`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L345)
- doc: Expert-only MoE module analogous to vllm's RoutedExperts.
- signature: `class JaxRoutedExperts(JaxModule):`
- members:
  - `load_weights(self, weights: Iterable)` — [`L457`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L457) — Used by JaxAutoWeightLoader to load HF weights into the layer.
  - `activation` — [`L416`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L416)
  - `activation_ffw_td` — [`L417`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L417)
  - `dtype` — [`L376`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L376)
  - `enable_return_routed_experts` — [`L388`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L388)
  - `hidden_act` — [`L380`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L380)
  - `hidden_size` — [`L378`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L378)
  - `intermediate_size_moe` — [`L379`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L379)
  - `kernel_down_proj_EFD` — [`L406`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L406)
  - `kernel_gating_EDF` — [`L396`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L396)
  - `kernel_up_proj_EDF` — [`L401`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L401)
  - `mesh` — [`L381`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L381)
  - `moe_backend` — [`L414`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L414)
  - `num_experts_per_tok` — [`L418`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L418)
  - `num_local_experts` — [`L377`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L377)
  - `prefix` — [`L387`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L387)
  - `quant_config` — [`L425`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L425)
  - `quant_method` — [`L429`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L429)
  - `qwix_quantized_weight_dtype` — [`L386`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L386)
  - `random_init` — [`L385`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L385)
  - `renormalize` — [`L384`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L384)
  - `scoring_func` — [`L383`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L383)
  - `top_k` — [`L382`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L382)
  - `use_ep` — [`L413`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L413)
- protocol/private: `__call__`[`L443`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L443), `__init__`[`L357`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L357), `_compute_use_ep`[`L435`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L435)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../common/sharding.md#ShardingAxisName), [`JaxModule`](../../__init__.md#JaxModule), [`create_param`](../base.md#create_param), [`QuantizeMethodBase`](../quantization/__init__.md#QuantizeMethodBase), [`get_quant_method`](../quantization/unquantized.md#UnquantizedConfig.get_quant_method), [`select_moe_backend`](utils.md#select_moe_backend), [`QuantizationConfig`](../quantization/configs.md#QuantizationConfig), [`apply_jax`](../quantization/__init__.md#QuantizeMethodBase.apply_jax), [`create_weights_jax`](../quantization/__init__.md#QuantizeMethodBase.create_weights_jax), [`load_weights`](../../../models/jax/gemma4.md#Gemma4MoE.load_weights), [`UnquantizedConfig`](../quantization/unquantized.md#UnquantizedConfig), [`Gemma4MoE`](../../../models/jax/gemma4.md#Gemma4MoE)
- used by: [`JaxModule`](../../__init__.md#JaxModule), [`moe_apply`](../../common/moe.md#moe_apply), [`apply_jax`](../quantization/unquantized.md#UnquantizedFusedMoEMethod.apply_jax), [`apply_jax`](../quantization/fp8.md#Fp8FusedMoEMethod.apply_jax), [`extra_backend_kwargs`](../quantization/unquantized.md#UnquantizedFusedMoEMethod.extra_backend_kwargs), [`get_quant_method`](../quantization/unquantized.md#UnquantizedConfig.get_quant_method), [`get_quant_method`](../quantization/fp8.md#Fp8Config.get_quant_method), [`process_weights_after_loading`](../quantization/unquantized.md#UnquantizedFusedMoEMethod.process_weights_after_loading), [`load_weights`](../../../models/jax/gemma4.md#Gemma4MoE.load_weights), [`__init__`](../../../models/jax/gemma4.md#Gemma4MoE.__init__), [`Gemma4MoE`](../../../models/jax/gemma4.md#Gemma4MoE)

### `Router`  ·  implements/extends Module
- def: [`tpu_inference/layers/jax/moe/moe.py:66`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L66)
- doc: Router module for Mixture-of-Experts (MoE) layers.
- signature: `class Router(nnx.Module):`
- members:
  - `__call__(self, x_TD: Float)` — [`L85`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L85) — Routes tokens to experts. — documented in [tpu_inference-layers-jax-moe](../../../../../concepts/tpu_inference-layers-jax-moe.md)
  - `__post_init__(self, rngs: nnx.Rngs)` — [`L119`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L119) — Generates the router kernel (weights) for routing.
  - `activation_ffw_td` — [`L79`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L79)
  - `dtype` — [`L73`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L73)
  - `ed_sharding` — [`L80`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L80)
  - `hidden_size` — [`L74`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L74)
  - `kernel_DE` — [`L122`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L122)
  - `mesh` — [`L83`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L83)
  - `moe_backend` — [`L82`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L82)
  - `num_experts` — [`L75`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L75)
  - `num_experts_per_tok` — [`L76`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L76)
  - `random_init` — [`L81`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L81)
  - `rngs` — [`L78`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L78)
  - `router_act` — [`L77`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L77)
- uses (calls/refs, reference-scoped): [`MoEBackend`](../../common/moe.md#MoEBackend), [`create_param`](../base.md#create_param), [`fused_moe_backends`](../../common/moe.md#MoEBackend.fused_moe_backends), [`ACT2FN`](../layers.md#FlaxUtils.ACT2FN), [`DENSE_MAT`](../../common/moe.md#MoEBackend.DENSE_MAT), [`GptOssRouter`](gpt_oss_moe.md#GptOssRouter), [`modeling_flax_utils`](moe.md#modeling_flax_utils)
- used by: [`end_layer`](../../../models/jax/llama4.md#Llama4ForCausalLM.end_layer), [`layers`](../../../models/jax/gpt_oss.md#GptOss.layers), [`bias_E`](gpt_oss_moe.md#GptOssRouter.bias_E), [`__call__`](gpt_oss_moe.md#GptOssRouter.__call__), [`GptOssRouter`](gpt_oss_moe.md#GptOssRouter), [`__post_init__`](gpt_oss_moe.md#GptOssRouter.__post_init__)

## Module values
- `logger` — [`L39`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L39)
- `modeling_flax_utils` — [`L38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/moe.py#L38)

