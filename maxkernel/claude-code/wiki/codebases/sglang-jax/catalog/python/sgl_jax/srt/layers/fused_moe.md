---
title: 'Module: python/sgl_jax/srt/layers/fused_moe.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/fused_moe.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.fused_moe`/
symbols:
  FusedEPMoE.__call__: FusedEPMoE#__call__().
  FusedEPMoEV2.__call__: FusedEPMoEV2#__call__().
  FusedEPMoE.quantize_weights: FusedEPMoE#quantize_weights().
  FusedEPMoE.num_experts: FusedEPMoE#num_experts.
  FusedEPMoE.w3_shared: FusedEPMoE#w3_shared.
  FusedEPMoE.quant_block_k: FusedEPMoE#quant_block_k.
  FusedEPMoE.quant_block_n: FusedEPMoE#quant_block_n.
  FusedEPMoE.w2_scale: FusedEPMoE#w2_scale.
  FusedEPMoE.quantized_dtype: FusedEPMoE#quantized_dtype.
  FusedEPMoE.w1_shared: FusedEPMoE#w1_shared.
  FusedEPMoE.w1: FusedEPMoE#w1.
  FusedEPMoE.w1_scale: FusedEPMoE#w1_scale.
  FusedEPMoE.w3_scale: FusedEPMoE#w3_scale.
  FusedEPMoE.w1_shared_scale: FusedEPMoE#w1_shared_scale.
  FusedEPMoE.w3_shared_scale: FusedEPMoE#w3_shared_scale.
  FusedEPMoE.w2_shared_scale: FusedEPMoE#w2_shared_scale.
  FusedEPMoE.w3: FusedEPMoE#w3.
  FusedEPMoE.w2: FusedEPMoE#w2.
  FusedEPMoE.w2_shared: FusedEPMoE#w2_shared.
  FusedEPMoE.hidden_size: FusedEPMoE#hidden_size.
  FusedEPMoE.intermediate_dim: FusedEPMoE#intermediate_dim.
  FusedEPMoE.mesh: FusedEPMoE#mesh.
  FusedEPMoEV2: FusedEPMoEV2#
  FusedEPMoE: FusedEPMoE#
  FusedEPMoE.num_shared_experts: FusedEPMoE#num_shared_experts.
  FusedEPMoE.use_jax_allreduce_metadata: FusedEPMoE#use_jax_allreduce_metadata.
  _expand_moe_block_scale: _expand_moe_block_scale().
  FusedEPMoE.num_experts_per_tok: FusedEPMoE#num_experts_per_tok.
  FusedEPMoE.ep_size: FusedEPMoE#ep_size.
  FusedEPMoE.activation: FusedEPMoE#activation.
  FusedEPMoE.use_grouped_topk: FusedEPMoE#use_grouped_topk.
  FusedEPMoE.moe_shared_expert_intermediate_size: FusedEPMoE#moe_shared_expert_intermediate_size.
  FusedEPMoE.num_groups: FusedEPMoE#num_groups.
  FusedEPMoE.top_k_groups: FusedEPMoE#top_k_groups.
  FusedEPMoE.renormalize_topk_logits: FusedEPMoE#renormalize_topk_logits.
  FusedEPMoE.routed_scaling_factor: FusedEPMoE#routed_scaling_factor.
  FusedEPMoE.disable_a2a: FusedEPMoE#disable_a2a.
  FusedEPMoE.disable_dynamic_ffn1: FusedEPMoE#disable_dynamic_ffn1.
  FusedEPMoE.disable_dynamic_ffn2: FusedEPMoE#disable_dynamic_ffn2.
  FusedEPMoE.disable_weight_load: FusedEPMoE#disable_weight_load.
  FusedEPMoE.disable_a2a_s_tile_read: FusedEPMoE#disable_a2a_s_tile_read.
  FusedEPMoE.disable_a2a_s_acc_tile_write: FusedEPMoE#disable_a2a_s_acc_tile_write.
  FusedEPMoE.disable_shared_expert: FusedEPMoE#disable_shared_expert.
  FusedEPMoE.disable_all_reduce_metadata: FusedEPMoE#disable_all_reduce_metadata.
  FusedEPMoE.disable_sync_barrier: FusedEPMoE#disable_sync_barrier.
  FusedEPMoE.activation_quantized_dtype: FusedEPMoE#activation_quantized_dtype.
  FusedEPMoE.enable_act_quant_cfg: FusedEPMoE#enable_act_quant_cfg.
  FusedEPMoE.__init__: FusedEPMoE#__init__().
  FusedEPMoE.weight_dtype: FusedEPMoE#weight_dtype.
  FusedEPMoE.dtype: FusedEPMoE#dtype.
  FusedEPMoE.layer_id: FusedEPMoE#layer_id.
---
# Module: [`python/sgl_jax/srt/layers/fused_moe.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py)

## Classes
### `FusedEPMoE`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/layers/fused_moe.py:20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L20)
- doc: Expert Parallel MoE layer using fused TPU kernel.
- signature: `class FusedEPMoE(nnx.Module):`
- members:
  - `__call__(self, hidden_states: jax.Array, topk_weights: jax.Array, topk_ids: jax.Array, *, block_config: FusedMoEBlockConfig | None = None, out_sharding: jax.sharding.Sharding | None = None)` — [`L461`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L461) — Forward pass through the fused MoE layer. — documented in [python-sgl_jax-srt-kernels-fused_moe-v1-kernel](../../../../../concepts/python-sgl_jax-srt-kernels-fused_moe-v1-kernel.md)
  - `quantize_weights(self, is_static: bool = False)` — [`L211`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L211) — Quantize MoE weights in-place. Call once after model loading.
  - `activation` — [`L88`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L88)
  - `activation_quantized_dtype` — [`L124`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L124)
  - `disable_a2a` — [`L99`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L99)
  - `disable_a2a_s_acc_tile_write` — [`L104`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L104)
  - `disable_a2a_s_tile_read` — [`L103`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L103)
  - `disable_all_reduce_metadata` — [`L106`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L106)
  - `disable_dynamic_ffn1` — [`L100`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L100)
  - `disable_dynamic_ffn2` — [`L101`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L101)
  - `disable_shared_expert` — [`L105`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L105)
  - `disable_sync_barrier` — [`L107`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L107)
  - `disable_weight_load` — [`L102`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L102)
  - `dtype` — [`L85`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L85)
  - `enable_act_quant_cfg` — [`L129`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L129)
  - `ep_size` — [`L87`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L87)
  - `hidden_size` — [`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L81)
  - `intermediate_dim` — [`L83`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L83)
  - `layer_id` — [`L86`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L86)
  - `mesh` — [`L98`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L98)
  - `moe_shared_expert_intermediate_size` — [`L95`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L95)
  - `num_experts` — [`L112`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L112)
  - `num_experts_per_tok` — [`L82`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L82)
  - `num_groups` — [`L90`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L90)
  - `num_shared_experts` — [`L94`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L94)
  - `quant_block_k` — [`L205`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L205)
  - `quant_block_n` — [`L206`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L206)
  - `quantized_dtype` — [`L121`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L121)
  - `renormalize_topk_logits` — [`L92`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L92)
  - `routed_scaling_factor` — [`L93`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L93)
  - `top_k_groups` — [`L91`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L91)
  - `use_grouped_topk` — [`L89`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L89)
  - `use_jax_allreduce_metadata` — [`L108`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L108)
  - `w1` — [`L132`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L132)
  - `w1_scale` — [`L158`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L158)
  - `w1_shared` — [`L165`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L165)
  - `w1_shared_scale` — [`L196`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L196)
  - `w2` — [`L149`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L149)
  - `w2_scale` — [`L160`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L160)
  - `w2_shared` — [`L174`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L174)
  - `w2_shared_scale` — [`L198`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L198)
  - `w3` — [`L140`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L140)
  - `w3_scale` — [`L159`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L159)
  - `w3_shared` — [`L183`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L183)
  - `w3_shared_scale` — [`L197`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L197)
  - `weight_dtype` — [`L84`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L84)
- protocol/private: `__init__`[`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L48)
- uses (calls/refs, reference-scoped): [`fused_ep_moe`](../kernels/fused_moe/v1/kernel.md#fused_ep_moe), [`quantize_tensor`](../utils/quantization/quantization_utils.md#quantize_tensor), [`get_global_expert_location_metadata`](../eplb/expert_location.md#get_global_expert_location_metadata), [`FusedMoEBlockConfig`](../kernels/fused_moe/v1/kernel.md#FusedMoEBlockConfig), [`FusedEPMoEV2`](fused_moe.md#FusedEPMoEV2), [`_expand_moe_block_scale`](fused_moe.md#_expand_moe_block_scale)
- used by: [`__call__`](fused_moe.md#FusedEPMoEV2.__call__), [`apply_moe_quantization`](../utils/quantization/quantization_utils.md#apply_moe_quantization), [`_quantize_moe_recursive`](../utils/quantization/quantization_utils.md#apply_moe_quantization._quantize_moe_recursive), [`FusedEPMoEV2`](fused_moe.md#FusedEPMoEV2)

### `FusedEPMoEV2`
- def: [`python/sgl_jax/srt/layers/fused_moe.py:535`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L535)
- protocol/private: `__call__`[`L542`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L542)
- uses (calls/refs, reference-scoped): [`fused_ep_moe_v2`](../kernels/fused_moe/v2/kernel.md#fused_ep_moe_v2), [`get_tuned_fused_moe_v2_block_config`](../kernels/fused_moe/v2/tuned_block_configs.md#get_tuned_fused_moe_v2_block_config), [`num_experts`](fused_moe.md#FusedEPMoE.num_experts), [`w3_shared`](fused_moe.md#FusedEPMoE.w3_shared), [`quant_block_k`](fused_moe.md#FusedEPMoE.quant_block_k), [`w2_scale`](fused_moe.md#FusedEPMoE.w2_scale), [`w1_shared`](fused_moe.md#FusedEPMoE.w1_shared), [`w1`](fused_moe.md#FusedEPMoE.w1), [`w1_scale`](fused_moe.md#FusedEPMoE.w1_scale), [`w1_shared_scale`](fused_moe.md#FusedEPMoE.w1_shared_scale), [`w2_shared_scale`](fused_moe.md#FusedEPMoE.w2_shared_scale), [`w3_scale`](fused_moe.md#FusedEPMoE.w3_scale), [`w3_shared_scale`](fused_moe.md#FusedEPMoE.w3_shared_scale), [`w2`](fused_moe.md#FusedEPMoE.w2), [`w3`](fused_moe.md#FusedEPMoE.w3), [`w2_shared`](fused_moe.md#FusedEPMoE.w2_shared), [`hidden_size`](fused_moe.md#FusedEPMoE.hidden_size), [`intermediate_dim`](fused_moe.md#FusedEPMoE.intermediate_dim), [`mesh`](fused_moe.md#FusedEPMoE.mesh), [`FusedEPMoE`](fused_moe.md#FusedEPMoE), [`ep_size`](fused_moe.md#FusedEPMoE.ep_size), [`num_experts_per_tok`](fused_moe.md#FusedEPMoE.num_experts_per_tok), [`activation`](fused_moe.md#FusedEPMoE.activation), [`use_grouped_topk`](fused_moe.md#FusedEPMoE.use_grouped_topk), [`activation_quantized_dtype`](fused_moe.md#FusedEPMoE.activation_quantized_dtype), [`enable_act_quant_cfg`](fused_moe.md#FusedEPMoE.enable_act_quant_cfg)
- used by: [`quantize_weights`](fused_moe.md#FusedEPMoE.quantize_weights)

## Functions
- `_expand_moe_block_scale(scale_3d: jax.Array, n_out: int, block_n: int)` — [`L14`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/fused_moe.py#L14) — Expand compact 2D MoE block scales to the kernel's fast 1D-ready layout.

