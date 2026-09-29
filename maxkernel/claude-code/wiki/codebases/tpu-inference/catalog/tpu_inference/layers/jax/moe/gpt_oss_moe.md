---
title: 'Module: tpu_inference/layers/jax/moe/gpt_oss_moe.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/moe/gpt_oss_moe.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.moe.gpt_oss_moe`/
symbols:
  GptOssMoE.__call__: GptOssMoE#__call__().
  GptOssMoE.__post_init__: GptOssMoE#__post_init__().
  GptOssRouter.bias_E: GptOssRouter#bias_E.
  GptOssRouter.__call__: GptOssRouter#__call__().
  GptOssMoE.gate_proj_kernel: GptOssMoE#gate_proj_kernel.
  GptOssMoE.gate_proj_bias: GptOssMoE#gate_proj_bias.
  GptOssMoE.up_proj_kernel: GptOssMoE#up_proj_kernel.
  GptOssMoE.up_proj_bias: GptOssMoE#up_proj_bias.
  GptOssMoE.mlp2_weight_EFD: GptOssMoE#mlp2_weight_EFD.
  GptOssMoE.dtype: GptOssMoE#dtype.
  GptOssMoE.combine_experts: GptOssMoE#combine_experts.
  GptOssMoE.random_init: GptOssMoE#random_init.
  GptOssRouter: GptOssRouter#
  GptOssMoE.router: GptOssMoE#router.
  GptOssMoE.ed_sharding: GptOssMoE#ed_sharding.
  GptOssMoE.edf_sharding: GptOssMoE#edf_sharding.
  modeling_flax_utils: modeling_flax_utils.
  GptOssRouter.__post_init__: GptOssRouter#__post_init__().
  CombineExperts.__call__: CombineExperts#__call__().
  GptOssRouter.e_sharding: GptOssRouter#e_sharding.
  CombineExperts.dtype: CombineExperts#dtype.
  GptOssMoE: GptOssMoE#
  GptOssMoE.hidden_size: GptOssMoE#hidden_size.
  GptOssMoE.intermediate_size_moe: GptOssMoE#intermediate_size_moe.
  GptOssMoE.num_local_experts: GptOssMoE#num_local_experts.
  GptOssMoE.swiglu_limit: GptOssMoE#swiglu_limit.
  GptOssMoE.activation_ffw_td: GptOssMoE#activation_ffw_td.
  GptOssMoE.efd_sharding: GptOssMoE#efd_sharding.
  GptOssMoE.enable_return_routed_experts: GptOssMoE#enable_return_routed_experts.
  _swiglu_split: _swiglu_split().
  CombineExperts: CombineExperts#
  GptOssMoE.rngs: GptOssMoE#rngs.
  GptOssMoE.swiglu_alpha: GptOssMoE#swiglu_alpha.
  GptOssMoE.mlp2_bias_ED: GptOssMoE#mlp2_bias_ED.
---
# Module: [`tpu_inference/layers/jax/moe/gpt_oss_moe.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py)

## Classes
### `CombineExperts`  ·  implements/extends Module
- def: [`tpu_inference/layers/jax/moe/gpt_oss_moe.py:86`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L86)
- doc: Module for combining expert outputs with weighted sum.
- signature: `class CombineExperts(nnx.Module):`
- members:
  - `__call__(self, down_proj_TED: Float, weights_TX: Float, indices_TX: jax.Array)` — [`L90`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L90) — Combines expert outputs using weighted sum.
  - `dtype` — [`L88`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L88)
- used by: [`combine_experts`](gpt_oss_moe.md#GptOssMoE.combine_experts)

### `GptOssMoE`  ·  implements/extends Module
- def: [`tpu_inference/layers/jax/moe/gpt_oss_moe.py:114`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L114)
- doc: JAX implementation of the GPT-OSS Mixture-of-Experts MLP block.
- signature: `class GptOssMoE(nnx.Module):`
- members:
  - `__call__(self, x_TD: Float)` — [`L137`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L137) — Performs the forward pass for the GPT-OSS MoE layer.
  - `__post_init__(self, rngs: nnx.Rngs)` — [`L178`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L178) — Initializes all weights and biases for the MoE block.
  - `activation_ffw_td` — [`L129`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L129)
  - `combine_experts` — [`L182`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L182)
  - `dtype` — [`L118`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L118)
  - `ed_sharding` — [`L132`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L132)
  - `edf_sharding` — [`L130`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L130)
  - `efd_sharding` — [`L131`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L131)
  - `enable_return_routed_experts` — [`L135`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L135)
  - `gate_proj_bias` — [`L195`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L195)
  - `gate_proj_kernel` — [`L188`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L188)
  - `hidden_size` — [`L119`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L119)
  - `intermediate_size_moe` — [`L120`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L120)
  - `mlp2_bias_ED` — [`L227`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L227)
  - `mlp2_weight_EFD` — [`L220`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L220)
  - `num_local_experts` — [`L121`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L121)
  - `random_init` — [`L134`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L134)
  - `rngs` — [`L123`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L123)
  - `router` — [`L122`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L122)
  - `swiglu_alpha` — [`L126`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L126)
  - `swiglu_limit` — [`L125`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L125)
  - `up_proj_bias` — [`L211`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L211)
  - `up_proj_kernel` — [`L204`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L204)
- uses (calls/refs, reference-scoped): [`create_param`](../base.md#create_param), [`GptOssRouter`](gpt_oss_moe.md#GptOssRouter), [`dtype`](gpt_oss_moe.md#CombineExperts.dtype), [`CombineExperts`](gpt_oss_moe.md#CombineExperts), [`_swiglu_split`](gpt_oss_moe.md#_swiglu_split)
- used by: [`layers`](../../../models/jax/gpt_oss.md#GptOss.layers)

### `GptOssRouter`  ·  implements/extends Router
- def: [`tpu_inference/layers/jax/moe/gpt_oss_moe.py:32`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L32)
- doc: Router module for Mixture-of-Experts (MoE) layers.
- signature: `class GptOssRouter(Router):`
- members:
  - `__call__(self, x_TD: Float)` — [`L52`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L52) — Overrides the parent's forward pass to include the bias.
  - `__post_init__(self, rngs: nnx.Rngs)` — [`L40`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L40) — Initializes the parent's kernel and adds the new bias parameter.
  - `bias_E` — [`L46`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L46)
  - `e_sharding` — [`L38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L38)
- uses (calls/refs, reference-scoped): [`create_param`](../base.md#create_param), [`__post_init__`](moe.md#Router.__post_init__), [`dtype`](moe.md#Router.dtype), [`Router`](moe.md#Router), [`activation_ffw_td`](moe.md#Router.activation_ffw_td), [`num_experts`](moe.md#Router.num_experts), [`num_experts_per_tok`](moe.md#Router.num_experts_per_tok), [`random_init`](moe.md#Router.random_init), [`kernel_DE`](moe.md#Router.kernel_DE)
- used by: [`layers`](../../../models/jax/gpt_oss.md#GptOss.layers), [`Router`](moe.md#Router), [`router`](gpt_oss_moe.md#GptOssMoE.router)

## Functions
- `_swiglu_split(gate: Float, up: Float, alpha: Float, limit: Float)` — [`L73`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L73) — Implements SwiGLU using separate Gate and Up projections.

## Module values
- `modeling_flax_utils` — [`L28`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/gpt_oss_moe.py#L28)

