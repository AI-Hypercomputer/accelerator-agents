---
title: 'Module: tpu_inference/layers/jax/rope.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/rope.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.rope`/
symbols:
  DeepseekScalingRotaryEmbedding._compute_inv_freq: DeepseekScalingRotaryEmbedding#_compute_inv_freq().
  GptOssRotaryEmbedding._compute_concentration_and_inv_freq: GptOssRotaryEmbedding#_compute_concentration_and_inv_freq().
  DeepseekScalingRotaryEmbedding.initialize_cache: DeepseekScalingRotaryEmbedding#initialize_cache().
  DeepseekScalingRotaryEmbedding._compute_sin_cos: DeepseekScalingRotaryEmbedding#_compute_sin_cos().
  Llama4VisionRotaryEmbedding.__post_init__: Llama4VisionRotaryEmbedding#__post_init__().
  DeepseekScalingRotaryEmbedding.apply_rope: DeepseekScalingRotaryEmbedding#apply_rope().
  RotaryEmbedding.rotary_dim: RotaryEmbedding#rotary_dim.
  RotaryEmbedding.sin_cos_cache: RotaryEmbedding#sin_cos_cache.
  RotaryEmbedding._compute_inv_freq: RotaryEmbedding#_compute_inv_freq().
  RotaryEmbedding._compute_sin_cos: RotaryEmbedding#_compute_sin_cos().
  RotaryEmbedding.initialize_cache: RotaryEmbedding#initialize_cache().
  RotaryEmbedding.apply_rope: RotaryEmbedding#apply_rope().
  DeepseekScalingRotaryEmbedding.scaling_factor: DeepseekScalingRotaryEmbedding#scaling_factor.
  DeepseekScalingRotaryEmbedding.mscale: DeepseekScalingRotaryEmbedding#mscale.
  DeepseekScalingRotaryEmbedding: DeepseekScalingRotaryEmbedding#
  RotaryEmbedding.rope_theta: RotaryEmbedding#rope_theta.
  GptOssRotaryEmbedding.__call__: GptOssRotaryEmbedding#__call__().
  GptOssRotaryEmbedding._apply_rotation: GptOssRotaryEmbedding#_apply_rotation().
  Llama4VisionRotaryEmbedding.freqs_cis_stacked: Llama4VisionRotaryEmbedding#freqs_cis_stacked.
  RotaryEmbedding.original_max_position_embeddings: RotaryEmbedding#original_max_position_embeddings.
  GptOssRotaryEmbedding.head_dim: GptOssRotaryEmbedding#head_dim.
  GptOssRotaryEmbedding.rope_theta: GptOssRotaryEmbedding#rope_theta.
  GptOssRotaryEmbedding.rope_scaling_factor: GptOssRotaryEmbedding#rope_scaling_factor.
  Llama4VisionRotaryEmbedding: Llama4VisionRotaryEmbedding#
  RotaryEmbedding: RotaryEmbedding#
  _yarn_find_correction_range: _yarn_find_correction_range().
  GptOssRotaryEmbedding._compute_cos_sin: GptOssRotaryEmbedding#_compute_cos_sin().
  RotaryEmbedding.dtype: RotaryEmbedding#dtype.
  GptOssRotaryEmbedding.initial_context_length: GptOssRotaryEmbedding#initial_context_length.
  Llama4VisionRotaryEmbedding.__call__: Llama4VisionRotaryEmbedding#__call__().
  DeepseekScalingRotaryEmbedding.beta_fast: DeepseekScalingRotaryEmbedding#beta_fast.
  DeepseekScalingRotaryEmbedding.beta_slow: DeepseekScalingRotaryEmbedding#beta_slow.
  DeepseekScalingRotaryEmbedding.mscale_value: DeepseekScalingRotaryEmbedding#mscale_value.
  DeepseekScalingRotaryEmbedding.mscale_all_dim: DeepseekScalingRotaryEmbedding#mscale_all_dim.
  _yarn_get_mscale: _yarn_get_mscale().
  _yarn_find_correction_dim: _yarn_find_correction_dim().
  GptOssRotaryEmbedding: GptOssRotaryEmbedding#
  GptOssRotaryEmbedding.dtype: GptOssRotaryEmbedding#dtype.
  GptOssRotaryEmbedding.rope_ntk_alpha: GptOssRotaryEmbedding#rope_ntk_alpha.
  GptOssRotaryEmbedding.rope_ntk_beta: GptOssRotaryEmbedding#rope_ntk_beta.
  Llama4VisionRotaryEmbedding.image_size: Llama4VisionRotaryEmbedding#image_size.
  Llama4VisionRotaryEmbedding.patch_size: Llama4VisionRotaryEmbedding#patch_size.
  Llama4VisionRotaryEmbedding.hidden_size: Llama4VisionRotaryEmbedding#hidden_size.
  Llama4VisionRotaryEmbedding.num_attention_heads: Llama4VisionRotaryEmbedding#num_attention_heads.
  Llama4VisionRotaryEmbedding.rope_theta: Llama4VisionRotaryEmbedding#rope_theta.
  _yarn_linear_ramp_mask: _yarn_linear_ramp_mask().
  Llama4VisionRotaryEmbedding.dtype: Llama4VisionRotaryEmbedding#dtype.
---
# Module: [`tpu_inference/layers/jax/rope.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py)

## Classes
### `DeepseekScalingRotaryEmbedding`  ·  implements/extends RotaryEmbedding
- def: [`tpu_inference/layers/jax/rope.py:81`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L81)
- doc: Rotary Embedding for deepseek, with scaling and YaRN method.
- signature: `class DeepseekScalingRotaryEmbedding(RotaryEmbedding):`
- members:
  - `apply_rope(self, positions: jax.Array, x_TNH: jax.Array)` — [`L142`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L142)
  - `initialize_cache(self)` — [`L91`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L91) — Computes and caches the sin/cos embeddings.
  - `beta_fast` — [`L86`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L86)
  - `beta_slow` — [`L87`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L87)
  - `mscale` — [`L104`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L104)
  - `mscale_all_dim` — [`L89`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L89)
  - `mscale_value` — [`L88`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L88)
  - `scaling_factor` — [`L85`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L85)
- protocol/private: `_compute_inv_freq`[`L107`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L107), `_compute_sin_cos`[`L125`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L125)
- uses (calls/refs, reference-scoped): [`rotary_dim`](rope.md#RotaryEmbedding.rotary_dim), [`sin_cos_cache`](rope.md#RotaryEmbedding.sin_cos_cache), [`rope_theta`](rope.md#RotaryEmbedding.rope_theta), [`original_max_position_embeddings`](rope.md#RotaryEmbedding.original_max_position_embeddings), [`RotaryEmbedding`](rope.md#RotaryEmbedding), [`_yarn_find_correction_range`](rope.md#_yarn_find_correction_range), [`dtype`](rope.md#RotaryEmbedding.dtype), [`_yarn_get_mscale`](rope.md#_yarn_get_mscale), [`_yarn_linear_ramp_mask`](rope.md#_yarn_linear_ramp_mask)
- used by: [`rope_emb`](../../models/jax/deepseek_v3.md#DeepSeekV3.rope_emb), [`compute_kv_projection`](../../models/jax/deepseek_v3.md#DeepseekV3Attention.compute_kv_projection), [`compute_q_projection`](../../models/jax/deepseek_v3.md#DeepseekV3Attention.compute_q_projection), [`compute_q_projection`](../../models/jax/deepseek_v3.md#DeepseekV3MLA.compute_q_projection), [`qk_head_dim`](../../models/jax/deepseek_v3.md#DeepseekV3BaseAttention.qk_head_dim), [`compute_kv_projection`](../../models/jax/deepseek_v3.md#DeepseekV3MLA.compute_kv_projection), [`_compute_inv_freq`](rope.md#RotaryEmbedding._compute_inv_freq), [`_compute_sin_cos`](rope.md#RotaryEmbedding._compute_sin_cos), [`rope`](../../models/jax/deepseek_v3.md#DeepseekV3BaseAttention.rope), [`apply_rope`](rope.md#RotaryEmbedding.apply_rope), [`initialize_cache`](rope.md#RotaryEmbedding.initialize_cache), [`initialize_cache`](../../models/jax/deepseek_v3.md#DeepSeekV3.initialize_cache), [`RotaryEmbedding`](rope.md#RotaryEmbedding)

### `GptOssRotaryEmbedding`  ·  implements/extends Module
- def: [`tpu_inference/layers/jax/rope.py:203`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L203)
- doc: JAX implementation of the Rotary Positional Embedding with YaRN scaling.
- signature: `class GptOssRotaryEmbedding(nnx.Module):`
- members:
  - `__call__(self, query_TNH: jax.Array, key_TNH: jax.Array, positions: jax.Array)` — [`L263`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L263) — Applies rotary embeddings to query and key tensors.
  - `_compute_concentration_and_inv_freq(self)` — [`L215`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L215) — Computes the inverse frequencies and concentration factor for YaRN.
  - `_compute_cos_sin(self, positions: jax.Array)` — [`L248`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L248) — Computes cosine and sine embeddings for given positions.
  - `dtype` — [`L209`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L209)
  - `head_dim` — [`L207`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L207)
  - `initial_context_length` — [`L210`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L210)
  - `rope_ntk_alpha` — [`L212`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L212)
  - `rope_ntk_beta` — [`L213`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L213)
  - `rope_scaling_factor` — [`L211`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L211)
  - `rope_theta` — [`L208`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L208)
- protocol/private: `_apply_rotation`[`L279`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L279)
- used by: [`rope`](attention/gpt_oss_attention.md#GptOssAttention.rope)

### `Llama4VisionRotaryEmbedding`  ·  implements/extends Module
- def: [`tpu_inference/layers/jax/rope.py:298`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L298)
- doc: Calculates and stores the 2D Rotary Positional Embedding (RoPE) frequencies
- signature: `class Llama4VisionRotaryEmbedding(nnx.Module):`
- members:
  - `dtype` — [`L308`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L308)
  - `freqs_cis_stacked` — [`L362`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L362)
  - `hidden_size` — [`L305`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L305)
  - `image_size` — [`L303`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L303)
  - `num_attention_heads` — [`L306`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L306)
  - `patch_size` — [`L304`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L304)
  - `rope_theta` — [`L307`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L307)
- protocol/private: `__call__`[`L366`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L366), `__post_init__`[`L310`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L310)
- used by: [`load_weights`](../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.load_weights), [`vision_rope`](../../models/jax/llama_guard_4.md#LlamaGuard4ForCausalLM.vision_rope), [`__init__`](../../models/jax/llama4.md#JAXLlama4VisionModel.__init__)

### `RotaryEmbedding`  ·  implements/extends Module
- def: [`tpu_inference/layers/jax/rope.py:27`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L27)
- doc: An implementation of the original rotary positional embedding.
- signature: `class RotaryEmbedding(nnx.Module):`
- members:
  - `apply_rope(self, positions: jax.Array, x_TNH: jax.Array)` — [`L61`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L61)
  - `initialize_cache(self)` — [`L37`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L37) — Computes and caches the sin/cos embeddings.
  - `dtype` — [`L34`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L34)
  - `original_max_position_embeddings` — [`L33`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L33)
  - `rope_theta` — [`L32`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L32)
  - `rotary_dim` — [`L31`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L31)
  - `sin_cos_cache` — [`L35`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L35)
- protocol/private: `_compute_inv_freq`[`L42`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L42), `_compute_sin_cos`[`L48`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L48)
- uses (calls/refs, reference-scoped): [`_compute_inv_freq`](rope.md#DeepseekScalingRotaryEmbedding._compute_inv_freq), [`_compute_sin_cos`](rope.md#DeepseekScalingRotaryEmbedding._compute_sin_cos), [`initialize_cache`](rope.md#DeepseekScalingRotaryEmbedding.initialize_cache), [`apply_rope`](rope.md#DeepseekScalingRotaryEmbedding.apply_rope), [`DeepseekScalingRotaryEmbedding`](rope.md#DeepseekScalingRotaryEmbedding)
- used by: [`rope_emb`](../../models/jax/deepseek_v3.md#DeepSeekV3.rope_emb), [`_compute_inv_freq`](rope.md#DeepseekScalingRotaryEmbedding._compute_inv_freq), [`_compute_sin_cos`](rope.md#DeepseekScalingRotaryEmbedding._compute_sin_cos), [`initialize_cache`](rope.md#DeepseekScalingRotaryEmbedding.initialize_cache), [`apply_rope`](rope.md#DeepseekScalingRotaryEmbedding.apply_rope), [`mscale`](rope.md#DeepseekScalingRotaryEmbedding.mscale), [`DeepseekScalingRotaryEmbedding`](rope.md#DeepseekScalingRotaryEmbedding)

## Functions
- `_yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048)` — [`L169`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L169)
- `_yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048)` — [`L179`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L179)
- `_yarn_get_mscale(scale, mscale)` — [`L164`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L164)
- `_yarn_linear_ramp_mask(min, max, dim)` — [`L193`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope.py#L193)

