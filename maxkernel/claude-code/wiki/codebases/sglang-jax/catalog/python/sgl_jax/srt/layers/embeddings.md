---
title: 'Module: python/sgl_jax/srt/layers/embeddings.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/embeddings.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.embeddings`/
symbols:
  MRotaryEmbedding._forward_mrope: MRotaryEmbedding#_forward_mrope().
  YarnRotaryEmbedding.__call__: YarnRotaryEmbedding#__call__().
  RotaryEmbedding.rotary_dim: RotaryEmbedding#rotary_dim.
  YarnRotaryEmbedding._compute_yarn_inv_freq: YarnRotaryEmbedding#_compute_yarn_inv_freq().
  Embed.__call__: Embed#__call__().
  RotaryEmbedding.__call__: RotaryEmbedding#__call__().
  RotaryEmbedding: RotaryEmbedding#
  MRotaryEmbedding.mrope_section: MRotaryEmbedding#mrope_section.
  YarnRotaryEmbedding._inv_freq_np: YarnRotaryEmbedding#_inv_freq_np.
  Llama3RotaryEmbedding._compute_inv_freq: Llama3RotaryEmbedding#_compute_inv_freq().
  get_rope: get_rope().
  RotaryEmbedding.head_size: RotaryEmbedding#head_size.
  RotaryEmbedding.dtype: RotaryEmbedding#dtype.
  ParallelLMHead.bias: ParallelLMHead#bias.
  apply_rotary_emb: apply_rotary_emb().
  Embed: Embed#
  Embed.attend: Embed#attend().
  RotaryEmbedding._compute_cos_sin_cache: RotaryEmbedding#_compute_cos_sin_cache().
  RotaryEmbedding._compute_inv_freq: RotaryEmbedding#_compute_inv_freq().
  RotaryEmbedding.is_neox_style: RotaryEmbedding#is_neox_style.
  Llama3RotaryEmbedding.orig_max_position: Llama3RotaryEmbedding#orig_max_position.
  Embed.dtype: Embed#dtype.
  _ROPE_DICT._ROPE_DICT: _ROPE_DICT._ROPE_DICT.
  MRotaryEmbedding.__call__: MRotaryEmbedding#__call__().
  Embed.embedding: Embed#embedding.
  RotaryEmbedding._inv_freq_np: RotaryEmbedding#_inv_freq_np.
  MRotaryEmbedding.split_indices: MRotaryEmbedding#split_indices.
  RotaryEmbedding.__init__: RotaryEmbedding#__init__().
  Llama3RotaryEmbedding.low_freq_factor: Llama3RotaryEmbedding#low_freq_factor.
  Embed.mesh: Embed#mesh.
  MRotaryEmbedding.mrope_interleaved: MRotaryEmbedding#mrope_interleaved.
  ProportionalRotaryEmbedding: ProportionalRotaryEmbedding#
  Llama3RotaryEmbedding: Llama3RotaryEmbedding#
  _yarn_find_correction_range: _yarn_find_correction_range().
  YarnRotaryEmbedding.yarn_mscale_all_dim: YarnRotaryEmbedding#yarn_mscale_all_dim.
  YarnRotaryEmbedding: YarnRotaryEmbedding#
  Llama3RotaryEmbedding.high_freq_factor: Llama3RotaryEmbedding#high_freq_factor.
  YarnRotaryEmbedding.scaling_factor: YarnRotaryEmbedding#scaling_factor.
  Embed.promote_dtype: Embed#promote_dtype.
  ParallelLMHead: ParallelLMHead#
  ParallelLMHead.__init__: ParallelLMHead#__init__().
  ParallelLMHead.tie_weights: ParallelLMHead#tie_weights().
  ParallelLMHead.embedding: ParallelLMHead#embedding.
  MRotaryEmbedding: MRotaryEmbedding#
  MRotaryEmbedding.__init__: MRotaryEmbedding#__init__().
  ProportionalRotaryEmbedding.__init__: ProportionalRotaryEmbedding#__init__().
  rotary_embedding_forward: rotary_embedding_forward().
  Embed.num_embeddings: Embed#num_embeddings.
  Embed.features: Embed#features.
  RotaryEmbedding.base: RotaryEmbedding#base.
  apply_interleaved_rope: apply_interleaved_rope().
  Llama3RotaryEmbedding.scaling_factor: Llama3RotaryEmbedding#scaling_factor.
  _yarn_find_correction_dim: _yarn_find_correction_dim().
  _deepseek_yarn_get_mscale: _deepseek_yarn_get_mscale().
  YarnRotaryEmbedding._rope_mscale: YarnRotaryEmbedding#_rope_mscale.
  Embed.__init__: Embed#__init__().
  Embed.kernel_axes: Embed#kernel_axes.
  RotaryEmbedding.max_position_embeddings: RotaryEmbedding#max_position_embeddings.
  YarnRotaryEmbedding.original_max_position_embeddings: YarnRotaryEmbedding#original_max_position_embeddings.
  YarnRotaryEmbedding.beta_fast: YarnRotaryEmbedding#beta_fast.
  YarnRotaryEmbedding.beta_slow: YarnRotaryEmbedding#beta_slow.
  YarnRotaryEmbedding.yarn_mscale: YarnRotaryEmbedding#yarn_mscale.
  ParallelLMHead.__call__: ParallelLMHead#__call__().
  ProportionalRotaryEmbedding._inv_freq_np: ProportionalRotaryEmbedding#_inv_freq_np.
  Llama3RotaryEmbedding.__init__: Llama3RotaryEmbedding#__init__().
  _grok_yarn_get_mscale: _grok_yarn_get_mscale().
  YarnRotaryEmbedding.__init__: YarnRotaryEmbedding#__init__().
---
# Module: [`python/sgl_jax/srt/layers/embeddings.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py)

## Classes
### `Embed`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/layers/embeddings.py:32`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L32)
- doc: A parameterized function from integers \[0, n) to d-dimensional vectors.
- signature: `class Embed(nnx.Module):`
- members:
  - `__call__(self, inputs: jax.Array)` — [`L87`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L87) — Embeds the inputs along the last dimension.
  - `__init__(self, num_embeddings: int, features: int, dtype: jnp.dtype | None = None, param_dtype: jnp.dtype = jnp.bfloat16, promote_dtype: PromoteDtypeFn = dtypes.promote_dtype, kernel_axes: tuple[str | None, ...] = (None, "tensor"), mesh: jax.sharding.Mesh | None = None)` — [`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L44) — Sets up the embedding parameters for the model.
  - `attend(self, query: jax.Array)` — [`L110`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L110) — Attend over the embedding using a query array.
  - `dtype` — [`L82`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L82)
  - `embedding` — [`L71`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L71)
  - `features` — [`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L81)
  - `kernel_axes` — [`L79`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L79)
  - `mesh` — [`L84`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L84)
  - `num_embeddings` — [`L80`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L80)
  - `promote_dtype` — [`L83`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L83)
- uses (calls/refs, reference-scoped): [`named_scope`](../utils/profiling_utils.md#named_scope), [`ParallelLMHead`](embeddings.md#ParallelLMHead)
- used by: [`__call__`](logits_processor.md#LogitsProcessor.__call__), [`_get_logits`](logits_processor.md#LogitsProcessor._get_logits), [`bias`](embeddings.md#ParallelLMHead.bias), [`ParallelLMHead`](embeddings.md#ParallelLMHead), [`__init__`](embeddings.md#ParallelLMHead.__init__), [`embedding`](embeddings.md#ParallelLMHead.embedding), [`tie_weights`](embeddings.md#ParallelLMHead.tie_weights)

### `Llama3RotaryEmbedding`  ·  implements/extends RotaryEmbedding
- def: [`python/sgl_jax/srt/layers/embeddings.py:495`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L495)
- signature: `class Llama3RotaryEmbedding(RotaryEmbedding):`
- members:
  - `high_freq_factor` — [`L511`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L511)
  - `low_freq_factor` — [`L510`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L510)
  - `orig_max_position` — [`L512`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L512)
  - `scaling_factor` — [`L509`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L509)
- protocol/private: `__init__`[`L496`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L496), `_compute_inv_freq`[`L515`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L515)
- uses (calls/refs, reference-scoped): [`RotaryEmbedding`](embeddings.md#RotaryEmbedding), [`_compute_inv_freq`](embeddings.md#RotaryEmbedding._compute_inv_freq), [`__init__`](embeddings.md#RotaryEmbedding.__init__)
- used by: [`RotaryEmbedding`](embeddings.md#RotaryEmbedding), [`get_rope`](embeddings.md#get_rope), [`_compute_inv_freq`](embeddings.md#RotaryEmbedding._compute_inv_freq)

### `MRotaryEmbedding`  ·  implements/extends RotaryEmbedding
- def: [`python/sgl_jax/srt/layers/embeddings.py:306`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L306)
- doc: Rotary Embedding with Multimodal Sections for JAX.
- signature: `class MRotaryEmbedding(RotaryEmbedding):`
- members:
  - `__call__(self, positions: jax.Array, query: jax.Array, key: jax.Array)` — [`L357`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L357) — Args:
  - `mrope_interleaved` — [`L322`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L322)
  - `mrope_section` — [`L321`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L321)
  - `split_indices` — [`L355`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L355)
- protocol/private: `__init__`[`L309`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L309), `_forward_mrope`[`L385`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L385)
- uses (calls/refs, reference-scoped): [`rotary_dim`](embeddings.md#RotaryEmbedding.rotary_dim), [`__call__`](embeddings.md#RotaryEmbedding.__call__), [`RotaryEmbedding`](embeddings.md#RotaryEmbedding), [`head_size`](embeddings.md#RotaryEmbedding.head_size), [`dtype`](embeddings.md#RotaryEmbedding.dtype), [`apply_rotary_emb`](embeddings.md#apply_rotary_emb), [`is_neox_style`](embeddings.md#RotaryEmbedding.is_neox_style), [`_inv_freq_np`](embeddings.md#RotaryEmbedding._inv_freq_np), [`__init__`](embeddings.md#RotaryEmbedding.__init__), [`apply_interleaved_rope`](embeddings.md#apply_interleaved_rope)
- used by: [`RotaryEmbedding`](embeddings.md#RotaryEmbedding)

### `ParallelLMHead`  ·  implements/extends Embed
- def: [`python/sgl_jax/srt/layers/embeddings.py:127`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L127)
- doc: Language model head layer for vocabulary prediction.
- signature: `class ParallelLMHead(Embed):`
- members:
  - `__init__(self, num_embeddings: int, features: int, dtype: jnp.dtype | None = None, param_dtype: jnp.dtype = jnp.bfloat16, promote_dtype: PromoteDtypeFn = dtypes.promote_dtype, kernel_axes: tuple[str | None, ...] = ("tensor", None), mesh: jax.sharding.Mesh | None = None, use_bias: bool = False)` — [`L135`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L135) — Initialize the language model head.
  - `tie_weights(self, embed_tokens: Embed)` — [`L182`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L182) — Tie the weights with word embeddings.
  - `bias` — [`L171`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L171)
  - `embedding` — [`L184`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L184)
- protocol/private: `__call__`[`L187`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L187)
- uses (calls/refs, reference-scoped): [`Embed`](embeddings.md#Embed), [`embedding`](embeddings.md#Embed.embedding), [`features`](embeddings.md#Embed.features), [`num_embeddings`](embeddings.md#Embed.num_embeddings), [`__init__`](embeddings.md#Embed.__init__)
- used by: [`Embed`](embeddings.md#Embed)

### `ProportionalRotaryEmbedding`  ·  implements/extends RotaryEmbedding
- def: [`python/sgl_jax/srt/layers/embeddings.py:456`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L456)
- doc: Proportional RoPE (Gemma4 full-attention layers).
- signature: `class ProportionalRotaryEmbedding(RotaryEmbedding):`
- protocol/private: `__init__`[`L465`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L465), `_inv_freq_np`[`L492`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L492)
- uses (calls/refs, reference-scoped): [`RotaryEmbedding`](embeddings.md#RotaryEmbedding), [`__init__`](embeddings.md#RotaryEmbedding.__init__)
- used by: [`RotaryEmbedding`](embeddings.md#RotaryEmbedding), [`get_rope`](embeddings.md#get_rope)

### `RotaryEmbedding`
- def: [`python/sgl_jax/srt/layers/embeddings.py:192`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L192)
- doc: Rotary Position Embedding (safe to initialize inside JIT if needed).
- signature: `class RotaryEmbedding:`
- members:
  - `_compute_cos_sin_cache(self)` — [`L262`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L262) — Compute the cos and sin cache.
  - `_compute_inv_freq(self, base: int | float)` — [`L255`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L255) — Compute the inverse frequency.
  - `base` — [`L208`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L208)
  - `dtype` — [`L210`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L210)
  - `head_size` — [`L205`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L205)
  - `is_neox_style` — [`L209`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L209)
  - `max_position_embeddings` — [`L207`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L207)
  - `rotary_dim` — [`L206`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L206)
- protocol/private: `__call__`[`L216`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L216), `__init__`[`L195`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L195), `_inv_freq_np`[`L213`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L213)
- uses (calls/refs, reference-scoped): [`named_scope`](../utils/profiling_utils.md#named_scope), [`_compute_inv_freq`](embeddings.md#Llama3RotaryEmbedding._compute_inv_freq), [`apply_rotary_emb`](embeddings.md#apply_rotary_emb), [`Llama3RotaryEmbedding`](embeddings.md#Llama3RotaryEmbedding), [`ProportionalRotaryEmbedding`](embeddings.md#ProportionalRotaryEmbedding), [`MRotaryEmbedding`](embeddings.md#MRotaryEmbedding)
- used by: [`_forward_mrope`](embeddings.md#MRotaryEmbedding._forward_mrope), [`__call__`](embeddings.md#YarnRotaryEmbedding.__call__), [`_compute_yarn_inv_freq`](embeddings.md#YarnRotaryEmbedding._compute_yarn_inv_freq), [`_compute_inv_freq`](embeddings.md#Llama3RotaryEmbedding._compute_inv_freq), [`get_rope`](embeddings.md#get_rope), [`_ROPE_DICT`](embeddings.md#_ROPE_DICT._ROPE_DICT), [`orig_max_position`](embeddings.md#Llama3RotaryEmbedding.orig_max_position), [`__call__`](embeddings.md#MRotaryEmbedding.__call__), [`Llama3RotaryEmbedding`](embeddings.md#Llama3RotaryEmbedding), [`ProportionalRotaryEmbedding`](embeddings.md#ProportionalRotaryEmbedding), [`YarnRotaryEmbedding`](embeddings.md#YarnRotaryEmbedding), [`yarn_mscale_all_dim`](embeddings.md#YarnRotaryEmbedding.yarn_mscale_all_dim), [`MRotaryEmbedding`](embeddings.md#MRotaryEmbedding), [`__init__`](embeddings.md#MRotaryEmbedding.__init__), [`__init__`](embeddings.md#ProportionalRotaryEmbedding.__init__)

### `YarnRotaryEmbedding`
- def: [`python/sgl_jax/srt/layers/embeddings.py:764`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L764)
- members:
  - `beta_fast` — [`L789`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L789)
  - `beta_slow` — [`L790`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L790)
  - `original_max_position_embeddings` — [`L788`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L788)
  - `scaling_factor` — [`L787`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L787)
  - `yarn_mscale` — [`L791`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L791)
  - `yarn_mscale_all_dim` — [`L792`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L792)
- protocol/private: `__call__`[`L824`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L824), `__init__`[`L772`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L772), `_compute_yarn_inv_freq`[`L804`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L804), `_inv_freq_np`[`L797`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L797), `_rope_mscale`[`L802`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L802)
- uses (calls/refs, reference-scoped): [`named_scope`](../utils/profiling_utils.md#named_scope), [`rotary_dim`](embeddings.md#RotaryEmbedding.rotary_dim), [`RotaryEmbedding`](embeddings.md#RotaryEmbedding), [`head_size`](embeddings.md#RotaryEmbedding.head_size), [`dtype`](embeddings.md#RotaryEmbedding.dtype), [`apply_rotary_emb`](embeddings.md#apply_rotary_emb), [`is_neox_style`](embeddings.md#RotaryEmbedding.is_neox_style), [`__init__`](embeddings.md#RotaryEmbedding.__init__), [`_yarn_find_correction_range`](embeddings.md#_yarn_find_correction_range), [`_deepseek_yarn_get_mscale`](embeddings.md#_deepseek_yarn_get_mscale), [`base`](embeddings.md#RotaryEmbedding.base)
- used by: [`get_rope`](embeddings.md#get_rope)

## Functions
- `_deepseek_yarn_get_mscale(scale: float, mscale: float)` — [`L757`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L757) — Compute mscale factor for DeepSeek-style YaRN (parameterized by mscale coefficient).
- `_grok_yarn_get_mscale(scaling_factor: float)` — [`L726`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L726) — YaRN mscale variant used by Grok (sqrt form).
- `_yarn_find_correction_dim(num_rotations: int, dim: int, base: float = 10000, max_position_embeddings: int = 2048)` — [`L734`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L734)
- `_yarn_find_correction_range(low_rot: int, high_rot: int, dim: int, base: int, max_position_embeddings: int)` — [`L745`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L745)
- `apply_interleaved_rope(x: jax.Array, mrope_section: list[int])` — [`L272`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L272) — Apply interleaved MRoPE to 3D rotary embeddings in JAX.
- `apply_rotary_emb(x: jax.Array, cos: jax.Array, sin: jax.Array, is_neox_style: bool)` — [`L572`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L572) — Args:
- `get_rope(head_size: int, rotary_dim: int, max_position: int, base: int, is_neox_style: bool = True, rope_scaling: dict[str, Any] | None = None, dtype: jnp.dtype | None = jnp.bfloat16, partial_rotary_factor: float = 1, dual_chunk_attention_config: dict[str, Any] | None = None)` — [`L605`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L605)
- `rotary_embedding_forward(positions: jax.Array, query: jax.Array, key: jax.Array, cos_sin_cache: jax.Array, rotary_dim: int, head_size: int, is_neox_style: bool)` — [`L540`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L540) — Rotary Position Embedding.

## Module values
- `_ROPE_DICT` — [`L602`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/embeddings.py#L602)

