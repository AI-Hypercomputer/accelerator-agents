---
title: 'Module: easydel/layers/embeddings/_embeddings.py'
type: catalog
provenance: extracted
module: easydel/layers/embeddings/_embeddings.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.embeddings._embeddings`/
symbols:
  Embed: Embed#
  Embed.__call__: Embed#__call__().
  Embed.embedding: Embed#embedding.
  Embed.attend: Embed#attend().
  Embed.craft_sharding: Embed#craft_sharding().
  Embed.dtype: Embed#dtype.
  Embed.__init__: Embed#__init__().
  Embed.promote_dtype: Embed#promote_dtype.
  default_embed_init: default_embed_init.
  Embed.num_embeddings: Embed#num_embeddings.
  Embed.features: Embed#features.
  Embed.param_dtype: Embed#param_dtype.
  Embed.embedding_init: Embed#embedding_init.
---
# Module: [`easydel/layers/embeddings/_embeddings.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/embeddings/_embeddings.py)

## Classes
### `Embed`  ·  implements/extends Module
- def: [`easydel/layers/embeddings/_embeddings.py:66`](../../../../../../../raw/code/EasyDeL/easydel/layers/embeddings/_embeddings.py#L66)
- doc: Embedding layer that converts integer indices to dense vector representations.
- signature: `class Embed(nn.Module):`
- members:
  - `__call__(self, inputs: Array)` — [`L160`](../../../../../../../raw/code/EasyDeL/easydel/layers/embeddings/_embeddings.py#L160) — Embed the input indices along the last dimension.
  - `__init__(self, num_embeddings: int, features: int, *, dtype: Dtype | None = None, param_dtype: Dtype = jnp.float32, embedding_init: Initializer = default_embed_init, promote_dtype: PromoteDtypeFn = dtypes.promote_dtype, rngs: nn.Rngs)` — [`L112`](../../../../../../../raw/code/EasyDeL/easydel/layers/embeddings/_embeddings.py#L112) — Initialize the embedding layer.
  - `attend(self, query: Array)` — [`L203`](../../../../../../../raw/code/EasyDeL/easydel/layers/embeddings/_embeddings.py#L203) — Compute attention scores between query vectors and embeddings.
  - `craft_sharding(self, *, partition_manager=None, **_kwargs)` — [`L192`](../../../../../../../raw/code/EasyDeL/easydel/layers/embeddings/_embeddings.py#L192) — Return dynamic partition specs for this module's parameters.
  - `dtype` — [`L155`](../../../../../../../raw/code/EasyDeL/easydel/layers/embeddings/_embeddings.py#L155)
  - `embedding` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/layers/embeddings/_embeddings.py#L151)
  - `embedding_init` — [`L157`](../../../../../../../raw/code/EasyDeL/easydel/layers/embeddings/_embeddings.py#L157)
  - `features` — [`L154`](../../../../../../../raw/code/EasyDeL/easydel/layers/embeddings/_embeddings.py#L154)
  - `num_embeddings` — [`L153`](../../../../../../../raw/code/EasyDeL/easydel/layers/embeddings/_embeddings.py#L153)
  - `param_dtype` — [`L156`](../../../../../../../raw/code/EasyDeL/easydel/layers/embeddings/_embeddings.py#L156)
  - `promote_dtype` — [`L158`](../../../../../../../raw/code/EasyDeL/easydel/layers/embeddings/_embeddings.py#L158)
- uses (calls/refs, reference-scoped): [`resolve_safe_sharding`](../_sharding.md#resolve_safe_sharding), [`default_embed_init`](_embeddings.md#default_embed_init)
- used by: [`get_embedding`](../../infra/base_module.md#EasyDeLBaseModule.get_embedding), [`__all__`](../../modules/whisper/modeling_whisper.md#__all__), [`_build_transform_fn`](../../infra/base_module.md#EasyDeLBaseModule._build_transform_fn), [`codec_embedding`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeTalkerCodePredictorModel.codec_embedding), [`embed_tokens`](../../modules/glm4v/modeling_glm4v.md#Glm4vTextModel.embed_tokens), [`embed_tokens`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeThinkerTextModel.embed_tokens), [`embed_tokens`](../../modules/qwen3_vl/modeling_qwen3_vl.md#Qwen3VLTextModel.embed_tokens), [`embed_tokens`](../../modules/qwen3_vl_moe/modeling_qwen3_vl_moe.md#Qwen3VLMoeTextModel.embed_tokens), [`embed_tokens`](../../modules/glm4_moe_lite/modeling_glm4_moe_lite.md#Glm4MoeLiteModel.embed_tokens), [`embed_tokens`](../../modules/glm_moe_dsa/modeling_glm_moe_dsa.md#GlmMoeDsaModel.embed_tokens), [`embed_tokens`](../../modules/gemma4/modeling_gemma4.md#Gemma4TextModel.embed_tokens), [`make_lm_head_fn`](../../infra/base_module.md#EasyDeLBaseModule.make_lm_head_fn), [`codec_embedding`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeTalkerModel.codec_embedding), [`embed_tokens`](../../modules/cohere/modeling_cohere.md#CohereModel.embed_tokens), [`embed_tokens`](../../modules/cohere2/modeling_cohere2.md#Cohere2Model.embed_tokens), [`embed_tokens`](../../modules/glm4v_moe/modeling_glm4v_moe.md#Glm4vMoeTextModel.embed_tokens), [`embed_tokens`](../../modules/kimi_linear/modeling_kimi_linear.md#KimiLinearModel.embed_tokens), [`embed_tokens`](../../modules/minimax/modeling_minimax.md#MiniMaxModel.embed_tokens), [`embed_tokens`](../../modules/mistral/modeling_mistral.md#MistralModel.embed_tokens), [`embed_tokens`](../../modules/olmo/modeling_olmo.md#OlmoModel.embed_tokens), [`embed_tokens`](../../modules/olmo2/modeling_olmo2.md#Olmo2Model.embed_tokens), [`embed_tokens`](../../modules/olmo3/modeling_olmo3.md#Olmo3Model.embed_tokens), [`embed_tokens`](../../modules/qwen2_moe/modeling_qwen2_moe.md#Qwen2MoeModel.embed_tokens), [`embed_tokens`](../../modules/qwen3/modeling_qwen3.md#Qwen3Model.embed_tokens), [`embed_tokens`](../../modules/qwen3_moe/modeling_qwen3_moe.md#Qwen3MoeModel.embed_tokens), [`embed_tokens`](../../modules/qwen3_next/modeling_qwen3_next.md#Qwen3NextModel.embed_tokens), [`embed_tokens`](../../modules/xerxes2/modeling_xerxes2.md#Xerxes2Model.embed_tokens), [`tok_embeddings`](../../modules/internlm2/modeling_internlm2.md#InternLM2Model.tok_embeddings), [`embed_tokens`](../../modules/qwen3_omni_moe/modeling_qwen3_omni_moe.md#Qwen3OmniMoeModel.embed_tokens), [`embed_tokens`](../../modules/exaone4/modeling_exaone4.md#Exaone4Model.embed_tokens), [`embed_tokens`](../../modules/falcon_h1/modeling_falcon_h1.md#FalconH1Model.embed_tokens), [`embed_tokens`](../../modules/gemma/modeling_gemma.md#GemmaModel.embed_tokens), [`embed_tokens`](../../modules/gemma2/modeling_gemma2.md#Gemma2Model.embed_tokens), [`embed_tokens`](../../modules/gemma3/modeling_gemma3.md#Gemma3TextModel.embed_tokens), [`embed_tokens`](../../modules/gpt_oss/modeling_gpt_oss.md#GptOssModel.embed_tokens), [`embed_tokens`](../../modules/mixtral/modeling_mixtral.md#MixtralModel.embed_tokens), [`embed_tokens`](../../modules/smollm3/modeling_smollm3.md#SmolLM3Model.embed_tokens), [`embed_tokens`](../../modules/stablelm/modeling_stablelm.md#StableLmModel.embed_tokens), [`embed_tokens`](../../modules/xerxes/modeling_xerxes.md#XerxesModel.embed_tokens), [`word_embeddings`](../../modules/falcon/modeling_falcon.md#FalconModel.word_embeddings)  (+54 more)

## Module values
- `default_embed_init` — [`L58`](../../../../../../../raw/code/EasyDeL/easydel/layers/embeddings/_embeddings.py#L58)

