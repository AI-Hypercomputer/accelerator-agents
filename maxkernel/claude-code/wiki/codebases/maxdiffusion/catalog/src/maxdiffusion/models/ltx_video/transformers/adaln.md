---
title: 'Module: src/maxdiffusion/models/ltx_video/transformers/adaln.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/transformers/adaln.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.transformers.adaln`/
symbols:
  AlphaCombinedTimestepSizeEmbeddings.timestep_embedder: AlphaCombinedTimestepSizeEmbeddings#timestep_embedder.
  AdaLayerNormSingle.emb: AdaLayerNormSingle#emb.
  TimestepEmbedding.linear_1: TimestepEmbedding#linear_1.
  AdaLayerNormSingle.linear: AdaLayerNormSingle#linear.
  Timesteps.__call__: Timesteps#__call__().
  TimestepEmbedding.linear_2: TimestepEmbedding#linear_2.
  AlphaCombinedTimestepSizeEmbeddings.time_proj: AlphaCombinedTimestepSizeEmbeddings#time_proj.
  TimestepEmbedding.act: TimestepEmbedding#act.
  TimestepEmbedding.__call__: TimestepEmbedding#__call__().
  AdaLayerNormSingle.__call__: AdaLayerNormSingle#__call__().
  AlphaCombinedTimestepSizeEmbeddings.__call__: AlphaCombinedTimestepSizeEmbeddings#__call__().
  TimestepEmbedding.time_embed_dim: TimestepEmbedding#time_embed_dim.
  TimestepEmbedding.dtype: TimestepEmbedding#dtype.
  TimestepEmbedding.weight_dtype: TimestepEmbedding#weight_dtype.
  TimestepEmbedding.matmul_precision: TimestepEmbedding#matmul_precision.
  AdaLayerNormSingle.embedding_dim: AdaLayerNormSingle#embedding_dim.
  AlphaCombinedTimestepSizeEmbeddings.outdim: AlphaCombinedTimestepSizeEmbeddings#outdim.
  TimestepEmbedding.out_dim: TimestepEmbedding#out_dim.
  TimestepEmbedding.sample_proj_bias: TimestepEmbedding#sample_proj_bias.
  Timesteps.num_channels: Timesteps#num_channels.
  Timesteps.flip_sin_to_cos: Timesteps#flip_sin_to_cos.
  Timesteps.downscale_freq_shift: Timesteps#downscale_freq_shift.
  AlphaCombinedTimestepSizeEmbeddings.size_emb_dim: AlphaCombinedTimestepSizeEmbeddings#size_emb_dim.
  AlphaCombinedTimestepSizeEmbeddings.dtype: AlphaCombinedTimestepSizeEmbeddings#dtype.
  AlphaCombinedTimestepSizeEmbeddings.weight_dtype: AlphaCombinedTimestepSizeEmbeddings#weight_dtype.
  AlphaCombinedTimestepSizeEmbeddings.matmul_precision: AlphaCombinedTimestepSizeEmbeddings#matmul_precision.
  AdaLayerNormSingle.dtype: AdaLayerNormSingle#dtype.
  AdaLayerNormSingle.weight_dtype: AdaLayerNormSingle#weight_dtype.
  AdaLayerNormSingle.matmul_precision: AdaLayerNormSingle#matmul_precision.
  get_timestep_embedding_multidim: get_timestep_embedding_multidim().
  TimestepEmbedding: TimestepEmbedding#
  TimestepEmbedding.in_channels: TimestepEmbedding#in_channels.
  TimestepEmbedding.act_fn: TimestepEmbedding#act_fn.
  Timesteps: Timesteps#
  Timesteps.scale: Timesteps#scale.
  AlphaCombinedTimestepSizeEmbeddings: AlphaCombinedTimestepSizeEmbeddings#
  AlphaCombinedTimestepSizeEmbeddings.embedding_dim: AlphaCombinedTimestepSizeEmbeddings#embedding_dim.
  AdaLayerNormSingle.embedding_coefficient: AdaLayerNormSingle#embedding_coefficient.
  AdaLayerNormSingle.silu: AdaLayerNormSingle#silu.
  TimestepEmbedding.setup: TimestepEmbedding#setup().
  AlphaCombinedTimestepSizeEmbeddings.setup: AlphaCombinedTimestepSizeEmbeddings#setup().
  AdaLayerNormSingle: AdaLayerNormSingle#
  AdaLayerNormSingle.setup: AdaLayerNormSingle#setup().
---
# Module: [`src/maxdiffusion/models/ltx_video/transformers/adaln.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py)

## Classes
### `AdaLayerNormSingle`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/transformers/adaln.py:154`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L154)
- doc: Norm layer adaptive layer norm single (adaLN-single).
- signature: `class AdaLayerNormSingle(nn.Module):`
- members:
  - `__call__(self, timestep: jnp.ndarray, added_cond_kwargs: Optional[Dict[str, jnp.ndarray]] = None, batch_size: Optional[int] = None, hidden_dtype: Optional[jnp.dtype] = None)` — [`L191`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L191) — Compute AdaLayerNorm-Single modulation.
  - `setup(self)` — [`L170`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L170)
  - `dtype` — [`L166`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L166)
  - `emb` — [`L171`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L171)
  - `embedding_coefficient` — [`L165`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L165)
  - `embedding_dim` — [`L164`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L164)
  - `linear` — [`L181`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L181)
  - `matmul_precision` — [`L168`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L168)
  - `silu` — [`L180`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L180) — ---
  - `weight_dtype` — [`L167`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L167)
- uses (calls/refs, reference-scoped): [`dtype`](adaln.md#AlphaCombinedTimestepSizeEmbeddings.dtype), [`matmul_precision`](adaln.md#AlphaCombinedTimestepSizeEmbeddings.matmul_precision), [`size_emb_dim`](adaln.md#AlphaCombinedTimestepSizeEmbeddings.size_emb_dim), [`weight_dtype`](adaln.md#AlphaCombinedTimestepSizeEmbeddings.weight_dtype), [`AlphaCombinedTimestepSizeEmbeddings`](adaln.md#AlphaCombinedTimestepSizeEmbeddings)

### `AlphaCombinedTimestepSizeEmbeddings`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/transformers/adaln.py:128`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L128)
- signature: `class AlphaCombinedTimestepSizeEmbeddings(nn.Module):`
- members:
  - `setup(self)` — [`L135`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L135) — Initialize sub-modules.
  - `dtype` — [`L131`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L131)
  - `embedding_dim` — [`L129`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L129)
  - `matmul_precision` — [`L133`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L133)
  - `outdim` — [`L137`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L137)
  - `size_emb_dim` — [`L130`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L130)
  - `time_proj` — [`L138`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L138)
  - `timestep_embedder` — [`L139`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L139)
  - `weight_dtype` — [`L132`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L132)
- protocol/private: `__call__`[`L148`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L148)
- uses (calls/refs, reference-scoped): [`dtype`](adaln.md#TimestepEmbedding.dtype), [`matmul_precision`](adaln.md#TimestepEmbedding.matmul_precision), [`time_embed_dim`](adaln.md#TimestepEmbedding.time_embed_dim), [`weight_dtype`](adaln.md#TimestepEmbedding.weight_dtype), [`downscale_freq_shift`](adaln.md#Timesteps.downscale_freq_shift), [`flip_sin_to_cos`](adaln.md#Timesteps.flip_sin_to_cos), [`num_channels`](adaln.md#Timesteps.num_channels), [`TimestepEmbedding`](adaln.md#TimestepEmbedding), [`Timesteps`](adaln.md#Timesteps), [`in_channels`](adaln.md#TimestepEmbedding.in_channels)
- used by: [`emb`](adaln.md#AdaLayerNormSingle.emb)

### `TimestepEmbedding`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/transformers/adaln.py:69`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L69)
- signature: `class TimestepEmbedding(nn.Module):`
- members:
  - `setup(self)` — [`L79`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L79) — Initialize layers efficiently
  - `act` — [`L91`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L91)
  - `act_fn` — [`L72`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L72)
  - `dtype` — [`L75`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L75)
  - `in_channels` — [`L70`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L70)
  - `linear_1` — [`L81`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L81)
  - `linear_2` — [`L93`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L93)
  - `matmul_precision` — [`L77`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L77)
  - `out_dim` — [`L73`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L73)
  - `sample_proj_bias` — [`L74`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L74)
  - `time_embed_dim` — [`L71`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L71)
  - `weight_dtype` — [`L76`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L76)
- protocol/private: `__call__`[`L103`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L103)
- used by: [`timestep_embedder`](adaln.md#AlphaCombinedTimestepSizeEmbeddings.timestep_embedder)

### `Timesteps`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/transformers/adaln.py:111`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L111)
- signature: `class Timesteps(nn.Module):`
- members:
  - `downscale_freq_shift` — [`L114`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L114)
  - `flip_sin_to_cos` — [`L113`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L113)
  - `num_channels` — [`L112`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L112)
  - `scale` — [`L115`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L115)
- protocol/private: `__call__`[`L117`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L117)
- uses (calls/refs, reference-scoped): [`get_timestep_embedding_multidim`](adaln.md#get_timestep_embedding_multidim)
- used by: [`time_proj`](adaln.md#AlphaCombinedTimestepSizeEmbeddings.time_proj)

## Functions
- `get_timestep_embedding_multidim(timesteps: jnp.ndarray, embedding_dim: int, flip_sin_to_cos: bool = False, downscale_freq_shift: float = 1, scale: float = 1, max_period: int = 10000)` — [`L28`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/adaln.py#L28) — Computes sinusoidal timestep embeddings while preserving the original dimensions.

