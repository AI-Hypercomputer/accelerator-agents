---
title: 'Module: src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx2.logical_sharding_ltx2`/
symbols:
  STRATEGIES: STRATEGIES.
  LTX2DiTShardingSpecs.norm_scale: LTX2DiTShardingSpecs#norm_scale.
  get_sharding_specs: get_sharding_specs().
  LTX2DiTShardingSpecs: LTX2DiTShardingSpecs#
  LTX2DiTShardingSpecs.embed_kernel: LTX2DiTShardingSpecs#embed_kernel.
  LTX2DiTShardingSpecs.embed_bias: LTX2DiTShardingSpecs#embed_bias.
  LTX2DiTShardingSpecs.scale_shift_table: LTX2DiTShardingSpecs#scale_shift_table.
  LTX2DiTShardingSpecs.out_embed_kernel: LTX2DiTShardingSpecs#out_embed_kernel.
  LTX2DiTShardingSpecs.out_embed_bias: LTX2DiTShardingSpecs#out_embed_bias.
  LTX2DiTShardingSpecs.qkv_kernel: LTX2DiTShardingSpecs#qkv_kernel.
  LTX2DiTShardingSpecs.out_kernel: LTX2DiTShardingSpecs#out_kernel.
  LTX2DiTShardingSpecs.out_bias: LTX2DiTShardingSpecs#out_bias.
  LTX2DiTShardingSpecs.qkv_bias: LTX2DiTShardingSpecs#qkv_bias.
  LTX2DiTShardingSpecs.gate_logits_kernel: LTX2DiTShardingSpecs#gate_logits_kernel.
  LTX2DiTShardingSpecs.gate_logits_bias: LTX2DiTShardingSpecs#gate_logits_bias.
  LTX2DiTShardingSpecs.adaln_kernel: LTX2DiTShardingSpecs#adaln_kernel.
  LTX2DiTShardingSpecs.adaln_bias: LTX2DiTShardingSpecs#adaln_bias.
  TextConnectorShardingSpecs: TextConnectorShardingSpecs#
  VAEShardingSpecs: VAEShardingSpecs#
  LTX2DiTShardingSpecs.net_0_kernel: LTX2DiTShardingSpecs#net_0_kernel.
  LTX2DiTShardingSpecs.net_2_kernel: LTX2DiTShardingSpecs#net_2_kernel.
  LTX2DiTShardingSpecs.net_2_bias: LTX2DiTShardingSpecs#net_2_bias.
  LTX2DiTShardingSpecs.emb_linear_1_kernel: LTX2DiTShardingSpecs#emb_linear_1_kernel.
  LTX2DiTShardingSpecs.emb_linear_2_kernel: LTX2DiTShardingSpecs#emb_linear_2_kernel.
  LTX2DiTShardingSpecs.emb_linear_2_bias: LTX2DiTShardingSpecs#emb_linear_2_bias.
  TextConnectorShardingSpecs.net_0_kernel: TextConnectorShardingSpecs#net_0_kernel.
  TextConnectorShardingSpecs.net_2_kernel: TextConnectorShardingSpecs#net_2_kernel.
  TextConnectorShardingSpecs.net_2_bias: TextConnectorShardingSpecs#net_2_bias.
  TextConnectorShardingSpecs.qkv_kernel: TextConnectorShardingSpecs#qkv_kernel.
  TextConnectorShardingSpecs.out_kernel: TextConnectorShardingSpecs#out_kernel.
  TextConnectorShardingSpecs.out_bias: TextConnectorShardingSpecs#out_bias.
  TextConnectorShardingSpecs.qkv_bias: TextConnectorShardingSpecs#qkv_bias.
  TextConnectorShardingSpecs.gate_logits_kernel: TextConnectorShardingSpecs#gate_logits_kernel.
  TextConnectorShardingSpecs.gate_logits_bias: TextConnectorShardingSpecs#gate_logits_bias.
  TextConnectorShardingSpecs.norm_scale: TextConnectorShardingSpecs#norm_scale.
  VAEShardingSpecs.emb_linear_1_kernel: VAEShardingSpecs#emb_linear_1_kernel.
  VAEShardingSpecs.emb_linear_2_kernel: VAEShardingSpecs#emb_linear_2_kernel.
  VAEShardingSpecs.emb_linear_2_bias: VAEShardingSpecs#emb_linear_2_bias.
  LTX2DiTShardingSpecs.net_0_bias: LTX2DiTShardingSpecs#net_0_bias.
  LTX2DiTShardingSpecs.emb_linear_1_bias: LTX2DiTShardingSpecs#emb_linear_1_bias.
  TextConnectorShardingSpecs.net_0_bias: TextConnectorShardingSpecs#net_0_bias.
  TextConnectorShardingSpecs.proj_kernel: TextConnectorShardingSpecs#proj_kernel.
  TextConnectorShardingSpecs.proj_bias: TextConnectorShardingSpecs#proj_bias.
  VAEShardingSpecs.emb_linear_1_bias: VAEShardingSpecs#emb_linear_1_bias.
  VAEShardingSpecs.scale_shift_table: VAEShardingSpecs#scale_shift_table.
  VAEShardingSpecs.per_channel_scale: VAEShardingSpecs#per_channel_scale.
---
# Module: [`src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py)

## Classes
### `LTX2DiTShardingSpecs`
- def: [`src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py:25`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L25)
- doc: Sharding specs for the LTX2 Diffusion Transformer.
- signature: `class LTX2DiTShardingSpecs:`
- members:
  - `adaln_bias` — [`L46`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L46)
  - `adaln_kernel` — [`L45`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L45)
  - `emb_linear_1_bias` — [`L53`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L53)
  - `emb_linear_1_kernel` — [`L52`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L52)
  - `emb_linear_2_bias` — [`L55`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L55)
  - `emb_linear_2_kernel` — [`L54`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L54)
  - `embed_bias` — [`L44`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L44)
  - `embed_kernel` — [`L43`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L43)
  - `gate_logits_bias` — [`L34`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L34)
  - `gate_logits_kernel` — [`L33`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L33)
  - `net_0_bias` — [`L38`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L38)
  - `net_0_kernel` — [`L37`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L37)
  - `net_2_bias` — [`L40`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L40)
  - `net_2_kernel` — [`L39`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L39)
  - `norm_scale` — [`L58`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L58)
  - `out_bias` — [`L31`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L31)
  - `out_embed_bias` — [`L49`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L49)
  - `out_embed_kernel` — [`L48`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L48)
  - `out_kernel` — [`L30`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L30)
  - `qkv_bias` — [`L32`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L32)
  - `qkv_kernel` — [`L29`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L29)
  - `scale_shift_table` — [`L47`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L47)
- used by: [`STRATEGIES`](logical_sharding_ltx2.md#STRATEGIES), [`audio_proj_in`](transformer_ltx2.md#LTX2VideoTransformer3DModel.audio_proj_in), [`dropout_rate`](attention_ltx2.md#LTX2Attention.dropout_rate), [`proj_in`](transformer_ltx2.md#LTX2VideoTransformer3DModel.proj_in), [`audio_proj_out`](transformer_ltx2.md#LTX2VideoTransformer3DModel.audio_proj_out), [`av_cross_attn_audio_v2a_gate`](transformer_ltx2.md#LTX2VideoTransformer3DModel.av_cross_attn_audio_v2a_gate), [`norm1`](transformer_ltx2.md#LTX2VideoTransformerBlock.norm1), [`norm2`](transformer_ltx2.md#LTX2VideoTransformerBlock.norm2), [`proj_out`](transformer_ltx2.md#LTX2VideoTransformer3DModel.proj_out), [`audio_norm1`](transformer_ltx2.md#LTX2VideoTransformerBlock.audio_norm1), [`audio_norm2`](transformer_ltx2.md#LTX2VideoTransformerBlock.audio_norm2), [`audio_norm3`](transformer_ltx2.md#LTX2VideoTransformerBlock.audio_norm3), [`audio_to_video_norm`](transformer_ltx2.md#LTX2VideoTransformerBlock.audio_to_video_norm), [`norm3`](transformer_ltx2.md#LTX2VideoTransformerBlock.norm3), [`video_to_audio_norm`](transformer_ltx2.md#LTX2VideoTransformerBlock.video_to_audio_norm), [`__init__`](attention_ltx2.md#LTX2Attention.__init__), [`cross_attn_mod`](transformer_ltx2.md#LTX2VideoTransformerBlock.cross_attn_mod), [`linear`](transformer_ltx2.md#LTX2AdaLayerNormSingle.linear), [`to_gate_logits`](attention_ltx2.md#LTX2Attention.to_gate_logits), [`__init__`](transformer_ltx2.md#LTX2VideoTransformer3DModel.__init__), [`__init__`](transformer_ltx2.md#LTX2AdaLayerNormSingle.__init__), [`__init__`](transformer_ltx2.md#LTX2VideoTransformerBlock.__init__)

### `TextConnectorShardingSpecs`
- def: [`src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py:62`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L62)
- doc: Specs for the Text Connector execution.
- signature: `class TextConnectorShardingSpecs:`
- members:
  - `gate_logits_bias` — [`L77`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L77)
  - `gate_logits_kernel` — [`L76`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L76)
  - `net_0_bias` — [`L67`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L67)
  - `net_0_kernel` — [`L66`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L66)
  - `net_2_bias` — [`L69`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L69)
  - `net_2_kernel` — [`L68`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L68)
  - `norm_scale` — [`L78`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L78)
  - `out_bias` — [`L74`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L74)
  - `out_kernel` — [`L73`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L73)
  - `proj_bias` — [`L82`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L82)
  - `proj_kernel` — [`L81`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L81)
  - `qkv_bias` — [`L75`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L75)
  - `qkv_kernel` — [`L72`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L72)
- used by: [`STRATEGIES`](logical_sharding_ltx2.md#STRATEGIES)

### `VAEShardingSpecs`
- def: [`src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py:86`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L86)
- doc: Sharding specs for the VAE.
- signature: `class VAEShardingSpecs:`
- members:
  - `emb_linear_1_bias` — [`L91`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L91)
  - `emb_linear_1_kernel` — [`L90`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L90)
  - `emb_linear_2_bias` — [`L93`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L93)
  - `emb_linear_2_kernel` — [`L92`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L92)
  - `per_channel_scale` — [`L97`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L97)
  - `scale_shift_table` — [`L96`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L96)
- used by: [`STRATEGIES`](logical_sharding_ltx2.md#STRATEGIES)

## Functions
- `get_sharding_specs(strategy_name: str, component_name: str)` — [`L151`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L151) — Unified factory to get specs for any component.

## Module values
- `STRATEGIES` — [`L101`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/logical_sharding_ltx2.py#L101)

