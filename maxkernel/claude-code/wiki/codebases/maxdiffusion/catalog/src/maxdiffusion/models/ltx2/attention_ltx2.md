---
title: 'Module: src/maxdiffusion/models/ltx2/attention_ltx2.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx2/attention_ltx2.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx2.attention_ltx2`/
symbols:
  LTX2Attention.__call__: LTX2Attention#__call__().
  LTX2RotaryPosEmbed.__call__: LTX2RotaryPosEmbed#__call__().
  Array: Array.
  LTX2Attention.dropout_rate: LTX2Attention#dropout_rate.
  LTX2RotaryPosEmbed.prepare_audio_coords: LTX2RotaryPosEmbed#prepare_audio_coords().
  LTX2Attention.attention_op: LTX2Attention#attention_op.
  LTX2RotaryPosEmbed.prepare_video_coords: LTX2RotaryPosEmbed#prepare_video_coords().
  LTX2Attention.__init__: LTX2Attention#__init__().
  LTX2Attention.dropout_layer: LTX2Attention#dropout_layer.
  LTX2RotaryPosEmbed.prepare_coords: LTX2RotaryPosEmbed#prepare_coords().
  LTX2RotaryPosEmbed.num_attention_heads: LTX2RotaryPosEmbed#num_attention_heads.
  LTX2Attention.inner_dim: LTX2Attention#inner_dim.
  LTX2Attention.to_out: LTX2Attention#to_out.
  LTX2Attention.to_gate_logits: LTX2Attention#to_gate_logits.
  apply_rotary_emb: apply_rotary_emb().
  apply_split_rotary_emb: apply_split_rotary_emb().
  LTX2RotaryPosEmbed.modality: LTX2RotaryPosEmbed#modality.
  LTX2RotaryPosEmbed.dim: LTX2RotaryPosEmbed#dim.
  LTX2RotaryPosEmbed.patch_size: LTX2RotaryPosEmbed#patch_size.
  LTX2RotaryPosEmbed.patch_size_t: LTX2RotaryPosEmbed#patch_size_t.
  Mesh: Mesh.
  DType: DType.
  BlockSizes: BlockSizes.
  LTX2Attention.to_q: LTX2Attention#to_q.
  LTX2Attention.to_k: LTX2Attention#to_k.
  LTX2Attention.to_v: LTX2Attention#to_v.
  LTX2Attention.norm_q: LTX2Attention#norm_q.
  LTX2Attention.norm_k: LTX2Attention#norm_k.
  LTX2RotaryPosEmbed.scale_factors: LTX2RotaryPosEmbed#scale_factors.
  LTX2RotaryPosEmbed.causal_offset: LTX2RotaryPosEmbed#causal_offset.
  LTX2RotaryPosEmbed.rope_type: LTX2RotaryPosEmbed#rope_type.
  LTX2RotaryPosEmbed.base_num_frames: LTX2RotaryPosEmbed#base_num_frames.
  LTX2RotaryPosEmbed.sampling_rate: LTX2RotaryPosEmbed#sampling_rate.
  LTX2RotaryPosEmbed.hop_length: LTX2RotaryPosEmbed#hop_length.
  LTX2RotaryPosEmbed.base_height: LTX2RotaryPosEmbed#base_height.
  LTX2RotaryPosEmbed.base_width: LTX2RotaryPosEmbed#base_width.
  LTX2RotaryPosEmbed.theta: LTX2RotaryPosEmbed#theta.
  LTX2RotaryPosEmbed.double_precision: LTX2RotaryPosEmbed#double_precision.
  LTX2Attention.heads: LTX2Attention#heads.
  LTX2Attention.rope_type: LTX2Attention#rope_type.
  LTX2Attention.dim_head: LTX2Attention#dim_head.
  LTX2RotaryPosEmbed: LTX2RotaryPosEmbed#
  LTX2RotaryPosEmbed.__init__: LTX2RotaryPosEmbed#__init__().
  LTX2RotaryPosEmbed.audio_latents_per_second: LTX2RotaryPosEmbed#audio_latents_per_second.
  LTX2Attention: LTX2Attention#
---
# Module: [`src/maxdiffusion/models/ltx2/attention_ltx2.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py)

## Classes
### `LTX2Attention`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx2/attention_ltx2.py:334`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L334)
- signature: `class LTX2Attention(nnx.Module):`
- members:
  - `attention_op` — [`L448`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L448)
  - `dim_head` — [`L358`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L358)
  - `dropout_layer` — [`L433`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L433)
  - `dropout_rate` — [`L360`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L360)
  - `heads` — [`L356`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L356)
  - `inner_dim` — [`L359`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L359)
  - `norm_k` — [`L411`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L411)
  - `norm_q` — [`L402`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L402)
  - `rope_type` — [`L357`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L357)
  - `to_gate_logits` — [`L438`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L438)
  - `to_k` — [`L394`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L394)
  - `to_out` — [`L422`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L422)
  - `to_q` — [`L382`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L382)
  - `to_v` — [`L397`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L397)
- protocol/private: `__call__`[`L461`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L461), `__init__`[`L336`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L336)
- uses (calls/refs, reference-scoped): [`apply_attention`](../attention_flax.md#NNXAttentionOp.apply_attention), [`Array`](attention_ltx2.md#Array), [`norm_scale`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs.norm_scale), [`get_sharding_specs`](logical_sharding_ltx2.md#get_sharding_specs), [`LTX2DiTShardingSpecs`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs), [`BATCH`](../../common_types.md#BATCH), [`D_KV`](../../common_types.md#D_KV), [`SELF_ATTN_HEAD`](../../common_types.md#SELF_ATTN_HEAD), [`SELF_ATTN_KV_LENGTH`](../../common_types.md#SELF_ATTN_KV_LENGTH), [`SELF_ATTN_Q_LENGTH`](../../common_types.md#SELF_ATTN_Q_LENGTH), [`apply_rotary_emb`](attention_ltx2.md#apply_rotary_emb), [`apply_split_rotary_emb`](attention_ltx2.md#apply_split_rotary_emb), [`NNXAttentionOp`](../attention_flax.md#NNXAttentionOp), [`BlockSizes`](attention_ltx2.md#BlockSizes), [`DType`](attention_ltx2.md#DType), [`Mesh`](attention_ltx2.md#Mesh), [`gate_logits_bias`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs.gate_logits_bias), [`gate_logits_kernel`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs.gate_logits_kernel), [`out_bias`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs.out_bias), [`out_kernel`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs.out_kernel), [`qkv_bias`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs.qkv_bias), [`qkv_kernel`](logical_sharding_ltx2.md#LTX2DiTShardingSpecs.qkv_kernel)

### `LTX2RotaryPosEmbed`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx2/attention_ltx2.py:108`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L108)
- doc: Video and audio rotary positional embeddings (RoPE) for the LTX-2.0 model.
- signature: `class LTX2RotaryPosEmbed(nnx.Module):`
- members:
  - `prepare_audio_coords(self, batch_size: int, num_frames: int, shift: int = 0)` — [`L204`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L204)
  - `prepare_coords(self, *args, **kwargs)` — [`L234`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L234)
  - `prepare_video_coords(self, batch_size: int, num_frames: int, height: int, width: int, fps: float = 24)` — [`L156`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L156)
  - `audio_latents_per_second` — [`L154`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L154)
  - `base_height` — [`L136`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L136)
  - `base_num_frames` — [`L135`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L135)
  - `base_width` — [`L137`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L137)
  - `causal_offset` — [`L142`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L142)
  - `dim` — [`L132`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L132)
  - `double_precision` — [`L144`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L144)
  - `hop_length` — [`L139`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L139)
  - `modality` — [`L143`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L143)
  - `num_attention_heads` — [`L146`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L146)
  - `patch_size` — [`L133`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L133)
  - `patch_size_t` — [`L134`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L134)
  - `rope_type` — [`L145`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L145)
  - `sampling_rate` — [`L138`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L138)
  - `scale_factors` — [`L140`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L140)
  - `theta` — [`L141`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L141)
- protocol/private: `__call__`[`L241`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L241), `__init__`[`L114`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L114)
- uses (calls/refs, reference-scoped): [`Array`](attention_ltx2.md#Array)

## Functions
- `apply_rotary_emb(x: Array, freqs: Tuple[Array, Array])` — [`L31`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L31) — Applies Interleaved RoPE to input x.
- `apply_split_rotary_emb(x: Array, freqs: Tuple[Array, Array])` — [`L60`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L60) — Applies Split RoPE to input x.

## Module values
- `Array` — [`L25`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L25)
- `BlockSizes` — [`L28`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L28)
- `DType` — [`L27`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L27)
- `Mesh` — [`L26`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/attention_ltx2.py#L26)

