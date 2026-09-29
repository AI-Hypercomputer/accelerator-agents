---
title: 'Module: src/maxdiffusion/models/ltx_video/transformers/transformer3d.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/transformers/transformer3d.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.transformers.transformer3d`/
symbols:
  Transformer3DModel.transformer_blocks: Transformer3DModel#transformer_blocks.
  Transformer3DModel.__call__: Transformer3DModel#__call__().
  Transformer3DModel.caption_projection: Transformer3DModel#caption_projection.
  Transformer3DModel.adaln_single: Transformer3DModel#adaln_single.
  Transformer3DModel.inner_dim: Transformer3DModel#inner_dim.
  Transformer3DModel.patchify_proj: Transformer3DModel#patchify_proj.
  Transformer3DModel.freq_cis_pre_computer: Transformer3DModel#freq_cis_pre_computer.
  Transformer3DModel.proj_out: Transformer3DModel#proj_out.
  FreqsCisPrecomputer.__call__: FreqsCisPrecomputer#__call__().
  Transformer3DModel.use_rope: Transformer3DModel#use_rope.
  Transformer3DModel.dtype: Transformer3DModel#dtype.
  Transformer3DModel.weight_dtype: Transformer3DModel#weight_dtype.
  Transformer3DModel.matmul_precision: Transformer3DModel#matmul_precision.
  Transformer3DModel.create_skip_layer_mask: Transformer3DModel#create_skip_layer_mask().
  Transformer3DModel.num_layers: Transformer3DModel#num_layers.
  Transformer3DModel.scale_shift_table_init: Transformer3DModel#scale_shift_table_init().
  Transformer3DModel.scale_shift_table: Transformer3DModel#scale_shift_table.
  FreqsCisPrecomputer.get_fractional_positions: FreqsCisPrecomputer#get_fractional_positions().
  Transformer3DModel.setup: Transformer3DModel#setup().
  Transformer3DModel: Transformer3DModel#
  Transformer3DModel.num_attention_heads: Transformer3DModel#num_attention_heads.
  Transformer3DModel.attention_head_dim: Transformer3DModel#attention_head_dim.
  Transformer3DModel.out_channels: Transformer3DModel#out_channels.
  Transformer3DModel.adaptive_norm: Transformer3DModel#adaptive_norm.
  Transformer3DModel.caption_channels: Transformer3DModel#caption_channels.
  Transformer3DModel.timestep_scale_multiplier: Transformer3DModel#timestep_scale_multiplier.
  Transformer3DModel.output_scale: Transformer3DModel#output_scale.
  log_base: log_base().
  Transformer3DModel.dropout: Transformer3DModel#dropout.
  Transformer3DModel.cross_attention_dim: Transformer3DModel#cross_attention_dim.
  Transformer3DModel.attention_bias: Transformer3DModel#attention_bias.
  Transformer3DModel.activation_fn: Transformer3DModel#activation_fn.
  Transformer3DModel.num_embeds_ada_norm: Transformer3DModel#num_embeds_ada_norm.
  Transformer3DModel.only_cross_attention: Transformer3DModel#only_cross_attention.
  Transformer3DModel.double_self_attention: Transformer3DModel#double_self_attention.
  Transformer3DModel.upcast_attention: Transformer3DModel#upcast_attention.
  Transformer3DModel.standardization_norm: Transformer3DModel#standardization_norm.
  Transformer3DModel.norm_elementwise_affine: Transformer3DModel#norm_elementwise_affine.
  Transformer3DModel.norm_eps: Transformer3DModel#norm_eps.
  Transformer3DModel.attention_type: Transformer3DModel#attention_type.
  Transformer3DModel.use_tpu_flash_attention: Transformer3DModel#use_tpu_flash_attention.
  Transformer3DModel.qk_norm: Transformer3DModel#qk_norm.
  Transformer3DModel.positional_embedding_type: Transformer3DModel#positional_embedding_type.
  Transformer3DModel.positional_embedding_theta: Transformer3DModel#positional_embedding_theta.
  Transformer3DModel.positional_embedding_max_pos: Transformer3DModel#positional_embedding_max_pos.
  Transformer3DModel.ffn_dim_mult: Transformer3DModel#ffn_dim_mult.
  Transformer3DModel.attention_op: Transformer3DModel#attention_op.
  Transformer3DModel.sharding_mesh: Transformer3DModel#sharding_mesh.
  Transformer3DModel.param_scan_axis: Transformer3DModel#param_scan_axis.
  Transformer3DModel.gradient_checkpointing: Transformer3DModel#gradient_checkpointing.
  Transformer3DModel.norm_out: Transformer3DModel#norm_out.
  FreqsCisPrecomputer.positional_embedding_max_pos: FreqsCisPrecomputer#positional_embedding_max_pos.
  FreqsCisPrecomputer.positional_embedding_theta: FreqsCisPrecomputer#positional_embedding_theta.
  FreqsCisPrecomputer.inner_dim: FreqsCisPrecomputer#inner_dim.
  FreqsCisPrecomputer: FreqsCisPrecomputer#
  Transformer3DModel.init_weights: Transformer3DModel#init_weights().
---
# Module: [`src/maxdiffusion/models/ltx_video/transformers/transformer3d.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py)

## Classes
### `FreqsCisPrecomputer`
- def: [`src/maxdiffusion/models/ltx_video/transformers/transformer3d.py:272`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L272)
- members:
  - `get_fractional_positions(self, indices_grid: jax.Array)` — [`L282`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L282)
  - `inner_dim` — [`L280`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L280)
  - `positional_embedding_max_pos` — [`L278`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L278)
  - `positional_embedding_theta` — [`L279`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L279)
- protocol/private: `__call__`[`L290`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L290)
- uses (calls/refs, reference-scoped): [`log_base`](transformer3d.md#log_base)
- used by: [`freq_cis_pre_computer`](transformer3d.md#Transformer3DModel.freq_cis_pre_computer)

### `Transformer3DModel`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx_video/transformers/transformer3d.py:31`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L31)
- signature: `class Transformer3DModel(nn.Module):`
- members:
  - `create_skip_layer_mask(self, batch_size: int, num_conds: int, ptb_index: int, skip_block_list: Optional[List[int]] = None)` — [`L181`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L181)
  - `init_weights(self, in_channels, key, caption_channels, eval_only=True)` — [`L156`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L156)
  - `scale_shift_table_init(key)` — [`L89`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L89)
  - `setup(self)` — [`L66`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L66)
  - `activation_fn` — [`L39`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L39)
  - `adaln_single` — [`L81`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L81)
  - `adaptive_norm` — [`L44`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L44)
  - `attention_bias` — [`L38`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L38)
  - `attention_head_dim` — [`L33`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L33)
  - `attention_op` — [`L58`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L58)
  - `attention_type` — [`L48`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L48)
  - `caption_channels` — [`L49`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L49)
  - `caption_projection` — [`L148`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L148)
  - `cross_attention_dim` — [`L37`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L37)
  - `double_self_attention` — [`L42`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L42)
  - `dropout` — [`L36`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L36)
  - `dtype` — [`L59`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L59)
  - `ffn_dim_mult` — [`L56`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L56)
  - `freq_cis_pre_computer` — [`L78`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L78)
  - `gradient_checkpointing` — [`L64`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L64)
  - `inner_dim` — [`L68`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L68)
  - `matmul_precision` — [`L61`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L61)
  - `norm_elementwise_affine` — [`L46`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L46)
  - `norm_eps` — [`L47`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L47)
  - `norm_out` — [`L96`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L96)
  - `num_attention_heads` — [`L32`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L32)
  - `num_embeds_ada_norm` — [`L40`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L40)
  - `num_layers` — [`L35`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L35)
  - `only_cross_attention` — [`L41`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L41)
  - `out_channels` — [`L34`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L34)
  - `output_scale` — [`L57`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L57)
  - `param_scan_axis` — [`L63`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L63)
  - `patchify_proj` — [`L69`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L69)
  - `positional_embedding_max_pos` — [`L54`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L54)
  - `positional_embedding_theta` — [`L53`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L53)
  - `positional_embedding_type` — [`L52`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L52)
  - `proj_out` — [`L97`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L97)
  - `qk_norm` — [`L51`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L51)
  - `scale_shift_table` — [`L92`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L92)
  - `sharding_mesh` — [`L62`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L62)
  - `standardization_norm` — [`L45`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L45)
  - `timestep_scale_multiplier` — [`L55`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L55)
  - `transformer_blocks` — [`L112`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L112)
  - `upcast_attention` — [`L43`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L43)
  - `use_rope` — [`L106`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L106)
  - `use_tpu_flash_attention` — [`L50`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L50)
  - `weight_dtype` — [`L60`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L60)
- protocol/private: `__call__`[`L197`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L197)
- uses (calls/refs, reference-scoped): [`FreqsCisPrecomputer`](transformer3d.md#FreqsCisPrecomputer)
- used by: [`_import_structure`](../../../__init__.md#_import_structure), [`_import_structure`](../../__init__.md#_import_structure)

## Functions
- `log_base(x: jax.Array, base: jax.Array)` — [`L258`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers/transformer3d.py#L258) — Computes log of x with defined base.

