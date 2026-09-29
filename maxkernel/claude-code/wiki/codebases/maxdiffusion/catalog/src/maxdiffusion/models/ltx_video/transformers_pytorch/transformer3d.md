---
title: 'Module: src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.transformers_pytorch.transformer3d`/
symbols:
  Transformer3DModel.forward: Transformer3DModel#forward().
  Transformer3DModel.from_pretrained: Transformer3DModel#from_pretrained().
  Transformer3DModel.precompute_freqs_cis: Transformer3DModel#precompute_freqs_cis().
  Transformer3DModel.set_use_tpu_flash_attention: Transformer3DModel#set_use_tpu_flash_attention().
  Transformer3DModel.transformer_blocks: Transformer3DModel#transformer_blocks.
  Transformer3DModel.caption_projection: Transformer3DModel#caption_projection.
  Transformer3DModel.timestep_scale_multiplier: Transformer3DModel#timestep_scale_multiplier.
  Transformer3DModel.adaln_single: Transformer3DModel#adaln_single.
  Transformer3DModel.use_rope: Transformer3DModel#use_rope.
  Transformer3DModel.proj_out: Transformer3DModel#proj_out.
  Transformer3DModel.get_fractional_positions: Transformer3DModel#get_fractional_positions().
  Transformer3DModel.create_custom_forward: Transformer3DModel#create_custom_forward().
  Transformer3DModel.positional_embedding_type: Transformer3DModel#positional_embedding_type.
  Transformer3DModel.create_skip_layer_mask: Transformer3DModel#create_skip_layer_mask().
  Transformer3DModel.use_tpu_flash_attention: Transformer3DModel#use_tpu_flash_attention.
  logger: logger.
  Transformer3DModelOutput: Transformer3DModelOutput#
  Transformer3DModelOutput.sample: Transformer3DModelOutput#sample.
  Transformer3DModel: Transformer3DModel#
  Transformer3DModel.inner_dim: Transformer3DModel#inner_dim.
  Transformer3DModel.patchify_proj: Transformer3DModel#patchify_proj.
  Transformer3DModel.positional_embedding_theta: Transformer3DModel#positional_embedding_theta.
  Transformer3DModel.positional_embedding_max_pos: Transformer3DModel#positional_embedding_max_pos.
  Transformer3DModel.out_channels: Transformer3DModel#out_channels.
  Transformer3DModel.norm_out: Transformer3DModel#norm_out.
  Transformer3DModel.scale_shift_table: Transformer3DModel#scale_shift_table.
  Transformer3DModel.gradient_checkpointing: Transformer3DModel#gradient_checkpointing.
  Transformer3DModel.custom_forward: Transformer3DModel#custom_forward().
  Transformer3DModel._supports_gradient_checkpointing: Transformer3DModel#_supports_gradient_checkpointing.
  Transformer3DModel.__init__: Transformer3DModel#__init__().
  Transformer3DModel.use_linear_projection: Transformer3DModel#use_linear_projection.
  Transformer3DModel.num_attention_heads: Transformer3DModel#num_attention_heads.
  Transformer3DModel.attention_head_dim: Transformer3DModel#attention_head_dim.
  Transformer3DModel._set_gradient_checkpointing: Transformer3DModel#_set_gradient_checkpointing().
  Transformer3DModel.load_state_dict: Transformer3DModel#load_state_dict().
---
# Module: [`src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py)

## Classes
### `Transformer3DModel`
- def: [`src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py:64`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L64)
- signature: `class Transformer3DModel(ModelMixin, ConfigMixin):`
- members:
  - `create_custom_forward(module, return_dict=None)` — [`L424`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L424)
  - `create_skip_layer_mask(self, batch_size: int, num_conds: int, ptb_index: int, skip_block_list: Optional[List[int]] = None)` — [`L177`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L177)
  - `custom_forward(*inputs)` — [`L425`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L425)
  - `forward(self, hidden_states: torch.Tensor, indices_grid: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None, timestep: Optional[torch.LongTensor] = None, class_labels: Optional[torch.LongTensor] = None, cross_attention_kwargs: Dict[str, Any] = None, attention_mask: Optional[torch.Tensor] = None, encoder_attention_mask: Optional[torch.Tensor] = None, skip_layer_mask: Optional[torch.Tensor] = None, skip_layer_strategy: Optional[SkipLayerStrategy] = None, return_dict: bool = True)` — [`L313`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L313) — The [`Transformer2DModel`] forward method.
  - `from_pretrained(cls, pretrained_model_path: Optional[Union[str, os.PathLike]], *args, **kwargs)` — [`L263`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L263)
  - `get_fractional_positions(self, indices_grid)` — [`L196`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L196)
  - `load_state_dict(self, state_dict: Dict, *args, **kwargs)` — [`L248`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L248)
  - `precompute_freqs_cis(self, indices_grid, spacing="exp")` — [`L203`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L203)
  - `set_use_tpu_flash_attention(self)` — [`L166`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L166) — Function sets the flag in this object and propagates down the children. The flag will enforce the usage of TPU
  - `adaln_single` — [`L156`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L156)
  - `attention_head_dim` — [`L104`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L104)
  - `caption_projection` — [`L160`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L160)
  - `gradient_checkpointing` — [`L164`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L164)
  - `inner_dim` — [`L106`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L106)
  - `norm_out` — [`L152`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L152)
  - `num_attention_heads` — [`L103`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L103)
  - `out_channels` — [`L151`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L151)
  - `patchify_proj` — [`L107`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L107)
  - `positional_embedding_max_pos` — [`L110`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L110)
  - `positional_embedding_theta` — [`L109`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L109)
  - `positional_embedding_type` — [`L108`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L108)
  - `proj_out` — [`L154`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L154)
  - `scale_shift_table` — [`L153`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L153)
  - `timestep_scale_multiplier` — [`L112`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L112)
  - `transformer_blocks` — [`L123`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L123)
  - `use_linear_projection` — [`L102`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L102)
  - `use_rope` — [`L111`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L111)
  - `use_tpu_flash_attention` — [`L101`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L101)
- protocol/private: `__init__`[`L68`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L68), `_set_gradient_checkpointing`[`L192`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L192), `_supports_gradient_checkpointing`[`L65`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L65)
- uses (calls/refs, reference-scoped): [`diffusers_and_ours_config_mapping`](../utils/diffusers_config_mapping.md#diffusers_and_ours_config_mapping), [`SkipLayerStrategy`](../utils/skip_layer_strategy.md#SkipLayerStrategy), [`make_hashable_key`](../utils/diffusers_config_mapping.md#make_hashable_key), [`TRANSFORMER_KEYS_RENAME_DICT`](../utils/diffusers_config_mapping.md#TRANSFORMER_KEYS_RENAME_DICT), [`Transformer3DModelOutput`](transformer3d.md#Transformer3DModelOutput), [`logger`](transformer3d.md#logger), [`sample`](transformer3d.md#Transformer3DModelOutput.sample)

### `Transformer3DModelOutput`
- def: [`src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py:51`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L51)
- doc: The output of \[`Transformer2DModel`\].
- signature: `class Transformer3DModelOutput(BaseOutput):`
- members:
  - `sample` — [`L61`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L61)
- used by: [`forward`](transformer3d.md#Transformer3DModel.forward)

## Module values
- `logger` — [`L47`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/transformers_pytorch/transformer3d.py#L47)

