---
title: 'Module: src/alphafold3/model/model_config.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/model_config.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.model_config`/
symbols:
  GlobalConfig: GlobalConfig#
  GlobalConfig.final_init: GlobalConfig#final_init.
  GlobalConfig.pair_transition_shard_spec: GlobalConfig#pair_transition_shard_spec.
  GlobalConfig.bfloat16: GlobalConfig#bfloat16.
  GlobalConfig.pair_attention_chunk_size: GlobalConfig#pair_attention_chunk_size.
  _Shape2DType._Shape2DType: _Shape2DType._Shape2DType.
  GlobalConfig.flash_attention_implementation: GlobalConfig#flash_attention_implementation.
---
# Module: [`src/alphafold3/model/model_config.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model_config.py)

## Classes
### `GlobalConfig`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/model_config.py:31`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model_config.py#L31) — documented in [alphafold3-model-model_config](../../../../concepts/alphafold3-model-model_config.md)
- doc: Global configuration for the AlphaFold3 model.
- signature: `class GlobalConfig(base_config.BaseConfig):`
- members:
  - `bfloat16` — [`L34`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model_config.py#L34) — documented in [alphafold3-model-model_config](../../../../concepts/alphafold3-model-model_config.md)
  - `final_init` — [`L35`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model_config.py#L35) — documented in [alphafold3-model-model_config](../../../../concepts/alphafold3-model-model_config.md)
  - `flash_attention_implementation` — [`L42`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model_config.py#L42)
  - `pair_attention_chunk_size` — [`L36`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model_config.py#L36) — documented in [alphafold3-model-network-modules](../../../../concepts/alphafold3-model-network-modules.md)
  - `pair_transition_shard_spec` — [`L37`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model_config.py#L37) — documented in [alphafold3-model-model_config](../../../../concepts/alphafold3-model-model_config.md)
- uses (calls/refs, reference-scoped): [`_Shape2DType`](model_config.md#_Shape2DType._Shape2DType)
- used by: [`atom_cross_att_encoder`](network/atom_cross_attention.md#atom_cross_att_encoder), [`atom_cross_att_decoder`](network/atom_cross_attention.md#atom_cross_att_decoder), [`__call__`](network/evoformer.md#Evoformer.__call__), [`__call__`](network/modules.md#PairFormerIteration.__call__), [`__call__`](network/confidence_head.md#ConfidenceHead.__call__), [`__call__`](network/diffusion_head.md#DiffusionHead.__call__), [`__call__`](network/modules.md#EvoformerIteration.__call__), [`__call__`](network/distogram_head.md#DistogramHead.__call__), [`create_target_feat_embedding`](model.md#create_target_feat_embedding), [`__call__`](network/modules.md#GridSelfAttention.__call__), [`__call__`](network/modules.md#OuterProductMean.__call__), [`self_attention`](network/diffusion_transformer.md#self_attention), [`cross_attention`](network/diffusion_transformer.md#cross_attention), [`__call__`](network/modules.md#TransitionBlock.__call__), [`__call__`](network/modules.md#TriangleMultiplication.__call__), [`transition_block`](network/diffusion_transformer.md#transition_block), [`_attention`](network/modules.md#GridSelfAttention._attention), [`__call__`](network/modules.md#MSAAttention.__call__), [`adaptive_zero_init`](network/diffusion_transformer.md#adaptive_zero_init), [`__init__`](network/template_modules.md#SingleTemplateEmbedding.__init__), [`__init__`](network/confidence_head.md#ConfidenceHead.__init__), [`__init__`](network/diffusion_head.md#DiffusionHead.__init__), [`__init__`](network/diffusion_transformer.md#CrossAttTransformer.__init__), [`__init__`](network/diffusion_transformer.md#Transformer.__init__), [`__init__`](network/distogram_head.md#DistogramHead.__init__), [`__init__`](network/evoformer.md#Evoformer.__init__), [`__init__`](network/modules.md#EvoformerIteration.__init__), [`__init__`](network/modules.md#GridSelfAttention.__init__), [`__init__`](network/modules.md#MSAAttention.__init__), [`__init__`](network/modules.md#OuterProductMean.__init__), [`__init__`](network/modules.md#PairFormerIteration.__init__), [`__init__`](network/modules.md#TransitionBlock.__init__), [`__init__`](network/modules.md#TriangleMultiplication.__init__), [`__init__`](network/template_modules.md#TemplateEmbedding.__init__), [`global_config`](model.md#Model.Config.global_config)

## Module values
- `_Shape2DType` — [`L28`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/model_config.py#L28) — documented in [alphafold3-model-model_config](../../../../concepts/alphafold3-model-model_config.md)

