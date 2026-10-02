---
title: 'Module: src/alphafold3/model/network/modules.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/network/modules.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.network.modules`/
symbols:
  PairFormerIteration.__call__: PairFormerIteration#__call__().
  EvoformerIteration.__call__: EvoformerIteration#__call__().
  GridSelfAttention.__call__: GridSelfAttention#__call__().
  OuterProductMean.__call__: OuterProductMean#__call__().
  TransitionBlock.__call__: TransitionBlock#__call__().
  TriangleMultiplication.__call__: TriangleMultiplication#__call__().
  GridSelfAttention._attention: GridSelfAttention#_attention().
  MSAAttention.__call__: MSAAttention#__call__().
  PairFormerIteration.config: PairFormerIteration#config.
  EvoformerIteration.config: EvoformerIteration#config.
  EvoformerIteration.global_config: EvoformerIteration#global_config.
  TransitionBlock: TransitionBlock#
  TriangleMultiplication: TriangleMultiplication#
  PairFormerIteration.global_config: PairFormerIteration#global_config.
  PairFormerIteration.Config.pair_attention: PairFormerIteration#Config#pair_attention.
  EvoformerIteration.Config.pair_attention: EvoformerIteration#Config#pair_attention.
  GridSelfAttention: GridSelfAttention#
  PairFormerIteration: PairFormerIteration#
  PairFormerIteration.Config.pair_transition: PairFormerIteration#Config#pair_transition.
  PairFormerIteration.Config.single_transition: PairFormerIteration#Config#single_transition.
  PairFormerIteration.Config.triangle_multiplication_incoming: PairFormerIteration#Config#triangle_multiplication_incoming.
  PairFormerIteration.Config.triangle_multiplication_outgoing: PairFormerIteration#Config#triangle_multiplication_outgoing.
  EvoformerIteration.Config.msa_attention: EvoformerIteration#Config#msa_attention.
  EvoformerIteration.Config.outer_product_mean: EvoformerIteration#Config#outer_product_mean.
  EvoformerIteration.Config.msa_transition: EvoformerIteration#Config#msa_transition.
  EvoformerIteration.Config.pair_transition: EvoformerIteration#Config#pair_transition.
  EvoformerIteration.Config.triangle_multiplication_incoming: EvoformerIteration#Config#triangle_multiplication_incoming.
  EvoformerIteration.Config.triangle_multiplication_outgoing: EvoformerIteration#Config#triangle_multiplication_outgoing.
  PairFormerIteration.Config.single_attention: PairFormerIteration#Config#single_attention.
  TransitionBlock.Config: TransitionBlock#Config#
  TransitionBlock.__init__: TransitionBlock#__init__().
  MSAAttention.__init__: MSAAttention#__init__().
  GridSelfAttention.Config.num_head: GridSelfAttention#Config#num_head.
  GridSelfAttention.__init__: GridSelfAttention#__init__().
  GridSelfAttention.config: GridSelfAttention#config.
  TriangleMultiplication.Config: TriangleMultiplication#Config#
  TriangleMultiplication.__init__: TriangleMultiplication#__init__().
  OuterProductMean.__init__: OuterProductMean#__init__().
  OuterProductMean.config: OuterProductMean#config.
  PairFormerIteration.__init__: PairFormerIteration#__init__().
  EvoformerIteration.__init__: EvoformerIteration#__init__().
  MSAAttention.Config.num_head: MSAAttention#Config#num_head.
  MSAAttention.config: MSAAttention#config.
  TriangleMultiplication.global_config: TriangleMultiplication#global_config.
  OuterProductMean.Config.num_outer_channel: OuterProductMean#Config#num_outer_channel.
  PairFormerIteration.Config: PairFormerIteration#Config#
  PairFormerIteration.Config.num_layer: PairFormerIteration#Config#num_layer.
  get_shard_size: get_shard_size().
  GridSelfAttention.Config: GridSelfAttention#Config#
  GridSelfAttention.global_config: GridSelfAttention#global_config.
  TransitionBlock.config: TransitionBlock#config.
  MSAAttention: MSAAttention#
  MSAAttention.Config: MSAAttention#Config#
  GridSelfAttention.transpose: GridSelfAttention#transpose.
  TriangleMultiplication.config: TriangleMultiplication#config.
  OuterProductMean: OuterProductMean#
  OuterProductMean.Config: OuterProductMean#Config#
  OuterProductMean.num_output_channel: OuterProductMean#num_output_channel.
  EvoformerIteration: EvoformerIteration#
  EvoformerIteration.Config: EvoformerIteration#Config#
  TransitionBlock.Config.num_intermediate_factor: TransitionBlock#Config#num_intermediate_factor.
  TransitionBlock.Config.use_glu_kernel: TransitionBlock#Config#use_glu_kernel.
  TransitionBlock.global_config: TransitionBlock#global_config.
  MSAAttention.global_config: MSAAttention#global_config.
  TriangleMultiplication.Config.equation: TriangleMultiplication#Config#equation.
  TriangleMultiplication.Config.use_glu_kernel: TriangleMultiplication#Config#use_glu_kernel.
  OuterProductMean.Config.chunk_size: OuterProductMean#Config#chunk_size.
  OuterProductMean.global_config: OuterProductMean#global_config.
  OuterProductMean.compute_chunk: OuterProductMean#compute_chunk().
  PairFormerIteration.Config.shard_transition_blocks: PairFormerIteration#Config#shard_transition_blocks.
  PairFormerIteration.with_single: PairFormerIteration#with_single.
  EvoformerIteration.Config.num_layer: EvoformerIteration#Config#num_layer.
  EvoformerIteration.Config.shard_transition_blocks: EvoformerIteration#Config#shard_transition_blocks.
---
# Module: [`src/alphafold3/model/network/modules.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py)

## Classes
### `Config`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/network/modules.py:545`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L545)
- doc: Configuration for EvoformerIteration.
- signature: `class Config(base_config.BaseConfig):`
- members:
  - `chunk_size` — [`L347`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L347)
  - `equation` — [`L258`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L258)
  - `msa_attention` — [`L549`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L549) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)
  - `msa_transition` — [`L551`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L551)
  - `num_head` — [`L100`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L100)
  - `num_head` — [`L144`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L144)
  - `num_intermediate_factor` — [`L54`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L54)
  - `num_layer` — [`L428`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L428) — documented in [alphafold3-model-network-template_modules](../../../../../concepts/alphafold3-model-network-template_modules.md)
  - `num_layer` — [`L548`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L548)
  - `num_outer_channel` — [`L348`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L348) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)
  - `outer_product_mean` — [`L550`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L550) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)
  - `pair_attention` — [`L429`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L429)
  - `pair_attention` — [`L552`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L552) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)
  - `pair_transition` — [`L430`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L430)
  - `pair_transition` — [`L553`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L553) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)
  - `shard_transition_blocks` — [`L439`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L439)
  - `shard_transition_blocks` — [`L560`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L560)
  - `single_attention` — [`L431`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L431) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)
  - `single_transition` — [`L432`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L432) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)
  - `triangle_multiplication_incoming` — [`L433`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L433)
  - `triangle_multiplication_incoming` — [`L554`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L554)
  - `triangle_multiplication_outgoing` — [`L436`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L436)
  - `triangle_multiplication_outgoing` — [`L557`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L557) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)
  - `use_glu_kernel` — [`L55`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L55)
  - `use_glu_kernel` — [`L259`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L259)
- uses (calls/refs, reference-scoped): [`TransitionBlock`](modules.md#TransitionBlock), [`TriangleMultiplication`](modules.md#TriangleMultiplication), [`GridSelfAttention`](modules.md#GridSelfAttention), [`Config`](modules.md#TransitionBlock.Config), [`Config`](modules.md#TriangleMultiplication.Config), [`Config`](modules.md#GridSelfAttention.Config), [`SelfAttentionConfig`](diffusion_transformer.md#SelfAttentionConfig), [`Config`](modules.md#MSAAttention.Config), [`Config`](modules.md#OuterProductMean.Config), [`MSAAttention`](modules.md#MSAAttention), [`OuterProductMean`](modules.md#OuterProductMean)
- used by: [`__call__`](evoformer.md#Evoformer.__call__), [`__call__`](modules.md#PairFormerIteration.__call__), [`__call__`](confidence_head.md#ConfidenceHead.__call__), [`__call__`](modules.md#EvoformerIteration.__call__), [`_embed_process_msa`](evoformer.md#Evoformer._embed_process_msa), [`__call__`](modules.md#GridSelfAttention.__call__), [`__call__`](modules.md#OuterProductMean.__call__), [`__call__`](template_modules.md#SingleTemplateEmbedding.__call__), [`__call__`](modules.md#TransitionBlock.__call__), [`__call__`](modules.md#TriangleMultiplication.__call__), [`_attention`](modules.md#GridSelfAttention._attention), [`__call__`](modules.md#MSAAttention.__call__), [`msa_stack`](evoformer.md#Evoformer.Config.msa_stack), [`__init__`](modules.md#EvoformerIteration.__init__)

### `EvoformerIteration`
- def: [`src/alphafold3/model/network/modules.py:542`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L542)
- doc: Single Iteration of Evoformer Main Stack.
- signature: `class EvoformerIteration(hk.Module):`
- members:
  - `config` — [`L569`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L569) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)
  - `global_config` — [`L570`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L570)
- protocol/private: `__call__`[`L572`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L572), `__init__`[`L562`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L562)
- uses (calls/refs, reference-scoped): [`GlobalConfig`](../model_config.md#GlobalConfig), [`sharded_apply`](../components/mapping.md#sharded_apply), [`TransitionBlock`](modules.md#TransitionBlock), [`TriangleMultiplication`](modules.md#TriangleMultiplication), [`pair_attention`](modules.md#EvoformerIteration.Config.pair_attention), [`GridSelfAttention`](modules.md#GridSelfAttention), [`msa_attention`](modules.md#EvoformerIteration.Config.msa_attention), [`msa_transition`](modules.md#EvoformerIteration.Config.msa_transition), [`outer_product_mean`](modules.md#EvoformerIteration.Config.outer_product_mean), [`pair_transition`](modules.md#EvoformerIteration.Config.pair_transition), [`triangle_multiplication_incoming`](modules.md#EvoformerIteration.Config.triangle_multiplication_incoming), [`triangle_multiplication_outgoing`](modules.md#EvoformerIteration.Config.triangle_multiplication_outgoing), [`pair_transition_shard_spec`](../model_config.md#GlobalConfig.pair_transition_shard_spec), [`get_shard_size`](modules.md#get_shard_size), [`Config`](modules.md#EvoformerIteration.Config), [`MSAAttention`](modules.md#MSAAttention), [`OuterProductMean`](modules.md#OuterProductMean), [`shard_transition_blocks`](modules.md#EvoformerIteration.Config.shard_transition_blocks)
- used by: [`evoformer_fn`](evoformer.md#Evoformer.evoformer_fn), [`msa_stack`](evoformer.md#Evoformer.Config.msa_stack)

### `GridSelfAttention`
- def: [`src/alphafold3/model/network/modules.py:140`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L140) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)
- doc: Self attention that is either per-sequence or per-residue.
- signature: `class GridSelfAttention(hk.Module):`
- members:
  - `__call__(self, act, pair_mask)` — [`L208`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L208) — Builds a module. — documented in [alphafold3-model-components-mapping](../../../../../concepts/alphafold3-model-components-mapping.md)
  - `config` — [`L155`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L155)
  - `global_config` — [`L156`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L156)
  - `transpose` — [`L157`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L157) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)
- protocol/private: `__init__`[`L146`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L146), `_attention`[`L160`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L160)
- uses (calls/refs, reference-scoped): [`Linear`](../components/haiku_modules.md#Linear), [`LayerNorm`](../components/haiku_modules.md#LayerNorm), [`GlobalConfig`](../model_config.md#GlobalConfig), [`final_init`](../model_config.md#GlobalConfig.final_init), [`inference_subbatch`](../components/mapping.md#inference_subbatch), [`num_head`](modules.md#GridSelfAttention.Config.num_head), [`pair_attention_chunk_size`](../model_config.md#GlobalConfig.pair_attention_chunk_size), [`Config`](modules.md#GridSelfAttention.Config), [`get_shard_size`](modules.md#get_shard_size), [`flash_attention_implementation`](../model_config.md#GlobalConfig.flash_attention_implementation)
- used by: [`__call__`](modules.md#PairFormerIteration.__call__), [`__call__`](modules.md#EvoformerIteration.__call__), [`pair_attention`](modules.md#EvoformerIteration.Config.pair_attention), [`pair_attention`](modules.md#PairFormerIteration.Config.pair_attention)

### `MSAAttention`
- def: [`src/alphafold3/model/network/modules.py:96`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L96) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)
- doc: MSA Attention.
- signature: `class MSAAttention(hk.Module):`
- members:
  - `config` — [`L106`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L106)
  - `global_config` — [`L107`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L107)
- protocol/private: `__call__`[`L109`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L109), `__init__`[`L102`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L102)
- uses (calls/refs, reference-scoped): [`Linear`](../components/haiku_modules.md#Linear), [`LayerNorm`](../components/haiku_modules.md#LayerNorm), [`GlobalConfig`](../model_config.md#GlobalConfig), [`final_init`](../model_config.md#GlobalConfig.final_init), [`num_head`](modules.md#MSAAttention.Config.num_head), [`Config`](modules.md#MSAAttention.Config)
- used by: [`__call__`](modules.md#EvoformerIteration.__call__), [`msa_attention`](modules.md#EvoformerIteration.Config.msa_attention)

### `OuterProductMean`
- def: [`src/alphafold3/model/network/modules.py:343`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L343) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)
- doc: Computed mean outer product.
- signature: `class OuterProductMean(hk.Module):`
- members:
  - `compute_chunk(left_act)` — [`L401`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L401)
  - `config` — [`L360`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L360)
  - `global_config` — [`L359`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L359)
  - `num_output_channel` — [`L361`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L361)
- protocol/private: `__call__`[`L363`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L363), `__init__`[`L350`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L350)
- uses (calls/refs, reference-scoped): [`Linear`](../components/haiku_modules.md#Linear), [`LayerNorm`](../components/haiku_modules.md#LayerNorm), [`GlobalConfig`](../model_config.md#GlobalConfig), [`final_init`](../model_config.md#GlobalConfig.final_init), [`inference_subbatch`](../components/mapping.md#inference_subbatch), [`num_outer_channel`](modules.md#OuterProductMean.Config.num_outer_channel), [`Config`](modules.md#OuterProductMean.Config), [`chunk_size`](modules.md#OuterProductMean.Config.chunk_size)
- used by: [`__call__`](modules.md#EvoformerIteration.__call__), [`outer_product_mean`](modules.md#EvoformerIteration.Config.outer_product_mean)

### `PairFormerIteration`
- def: [`src/alphafold3/model/network/modules.py:422`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L422) — documented in [alphafold3-model-network-evoformer](../../../../../concepts/alphafold3-model-network-evoformer.md)
- doc: Single Iteration of Pair Former.
- signature: `class PairFormerIteration(hk.Module):`
- members:
  - `__call__(self, act, pair_mask, single_act=None, seq_mask=None)` — [`L454`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L454) — Build a single iteration of the pair former. — documented in [alphafold3-model-components-mapping](../../../../../concepts/alphafold3-model-components-mapping.md)
  - `config` — [`L450`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L450) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)
  - `global_config` — [`L451`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L451)
  - `with_single` — [`L452`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L452)
- protocol/private: `__init__`[`L441`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L441)
- uses (calls/refs, reference-scoped): [`Linear`](../components/haiku_modules.md#Linear), [`LayerNorm`](../components/haiku_modules.md#LayerNorm), [`GlobalConfig`](../model_config.md#GlobalConfig), [`self_attention`](diffusion_transformer.md#self_attention), [`sharded_apply`](../components/mapping.md#sharded_apply), [`TransitionBlock`](modules.md#TransitionBlock), [`TriangleMultiplication`](modules.md#TriangleMultiplication), [`pair_attention`](modules.md#PairFormerIteration.Config.pair_attention), [`GridSelfAttention`](modules.md#GridSelfAttention), [`pair_transition`](modules.md#PairFormerIteration.Config.pair_transition), [`single_transition`](modules.md#PairFormerIteration.Config.single_transition), [`triangle_multiplication_incoming`](modules.md#PairFormerIteration.Config.triangle_multiplication_incoming), [`triangle_multiplication_outgoing`](modules.md#PairFormerIteration.Config.triangle_multiplication_outgoing), [`single_attention`](modules.md#PairFormerIteration.Config.single_attention), [`pair_transition_shard_spec`](../model_config.md#GlobalConfig.pair_transition_shard_spec), [`Config`](modules.md#PairFormerIteration.Config), [`get_shard_size`](modules.md#get_shard_size), [`num_head`](diffusion_transformer.md#SelfAttentionConfig.num_head), [`shard_transition_blocks`](modules.md#PairFormerIteration.Config.shard_transition_blocks)
- used by: [`pairformer_fn`](evoformer.md#Evoformer.pairformer_fn), [`pairformer_fn`](confidence_head.md#ConfidenceHead.pairformer_fn), [`template_stack`](template_modules.md#TemplateEmbedding.Config.template_stack), [`pairformer`](confidence_head.md#ConfidenceHead.Config.pairformer), [`PairformerConfig`](evoformer.md#Evoformer.PairformerConfig), [`template_iteration_fn`](template_modules.md#SingleTemplateEmbedding.template_iteration_fn)

### `TransitionBlock`
- def: [`src/alphafold3/model/network/modules.py:50`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L50) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)
- doc: Transition block for transformer.
- signature: `class TransitionBlock(hk.Module):`
- members:
  - `config` — [`L61`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L61)
  - `global_config` — [`L62`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L62)
- protocol/private: `__call__`[`L64`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L64), `__init__`[`L57`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L57)
- uses (calls/refs, reference-scoped): [`Linear`](../components/haiku_modules.md#Linear), [`LayerNorm`](../components/haiku_modules.md#LayerNorm), [`GlobalConfig`](../model_config.md#GlobalConfig), [`final_init`](../model_config.md#GlobalConfig.final_init), [`haiku_linear_get_params`](../components/haiku_modules.md#haiku_linear_get_params), [`Config`](modules.md#TransitionBlock.Config), [`num_intermediate_factor`](modules.md#TransitionBlock.Config.num_intermediate_factor), [`use_glu_kernel`](modules.md#TransitionBlock.Config.use_glu_kernel)
- used by: [`__call__`](modules.md#PairFormerIteration.__call__), [`__call__`](modules.md#EvoformerIteration.__call__), [`msa_transition`](modules.md#EvoformerIteration.Config.msa_transition), [`pair_transition`](modules.md#EvoformerIteration.Config.pair_transition), [`pair_transition`](modules.md#PairFormerIteration.Config.pair_transition), [`single_transition`](modules.md#PairFormerIteration.Config.single_transition)

### `TriangleMultiplication`
- def: [`src/alphafold3/model/network/modules.py:254`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L254) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)
- doc: Triangle Multiplication.
- signature: `class TriangleMultiplication(hk.Module):`
- members:
  - `__call__(self, act, mask)` — [`L268`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L268) — Applies Module.
  - `config` — [`L265`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L265)
  - `global_config` — [`L266`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L266)
- protocol/private: `__init__`[`L261`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L261)
- uses (calls/refs, reference-scoped): [`Linear`](../components/haiku_modules.md#Linear), [`LayerNorm`](../components/haiku_modules.md#LayerNorm), [`GlobalConfig`](../model_config.md#GlobalConfig), [`final_init`](../model_config.md#GlobalConfig.final_init), [`haiku_linear_get_params`](../components/haiku_modules.md#haiku_linear_get_params), [`Config`](modules.md#TriangleMultiplication.Config), [`equation`](modules.md#TriangleMultiplication.Config.equation), [`use_glu_kernel`](modules.md#TriangleMultiplication.Config.use_glu_kernel)
- used by: [`__call__`](modules.md#PairFormerIteration.__call__), [`__call__`](modules.md#EvoformerIteration.__call__), [`triangle_multiplication_incoming`](modules.md#EvoformerIteration.Config.triangle_multiplication_incoming), [`triangle_multiplication_incoming`](modules.md#PairFormerIteration.Config.triangle_multiplication_incoming), [`triangle_multiplication_outgoing`](modules.md#EvoformerIteration.Config.triangle_multiplication_outgoing), [`triangle_multiplication_outgoing`](modules.md#PairFormerIteration.Config.triangle_multiplication_outgoing)

## Functions
- `get_shard_size(num_residues: int, shard_spec: Sequence[tuple[int | None, int | None]])` — [`L36`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/modules.py#L36) — documented in [alphafold3-model-network-modules](../../../../../concepts/alphafold3-model-network-modules.md)

