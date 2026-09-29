---
title: 'Module: tpu_inference/models/jax/qwen2_5_vl.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/qwen2_5_vl.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.qwen2_5_vl`/
symbols:
  Qwen2_5_VisionAttention.__call__: Qwen2_5_VisionAttention#__call__().
  Qwen2_5_VLForConditionalGeneration.visual: Qwen2_5_VLForConditionalGeneration#visual.
  Qwen2_5_VLForConditionalGeneration.embed_input_ids: Qwen2_5_VLForConditionalGeneration#embed_input_ids().
  Qwen2_5_VLForConditionalGeneration.precompile_vision_encoder: Qwen2_5_VLForConditionalGeneration#precompile_vision_encoder().
  init_fn: init_fn.
  Qwen2_5_VLForConditionalGeneration.pp_missing_layers: Qwen2_5_VLForConditionalGeneration#pp_missing_layers.
  Qwen2_5_VisionTransformer.spatial_merge_size: Qwen2_5_VisionTransformer#spatial_merge_size.
  Qwen2_5_VisionTransformer.compute_hidden_states: Qwen2_5_VisionTransformer#compute_hidden_states().
  Qwen2_5_VLForConditionalGeneration._parse_and_validate_image_input: Qwen2_5_VLForConditionalGeneration#_parse_and_validate_image_input().
  Qwen2_5_VLForConditionalGeneration._process_image_input: Qwen2_5_VLForConditionalGeneration#_process_image_input().
  Qwen2_5_VisionMLP.__init__: Qwen2_5_VisionMLP#__init__().
  Qwen2_5_VisionPatchMerger.__call__: Qwen2_5_VisionPatchMerger#__call__().
  Qwen2_5_VisionTransformer.__call__: Qwen2_5_VisionTransformer#__call__().
  Qwen2_5_VLForConditionalGeneration.get_mrope_input_positions: Qwen2_5_VLForConditionalGeneration#get_mrope_input_positions().
  Qwen2_5_VLForConditionalGeneration.lm_head: Qwen2_5_VLForConditionalGeneration#lm_head.
  Qwen2_5_VisionTransformer.get_window_index_thw: Qwen2_5_VisionTransformer#get_window_index_thw().
  Qwen2_5_VisionTransformer.pad_inputs: Qwen2_5_VisionTransformer#pad_inputs().
  Qwen2_5_VLForConditionalGeneration.model: Qwen2_5_VLForConditionalGeneration#model.
  Qwen2_5_VisionMLP.__call__: Qwen2_5_VisionMLP#__call__().
  Qwen2_5_VisionAttention.head_dim: Qwen2_5_VisionAttention#head_dim.
  Qwen2_5_VisionBlock.__call__: Qwen2_5_VisionBlock#__call__().
  Qwen2_5_VisionPatchEmbed.__call__: Qwen2_5_VisionPatchEmbed#__call__().
  Qwen2_5_VLForConditionalGeneration.embed_multimodal: Qwen2_5_VLForConditionalGeneration#embed_multimodal().
  Qwen2_5_VLForConditionalGeneration.__call__: Qwen2_5_VLForConditionalGeneration#__call__().
  Qwen2_5_VLForConditionalGeneration.compute_logits: Qwen2_5_VLForConditionalGeneration#compute_logits().
  Qwen2_5_VisionAttention.head_dim_original: Qwen2_5_VisionAttention#head_dim_original.
  Qwen2_5_VLForConditionalGeneration.vllm_config: Qwen2_5_VLForConditionalGeneration#vllm_config.
  Qwen2_5_VLForConditionalGeneration.is_first_rank: Qwen2_5_VLForConditionalGeneration#is_first_rank.
  generate_window_segment_ids: generate_window_segment_ids().
  Qwen2_5_VisionTransformer.spatial_merge_unit: Qwen2_5_VisionTransformer#spatial_merge_unit.
  Qwen2_5_VisionTransformer.rotary_pos_emb_thw: Qwen2_5_VisionTransformer#rotary_pos_emb_thw().
  Qwen2_5_VLForConditionalGeneration.is_last_rank: Qwen2_5_VLForConditionalGeneration#is_last_rank.
  Qwen2_5_VLImageInputs: Qwen2_5_VLImageInputs.
  Qwen2_5_VisionAttention.num_heads: Qwen2_5_VisionAttention#num_heads.
  Qwen2_5_VisionTransformer.compute_aux_arrays: Qwen2_5_VisionTransformer#compute_aux_arrays().
  Qwen2_5_VisionTransformer.encode_jit: Qwen2_5_VisionTransformer#encode_jit().
  Qwen2_5_VisionAttention.qkv_proj: Qwen2_5_VisionAttention#qkv_proj.
  Qwen2_5_VisionAttention.proj: Qwen2_5_VisionAttention#proj.
  Qwen2_5_VisionAttention.flash_attention: Qwen2_5_VisionAttention#flash_attention.
  Qwen2_5_VisionPatchMerger.mlp_fc1: Qwen2_5_VisionPatchMerger#mlp_fc1.
  Qwen2_5_VisionPatchMerger.mlp_act: Qwen2_5_VisionPatchMerger#mlp_act.
  Qwen2_5_VisionPatchMerger.mlp_fc2: Qwen2_5_VisionPatchMerger#mlp_fc2.
  Qwen2_5_VisionTransformer.patch_embed: Qwen2_5_VisionTransformer#patch_embed.
  Qwen2_5_VisionTransformer.get_rope_by_thw: Qwen2_5_VisionTransformer#get_rope_by_thw().
  Qwen2_5_VisionAttention.hidden_size: Qwen2_5_VisionAttention#hidden_size.
  Qwen2_5_VisionBlock.__init__: Qwen2_5_VisionBlock#__init__().
  Qwen2_5_VisionRotaryEmbedding.__call__: Qwen2_5_VisionRotaryEmbedding#__call__().
  modeling_flax_utils: modeling_flax_utils.
  Qwen2_5_VisionAttention.num_kv_heads: Qwen2_5_VisionAttention#num_kv_heads.
  Qwen2_5_VisionTransformer.encode_padded_jit: Qwen2_5_VisionTransformer#encode_padded_jit().
  Qwen2_5_VisionPatchEmbed.patch_size: Qwen2_5_VisionPatchEmbed#patch_size.
  Qwen2_5_VisionPatchMerger.hidden_size: Qwen2_5_VisionPatchMerger#hidden_size.
  Qwen2_5_VLForConditionalGeneration.rng: Qwen2_5_VLForConditionalGeneration#rng.
  logger: logger.
  Qwen2_5_VisionMLP.gate_proj: Qwen2_5_VisionMLP#gate_proj.
  Qwen2_5_VisionMLP.up_proj: Qwen2_5_VisionMLP#up_proj.
  Qwen2_5_VisionMLP.down_proj: Qwen2_5_VisionMLP#down_proj.
  Qwen2_5_VisionBlock.attn: Qwen2_5_VisionBlock#attn.
  Qwen2_5_VisionBlock.mlp: Qwen2_5_VisionBlock#mlp.
  Qwen2_5_VisionPatchEmbed.proj: Qwen2_5_VisionPatchEmbed#proj.
  Qwen2_5_VisionPatchMerger.ln_q: Qwen2_5_VisionPatchMerger#ln_q.
  Qwen2_5_VisionTransformer.rotary_pos_emb: Qwen2_5_VisionTransformer#rotary_pos_emb.
  Qwen2_5_VisionTransformer.blocks: Qwen2_5_VisionTransformer#blocks.
  Qwen2_5_VisionTransformer.merger: Qwen2_5_VisionTransformer#merger.
  Qwen2_5_VLForConditionalGeneration.WeightLoader: Qwen2_5_VLForConditionalGeneration#WeightLoader.
  Qwen2_5_VLForConditionalGeneration._parse_and_validate_multimodal_inputs: Qwen2_5_VLForConditionalGeneration#_parse_and_validate_multimodal_inputs().
  Qwen2_5_VLForConditionalGeneration.get_single_image_embedding: Qwen2_5_VLForConditionalGeneration#get_single_image_embedding().
  Qwen2_5_VLForConditionalGeneration.config: Qwen2_5_VLForConditionalGeneration#config.
  Qwen2_5_VisionAttention.__init__: Qwen2_5_VisionAttention#__init__().
  Qwen2_5_VisionAttention.rope_theta: Qwen2_5_VisionAttention#rope_theta.
  Qwen2_5_VisionAttention.rope_scaling: Qwen2_5_VisionAttention#rope_scaling.
  Qwen2_5_VLForConditionalGeneration.load_weights: Qwen2_5_VLForConditionalGeneration#load_weights().
  SegmentIds: SegmentIds#
  Qwen2_5_VLImagePixelInputs: Qwen2_5_VLImagePixelInputs#
  apply_rotary_pos_emb_vision: apply_rotary_pos_emb_vision().
  Qwen2_5_VisionPatchEmbed.temporal_patch_size: Qwen2_5_VisionPatchEmbed#temporal_patch_size.
  Qwen2_5_VisionRotaryEmbedding.dim: Qwen2_5_VisionRotaryEmbedding#dim.
  Qwen2_5_VisionTransformer.window_size: Qwen2_5_VisionTransformer#window_size.
  Qwen2_5_VisionTransformer.patch_size: Qwen2_5_VisionTransformer#patch_size.
  Qwen2_5_VisionTransformer.enable_dynamic_image_sizes: Qwen2_5_VisionTransformer#enable_dynamic_image_sizes.
  Qwen2_5_VLForConditionalGeneration: Qwen2_5_VLForConditionalGeneration#
  DEFAULT_BLOCK_K_MAJOR: DEFAULT_BLOCK_K_MAJOR.
  SegmentIds.q: SegmentIds#q.
  SegmentIds.kv: SegmentIds#kv.
  Qwen2_5_VLImagePixelInputs.pixel_values: Qwen2_5_VLImagePixelInputs#pixel_values.
  Qwen2_5_VLImagePixelInputs.image_grid_thw: Qwen2_5_VLImagePixelInputs#image_grid_thw.
  Qwen2_5_VLImageEmbeddingInputs: Qwen2_5_VLImageEmbeddingInputs#
  Qwen2_5_VisionMLP: Qwen2_5_VisionMLP#
  Qwen2_5_VisionMLP.act_fn: Qwen2_5_VisionMLP#act_fn.
  Qwen2_5_VisionAttention: Qwen2_5_VisionAttention#
  Qwen2_5_VisionBlock: Qwen2_5_VisionBlock#
  Qwen2_5_VisionBlock.norm1: Qwen2_5_VisionBlock#norm1.
  Qwen2_5_VisionBlock.norm2: Qwen2_5_VisionBlock#norm2.
  Qwen2_5_VisionPatchEmbed: Qwen2_5_VisionPatchEmbed#
  Qwen2_5_VisionPatchEmbed.hidden_size: Qwen2_5_VisionPatchEmbed#hidden_size.
  Qwen2_5_VisionPatchMerger: Qwen2_5_VisionPatchMerger#
  Qwen2_5_VisionRotaryEmbedding: Qwen2_5_VisionRotaryEmbedding#
  Qwen2_5_VisionRotaryEmbedding.theta: Qwen2_5_VisionRotaryEmbedding#theta.
  Qwen2_5_VisionTransformer: Qwen2_5_VisionTransformer#
  Qwen2_5_VisionTransformer.config: Qwen2_5_VisionTransformer#config.
  Qwen2_5_VisionTransformer.dtype: Qwen2_5_VisionTransformer#dtype.
  Qwen2_5_VisionTransformer.hidden_size: Qwen2_5_VisionTransformer#hidden_size.
  Qwen2_5_VisionTransformer.fullatt_block_indexes: Qwen2_5_VisionTransformer#fullatt_block_indexes.
  Qwen2_5_VLForConditionalGeneration.mesh: Qwen2_5_VLForConditionalGeneration#mesh.
  Qwen2_5_VLForConditionalGeneration._validate_and_reshape_mm_tensor: Qwen2_5_VLForConditionalGeneration#_validate_and_reshape_mm_tensor().
  Qwen2_5_VLImageEmbeddingInputs.image_embeds: Qwen2_5_VLImageEmbeddingInputs#image_embeds.
  Qwen2_5_VLImageEmbeddingInputs.image_grid_thw: Qwen2_5_VLImageEmbeddingInputs#image_grid_thw.
  Qwen2_5_VisionAttention.mesh: Qwen2_5_VisionAttention#mesh.
  Qwen2_5_VisionPatchEmbed.__init__: Qwen2_5_VisionPatchEmbed#__init__().
  Qwen2_5_VisionPatchMerger.__init__: Qwen2_5_VisionPatchMerger#__init__().
  Qwen2_5_VisionRotaryEmbedding.__init__: Qwen2_5_VisionRotaryEmbedding#__init__().
  Qwen2_5_VisionTransformer.__init__: Qwen2_5_VisionTransformer#__init__().
  Qwen2_5_VisionTransformer.num_heads: Qwen2_5_VisionTransformer#num_heads.
  Qwen2_5_VisionTransformer.compute_attn_mask_seqlen: Qwen2_5_VisionTransformer#compute_attn_mask_seqlen().
  Qwen2_5_VLForConditionalGeneration.__init__: Qwen2_5_VLForConditionalGeneration#__init__().
  Qwen2_5_VLForConditionalGeneration.multimodal_config: Qwen2_5_VLForConditionalGeneration#multimodal_config.
---
# Module: [`tpu_inference/models/jax/qwen2_5_vl.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py)

## Classes
### `Qwen2_5_VLForConditionalGeneration`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/qwen2_5_vl.py:762`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L762)
- signature: `class Qwen2_5_VLForConditionalGeneration(nnx.Module):`
- members:
  - `compute_logits(self, hidden_states: jax.Array)` — [`L1149`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L1149)
  - `embed_input_ids(self, input_ids: jax.Array, multimodal_embeddings: jax.Array | None, *, is_multimodal: jax.Array | None = None)` — [`L1092`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L1092)
  - `embed_multimodal(self, **kwargs: object)` — [`L1065`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L1065)
  - `get_mrope_input_positions(self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec])` — [`L808`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L808) — Get mrope input positions and delta value.
  - `get_single_image_embedding(self, image_pixel_values, image_grid_thw)` — [`L1026`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L1026)
  - `load_weights(self, rng_key: jax.Array)` — [`L1155`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L1155)
  - `precompile_vision_encoder(self, run_compilation_fn: Callable)` — [`L1203`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L1203)
  - `WeightLoader` — [`L763`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L763)
  - `config` — [`L775`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L775)
  - `is_first_rank` — [`L773`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L773)
  - `is_last_rank` — [`L778`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L778)
  - `lm_head` — [`L799`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L799)
  - `mesh` — [`L772`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L772)
  - `model` — [`L791`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L791)
  - `multimodal_config` — [`L776`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L776)
  - `pp_missing_layers` — [`L1160`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L1160)
  - `rng` — [`L771`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L771)
  - `visual` — [`L781`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L781)
  - `vllm_config` — [`L770`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L770)
- protocol/private: `__call__`[`L1116`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L1116), `__init__`[`L765`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L765), `_parse_and_validate_image_input`[`L957`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L957), `_parse_and_validate_multimodal_inputs`[`L991`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L991), `_process_image_input`[`L1029`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L1029), `_validate_and_reshape_mm_tensor`[`L937`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L937)
- uses (calls/refs, reference-scoped): [`AttentionMetadata`](../../layers/common/attention_metadata.md#AttentionMetadata), [`JaxIntermediateTensors`](jax_intermediate_tensor.md#JaxIntermediateTensors), [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`get_pp_group`](../../distributed/jax_parallel_state.md#get_pp_group), [`JaxLmHead`](../../layers/jax/linear.md#JaxLmHead), [`tensors`](jax_intermediate_tensor.md#JaxIntermediateTensors.tensors), [`embed_tokens`](qwen2.md#Qwen2Model.embed_tokens), [`StandardWeightLoader`](utils/weight_utils.md#StandardWeightLoader), [`is_first_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_first_rank), [`load_weights`](utils/weight_utils.md#StandardWeightLoader.load_weights), [`is_last_rank`](../../distributed/jax_parallel_state.md#GroupCoordinator.is_last_rank), [`Qwen2Model`](qwen2.md#Qwen2Model), [`decode`](../../layers/jax/embed.md#JaxEmbed.decode), [`merge_multimodal_embeddings`](utils/multi_modal_utils.md#merge_multimodal_embeddings), [`Qwen2_5_VLImageInputs`](qwen2_5_vl.md#Qwen2_5_VLImageInputs), [`encode_jit`](qwen2_5_vl.md#Qwen2_5_VisionTransformer.encode_jit), [`encode_padded_jit`](qwen2_5_vl.md#Qwen2_5_VisionTransformer.encode_padded_jit), `vision_config`, [`logger`](qwen2_5_vl.md#logger), [`MultiModalEmbeddings`](utils/multi_modal_utils.md#MultiModalEmbeddings), [`Qwen2_5_VLImagePixelInputs`](qwen2_5_vl.md#Qwen2_5_VLImagePixelInputs), [`enable_dynamic_image_sizes`](qwen2_5_vl.md#Qwen2_5_VisionTransformer.enable_dynamic_image_sizes), `image_token_id`, `video_token_id`, [`Qwen2_5_VisionTransformer`](qwen2_5_vl.md#Qwen2_5_VisionTransformer), [`config`](qwen2_5_vl.md#Qwen2_5_VisionTransformer.config), [`dtype`](qwen2_5_vl.md#Qwen2_5_VisionTransformer.dtype), [`image_grid_thw`](qwen2_5_vl.md#Qwen2_5_VLImagePixelInputs.image_grid_thw), [`pixel_values`](qwen2_5_vl.md#Qwen2_5_VLImagePixelInputs.pixel_values), `text_config`, `vision_start_token_id`
- used by: [`_get_model_architecture`](../common/model_loader.md#_get_model_architecture)

### `Qwen2_5_VLImageEmbeddingInputs`  ·  implements/extends _TypedDict
- def: [`tpu_inference/models/jax/qwen2_5_vl.py:92`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L92)
- signature: `class Qwen2_5_VLImageEmbeddingInputs(TypedDict):`
- members:
  - `image_embeds` — [`L94`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L94) — ---
  - `image_grid_thw` — [`L107`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L107) — ---
- used by: [`Qwen2_5_VLImageInputs`](qwen2_5_vl.md#Qwen2_5_VLImageInputs)

### `Qwen2_5_VLImagePixelInputs`  ·  implements/extends _TypedDict
- def: [`tpu_inference/models/jax/qwen2_5_vl.py:76`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L76)
- signature: `class Qwen2_5_VLImagePixelInputs(TypedDict):`
- members:
  - `image_grid_thw` — [`L83`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L83) — ---
  - `pixel_values` — [`L78`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L78) — ---
- used by: [`_parse_and_validate_image_input`](qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration._parse_and_validate_image_input), [`Qwen2_5_VLImageInputs`](qwen2_5_vl.md#Qwen2_5_VLImageInputs)

### `Qwen2_5_VisionAttention`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/qwen2_5_vl.py:210`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L210)
- signature: `class Qwen2_5_VisionAttention(nnx.Module):`
- members:
  - `flash_attention` — [`L254`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L254)
  - `head_dim` — [`L228`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L228)
  - `head_dim_original` — [`L221`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L221)
  - `hidden_size` — [`L215`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L215)
  - `mesh` — [`L233`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L233)
  - `num_heads` — [`L216`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L216)
  - `num_kv_heads` — [`L217`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L217)
  - `proj` — [`L245`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L245)
  - `qkv_proj` — [`L235`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L235)
  - `rope_scaling` — [`L220`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L220)
  - `rope_theta` — [`L219`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L219)
- protocol/private: `__call__`[`L261`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L261), `__init__`[`L212`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L212)
- uses (calls/refs, reference-scoped): [`init_fn`](qwen2_5_vl.md#init_fn), [`sharded_flash_attention`](../../layers/common/attention_interface.md#sharded_flash_attention), [`get_rope_scaling`](../../layers/jax/rope_interface.md#get_rope_scaling), [`generate_window_segment_ids`](qwen2_5_vl.md#generate_window_segment_ids), [`get_rope_theta`](../../layers/jax/rope_interface.md#get_rope_theta), `vision_config`, [`apply_rotary_pos_emb_vision`](qwen2_5_vl.md#apply_rotary_pos_emb_vision), [`DEFAULT_BLOCK_K_MAJOR`](qwen2_5_vl.md#DEFAULT_BLOCK_K_MAJOR)
- used by: [`attn`](qwen2_5_vl.md#Qwen2_5_VisionBlock.attn)

### `Qwen2_5_VisionBlock`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/qwen2_5_vl.py:331`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L331)
- signature: `class Qwen2_5_VisionBlock(nnx.Module):`
- members:
  - `attn` — [`L345`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L345)
  - `mlp` — [`L349`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L349)
  - `norm1` — [`L343`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L343)
  - `norm2` — [`L344`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L344)
- protocol/private: `__call__`[`L353`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L353), `__init__`[`L333`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L333)
- uses (calls/refs, reference-scoped): [`init_fn`](qwen2_5_vl.md#init_fn), `vision_config`, [`Qwen2_5_VisionAttention`](qwen2_5_vl.md#Qwen2_5_VisionAttention), [`Qwen2_5_VisionMLP`](qwen2_5_vl.md#Qwen2_5_VisionMLP)
- used by: [`blocks`](qwen2_5_vl.md#Qwen2_5_VisionTransformer.blocks)

### `Qwen2_5_VisionMLP`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/qwen2_5_vl.py:117`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L117)
- signature: `class Qwen2_5_VisionMLP(nnx.Module):`
- members:
  - `act_fn` — [`L151`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L151)
  - `down_proj` — [`L142`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L142)
  - `gate_proj` — [`L124`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L124)
  - `up_proj` — [`L133`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L133)
- protocol/private: `__call__`[`L153`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L153), `__init__`[`L119`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L119)
- uses (calls/refs, reference-scoped): [`init_fn`](qwen2_5_vl.md#init_fn), [`ACT2FN`](../../layers/jax/layers.md#FlaxUtils.ACT2FN), [`modeling_flax_utils`](qwen2_5_vl.md#modeling_flax_utils), `hidden_act`, `hidden_size`, `intermediate_size`
- used by: [`mlp`](qwen2_5_vl.md#Qwen2_5_VisionBlock.mlp)

### `Qwen2_5_VisionPatchEmbed`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/qwen2_5_vl.py:366`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L366)
- signature: `class Qwen2_5_VisionPatchEmbed(nnx.Module):`
- members:
  - `hidden_size` — [`L379`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L379)
  - `patch_size` — [`L377`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L377)
  - `proj` — [`L381`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L381)
  - `temporal_patch_size` — [`L378`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L378)
- protocol/private: `__call__`[`L391`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L391), `__init__`[`L368`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L368)
- uses (calls/refs, reference-scoped): [`init_fn`](qwen2_5_vl.md#init_fn)
- used by: [`patch_embed`](qwen2_5_vl.md#Qwen2_5_VisionTransformer.patch_embed)

### `Qwen2_5_VisionPatchMerger`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/qwen2_5_vl.py:409`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L409)
- signature: `class Qwen2_5_VisionPatchMerger(nnx.Module):`
- members:
  - `hidden_size` — [`L413`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L413)
  - `ln_q` — [`L414`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L414)
  - `mlp_act` — [`L428`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L428)
  - `mlp_fc1` — [`L420`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L420)
  - `mlp_fc2` — [`L429`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L429)
- protocol/private: `__call__`[`L438`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L438), `__init__`[`L411`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L411)
- uses (calls/refs, reference-scoped): [`init_fn`](qwen2_5_vl.md#init_fn), [`ACT2FN`](../../layers/jax/layers.md#FlaxUtils.ACT2FN), [`modeling_flax_utils`](qwen2_5_vl.md#modeling_flax_utils)
- used by: [`merger`](qwen2_5_vl.md#Qwen2_5_VisionTransformer.merger)

### `Qwen2_5_VisionRotaryEmbedding`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/qwen2_5_vl.py:447`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L447)
- signature: `class Qwen2_5_VisionRotaryEmbedding(nnx.Module):`
- members:
  - `dim` — [`L450`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L450)
  - `theta` — [`L451`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L451)
- protocol/private: `__call__`[`L453`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L453), `__init__`[`L449`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L449)
- used by: [`rotary_pos_emb`](qwen2_5_vl.md#Qwen2_5_VisionTransformer.rotary_pos_emb)

### `Qwen2_5_VisionTransformer`  ·  implements/extends Module
- def: [`tpu_inference/models/jax/qwen2_5_vl.py:461`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L461)
- signature: `class Qwen2_5_VisionTransformer(nnx.Module):`
- members:
  - `compute_attn_mask_seqlen(self, cu_seqlens: jax.Array)` — [`L606`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L606)
  - `compute_aux_arrays(self, grid_thw: tuple[tuple[int, int, int]])` — [`L615`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L615)
  - `compute_hidden_states(self, x: jax.Array, window_index: jax.Array, rotary_pos_emb: jax.Array, cu_seqlens: jax.Array, cu_window_seqlens: jax.Array)` — [`L689`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L689)
  - `encode_jit(self, x, grid_thw)` — [`L730`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L730)
  - `encode_padded_jit(self, x_padded, window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens)` — [`L723`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L723)
  - `get_rope_by_thw(self, t, h, w)` — [`L592`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L592)
  - `get_window_index_thw(self, grid_t, grid_h, grid_w)` — [`L550`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L550)
  - `pad_inputs(self, x, window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens)` — [`L662`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L662)
  - `rotary_pos_emb_thw(self, t, h, w)` — [`L522`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L522)
  - `blocks` — [`L500`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L500)
  - `config` — [`L473`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L473)
  - `dtype` — [`L474`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L474)
  - `enable_dynamic_image_sizes` — [`L519`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L519)
  - `fullatt_block_indexes` — [`L486`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L486)
  - `hidden_size` — [`L479`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L479)
  - `merger` — [`L509`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L509)
  - `num_heads` — [`L480`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L480)
  - `patch_embed` — [`L489`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L489)
  - `patch_size` — [`L484`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L484)
  - `rotary_pos_emb` — [`L498`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L498)
  - `spatial_merge_size` — [`L485`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L485)
  - `spatial_merge_unit` — [`L487`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L487)
  - `window_size` — [`L483`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L483)
- protocol/private: `__call__`[`L736`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L736), `__init__`[`L463`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L463)
- uses (calls/refs, reference-scoped): [`Qwen2_5_VisionBlock`](qwen2_5_vl.md#Qwen2_5_VisionBlock), [`Qwen2_5_VisionPatchEmbed`](qwen2_5_vl.md#Qwen2_5_VisionPatchEmbed), [`Qwen2_5_VisionPatchMerger`](qwen2_5_vl.md#Qwen2_5_VisionPatchMerger), [`Qwen2_5_VisionRotaryEmbedding`](qwen2_5_vl.md#Qwen2_5_VisionRotaryEmbedding)
- used by: [`visual`](qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration.visual), [`precompile_vision_encoder`](qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration.precompile_vision_encoder), [`_process_image_input`](qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration._process_image_input)

### `SegmentIds`  ·  implements/extends NamedTuple
- def: [`tpu_inference/models/jax/qwen2_5_vl.py:59`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L59)
- doc: SegmentIds for Q and KV sequences.
- signature: `class SegmentIds(NamedTuple):`
- members:
  - `kv` — [`L73`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L73)
  - `q` — [`L72`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L72)
- used by: [`generate_window_segment_ids`](qwen2_5_vl.md#generate_window_segment_ids)

## Functions
- `apply_rotary_pos_emb_vision(x: jax.Array, rotary_pos_emb: jax.Array)` — [`L161`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L161)
- `generate_window_segment_ids(cu_seqlens: jax.Array, seq_len: int, padded_seq_len: int)` — [`L190`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L190) — Generates segment IDs for windowed attention

## Module values
- `DEFAULT_BLOCK_K_MAJOR` — [`L56`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L56)
- `Qwen2_5_VLImageInputs` — [`L113`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L113)
- `init_fn` — [`L53`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L53)
- `logger` — [`L51`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L51)
- `modeling_flax_utils` — [`L54`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/qwen2_5_vl.py#L54)

