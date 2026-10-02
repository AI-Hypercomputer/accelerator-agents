---
title: 'Module: tpu_inference/models/jax/gemma4_mm.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/gemma4_mm.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.gemma4_mm`/
symbols:
  Gemma4VisionFlashAttention.__call__: Gemma4VisionFlashAttention#__call__().
  Gemma4VisionModel.__call__: Gemma4VisionModel#__call__().
  init_fn: init_fn.
  Gemma4VisionFlashAttention.q_proj: Gemma4VisionFlashAttention#q_proj.
  Gemma4VisionFlashAttention.k_proj: Gemma4VisionFlashAttention#k_proj.
  Gemma4VisionFlashAttention.v_proj: Gemma4VisionFlashAttention#v_proj.
  Gemma4VisionFlashAttention.o_proj: Gemma4VisionFlashAttention#o_proj.
  Gemma4ForConditionalGeneration.filter_weights: Gemma4ForConditionalGeneration#filter_weights().
  Gemma4VisionEncoderLayer.__call__: Gemma4VisionEncoderLayer#__call__().
  Gemma4ForConditionalGeneration.prepare_encoder_cudagraph_capture_inputs: Gemma4ForConditionalGeneration#prepare_encoder_cudagraph_capture_inputs().
  Gemma4ForConditionalGeneration.precompile_vision_encoder: Gemma4ForConditionalGeneration#precompile_vision_encoder().
  Gemma4VisionMLP.gate_proj: Gemma4VisionMLP#gate_proj.
  Gemma4VisionMLP.up_proj: Gemma4VisionMLP#up_proj.
  Gemma4VisionMLP.down_proj: Gemma4VisionMLP#down_proj.
  Gemma4ForConditionalGeneration.embed_input_ids: Gemma4ForConditionalGeneration#embed_input_ids().
  Gemma4ForConditionalGeneration._process_image_input: Gemma4ForConditionalGeneration#_process_image_input().
  Gemma4ForConditionalGeneration.encoder_cudagraph_forward: Gemma4ForConditionalGeneration#encoder_cudagraph_forward().
  Gemma4ForConditionalGeneration.__call__: Gemma4ForConditionalGeneration#__call__().
  Gemma4ForConditionalGeneration.compute_logits: Gemma4ForConditionalGeneration#compute_logits().
  Gemma4VisionFlashAttention.head_dim: Gemma4VisionFlashAttention#head_dim.
  Gemma4ForConditionalGeneration.lm_head: Gemma4ForConditionalGeneration#lm_head.
  Gemma4VisionPatchEmbedder.input_proj: Gemma4VisionPatchEmbedder#input_proj.
  Gemma4ForConditionalGeneration.model: Gemma4ForConditionalGeneration#model.
  Gemma4ForConditionalGeneration.final_logit_softcapping: Gemma4ForConditionalGeneration#final_logit_softcapping.
  Gemma4ForConditionalGeneration.get_encoder_cudagraph_item_specs: Gemma4ForConditionalGeneration#get_encoder_cudagraph_item_specs().
  Gemma4ForConditionalGeneration.get_single_image_embedding: Gemma4ForConditionalGeneration#get_single_image_embedding().
  Gemma4VisionFlashAttention.q_norm: Gemma4VisionFlashAttention#q_norm.
  Gemma4VisionFlashAttention.k_norm: Gemma4VisionFlashAttention#k_norm.
  Gemma4VisionModel.layers: Gemma4VisionModel#layers.
  Gemma4MultimodalEmbedder.embedding_projection: Gemma4MultimodalEmbedder#embedding_projection.
  Gemma4ForConditionalGeneration._parse_and_validate_image_input: Gemma4ForConditionalGeneration#_parse_and_validate_image_input().
  Gemma4VisionMLP.__call__: Gemma4VisionMLP#__call__().
  Gemma4VisionPooler.__call__: Gemma4VisionPooler#__call__().
  Gemma4ForConditionalGeneration.select_encoder_cudagraph_items: Gemma4ForConditionalGeneration#select_encoder_cudagraph_items().
  Gemma4ForConditionalGeneration: Gemma4ForConditionalGeneration#
  POSITIONS_PAD_VALUE: POSITIONS_PAD_VALUE.
  Gemma4VisionFlashAttention.v_norm: Gemma4VisionFlashAttention#v_norm.
  Gemma4VisionPatchEmbedder._factorized_posemb: Gemma4VisionPatchEmbedder#_factorized_posemb().
  Gemma4VisionEncoderLayer.input_layernorm: Gemma4VisionEncoderLayer#input_layernorm.
  Gemma4VisionEncoderLayer.post_attention_layernorm: Gemma4VisionEncoderLayer#post_attention_layernorm.
  Gemma4VisionEncoderLayer.pre_feedforward_layernorm: Gemma4VisionEncoderLayer#pre_feedforward_layernorm.
  Gemma4VisionEncoderLayer.post_feedforward_layernorm: Gemma4VisionEncoderLayer#post_feedforward_layernorm.
  Gemma4ForConditionalGeneration.mesh: Gemma4ForConditionalGeneration#mesh.
  Gemma4ForConditionalGeneration.embed_multimodal: Gemma4ForConditionalGeneration#embed_multimodal().
  Gemma4ForConditionalGeneration.vision_tower: Gemma4ForConditionalGeneration#vision_tower.
  Gemma4ForConditionalGeneration._get_pixel_position_ids_by_modality: Gemma4ForConditionalGeneration#_get_pixel_position_ids_by_modality().
  Gemma4VisionFlashAttention.features: Gemma4VisionFlashAttention#features.
  Gemma4VisionPatchEmbedder.__call__: Gemma4VisionPatchEmbedder#__call__().
  Gemma4MultimodalEmbedder.__call__: Gemma4MultimodalEmbedder#__call__().
  Gemma4ForConditionalGeneration.packed_modules_mapping: Gemma4ForConditionalGeneration#packed_modules_mapping.
  Gemma4ForConditionalGeneration.vllm_config: Gemma4ForConditionalGeneration#vllm_config.
  Gemma4ForConditionalGeneration.max_soft_tokens: Gemma4ForConditionalGeneration#max_soft_tokens.
  Gemma4ForConditionalGeneration.load_weights: Gemma4ForConditionalGeneration#load_weights().
  Gemma4ForConditionalGeneration.get_encoder_cudagraph_config: Gemma4ForConditionalGeneration#get_encoder_cudagraph_config().
  Gemma4ForConditionalGeneration.prepare_encoder_cudagraph_replay_buffers: Gemma4ForConditionalGeneration#prepare_encoder_cudagraph_replay_buffers().
  Gemma4VisionFlashAttention.num_kv_heads: Gemma4VisionFlashAttention#num_kv_heads.
  Gemma4VisionModel.patch_embedder: Gemma4VisionModel#patch_embedder.
  Gemma4VisionModel.standardize: Gemma4VisionModel#standardize.
  Gemma4ForConditionalGeneration.embed_vision: Gemma4ForConditionalGeneration#embed_vision.
  Gemma4ForConditionalGeneration._get_pixel_values_by_modality: Gemma4ForConditionalGeneration#_get_pixel_values_by_modality().
  Gemma4VisionFlashAttention.num_heads: Gemma4VisionFlashAttention#num_heads.
  logger: logger.
  Gemma4VisionFlashAttention: Gemma4VisionFlashAttention#
  Gemma4VisionPatchEmbedder: Gemma4VisionPatchEmbedder#
  Gemma4VisionMLP: Gemma4VisionMLP#
  Gemma4VisionEncoderLayer: Gemma4VisionEncoderLayer#
  Gemma4VisionEncoderLayer.self_attn: Gemma4VisionEncoderLayer#self_attn.
  Gemma4VisionEncoderLayer.mlp: Gemma4VisionEncoderLayer#mlp.
  Gemma4VisionPooler: Gemma4VisionPooler#
  Gemma4VisionModel: Gemma4VisionModel#
  Gemma4VisionModel.pooler: Gemma4VisionModel#pooler.
  Gemma4MultimodalEmbedder: Gemma4MultimodalEmbedder#
  Gemma4MultimodalEmbedder.embedding_pre_projection_norm: Gemma4MultimodalEmbedder#embedding_pre_projection_norm.
  Gemma4ForConditionalGeneration.pad_pixel_position_ids: Gemma4ForConditionalGeneration#pad_pixel_position_ids().
  Gemma4ImagePixelInputs: Gemma4ImagePixelInputs#
  Gemma4VisionMLP.features: Gemma4VisionMLP#features.
  Gemma4VisionMLP.hidden_dim: Gemma4VisionMLP#hidden_dim.
  Gemma4ForConditionalGeneration.get_input_modality: Gemma4ForConditionalGeneration#get_input_modality().
  Gemma4VisionFlashAttention.__init__: Gemma4VisionFlashAttention#__init__().
  Gemma4VisionMLP.__init__: Gemma4VisionMLP#__init__().
  Gemma4VisionEncoderLayer.__init__: Gemma4VisionEncoderLayer#__init__().
  Gemma4VisionModel.__init__: Gemma4VisionModel#__init__().
  Gemma4MultimodalEmbedder.__init__: Gemma4MultimodalEmbedder#__init__().
  Gemma4ForConditionalGeneration.WeightLoader: Gemma4ForConditionalGeneration#WeightLoader.
  Gemma4ForConditionalGeneration.get_encoder_cudagraph_budget_range: Gemma4ForConditionalGeneration#get_encoder_cudagraph_budget_range().
  Gemma4ForConditionalGeneration.encoder_eager_forward: Gemma4ForConditionalGeneration#encoder_eager_forward().
  apply_multidimensional_rope: apply_multidimensional_rope().
  Gemma4VisionFlashAttention.rope_base_frequency: Gemma4VisionFlashAttention#rope_base_frequency.
  Gemma4VisionPatchEmbedder.patch_size: Gemma4VisionPatchEmbedder#patch_size.
  Gemma4ForConditionalGeneration.image_token_id: Gemma4ForConditionalGeneration#image_token_id.
  Gemma4ForConditionalGeneration.pooling_kernel_size: Gemma4ForConditionalGeneration#pooling_kernel_size.
  Gemma4ImagePixelInputs.pixel_values: Gemma4ImagePixelInputs#pixel_values.
  Gemma4ImagePixelInputs.pixel_position_ids: Gemma4ImagePixelInputs#pixel_position_ids.
  SegmentIds: SegmentIds#
  SegmentIds.q: SegmentIds#q.
  SegmentIds.kv: SegmentIds#kv.
  Gemma4VisionFlashAttention.mesh: Gemma4VisionFlashAttention#mesh.
  Gemma4VisionPatchEmbedder.position_embedding_table: Gemma4VisionPatchEmbedder#position_embedding_table.
  Gemma4VisionPooler.config: Gemma4VisionPooler#config.
  Gemma4VisionPooler.d_model: Gemma4VisionPooler#d_model.
  Gemma4VisionPooler.param_dtype: Gemma4VisionPooler#param_dtype.
  Gemma4VisionModel.mesh: Gemma4VisionModel#mesh.
  Gemma4VisionModel.start_layer: Gemma4VisionModel#start_layer.
  Gemma4VisionModel.end_layer: Gemma4VisionModel#end_layer.
  Gemma4VisionModel.std_bias: Gemma4VisionModel#std_bias.
  Gemma4VisionModel.std_scale: Gemma4VisionModel#std_scale.
  Gemma4ForConditionalGeneration.patch_pixels: Gemma4ForConditionalGeneration#patch_pixels.
  Gemma4ForConditionalGeneration.map_name: Gemma4ForConditionalGeneration#map_name().
  Gemma4ForConditionalGeneration.process_tensor: Gemma4ForConditionalGeneration#process_tensor().
  Gemma4VisionPatchEmbedder.__init__: Gemma4VisionPatchEmbedder#__init__().
  Gemma4VisionPatchEmbedder.config: Gemma4VisionPatchEmbedder#config.
  Gemma4VisionPatchEmbedder.dtype: Gemma4VisionPatchEmbedder#dtype.
  Gemma4VisionPooler.__init__: Gemma4VisionPooler#__init__().
  Gemma4VisionModel.config: Gemma4VisionModel#config.
  Gemma4VisionModel.dtype: Gemma4VisionModel#dtype.
  Gemma4ForConditionalGeneration.supports_multimodal: Gemma4ForConditionalGeneration#supports_multimodal.
  Gemma4ForConditionalGeneration.supports_encoder_tp_data: Gemma4ForConditionalGeneration#supports_encoder_tp_data.
  Gemma4ForConditionalGeneration.supports_encoder_cudagraph: Gemma4ForConditionalGeneration#supports_encoder_cudagraph.
  Gemma4ForConditionalGeneration._processor_factory: Gemma4ForConditionalGeneration#_processor_factory.
  Gemma4ForConditionalGeneration.__init__: Gemma4ForConditionalGeneration#__init__().
  Gemma4ForConditionalGeneration.get_max_frames_per_video: Gemma4ForConditionalGeneration#get_max_frames_per_video().
  Gemma4ForConditionalGeneration.postprocess_encoder_output: Gemma4ForConditionalGeneration#postprocess_encoder_output().
---
# Module: [`tpu_inference/models/jax/gemma4_mm.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py)

## Classes
### `Gemma4ForConditionalGeneration`  ·  implements/extends JaxModule, LoadableWithIterator
- def: [`tpu_inference/models/jax/gemma4_mm.py:571`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L571)
- signature: `class Gemma4ForConditionalGeneration(JaxModule, LoadableWithIterator):`
- members:
  - `compute_logits(self, hidden_states: jax.Array)` — [`L1079`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L1079)
  - `embed_input_ids(self, input_ids: jax.Array, multimodal_embeddings: Optional[jax.Array] = None, **kwargs)` — [`L700`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L700)
  - `embed_multimodal(self, **kwargs)` — [`L808`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L808)
  - `encoder_cudagraph_forward(self, inputs: dict)` — [`L964`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L964) — Run the vision encoder on fixed-shape inputs.
  - `encoder_eager_forward(self, mm_kwargs: dict[str, Any])` — [`L1005`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L1005) — Fallback for inputs that exceed all budget sizes.
  - `filter_weights(weights_iterator)` — [`L663`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L663)
  - `get_encoder_cudagraph_budget_range(self, vllm_config)` — [`L846`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L846)
  - `get_encoder_cudagraph_config(self)` — [`L817`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L817)
  - `get_encoder_cudagraph_item_specs(self, mm_kwargs)` — [`L868`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L868)
  - `get_input_modality(self, mm_kwargs: dict[str, Any])` — [`L840`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L840)
  - `get_max_frames_per_video(self)` — [`L837`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L837)
  - `get_single_image_embedding(self, pixel_values: jax.Array, pixel_position_ids: jax.Array)` — [`L719`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L719)
  - `load_weights(self, weights: Iterable[Tuple[str, Any]])` — [`L638`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L638)
  - `map_name(name: str)` — [`L640`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L640)
  - `pad_pixel_position_ids(dst: torch.Tensor, src: torch.Tensor)` — [`L821`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L821)
  - `postprocess_encoder_output(self, output, indices, per_item_out_tokens, dest, clone: bool = False, batch_mm_kwargs=None)` — [`L1010`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L1010) — Split batch encoder output into per-image entries in dest.
  - `precompile_vision_encoder(self, run_compilation_fn: Callable)` — [`L1091`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L1091)
  - `prepare_encoder_cudagraph_capture_inputs(self, token_budget: int, max_batch_size: int, max_frames_per_batch: int, device: torch.device, dtype: torch.dtype)` — [`L914`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L914)
  - `prepare_encoder_cudagraph_replay_buffers(self, mm_kwargs, max_batch_size: int, max_frames_per_batch: int, path: str = "default")` — [`L947`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L947)
  - `process_tensor(mapped_name, tensor)` — [`L656`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L656)
  - `select_encoder_cudagraph_items(self, mm_kwargs: dict[str, Any], indices: list[int])` — [`L889`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L889)
  - `WeightLoader` — [`L573`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L573)
  - `embed_vision` — [`L608`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L608)
  - `final_logit_softcapping` — [`L616`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L616)
  - `image_token_id` — [`L595`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L595)
  - `lm_head` — [`L625`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L625)
  - `max_soft_tokens` — [`L598`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L598)
  - `mesh` — [`L583`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L583)
  - `model` — [`L587`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L587)
  - `packed_modules_mapping` — [`L572`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L572)
  - `patch_pixels` — [`L599`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L599)
  - `pooling_kernel_size` — [`L597`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L597)
  - `supports_encoder_cudagraph` — [`L576`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L576)
  - `supports_encoder_tp_data` — [`L575`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L575)
  - `supports_multimodal` — [`L574`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L574)
  - `vision_tower` — [`L601`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L601)
  - `vllm_config` — [`L581`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L581)
- protocol/private: `__call__`[`L1027`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L1027), `__init__`[`L579`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L579), `_get_pixel_position_ids_by_modality`[`L862`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L862), `_get_pixel_values_by_modality`[`L856`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L856), `_parse_and_validate_image_input`[`L749`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L749), `_process_image_input`[`L770`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L770), `_processor_factory`[`L577`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L577)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../layers/common/sharding.md#ShardingAxisName), [`JaxModule`](../../layers/__init__.md#JaxModule), [`JaxIntermediateTensors`](jax_intermediate_tensor.md#JaxIntermediateTensors), [`PPMissingLayer`](../../layers/jax/pp_utils.md#PPMissingLayer), [`LoadableWithIterator`](utils/weight_utils.md#LoadableWithIterator), [`embed_tokens`](gemma4.md#Gemma4Model.embed_tokens), [`JaxLmHead`](../../layers/jax/linear.md#JaxLmHead), [`load_weights`](utils/weight_utils.md#LoadableWithIterator.load_weights), [`tensors`](jax_intermediate_tensor.md#JaxIntermediateTensors.tensors), [`init_fn`](gemma4_mm.md#init_fn), [`StandardWeightLoader`](utils/weight_utils.md#StandardWeightLoader), [`layers`](gemma4.md#Gemma4Model.layers), [`decode`](../../layers/jax/embed.md#JaxEmbed.decode), [`Gemma4ForCausalLM`](gemma4.md#Gemma4ForCausalLM), [`is_last_rank`](gemma4.md#Gemma4Model.is_last_rank), [`merge_multimodal_embeddings`](utils/multi_modal_utils.md#merge_multimodal_embeddings), [`POSITIONS_PAD_VALUE`](gemma4_mm.md#POSITIONS_PAD_VALUE), [`Gemma4Model`](gemma4.md#Gemma4Model), [`patch_embedder`](gemma4_mm.md#Gemma4VisionModel.patch_embedder), [`Gemma4MultimodalEmbedder`](gemma4_mm.md#Gemma4MultimodalEmbedder), [`Gemma4VisionModel`](gemma4_mm.md#Gemma4VisionModel), [`logger`](gemma4_mm.md#logger), [`Gemma4ImagePixelInputs`](gemma4_mm.md#Gemma4ImagePixelInputs), [`start_layer`](gemma4.md#Gemma4Model.start_layer), [`embedding_scale`](gemma4.md#Gemma4Model.embedding_scale), [`end_layer`](gemma4.md#Gemma4Model.end_layer), [`packed_modules_mapping`](gemma4.md#Gemma4ForCausalLM.packed_modules_mapping), [`patch_size`](gemma4_mm.md#Gemma4VisionPatchEmbedder.patch_size), [`pixel_position_ids`](gemma4_mm.md#Gemma4ImagePixelInputs.pixel_position_ids), [`pixel_values`](gemma4_mm.md#Gemma4ImagePixelInputs.pixel_values)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`_get_model_architecture`](../common/model_loader.md#_get_model_architecture), [`LoadableWithIterator`](utils/weight_utils.md#LoadableWithIterator), [`load_weights`](utils/weight_utils.md#LoadableWithIterator.load_weights)

### `Gemma4ImagePixelInputs`  ·  implements/extends _TypedDict
- def: [`tpu_inference/models/jax/gemma4_mm.py:53`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L53)
- doc: Pre-patchified image inputs from the Gemma4 image processor.
- signature: `class Gemma4ImagePixelInputs(TypedDict):`
- members:
  - `pixel_position_ids` — [`L73`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L73) — ---
  - `pixel_values` — [`L69`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L69) — ---
- used by: [`_process_image_input`](gemma4_mm.md#Gemma4ForConditionalGeneration._process_image_input), [`_parse_and_validate_image_input`](gemma4_mm.md#Gemma4ForConditionalGeneration._parse_and_validate_image_input)

### `Gemma4MultimodalEmbedder`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/gemma4_mm.py:532`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L532)
- signature: `class Gemma4MultimodalEmbedder(JaxModule):`
- members:
  - `embedding_pre_projection_norm` — [`L554`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L554)
  - `embedding_projection` — [`L542`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L542)
- protocol/private: `__call__`[`L565`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L565), `__init__`[`L534`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L534)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../layers/common/sharding.md#ShardingAxisName), [`JaxModule`](../../layers/__init__.md#JaxModule), [`JaxEinsum`](../../layers/jax/linear.md#JaxEinsum), [`VllmQuantConfig`](../../layers/vllm/quantization/configs.md#VllmQuantConfig), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`init_fn`](gemma4_mm.md#init_fn)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`embed_vision`](gemma4_mm.md#Gemma4ForConditionalGeneration.embed_vision)

### `Gemma4VisionEncoderLayer`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/gemma4_mm.py:379`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L379)
- signature: `class Gemma4VisionEncoderLayer(JaxModule):`
- members:
  - `input_layernorm` — [`L387`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L387)
  - `mlp` — [`L410`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L410)
  - `post_attention_layernorm` — [`L397`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L397)
  - `post_feedforward_layernorm` — [`L411`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L411)
  - `pre_feedforward_layernorm` — [`L404`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L404)
  - `self_attn` — [`L394`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L394)
- protocol/private: `__call__`[`L418`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L418), `__init__`[`L381`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L381)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`VllmQuantConfig`](../../layers/vllm/quantization/configs.md#VllmQuantConfig), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`init_fn`](gemma4_mm.md#init_fn), [`Gemma4VisionFlashAttention`](gemma4_mm.md#Gemma4VisionFlashAttention), [`Gemma4VisionMLP`](gemma4_mm.md#Gemma4VisionMLP)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`layers`](gemma4_mm.md#Gemma4VisionModel.layers)

### `Gemma4VisionFlashAttention`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/gemma4_mm.py:130`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L130)
- doc: Gemma 4 Vision Attention using TPU sharded_flash_attention.
- signature: `class Gemma4VisionFlashAttention(JaxModule):`
- members:
  - `features` — [`L141`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L141)
  - `head_dim` — [`L145`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L145)
  - `k_norm` — [`L191`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L191)
  - `k_proj` — [`L162`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L162)
  - `mesh` — [`L147`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L147)
  - `num_heads` — [`L142`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L142)
  - `num_kv_heads` — [`L143`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L143)
  - `o_proj` — [`L176`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L176)
  - `q_norm` — [`L185`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L185)
  - `q_proj` — [`L154`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L154)
  - `rope_base_frequency` — [`L152`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L152)
  - `v_norm` — [`L197`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L197)
  - `v_proj` — [`L169`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L169)
- protocol/private: `__call__`[`L204`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L204), `__init__`[`L135`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L135)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../layers/common/sharding.md#ShardingAxisName), [`JaxModule`](../../layers/__init__.md#JaxModule), [`JaxEinsum`](../../layers/jax/linear.md#JaxEinsum), [`VllmQuantConfig`](../../layers/vllm/quantization/configs.md#VllmQuantConfig), [`JaxRmsNorm`](../../layers/jax/norm.md#JaxRmsNorm), [`init_fn`](gemma4_mm.md#init_fn), [`sharded_flash_attention`](../../layers/common/attention_interface.md#sharded_flash_attention), [`apply_multidimensional_rope`](gemma4_mm.md#apply_multidimensional_rope), [`SegmentIds`](gemma4_mm.md#SegmentIds), [`kv`](gemma4_mm.md#SegmentIds.kv), [`q`](gemma4_mm.md#SegmentIds.q)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`self_attn`](gemma4_mm.md#Gemma4VisionEncoderLayer.self_attn)

### `Gemma4VisionMLP`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/gemma4_mm.py:330`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L330)
- doc: Feed forward module.
- signature: `class Gemma4VisionMLP(JaxModule):`
- members:
  - `down_proj` — [`L363`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L363)
  - `features` — [`L338`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L338)
  - `gate_proj` — [`L341`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L341)
  - `hidden_dim` — [`L339`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L339)
  - `up_proj` — [`L352`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L352)
- protocol/private: `__call__`[`L374`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L374), `__init__`[`L333`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L333)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../layers/common/sharding.md#ShardingAxisName), [`JaxModule`](../../layers/__init__.md#JaxModule), [`JaxEinsum`](../../layers/jax/linear.md#JaxEinsum), [`VllmQuantConfig`](../../layers/vllm/quantization/configs.md#VllmQuantConfig), [`init_fn`](gemma4_mm.md#init_fn)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`mlp`](gemma4_mm.md#Gemma4VisionEncoderLayer.mlp)

### `Gemma4VisionModel`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/gemma4_mm.py:476`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L476)
- doc: Top-level wrapper for the Gemma 4 Vision Encoder.
- signature: `class Gemma4VisionModel(JaxModule):`
- members:
  - `config` — [`L487`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L487)
  - `dtype` — [`L489`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L489)
  - `end_layer` — [`L496`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L496)
  - `layers` — [`L496`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L496)
  - `mesh` — [`L490`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L490)
  - `patch_embedder` — [`L492`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L492)
  - `pooler` — [`L500`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L500)
  - `standardize` — [`L502`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L502)
  - `start_layer` — [`L496`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L496)
  - `std_bias` — [`L504`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L504)
  - `std_scale` — [`L506`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L506)
- protocol/private: `__call__`[`L509`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L509), `__init__`[`L481`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L481)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule), [`VllmQuantConfig`](../../layers/vllm/quantization/configs.md#VllmQuantConfig), [`make_layers`](../../layers/jax/pp_utils.md#make_layers), [`Gemma4VisionEncoderLayer`](gemma4_mm.md#Gemma4VisionEncoderLayer), [`Gemma4VisionPatchEmbedder`](gemma4_mm.md#Gemma4VisionPatchEmbedder), [`Gemma4VisionPooler`](gemma4_mm.md#Gemma4VisionPooler)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`precompile_vision_encoder`](gemma4_mm.md#Gemma4ForConditionalGeneration.precompile_vision_encoder), [`vision_tower`](gemma4_mm.md#Gemma4ForConditionalGeneration.vision_tower)

### `Gemma4VisionPatchEmbedder`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/gemma4_mm.py:269`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L269)
- doc: Handles converting input \[B, H, W, C\] to patches \[B, L, D\],
- signature: `class Gemma4VisionPatchEmbedder(JaxModule):`
- members:
  - `config` — [`L276`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L276)
  - `dtype` — [`L277`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L277)
  - `input_proj` — [`L281`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L281)
  - `patch_size` — [`L279`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L279)
  - `position_embedding_table` — [`L293`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L293)
- protocol/private: `__call__`[`L312`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L312), `__init__`[`L275`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L275), `_factorized_posemb`[`L297`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L297)
- uses (calls/refs, reference-scoped): [`ShardingAxisName`](../../layers/common/sharding.md#ShardingAxisName), [`JaxModule`](../../layers/__init__.md#JaxModule), [`JaxEinsum`](../../layers/jax/linear.md#JaxEinsum), [`init_fn`](gemma4_mm.md#init_fn), [`POSITIONS_PAD_VALUE`](gemma4_mm.md#POSITIONS_PAD_VALUE)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`precompile_vision_encoder`](gemma4_mm.md#Gemma4ForConditionalGeneration.precompile_vision_encoder), [`patch_embedder`](gemma4_mm.md#Gemma4VisionModel.patch_embedder)

### `Gemma4VisionPooler`  ·  implements/extends JaxModule
- def: [`tpu_inference/models/jax/gemma4_mm.py:439`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L439)
- doc: Vision exit layer with dynamic spatial pooling.
- signature: `class Gemma4VisionPooler(JaxModule):`
- members:
  - `config` — [`L445`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L445)
  - `d_model` — [`L446`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L446)
  - `param_dtype` — [`L447`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L447)
- protocol/private: `__call__`[`L449`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L449), `__init__`[`L444`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L444)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../layers/__init__.md#JaxModule)
- used by: [`JaxModule`](../../layers/__init__.md#JaxModule), [`pooler`](gemma4_mm.md#Gemma4VisionModel.pooler)

### `SegmentIds`  ·  implements/extends NamedTuple
- def: [`tpu_inference/models/jax/gemma4_mm.py:124`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L124)
- doc: SegmentIds required by TPU sharded_flash_attention backend.
- signature: `class SegmentIds(NamedTuple):`
- members:
  - `kv` — [`L127`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L127)
  - `q` — [`L126`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L126)
- used by: [`__call__`](gemma4_mm.md#Gemma4VisionFlashAttention.__call__)

## Functions
- `apply_multidimensional_rope(inputs: jax.Array, positions: jax.Array, base_frequency: int, rotary_fraction: Optional[float] = None)` — [`L79`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L79) — Applies multidimensional RoPE.

## Module values
- `POSITIONS_PAD_VALUE` — [`L49`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L49)
- `init_fn` — [`L50`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L50)
- `logger` — [`L47`](../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/gemma4_mm.py#L47)

