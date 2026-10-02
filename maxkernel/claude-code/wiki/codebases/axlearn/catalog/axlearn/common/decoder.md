---
title: 'Module: axlearn/common/decoder.py'
type: catalog
provenance: extracted
module: axlearn/common/decoder.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.decoder`/
symbols:
  DecodingLayer.sample_decode: DecodingLayer#sample_decode().
  Decoder.__init__: Decoder#__init__().
  DecodingLayer.beam_search_decode: DecodingLayer#beam_search_decode().
  Decoder.extend_step: Decoder#extend_step().
  Decoder._forward_for_mode: Decoder#_forward_for_mode().
  Decoder.compute_logits: Decoder#compute_logits().
  Decoder.init_states: Decoder#init_states().
  Decoder.forward: Decoder#forward().
  Decoder.prefill_states: Decoder#prefill_states().
  DecodingLayer.tokens_to_scores: DecodingLayer#tokens_to_scores().
  Decoder._decoding: Decoder#_decoding.
  DecodingLayer._tokens_to_scores: DecodingLayer#_tokens_to_scores().
  Decoder.Config.emb: Decoder#Config#emb.
  Decoder.Config.transformer: Decoder#Config#transformer.
  Decoder.beam_search_decode: Decoder#beam_search_decode().
  Decoder.sample_decode: Decoder#sample_decode().
  LmHead._create_layer_parameter_specs: LmHead#_create_layer_parameter_specs().
  Decoder.Config: Decoder#Config#
  Decoder.Config.output_norm: Decoder#Config#output_norm.
  Decoder.compute_attention_logit_biases: Decoder#compute_attention_logit_biases().
  Decoder.Config.dim: Decoder#Config#dim.
  _temporary_output_collection: _temporary_output_collection().
  DecodingLayer.Config: DecodingLayer#Config#
  Decoder.Config.output_dropout: Decoder#Config#output_dropout.
  Decoder.Config.decoding: Decoder#Config#decoding.
  Decoder.Config.attention_mask: Decoder#Config#attention_mask.
  BaseDecoder.init_states: BaseDecoder#init_states().
  LmHead.Config: LmHead#Config#
  log_probs_from_logits: log_probs_from_logits().
  BaseDecoder.prefill_states: BaseDecoder#prefill_states().
  Decoder.Config.vocab_size: Decoder#Config#vocab_size.
  Decoder._output_logits_modifier: Decoder#_output_logits_modifier.
  Decoder: Decoder#
  BaseDecoder.extend_step: BaseDecoder#extend_step().
  Decoder.Config.output_logits_modifier: Decoder#Config#output_logits_modifier.
  Decoder.infer_fwd_dtype: Decoder#infer_fwd_dtype().
  DecodingLayer: DecodingLayer#
  DecodingLayer.Config.pad_token_id: DecodingLayer#Config#pad_token_id.
  DecodingLayer.Config.eos_token_id: DecodingLayer#Config#eos_token_id.
  DecodingLayer.__init__: DecodingLayer#__init__().
  LmHead.Config.vocab_size: LmHead#Config#vocab_size.
  LmHead.Config.embedding_dim: LmHead#Config#embedding_dim.
  LmHead.forward: LmHead#forward().
  DecodingLayer._pad: DecodingLayer#_pad().
  Decoder.Config.lm_head: Decoder#Config#lm_head.
  DecodingLayer._decoder: DecodingLayer#_decoder.
  _segment_ids_from_causal_input_ids: _segment_ids_from_causal_input_ids().
  BaseDecoder.prng_key: BaseDecoder#prng_key().
  DecodingLayer._get_init_scores: DecodingLayer#_get_init_scores().
  LmHead: LmHead#
  LmHead.default_config: LmHead#default_config().
  Decoder.Config.pad_token_id: Decoder#Config#pad_token_id.
  Decoder.Config.logits_forward_dtype: Decoder#Config#logits_forward_dtype.
  BaseDecoder: BaseDecoder#
  DecodingLayer._input_ids_after_prefill: DecodingLayer#_input_ids_after_prefill().
  Decoder.Config.dropout_rate: Decoder#Config#dropout_rate.
  Decoder.Config.eos_token_id: Decoder#Config#eos_token_id.
  Decoder.Config.logits_partition_spec: Decoder#Config#logits_partition_spec.
---
# Module: [`axlearn/common/decoder.py`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py)

## Classes
### `BaseDecoder`  ·  implements/extends Protocol
- def: [`axlearn/common/decoder.py:131`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L131)
- doc: Defines the interface that a Decoder must follow for compatibility with decoding.
- signature: `class BaseDecoder(Protocol):`
- members:
  - `extend_step(self, *, cached_states: Nested[Tensor], input_batch: Nested[Tensor], is_prefill: bool = False, segment_ids: Optional[Tensor] = None, **kwargs)` — [`L173`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L173) — Computes incremental outputs during autoregressive decoding.
  - `init_states(self, *, batch_size: int, max_sequence_length: int, dtype: jnp.dtype)` — [`L139`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L139) — Initializes cache for autoregressive cached decoding.
  - `prefill_states(self, *, time_step: Tensor, input_batch: Nested[Tensor], **kwargs)` — [`L154`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L154) — Initializes cache for autoregressive cached decoding, DEPRECATED.
  - `prng_key(self)` — [`L135`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L135) — A PRNG key for sampling.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`nowrap`](module.md#nowrap)
- used by: [`sample_decode`](decoder.md#DecodingLayer.sample_decode), [`beam_search_decode`](decoder.md#DecodingLayer.beam_search_decode), [`tokens_to_scores`](decoder.md#DecodingLayer.tokens_to_scores), [`__init__`](decoder.md#DecodingLayer.__init__)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/decoder.py:918`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L918)
- signature: `class Config(BaseLayer.Config):`
- members:
  - `attention_mask` — [`L486`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L486)
  - `decoding` — [`L518`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L518)
  - `dim` — [`L489`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L489)
  - `dropout_rate` — [`L492`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L492)
  - `emb` — [`L494`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L494)
  - `embedding_dim` — [`L921`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L921)
  - `eos_token_id` — [`L212`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L212)
  - `eos_token_id` — [`L505`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L505)
  - `lm_head` — [`L503`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L503)
  - `logits_forward_dtype` — [`L514`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L514)
  - `logits_partition_spec` — [`L507`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L507)
  - `output_dropout` — [`L501`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L501)
  - `output_logits_modifier` — [`L516`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L516)
  - `output_norm` — [`L498`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L498)
  - `pad_token_id` — [`L211`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L211)
  - `pad_token_id` — [`L504`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L504)
  - `transformer` — [`L496`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L496)
  - `vocab_size` — [`L487`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L487)
  - `vocab_size` — [`L919`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L919)
- uses (calls/refs, reference-scoped): [`default_config`](config.md#Configurable.default_config), [`BaseLayer`](base_layer.md#BaseLayer), [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`InstantiableConfig`](config.md#InstantiableConfig), [`Config`](base_layer.md#BaseLayer.Config), [`ConfigOr`](config.md#ConfigOr), [`StackedTransformerLayer`](attention.md#StackedTransformerLayer), [`Dropout`](layers.md#Dropout), [`BaseStackedTransformerLayer`](attention.md#BaseStackedTransformerLayer), [`Config`](attention.md#BaseStackedTransformerLayer.Config), [`BaseEmbedding`](embedding.md#BaseEmbedding), [`AttentionLogitBiasLayer`](attention.md#AttentionLogitBiasLayer), [`LogitsToLogitsFn`](logit_modifiers.md#LogitsToLogitsFn), [`Config`](embedding.md#BaseEmbedding.Config), [`Config`](layers.md#Dropout.Config), [`LayerNorm`](layers.md#LayerNorm), [`Config`](decoder.md#DecodingLayer.Config), [`TransformerTextEmbeddings`](embedding.md#TransformerTextEmbeddings), [`DecodingLayer`](decoder.md#DecodingLayer)
- used by: [`__init__`](decoder.md#Decoder.__init__), [`init_states`](decoder.md#Decoder.init_states), [`_decoding`](decoder.md#Decoder._decoding), [`_output_logits_modifier`](decoder.md#Decoder._output_logits_modifier)

### `Decoder`  ·  implements/extends BaseLayer
- def: [`axlearn/common/decoder.py:477`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L477)
- doc: Construct a decoder transformer to output hidden states and logits based on lm head.
- signature: `class Decoder(BaseLayer):`
- members:
  - `beam_search_decode(self, *, input_batch: Nested[Tensor], max_sequence_length: int, num_decodes: int, **kwargs)` — [`L845`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L845) — See configured `decoding` implementation for details.
  - `compute_attention_logit_biases(self, input_ids: Tensor, *, segment_ids: Optional[Tensor] = None, positions: Optional[Tensor] = None)` — [`L877`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L877) — Produces self-attention logit biases.
  - `compute_logits(self, forward_outputs: Nested[Tensor])` — [`L606`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L606) — Computes logits from decoder forward outputs.
  - `extend_step(self, *, cached_states: Nested[Tensor], input_batch: Nested[Tensor], is_prefill: bool = False, **kwargs)` — [`L768`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L768) — See `BaseDecoder.forward_step` for details.
  - `forward(self, input_batch: Nested[Tensor], *, cross_attention_data: Optional[Tensor] = None, cross_attention_logit_biases: Optional[Tensor] = None, **kwargs)` — [`L644`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L644) — Computes decoder hidden states from input ids and cross attention hidden states.
  - `infer_fwd_dtype()` — [`L744`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L744)
  - `init_states(self, *, batch_size: int, max_sequence_length: int, dtype: jnp.dtype)` — [`L694`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L694) — See `BaseDecoder.init_states` for details.
  - `prefill_states(self, *, time_step: Tensor, input_batch: Nested[Tensor], **kwargs)` — [`L715`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L715) — See `BaseDecoder.prefill_states` for details, DEPRECATED.
  - `sample_decode(self, *, input_batch: Nested[Tensor], max_sequence_length: int, num_decodes: int, **kwargs)` — [`L861`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L861) — See configured `decoding` implementation for details.
- protocol/private: `__init__`[`L520`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L520), `_decoding`[`L544`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L544), `_forward_for_mode`[`L548`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L548), `_output_logits_modifier`[`L543`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L543)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`BaseLayer`](base_layer.md#BaseLayer), [`Module`](module.md#Module), [`instantiate`](config.md#Configurable.Config.instantiate), [`_add_child`](module.md#Module._add_child), [`ForwardMode`](attention.md#ForwardMode), [`maybe_instantiate`](config.md#maybe_instantiate), [`__init__`](base_layer.md#BaseLayer.__init__), [`sample_decode`](decoder.md#DecodingLayer.sample_decode), [`child_context`](module.md#child_context), [`parameters`](base_layer.md#BaseLayer.parameters), [`with_sharding_constraint`](utils.md#with_sharding_constraint), [`nowrap`](module.md#nowrap), [`beam_search_decode`](decoder.md#DecodingLayer.beam_search_decode), [`PREFILL`](attention.md#ForwardMode.PREFILL), [`_add_tensor_stats`](base_layer.md#BaseLayer._add_tensor_stats), [`maybe_shard`](utils.md#maybe_shard), [`children`](module.md#Module.children), [`EXTEND_STEP`](attention.md#ForwardMode.EXTEND_STEP), [`FORWARD`](attention.md#ForwardMode.FORWARD), [`validate_contains_paths`](utils.md#validate_contains_paths), [`repeat`](ein_ops.md#repeat), [`dtype`](base_layer.md#BaseLayer.dtype), [`sequence_mask`](utils.md#sequence_mask), [`set_dropout_rate_recursively`](layers.md#set_dropout_rate_recursively), [`emb`](decoder.md#Decoder.Config.emb), [`transformer`](decoder.md#Decoder.Config.transformer), [`Config`](decoder.md#Decoder.Config), [`add_module_output`](module.md#Module.add_module_output), [`output_norm`](decoder.md#Decoder.Config.output_norm), [`dim`](decoder.md#Decoder.Config.dim), [`decoding`](decoder.md#Decoder.Config.decoding), [`output_dropout`](decoder.md#Decoder.Config.output_dropout), [`attention_mask`](decoder.md#Decoder.Config.attention_mask), [`vocab_size`](decoder.md#Decoder.Config.vocab_size), [`output_logits_modifier`](decoder.md#Decoder.Config.output_logits_modifier), [`DecodingLayer`](decoder.md#DecodingLayer)  (+6 more)
- used by: [`BaseLayer`](base_layer.md#BaseLayer), `_test_decoder_with_transformer`

### `DecodingLayer`  ·  implements/extends Configurable
- def: [`axlearn/common/decoder.py:204`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L204)
- doc: Base decoding API.
- signature: `class DecodingLayer(Configurable):`
- members:
  - `_pad(prefix: Tensor, *, max_sequence_length: int, pad_id: int)` — [`L461`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L461) — Accept token IDs input tensor and pad if necessary to max_sequence_length.
  - `_tokens_to_scores(self, *, num_decodes: int, cross_attention_data: Optional[Tensor] = None, cross_attention_logit_biases: Optional[Tensor] = None, logits_modifier: Optional[LogitsToLogitsFn] = None)` — [`L384`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L384) — Build a fn mapping current token IDs and model state to next logits and updated state.
  - `beam_search_decode(self, *, input_batch: Nested[Tensor], max_sequence_length: int, num_decodes: int, cross_attention_data: Optional[Tensor] = None, cross_attention_logit_biases: Optional[Tensor] = None, brevity_penalty: Optional[BrevityPenaltyFn] = None)` — [`L218`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L218) — Perform beam search decoding.
  - `sample_decode(self, *, input_batch: Nested[Tensor], max_sequence_length: int, num_decodes: int, cross_attention_data: Optional[Tensor] = None, cross_attention_logit_biases: Optional[Tensor] = None, logits_modifier: Optional[ConfigOr[LogitsToLogitsFn]] = None, stop_decoding_condition: Optional[StopDecodingCondition] = None)` — [`L287`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L287) — Perform sample-based decoding.
  - `tokens_to_scores(token_ids: Tensor, cache: NestedTensor)` — [`L417`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L417) — Maps current token IDs and model state to next logits and updated state.
- protocol/private: `__init__`[`L214`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L214), `_decoder`[`L216`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L216), `_get_init_scores`[`L356`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L356), `_input_ids_after_prefill`[`L342`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L342)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`maybe_instantiate`](config.md#maybe_instantiate), [`Configurable`](config.md#Configurable), [`ConfigOr`](config.md#ConfigOr), [`beam_search_decode`](decoding.md#beam_search_decode), [`sample_decode`](decoding.md#sample_decode), [`validate_contains_paths`](utils.md#validate_contains_paths), [`StopDecodingCondition`](decoding.md#StopDecodingCondition), [`sequence_mask`](utils.md#sequence_mask), [`LogitsToLogitsFn`](logit_modifiers.md#LogitsToLogitsFn), [`__init__`](config.md#Configurable.__init__), [`_temporary_output_collection`](decoder.md#_temporary_output_collection), [`log_probs_from_logits`](decoder.md#log_probs_from_logits), [`prefill_states`](decoder.md#BaseDecoder.prefill_states), [`extend_step`](decoder.md#BaseDecoder.extend_step), [`infer_initial_time_step`](decoding.md#infer_initial_time_step), [`StopOnSubsequence`](decoding.md#StopOnSubsequence), [`BeamSearchOutputs`](decoding.md#BeamSearchOutputs), [`BrevityPenaltyFn`](decoding.md#BrevityPenaltyFn), [`SampleOutputs`](decoding.md#SampleOutputs), [`prng_key`](decoder.md#BaseDecoder.prng_key), [`BaseDecoder`](decoder.md#BaseDecoder)
- used by: [`Configurable`](config.md#Configurable), [`_decoding`](decoder.md#Decoder._decoding), [`beam_search_decode`](decoder.md#Decoder.beam_search_decode), [`sample_decode`](decoder.md#Decoder.sample_decode), [`decoding`](decoder.md#Decoder.Config.decoding)

### `LmHead`  ·  implements/extends BaseLayer
- def: [`axlearn/common/decoder.py:914`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L914)
- doc: LM head layer for decoder to compute logits.
- signature: `class LmHead(BaseLayer):`
- members:
  - `default_config(cls)` — [`L924`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L924)
  - `forward(self, x: Tensor)` — [`L941`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L941) — Computes logits with token embedding.
- protocol/private: `_create_layer_parameter_specs`[`L929`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L929)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`default_config`](config.md#Configurable.default_config), [`config`](config.md#Configurable.config), [`BaseLayer`](base_layer.md#BaseLayer), [`ParameterSpec`](base_layer.md#ParameterSpec), [`shape`](utils.md#TensorSpec.shape), [`mesh_axes`](utils.md#TensorSpec.mesh_axes), [`parameters`](base_layer.md#BaseLayer.parameters)
- used by: [`default_config`](config.md#Configurable.default_config), [`BaseLayer`](base_layer.md#BaseLayer), [`_create_layer_parameter_specs`](base_layer.md#BaseLayer._create_layer_parameter_specs)

## Functions
- `_segment_ids_from_causal_input_ids(input_ids: Tensor, *, pad_token_id: int)` — [`L78`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L78) — Computes segment_ids from inputs.
- `_temporary_output_collection()` — [`L63`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L63) — Overrides the output collection without introducing a child context.
- `log_probs_from_logits(logits: Tensor, logits_modifier: Optional[LogitsToLogitsFn] = None)` — [`L108`](../../../../../../raw/code/axlearn/axlearn/common/decoder.py#L108) — Computes log probabilities from logits, with an optional modifier.

