---
title: 'Module: easydel/infra/mixins/protocol.py'
type: catalog
provenance: extracted
module: easydel/infra/mixins/protocol.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.mixins.protocol`/
symbols:
  BaseModuleProtocol: BaseModuleProtocol#
  BaseModuleProtocol.mesh_call: BaseModuleProtocol#mesh_call().
  BaseModuleProtocol.__call__: BaseModuleProtocol#__call__().
  Tokens: Tokens.
  AttentionMask: AttentionMask.
  PositionIds: PositionIds.
  TokenEmbeds: TokenEmbeds.
  RecurrentState: RecurrentState.
  get_module_repr: get_module_repr().
  TokenTypeIds: TokenTypeIds.
  PixelValues: PixelValues.
  Self: Self.
  HeadMask: HeadMask.
  BaseModuleProtocol.compute_loss: BaseModuleProtocol#compute_loss().
  prettify_nnx: prettify_nnx().
  CachePosition: CachePosition.
  AudioFeatures: AudioFeatures.
  BaseModuleProtocol.quantize: BaseModuleProtocol#quantize().
  BaseModuleProtocol.prepare_inputs_for_generation: BaseModuleProtocol#prepare_inputs_for_generation().
  printify_nnx: printify_nnx().
  EncoderHiddenStates: EncoderHiddenStates.
  DecoderInputIds: DecoderInputIds.
  return_type_adjuster.decorator: return_type_adjuster().decorator().
  Labels: Labels.
  _T: _T.
  return_type_adjuster: return_type_adjuster().
  BaseModuleProtocol.to_dtype: BaseModuleProtocol#to_dtype().
  BaseModuleProtocol._get_partition_rules: BaseModuleProtocol#_get_partition_rules().
  BaseModuleProtocol.shard_model: BaseModuleProtocol#shard_model().
  BaseModuleProtocol.gather_model: BaseModuleProtocol#gather_model().
  BaseModuleProtocol.to_state: BaseModuleProtocol#to_state().
  BaseModuleProtocol.lazy_init: BaseModuleProtocol#lazy_init().
  BaseModuleProtocol.apply_lora_to_layers: BaseModuleProtocol#apply_lora_to_layers().
  BaseModuleProtocol.merge_lora_params: BaseModuleProtocol#merge_lora_params().
  BaseModuleProtocol.split_lora_params: BaseModuleProtocol#split_lora_params().
  BaseModuleProtocol.generate: BaseModuleProtocol#generate().
  PartitionLike: PartitionLike.
  EncoderAttentionMask: EncoderAttentionMask.
  DecoderPositionIds: DecoderPositionIds.
  VideoPixelValues: VideoPixelValues.
  ImageSizes: ImageSizes.
  VisualPosMasks: VisualPosMasks.
  DeepstackVisualEmbeds: DeepstackVisualEmbeds.
  RopeDeltas: RopeDeltas.
  LogSnr: LogSnr.
  NoiseMask: NoiseMask.
  return_type_adjuster.decorator.wrapper: return_type_adjuster().decorator().wrapper().
  _CP: _CP.
  BaseModuleProtocol.config_class: BaseModuleProtocol#config_class.
  BaseModuleProtocol.config: BaseModuleProtocol#config.
  BaseModuleProtocol.graphdef: BaseModuleProtocol#graphdef().
  BaseModuleProtocol.graphstate: BaseModuleProtocol#graphstate().
  BaseModuleProtocol.graphother: BaseModuleProtocol#graphother().
  BaseModuleProtocol.half: BaseModuleProtocol#half().
  BaseModuleProtocol.float: BaseModuleProtocol#float().
  BaseModuleProtocol._reformat_dtype: BaseModuleProtocol#_reformat_dtype().
  BaseModuleProtocol._get_mesh: BaseModuleProtocol#_get_mesh().
  BaseModuleProtocol._apply_sharding_fns: BaseModuleProtocol#_apply_sharding_fns().
  BaseModuleProtocol._shard_fns: BaseModuleProtocol#_shard_fns().
  BaseModuleProtocol.to_torch: BaseModuleProtocol#to_torch().
  BaseModuleProtocol.prepare_inputs_for_call: BaseModuleProtocol#prepare_inputs_for_call().
  BaseModuleProtocol.get_static_arguments: BaseModuleProtocol#get_static_arguments().
  BaseModuleProtocol.unwrap_lora_to_layers: BaseModuleProtocol#unwrap_lora_to_layers().
  BaseModuleProtocol.transform_fn: BaseModuleProtocol#transform_fn().
  BaseModuleProtocol.pure_transform_fn: BaseModuleProtocol#pure_transform_fn().
  BaseModuleProtocol.params_sharding: BaseModuleProtocol#params_sharding().
  BaseModuleProtocol.merge_params: BaseModuleProtocol#merge_params().
  BaseModuleProtocol.split_params: BaseModuleProtocol#split_params().
  BaseModuleProtocol.split_params_dict: BaseModuleProtocol#split_params_dict().
  BaseModuleProtocol.merge_params_dict: BaseModuleProtocol#merge_params_dict().
  BaseModuleProtocol._flop: BaseModuleProtocol#_flop().
  BaseModuleProtocol.__str__: BaseModuleProtocol#__str__().
  BaseModuleProtocol.__repr__: BaseModuleProtocol#__repr__().
  AnyArray: AnyArray.
  SegmentIds: SegmentIds.
  DecoderAttentionMask: DecoderAttentionMask.
  BaseModuleProtocol.base_model_prefix: BaseModuleProtocol#base_model_prefix.
  BaseModuleProtocol._model_task: BaseModuleProtocol#_model_task.
  BaseModuleProtocol._model_type: BaseModuleProtocol#_model_type.
  BaseModuleProtocol.save_pretrained: BaseModuleProtocol#save_pretrained().
  BaseModuleProtocol.push_to_hub: BaseModuleProtocol#push_to_hub().
  BaseModuleProtocol.from_pretrained: BaseModuleProtocol#from_pretrained().
  BaseModuleProtocol.can_generate: BaseModuleProtocol#can_generate().
  BaseModuleProtocol.get_torch_loader: BaseModuleProtocol#get_torch_loader().
  BaseModuleProtocol.init_cache: BaseModuleProtocol#init_cache().
  BaseModuleProtocol.init_ragged_pages: BaseModuleProtocol#init_ragged_pages().
  BaseModuleProtocol.get_inference_cache_type: BaseModuleProtocol#get_inference_cache_type().
  BaseModuleProtocol.update_inputs_for_generation: BaseModuleProtocol#update_inputs_for_generation().
---
# Module: [`easydel/infra/mixins/protocol.py`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py)

## Classes
### `BaseModuleProtocol`
- def: [`easydel/infra/mixins/protocol.py:353`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L353)
- doc: Protocol defining the common interface for EasyDeL modules.
- signature: `class BaseModuleProtocol(metaclass=ABCMeta):`
- members:
  - `__call__(self, input_ids: Tokens | None = None, inputs_embeds: TokenEmbeds | None = None, attention_mask: AttentionMask | None = None, mask_info: MaskInfo | None = None, position_ids: PositionIds | None = None, mode: common_types.RUNTIME_MODE_TYPES | None = None, past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None, cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None, output_attentions: bool | None = None, output_hidden_states: bool | None = None)` — [`L435`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L435) — Forward pass for decoder-only backbones without a task head.
  - `__repr__(self)` — [`L3607`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3607) — Return a detailed string representation of the model.
  - `__str__(self)` — [`L3598`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3598) — Return a human-readable string representation of the model.
  - `_apply_sharding_fns(self, sharding_fns: collections.abc.Mapping[str, tp.Callable])` — [`L3050`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3050) — Apply sharding functions to the model's state.
  - `_flop(self, *args, **kwargs)` — [`L3357`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3357) — Calculate the FLOP count from JaxPr.
  - `_get_mesh(self, mesh: Mesh | None = None)` — [`L3021`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3021) — Retrieve the JAX mesh for distributed computation.
  - `_get_partition_rules(self, partition_rules: PartitionLike)` — [`L3035`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3035) — Retrieve partition rules for model sharding.
  - `_reformat_dtype(self, dtype)` — [`L3007`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3007) — Internal method to convert model parameters to a given data type.
  - `_shard_fns(self)` — [`L3095`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3095) — property shard functions for model state and parameters.
  - `apply_lora_to_layers(self: Self, lora_rank: int, lora_pattern: str | None = None, verbose: bool = False, rngs: nn.Rngs | None = None)` — [`L3191`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3191) — Apply LoRA (Low-Rank Adaptation) to specified linear layers.
  - `can_generate(cls)` — [`L3456`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3456) — Checks if the model can generate sequences with `.generate()`.
  - `compute_loss(self, *, labels: Labels | None = None, loss_config: LossConfig | None = None, loss_kwargs: dict | None = None, **batch)` — [`L2906`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L2906) — Compute loss for a forward pass.
  - `float(self, change_runtime_dtype: bool = True)` — [`L2995`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L2995) — Convert model parameters to float32 (full precision).
  - `from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, **kwargs)` — [`L3438`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3438) — Loads an EasyDeL model from a pretrained model or path.
  - `gather_model(self, partition_rules: PartitionLike = None, mesh: Mesh | None = None)` — [`L3078`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3078) — Gathers the model's parameters based on the specified partitioning rules and mesh.
  - `generate(self, input_ids: Tokens, generation_config: tp.Any | None = None, prng_key: AnyArray | None = None, trace: bool = True, logits_processor: tp.Any | None = None, **kwargs)` — [`L3475`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3475) — Generates sequences of token ids for models with a language modeling head.
  - `get_inference_cache_type(self)` — [`L3544`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3544) — Determine the appropriate cache type for inference.
  - `get_static_arguments(self)` — [`L3159`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3159) — Get static arguments for JIT compilation.
  - `get_torch_loader(cls)` — [`L3466`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3466) — Gets the appropriate PyTorch AutoModel loader for this model type.
  - `graphdef(self)` — [`L2933`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L2933) — Returns the static graph definition of the model.
  - `graphother(self)` — [`L2959`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L2959) — Returns the non-trainable state of the model.
  - `graphstate(self)` — [`L2946`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L2946) — Returns the trainable state (parameters) of the model.
  - `half(self, change_runtime_dtype: bool = True)` — [`L2983`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L2983) — Convert model parameters to float16 (half precision).
  - `init_cache(self, batch_size: int, max_length: int, starts: int | None = None, shardings: dict | None = None, pad_token_id: int | None = None)` — [`L3500`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3500) — Initializes and returns a standard (non-paged) Key-Value cache.
  - `init_ragged_pages(self, metadata: tp.Any | None = None, page_size: int | None = None, hbm_utilization: float | None = None, max_model_length: int | None = None)` — [`L3523`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3523) — Initializes and returns the actual Paged Attention KV Cache tensors.
  - `lazy_init(cls: type[Self], *args, **kwargs)` — [`L3173`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3173) — Initialize the model lazily using nnx.eval_shape.
  - `merge_lora_params(self: Self, pytree: dict)` — [`L3221`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3221) — Merge LoRA parameters into the base model parameters.
  - `merge_params(self, tree)` — [`L3305`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3305) — Merge a parameter tree into the current model.
  - `merge_params_dict(self, params_dict: dict)` — [`L3346`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3346) — Merge model parameters from a dictionary into the current model.
  - `mesh_call(self, input_ids: Tokens | None = None, inputs_embeds: TokenEmbeds | None = None, attention_mask: AttentionMask | None = None, mask_info: MaskInfo | None = None, position_ids: PositionIds | None = None, mode: common_types.RUNTIME_MODE_TYPES | None = None, past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None, cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None, output_attentions: bool | None = None, output_hidden_states: bool | None = None)` — [`L1634`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L1634) — Run the forward pass inside `self.mesh` (auto/normal mesh).
  - `params_sharding(self)` — [`L3292`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3292) — Get the sharding specification for model parameters.
  - `prepare_inputs_for_call(self, **kwargs)` — [`L3143`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3143) — Prepare and validate inputs before calling the model.
  - `prepare_inputs_for_generation(self, input_ids: Tokens, max_length: int, pad_token_id: int, starts: int | None = None, shardings: int | None = None, attention_mask: AttentionMask | None = None, token_type_ids: TokenTypeIds | None = None, mask_info: tp.Any | None = None)` — [`L3553`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3553) — Sets up the initial inputs required for starting autoregressive generation.
  - `pure_transform_fn(self)` — [`L3279`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3279) — Get a pure transform function for PyTorch to EasyDeL conversion.
  - `push_to_hub(self, repo_id: str, use_temp_dir: bool | None = None, commit_message: str | None = None, private: bool | None = None, token: bool | str | None = None, create_pr: bool = False, float_dtype: jnp.dtype | None = None, verbose: bool = True, mismatch_allowed: bool = True, revision: str | None = None, commit_description: str | None = None, upload_num_threads: int | None = None)` — [`L3399`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3399) — Pushes the model to the Hugging Face Hub.
  - `quantize(self: Self, quantization_config: QuantizationConfig | None = None, apply_quantization: bool = True, verbose: bool | None = None)` — [`L3099`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3099) — Quantizes the model's linear layers.
  - `save_pretrained(self, save_directory: str | os.PathLike, push_to_hub: bool = False, token: str | bool | None = None, float_dtype: jnp.dtype | None = None, step: int | None = None, upload_num_threads: int | None = None, **kwargs)` — [`L3374`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3374) — Saves the model, its configuration, and optionally pushes it to the Hugging Face Hub.
  - `shard_model(self, partition_rules: PartitionLike = None, mesh: Mesh | None = None)` — [`L3062`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3062) — Shards the model's parameters using the specified partitioning rules and mesh.
  - `split_lora_params(self: Self)` — [`L3237`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3237) — Split LoRA parameters from the base model parameters.
  - `split_params(self)` — [`L3317`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3317) — Split the model into its parameter tree.
  - `split_params_dict(self, params_dict: dict)` — [`L3329`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3329) — Split model parameters from a dictionary into state components.
  - `to_dtype(self: Self, dtype)` — [`L2971`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L2971) — Convert model parameters to the specified data type.
  - `to_state(self)` — [`L3117`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3117) — Convert the current model to an EasyDeLState object.
  - `to_torch(self)` — [`L3130`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3130) — Convert the current EasyDeL model to a HuggingFace PyTorch model.
  - `transform_fn(self)` — [`L3265`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3265) — Get the transform function for converting PyTorch to EasyDeL module.
  - `unwrap_lora_to_layers(self, verbose: bool = False)` — [`L3250`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3250) — Remove LoRA adapters from linear layers within the model.
  - `update_inputs_for_generation(self, model_outputs: tp.Any, model_kwargs: dict[str, tp.Any])` — [`L3582`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3582) — Updates the keyword arguments for the next generation step.
  - `base_model_prefix` — [`L430`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L430)
  - `config` — [`L429`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L429)
  - `config_class` — [`L428`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L428) — ---
- protocol/private: `_model_task`[`L431`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L431), `_model_type`[`L432`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L432)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../base_module.md#EasyDeLBaseModule), [`EasyDeLBaseConfig`](../base_config.md#EasyDeLBaseConfig), [`OperationsMetadata`](../../caching/_abstracts.md#OperationsMetadata), [`RaggedPagesMetadata`](../../caching/ragged_page/cache.md#RaggedPagesMetadata), [`TransformerMetadata`](../../caching/transformer/cache.md#TransformerMetadata), [`TransformerCache`](../../caching/transformer/cache.md#TransformerCache), [`EasyDeLState`](../base_state.md#EasyDeLState), [`HybridCache`](../../caching/hybrid/cache.md#HybridCache), [`RaggedPagesCache`](../../caching/ragged_page/cache.md#RaggedPagesCache), [`BaseModelOutput`](../modeling_outputs.md#BaseModelOutput), [`ModelOutput`](../modeling_outputs.md#ModelOutput), [`LossMetrics`](../loss_utils.md#LossMetrics), [`LossConfig`](../loss_utils.md#LossConfig), [`SequenceClassifierOutput`](../modeling_outputs.md#SequenceClassifierOutput), [`QuantizationConfig`](../../layers/quantization/_configs.md#QuantizationConfig), [`CausalLMOutput`](../modeling_outputs.md#CausalLMOutput), [`Tokens`](protocol.md#Tokens), [`AttentionMask`](protocol.md#AttentionMask), [`RecurrentCache`](../../caching/recurrent/cache.md#RecurrentCache), [`MoeModelOutput`](../modeling_outputs.md#MoeModelOutput), [`VLMCausalLMOutput`](../modeling_outputs.md#VLMCausalLMOutput), [`PositionIds`](protocol.md#PositionIds), [`to_state`](../base_module.md#EasyDeLBaseModule.to_state), [`MoeCausalLMOutput`](../modeling_outputs.md#MoeCausalLMOutput), [`TokenEmbeds`](protocol.md#TokenEmbeds), [`compute_loss`](../base_module.md#EasyDeLBaseModule.compute_loss), [`_get_partition_rules`](../base_module.md#EasyDeLBaseModule._get_partition_rules), [`RecurrentState`](protocol.md#RecurrentState), [`BaseModelOutputWithPooling`](../modeling_outputs.md#BaseModelOutputWithPooling), [`shard_model`](../base_module.md#EasyDeLBaseModule.shard_model), [`quantize`](../base_module.md#EasyDeLBaseModule.quantize), [`to_torch`](../base_module.md#EasyDeLBaseModule.to_torch), [`TokenTypeIds`](protocol.md#TokenTypeIds), [`gather_model`](../base_module.md#EasyDeLBaseModule.gather_model), [`ImageClassifierOutput`](../modeling_outputs.md#ImageClassifierOutput), [`_get_mesh`](../base_module.md#EasyDeLBaseModule._get_mesh), [`Seq2SeqLMOutput`](../modeling_outputs.md#Seq2SeqLMOutput), [`BaseModelOutputWithPastAndCrossAttentions`](../modeling_outputs.md#BaseModelOutputWithPastAndCrossAttentions), [`QuestionAnsweringModelOutput`](../modeling_outputs.md#QuestionAnsweringModelOutput), [`TokenClassifierOutput`](../modeling_outputs.md#TokenClassifierOutput)  (+52 more)
- used by: [`EasyDeLBaseModule`](../base_module.md#EasyDeLBaseModule)

## Functions
- `decorator(func: tp.Callable[..., nn.Module])` — [`L183`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L183)
- `get_module_repr(module: nn.Module)` — [`L192`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L192) — Get a string representation of module parameters.
- `prettify_nnx(module: nn.Module, indent: str = "", depth: int = 0, max_depth: int | None = None, module_param=None)` — [`L253`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L253) — Format the structure of a Flax NNX module for display.
- `printify_nnx(model: nn.Module)` — [`L3617`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L3617) — Create a printable string representation of an EasyDeL NNX module.
- `return_type_adjuster(original_return_type: type[_T])` — [`L159`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L159) — Decorator factory to adjust return type annotations for type checking.
- `wrapper(*args: tp.Any, **kwargs: tp.Any)` — [`L184`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L184)

## Module values
- `AnyArray` — [`L118`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L118)
- `AttentionMask` — [`L121`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L121)
- `AudioFeatures` — [`L140`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L140)
- `CachePosition` — [`L137`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L137)
- `DecoderAttentionMask` — [`L129`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L129)
- `DecoderInputIds` — [`L128`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L128)
- `DecoderPositionIds` — [`L130`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L130)
- `DeepstackVisualEmbeds` — [`L135`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L135)
- `EncoderAttentionMask` — [`L127`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L127)
- `EncoderHiddenStates` — [`L126`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L126)
- `HeadMask` — [`L125`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L125)
- `ImageSizes` — [`L133`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L133)
- `Labels` — [`L141`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L141)
- `LogSnr` — [`L138`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L138)
- `NoiseMask` — [`L139`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L139)
- `PartitionLike` — [`L113`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L113)
- `PixelValues` — [`L131`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L131)
- `PositionIds` — [`L122`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L122)
- `RecurrentState` — [`L142`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L142)
- `RopeDeltas` — [`L136`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L136)
- `SegmentIds` — [`L123`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L123)
- `Self` — [`L116`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L116)
- `TokenEmbeds` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L120)
- `TokenTypeIds` — [`L124`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L124)
- `Tokens` — [`L119`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L119)
- `VideoPixelValues` — [`L132`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L132)
- `VisualPosMasks` — [`L134`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L134)
- `_CP` — [`L114`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L114)
- `_T` — [`L115`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/protocol.py#L115)

