---
title: 'Module: simply/utils/module.py'
type: catalog
provenance: extracted
module: simply/utils/module.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.module`/
symbols:
  EinsumLinear.init: EinsumLinear#init().
  SimplyModule.apply: SimplyModule#apply().
  SimplyModule.init: SimplyModule#init().
  EinsumLinear.apply: EinsumLinear#apply().
  EmbeddingLinear.setup: EmbeddingLinear#setup().
  EmbeddingLinear.embed: EmbeddingLinear#embed().
  EmbeddingLinear.init: EmbeddingLinear#init().
  SimplyModule.setup: SimplyModule#setup().
  EinsumLinear: EinsumLinear#
  SimplyModule: SimplyModule#
  ModuleRegistry: ModuleRegistry#
  EinsumLinear.setup: EinsumLinear#setup().
  SimplyModule.__post_init__: SimplyModule#__post_init__().
  EmbeddingLinear: EmbeddingLinear#
  EinsumLinear.bias_partition: EinsumLinear#bias_partition.
  Array: Array.
  EinsumLinear.output_partition: EinsumLinear#output_partition.
  PyTree: PyTree.
  EinsumLinear.weight_dim_annotation: EinsumLinear#weight_dim_annotation.
  EinsumLinear.bias_dim_annotation: EinsumLinear#bias_dim_annotation.
  EmbeddingLinear.apply: EmbeddingLinear#apply().
  EinsumLinear.bias_term: EinsumLinear#bias_term.
  _reshape_bias: _reshape_bias().
  EinsumLinear.weight_init: EinsumLinear#weight_init.
  EinsumLinear.bias_init: EinsumLinear#bias_init.
  EinsumLinear.weight_name: EinsumLinear#weight_name.
  EinsumLinear.bias_name: EinsumLinear#bias_name.
  EinsumLinear.bias_shape: EinsumLinear#bias_shape.
  EmbeddingLinear.weight_init: EmbeddingLinear#weight_init.
  EmbeddingLinear.bias_init: EmbeddingLinear#bias_init.
  EmbeddingLinear.embed_init: EmbeddingLinear#embed_init.
  EmbeddingLinear.output_partition: EmbeddingLinear#output_partition.
  AnnotatedArray: AnnotatedArray.
  get_raw_arrays: get_raw_arrays.
  EinsumLinear.weight_partition: EinsumLinear#weight_partition.
  EmbeddingLinear.weight_partition: EmbeddingLinear#weight_partition.
  EinsumLinear.weight_shape: EinsumLinear#weight_shape.
  EmbeddingLinear.activation_dtype: EmbeddingLinear#activation_dtype.
  EmbeddingLinear.embed_name: EmbeddingLinear#embed_name.
  create_char_dict: create_char_dict().
  EinsumLinear.activation_dtype: EinsumLinear#activation_dtype.
  EmbeddingLinear.vocab_size: EmbeddingLinear#vocab_size.
  EmbeddingLinear.dim: EmbeddingLinear#dim.
  PRNGKey: PRNGKey.
  _reshape_bias.create_rearranged_bias_term: _reshape_bias().create_rearranged_bias_term().
  EinsumLinear.eqn: EinsumLinear#eqn.
  EinsumLinear.weight_dtype: EinsumLinear#weight_dtype.
  EmbeddingLinear.embedding_scale_by_sqrt_dim: EmbeddingLinear#embedding_scale_by_sqrt_dim.
  EmbeddingLinear.use_tied_embedding: EmbeddingLinear#use_tied_embedding.
  EmbeddingLinear.weight_dtype: EmbeddingLinear#weight_dtype.
  EmbeddingLinear.weight_name: EmbeddingLinear#weight_name.
  EmbeddingLinear.einsum_linear: EmbeddingLinear#einsum_linear.
  _parse_einsum_eqn: _parse_einsum_eqn().
  EinsumLinear.output_term: EinsumLinear#output_term.
  EmbeddingLinear.use_lookup: EmbeddingLinear#use_lookup.
  EmbeddingLinear.use_bias: EmbeddingLinear#use_bias.
  EmbeddingLinear.bias_name: EmbeddingLinear#bias_name.
  SimplyModule.__getattr__: SimplyModule#__getattr__().
  ModuleRegistry.namespace: ModuleRegistry#namespace.
---
# Module: [`simply/utils/module.py`](../../../../../../raw/code/simply/simply/utils/module.py)

## Classes
### `EinsumLinear`
- def: [`simply/utils/module.py:178`](../../../../../../raw/code/simply/simply/utils/module.py#L178)
- doc: An Einsum layer with learnable weights and optional bias.
- signature: `class EinsumLinear(SimplyModule):`
- members:
  - `apply(self, params: PyTree, x: Array)` — [`L324`](../../../../../../raw/code/simply/simply/utils/module.py#L324) — documented in [simply-utils-module](../../../concepts/simply-utils-module.md)
  - `init(self, prng_key: PRNGKey)` — [`L293`](../../../../../../raw/code/simply/simply/utils/module.py#L293) — documented in [simply-utils-module](../../../concepts/simply-utils-module.md)
  - `setup(self)` — [`L237`](../../../../../../raw/code/simply/simply/utils/module.py#L237)
  - `activation_dtype` — [`L227`](../../../../../../raw/code/simply/simply/utils/module.py#L227)
  - `bias_dim_annotation` — [`L271`](../../../../../../raw/code/simply/simply/utils/module.py#L271)
  - `bias_init` — [`L224`](../../../../../../raw/code/simply/simply/utils/module.py#L224)
  - `bias_name` — [`L235`](../../../../../../raw/code/simply/simply/utils/module.py#L235)
  - `bias_partition` — [`L286`](../../../../../../raw/code/simply/simply/utils/module.py#L286)
  - `bias_shape` — [`L282`](../../../../../../raw/code/simply/simply/utils/module.py#L282)
  - `bias_term` — [`L221`](../../../../../../raw/code/simply/simply/utils/module.py#L221)
  - `eqn` — [`L219`](../../../../../../raw/code/simply/simply/utils/module.py#L219)
  - `output_partition` — [`L230`](../../../../../../raw/code/simply/simply/utils/module.py#L230)
  - `output_term` — [`L270`](../../../../../../raw/code/simply/simply/utils/module.py#L270)
  - `weight_dim_annotation` — [`L222`](../../../../../../raw/code/simply/simply/utils/module.py#L222)
  - `weight_dtype` — [`L226`](../../../../../../raw/code/simply/simply/utils/module.py#L226)
  - `weight_init` — [`L223`](../../../../../../raw/code/simply/simply/utils/module.py#L223)
  - `weight_name` — [`L234`](../../../../../../raw/code/simply/simply/utils/module.py#L234)
  - `weight_partition` — [`L229`](../../../../../../raw/code/simply/simply/utils/module.py#L229)
  - `weight_shape` — [`L220`](../../../../../../raw/code/simply/simply/utils/module.py#L220)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`with_sharding_constraint`](sharding.md#with_sharding_constraint), [`SimplyModule`](module.md#SimplyModule), [`ModuleRegistry`](module.md#ModuleRegistry), [`convert_or_dequantize`](common.md#convert_or_dequantize), [`Initializer`](initializer.md#Initializer), [`create`](common.md#AnnotatedArray.create), [`XavierUniformInit`](initializer.md#XavierUniformInit), [`PartitionAnnotation`](common.md#PartitionAnnotation.PartitionAnnotation), [`Array`](module.md#Array), [`ZeroInit`](initializer.md#ZeroInit), [`PyTree`](module.md#PyTree), [`_reshape_bias`](module.md#_reshape_bias), [`NOT_ANNOTATED`](sharding.md#NOT_ANNOTATED), [`AnnotatedArray`](module.md#AnnotatedArray), [`get_raw_arrays`](module.md#get_raw_arrays), [`create_char_dict`](module.md#create_char_dict), [`PRNGKey`](module.md#PRNGKey), [`_parse_einsum_eqn`](module.md#_parse_einsum_eqn)
- used by: [`apply`](module.md#SimplyModule.apply), [`init`](module.md#SimplyModule.init), [`setup`](module.md#EmbeddingLinear.setup), [`setup`](module.md#SimplyModule.setup), [`v_proj`](../model_lib.md#Attention.v_proj), [`EinsumLinear`](../model_lib.md#EinsumLinear), [`test_dump`](module_test.md#EinsumLinearTest.test_dump), [`k_proj`](../model_lib.md#Attention.k_proj), [`o_proj`](../model_lib.md#Attention.o_proj), [`q_proj`](../model_lib.md#Attention.q_proj), [`test_gmoe_routing`](module_test.md#EinsumLinearTest.test_gmoe_routing), [`test_classic_linear`](module_test.md#EinsumLinearTest.test_classic_linear), [`test_classic_linear2`](module_test.md#EinsumLinearTest.test_classic_linear2), [`test_combined_qkv`](module_test.md#EinsumLinearTest.test_combined_qkv), [`test_gmoe_linear`](module_test.md#EinsumLinearTest.test_gmoe_linear), [`test_gmoe_linear2`](module_test.md#EinsumLinearTest.test_gmoe_linear2), [`test_mha_o`](module_test.md#EinsumLinearTest.test_mha_o), [`test_mha_qkv`](module_test.md#EinsumLinearTest.test_mha_qkv)

### `EmbeddingLinear`
- def: [`simply/utils/module.py:345`](../../../../../../raw/code/simply/simply/utils/module.py#L345)
- doc: A EinsumLinear layer that also supports embedding lookup.
- signature: `class EmbeddingLinear(SimplyModule):`
- members:
  - `apply(self, params: PyTree, x: Array)` — [`L468`](../../../../../../raw/code/simply/simply/utils/module.py#L468)
  - `embed(self, params: PyTree, x: Array)` — [`L435`](../../../../../../raw/code/simply/simply/utils/module.py#L435) — Embeds token IDs into embedding vectors.
  - `init(self, prng_key: PRNGKey)` — [`L420`](../../../../../../raw/code/simply/simply/utils/module.py#L420)
  - `setup(self)` — [`L401`](../../../../../../raw/code/simply/simply/utils/module.py#L401)
  - `activation_dtype` — [`L390`](../../../../../../raw/code/simply/simply/utils/module.py#L390)
  - `bias_init` — [`L384`](../../../../../../raw/code/simply/simply/utils/module.py#L384)
  - `bias_name` — [`L398`](../../../../../../raw/code/simply/simply/utils/module.py#L398)
  - `dim` — [`L378`](../../../../../../raw/code/simply/simply/utils/module.py#L378)
  - `einsum_linear` — [`L402`](../../../../../../raw/code/simply/simply/utils/module.py#L402)
  - `embed_init` — [`L387`](../../../../../../raw/code/simply/simply/utils/module.py#L387)
  - `embed_name` — [`L399`](../../../../../../raw/code/simply/simply/utils/module.py#L399)
  - `embedding_scale_by_sqrt_dim` — [`L380`](../../../../../../raw/code/simply/simply/utils/module.py#L380)
  - `output_partition` — [`L393`](../../../../../../raw/code/simply/simply/utils/module.py#L393)
  - `use_bias` — [`L382`](../../../../../../raw/code/simply/simply/utils/module.py#L382)
  - `use_lookup` — [`L379`](../../../../../../raw/code/simply/simply/utils/module.py#L379)
  - `use_tied_embedding` — [`L385`](../../../../../../raw/code/simply/simply/utils/module.py#L385)
  - `vocab_size` — [`L377`](../../../../../../raw/code/simply/simply/utils/module.py#L377)
  - `weight_dtype` — [`L389`](../../../../../../raw/code/simply/simply/utils/module.py#L389)
  - `weight_init` — [`L381`](../../../../../../raw/code/simply/simply/utils/module.py#L381)
  - `weight_name` — [`L397`](../../../../../../raw/code/simply/simply/utils/module.py#L397)
  - `weight_partition` — [`L392`](../../../../../../raw/code/simply/simply/utils/module.py#L392)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`with_sharding_constraint`](sharding.md#with_sharding_constraint), [`EinsumLinear`](module.md#EinsumLinear), [`SimplyModule`](module.md#SimplyModule), [`ModuleRegistry`](module.md#ModuleRegistry), [`convert_or_dequantize`](common.md#convert_or_dequantize), [`Initializer`](initializer.md#Initializer), [`PartitionAnnotation`](common.md#PartitionAnnotation.PartitionAnnotation), [`LecunNormalInit`](initializer.md#LecunNormalInit), [`Array`](module.md#Array), [`ZeroInit`](initializer.md#ZeroInit), [`PyTree`](module.md#PyTree), [`NOT_ANNOTATED`](sharding.md#NOT_ANNOTATED), [`get_raw_arrays`](module.md#get_raw_arrays), [`PRNGKey`](module.md#PRNGKey)
- used by: [`apply`](module.md#SimplyModule.apply), [`init`](module.md#SimplyModule.init), [`setup`](module.md#SimplyModule.setup), [`embed_linear`](../model_lib.md#TransformerLM.embed_linear), [`test_embedding_linear_apply`](module_test.md#EmbeddingLinearTest.test_embedding_linear_apply), [`test_embedding_linear_embed`](module_test.md#EmbeddingLinearTest.test_embedding_linear_embed), [`test_embedding_linear_tied_embedding`](module_test.md#EmbeddingLinearTest.test_embedding_linear_tied_embedding), [`test_embedding_linear_untied_embedding`](module_test.md#EmbeddingLinearTest.test_embedding_linear_untied_embedding)

### `ModuleRegistry`
- def: [`simply/utils/module.py:71`](../../../../../../raw/code/simply/simply/utils/module.py#L71)
- members:
  - `namespace` — [`L74`](../../../../../../raw/code/simply/simply/utils/module.py#L74)
- uses (calls/refs, reference-scoped): [`RootRegistry`](registry.md#RootRegistry)
- used by: [`TransformerLM`](../model_lib.md#TransformerLM), [`EinsumLinear`](module.md#EinsumLinear), [`test_dump_load_baseline_config`](../config_lib_test.md#ConfigLibTest.test_dump_load_baseline_config), [`LayerNorm`](../model_lib.md#LayerNorm), [`EmbeddingLinear`](module.md#EmbeddingLinear), [`__post_init__`](module.md#SimplyModule.__post_init__), [`MoEFeedForward`](../model_lib.md#MoEFeedForward), [`AnnotatedArray`](vision_encoder.md#AnnotatedArray), [`FeedForward`](../model_lib.md#FeedForward), [`TransformerBlock`](../model_lib.md#TransformerBlock), [`create_model`](../model_lib.md#create_model), [`Attention`](../model_lib.md#Attention), [`ControlFlow`](control_flow.md#ControlFlow), [`PerDimScale`](../model_lib.md#PerDimScale), [`ScanModule`](control_flow.md#ScanModule), [`VisionTransformer`](vision_encoder.md#VisionTransformer), [`get_scaling_info`](../model_lib.md#get_scaling_info)

### `SimplyModule`  ·  implements/extends ABC
- def: [`simply/utils/module.py:38`](../../../../../../raw/code/simply/simply/utils/module.py#L38)
- doc: An ultra-simplified version of `flax.nn.Module`.
- signature: `class SimplyModule(abc.ABC):`
- members:
  - `apply(self, params: PyTree, x: Any, **kwargs: Any)` — [`L67`](../../../../../../raw/code/simply/simply/utils/module.py#L67) — Run forward pass of the module with parameters and inputs. — documented in [simply-utils-module](../../../concepts/simply-utils-module.md)
  - `init(self, prng_key: jax.Array)` — [`L63`](../../../../../../raw/code/simply/simply/utils/module.py#L63) — initialize the parameters associated with the module. — documented in [simply-utils-module](../../../concepts/simply-utils-module.md)
  - `setup(self)` — [`L60`](../../../../../../raw/code/simply/simply/utils/module.py#L60) — Setup any attributes. Typically used for instantiating sub-modules.
- protocol/private: `__getattr__`[`L55`](../../../../../../raw/code/simply/simply/utils/module.py#L55), `__post_init__`[`L42`](../../../../../../raw/code/simply/simply/utils/module.py#L42)
- uses (calls/refs, reference-scoped): [`apply`](../model_lib.md#Attention.apply), [`apply`](../model_lib.md#TransformerLM.apply), [`init`](module.md#EinsumLinear.init), [`apply`](../model_lib.md#TransformerBlock.apply), [`get`](registry.md#RootRegistry.get), [`apply`](module.md#EinsumLinear.apply), [`init`](../model_lib.md#TransformerBlock.init), [`init`](../model_lib.md#LayerNorm.init), [`setup`](module.md#EmbeddingLinear.setup), [`apply`](control_flow.md#ControlFlow.apply), [`apply`](vision_encoder.md#VisionTransformer.apply), [`init`](module.md#EmbeddingLinear.init), [`apply`](../model_lib.md#FeedForward.apply), [`apply`](../model_lib.md#LayerNorm.apply), [`init`](vision_encoder.md#PatchEncoder.init), [`apply`](vision_encoder.md#PatchEncoder.apply), [`init`](../model_lib.md#Attention.init), [`ModuleRegistry`](module.md#ModuleRegistry), [`init`](../model_lib.md#TransformerLM.init), [`init`](vision_encoder.md#VisionTransformer.init), [`init`](../model_lib.md#FeedForward.init), [`init`](../model_lib.md#PerDimScale.init), [`setup`](module.md#EinsumLinear.setup), [`apply`](../model_lib.md#PerDimScale.apply), [`apply`](control_flow.md#ScanModule.apply), [`apply`](../model_lib.md#InputEncoderInterface.apply), [`fullname`](registry.md#RootRegistry.fullname), [`init`](control_flow.md#ScanModule.init), [`PyTree`](module.md#PyTree), [`apply`](module.md#EmbeddingLinear.apply), [`init`](control_flow.md#ControlFlow.init), [`setup`](../model_lib.md#TransformerLM.setup), [`setup`](control_flow.md#ScanModule.setup), [`InputEncoderInterface`](../model_lib.md#InputEncoderInterface), [`setup`](../model_lib.md#Attention.setup), [`setup`](../model_lib.md#TransformerBlock.setup), [`setup`](../model_lib.md#FeedForward.setup), [`setup`](control_flow.md#ControlFlow.setup), [`setup`](vision_encoder.md#VisionTransformer.setup)
- used by: [`score_tokens`](../model_lib.md#LMInterface.score_tokens), [`TransformerLM`](../model_lib.md#TransformerLM), [`EinsumLinear`](module.md#EinsumLinear), [`init`](../model_lib.md#TransformerLM.init), [`LayerNorm`](../model_lib.md#LayerNorm), [`_process_module`](control_flow.md#ScanModule._process_module), [`get_abstract_params`](checkpoint_lib.md#get_abstract_params), [`EmbeddingLinear`](module.md#EmbeddingLinear), [`__init__`](../model_lib.md#LMInterface.__init__), [`init`](control_flow.md#ScanModule.init), [`FeedForward`](../model_lib.md#FeedForward), [`TransformerBlock`](../model_lib.md#TransformerBlock), [`modules`](control_flow.md#ControlFlow.modules), [`Attention`](../model_lib.md#Attention), [`ControlFlow`](control_flow.md#ControlFlow), [`PerDimScale`](../model_lib.md#PerDimScale), [`ScanModule`](control_flow.md#ScanModule), [`decode_fn`](../model_lib.md#LMInterface.decode_fn), [`prefill_fn`](../model_lib.md#LMInterface.prefill_fn), [`VisionTransformer`](vision_encoder.md#VisionTransformer), [`module`](control_flow.md#ScanModule.module), [`setup`](control_flow.md#ScanModule.setup), [`InputEncoderInterface`](../model_lib.md#InputEncoderInterface), [`PatchEncoder`](vision_encoder.md#PatchEncoder), [`SimplyModule`](../model_lib.md#SimplyModule), [`module`](control_flow.md#ControlStep.module)

## Functions
- `_parse_einsum_eqn(eqn: str)` — [`L123`](../../../../../../raw/code/simply/simply/utils/module.py#L123) — Parses einsum equation into weight, input, and output strings.
- `_reshape_bias(bias: Array, *, output_term: str, bias_term: str, output_shape: Sequence[int])` — [`L77`](../../../../../../raw/code/simply/simply/utils/module.py#L77) — Reshapes bias tensor to be broadcastable to output tensor.
- `create_char_dict(term: str, seq: Sequence[Any] | None)` — [`L139`](../../../../../../raw/code/simply/simply/utils/module.py#L139) — Creates a dictionary mapping dimension characters to sequence elements.
- `create_rearranged_bias_term(string, bias_term)` — [`L100`](../../../../../../raw/code/simply/simply/utils/module.py#L100)

## Module values
- `AnnotatedArray` — [`L32`](../../../../../../raw/code/simply/simply/utils/module.py#L32)
- `Array` — [`L34`](../../../../../../raw/code/simply/simply/utils/module.py#L34)
- `PRNGKey` — [`L35`](../../../../../../raw/code/simply/simply/utils/module.py#L35)
- `PyTree` — [`L31`](../../../../../../raw/code/simply/simply/utils/module.py#L31)
- `get_raw_arrays` — [`L33`](../../../../../../raw/code/simply/simply/utils/module.py#L33)

