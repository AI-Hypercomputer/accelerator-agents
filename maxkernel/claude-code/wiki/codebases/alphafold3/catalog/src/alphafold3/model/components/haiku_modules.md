---
title: 'Module: src/alphafold3/model/components/haiku_modules.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/components/haiku_modules.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.components.haiku_modules`/
symbols:
  Linear: Linear#
  LayerNorm: LayerNorm#
  Linear.__call__: Linear#__call__().
  Linear.output_shape: Linear#output_shape.
  LayerNorm.__call__: LayerNorm#__call__().
  haiku_linear_get_params: haiku_linear_get_params().
  Linear.num_output_dims: Linear#num_output_dims.
  _get_initializer_scale: _get_initializer_scale().
  Linear.num_input_dims: Linear#num_input_dims.
  Linear.precision: Linear#precision.
  Linear.__init__: Linear#__init__().
  TRUNCATED_NORMAL_STDDEV_FACTOR: TRUNCATED_NORMAL_STDDEV_FACTOR.
  LayerNorm.upcast: LayerNorm#upcast.
  Linear.initializer: Linear#initializer.
  DEFAULT_PRECISION: DEFAULT_PRECISION.
  LayerNorm._temp_create_scale: LayerNorm#_temp_create_scale.
  LayerNorm._temp_create_offset: LayerNorm#_temp_create_offset.
  Linear.use_bias: Linear#use_bias.
  Linear.bias_init: Linear#bias_init.
  Linear.fast_scalar_mode: Linear#fast_scalar_mode.
  Linear.transpose_weights: Linear#transpose_weights.
  PRECISION.PRECISION: PRECISION.PRECISION.
  LayerNorm.__init__: LayerNorm#__init__().
---
# Module: [`src/alphafold3/model/components/haiku_modules.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py)

## Classes
### `LayerNorm`
- def: [`src/alphafold3/model/components/haiku_modules.py:50`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L50) — documented in [alphafold3-model-components-haiku_modules](../../../../../concepts/alphafold3-model-components-haiku_modules.md)
- doc: LayerNorm module.
- signature: `class LayerNorm(hk.LayerNorm):`
- members:
  - `upcast` — [`L87`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L87)
- protocol/private: `__call__`[`L91`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L91), `__init__`[`L62`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L62), `_temp_create_offset`[`L89`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L89), `_temp_create_scale`[`L88`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L88)
- used by: [`atom_cross_att_encoder`](../network/atom_cross_attention.md#atom_cross_att_encoder), [`atom_cross_att_decoder`](../network/atom_cross_attention.md#atom_cross_att_decoder), [`__call__`](../network/evoformer.md#Evoformer.__call__), [`__call__`](../network/modules.md#PairFormerIteration.__call__), [`__call__`](../network/confidence_head.md#ConfidenceHead.__call__), [`__call__`](../network/diffusion_head.md#DiffusionHead.__call__), [`construct_input`](../network/template_modules.md#SingleTemplateEmbedding.construct_input), [`_conditioning`](../network/diffusion_head.md#DiffusionHead._conditioning), [`__call__`](../network/modules.md#GridSelfAttention.__call__), [`__call__`](../network/modules.md#OuterProductMean.__call__), [`__call__`](../network/template_modules.md#SingleTemplateEmbedding.__call__), [`__call__`](../network/diffusion_transformer.md#CrossAttTransformer.__call__), [`__call__`](../network/modules.md#TransitionBlock.__call__), [`__call__`](../network/modules.md#TriangleMultiplication.__call__), [`__call__`](../network/modules.md#MSAAttention.__call__), [`__call__`](../network/diffusion_transformer.md#Transformer.__call__), [`adaptive_layernorm`](../network/diffusion_transformer.md#adaptive_layernorm)

### `Linear`
- def: [`src/alphafold3/model/components/haiku_modules.py:197`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L197) — documented in [alphafold3-model-components-haiku_modules](../../../../../concepts/alphafold3-model-components-haiku_modules.md)
- doc: Custom Linear Module.
- signature: `class Linear(hk.Module):`
- members:
  - `__call__(self, inputs: jnp.ndarray)` — [`L253`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L253) — Connects Module. — documented in [alphafold3-model-components-haiku_modules](../../../../../concepts/alphafold3-model-components-haiku_modules.md)
  - `__init__(self, num_output: int | Sequence[int], *, initializer: str = 'linear', num_input_dims: int = 1, use_bias: bool = False, bias_init: float = 0, precision: PRECISION = None, fast_scalar_mode: bool = True, transpose_weights: bool = False, name: str)` — [`L208`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L208) — Constructs Linear Module.
  - `bias_init` — [`L246`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L246) — documented in [alphafold3-model-components-haiku_modules](../../../../../concepts/alphafold3-model-components-haiku_modules.md)
  - `fast_scalar_mode` — [`L250`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L250) — documented in [alphafold3-model-components-haiku_modules](../../../../../concepts/alphafold3-model-components-haiku_modules.md)
  - `initializer` — [`L244`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L244) — documented in [alphafold3-model-components-haiku_modules](../../../../../concepts/alphafold3-model-components-haiku_modules.md)
  - `num_input_dims` — [`L247`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L247) — documented in [alphafold3-model-components-haiku_modules](../../../../../concepts/alphafold3-model-components-haiku_modules.md)
  - `num_output_dims` — [`L248`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L248) — documented in [alphafold3-model-components-haiku_modules](../../../../../concepts/alphafold3-model-components-haiku_modules.md)
  - `output_shape` — [`L241`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L241) — documented in [alphafold3-model-components-haiku_modules](../../../../../concepts/alphafold3-model-components-haiku_modules.md)
  - `precision` — [`L249`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L249) — documented in [alphafold3-model-components-haiku_modules](../../../../../concepts/alphafold3-model-components-haiku_modules.md)
  - `transpose_weights` — [`L251`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L251) — documented in [alphafold3-model-components-haiku_modules](../../../../../concepts/alphafold3-model-components-haiku_modules.md)
  - `use_bias` — [`L245`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L245) — documented in [alphafold3-model-components-haiku_modules](../../../../../concepts/alphafold3-model-components-haiku_modules.md)
- uses (calls/refs, reference-scoped): [`_get_initializer_scale`](haiku_modules.md#_get_initializer_scale), [`TRUNCATED_NORMAL_STDDEV_FACTOR`](haiku_modules.md#TRUNCATED_NORMAL_STDDEV_FACTOR), [`DEFAULT_PRECISION`](haiku_modules.md#DEFAULT_PRECISION), [`PRECISION`](haiku_modules.md#PRECISION.PRECISION)
- used by: [`atom_cross_att_encoder`](../network/atom_cross_attention.md#atom_cross_att_encoder), [`atom_cross_att_decoder`](../network/atom_cross_attention.md#atom_cross_att_decoder), [`__call__`](../network/evoformer.md#Evoformer.__call__), [`__call__`](../network/modules.md#PairFormerIteration.__call__), [`__call__`](../network/confidence_head.md#ConfidenceHead.__call__), [`__call__`](../network/diffusion_head.md#DiffusionHead.__call__), [`construct_input`](../network/template_modules.md#SingleTemplateEmbedding.construct_input), [`_conditioning`](../network/diffusion_head.md#DiffusionHead._conditioning), [`_embed_bonds`](../network/evoformer.md#Evoformer._embed_bonds), [`_embed_process_msa`](../network/evoformer.md#Evoformer._embed_process_msa), [`__call__`](../network/distogram_head.md#DistogramHead.__call__), [`_per_atom_conditioning`](../network/atom_cross_attention.md#_per_atom_conditioning), [`__call__`](../network/modules.md#GridSelfAttention.__call__), [`__call__`](../network/modules.md#OuterProductMean.__call__), [`__call__`](../network/template_modules.md#TemplateEmbedding.__call__), [`self_attention`](../network/diffusion_transformer.md#self_attention), [`_relative_encoding`](../network/evoformer.md#Evoformer._relative_encoding), [`cross_attention`](../network/diffusion_transformer.md#cross_attention), [`__call__`](../network/diffusion_transformer.md#CrossAttTransformer.__call__), [`__call__`](../network/modules.md#TransitionBlock.__call__), [`__call__`](../network/modules.md#TriangleMultiplication.__call__), [`transition_block`](../network/diffusion_transformer.md#transition_block), [`_attention`](../network/modules.md#GridSelfAttention._attention), [`super_block`](../network/diffusion_transformer.md#Transformer.super_block), [`__call__`](../network/modules.md#MSAAttention.__call__), [`_embed_features`](../network/confidence_head.md#ConfidenceHead._embed_features), [`_seq_pair_embedding`](../network/evoformer.md#Evoformer._seq_pair_embedding), [`adaptive_zero_init`](../network/diffusion_transformer.md#adaptive_zero_init), [`adaptive_layernorm`](../network/diffusion_transformer.md#adaptive_layernorm)

## Functions
- `_get_initializer_scale(initializer_name, input_shape)` — [`L327`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L327) — Get initializer for weights. — documented in [alphafold3-model-components-haiku_modules](../../../../../concepts/alphafold3-model-components-haiku_modules.md)
- `haiku_linear_get_params(inputs: jax.Array | jax.ShapeDtypeStruct, *, num_output: int | Sequence[int], use_bias: bool = False, num_input_dims: int = 1, initializer: str = 'linear', bias_init: float = 0, transpose_weights: bool = False, name: str | None = None)` — [`L124`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L124) — Get parameters for linear layer. — documented in [alphafold3-model-network-diffusion_transformer](../../../../../concepts/alphafold3-model-network-diffusion_transformer.md)

## Module values
- `DEFAULT_PRECISION` — [`L42`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L42) — documented in [alphafold3-model-components-haiku_modules](../../../../../concepts/alphafold3-model-components-haiku_modules.md)
- `PRECISION` — [`L33`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L33)
- `TRUNCATED_NORMAL_STDDEV_FACTOR` — [`L45`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/components/haiku_modules.py#L45) — documented in [alphafold3-model-components-haiku_modules](../../../../../concepts/alphafold3-model-components-haiku_modules.md)

