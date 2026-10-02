---
title: 'Module: axlearn/common/quantizer.py'
type: catalog
provenance: extracted
module: axlearn/common/quantizer.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.quantizer`/
symbols:
  _einsum_dims: _einsum_dims.
  compute_code_histogram: compute_code_histogram().
  compute_code_pplx: compute_code_pplx().
  compute_code_coverage: compute_code_coverage().
  BaseQuantizer: BaseQuantizer#
  BaseQuantizer.Config: BaseQuantizer#Config#
  BaseQuantizer.Config.codebook_dim: BaseQuantizer#Config#codebook_dim.
  BaseQuantizer.Config.codebook_size: BaseQuantizer#Config#codebook_size.
  BaseQuantizer.Config.num_codebooks: BaseQuantizer#Config#num_codebooks.
  BaseQuantizer.Output: BaseQuantizer#Output#
  BaseQuantizer.Output.ids: BaseQuantizer#Output#ids.
  BaseQuantizer.Output.quantized_vectors: BaseQuantizer#Output#quantized_vectors.
  BaseQuantizer.Output.loss: BaseQuantizer#Output#loss.
  BaseQuantizer.default_config: BaseQuantizer#default_config().
  BaseQuantizer._create_layer_parameter_specs: BaseQuantizer#_create_layer_parameter_specs().
  BaseQuantizer.forward: BaseQuantizer#forward().
  BaseQuantizer.lookup: BaseQuantizer#lookup().
  _lookup: _lookup().
  SimilarityMetric: SimilarityMetric#
  SimilarityMetric.L2_DISTANCE: SimilarityMetric#L2_DISTANCE.
  SimilarityMetric.DOT_PRODUCT: SimilarityMetric#DOT_PRODUCT.
  quantize_by_nearest_neighbor: quantize_by_nearest_neighbor().
  _apply_paddings: _apply_paddings().
  _ids_to_onehots: _ids_to_onehots().
  _add_codebook_summaries: _add_codebook_summaries().
  RandomVectorQuantizer: RandomVectorQuantizer#
  RandomVectorQuantizer.Config: RandomVectorQuantizer#Config#
  RandomVectorQuantizer.Config.input_dim: RandomVectorQuantizer#Config#input_dim.
  RandomVectorQuantizer.Config.rand_proj: RandomVectorQuantizer#Config#rand_proj.
  RandomVectorQuantizer.Config.normalize_codebook: RandomVectorQuantizer#Config#normalize_codebook.
  RandomVectorQuantizer.Config.normalize_inputs: RandomVectorQuantizer#Config#normalize_inputs.
  RandomVectorQuantizer.default_config: RandomVectorQuantizer#default_config().
  RandomVectorQuantizer.__init__: RandomVectorQuantizer#__init__().
  RandomVectorQuantizer.initialize_parameters_recursively: RandomVectorQuantizer#initialize_parameters_recursively().
  RandomVectorQuantizer.forward: RandomVectorQuantizer#forward().
  KmeansVectorQuantizer: KmeansVectorQuantizer#
  KmeansVectorQuantizer.Config: KmeansVectorQuantizer#Config#
  KmeansVectorQuantizer.Config.beta: KmeansVectorQuantizer#Config#beta.
  KmeansVectorQuantizer.Config.normalize_codebook: KmeansVectorQuantizer#Config#normalize_codebook.
  KmeansVectorQuantizer.Config.normalize_inputs: KmeansVectorQuantizer#Config#normalize_inputs.
  KmeansVectorQuantizer.default_config: KmeansVectorQuantizer#default_config().
  KmeansVectorQuantizer._compute_fan_axes: KmeansVectorQuantizer#_compute_fan_axes().
  KmeansVectorQuantizer.forward: KmeansVectorQuantizer#forward().
  GumbelSoftmaxVectorQuantizer: GumbelSoftmaxVectorQuantizer#
  GumbelSoftmaxVectorQuantizer.Config: GumbelSoftmaxVectorQuantizer#Config#
  GumbelSoftmaxVectorQuantizer.Config.input_dim: GumbelSoftmaxVectorQuantizer#Config#input_dim.
  GumbelSoftmaxVectorQuantizer.Config.input_proj: GumbelSoftmaxVectorQuantizer#Config#input_proj.
  GumbelSoftmaxVectorQuantizer.Config.temperature_schedule: GumbelSoftmaxVectorQuantizer#Config#temperature_schedule.
  GumbelSoftmaxVectorQuantizer.__init__: GumbelSoftmaxVectorQuantizer#__init__().
  GumbelSoftmaxVectorQuantizer.default_config: GumbelSoftmaxVectorQuantizer#default_config().
  GumbelSoftmaxVectorQuantizer._create_layer_parameter_specs: GumbelSoftmaxVectorQuantizer#_create_layer_parameter_specs().
  GumbelSoftmaxVectorQuantizer.forward: GumbelSoftmaxVectorQuantizer#forward().
---
# Module: [`axlearn/common/quantizer.py`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py)

## Classes
### `BaseQuantizer`
- def: [`axlearn/common/quantizer.py:95`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L95)
- doc: An abstract class to define the common interface of vector quantizer layers.
- signature: `class BaseQuantizer(BaseLayer):`
- members:
  - `default_config(cls)` — [`L116`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L116)
  - `forward(self, inputs: Tensor, *, paddings: Tensor)` — [`L133`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L133) — Quantizes input sequences.
  - `lookup(self, ids: Tensor)` — [`L147`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L147) — Codebook look up with ids.
- protocol/private: `_create_layer_parameter_specs`[`L124`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L124)

### `Config`
- def: [`axlearn/common/quantizer.py:569`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L569)
- signature: `class Config(BaseQuantizer.Config):`
- members:
  - `beta` — [`L428`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L428)
  - `codebook_dim` — [`L101`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L101)
  - `codebook_size` — [`L103`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L103)
  - `input_dim` — [`L326`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L326)
  - `input_dim` — [`L570`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L570)
  - `input_proj` — [`L571`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L571)
  - `normalize_codebook` — [`L328`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L328)
  - `normalize_codebook` — [`L429`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L429)
  - `normalize_inputs` — [`L329`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L329)
  - `normalize_inputs` — [`L430`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L430)
  - `num_codebooks` — [`L105`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L105)
  - `rand_proj` — [`L327`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L327)
  - `temperature_schedule` — [`L572`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L572)

### `GumbelSoftmaxVectorQuantizer`
- def: [`axlearn/common/quantizer.py:558`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L558)
- doc: Vector quantizer with Gumbel softmax trick.
- signature: `class GumbelSoftmaxVectorQuantizer(BaseQuantizer):`
- members:
  - `default_config(cls)` — [`L588`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L588)
  - `forward(self, inputs: Tensor, *, paddings: Tensor)` — [`L610`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L610) — Quantization using Gumbel softmax trick.
- protocol/private: `__init__`[`L574`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L574), `_create_layer_parameter_specs`[`L599`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L599)

### `KmeansVectorQuantizer`
- def: [`axlearn/common/quantizer.py:410`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L410)
- doc: Vector quantizer with mse loss.
- signature: `class KmeansVectorQuantizer(BaseQuantizer):`
- members:
  - `default_config(cls)` — [`L433`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L433)
  - `forward(self, inputs: Tensor, *, paddings: Tensor)` — [`L457`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L457) — Quantization with mse loss.
- protocol/private: `_compute_fan_axes`[`L448`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L448)

### `Output`
- def: [`axlearn/common/quantizer.py:107`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L107)
- signature: `class Output(NamedTuple):`
- members:
  - `ids` — [`L109`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L109)
  - `loss` — [`L113`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L113)
  - `quantized_vectors` — [`L111`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L111)

### `RandomVectorQuantizer`
- def: [`axlearn/common/quantizer.py:313`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L313)
- doc: Random-projection Quantizer.
- signature: `class RandomVectorQuantizer(BaseQuantizer):`
- members:
  - `default_config(cls)` — [`L332`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L332)
  - `forward(self, inputs: Tensor, *, paddings: Tensor)` — [`L362`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L362) — Computes random projection and quantization.
  - `initialize_parameters_recursively(self, prng_key: Tensor, *, prebuilt: Optional[Nested[Optional[ParameterSpec]]]=None)` — [`L352`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L352)
- protocol/private: `__init__`[`L341`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L341)

### `SimilarityMetric`
- def: [`axlearn/common/quantizer.py:196`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L196)
- signature: `class SimilarityMetric(Enum):`
- members:
  - `DOT_PRODUCT` — [`L198`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L198)
  - `L2_DISTANCE` — [`L197`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L197)

## Functions
- `_add_codebook_summaries(*, context: InvocationContext, onehots: Tensor, paddings: Tensor)` — [`L276`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L276) — Helper function to compute codebook distribution statistics and add to summaries.
- `_apply_paddings(*, outputs: BaseQuantizer.Output, paddings: Tensor)` — [`L246`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L246) — Applies paddings to quantizer outputs.
- `_ids_to_onehots(ids: Tensor, *, codebook_size: int, dtype: jnp.dtype)` — [`L271`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L271)
- `_lookup(*, ids: Tensor, codebook: Tensor)` — [`L165`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L165) — Codebook look up with ids.
- `compute_code_coverage(onehots: Tensor, paddings: Tensor)` — [`L86`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L86) — Computes codebook coverage.
- `compute_code_histogram(onehots: Tensor, paddings: Tensor)` — [`L56`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L56) — Computes histograms of the quantized codes over the codebook vocabulary.
- `compute_code_pplx(onehots: Tensor, paddings: Tensor)` — [`L72`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L72) — Computes pplx and entropy of the quantized codes distribution.
- `quantize_by_nearest_neighbor(inputs: Tensor, *, codebook: Tensor, metric: SimilarityMetric)` — [`L201`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L201) — Quantizes inputs by the nearest neighbor look-up in the codebook.

## Module values
- `_einsum_dims` — [`L53`](../../../../../../raw/code/axlearn/axlearn/common/quantizer.py#L53)

