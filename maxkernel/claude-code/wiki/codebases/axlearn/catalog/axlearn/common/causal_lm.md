---
title: 'Module: axlearn/common/causal_lm.py'
type: catalog
provenance: extracted
module: axlearn/common/causal_lm.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.causal_lm`/
symbols:
  layer_norm_config: layer_norm_config().
  _infer_live_targets: _infer_live_targets().
  CrossEntropyLossMetrics: CrossEntropyLossMetrics#
  CrossEntropyLossMetrics.Config: CrossEntropyLossMetrics#Config#
  CrossEntropyLossMetrics.Config.z_loss_scale: CrossEntropyLossMetrics#Config#z_loss_scale.
  CrossEntropyLossMetrics.forward: CrossEntropyLossMetrics#forward().
  AuxLossMetrics: AuxLossMetrics#
  AuxLossMetrics.Config: AuxLossMetrics#Config#
  AuxLossMetrics.Config.aux_loss_regex: AuxLossMetrics#Config#aux_loss_regex.
  AuxLossMetrics.forward: AuxLossMetrics#forward().
  _update: _update().
  CompositeLossWeights: CompositeLossWeights#
  CompositeLossWeights.forward: CompositeLossWeights#forward().
  CompositeLossMetrics: CompositeLossMetrics#
  CompositeLossMetrics.Config: CompositeLossMetrics#Config#
  CompositeLossMetrics.Config.metrics: CompositeLossMetrics#Config#metrics.
  CompositeLossMetrics.Config.loss_weights: CompositeLossMetrics#Config#loss_weights.
  CompositeLossMetrics.Config.flatten_metrics: CompositeLossMetrics#Config#flatten_metrics.
  CompositeLossMetrics.__init__: CompositeLossMetrics#__init__().
  CompositeLossMetrics.forward: CompositeLossMetrics#forward().
  metrics_config: metrics_config().
  Model: Model#
  Model.Config: Model#Config#
  Model.Config.decoder: Model#Config#decoder.
  Model.Config.batch_axis_names: Model#Config#batch_axis_names.
  Model.Config.seq_axis_names: Model#Config#seq_axis_names.
  Model.Config.metrics: Model#Config#metrics.
  Model.Config.scan_chunk: Model#Config#scan_chunk.
  Model.__init__: Model#__init__().
  Model.default_config: Model#default_config().
  Model.forward: Model#forward().
  Model.beam_search_decode: Model#beam_search_decode().
  Model.sample_decode: Model#sample_decode().
  Model.extract_logits: Model#extract_logits().
  Model.compute_logits: Model#compute_logits().
  Model.score: Model#score().
  Model.predict: Model#predict().
  Model._metrics: Model#_metrics().
  Model._chunked_metrics: Model#_chunked_metrics().
  Model._constrain_input_batch: Model#_constrain_input_batch().
  TransformerStackConfig: TransformerStackConfig.
  residual_initializer_cfg: residual_initializer_cfg().
  gpt_decoder_config: gpt_decoder_config().
---
# Module: [`axlearn/common/causal_lm.py`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py)

## Classes
### `AuxLossMetrics`
- def: [`axlearn/common/causal_lm.py:151`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L151)
- doc: Computes aux loss by aggregating across layers.
- signature: `class AuxLossMetrics(BaseLossMetrics):`
- members:
  - `forward(self, input_batch: Nested[Tensor], *, predict_outputs: Nested[Tensor], module_outputs: Nested[Tensor])` — [`L171`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L171) — Computes aux loss by aggregating module outputs from all layers.

### `CompositeLossMetrics`
- def: [`axlearn/common/causal_lm.py:248`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L248)
- doc: Computes a composite loss from multiple child metrics.
- signature: `class CompositeLossMetrics(BaseLossMetrics):`
- members:
  - `forward(self, input_batch: Nested[Tensor], *, predict_outputs: Nested[Tensor], module_outputs: Nested[Tensor])` — [`L278`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L278) — Combines losses and metrics from the configured children.
- protocol/private: `__init__`[`L267`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L267)

### `CompositeLossWeights`
- def: [`axlearn/common/causal_lm.py:233`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L233)
- doc: Computes loss weights.
- signature: `class CompositeLossWeights(Module):`
- members:
  - `forward(self, child_metrics: dict[str, tuple[Tensor, Nested[Tensor]]])` — [`L236`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L236) — Computes per-child loss weights from child metrics.

### `Config`
- def: [`axlearn/common/causal_lm.py:366`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L366)
- doc: Configuration for a causal-lm.
- signature: `class Config(BaseModel.Config):`
- members:
  - `aux_loss_regex` — [`L169`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L169)
  - `batch_axis_names` — [`L376`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L376)
  - `decoder` — [`L370`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L370)
  - `flatten_metrics` — [`L265`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L265)
  - `loss_weights` — [`L264`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L264)
  - `metrics` — [`L263`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L263)
  - `metrics` — [`L382`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L382)
  - `scan_chunk` — [`L388`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L388)
  - `seq_axis_names` — [`L380`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L380)
  - `z_loss_scale` — [`L74`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L74)

### `CrossEntropyLossMetrics`
- def: [`axlearn/common/causal_lm.py:62`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L62)
- doc: Computes cross entropy loss and related training summaries.
- signature: `class CrossEntropyLossMetrics(BaseLossMetrics):`
- members:
  - `forward(self, input_batch: Nested[Tensor], *, predict_outputs: Nested[Tensor], module_outputs: Nested[Tensor])` — [`L76`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L76) — Computes cross entropy loss.

### `Model`
- def: [`axlearn/common/causal_lm.py:362`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L362)
- doc: Autoregressive decoder-only transformer sequence model.
- signature: `class Model(BaseModel):`
- members:
  - `_chunked_metrics(self, *, input_batch: Nested[Tensor], predict_outputs: Nested[Tensor], target_labels: Tensor, scan_chunk: int)` — [`L640`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L640) — Computes metrics by scanning over sequence chunks, avoiding [B, T, V] logits.
  - `_constrain_input_batch(self, input_batch: NestedTensor)` — [`L811`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L811) — Applies sharding constraints in-place for relevant named tensors in the input batch.
  - `beam_search_decode(self, input_batch: NestedTensor, num_decodes: int=1, brevity_penalty: Optional[Callable[[jnp.array, Tensor], jnp.array]]=brevity_penalty_fn(alpha=0.0))` — [`L456`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L456) — Perform beam search decoding given prefix prompt.
  - `compute_logits(self, predictions: Nested[Tensor])` — [`L536`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L536) — Computes logits from decoder hidden states.
  - `default_config(cls)` — [`L397`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L397)
  - `extract_logits(self, input_batch: NestedTensor)` — [`L519`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L519) — Obtains logits from the language model.
  - `forward(self, input_batch: NestedTensor, return_aux: bool=False)` — [`L410`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L410) — Produce decoder-only loss and predictions including decoder hidden states in
  - `predict(self, input_batch: dict[str, Tensor])` — [`L569`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L569) — Produce decoder hidden states.
  - `sample_decode(self, input_batch: NestedTensor, *, num_decodes: int=1, logits_modifier: Optional[ConfigOr[LogitsToLogitsFn]]=None, stop_decoding_condition: Optional[StopDecodingCondition]=None)` — [`L486`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L486) — Perform sample decoding given prefix prompt.
  - `score(self, input_batch: Nested[Tensor])` — [`L549`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L549) — Produce decoder score like per_token_loss and live_targets.
- protocol/private: `__init__`[`L390`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L390), `_metrics`[`L595`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L595)

## Functions
- `_infer_live_targets(input_batch: Nested[Tensor])` — [`L53`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L53) — Uses `live_targets` (if present), otherwise infers from `target_labels >= 0`.
- `_update(x: dict, updates: dict)` — [`L226`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L226) — Equivalent to `x.update(updates)` but raises upon key conflicts.
- `gpt_decoder_config(stack_cfg: TransformerStackConfig, num_layers: int, hidden_dim: int, num_heads: int, vocab_size: int, max_position_embeddings: int, activation_function: str='nn.relu', layer_norm_epsilon: float=1e-08, dropout_rate: float=0.0, layer_remat: Optional[RematSpec]=None)` — [`L873`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L873) — Build a decoder transformer config in the style of GPT.
- `layer_norm_config(eps=1e-05)` — [`L49`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L49)
- `metrics_config(*, z_loss_scale: Optional[float]=None, aux_loss_regex: Optional[str]=None)` — [`L340`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L340) — Constructs a default causal-lm metrics config.
- `residual_initializer_cfg(num_layers, scale=0.02)` — [`L858`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L858)

## Module values
- `TransformerStackConfig` — [`L851`](../../../../../../raw/code/axlearn/axlearn/common/causal_lm.py#L851)

