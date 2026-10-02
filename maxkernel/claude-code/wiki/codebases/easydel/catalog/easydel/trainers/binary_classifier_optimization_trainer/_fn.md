---
title: 'Module: easydel/trainers/binary_classifier_optimization_trainer/_fn.py'
type: catalog
provenance: extracted
module: easydel/trainers/binary_classifier_optimization_trainer/_fn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.binary_classifier_optimization_trainer._fn`/
symbols:
  training_step: training_step().
  concatenated_forward: concatenated_forward().
  training_step.calculate_loss: training_step().calculate_loss().
  evaluation_step: evaluation_step().
  RunningMoments.update: RunningMoments#update().
  RunningMoments.as_dict: RunningMoments#as_dict().
  RunningMoments.load_dict: RunningMoments#load_dict().
  detach_metrics: detach_metrics().
  RunningMoments.count: RunningMoments#count.
  RunningMoments.mean: RunningMoments#mean.
  RunningMoments.var: RunningMoments#var.
  RunningMoments: RunningMoments#
  compute_bco_loss: compute_bco_loss().
  RunningMoments.__init__: RunningMoments#__init__().
  __all__: __all__.
---
# Module: [`easydel/trainers/binary_classifier_optimization_trainer/_fn.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/_fn.py)

## Classes
### `RunningMoments`
- def: [`easydel/trainers/binary_classifier_optimization_trainer/_fn.py:44`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/_fn.py#L44)
- doc: Simple running mean/variance tracker for BCO delta parameter.
- signature: `class RunningMoments:`
- members:
  - `as_dict(self)` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/_fn.py#L82) — Export statistics as dictionary.
  - `load_dict(self, data: dict[str, float])` — [`L90`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/_fn.py#L90) — Load statistics from dictionary.
  - `update(self, values: collections.abc.Sequence[float] | jnp.ndarray | None)` — [`L56`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/_fn.py#L56) — Update running statistics with new values.
  - `count` — [`L54`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/_fn.py#L54)
  - `mean` — [`L52`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/_fn.py#L52)
  - `var` — [`L53`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/_fn.py#L53)
- protocol/private: `__init__`[`L51`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/_fn.py#L51)
- used by: [`_preprocess_batch_input`](bco_trainer.md#BCOTrainer._preprocess_batch_input), [`running`](bco_trainer.md#BCOTrainer.running)

## Functions
- `calculate_loss(tree: jax.ArrayTree, call_batch: dict[str, jax.Array])` — [`L351`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/_fn.py#L351)
- `compute_bco_loss(policy_logps: jax.Array, reference_logps: jax.Array, chosen_mask: jax.Array, rejected_mask: jax.Array, *, beta: float, delta: float, udm_weights: jax.Array | None = None)` — [`L249`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/_fn.py#L249) — Compute BCO loss and rewards for desirable/undesirable examples.
- `concatenated_forward(model: EasyDeLBaseModule, batch: dict[str, jax.Array], *, is_encoder_decoder: bool, label_pad_token_id: int, padding_value: int, max_length: int | None = None, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end", aux_loss_enabled: bool = False, logprob_vocab_chunk_size: int | None = None)` — [`L101`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/_fn.py#L101) — Run model forward pass to compute completion log probabilities.
- `detach_metrics(metrics: LossMetrics)` — [`L529`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/_fn.py#L529) — Convert metrics to host scalars for serialization.
- `evaluation_step(state: EasyDeLState, batch: dict[str, jax.Array], reference_state: EasyDeLState | None, concatenated_forward_fn: tp.Callable[..., dict[str, jax.Array]], beta: float)` — [`L440`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/_fn.py#L440) — Execute BCO evaluation step without gradients.
- `training_step(state: EasyDeLState, batch: dict[str, jax.Array], reference_state: EasyDeLState | None, learning_rate_fn: tp.Callable | None, concatenated_forward_fn: tp.Callable[..., dict[str, jax.Array]], beta: float, loss_config: LossConfig | None, partition_spec: PartitionSpec | None, gradient_accumulation_steps: int, straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None)` — [`L306`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/_fn.py#L306) — Execute BCO training step with gradient computation.

## Module values
- `__all__` — [`L556`](../../../../../../../raw/code/EasyDeL/easydel/trainers/binary_classifier_optimization_trainer/_fn.py#L556)

