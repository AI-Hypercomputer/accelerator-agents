---
title: 'Module: easydel/trainers/reward_trainer/_fn.py'
type: catalog
provenance: extracted
module: easydel/trainers/reward_trainer/_fn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.reward_trainer._fn`/
symbols:
  training_step: training_step().
  evaluation_step: evaluation_step().
  training_step.loss_fn: training_step().loss_fn().
  evaluation_step.loss_fn: evaluation_step().loss_fn().
---
# Module: [`easydel/trainers/reward_trainer/_fn.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/_fn.py)

## Functions
- `evaluation_step(state: EasyDeLState, batch: collections.abc.Mapping[str, jax.Array], loss_config: LossConfig | None = None, partition_spec: PartitionSpec | None = None, center_rewards_coefficient: float | None = None)` — [`L167`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/_fn.py#L167) — Performs a single evaluation step by computing loss metrics for the input batch.
- `loss_fn(tree, minibatch)` — [`L102`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/_fn.py#L102) — Computes the loss and additional metrics for a given minibatch and tree state.
- `loss_fn(tree)` — [`L206`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/_fn.py#L206) — Computes loss metrics for the evaluation batch given a merged graph state.
- `training_step(state: EasyDeLState, batch: collections.abc.Mapping[str, jax.Array], loss_config: LossConfig | None = None, learning_rate_fn: optax.Schedule = None, partition_spec: PartitionSpec | None = None, gradient_accumulation_steps: int = 1, center_rewards_coefficient: float | None = None, straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None)` — [`L53`](../../../../../../../raw/code/EasyDeL/easydel/trainers/reward_trainer/_fn.py#L53) — Performs a single training step by computing gradients via minibatch processing,

