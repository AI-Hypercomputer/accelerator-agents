---
title: 'Module: easydel/trainers/kto_trainer/_fn.py'
type: catalog
provenance: extracted
module: easydel/trainers/kto_trainer/_fn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.kto_trainer._fn`/
symbols:
  training_step: training_step().
  evaluation_step: evaluation_step().
  training_step._loss_fn: training_step()._loss_fn().
  kto_objective: kto_objective().
  _build_kl_batch: _build_kl_batch().
  kto_objective._safe_sigmoid: kto_objective()._safe_sigmoid().
  KTO_LOSS_TYPES: KTO_LOSS_TYPES.
  _build_kl_batch._rolled: _build_kl_batch()._rolled().
---
# Module: [`easydel/trainers/kto_trainer/_fn.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/_fn.py)

## Functions
- `_build_kl_batch(batch: dict[str, jax.Array])` — [`L38`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/_fn.py#L38) — Create mismatched prompt/completion batch for KL estimation.
- `_loss_fn(tree: flax.nnx.GraphState, minibatch: dict[str, jax.Array])` — [`L201`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/_fn.py#L201)
- `_rolled(name: str)` — [`L60`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/_fn.py#L60)
- `_safe_sigmoid(x)` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/_fn.py#L120)
- `evaluation_step(state: EasyDeLState, batch: dict[str, jax.Array], reference_state: EasyDeLState, forward_fn: tp.Callable[[EasyDeLState | EasyDeLState.model, dict[str, jax.Array]], dict[str, jax.Array]], beta: float, desirable_weight: float, undesirable_weight: float, loss_type: str, calculate_kl: bool, aux_loss_coef: float, partition_spec: PartitionSpec | None = None)` — [`L262`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/_fn.py#L262) — Execute KTO evaluation step without gradients.
- `kto_objective(policy_logps: jax.Array, reference_logps: jax.Array, labels: jax.Array, *, beta: float, desirable_weight: float, undesirable_weight: float, loss_type: str, policy_kl_logps: jax.Array | None = None, reference_kl_logps: jax.Array | None = None)` — [`L75`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/_fn.py#L75) — Compute KTO or APO unpaired losses and rewards.
- `training_step(state: EasyDeLState, batch: dict[str, jax.Array], reference_state: EasyDeLState, learning_rate_fn: tp.Callable[[jax.Array], jax.Array], forward_fn: tp.Callable[[EasyDeLState | EasyDeLState.model, dict[str, jax.Array]], dict[str, jax.Array]], beta: float, desirable_weight: float, undesirable_weight: float, loss_type: str, calculate_kl: bool, aux_loss_coef: float, loss_config: LossConfig | None = None, partition_spec: PartitionSpec | None = None, gradient_accumulation_steps: int = 1, straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None)` — [`L145`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/_fn.py#L145) — Execute KTO training step with gradient computation.

## Module values
- `KTO_LOSS_TYPES` — [`L35`](../../../../../../../raw/code/EasyDeL/easydel/trainers/kto_trainer/_fn.py#L35)

