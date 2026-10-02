---
title: 'Module: easydel/trainers/on_policy_distillation_trainer/_fn.py'
type: catalog
provenance: extracted
module: easydel/trainers/on_policy_distillation_trainer/_fn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.on_policy_distillation_trainer._fn`/
symbols:
  on_policy_distillation_step: on_policy_distillation_step().
  on_policy_distillation_step.loss_fn: on_policy_distillation_step().loss_fn().
  on_policy_distillation_step.loss_fn._teacher_fwd: on_policy_distillation_step().loss_fn()._teacher_fwd().
  _stop_gradient_tree: _stop_gradient_tree().
---
# Module: [`easydel/trainers/on_policy_distillation_trainer/_fn.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/_fn.py)

## Functions
- `_stop_gradient_tree(tree)` — [`L54`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/_fn.py#L54)
- `_teacher_fwd(kw, t_graphstate)` — [`L141`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/_fn.py#L141)
- `loss_fn(tree, minibatch)` — [`L110`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/_fn.py#L110)
- `on_policy_distillation_step(student_state: EasyDeLState, batch: collections.abc.Mapping[str, jax.Array], teacher_state: EasyDeLState, loss_config: LossConfig | None = None, learning_rate_fn: optax.Schedule = None, partition_spec: PartitionSpec | None = None, gradient_accumulation_steps: int = 1, is_training: bool = True, temperature: float = 4, alpha: float = 0.9, straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None, logits_chunk_size: int | None = None)` — [`L58`](../../../../../../../raw/code/EasyDeL/easydel/trainers/on_policy_distillation_trainer/_fn.py#L58) — Training/evaluation step for on-policy distillation.

