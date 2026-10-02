---
title: 'Module: easydel/trainers/generalized_knowledge_distillation_trainer/_fn.py'
type: catalog
provenance: extracted
module: easydel/trainers/generalized_knowledge_distillation_trainer/_fn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.generalized_knowledge_distillation_trainer._fn`/
symbols:
  gkd_step: gkd_step().
  gkd_step.loss_fn: gkd_step().loss_fn().
  gkd_step.teacher_forward: gkd_step().teacher_forward().
  _kl_div: _kl_div().
  generalized_jsd_loss: generalized_jsd_loss().
  gkd_step.teacher_forward._teacher_fwd: gkd_step().teacher_forward()._teacher_fwd().
  _stop_gradient_tree: _stop_gradient_tree().
  _stop_gradient_tree._maybe_stop: _stop_gradient_tree()._maybe_stop().
---
# Module: [`easydel/trainers/generalized_knowledge_distillation_trainer/_fn.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/_fn.py)

## Functions
- `_kl_div(log_target: jax.Array, log_input: jax.Array)` — [`L57`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/_fn.py#L57) — Compute KL divergence KL(target || input) given log-probabilities.
- `_maybe_stop(x)` — [`L49`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/_fn.py#L49)
- `_stop_gradient_tree(tree)` — [`L39`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/_fn.py#L39) — Detach a pytree by applying stop_gradient to every array leaf.
- `_teacher_fwd(kw, t_graphstate)` — [`L186`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/_fn.py#L186)
- `generalized_jsd_loss(student_logits: jax.Array, teacher_logits: jax.Array, *, labels: jax.Array | None = None, mask: jax.Array | None = None, beta: float = 0.5, temperature: float = 1)` — [`L71`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/_fn.py#L71) — Compute generalized Jensen-Shannon divergence for knowledge distillation.
- `gkd_step(student_state: EasyDeLState, batch: collections.abc.Mapping[str, jax.Array], teacher_state: EasyDeLState, loss_config: LossConfig | None = None, learning_rate_fn=None, partition_spec: PartitionSpec | None = None, gradient_accumulation_steps: int = 1, is_training: bool = True, beta: float = 0.5, temperature: float = 1, straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None)` — [`L131`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/_fn.py#L131) — Execute GKD training or evaluation step.
- `loss_fn(tree, minibatch)` — [`L196`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/_fn.py#L196)
- `teacher_forward(minibatch: collections.abc.Mapping[str, jax.Array])` — [`L168`](../../../../../../../raw/code/EasyDeL/easydel/trainers/generalized_knowledge_distillation_trainer/_fn.py#L168)

