---
title: 'Module: easydel/trainers/sparse_distillation_trainer/_fn.py'
type: catalog
provenance: extracted
module: easydel/trainers/sparse_distillation_trainer/_fn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.sparse_distillation_trainer._fn`/
symbols:
  sparse_distillation_step: sparse_distillation_step().
  sparse_distillation_step.loss_fn: sparse_distillation_step().loss_fn().
  partial_kl_distillation_loss: partial_kl_distillation_loss().
---
# Module: [`easydel/trainers/sparse_distillation_trainer/_fn.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/_fn.py)

## Functions
- `loss_fn(tree, minibatch)` — [`L205`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/_fn.py#L205)
- `partial_kl_distillation_loss(student_logits: Array, teacher_top_k_indices: Array, teacher_top_k_logprobs: Array, attention_mask: Array | None = None, loss_mask: Array | None = None, labels: Array | None = None, use_hard_labels: bool = False, temperature: float = 4, alpha: float = 0.9)` — [`L49`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/_fn.py#L49) — Compute partial KL distillation loss from sparse teacher logprobs.
- `sparse_distillation_step(state: EasyDeLState, batch: collections.abc.Mapping[str, jax.Array], loss_config: LossConfig | None = None, learning_rate_fn: optax.Schedule | None = None, partition_spec: PartitionSpec | None = None, gradient_accumulation_steps: int = 1, is_training: bool = True, temperature: float = 4, alpha: float = 0.9, straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None)` — [`L160`](../../../../../../../raw/code/EasyDeL/easydel/trainers/sparse_distillation_trainer/_fn.py#L160) — Training/evaluation step for sparse (gray-box) distillation.

