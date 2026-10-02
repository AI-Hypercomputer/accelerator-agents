---
title: 'Module: easydel/trainers/self_distillation_policy_optimization/_fn.py'
type: catalog
provenance: extracted
module: easydel/trainers/self_distillation_policy_optimization/_fn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.self_distillation_policy_optimization._fn`/sdpo_step().
symbols:
  sdpo_step: ''
  sdpo_step.loss_fn: loss_fn().
  sdpo_step.loss_fn.masked_mean: loss_fn().masked_mean().
---
# Module: [`easydel/trainers/self_distillation_policy_optimization/_fn.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/_fn.py)

## Functions
- `loss_fn(tree, minibatch)` — [`L132`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/_fn.py#L132)
- `masked_mean(x)` — [`L289`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/_fn.py#L289)
- `sdpo_step(state: EasyDeLState, batch: collections.abc.Mapping[str, jax.Array], num_generations: int, teacher_prompt_length: int, beta: float, distillation_type: str, logprob_vocab_chunk_size: int | None, max_loss_completion_tokens: int | None, completion_chunk_size: int | None, loss_config: LossConfig | None = None, learning_rate_fn: optax.Schedule = None, partition_spec: PartitionSpec | None = None, gradient_accumulation_steps: int = 1, is_training: bool = True, straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None)` — [`L60`](../../../../../../../raw/code/EasyDeL/easydel/trainers/self_distillation_policy_optimization/_fn.py#L60) — Single SDPO training / evaluation step.

