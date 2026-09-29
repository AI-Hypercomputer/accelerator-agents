---
title: 'Module: easydel/trainers/nash_md_trainer/_fn.py'
type: catalog
provenance: extracted
module: easydel/trainers/nash_md_trainer/_fn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.nash_md_trainer._fn`/
symbols:
  nash_md_step: nash_md_step().
  nash_md_step.loss_fn: nash_md_step().loss_fn().
  _compute_policy_logps: _compute_policy_logps().
---
# Module: [`easydel/trainers/nash_md_trainer/_fn.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/_fn.py)

## Functions
- `_compute_policy_logps(module: flax.nnx.Module, prompt_ids: jax.Array, prompt_mask: jax.Array, completion_ids: jax.Array, completion_mask: jax.Array, logprob_vocab_chunk_size: int | None)` — [`L37`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/_fn.py#L37) — Compute policy log probabilities for completion tokens.
- `loss_fn(tree: flax.nnx.GraphState, minibatch: dict[str, jax.Array])` — [`L106`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/_fn.py#L106)
- `nash_md_step(state: EasyDeLState, batch: dict[str, jax.Array], beta: float, logprob_vocab_chunk_size: int | None, loss_config: LossConfig | None, learning_rate_fn, partition_spec: PartitionSpec | None, gradient_accumulation_steps: int, is_train: bool, straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None)` — [`L70`](../../../../../../../raw/code/EasyDeL/easydel/trainers/nash_md_trainer/_fn.py#L70) — Execute Nash-MD training or evaluation step.

