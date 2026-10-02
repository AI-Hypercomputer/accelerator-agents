---
title: 'Module: easydel/trainers/xpo_trainer/_fn.py'
type: catalog
provenance: extracted
module: easydel/trainers/xpo_trainer/_fn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.xpo_trainer._fn`/
symbols:
  xpo_step: xpo_step().
  xpo_step.loss_fn: xpo_step().loss_fn().
  _compute_logps: _compute_logps().
  _sum_logps: _sum_logps().
---
# Module: [`easydel/trainers/xpo_trainer/_fn.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/_fn.py)

## Functions
- `_compute_logps(module: flax.nnx.Module, prompt_ids: jax.Array, prompt_mask: jax.Array, completion_ids: jax.Array, completion_mask: jax.Array, logprob_vocab_chunk_size: int | None)` — [`L38`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/_fn.py#L38) — Compute per-token log probabilities for completion tokens.
- `_sum_logps(token_logps: jax.Array, completion_mask: jax.Array)` — [`L73`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/_fn.py#L73) — Sum log probabilities over completion tokens, respecting the attention mask.
- `loss_fn(tree: flax.nnx.GraphState, minibatch: dict[str, jax.Array])` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/_fn.py#L151)
- `xpo_step(state: EasyDeLState, batch: dict[str, jax.Array], reference_state: EasyDeLState, logprob_vocab_chunk_size: int | None, loss_config: LossConfig | None, learning_rate_fn, partition_spec: PartitionSpec | None, gradient_accumulation_steps: int, is_train: bool, straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None)` — [`L87`](../../../../../../../raw/code/EasyDeL/easydel/trainers/xpo_trainer/_fn.py#L87) — Execute a single XPO training or evaluation step.

