---
title: 'Module: easydel/trainers/proximal_policy_optimization_trainer/_fn.py'
type: catalog
provenance: extracted
module: easydel/trainers/proximal_policy_optimization_trainer/_fn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.proximal_policy_optimization_trainer._fn`/
symbols:
  ppo_step: ppo_step().
  ppo_step.loss_fn: ppo_step().loss_fn().
  _masked_mean: _masked_mean().
  get_per_token_logps_values_entropies: get_per_token_logps_values_entropies().
  compute_per_token_logps: compute_per_token_logps().
---
# Module: [`easydel/trainers/proximal_policy_optimization_trainer/_fn.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/_fn.py)

## Functions
- `_masked_mean(x: jax.Array, mask: jax.Array)` — [`L50`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/_fn.py#L50) — Compute the mean of masked elements.
- `compute_per_token_logps(logits: jax.Array, input_ids: jax.Array, prompt_length: int)` — [`L64`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/_fn.py#L64) — Compute per-token log probabilities for completion tokens.
- `get_per_token_logps_values_entropies(model, input_ids: jax.Array, attention_mask: jax.Array, prompt_length: int, logprob_vocab_chunk_size: int | None = None)` — [`L89`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/_fn.py#L89) — Compute per-token log probabilities, values, and entropies for PPO.
- `loss_fn(tree, minibatch)` — [`L232`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/_fn.py#L232)
- `ppo_step(state: EasyDeLState, batch: collections.abc.Mapping[str, jax.Array], prompt_length: int, cliprange: float, vf_coef: float, cliprange_value: float, entropy_coef: float, logprob_vocab_chunk_size: int | None, loss_config: LossConfig | None = None, learning_rate_fn: optax.Schedule = None, partition_spec: PartitionSpec | None = None, gradient_accumulation_steps: int = 1, is_training: bool = True, straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None)` — [`L178`](../../../../../../../raw/code/EasyDeL/easydel/trainers/proximal_policy_optimization_trainer/_fn.py#L178) — Execute a single PPO training or evaluation step.

