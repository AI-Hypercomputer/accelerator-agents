---
title: 'Module: easydel/trainers/group_relative_policy_optimization/_fn.py'
type: catalog
provenance: extracted
module: easydel/trainers/group_relative_policy_optimization/_fn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.group_relative_policy_optimization._fn`/
symbols:
  get_per_token_logps: get_per_token_logps().
  grpo_step: grpo_step().
  grpo_step.loss_fn: grpo_step().loss_fn().
  get_per_token_logps_and_entropies: get_per_token_logps_and_entropies().
  _masked_sum_and_count: _masked_sum_and_count().
  grpo_step.loss_fn.masked_mean: grpo_step().loss_fn().masked_mean().
  RewardFunc: RewardFunc.
  _maybe_extend_inputs_embeds_for_scoring: _maybe_extend_inputs_embeds_for_scoring().
  compute_per_token_logps: compute_per_token_logps().
---
# Module: [`easydel/trainers/group_relative_policy_optimization/_fn.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/_fn.py)

## Functions
- `_masked_sum_and_count(x: jax.Array, mask: jax.Array)` — [`L64`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/_fn.py#L64) — Return numerator/denominator matching the masked_mean semantics used below.
- `_maybe_extend_inputs_embeds_for_scoring(model, input_ids, model_kwargs, *, prompt_length: int)` — [`L286`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/_fn.py#L286) — Extend prompt-side embeddings so GRPO scores the same prompt representation it sampled from.
- `compute_per_token_logps(logits, input_ids, prompt_length)` — [`L166`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/_fn.py#L166) — Compute per-token log probabilities in a vectorized way.
- `get_per_token_logps(model, input_ids, attention_mask, prompt_length, model_kwargs=None, logprob_vocab_chunk_size: int | None = None)` — [`L72`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/_fn.py#L72) — Compute per-token log probabilities for generated sequences.
- `get_per_token_logps_and_entropies(model, input_ids, attention_mask, prompt_length, model_kwargs=None, logprob_vocab_chunk_size: int | None = None)` — [`L194`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/_fn.py#L194) — Compute per-token log probabilities and entropy for the completion portion.
- `grpo_step(state: EasyDeLState, batch: collections.abc.Mapping[str, jax.Array], num_generations: int, beta: float, loss_config: LossConfig | None = None, learning_rate_fn: optax.Schedule = None, partition_spec: PartitionSpec | None = None, gradient_accumulation_steps: int = 1, is_training: bool = True, loss_type: str = "dapo", epsilon: float = 0.2, epsilon_high: float = 0.2, delta: float | None = None, importance_sampling_level: str = "token", top_entropy_quantile: float = 1, completion_chunk_size: int | None = None, max_loss_completion_tokens: int | None = None, logprob_vocab_chunk_size: int | None = None, straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None)` — [`L330`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/_fn.py#L330) — Perform a single GRPO training or evaluation step.
- `loss_fn(tree, minibatch)` — [`L406`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/_fn.py#L406)
- `masked_mean(x)` — [`L753`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/_fn.py#L753)

## Module values
- `RewardFunc` — [`L61`](../../../../../../../raw/code/EasyDeL/easydel/trainers/group_relative_policy_optimization/_fn.py#L61)

