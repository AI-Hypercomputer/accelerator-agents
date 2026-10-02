---
title: 'Module: easydel/trainers/contrastive_preference_optimization_trainer/_fn.py'
type: catalog
provenance: extracted
module: easydel/trainers/contrastive_preference_optimization_trainer/_fn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.contrastive_preference_optimization_trainer._fn`/
symbols:
  training_step: training_step().
  concatenated_forward: concatenated_forward().
  evaluation_step: evaluation_step().
  training_step.calculate_loss: training_step().calculate_loss().
  cpo_loss: cpo_loss().
  LOSS_TYPES: LOSS_TYPES.
  _policy_nll_loss: _policy_nll_loss().
---
# Module: [`easydel/trainers/contrastive_preference_optimization_trainer/_fn.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/_fn.py)

## Functions
- `_policy_nll_loss(chosen_logps_raw: jax.Array, chosen_lengths: jax.Array)` — [`L276`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/_fn.py#L276) — Compute negative log-likelihood loss for policy regularization.
- `calculate_loss(tree: flax.nnx.GraphState, call_batch: dict[str, jax.Array])` — [`L338`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/_fn.py#L338)
- `concatenated_forward(model: EasyDeLBaseModule, batch: dict[str, tp.Any], *, is_encoder_decoder: bool, label_pad_token_id: int, padding_value: int, max_length: int | None = None, truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end", aux_loss_enabled: bool = False, loss_type: LOSS_TYPES = "sigmoid", logprob_vocab_chunk_size: int | None = None)` — [`L49`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/_fn.py#L49) — Runs the policy model on concatenated chosen/rejected sequences.
- `cpo_loss(policy_chosen_logps: jax.Array, policy_rejected_logps: jax.Array, *, beta: float, label_smoothing: float, loss_type: LOSS_TYPES, simpo_gamma: float, alpha: float)` — [`L217`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/_fn.py#L217) — Compute CPO losses and rewards for chosen/rejected pairs.
- `evaluation_step(state: EasyDeLState, batch: dict[str, jax.Array], concatenated_forward_fn: tp.Callable[..., dict[str, jax.Array]], beta: float, label_smoothing: float, loss_type: LOSS_TYPES, cpo_alpha: float, simpo_gamma: float, alpha: float, partition_spec: PartitionSpec | None = None)` — [`L407`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/_fn.py#L407) — Execute CPO evaluation step without gradients.
- `training_step(state: EasyDeLState, batch: dict[str, jax.Array], learning_rate_fn: tp.Callable[[jax.Array], jax.Array] | None, concatenated_forward_fn: tp.Callable[..., dict[str, jax.Array]], beta: float, label_smoothing: float, loss_type: LOSS_TYPES, cpo_alpha: float, simpo_gamma: float, alpha: float, loss_config: LossConfig | None = None, partition_spec: PartitionSpec | None = None, gradient_accumulation_steps: int = 1, straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None)` — [`L294`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/_fn.py#L294) — Execute CPO training step with gradient computation.

## Module values
- `LOSS_TYPES` — [`L46`](../../../../../../../raw/code/EasyDeL/easydel/trainers/contrastive_preference_optimization_trainer/_fn.py#L46)

