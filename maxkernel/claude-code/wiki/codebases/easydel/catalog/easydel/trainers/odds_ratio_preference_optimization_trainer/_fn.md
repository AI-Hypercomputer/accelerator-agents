---
title: 'Module: easydel/trainers/odds_ratio_preference_optimization_trainer/_fn.py'
type: catalog
provenance: extracted
module: easydel/trainers/odds_ratio_preference_optimization_trainer/_fn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.odds_ratio_preference_optimization_trainer._fn`/
symbols:
  orpo_step: orpo_step().
  concatenated_forward: concatenated_forward().
  orpo_step.calculate_loss: orpo_step().calculate_loss().
  concatenated_forward.cross_entropy_loss: concatenated_forward().cross_entropy_loss().
  get_batch_logps: get_batch_logps().
  get_batch_mean_logit_summaries: get_batch_mean_logit_summaries().
  concatenated_inputs: concatenated_inputs().
  odds_ratio_loss: odds_ratio_loss().
---
# Module: [`easydel/trainers/odds_ratio_preference_optimization_trainer/_fn.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/_fn.py)

## Functions
- `calculate_loss(tree: nn.GraphState, batch: dict)` — [`L505`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/_fn.py#L505) — Computes the loss and metrics for a given minibatch.
- `concatenated_forward(state: EasyDeLState, batch: collections.abc.Mapping[str, list | Array], is_encoder_decoder: bool, label_pad_token_id: int, padding_value: tp.Any, max_length: int | None = None, logprob_vocab_chunk_size: int | None = None)` — [`L62`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/_fn.py#L62) — Computes log-probabilities and logits for both chosen and rejected examples by concatenating
- `concatenated_inputs(batch: dict[str, list | Array], is_encoder_decoder: bool = False)` — [`L370`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/_fn.py#L370) — Concatenates chosen and rejected examples from the batch into unified arrays.
- `cross_entropy_loss(logits, labels)` — [`L152`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/_fn.py#L152) — Computes the cross entropy loss and accuracy between the logits and labels.
- `get_batch_logps(logits: Array, labels: Array, average_log_prob: bool = False, label_pad_token_id: int = -100, is_encoder_decoder: bool = False, logprob_vocab_chunk_size: int | None = None)` — [`L261`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/_fn.py#L261) — Computes the log probabilities for a batch of sequences given the model logits and labels.
- `get_batch_mean_logit_summaries(logits: Array, labels: Array, label_pad_token_id: int = -100, is_encoder_decoder: bool = False)` — [`L313`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/_fn.py#L313) — Compute a per-example mean logit summary over loss-bearing token positions.
- `odds_ratio_loss(beta: float, policy_chosen_logps: Array, policy_rejected_logps: Array)` — [`L415`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/_fn.py#L415) — Computes the odds ratio loss used for training based on the log probabilities of chosen and rejected examples.
- `orpo_step(state: EasyDeLState, batch: dict, concatenated_forward: tp.Callable, beta: float = 0.1, learning_rate_fn: tp.Callable | None = None, mode: tp.Literal["train", "eval"] = "train", loss_config: LossConfig | None = None, partition_spec: PartitionSpec | None = None, gradient_accumulation_steps: int = 1, straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None)` — [`L455`](../../../../../../../raw/code/EasyDeL/easydel/trainers/odds_ratio_preference_optimization_trainer/_fn.py#L455) — Performs a single training or evaluation step for the ORPO method.

