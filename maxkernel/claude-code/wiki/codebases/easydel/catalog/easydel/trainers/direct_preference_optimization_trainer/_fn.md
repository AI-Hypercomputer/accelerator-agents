---
title: 'Module: easydel/trainers/direct_preference_optimization_trainer/_fn.py'
type: catalog
provenance: extracted
module: easydel/trainers/direct_preference_optimization_trainer/_fn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.direct_preference_optimization_trainer._fn`/
symbols:
  get_loss_function: get_loss_function().
  training_step: training_step().
  concatenated_forward: concatenated_forward().
  evaluation_step: evaluation_step().
  _compute_dpo_outputs_from_hidden_states: _compute_dpo_outputs_from_hidden_states().
  _compute_token_logps_chunked: _compute_token_logps_chunked().
  evaluation_step.calculate_loss: evaluation_step().calculate_loss().
  training_step.calculate_loss: training_step().calculate_loss().
  _compute_dpo_outputs_from_hidden_states._chunk_contributions: _compute_dpo_outputs_from_hidden_states()._chunk_contributions().
  concatenated_inputs: concatenated_inputs().
  _compute_dpo_outputs_from_hidden_states._accumulate_chunk: _compute_dpo_outputs_from_hidden_states()._accumulate_chunk().
  _compute_token_logps_chunked._max_step: _compute_token_logps_chunked()._max_step().
  _compute_token_logps_chunked._sum_step: _compute_token_logps_chunked()._sum_step().
  _compute_token_logps_chunked.max_body: _compute_token_logps_chunked().max_body().
  _compute_token_logps_chunked.sum_body: _compute_token_logps_chunked().sum_body().
  _compute_dpo_outputs_from_hidden_states._full_body: _compute_dpo_outputs_from_hidden_states()._full_body().
  get_loss_function._sigmoid_dpo_loss: get_loss_function()._sigmoid_dpo_loss().
  get_loss_function._discopop_dpo_loss: get_loss_function()._discopop_dpo_loss().
  _compute_dpo_outputs_from_hidden_states._project_chunk: _compute_dpo_outputs_from_hidden_states()._project_chunk().
  _get_reference_logps_from_batch: _get_reference_logps_from_batch().
  get_loss_function._base_dpo_loss: get_loss_function()._base_dpo_loss().
  get_loss_function._aot_dpo_loss: get_loss_function()._aot_dpo_loss().
  _resolve_dpo_lmhead_chunksize: _resolve_dpo_lmhead_chunksize().
  get_loss_function._nca_pair_dpo_loss: get_loss_function()._nca_pair_dpo_loss().
  get_loss_function._hinge_dpo_loss: get_loss_function()._hinge_dpo_loss().
  get_loss_function._ipo_dpo_loss: get_loss_function()._ipo_dpo_loss().
  get_loss_function._kto_pair_dpo_loss: get_loss_function()._kto_pair_dpo_loss().
  get_loss_function._robust_dpo_loss: get_loss_function()._robust_dpo_loss().
  get_loss_function._exo_pair_dpo_loss: get_loss_function()._exo_pair_dpo_loss().
  get_loss_function._bco_pair_dpo_loss: get_loss_function()._bco_pair_dpo_loss().
  get_loss_function._sppo_hard_dpo_loss: get_loss_function()._sppo_hard_dpo_loss().
  get_loss_function._aot_pair_dpo_loss: get_loss_function()._aot_pair_dpo_loss().
  get_loss_function._apo_zero_dpo_loss: get_loss_function()._apo_zero_dpo_loss().
  get_loss_function._apo_down_dpo_loss: get_loss_function()._apo_down_dpo_loss().
---
# Module: [`easydel/trainers/direct_preference_optimization_trainer/_fn.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py)

## Functions
- `_accumulate_chunk(start: int, size: int, carry: tuple[Array, Array, Array, Array, Array])` — [`L290`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L290)
- `_aot_dpo_loss(chosen_logps: Array, rejected_logps: Array, ref_chosen_logps: Array, ref_rejected_logps: Array, beta: float, label_smoothing: float, **kwargs)` — [`L560`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L560) — Computes the DPO loss using the AOT (All Ordered Terms) loss formulation.
- `_aot_pair_dpo_loss(chosen_logps: Array, rejected_logps: Array, ref_chosen_logps: Array, ref_rejected_logps: Array, beta: float, label_smoothing: float)` — [`L776`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L776) — Computes the AOT pair variant of the DPO loss.
- `_apo_down_dpo_loss(chosen_logps: Array, rejected_logps: Array, ref_chosen_logps: Array, ref_rejected_logps: Array, beta: float, label_smoothing: float)` — [`L849`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L849) — Computes the APO down variant of the DPO loss.
- `_apo_zero_dpo_loss(chosen_logps: Array, rejected_logps: Array, ref_chosen_logps: Array, ref_rejected_logps: Array, beta: float, label_smoothing: float)` — [`L826`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L826) — Computes the APO zero variant of the DPO loss.
- `_base_dpo_loss(chosen_logps: Array, rejected_logps: Array, ref_chosen_logps: Array, ref_rejected_logps: Array, beta: float, label_smoothing: float, **kwargs)` — [`L467`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L467) — Base computation for DPO loss.
- `_bco_pair_dpo_loss(chosen_logps: Array, rejected_logps: Array, ref_chosen_logps: Array, ref_rejected_logps: Array, beta: float, label_smoothing: float)` — [`L731`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L731) — Computes the BCO pair variant of the DPO loss.
- `_chunk_contributions(chunk_hidden_states: Array, chunk_labels: Array, chunk_loss_mask: Array)` — [`L262`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L262)
- `_compute_dpo_outputs_from_hidden_states(model: tp.Any, hidden_states: Array, labels: Array, loss_mask: Array, *, num_examples: int, chunk_size: int, logprob_vocab_chunk_size: int | None, loss_type: LOSS_FN_VARIANTS)` — [`L175`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L175) — Project DPO hidden states through the LM head chunk-by-chunk across the sequence dimension.
- `_compute_token_logps_chunked(logits: Array, targets: Array, *, chunk_size: int | None)` — [`L63`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L63) — Compute selected token log-probabilities without materializing a full-vocab log-softmax.
- `_discopop_dpo_loss(chosen_logps: Array, rejected_logps: Array, ref_chosen_logps: Array, ref_rejected_logps: Array, beta: float, label_smoothing: float, discopop_tau: float = 1, **kwargs)` — [`L590`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L590) — Computes the DPO loss using a Discopo-based modulation.
- `_exo_pair_dpo_loss(chosen_logps: Array, rejected_logps: Array, ref_chosen_logps: Array, ref_rejected_logps: Array, beta: float, label_smoothing: float)` — [`L706`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L706) — Computes the exo-pair variant of the DPO loss.
- `_full_body(i: int, inner_carry: tuple[Array, Array, Array, Array, Array])` — [`L317`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L317)
- `_get_reference_logps_from_batch(batch: dict[str, tp.Any])` — [`L340`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L340) — Read reference log-prob columns from either the canonical or legacy keys.
- `_hinge_dpo_loss(chosen_logps: Array, rejected_logps: Array, ref_chosen_logps: Array, ref_rejected_logps: Array, beta: float, label_smoothing: float)` — [`L624`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L624) — Computes the hinge loss version of the DPO loss.
- `_ipo_dpo_loss(chosen_logps: Array, rejected_logps: Array, ref_chosen_logps: Array, ref_rejected_logps: Array, beta: float, label_smoothing: float)` — [`L644`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L644) — Computes the IPO loss variant of the DPO loss.
- `_kto_pair_dpo_loss(chosen_logps: Array, rejected_logps: Array, ref_chosen_logps: Array, ref_rejected_logps: Array, beta: float, label_smoothing: float)` — [`L664`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L664) — Computes the KTO pair loss variant.
- `_max_step(start: int, size: int, running_max: Array)` — [`L108`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L108)
- `_nca_pair_dpo_loss(chosen_logps: Array, rejected_logps: Array, ref_chosen_logps: Array, ref_rejected_logps: Array, beta: float, label_smoothing: float, **kwargs)` — [`L534`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L534) — Computes the DPO loss using an NCA pair formulation.
- `_project_chunk(chunk_hidden_states: Array)` — [`L255`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L255)
- `_resolve_dpo_lmhead_chunksize(model: tp.Any)` — [`L145`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L145) — Return the configured LM-head token chunk size when headless DPO is supported.
- `_robust_dpo_loss(chosen_logps: Array, rejected_logps: Array, ref_chosen_logps: Array, ref_rejected_logps: Array, beta: float, label_smoothing: float)` — [`L684`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L684) — Computes a robust variant of the DPO loss.
- `_sigmoid_dpo_loss(chosen_logps: Array, rejected_logps: Array, ref_chosen_logps: Array, ref_rejected_logps: Array, beta: float, label_smoothing: float, **kwargs)` — [`L499`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L499) — Computes the DPO loss using a sigmoid-based formulation.
- `_sppo_hard_dpo_loss(chosen_logps: Array, rejected_logps: Array, ref_chosen_logps: Array, ref_rejected_logps: Array, beta: float, label_smoothing: float)` — [`L755`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L755) — Computes the SPO PPO hard variant of the DPO loss.
- `_sum_step(start: int, size: int, running_sum: Array)` — [`L124`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L124)
- `calculate_loss(tree: flax.nnx.GraphState, call_batch)` — [`L1148`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L1148) — Inner function to compute loss and metrics for a given minibatch.
- `calculate_loss(tree: flax.nnx.GraphState)` — [`L1261`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L1261) — Inner function to compute loss metrics for evaluation.
- `concatenated_forward(model: EasyDeLBaseModule, batch: dict[str, list | Array], is_encoder_decoder: bool, label_pad_token_id: int, padding_value: int, max_length: int | None = None, truncation_mode: str = "keep_end", aux_loss_enabled: bool = False, loss_type: str = "sigmoid", logprob_vocab_chunk_size: int | None = None)` — [`L894`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L894) — Runs the model on concatenated chosen/rejected inputs for efficiency.
- `concatenated_inputs(batch: dict[str, list | Array], padding_value: int)` — [`L353`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L353) — Concatenates chosen and rejected examples from the batch, and pads the inputs to a uniform length.
- `evaluation_step(state: EasyDeLState, batch: dict, reference_state: EasyDeLState | None, concatenated_forward: tp.Callable, beta: float = 0.1, label_smoothing: float = 0, loss_type: LOSS_FN_VARIANTS = "sigmoid", reference_free: bool = False, partition_spec: PartitionSpec | None = None)` — [`L1216`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L1216) — Performs a single evaluation step.
- `get_loss_function(loss_type: LOSS_FN_VARIANTS, beta: float, label_smoothing: float | int)` — [`L445`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L445) — Returns a loss function based on the specified loss type.
- `max_body(i: int, running_max: Array)` — [`L114`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L114)
- `sum_body(i: int, running_sum: Array)` — [`L130`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L130)
- `training_step(state: EasyDeLState, batch: dict, reference_state: EasyDeLState, learning_rate_fn: tp.Callable, concatenated_forward: tp.Callable, beta: float = 0.1, label_smoothing: float = 0, loss_type: LOSS_FN_VARIANTS = "sigmoid", reference_free: bool = False, loss_config: LossConfig | None = None, partition_spec: PartitionSpec | None = None, gradient_accumulation_steps: int = 1, straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None)` — [`L1075`](../../../../../../../raw/code/EasyDeL/easydel/trainers/direct_preference_optimization_trainer/_fn.py#L1075) — Performs a single training step.

