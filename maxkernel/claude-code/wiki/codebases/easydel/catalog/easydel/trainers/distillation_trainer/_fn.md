---
title: 'Module: easydel/trainers/distillation_trainer/_fn.py'
type: catalog
provenance: extracted
module: easydel/trainers/distillation_trainer/_fn.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.distillation_trainer._fn`/
symbols:
  distillation_step.loss_fn: distillation_step().loss_fn().
  distillation_step: distillation_step().
  distillation_step.teacher_forward: distillation_step().teacher_forward().
  chunked_distillation_loss: chunked_distillation_loss().
  distillation_step.teacher_forward._teacher_fwd: distillation_step().teacher_forward()._teacher_fwd().
  distillation_loss: distillation_loss().
  _resolve_indices: _resolve_indices().
  _compute_kl_and_ce: _compute_kl_and_ce().
  chunked_distillation_loss._chunk_kl_ce: chunked_distillation_loss()._chunk_kl_ce().
  chunked_distillation_loss._scan_body: chunked_distillation_loss()._scan_body().
  _per_token_xent: _per_token_xent().
  _masked_mse: _masked_mse().
  _normalize_attention: _normalize_attention().
  _stop_gradient_tree: _stop_gradient_tree().
  _finalize_distillation_metrics: _finalize_distillation_metrics().
  _build_mask_and_labels: _build_mask_and_labels().
  _build_attention_mask: _build_attention_mask().
---
# Module: [`easydel/trainers/distillation_trainer/_fn.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py)

## Functions
- `_build_attention_mask(attention_mask: jax.Array | None, *, dtype: jnp.dtype)` — [`L428`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L428)
- `_build_mask_and_labels(attention_mask: Array | None, loss_mask: Array | None, labels: Array | None, dtype: jnp.dtype, seq_len: int, batch_size: int)` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L151) — Build a combined per-token mask and safe labels array.
- `_chunk_kl_ce(s_h, t_h, m, sl)` — [`L355`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L355)
- `_compute_kl_and_ce(student_logits: Array, teacher_logits: Array, mask: Array, safe_labels: Array, use_hard_labels: bool, temperature: float, dtype: jnp.dtype)` — [`L81`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L81) — Per-token distillation sums for one chunk of logits.
- `_finalize_distillation_metrics(distill_xent_sum: Array, teacher_entropy_sum: Array, ce_sum: Array, mask_sum: Array, temperature: float, alpha: float, use_hard_labels: bool, dtype: jnp.dtype)` — [`L117`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L117) — Normalise accumulated distillation/CE sums into final scalar metrics/loss.
- `_masked_mse(values: jax.Array, targets: jax.Array, mask: jax.Array | None)` — [`L413`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L413)
- `_normalize_attention(tensor: jax.Array)` — [`L435`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L435)
- `_per_token_xent(teacher_logits: Array, student_logits: Array, temperature: float, dtype: jnp.dtype)` — [`L55`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L55) — Compute per-token distillation cross-entropy and teacher entropy.
- `_resolve_indices(collection_length: int, indices: tuple[int, ...] | None, *, default_all: bool)` — [`L392`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L392)
- `_scan_body(carry, xs)` — [`L368`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L368)
- `_stop_gradient_tree(tree)` — [`L441`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L441)
- `_teacher_fwd(kw, t_graphstate)` — [`L553`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L553)
- `chunked_distillation_loss(student_hidden: Array, teacher_hidden: Array, student_lm_head_fn: tp.Callable[[Array], Array], teacher_lm_head_fn: tp.Callable[[Array], Array], attention_mask: Array | None = None, loss_mask: Array | None = None, labels: Array | None = None, use_hard_labels: bool = False, temperature: float = 4, alpha: float = 0.9, chunk_size: int = 128)` — [`L288`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L288) — Memory-efficient distillation loss that avoids materialising full logits.
- `distillation_loss(student_logits: Array, teacher_logits: Array, attention_mask: Array | None = None, loss_mask: Array | None = None, labels: Array | None = None, use_hard_labels: bool = False, temperature: float = 4, alpha: float = 0.9)` — [`L181`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L181) — Compute knowledge distillation loss between student and teacher models.
- `distillation_step(student_state: EasyDeLState, batch: collections.abc.Mapping[str, jax.Array], teacher_state: EasyDeLState, loss_config: LossConfig | None = None, learning_rate_fn: optax.Schedule = None, partition_spec: PartitionSpec | None = None, gradient_accumulation_steps: int = 1, is_training: bool = True, temperature: float = 4, alpha: float = 0.9, hidden_state_weight: float = 0, hidden_state_layers: tuple[int, ...] | None = None, hidden_state_loss: tp.Literal["mse"] = "mse", attention_weight: float = 0, attention_layers: tuple[int, ...] | None = None, attention_normalize: bool = False, straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None, logits_chunk_size: int | None = None)` — [`L445`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L445) — Perform a single knowledge-distillation training or evaluation step.
- `loss_fn(tree, minibatch)` — [`L576`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L576)
- `teacher_forward(minibatch: collections.abc.Mapping[str, jax.Array])` — [`L529`](../../../../../../../raw/code/EasyDeL/easydel/trainers/distillation_trainer/_fn.py#L529)

