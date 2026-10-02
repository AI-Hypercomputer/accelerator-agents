---
title: 'Module: axlearn/common/loss.py'
type: catalog
provenance: extracted
module: axlearn/common/loss.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.loss`/
symbols:
  _reduce_loss: _reduce_loss().
  giou_loss: giou_loss().
  negative_cosine_similarity_loss: negative_cosine_similarity_loss().
  cross_entropy: cross_entropy().
  asymmetric_contrastive_loss_from_logits: asymmetric_contrastive_loss_from_logits().
  ReductionMethod: ReductionMethod#
  huber_loss: huber_loss().
  _weighted_mean: _weighted_mean().
  mean_squared_error: mean_squared_error().
  bilinear_mean_squared_error: bilinear_mean_squared_error().
  l1_loss: l1_loss().
  asymmetric_contrastive_loss_from_features: asymmetric_contrastive_loss_from_features().
  symmetric_contrastive_loss_from_features: symmetric_contrastive_loss_from_features().
  binary_cross_entropy: binary_cross_entropy().
  contrastive_logits: contrastive_logits().
  giou_loss._compute_area: giou_loss()._compute_area().
  symmetric_contrastive_loss_from_logits: symmetric_contrastive_loss_from_logits().
  focal_loss: focal_loss().
  koleo_loss: koleo_loss().
  ranking_pairwise_loss: ranking_pairwise_loss().
  _stable_cross_entropy: _stable_cross_entropy().
  sigmoid_cross_entropy_with_logits: sigmoid_cross_entropy_with_logits().
  categorical_hinge_loss: categorical_hinge_loss().
  _one_hot_with_label_smoothing: _one_hot_with_label_smoothing().
  _stable_cross_entropy_fwd: _stable_cross_entropy_fwd().
  _stable_cross_entropy_bwd: _stable_cross_entropy_bwd().
  giou_loss._valid_boxes: giou_loss()._valid_boxes().
  pairwise_loss: pairwise_loss().
  ReductionMethod.NONE: ReductionMethod#NONE.
  flops_loss: flops_loss().
  large_margin_cosine_loss: large_margin_cosine_loss().
  kl_divergence: kl_divergence().
  ReductionMethod.MEAN: ReductionMethod#MEAN.
  NEG_INF: NEG_INF.
  ReductionMethod.SUM: ReductionMethod#SUM.
---
# Module: [`axlearn/common/loss.py`](../../../../../../raw/code/axlearn/axlearn/common/loss.py)

## Classes
### `ReductionMethod`  ·  implements/extends Enum, str
- def: [`axlearn/common/loss.py:46`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L46)
- signature: `class ReductionMethod(str, enum.Enum):`
- members:
  - `MEAN` — [`L49`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L49)
  - `NONE` — [`L47`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L47)
  - `SUM` — [`L48`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L48)
- used by: [`_reduce_loss`](loss.md#_reduce_loss), [`giou_loss`](loss.md#giou_loss), [`negative_cosine_similarity_loss`](loss.md#negative_cosine_similarity_loss), [`huber_loss`](loss.md#huber_loss)

## Functions
- `_compute_area(y1: Tensor, x1: Tensor, y2: Tensor, x2: Tensor)` — [`L969`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L969)
- `_one_hot_with_label_smoothing(labels: Tensor, num_classes: int, label_smoothing: float = 0)` — [`L221`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L221)
- `_reduce_loss(*, loss: Tensor, reduction: ReductionMethod, sample_weight: Optional[Tensor], eps: float = 0.000001)` — [`L52`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L52) — Reduces loss tensor.
- `_stable_cross_entropy(logits: Tensor, targets: Tensor, z_loss_scale: float)` — [`L234`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L234) — Computes cross entropy loss with stable custom gradient.
- `_stable_cross_entropy_bwd(res: tuple, g: tuple[Tensor, Tensor, Tensor])` — [`L298`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L298) — Backward-mode of `cross_entropy_with_logits`.
- `_stable_cross_entropy_fwd(logits: Tensor, targets: Tensor, z_loss_scale: float = 0)` — [`L271`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L271) — Forward-mode of `cross_entropy_with_logits`.
- `_valid_boxes(y1: Tensor, x1: Tensor, y2: Tensor, x2: Tensor)` — [`L966`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L966)
- `_weighted_mean(arr: Tensor, *, sample_weight: Optional[Tensor] = None, eps: float = 1e-7)` — [`L333`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L333) — Computes the weighted average of arr without dividing by 0.
- `asymmetric_contrastive_loss_from_features(queries: Tensor, positive_keys: Tensor, *, negative_keys: Tensor = None, negative_key_paddings: Tensor = None, temperature: Union[Tensor, float] = 1, soft_labels: Optional[Tensor] = None)` — [`L530`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L530) — Asymmetric contrastive loss from features.
- `asymmetric_contrastive_loss_from_logits(logits: Tensor, *, key_paddings: Tensor = None, temperature: Union[Tensor, float] = 1, soft_labels: Optional[Tensor] = None)` — [`L464`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L464) — Asymmetric contrastive loss from logits.
- `bilinear_mean_squared_error(preds: Tensor, targets: Tensor, *, shape: tuple[int, ...], sample_weight: Optional[Tensor] = None, eps: float = 1e-7)` — [`L383`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L383) — Computes the mean squared error loss after bilinear downsampling to shape `shape`.
- `binary_cross_entropy(logits: Tensor, *, target_labels: Tensor, live_targets: Optional[Tensor] = None)` — [`L176`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L176) — Compute the binary cross entropy loss between logits and targets.
- `categorical_hinge_loss(logits: Tensor, targets: Tensor)` — [`L732`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L732) — Computes the categorical hinge loss between `y_true` & `logits`.
- `contrastive_logits(x: Tensor, y: Tensor)` — [`L442`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L442) — Computes contrastive logits between two tensors.
- `cross_entropy(logits: Tensor, target_labels: Tensor, *, live_targets: Optional[Tensor] = None, z_loss_scale: float = 0, label_smoothing: float = 0, soft_target_labels: Optional[Tensor] = None)` — [`L89`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L89) — Compute the cross entropy loss between logits and target_labels.
- `flops_loss(*, embeddings: Tensor, paddings: Optional[Tensor] = None, sparsity_threshold: float = 0)` — [`L843`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L843) — The FLOPs loss in 'Minimizing FLOPs to learn efficient sparse representations' ICLR2020.
- `focal_loss(logits: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 1.5, sample_weight: Optional[Tensor] = None)` — [`L760`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L760) — Compute the focal loss between `logits` and the golden `targets` values.
- `giou_loss(*, predictions: Tensor, targets: Tensor, reduction: ReductionMethod = ReductionMethod.NONE, sample_weight: Optional[Tensor] = None, eps: float = 1e-7)` — [`L940`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L940) — Generalized Intersection over Union loss.
- `huber_loss(*, predictions: Tensor, targets: Tensor, delta: float = 1, sample_weight: Optional[Tensor] = None, reduce_axis: Optional[int] = None, reduction: ReductionMethod = ReductionMethod.NONE)` — [`L810`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L810) — Extend the optax.huber_loss with more functionalities.
- `kl_divergence(log_predictions: Tensor, targets: Tensor, *, is_log_targets: bool = False)` — [`L1052`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L1052) — Computes the Kullback-Leibler divergence (relative entropy) loss.
- `koleo_loss(embeddings: Tensor, *, normalize_embedding: bool = True, eps: float = 1e-8)` — [`L1091`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L1091) — Compute KoLeo loss for the given embedding.
- `l1_loss(preds: Tensor, targets: Tensor, sample_weight: Optional[Tensor] = None, eps: float = 1e-7)` — [`L419`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L419) — Computes mean l1 loss.
- `large_margin_cosine_loss(logits: Tensor, *, labels: Optional[Tensor] = None, soft_labels: Optional[Tensor] = None, alpha: float = 1, margin: float = 0)` — [`L887`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L887) — Loss based on https://arxiv.org/pdf/1801.09414.pdf.
- `mean_squared_error(preds: Tensor, targets: Tensor, sample_weight: Optional[Tensor] = None, eps: float = 1e-7)` — [`L360`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L360) — Computes mean squared error loss.
- `negative_cosine_similarity_loss(predictions: Tensor, targets: Tensor, *, normalize_embedding: bool = True, eps: float = 1e-8, live_targets: Optional[Tensor] = None, reduction: ReductionMethod = ReductionMethod.MEAN)` — [`L1007`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L1007) — Compute the negative cross similarity loss between predictions and targets.
- `pairwise_loss(*, logits: Tensor, pair_weights: Tensor, loss_scale: Tensor)` — [`L1135`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L1135) — Computes the mean pairwise loss from logits.
- `ranking_pairwise_loss(*, logits: Tensor, ranks: Tensor, loss_scale: Tensor)` — [`L1164`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L1164) — Computes pairwise loss among ranked docs (ranks > 0).
- `sigmoid_cross_entropy_with_logits(logits: Tensor, targets: Tensor)` — [`L724`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L724) — Implementation based on `tf.nn.sigmoid_cross_entropy_with_logits`.
- `symmetric_contrastive_loss_from_features(x: Tensor, y: Tensor, *, x_negatives: Tensor = None, y_negatives: Tensor = None, x_negative_paddings: Tensor = None, y_negative_paddings: Tensor = None, temperature: Union[Tensor, float] = 1, y_as_key_soft_labels: Optional[Tensor] = None, x_as_key_soft_labels: Optional[Tensor] = None)` — [`L634`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L634) — Symmetric contrastive loss from features.
- `symmetric_contrastive_loss_from_logits(x_y_logits: Tensor, y_x_logits: Tensor, *, y_as_key_paddings: Tensor = None, x_as_key_paddings: Tensor = None, temperature: Union[float, Tensor] = 1, y_as_key_soft_labels: Optional[Tensor] = None, x_as_key_soft_labels: Optional[Tensor] = None)` — [`L586`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L586) — Symmetric contrastive loss from logits.

## Module values
- `NEG_INF` — [`L43`](../../../../../../raw/code/axlearn/axlearn/common/loss.py#L43)

