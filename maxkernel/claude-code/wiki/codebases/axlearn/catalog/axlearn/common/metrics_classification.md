---
title: 'Module: axlearn/common/metrics_classification.py'
type: catalog
provenance: extracted
module: axlearn/common/metrics_classification.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.metrics_classification`/
symbols:
  precision_recall_f_score: precision_recall_f_score().
  _compute_area_under_the_curve: _compute_area_under_the_curve().
  roc_curve: roc_curve().
  f_score: f_score().
  precision_recall_curve: precision_recall_curve().
  binary_classification_roc_auc_score: binary_classification_roc_auc_score().
  binary_clf_curve: binary_clf_curve().
  confusion_matrix: confusion_matrix().
  brier_score: brier_score().
---
# Module: [`axlearn/common/metrics_classification.py`](../../../../../../raw/code/axlearn/axlearn/common/metrics_classification.py)

## Functions
- `_compute_area_under_the_curve(y_true: Tensor, y_score: Tensor, sample_weight: Optional[Tensor] = None)` — [`L282`](../../../../../../raw/code/axlearn/axlearn/common/metrics_classification.py#L282) — Helper function to compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).
- `binary_classification_roc_auc_score(y_true: Tensor, y_score: Tensor, sample_weight: Optional[Tensor] = None)` — [`L234`](../../../../../../raw/code/axlearn/axlearn/common/metrics_classification.py#L234) — Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC) for binary
- `binary_clf_curve(y_true: Tensor, y_score: Tensor, *, weight: Optional[Tensor] = None)` — [`L118`](../../../../../../raw/code/axlearn/axlearn/common/metrics_classification.py#L118) — Calculate true and false positives per binary classification threshold.
- `brier_score(labels: Tensor, logits: Tensor)` — [`L330`](../../../../../../raw/code/axlearn/axlearn/common/metrics_classification.py#L330) — Compute Brier score for a probabilistic prediction.
- `confusion_matrix(y_true: Tensor, y_pred: Tensor, *, num_classes: int, weight: Optional[Tensor] = None)` — [`L23`](../../../../../../raw/code/axlearn/axlearn/common/metrics_classification.py#L23) — Computes confusion matrix.
- `f_score(y_true: Tensor, y_pred: Tensor, *, beta: float = 1, eps: float = 1e-8, weight: Optional[Tensor] = None)` — [`L102`](../../../../../../raw/code/axlearn/axlearn/common/metrics_classification.py#L102) — Computes F-beta score for the positive class in binary classification.
- `precision_recall_curve(y_true: Tensor, y_score: Tensor, *, weight: Optional[Tensor] = None)` — [`L182`](../../../../../../raw/code/axlearn/axlearn/common/metrics_classification.py#L182) — Compute precision-recall pairs for different probability thresholds.
- `precision_recall_f_score(y_true: Tensor, y_pred: Tensor, *, beta: float = 1, eps: float = 1e-8, weight: Optional[Tensor] = None)` — [`L56`](../../../../../../raw/code/axlearn/axlearn/common/metrics_classification.py#L56) — Computes precision, recall, and F-beta score for binary classification.
- `roc_curve(y_true: Tensor, y_score: Tensor, sample_weight: Optional[Tensor] = None)` — [`L296`](../../../../../../raw/code/axlearn/axlearn/common/metrics_classification.py#L296) — Computes Receiver Operating Characteristic (ROC).

