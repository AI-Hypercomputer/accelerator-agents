---
title: 'Module: src/maxtext/utils/qk_clip_utils.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/qk_clip_utils.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.qk_clip_utils`/
symbols:
  apply_qk_clip.clip_mla_weights: apply_qk_clip().clip_mla_weights().
  apply_qk_clip_nnx.clip_mla_weights: apply_qk_clip_nnx().clip_mla_weights().
  _get_key_name: _get_key_name().
  apply_qk_clip: apply_qk_clip().
  apply_qk_clip_nnx: apply_qk_clip_nnx().
  calculate_max_logit_metric.extract_logits: calculate_max_logit_metric().extract_logits().
  calculate_max_logit_metric: calculate_max_logit_metric().
  _check_attention_type: _check_attention_type().
  _max_logits_at: _max_logits_at().
  _scale_from_max_logits: _scale_from_max_logits().
  _clip_mla_weight: _clip_mla_weight().
---
# Module: [`src/maxtext/utils/qk_clip_utils.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/qk_clip_utils.py)

## Functions
- `_check_attention_type(config)` — [`L58`](../../../../../../../raw/code/maxtext/src/maxtext/utils/qk_clip_utils.py#L58)
- `_clip_mla_weight(layer_name, param, scale, qk_nope)` — [`L91`](../../../../../../../raw/code/maxtext/src/maxtext/utils/qk_clip_utils.py#L91) — Apply the per-head scale to a wq_b or wkv_b kernel.
- `_get_key_name(k)` — [`L22`](../../../../../../../raw/code/maxtext/src/maxtext/utils/qk_clip_utils.py#L22) — Helper to unwrap JAX path keys.
- `_max_logits_at(curr)` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/utils/qk_clip_utils.py#L66) — Read max_logits from a node in the intermediates tree.
- `_scale_from_max_logits(max_logits_batch, tau)` — [`L85`](../../../../../../../raw/code/maxtext/src/maxtext/utils/qk_clip_utils.py#L85)
- `apply_qk_clip(state, intermediate_outputs, config)` — [`L106`](../../../../../../../raw/code/maxtext/src/maxtext/utils/qk_clip_utils.py#L106) — Applies QK-Clip to MLA weights based on max_logits (Linen path).
- `apply_qk_clip_nnx(state, intermediate_outputs, config)` — [`L140`](../../../../../../../raw/code/maxtext/src/maxtext/utils/qk_clip_utils.py#L140) — Applies QK-Clip to MLA weights on an NNX TrainStateNNX.
- `calculate_max_logit_metric(intermediate_outputs)` — [`L31`](../../../../../../../raw/code/maxtext/src/maxtext/utils/qk_clip_utils.py#L31) — Extracts and computes the global maximum logit from intermediate outputs.
- `clip_mla_weights(path, param)` — [`L114`](../../../../../../../raw/code/maxtext/src/maxtext/utils/qk_clip_utils.py#L114)
- `clip_mla_weights(path, param)` — [`L158`](../../../../../../../raw/code/maxtext/src/maxtext/utils/qk_clip_utils.py#L158)
- `extract_logits(path, val)` — [`L42`](../../../../../../../raw/code/maxtext/src/maxtext/utils/qk_clip_utils.py#L42)

