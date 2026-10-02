---
title: 'Module: src/maxtext/utils/vocabulary_tiling.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/vocabulary_tiling.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.vocabulary_tiling`/
symbols:
  vocab_tiling_nnx_loss: vocab_tiling_nnx_loss().
  vocab_tiling_linen_loss: vocab_tiling_linen_loss().
  vocab_tiling_linen_loss._chunked_cross_entropy_loss_fwd: vocab_tiling_linen_loss()._chunked_cross_entropy_loss_fwd().
  vocab_tiling_nnx_loss._chunked_cross_entropy_loss_fwd: vocab_tiling_nnx_loss()._chunked_cross_entropy_loss_fwd().
  _is_output_head_param_path: _is_output_head_param_path().
  vocab_tiling_linen_loss._chunked_cross_entropy_loss_bwd: vocab_tiling_linen_loss()._chunked_cross_entropy_loss_bwd().
  vocab_tiling_nnx_loss._chunked_cross_entropy_loss_bwd: vocab_tiling_nnx_loss()._chunked_cross_entropy_loss_bwd().
  vocab_tiling_linen_loss.chunked_cross_entropy_loss: vocab_tiling_linen_loss().chunked_cross_entropy_loss().
  vocab_tiling_nnx_loss.chunked_cross_entropy_loss: vocab_tiling_nnx_loss().chunked_cross_entropy_loss().
  vocab_tiling_linen_loss._reshape: vocab_tiling_linen_loss()._reshape().
  vocab_tiling_nnx_loss._reshape: vocab_tiling_nnx_loss()._reshape().
  vocab_tiling_linen_loss._chunked_cross_entropy_loss_bwd._bwd_scan_body: vocab_tiling_linen_loss()._chunked_cross_entropy_loss_bwd()._bwd_scan_body().
  vocab_tiling_nnx_loss._chunked_cross_entropy_loss_fwd._fwd_scan_body: vocab_tiling_nnx_loss()._chunked_cross_entropy_loss_fwd()._fwd_scan_body().
  vocab_tiling_nnx_loss._chunked_cross_entropy_loss_bwd._single_chunk_loss_fn: vocab_tiling_nnx_loss()._chunked_cross_entropy_loss_bwd()._single_chunk_loss_fn().
  vocab_tiling_nnx_loss._chunked_cross_entropy_loss_bwd._bwd_scan_body: vocab_tiling_nnx_loss()._chunked_cross_entropy_loss_bwd()._bwd_scan_body().
  vocab_tiling_nnx_loss._logits_for_chunk: vocab_tiling_nnx_loss()._logits_for_chunk().
  _OUTPUT_HEAD_PATH_KEYS: _OUTPUT_HEAD_PATH_KEYS.
  _is_output_head_param_path._name: _is_output_head_param_path()._name().
  vocab_tiling_linen_loss._chunked_cross_entropy_loss_fwd._fwd_scan_body: vocab_tiling_linen_loss()._chunked_cross_entropy_loss_fwd()._fwd_scan_body().
  vocab_tiling_linen_loss._chunked_cross_entropy_loss_bwd._single_chunk_loss_fn: vocab_tiling_linen_loss()._chunked_cross_entropy_loss_bwd()._single_chunk_loss_fn().
---
# Module: [`src/maxtext/utils/vocabulary_tiling.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py)

## Functions
- `_bwd_scan_body(grad_params_acc, chunk_data)` — [`L217`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L217)
- `_bwd_scan_body(grad_head_acc, chunk_data)` — [`L444`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L444)
- `_chunked_cross_entropy_loss_bwd(residuals, cotangents)` — [`L196`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L196)
- `_chunked_cross_entropy_loss_bwd(residuals, cotangents)` — [`L422`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L422)
- `_chunked_cross_entropy_loss_fwd(gathered_params, hidden_states, labels, segmentation)` — [`L142`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L142)
- `_chunked_cross_entropy_loss_fwd(chunk_head_params, chunk_other_params, chunk_rest, hidden_states, labels, segmentation)` — [`L373`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L373)
- `_fwd_scan_body(accumulators, chunk_data)` — [`L153`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L153)
- `_fwd_scan_body(accumulators, chunk_data)` — [`L385`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L385)
- `_is_output_head_param_path(path, _value)` — [`L39`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L39) — Filter for nnx.split: True when the param path belongs to the output head.
- `_logits_for_chunk(chunk_head_params, chunk_other_params, chunk_rest, hidden_chunk)` — [`L361`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L361)
- `_name(k)` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L45)
- `_reshape(inputs, out_shape, out_sharding)` — [`L122`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L122)
- `_reshape(inputs, out_shape, out_sharding)` — [`L337`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L337)
- `_single_chunk_loss_fn(input_params, input_hidden_chunk, input_label_chunk, input_segmentation_chunk)` — [`L205`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L205)
- `_single_chunk_loss_fn(input_head_params, input_hidden_chunk, input_label_chunk, input_segmentation_chunk)` — [`L438`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L438)
- `chunked_cross_entropy_loss(gathered_params, hidden_states, labels, segmentation)` — [`L135`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L135) — Calculates the total cross-entropy loss using vocab tiling.
- `chunked_cross_entropy_loss(chunk_head_params, chunk_other_params, chunk_rest, hidden_states, labels, segmentation)` — [`L367`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L367)
- `vocab_tiling_linen_loss(hidden_states, data, config, model, params, is_train)` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L56) — Calculates cross-entropy loss using vocab tiling for Linen models.
- `vocab_tiling_nnx_loss(model, hidden_states, data, config, is_train)` — [`L276`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L276) — Computes cross-entropy loss with vocab tiling for NNX models.

## Module values
- `_OUTPUT_HEAD_PATH_KEYS` — [`L36`](../../../../../../../raw/code/maxtext/src/maxtext/utils/vocabulary_tiling.py#L36)

