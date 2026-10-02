---
title: 'Module: tpu_inference/models/jax/utils/multi_modal_utils.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/utils/multi_modal_utils.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.utils.multi_modal_utils`/
symbols:
  merge_multimodal_embeddings: merge_multimodal_embeddings().
  flatten_embeddings: flatten_embeddings().
  _embedding_count_expression: _embedding_count_expression().
  sanity_check_mm_encoder_outputs: sanity_check_mm_encoder_outputs().
  flatten_pad_mm_embeds: flatten_pad_mm_embeds().
  MultiModalEmbeddings: MultiModalEmbeddings.
  _merge_multimodal_embeddings: _merge_multimodal_embeddings().
  NestedTensors.NestedTensors: NestedTensors.NestedTensors.
  logger: logger.
  normalize_mm_grid_thw: normalize_mm_grid_thw().
  reshape_mm_tensor: reshape_mm_tensor().
  split_mm_embeddings_by_grid: split_mm_embeddings_by_grid().
---
# Module: [`tpu_inference/models/jax/utils/multi_modal_utils.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/multi_modal_utils.py)

## Functions
- `_embedding_count_expression(embeddings: NestedTensors)` — [`L80`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/multi_modal_utils.py#L80) — Constructs a debugging representation of the number of embeddings in the
- `_merge_multimodal_embeddings(inputs_embeds: jax.Array, is_multimodal: jax.Array, multimodal_embeddings: jax.Array)` — [`L204`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/multi_modal_utils.py#L204) — Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
- `flatten_embeddings(embeddings: NestedTensors)` — [`L68`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/multi_modal_utils.py#L68) — Recursively flattens and concatenates NestedTensors on all but the last
- `flatten_pad_mm_embeds(mm_embeds: list[jax.Array] | None, target_pad_len: int)` — [`L292`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/multi_modal_utils.py#L292)
- `merge_multimodal_embeddings(input_ids: jax.Array, inputs_embeds: jax.Array, multimodal_embeddings: jax.Array, placeholder_token_id: Union[int, list[int]])` — [`L246`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/multi_modal_utils.py#L246) — Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
- `normalize_mm_grid_thw(grid_thw: object)` — [`L93`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/multi_modal_utils.py#L93) — Normalize grid_thw into a tuple-of-tuples.
- `reshape_mm_tensor(mm_input: object, name: str)` — [`L139`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/multi_modal_utils.py#L139) — Normalize multimodal tensor input to a 2D JAX array.
- `sanity_check_mm_encoder_outputs(mm_embeddings: MultiModalEmbeddings, expected_num_items: int)` — [`L41`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/multi_modal_utils.py#L41) — Perform sanity checks for the result of
- `split_mm_embeddings_by_grid(embeddings: jax.Array, grid_thw: tuple[tuple[int, int, int], ...], spatial_merge_size: int, deepstack_embeddings: Optional[list[jax.Array]] = None)` — [`L164`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/multi_modal_utils.py#L164) — Split concatenated multimodal embeddings back into per-item chunks.

## Module values
- `MultiModalEmbeddings` — [`L31`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/multi_modal_utils.py#L31)
- `NestedTensors` — [`L25`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/multi_modal_utils.py#L25)
- `logger` — [`L23`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/multi_modal_utils.py#L23)

