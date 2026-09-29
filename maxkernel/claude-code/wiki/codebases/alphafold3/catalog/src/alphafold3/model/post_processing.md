---
title: 'Module: src/alphafold3/model/post_processing.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/post_processing.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.post_processing`/
symbols:
  post_process_inference_result: post_process_inference_result().
  write_output: write_output().
  ProcessedInferenceResult: ProcessedInferenceResult#
  ProcessedInferenceResult.cif: ProcessedInferenceResult#cif.
  ProcessedInferenceResult.structure_confidence_summary_json: ProcessedInferenceResult#structure_confidence_summary_json.
  ProcessedInferenceResult.structure_full_data_json: ProcessedInferenceResult#structure_full_data_json.
  write_output.write_bytes: write_output().write_bytes().
  ProcessedInferenceResult.mean_confidence_1d: ProcessedInferenceResult#mean_confidence_1d.
  ProcessedInferenceResult.ranking_score: ProcessedInferenceResult#ranking_score.
  ProcessedInferenceResult.model_id: ProcessedInferenceResult#model_id.
  write_embeddings: write_embeddings().
---
# Module: [`src/alphafold3/model/post_processing.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/post_processing.py)

## Classes
### `ProcessedInferenceResult`
- def: [`src/alphafold3/model/post_processing.py:34`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/post_processing.py#L34)
- doc: Stores attributes of a processed inference result.
- signature: `class ProcessedInferenceResult:`
- members:
  - `cif` — [`L48`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/post_processing.py#L48)
  - `mean_confidence_1d` — [`L49`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/post_processing.py#L49)
  - `model_id` — [`L53`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/post_processing.py#L53)
  - `ranking_score` — [`L50`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/post_processing.py#L50)
  - `structure_confidence_summary_json` — [`L51`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/post_processing.py#L51)
  - `structure_full_data_json` — [`L52`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/post_processing.py#L52)
- used by: [`post_process_inference_result`](post_processing.md#post_process_inference_result), [`write_output`](post_processing.md#write_output)

## Functions
- `post_process_inference_result(inference_result: model.InferenceResult)` — [`L56`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/post_processing.py#L56) — Returns cif, confidence_1d_json, confidence_2d_json, mean_confidence_1d, and ranking confidence.
- `write_bytes(path: epath.Path, data: bytes)` — [`L115`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/post_processing.py#L115)
- `write_embeddings(embeddings: dict[str, np.ndarray], output_dir: epath.PathLike, name: str | None = None)` — [`L133`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/post_processing.py#L133) — Writes embeddings to a directory.
- `write_output(inference_result: model.InferenceResult, output_dir: epath.PathLike, terms_of_use: str | None = None, name: str | None = None, compress: bool = False)` — [`L98`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/post_processing.py#L98) — Writes processed inference result to a directory.

