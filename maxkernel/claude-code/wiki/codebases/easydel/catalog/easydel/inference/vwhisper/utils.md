---
title: 'Module: easydel/inference/vwhisper/utils.py'
type: catalog
provenance: extracted
module: easydel/inference/vwhisper/utils.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.vwhisper.utils`/
symbols:
  chunk_iter_with_batch: chunk_iter_with_batch().
  process_audio_input: process_audio_input().
---
# Module: [`easydel/inference/vwhisper/utils.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/utils.py)

## Functions
- `chunk_iter_with_batch(audio_array: np.ndarray, chunk_length: int, stride_left: int, stride_right: int, batch_size: int, feature_extractor)` — [`L57`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/utils.py#L57) — Generate batched chunks from a long audio array with overlapping strides.
- `process_audio_input(audio_input: str | bytes | np.ndarray | dict[str, np.ndarray | int], feature_extractor)` — [`L164`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/utils.py#L164) — Process various audio input formats into a normalized numpy array.

