---
title: 'Module: easydel/inference/esurge/multimodal/types.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/multimodal/types.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.multimodal.types`/
symbols:
  MultiModalFeature: MultiModalFeature#
  BatchedMultiModalInputs.from_features: BatchedMultiModalInputs#from_features().
  MultiModalFeature.from_image: MultiModalFeature#from_image().
  MultiModalFeature.from_video: MultiModalFeature#from_video().
  MultiModalFeature.request_idx: MultiModalFeature#request_idx.
  MultiModalFeature.clear_pixel_values: MultiModalFeature#clear_pixel_values().
  PlaceholderRange.end: PlaceholderRange#end().
  BatchedMultiModalInputs.has_vision: BatchedMultiModalInputs#has_vision().
  PlaceholderRange: PlaceholderRange#
  BatchedMultiModalInputs.image_features: BatchedMultiModalInputs#image_features.
  BatchedMultiModalInputs.video_features: BatchedMultiModalInputs#video_features.
  BatchedMultiModalInputs.has_images: BatchedMultiModalInputs#has_images().
  BatchedMultiModalInputs.has_videos: BatchedMultiModalInputs#has_videos().
  MultiModalFeature.pixel_values: MultiModalFeature#pixel_values.
  BatchedMultiModalInputs: BatchedMultiModalInputs#
  MultiModalFeature.placeholder_range: MultiModalFeature#placeholder_range.
  MultiModalFeature.has_cached_embeddings: MultiModalFeature#has_cached_embeddings().
  MultiModalFeature.set_cached_embeddings: MultiModalFeature#set_cached_embeddings().
  BatchedMultiModalInputs.empty: BatchedMultiModalInputs#empty().
  BatchedMultiModalInputs.num_images: BatchedMultiModalInputs#num_images().
  BatchedMultiModalInputs.num_videos: BatchedMultiModalInputs#num_videos().
  BatchedMultiModalInputs.get_request_image_count: BatchedMultiModalInputs#get_request_image_count().
  BatchedMultiModalInputs.get_request_video_count: BatchedMultiModalInputs#get_request_video_count().
  MultiModalFeature.modality: MultiModalFeature#modality.
  MultiModalFeature.cached_embeddings: MultiModalFeature#cached_embeddings.
  MultiModalFeature._compute_hash: MultiModalFeature#_compute_hash().
  PlaceholderRange.offset: PlaceholderRange#offset.
  PlaceholderRange.length: PlaceholderRange#length.
  MultiModalFeature.grid_thw: MultiModalFeature#grid_thw.
  BatchedMultiModalInputs.pixel_values: BatchedMultiModalInputs#pixel_values.
  BatchedMultiModalInputs.pixel_values_videos: BatchedMultiModalInputs#pixel_values_videos.
  BatchedMultiModalInputs.request_to_image_indices: BatchedMultiModalInputs#request_to_image_indices.
  BatchedMultiModalInputs.request_to_video_indices: BatchedMultiModalInputs#request_to_video_indices.
  PlaceholderRange.modality: PlaceholderRange#modality.
  MultiModalFeature.mm_hash: MultiModalFeature#mm_hash.
  BatchedMultiModalInputs.image_grid_thw: BatchedMultiModalInputs#image_grid_thw.
  BatchedMultiModalInputs.video_grid_thw: BatchedMultiModalInputs#video_grid_thw.
---
# Module: [`easydel/inference/esurge/multimodal/types.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py)

## Classes
### `BatchedMultiModalInputs`
- def: [`easydel/inference/esurge/multimodal/types.py:395`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L395)
- doc: Batched multimodal inputs for model forward pass.
- signature: `class BatchedMultiModalInputs:`
- members:
  - `empty(cls)` — [`L563`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L563) — Create an empty BatchedMultiModalInputs.
  - `from_features(cls, features: list[MultiModalFeature])` — [`L470`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L470) — Create batched inputs from a list of features.
  - `get_request_image_count(self, request_idx: int)` — [`L635`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L635) — Get number of images for a specific request.
  - `get_request_video_count(self, request_idx: int)` — [`L651`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L651) — Get number of videos for a specific request.
  - `has_images(self)` — [`L580`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L580) — Check if batch contains any images.
  - `has_videos(self)` — [`L590`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L590) — Check if batch contains any videos.
  - `has_vision(self)` — [`L600`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L600) — Check if batch contains any vision data (images or videos).
  - `num_images(self)` — [`L614`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L614) — Return number of images in the batch.
  - `num_videos(self)` — [`L625`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L625) — Return number of videos in the batch.
  - `image_features` — [`L464`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L464)
  - `image_grid_thw` — [`L461`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L461)
  - `pixel_values` — [`L460`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L460)
  - `pixel_values_videos` — [`L462`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L462)
  - `request_to_image_indices` — [`L466`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L466)
  - `request_to_video_indices` — [`L467`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L467)
  - `video_features` — [`L465`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L465)
  - `video_grid_thw` — [`L463`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L463)
- uses (calls/refs, reference-scoped): [`MultiModalFeature`](types.md#MultiModalFeature), [`request_idx`](types.md#MultiModalFeature.request_idx), [`pixel_values`](types.md#MultiModalFeature.pixel_values), [`modality`](types.md#MultiModalFeature.modality)

### `MultiModalFeature`
- def: [`easydel/inference/esurge/multimodal/types.py:146`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L146)
- doc: Single multimodal feature with metadata for caching and batching.
- signature: `class MultiModalFeature:`
- members:
  - `_compute_hash(pixel_values: np.ndarray)` — [`L308`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L308) — Compute content-based hash for caching.
  - `clear_pixel_values(self)` — [`L373`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L373) — Clear pixel values after encoding to free memory.
  - `from_image(cls, pixel_values: np.ndarray, grid_thw: np.ndarray | None = None, placeholder_range: PlaceholderRange | None = None, request_idx: int = 0)` — [`L211`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L211) — Create a feature from image data.
  - `from_video(cls, pixel_values: np.ndarray, grid_thw: np.ndarray | None = None, placeholder_range: PlaceholderRange | None = None, request_idx: int = 0)` — [`L259`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L259) — Create a feature from video data.
  - `has_cached_embeddings(self)` — [`L340`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L340) — Check if embeddings have been cached.
  - `set_cached_embeddings(self, embeddings: jax.Array)` — [`L354`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L354) — Cache the vision encoder embeddings.
  - `cached_embeddings` — [`L207`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L207)
  - `grid_thw` — [`L205`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L205)
  - `mm_hash` — [`L202`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L202)
  - `modality` — [`L203`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L203)
  - `pixel_values` — [`L204`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L204)
  - `placeholder_range` — [`L206`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L206)
  - `request_idx` — [`L208`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L208)
- uses (calls/refs, reference-scoped): [`PlaceholderRange`](types.md#PlaceholderRange)
- used by: [`clear_vision_data`](../request.md#EngineRequest.clear_vision_data), [`from_features`](types.md#BatchedMultiModalInputs.from_features), [`process_images_to_features`](manager.md#MultiModalManager.process_images_to_features), [`process_videos_to_features`](manager.md#MultiModalManager.process_videos_to_features), [`mm_features`](../request.md#EngineRequest.mm_features), [`__init__`](../request.md#EngineRequest.__init__), [`mm_features`](../scheduler/output.md#NewRequestData.mm_features), [`image_features`](types.md#BatchedMultiModalInputs.image_features), [`video_features`](types.md#BatchedMultiModalInputs.video_features)

### `PlaceholderRange`
- def: [`easydel/inference/esurge/multimodal/types.py:85`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L85)
- doc: Tracks multimodal placeholder positions in a token sequence.
- signature: `class PlaceholderRange:`
- members:
  - `end(self)` — [`L127`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L127) — Return the end position (exclusive) of the placeholder.
  - `length` — [`L123`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L123)
  - `modality` — [`L124`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L124)
  - `offset` — [`L122`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/types.py#L122)
- used by: [`from_image`](types.md#MultiModalFeature.from_image), [`from_video`](types.md#MultiModalFeature.from_video), [`placeholder_range`](types.md#MultiModalFeature.placeholder_range)

