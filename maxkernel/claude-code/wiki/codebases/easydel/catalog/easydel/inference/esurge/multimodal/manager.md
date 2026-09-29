---
title: 'Module: easydel/inference/esurge/multimodal/manager.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/multimodal/manager.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.multimodal.manager`/
symbols:
  MultiModalManager.process_images: MultiModalManager#process_images().
  MultiModalManager.process_videos: MultiModalManager#process_videos().
  MultiModalManager.tokenize_multimodal: MultiModalManager#tokenize_multimodal().
  MultiModalManager.processor: MultiModalManager#processor.
  MultiModalManager.model: MultiModalManager#model.
  MultiModalManager._get_resize_buckets_for_model: MultiModalManager#_get_resize_buckets_for_model().
  MultiModalManager._supports_flat_patch_inputs: MultiModalManager#_supports_flat_patch_inputs().
  MultiModalManager.process_images_to_features: MultiModalManager#process_images_to_features().
  MultiModalManager.process_videos_to_features: MultiModalManager#process_videos_to_features().
  MultiModalManager._normalize_rgb: MultiModalManager#_normalize_rgb().
  MultiModalManager.cache: MultiModalManager#cache.
  MultiModalManager.extract_media_from_messages: MultiModalManager#extract_media_from_messages().
  MultiModalManager._get_vision_config: MultiModalManager#_get_vision_config().
  MultiModalManager: MultiModalManager#
  MultiModalManager.clear_cache: MultiModalManager#clear_cache().
  MultiModalManager.get_cache_stats: MultiModalManager#get_cache_stats().
  MultiModalManager.resolution_buckets: MultiModalManager#resolution_buckets.
  MultiModalManager._placeholder_text: MultiModalManager#_placeholder_text().
  MultiModalManager._pad_flat_patches_for_merge: MultiModalManager#_pad_flat_patches_for_merge().
  MultiModalManager._normalize_messages_for_chat_template: MultiModalManager#_normalize_messages_for_chat_template().
  MultiModalManager._get_text_tokenizer: MultiModalManager#_get_text_tokenizer().
  MultiModalManager._token_str_for_id: MultiModalManager#_token_str_for_id().
  MultiModalManager.resize_to_bucket: MultiModalManager#resize_to_bucket().
  MultiModalManager._align_dim_to_multiple: MultiModalManager#_align_dim_to_multiple().
  MultiModalManager._patchify_spatiotemporal: MultiModalManager#_patchify_spatiotemporal().
  MultiModalManager._counts: MultiModalManager#_counts().
  CLIP_IMAGE_MEAN: CLIP_IMAGE_MEAN.
  CLIP_IMAGE_STD: CLIP_IMAGE_STD.
  DEFAULT_RESOLUTION_BUCKETS: DEFAULT_RESOLUTION_BUCKETS.
  MultiModalManager._resize_to_buckets: MultiModalManager#_resize_to_buckets().
  MultiModalManager._append_image: MultiModalManager#_append_image().
  MultiModalManager._append_image_url: MultiModalManager#_append_image_url().
  MultiModalManager.__init__: MultiModalManager#__init__().
---
# Module: [`easydel/inference/esurge/multimodal/manager.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py)

## Classes
### `MultiModalManager`
- def: [`easydel/inference/esurge/multimodal/manager.py:120`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L120)
- doc: Manager for multimodal (vision-language) processing.
- signature: `class MultiModalManager:`
- members:
  - `__init__(self, processor: Any | None = None, model: Any | None = None, resolution_buckets: list[tuple[int, int]] | None = None, cache_capacity_mb: int = 1024, enable_cache: bool = True)` — [`L140`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L140) — Initialize MultiModalManager.
  - `_align_dim_to_multiple(dim: int, multiple: int)` — [`L210`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L210) — Align a dimension to the nearest multiple.
  - `_get_resize_buckets_for_model(self)` — [`L236`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L236) — Return effective resize buckets for the current model/processor.
  - `_get_text_tokenizer(self)` — [`L456`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L456) — Return a tokenizer-like object for chat templating and text tokenization.
  - `_get_vision_config(self)` — [`L434`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L434) — Get the vision config from the model.
  - `_normalize_messages_for_chat_template(self, messages: list[dict], cfg: Any, tokenizer: Any)` — [`L376`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L376) — Convert OpenAI-style multimodal content into template-friendly text.
  - `_normalize_rgb(self, rgb: np.ndarray)` — [`L663`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L663) — Apply CLIP-style normalization to RGB values.
  - `_pad_flat_patches_for_merge(self, pixel_values: np.ndarray, grid_thw: np.ndarray, *, spatial_merge_size: int)` — [`L503`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L503) — Ensure flat-patch inputs are compatible with spatial merge.
  - `_patchify_spatiotemporal(self, frames: np.ndarray, *, patch_size: int, temporal_patch_size: int, spatial_merge_size: int = 1)` — [`L584`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L584) — Convert frames into flattened patches for GLM/Qwen-style vision towers.
  - `_placeholder_text(self, cfg: Any, tokenizer: Any, kind: str)` — [`L339`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L339) — Build a textual placeholder sequence for an image/video item.
  - `_resize_to_buckets(self, image: Image.Image, buckets: list[tuple[int, int]])` — [`L280`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L280) — Resize image to the nearest resolution bucket.
  - `_supports_flat_patch_inputs(self)` — [`L480`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L480) — Best-effort detection for models expecting flattened patch tokens.
  - `_token_str_for_id(self, tokenizer: Any, token_id: int | None)` — [`L301`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L301) — Convert a token ID to its string representation.
  - `clear_cache(self)` — [`L1233`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L1233) — Clear the vision encoder cache.
  - `extract_media_from_messages(self, messages: list[dict])` — [`L921`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L921) — Extract images and videos from OpenAI-style messages.
  - `get_cache_stats(self)` — [`L1400`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L1400) — Get vision encoder cache statistics.
  - `process_images(self, images: list[Image.Image] | None)` — [`L684`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L684) — Process images with resolution bucketing.
  - `process_images_to_features(self, images: list[Image.Image] | None, request_idx: int = 0)` — [`L1247`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L1247) — Process images and create MultiModalFeature objects.
  - `process_videos(self, videos: list[np.ndarray] | None)` — [`L796`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L796) — Process videos with spatial resolution bucketing.
  - `process_videos_to_features(self, videos: list[np.ndarray] | None, request_idx: int = 0)` — [`L1323`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L1323) — Process videos and create MultiModalFeature objects.
  - `resize_to_bucket(self, image: Image.Image)` — [`L186`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L186) — Resize image to nearest resolution bucket.
  - `tokenize_multimodal(self, messages: list[dict], images: list[Image.Image] | None = None, videos: list[np.ndarray] | None = None, image_grid_thw: np.ndarray | None = None, video_grid_thw: np.ndarray | None = None)` — [`L1033`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L1033) — Tokenize multimodal messages with placeholder insertion.
  - `cache` — [`L184`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L184)
  - `model` — [`L182`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L182)
  - `processor` — [`L181`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L181)
  - `resolution_buckets` — [`L183`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L183)
- protocol/private: `_append_image`[`L960`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L960), `_append_image_url`[`L982`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L982), `_counts`[`L1161`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L1161)
- uses (calls/refs, reference-scoped): [`get_stats`](cache.md#VisionEncoderCache.get_stats), [`MultiModalFeature`](types.md#MultiModalFeature), [`clear`](cache.md#VisionEncoderCache.clear), [`from_image`](types.md#MultiModalFeature.from_image), [`from_video`](types.md#MultiModalFeature.from_video), [`VisionEncoderCache`](cache.md#VisionEncoderCache), [`CLIP_IMAGE_MEAN`](manager.md#CLIP_IMAGE_MEAN), [`CLIP_IMAGE_STD`](manager.md#CLIP_IMAGE_STD), [`DEFAULT_RESOLUTION_BUCKETS`](manager.md#DEFAULT_RESOLUTION_BUCKETS)
- used by: [`_worker_startup_timeout`](../esurge_engine.md#eSurge._worker_startup_timeout), [`_multimodal_manager`](../esurge_engine.md#eSurge._multimodal_manager)

## Module values
- `CLIP_IMAGE_MEAN` — [`L98`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L98)
- `CLIP_IMAGE_STD` — [`L101`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L101)
- `DEFAULT_RESOLUTION_BUCKETS` — [`L108`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/multimodal/manager.py#L108)

