---
title: 'Module: python/sgl_jax/srt/multimodal/common/modality_enum.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/multimodal/common/modality_enum.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.multimodal.common.modality_enum`/
symbols:
  MultimodalDataItem.set_pad_value: MultimodalDataItem#set_pad_value().
  pad_input_tokens: pad_input_tokens().
  MultimodalInputs.from_dict: MultimodalInputs#from_dict().
  MultimodalDataItem.is_image: MultimodalDataItem#is_image().
  MultimodalInputs.merge: MultimodalInputs#merge().
  Modality: Modality#
  MultimodalInputs.mm_items: MultimodalInputs#mm_items.
  MultimodalDataItem.feature: MultimodalDataItem#feature.
  MultimodalDataItem.from_dict: MultimodalDataItem#from_dict().
  MultimodalDataItem.is_audio: MultimodalDataItem#is_audio().
  MultimodalDataItem.is_video: MultimodalDataItem#is_video().
  Modality.all: Modality#all().
  MultimodalDataItem.merge: MultimodalDataItem#merge().
  MultimodalDataItem.is_valid: MultimodalDataItem#is_valid().
  flatten_nested_list: flatten_nested_list().
  hash_feature: hash_feature().
  MultimodalDataItem.modality: MultimodalDataItem#modality.
  tensor_hash: tensor_hash().
  MultimodalDataItem.hash: MultimodalDataItem#hash.
  MultimodalDataItem.pad_value: MultimodalDataItem#pad_value.
  MultimodalDataItem.is_modality: MultimodalDataItem#is_modality().
  MultimodalInputs.contains_image_inputs: MultimodalInputs#contains_image_inputs().
  MultimodalInputs.contains_video_inputs: MultimodalInputs#contains_video_inputs().
  MultimodalInputs.contains_audio_inputs: MultimodalInputs#contains_audio_inputs().
  MultimodalInputs.contains_mm_input: MultimodalInputs#contains_mm_input().
  MultimodalDataItem: MultimodalDataItem#
  data_hash: data_hash().
  MultimodalInputs.num_image_tokens: MultimodalInputs#num_image_tokens.
  MultimodalInputs.mrope_positions: MultimodalInputs#mrope_positions.
  MultimodalInputs.mrope_position_delta: MultimodalInputs#mrope_position_delta.
  MultimodalDataItem.__setitem__: MultimodalDataItem#__setitem__().
  Modality.from_str: Modality#from_str().
  MultimodalDataItem.set: MultimodalDataItem#set().
  MultimodalDataItem.is_empty_list: MultimodalDataItem#is_empty_list().
  Modality.IMAGE: Modality#IMAGE.
  Modality.VIDEO: Modality#VIDEO.
  Modality.AUDIO: Modality#AUDIO.
  MultimodalDataItem.offsets: MultimodalDataItem#offsets.
  MultimodalInputs.image_pad_len: MultimodalInputs#image_pad_len.
  Modality.MULTI_IMAGES: Modality#MULTI_IMAGES.
  MultimodalDataItem.precomputed_embeddings: MultimodalDataItem#precomputed_embeddings.
  MultimodalDataItem.model_specific_data: MultimodalDataItem#model_specific_data.
  MultimodalDataItem.validate: MultimodalDataItem#validate().
  MultimodalInputs: MultimodalInputs#
  TensorTransportMode: TensorTransportMode.
  MultimodalDataItem.__getattr__: MultimodalDataItem#__getattr__().
  MultimodalInputs.im_token_id: MultimodalInputs#im_token_id.
  MultimodalInputs.im_start_id: MultimodalInputs#im_start_id.
  MultimodalInputs.im_end_id: MultimodalInputs#im_end_id.
  MultimodalInputs.slice_start_id: MultimodalInputs#slice_start_id.
  MultimodalInputs.slice_end_id: MultimodalInputs#slice_end_id.
  MultimodalInputs.video_token_id: MultimodalInputs#video_token_id.
  MultimodalInputs.audio_token_id: MultimodalInputs#audio_token_id.
  MultimodalInputs.audio_start_id: MultimodalInputs#audio_start_id.
  MultimodalInputs.audio_end_id: MultimodalInputs#audio_end_id.
---
# Module: [`python/sgl_jax/srt/multimodal/common/modality_enum.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py)

## Classes
### `Modality`  ·  implements/extends Enum
- def: [`python/sgl_jax/srt/multimodal/common/modality_enum.py:151`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L151)
- signature: `class Modality(Enum):`
- members:
  - `all()` — [`L167`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L167)
  - `from_str(modality_str: str)` — [`L158`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L158)
  - `AUDIO` — [`L155`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L155)
  - `IMAGE` — [`L152`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L152)
  - `MULTI_IMAGES` — [`L153`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L153)
  - `VIDEO` — [`L154`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L154)
- used by: [`is_image`](modality_enum.md#MultimodalDataItem.is_image), [`from_dict`](modality_enum.md#MultimodalDataItem.from_dict), [`is_audio`](modality_enum.md#MultimodalDataItem.is_audio), [`is_video`](modality_enum.md#MultimodalDataItem.is_video), [`modality`](modality_enum.md#MultimodalDataItem.modality), [`is_modality`](modality_enum.md#MultimodalDataItem.is_modality)

### `MultimodalDataItem`
- def: [`python/sgl_jax/srt/multimodal/common/modality_enum.py:172`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L172)
- members:
  - `from_dict(obj: dict)` — [`L250`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L250)
  - `is_audio(self)` — [`L233`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L233)
  - `is_empty_list(lst)` — [`L209`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L209)
  - `is_image(self)` — [`L236`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L236)
  - `is_modality(self, modality: Modality)` — [`L230`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L230)
  - `is_valid(self)` — [`L242`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L242)
  - `is_video(self)` — [`L239`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L239)
  - `merge(self, other)` — [`L259`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L259)
  - `set(self, key: str, value: Any)` — [`L205`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L205)
  - `set_pad_value(self)` — [`L214`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L214) — Set padding value after hashing the data first
  - `validate(self)` — [`L245`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L245)
  - `feature` — [`L185`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L185)
  - `hash` — [`L180`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L180)
  - `modality` — [`L179`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L179)
  - `model_specific_data` — [`L191`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L191)
  - `offsets` — [`L182`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L182)
  - `pad_value` — [`L181`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L181)
  - `precomputed_embeddings` — [`L188`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L188)
- protocol/private: `__getattr__`[`L193`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L193), `__setitem__`[`L199`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L199)
- uses (calls/refs, reference-scoped): [`Modality`](modality_enum.md#Modality), [`flatten_nested_list`](modality_enum.md#flatten_nested_list), [`hash_feature`](modality_enum.md#hash_feature), [`AUDIO`](modality_enum.md#Modality.AUDIO), [`IMAGE`](modality_enum.md#Modality.IMAGE), [`VIDEO`](modality_enum.md#Modality.VIDEO), [`MULTI_IMAGES`](modality_enum.md#Modality.MULTI_IMAGES)
- used by: [`from_dict`](modality_enum.md#MultimodalInputs.from_dict), [`pad_input_tokens`](modality_enum.md#pad_input_tokens), [`mm_items`](modality_enum.md#MultimodalInputs.mm_items), [`contains_audio_inputs`](modality_enum.md#MultimodalInputs.contains_audio_inputs), [`contains_image_inputs`](modality_enum.md#MultimodalInputs.contains_image_inputs), [`contains_mm_input`](modality_enum.md#MultimodalInputs.contains_mm_input), [`contains_video_inputs`](modality_enum.md#MultimodalInputs.contains_video_inputs)

### `MultimodalInputs`
- def: [`python/sgl_jax/srt/multimodal/common/modality_enum.py:282`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L282)
- doc: Inputs related to multimodal data
- signature: `class MultimodalInputs:`
- members:
  - `contains_audio_inputs(self)` — [`L357`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L357)
  - `contains_image_inputs(self)` — [`L351`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L351)
  - `contains_mm_input(self)` — [`L360`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L360)
  - `contains_video_inputs(self)` — [`L354`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L354)
  - `from_dict(obj: dict)` — [`L310`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L310)
  - `merge(self, other)` — [`L363`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L363) — Merge multimodal inputs when merging requests
  - `audio_end_id` — [`L303`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L303)
  - `audio_start_id` — [`L302`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L302)
  - `audio_token_id` — [`L301`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L301)
  - `im_end_id` — [`L293`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L293)
  - `im_start_id` — [`L292`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L292)
  - `im_token_id` — [`L291`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L291)
  - `image_pad_len` — [`L287`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L287)
  - `mm_items` — [`L286`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L286)
  - `mrope_position_delta` — [`L307`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L307)
  - `mrope_positions` — [`L306`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L306)
  - `num_image_tokens` — [`L288`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L288)
  - `slice_end_id` — [`L295`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L295)
  - `slice_start_id` — [`L294`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L294)
  - `video_token_id` — [`L298`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L298)
- uses (calls/refs, reference-scoped): [`set_pad_value`](modality_enum.md#MultimodalDataItem.set_pad_value), [`is_image`](modality_enum.md#MultimodalDataItem.is_image), [`from_dict`](modality_enum.md#MultimodalDataItem.from_dict), [`is_audio`](modality_enum.md#MultimodalDataItem.is_audio), [`is_video`](modality_enum.md#MultimodalDataItem.is_video), [`is_valid`](modality_enum.md#MultimodalDataItem.is_valid), [`MultimodalDataItem`](modality_enum.md#MultimodalDataItem)

## Functions
- `data_hash(data: Any)` — [`L38`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L38) — Hash raw data bytes
- `flatten_nested_list(nested_list)` — [`L12`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L12)
- `hash_feature(f: Any)` — [`L24`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L24) — Hash multimodal features
- `pad_input_tokens(input_ids: list[int], mm_items: list[MultimodalDataItem], im_token_id: int = None, video_token_id: int = None, audio_token_id: int = None)` — [`L76`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L76) — Replace multimodal placeholder tokens in input_ids with corresponding pad_values from mm_items.
- `tensor_hash(tensor_list: Any)` — [`L44`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L44) — Hash JAX tensors or tensor lists using CPU-based hashing

## Module values
- `TensorTransportMode` — [`L21`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/multimodal/common/modality_enum.py#L21)

