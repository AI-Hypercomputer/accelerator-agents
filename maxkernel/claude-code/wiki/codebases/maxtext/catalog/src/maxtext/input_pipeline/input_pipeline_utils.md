---
title: 'Module: src/maxtext/input_pipeline/input_pipeline_utils.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/input_pipeline_utils.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.input_pipeline_utils`/
symbols:
  HFDataSource._update_shard: HFDataSource#_update_shard().
  PadOrTrimToMaxLength.map: PadOrTrimToMaxLength#map().
  HFDataSource.n_shards: HFDataSource#n_shards.
  PadOrTrimToMaxLength._pad_image_and_mask: PadOrTrimToMaxLength#_pad_image_and_mask().
  HFDataSource.datasets: HFDataSource#datasets.
  SFTPromptMasking.map: SFTPromptMasking#map().
  SFTPromptMaskingVision.map: SFTPromptMaskingVision#map().
  ComputeQwen3OmniPositions.map: ComputeQwen3OmniPositions#map().
  HFDataSource._check_shard_count: HFDataSource#_check_shard_count().
  HFDataSource.dataset_shards: HFDataSource#dataset_shards.
  HFDataSource.__getitem__: HFDataSource#__getitem__().
  _IndexShardDatasetIterator.__next__: _IndexShardDatasetIterator#__next__().
  IndexShardIterDataset.__iter__: IndexShardIterDataset#__iter__().
  ShiftData.map: ShiftData#map().
  TokenizeOp: TokenizeOp().
  HFDataSource.num_threads: HFDataSource#num_threads.
  GCSTFRecordIterDataset.__iter__: GCSTFRecordIterDataset#__iter__().
  NormalizeFeatures.map: NormalizeFeatures#map().
  KeepFeatures.map: KeepFeatures#map().
  Rekey.map: Rekey#map().
  extract_token_ids: extract_token_ids().
  FoldImagesIntoBatch.target_shape: FoldImagesIntoBatch#target_shape.
  INPUT_TOKENS_KEY: INPUT_TOKENS_KEY.
  HFDataSource.data_iters: HFDataSource#data_iters.
  _IndexShardDatasetIterator._next_index: _IndexShardDatasetIterator#_next_index.
  PadOrTrimToMaxLength.config: PadOrTrimToMaxLength#config.
  ComputeQwen3OmniPositions.data_column: ComputeQwen3OmniPositions#data_column.
  _get_completion_in_chat_template: _get_completion_in_chat_template().
  PadOrTrimToMaxLength._pad_text: PadOrTrimToMaxLength#_pad_text().
  shift_and_refine: shift_and_refine().
  HFDataSource.dataloading_host_count: HFDataSource#dataloading_host_count.
  PadOrTrimToMaxLength.max_length: PadOrTrimToMaxLength#max_length.
  pre_process_image_sft: pre_process_image_sft().
  apply_chat_template: apply_chat_template().
  HFNormalizeFeatures.map: HFNormalizeFeatures#map().
  _GCSTFRecordDatasetIterator._reader: _GCSTFRecordDatasetIterator#_reader.
  make_tfrecord_iter_dataset: make_tfrecord_iter_dataset().
  _IndexShardDatasetIterator.get_state: _IndexShardDatasetIterator#get_state().
  _IndexShardDatasetIterator.set_state: _IndexShardDatasetIterator#set_state().
  ParseFeatures.map: ParseFeatures#map().
  ReformatPacking.map: ReformatPacking#map().
  FoldImagesIntoBatch.map: FoldImagesIntoBatch#map().
  Features: Features.
  SFTPromptMasking.max_target_length: SFTPromptMasking#max_target_length.
  SFTPromptMaskingVision.query_column: SFTPromptMaskingVision#query_column.
  SFTPromptMaskingVision.response_column: SFTPromptMaskingVision#response_column.
  SFTPromptMaskingVision.max_target_length: SFTPromptMaskingVision#max_target_length.
  HFNormalizeFeatures.column_name: HFNormalizeFeatures#column_name.
  HFDataSource._split_dataset_by_node: HFDataSource#_split_dataset_by_node.
  HFDataSource.dataloading_host_index: HFDataSource#dataloading_host_index.
  _IndexShardDatasetIterator: _IndexShardDatasetIterator#
  ParseFeatures.data_columns: ParseFeatures#data_columns.
  NormalizeFeatures.column_names: NormalizeFeatures#column_names.
  KeepFeatures.feature_names: KeepFeatures#feature_names.
  PadOrTrimToMaxLength.pad_id: PadOrTrimToMaxLength#pad_id.
  PadOrTrimToMaxLength.max_num_images_per_example: PadOrTrimToMaxLength#max_num_images_per_example.
  PadOrTrimToMaxLength._pad: PadOrTrimToMaxLength#_pad().
  shift_left: shift_left().
  TokenizeOp._process_string: TokenizeOp()._process_string().
  pre_process_image_sft._process_image_fn: pre_process_image_sft()._process_image_fn().
  SFTPromptMasking.text_column_name: SFTPromptMasking#text_column_name.
  SFTPromptMasking.completion_only: SFTPromptMasking#completion_only.
  SFTPromptMasking.unk_id: SFTPromptMasking#unk_id.
  SFTPromptMaskingVision.pad_id: SFTPromptMaskingVision#pad_id.
  HFDataSource.dataset: HFDataSource#dataset.
  _GCSTFRecordReader: _GCSTFRecordReader#
  _GCSTFRecordDatasetIterator: _GCSTFRecordDatasetIterator#
  GCSTFRecordIterDataset: GCSTFRecordIterDataset#
  _IndexShardDatasetIterator._host_index: _IndexShardDatasetIterator#_host_index.
  _IndexShardDatasetIterator._host_count: _IndexShardDatasetIterator#_host_count.
  IndexShardIterDataset._host_index: IndexShardIterDataset#_host_index.
  IndexShardIterDataset._host_count: IndexShardIterDataset#_host_count.
  NormalizeFeatures.tokenize: NormalizeFeatures#tokenize.
  KeepFeatures.tokenize: KeepFeatures#tokenize.
  Rekey.mapping_dict: Rekey#mapping_dict.
  Rekey.keep_old_keys: Rekey#keep_old_keys.
  ReformatPacking.column_names: ReformatPacking#column_names.
  PadOrTrimToMaxLength.add_true_length: PadOrTrimToMaxLength#add_true_length.
  FoldImagesIntoBatch.model_name: FoldImagesIntoBatch#model_name.
  ShiftData.ignored_ids: ShiftData#ignored_ids.
  ShiftData.axis: ShiftData#axis.
  ComputeQwen3OmniPositions.spatial_merge_size: ComputeQwen3OmniPositions#spatial_merge_size.
  ComputeQwen3OmniPositions.position_id_per_seconds: ComputeQwen3OmniPositions#position_id_per_seconds.
  ComputeQwen3OmniPositions.use_audio_in_video: ComputeQwen3OmniPositions#use_audio_in_video.
  normalize_features: normalize_features().
  get_tokenizer: get_tokenizer().
  truncate_to_max_allowable_length: truncate_to_max_allowable_length().
  shift_data_by_truncation: shift_data_by_truncation().
  add_segmentation_and_position: add_segmentation_and_position().
  reformat_prompt: reformat_prompt().
  reformat_response: reformat_response().
  merge_image_columns: merge_image_columns().
  prepare_text_for_image_fusion: prepare_text_for_image_fusion().
  combine_columns: combine_columns().
  is_conversational: is_conversational().
  tokenization: tokenization().
  SFTPromptMasking: SFTPromptMasking#
  SFTPromptMasking.__init__: SFTPromptMasking#__init__().
  SFTPromptMaskingVision: SFTPromptMaskingVision#
  SFTPromptMaskingVision.__init__: SFTPromptMaskingVision#__init__().
  HFNormalizeFeatures: HFNormalizeFeatures#
  HFNormalizeFeatures.__init__: HFNormalizeFeatures#__init__().
  HFDataSource: HFDataSource#
  HFDataSource.__init__: HFDataSource#__init__().
  HFDataSource.max_target_lenth: HFDataSource#max_target_lenth.
  HFDataSource.data_column_names: HFDataSource#data_column_names.
  HFDataSource.__len__: HFDataSource#__len__().
  _GCSTFRecordReader.__init__: _GCSTFRecordReader#__init__().
  _GCSTFRecordReader._reader: _GCSTFRecordReader#_reader.
  _GCSTFRecordDatasetIterator.__init__: _GCSTFRecordDatasetIterator#__init__().
  compute_file_sharding: compute_file_sharding().
  _IndexShardDatasetIterator.__init__: _IndexShardDatasetIterator#__init__().
  IndexShardIterDataset: IndexShardIterDataset#
  IndexShardIterDataset.__init__: IndexShardIterDataset#__init__().
  ParseFeatures: ParseFeatures#
  ParseFeatures.__init__: ParseFeatures#__init__().
  ParseFeatures.tokenize: ParseFeatures#tokenize.
  NormalizeFeatures: NormalizeFeatures#
  NormalizeFeatures.__init__: NormalizeFeatures#__init__().
  KeepFeatures: KeepFeatures#
  KeepFeatures.__init__: KeepFeatures#__init__().
  Rekey: Rekey#
  Rekey.__init__: Rekey#__init__().
  ReformatPacking: ReformatPacking#
  ReformatPacking.__init__: ReformatPacking#__init__().
  PadOrTrimToMaxLength: PadOrTrimToMaxLength#
  PadOrTrimToMaxLength.__init__: PadOrTrimToMaxLength#__init__().
  ExtractImagesAndMasks: ExtractImagesAndMasks#
  ExtractImagesAndMasks.map: ExtractImagesAndMasks#map().
  FoldImagesIntoBatch: FoldImagesIntoBatch#
  FoldImagesIntoBatch.__post_init__: FoldImagesIntoBatch#__post_init__().
  shift_right: shift_right().
  ShiftData: ShiftData#
  ShiftData.__init__: ShiftData#__init__().
  ComputeQwen3OmniPositions: ComputeQwen3OmniPositions#
  ComputeQwen3OmniPositions.__init__: ComputeQwen3OmniPositions#__init__().
---
# Module: [`src/maxtext/input_pipeline/input_pipeline_utils.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py)

## Classes
### `ComputeQwen3OmniPositions`  ·  implements/extends Map
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:980`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L980)
- doc: Computes 3D position IDs for Qwen3-Omni multimodal sequences.
- signature: `class ComputeQwen3OmniPositions(grain.MapTransform):`
- members:
  - `__init__(self, data_column: str = "inputs", spatial_merge_size: int = 2, position_id_per_seconds: int = 25, use_audio_in_video: bool = False)` — [`L994`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L994) — Initialize the Qwen3-Omni position computation transform.
  - `map(self, element: dict[str, np.ndarray])` — [`L1014`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L1014) — Compute 3D position IDs for the batch element.
  - `data_column` — [`L1009`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L1009)
  - `position_id_per_seconds` — [`L1011`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L1011)
  - `spatial_merge_size` — [`L1010`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L1010)
  - `use_audio_in_video` — [`L1012`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L1012)

### `ExtractImagesAndMasks`  ·  implements/extends Map
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:862`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L862)
- doc: Extracts images and masks from a PreprocessorOutput object.
- signature: `class ExtractImagesAndMasks(grain.MapTransform):`
- members:
  - `map(self, element: dict[str, np.ndarray])` — [`L874`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L874) — Applies the extraction transformation to the 'images' field if present.

### `FoldImagesIntoBatch`  ·  implements/extends Map
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:892`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L892)
- doc: Folds the 'image' dimension into the batch dimension.
- signature: `class FoldImagesIntoBatch(grain.MapTransform):`
- members:
  - `__post_init__(self)` — [`L909`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L909) — Initializes the target shape after the dataclass is created.
  - `map(self, element: dict[str, np.ndarray])` — [`L913`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L913) — Applies the folding transformation to the 'images' field if present.
  - `model_name` — [`L907`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L907)
  - `target_shape` — [`L911`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L911)

### `GCSTFRecordIterDataset`  ·  implements/extends TFRecordIterDataset
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:485`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L485)
- doc: Extends Grain's TFRecordIterDataset to support GCS paths.
- signature: `class GCSTFRecordIterDataset(TFRecordIterDataset):`
- protocol/private: `__iter__`[`L488`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L488)
- uses (calls/refs, reference-scoped): [`_GCSTFRecordDatasetIterator`](input_pipeline_utils.md#_GCSTFRecordDatasetIterator), `_path`
- used by: [`make_tfrecord_iter_dataset`](input_pipeline_utils.md#make_tfrecord_iter_dataset)

### `HFDataSource`  ·  implements/extends RandomAccessDataSource
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:389`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L389)
- doc: A class that makes HuggingFace IterableDataset a grain datasource without random access support
- signature: `class HFDataSource(grain.RandomAccessDataSource):`
- members:
  - `__getitem__(self, index)` — [`L449`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L449) — Since HuggingFace IterableDataset does not support random access by index. — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
  - `__len__(self)` — [`L444`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L444) — Return length of the HF dataset. Since HuggingFace IterableDataset does not have length,
  - `_update_shard(self, idx)` — [`L428`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L428) — update shard — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
  - `_split_dataset_by_node` — [`L403`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L403) — --- — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
  - `data_column_names` — [`L409`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L409)
  - `data_iters` — [`L417`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L417) — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
  - `dataloading_host_count` — [`L406`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L406) — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
  - `dataloading_host_index` — [`L407`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L407)
  - `dataset` — [`L404`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L404) — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
  - `dataset_shards` — [`L415`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L415) — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
  - `datasets` — [`L416`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L416) — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
  - `max_target_lenth` — [`L408`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L408)
  - `n_shards` — [`L411`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L411) — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
  - `num_threads` — [`L405`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L405) — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
- protocol/private: `__init__`[`L392`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L392), `_check_shard_count`[`L419`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L419)

### `HFNormalizeFeatures`  ·  implements/extends Map
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:376`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L376)
- doc: Normalize feature keys for HuggingFace input
- signature: `class HFNormalizeFeatures(grain.MapTransform):`
- members:
  - `map(self, element)` — [`L382`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L382)
  - `column_name` — [`L380`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L380)
- protocol/private: `__init__`[`L379`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L379)

### `IndexShardIterDataset`  ·  implements/extends IterDataset
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:552`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L552)
- doc: Shards an IterDataset across hosts by element index (host i keeps records where idx % N == i).
- signature: `class IndexShardIterDataset(grain.IterDataset):`
- protocol/private: `__init__`[`L559`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L559), `__iter__`[`L564`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L564), `_host_count`[`L562`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L562), `_host_index`[`L561`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L561)
- uses (calls/refs, reference-scoped): [`_IndexShardDatasetIterator`](input_pipeline_utils.md#_IndexShardDatasetIterator)

### `KeepFeatures`  ·  implements/extends Map
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:640`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L640)
- doc: Filter dataset elements to specified features for parquet and other non-proto formats.
- signature: `class KeepFeatures(grain.MapTransform):`
- members:
  - `map(self, element: dict[str, Any])` — [`L653`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L653) — Applies the feature filtering to the input element.
  - `feature_names` — [`L650`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L650)
  - `tokenize` — [`L651`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L651)
- protocol/private: `__init__`[`L649`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L649)

### `NormalizeFeatures`  ·  implements/extends Map
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:625`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L625)
- doc: Normalize text feature keys.
- signature: `class NormalizeFeatures(grain.MapTransform):`
- members:
  - `map(self, element)` — [`L632`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L632)
  - `column_names` — [`L629`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L629)
  - `tokenize` — [`L630`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L630)
- protocol/private: `__init__`[`L628`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L628)

### `PadOrTrimToMaxLength`  ·  implements/extends Map
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:719`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L719)
- doc: Pads or trims each input to the specified length.
- signature: `class PadOrTrimToMaxLength(grain.MapTransform):`
- members:
  - `_pad_image_and_mask(self, preprocessed_image: mm_utils.PreprocessorOutput)` — [`L742`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L742) — Pads the input tensors (image and mask) of a PreprocessorOutput to a maximum number of items. — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
  - `map(self, element: dict[str, np.ndarray | mm_utils.PreprocessorOutput])` — [`L833`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L833) — map to each element — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
  - `add_true_length` — [`L734`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L734) — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
  - `config` — [`L733`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L733) — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
  - `max_length` — [`L731`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L731) — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
  - `max_num_images_per_example` — [`L735`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L735) — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
  - `pad_id` — [`L732`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L732) — documented in [maxtext-input_pipeline-input_pipeline_utils](../../../../concepts/maxtext-input_pipeline-input_pipeline_utils.md)
- protocol/private: `__init__`[`L723`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L723), `_pad`[`L796`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L796), `_pad_text`[`L737`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L737)

### `ParseFeatures`  ·  implements/extends Map
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:573`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L573)
- doc: Parse serialized tf.train.Example protos for arrayrecord/tfrecord datasets.
- signature: `class ParseFeatures(grain.MapTransform):`
- members:
  - `map(self, element)` — [`L585`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L585) — Parse a serialized tf.train.Example proto and extract features.
  - `data_columns` — [`L582`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L582)
  - `tokenize` — [`L583`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L583)
- protocol/private: `__init__`[`L581`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L581)

### `ReformatPacking`  ·  implements/extends Map
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:703`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L703)
- doc: Reformat packing outputs.
- signature: `class ReformatPacking(grain.MapTransform):`
- members:
  - `map(self, element)` — [`L709`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L709)
  - `column_names` — [`L707`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L707)
- protocol/private: `__init__`[`L706`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L706)

### `Rekey`  ·  implements/extends Map
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:684`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L684)
- doc: Rename keys according to a mapping dict
- signature: `class Rekey(grain.MapTransform):`
- members:
  - `map(self, element)` — [`L691`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L691)
  - `keep_old_keys` — [`L689`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L689)
  - `mapping_dict` — [`L688`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L688)
- protocol/private: `__init__`[`L687`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L687)

### `SFTPromptMasking`  ·  implements/extends Map
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:324`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L324)
- doc: Construct inputs and targets for SFT training. Concat prompt and completion to generate inputs.
- signature: `class SFTPromptMasking(grain.MapTransform):`
- members:
  - `map(self, element)` — [`L335`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L335) — Maps a single dataset element to an SFT training instance.
  - `completion_only` — [`L331`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L331)
  - `max_target_length` — [`L332`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L332)
  - `text_column_name` — [`L330`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L330)
  - `unk_id` — [`L333`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L333)
- protocol/private: `__init__`[`L329`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L329)

### `SFTPromptMaskingVision`  ·  implements/extends Map
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:356`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L356)
- doc: SFT prompt masking for multimodal
- signature: `class SFTPromptMaskingVision(grain.MapTransform):`
- members:
  - `map(self, element)` — [`L365`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L365)
  - `max_target_length` — [`L362`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L362)
  - `pad_id` — [`L363`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L363)
  - `query_column` — [`L360`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L360)
  - `response_column` — [`L361`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L361)
- protocol/private: `__init__`[`L359`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L359)

### `ShiftData`  ·  implements/extends Map
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:968`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L968)
- doc: Shift inputs and refine annotations.
- signature: `class ShiftData(grain.MapTransform):`
- members:
  - `map(self, element)` — [`L975`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L975)
  - `axis` — [`L973`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L973)
  - `ignored_ids` — [`L972`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L972)
- protocol/private: `__init__`[`L971`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L971)
- uses (calls/refs, reference-scoped): [`shift_and_refine`](input_pipeline_utils.md#shift_and_refine)

### `_GCSTFRecordDatasetIterator`  ·  implements/extends _TFRecordDatasetIterator
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:476`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L476)
- doc: Extends Grain's _TFRecordDatasetIterator to use _GCSTFRecordReader for GCS paths.
- signature: `class _GCSTFRecordDatasetIterator(_TFRecordDatasetIterator):`
- protocol/private: `__init__`[`L479`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L479), `_reader`[`L482`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L482)
- uses (calls/refs, reference-scoped): [`_GCSTFRecordReader`](input_pipeline_utils.md#_GCSTFRecordReader)
- used by: [`__iter__`](input_pipeline_utils.md#GCSTFRecordIterDataset.__iter__)

### `_GCSTFRecordReader`  ·  implements/extends _TFRecordReader
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:467`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L467)
- doc: Extends Grain's _TFRecordReader to open TFRecord files from GCS via streaming BlobReader.
- signature: `class _GCSTFRecordReader(_TFRecordReader):`
- protocol/private: `__init__`[`L470`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L470), `_reader`[`L473`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L473)
- used by: [`_reader`](input_pipeline_utils.md#_GCSTFRecordDatasetIterator._reader)

### `_IndexShardDatasetIterator`  ·  implements/extends DatasetIterator
- def: [`src/maxtext/input_pipeline/input_pipeline_utils.py:524`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L524)
- doc: Iterator that yields every nth element of its parent (round-robin by index).
- signature: `class _IndexShardDatasetIterator(grain.DatasetIterator):`
- members:
  - `get_state(self)` — [`L541`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L541)
  - `set_state(self, state)` — [`L547`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L547)
- protocol/private: `__init__`[`L527`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L527), `__next__`[`L533`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L533), `_host_count`[`L530`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L530), `_host_index`[`L529`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L529), `_next_index`[`L531`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L531)
- used by: [`__iter__`](input_pipeline_utils.md#IndexShardIterDataset.__iter__)

## Functions
- `TokenizeOp(tokenizer_model, features: Features, data_keys: Iterable[str] = ("inputs", "targets"))` — [`L74`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L74) — Op for tokenization
- `_get_completion_in_chat_template(tokenizer_model, round_msgs)` — [`L211`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L211) — Calculates the completion part of a conversation turn formatted with a chat template.
- `_process_image_fn(image)` — [`L125`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L125)
- `_process_string(string_tensor)` — [`L78`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L78)
- `add_segmentation_and_position(x, data_columns, padding_token=0)` — [`L63`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L63)
- `apply_chat_template(example, tokenizer_model, data_column_name)` — [`L260`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L260) — Formats conversational data by applying the tokenizer's chat template
- `combine_columns(example, columns, data_column)` — [`L146`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L146) — Combine columns such as 'prompt' and 'completion' for sft training
- `compute_file_sharding(file_count, host_index, host_count)` — [`L499`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L499) — Compute per-host file slicing and optional row-shard parameters.
- `extract_token_ids(tokens)` — [`L181`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L181) — Extracts token IDs from various tokenizer output formats.
- `get_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token=None)` — [`L47`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L47)
- `is_conversational(features, data_columns)` — [`L157`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L157) — Check if data is in a conversational format.
- `make_tfrecord_iter_dataset(path: str)` — [`L492`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L492) — Returns the appropriate TFRecordIterDataset for local or GCS paths.
- `merge_image_columns(example, image_columns, max_num_images_per_example)` — [`L109`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L109) — Merge multiple image columns into a single list of images.
- `normalize_features(x, column_name)` — [`L43`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L43)
- `pre_process_image_sft(example, image_column, config)` — [`L122`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L122) — pre-process image for multimodal SFT
- `prepare_text_for_image_fusion(example, column_name, config)` — [`L138`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L138) — prepare text for image fusion for multimodal SFT
- `reformat_prompt(example, column, image_placeholder, model_name)` — [`L93`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L93) — reformat prompt for multimodal SFT
- `reformat_response(example, column, model_name)` — [`L103`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L103) — reformat response for multimodal SFT
- `shift_and_refine(x, ignored_ids, axis=1)` — [`L957`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L957) — Shift inputs, set segmentation to 0 when target element is in ignored_ids if provided
- `shift_data_by_truncation(x)` — [`L57`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L57)
- `shift_left(x, pad_id, axis=1)` — [`L945`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L945) — Shift to the left and pad.
- `shift_right(x, axis=1)` — [`L933`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L933) — Shift the input to the right by padding and slicing on axis.
- `tokenization(example, hf_tokenizer, truncation, max_length, column_names)` — [`L311`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L311) — Tokenize a HuggingFace dataset
- `truncate_to_max_allowable_length(x, max_length)` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L53)

## Module values
- `Features` — [`L37`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L37)
- `INPUT_TOKENS_KEY` — [`L38`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_utils.py#L38)

