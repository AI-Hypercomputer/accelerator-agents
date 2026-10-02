---
title: 'Module: simply/data_lib.py'
type: catalog
provenance: extracted
module: simply/data_lib.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.data_lib`/
symbols:
  create_iter_dataset: create_iter_dataset().
  _create_map_dataset: _create_map_dataset().
  DatasetConfig: DatasetConfig#
  DataSourceRegistry: DataSourceRegistry#
  register_spm_vocabs: register_spm_vocabs().
  _to_fixed_length: _to_fixed_length().
  ChatFormatTransform.map: ChatFormatTransform#map().
  PACKING_NONE: PACKING_NONE.
  get_data_source: get_data_source().
  VOCABS_DIR: VOCABS_DIR.
  TokenizeTransform.map: TokenizeTransform#map().
  GSM8KSource._examples: GSM8KSource#_examples().
  GSM8KSFTSource._examples: GSM8KSFTSource#_examples().
  MMLUSource._examples: MMLUSource#_examples().
  AIME24Source._examples: AIME24Source#_examples().
  QualityTrainSource._examples: QualityTrainSource#_examples().
  register_hf_vocabs: register_hf_vocabs().
  TFDSSource: TFDSSource#
  SimpleQANumSource: SimpleQANumSource#
  HFSource._source: HFSource#_source().
  DeepScaleRSource._examples: DeepScaleRSource#_examples().
  MATH500Source._examples: MATH500Source#_examples().
  GPQADiamondSource._examples: GPQADiamondSource#_examples().
  DATASETS_DIR: DATASETS_DIR.
  HF_VOCABS: HF_VOCABS.
  _register_gsm8k_variants: _register_gsm8k_variants().
  _get_tokenizer: _get_tokenizer().
  GSM8KTestSource: GSM8KTestSource#
  GSM8KSFTTestSource: GSM8KSFTTestSource#
  AIME25Source: AIME25Source#
  QualityValSource: QualityValSource#
  PadTransform.map: PadTransform#map().
  TFDSSource._source: TFDSSource#_source().
  GSM8KSource: GSM8KSource#
  SimpleQASource._examples: SimpleQASource#_examples().
  AIME24Source: AIME24Source#
  create_iter_dataset._pack: create_iter_dataset()._pack().
  DatasetConfigRegistry: DatasetConfigRegistry#
  OPENMIX_V1_VOCABS: OPENMIX_V1_VOCABS.
  QualityTrainSource: QualityTrainSource#
  QualityTrainSource._source: QualityTrainSource#_source().
  _get_lm_format: _get_lm_format().
  PACKING_PAD_OR_TRUNCATE: PACKING_PAD_OR_TRUNCATE.
  get_batch_fn: get_batch_fn().
  grain_iter_global_state._is_leaf: grain_iter_global_state()._is_leaf().
  HFSource: HFSource#
  ArrayRecordSource: ArrayRecordSource#
  BagzSource: BagzSource#
  DatasetConfig.trainable_roles: DatasetConfig#trainable_roles.
  MixtureConfig: MixtureConfig#
  GSM8KTestSource.split: GSM8KTestSource#split.
  GSM8KSFTTestSource.split: GSM8KSFTTestSource#split.
  SimpleQANumSource.path: SimpleQANumSource#path.
  DeepScaleRSource: DeepScaleRSource#
  AIME25Source.year: AIME25Source#year.
  AIME26Source.year: AIME26Source#year.
  GPQADiamondSource: GPQADiamondSource#
  pt_dataset_v1: pt_dataset_v1().
  TokenizeTransform.tokenizer_name: TokenizeTransform#tokenizer_name.
  TokenizeTransform.add_eos: TokenizeTransform#add_eos.
  PadTransform.seq_len: PadTransform#seq_len.
  PadTransform.pad_id: PadTransform#pad_id.
  NumpyTransform.type_cast_map: NumpyTransform#type_cast_map.
  QWEN2P5_VOCAB: QWEN2P5_VOCAB.
  ArrayRecordSource._source: ArrayRecordSource#_source().
  BagzSource._source: BagzSource#_source().
  MixtureConfig.datasets: MixtureConfig#datasets.
  create_iter_dataset._finalize: create_iter_dataset()._finalize().
  TokenizeTransform: TokenizeTransform#
  TokenizeTransform.add_bos: TokenizeTransform#add_bos.
  NextTokenPredTransform: NextTokenPredTransform#
  TruncateTransform: TruncateTransform#
  PadTransform: PadTransform#
  RekeyTransform.key_map: RekeyTransform#key_map.
  PACKING_CONCAT_SPLIT: PACKING_CONCAT_SPLIT.
  PACKING_FIRST_FIT: PACKING_FIRST_FIT.
  BATCH_UNSTACKED: BATCH_UNSTACKED.
  OPENMIX_V1_32768_VOCAB: OPENMIX_V1_32768_VOCAB.
  OPENMIX_V1_100864_VOCAB: OPENMIX_V1_100864_VOCAB.
  OPENMIX_V2_EDU_100864_V1P1_VOCAB: OPENMIX_V2_EDU_100864_V1P1_VOCAB.
  OPENMIX_V3_100864_V2_VOCAB: OPENMIX_V3_100864_V2_VOCAB.
  GEMMA2_VOCAB: GEMMA2_VOCAB.
  GEMMA3_VOCAB: GEMMA3_VOCAB.
  DEEPSEEK_R1_DISTILL_QWEN_VOCAB: DEEPSEEK_R1_DISTILL_QWEN_VOCAB.
  QWEN3_VOCAB: QWEN3_VOCAB.
  OPENMIX_V2_VOCABS: OPENMIX_V2_VOCABS.
  OPENMIX_V3_VOCABS: OPENMIX_V3_VOCABS.
  GEMMA2_VOCABS: GEMMA2_VOCABS.
  GEMMA3_VOCABS: GEMMA3_VOCABS.
  GSM8KSource.path: GSM8KSource#path.
  GSM8KSFTSource.path: GSM8KSFTSource#path.
  SimpleQASource.path: SimpleQASource#path.
  MMLUSource.path: MMLUSource#path.
  DeepScaleRSource.path: DeepScaleRSource#path.
  AIME24Source.path: AIME24Source#path.
  MATH500Source.path: MATH500Source#path.
  GPQADiamondSource.path: GPQADiamondSource#path.
  grain_iter_global_state: grain_iter_global_state().
  grain_iter_local_state: grain_iter_local_state().
  ArrayRecordSource.paths: ArrayRecordSource#paths.
  BagzSource.paths: BagzSource#paths.
  ChatFormatTransform.tokenizer_name: ChatFormatTransform#tokenizer_name.
  ChatFormatTransform.lm_format_name: ChatFormatTransform#lm_format_name.
  ChatFormatTransform.add_bos: ChatFormatTransform#add_bos.
  ChatFormatTransform.trainable_roles: ChatFormatTransform#trainable_roles.
  TruncateTransform.seq_len: TruncateTransform#seq_len.
  NumpyTransform: NumpyTransform#
  BATCH_STACKED: BATCH_STACKED.
  FWEDU_100864_V1_VOCAB: FWEDU_100864_V1_VOCAB.
  OPENMIX_V2_EDU_100864_VOCAB: OPENMIX_V2_EDU_100864_VOCAB.
  OPENMIX_V3_100864_V1_VOCAB: OPENMIX_V3_100864_V1_VOCAB.
  QWQ_VOCAB: QWQ_VOCAB.
  TFDSSource.__len__: TFDSSource#__len__().
  TFDSSource.__getitem__: TFDSSource#__getitem__().
  HFSource.__len__: HFSource#__len__().
  HFSource.__getitem__: HFSource#__getitem__().
  ArrayRecordSource.__len__: ArrayRecordSource#__len__().
  ArrayRecordSource.__getitem__: ArrayRecordSource#__getitem__().
  BagzSource.__len__: BagzSource#__len__().
  BagzSource.__getitem__: BagzSource#__getitem__().
  DatasetConfig.source: DatasetConfig#source.
  MixtureConfig.__post_init__: MixtureConfig#__post_init__().
  GSM8KSource.__len__: GSM8KSource#__len__().
  GSM8KSource.__getitem__: GSM8KSource#__getitem__().
  GSM8KSFTSource.__len__: GSM8KSFTSource#__len__().
  GSM8KSFTSource.__getitem__: GSM8KSFTSource#__getitem__().
  SimpleQASource.__len__: SimpleQASource#__len__().
  SimpleQASource.__getitem__: SimpleQASource#__getitem__().
  MMLUSource.__len__: MMLUSource#__len__().
  MMLUSource.__getitem__: MMLUSource#__getitem__().
  DeepScaleRSource.__len__: DeepScaleRSource#__len__().
  DeepScaleRSource.__getitem__: DeepScaleRSource#__getitem__().
  AIME24Source.__len__: AIME24Source#__len__().
  AIME24Source.__getitem__: AIME24Source#__getitem__().
  AIME26Source: AIME26Source#
  MATH500Source.__len__: MATH500Source#__len__().
  MATH500Source.__getitem__: MATH500Source#__getitem__().
  GPQADiamondSource.__len__: GPQADiamondSource#__len__().
  GPQADiamondSource.__getitem__: GPQADiamondSource#__getitem__().
  QualityTrainSource.__len__: QualityTrainSource#__len__().
  QualityTrainSource.__getitem__: QualityTrainSource#__getitem__().
  TruncateTransform.map: TruncateTransform#map().
  RekeyTransform.map: RekeyTransform#map().
  NumpyTransform.map: NumpyTransform#map().
  SimpleDataSource: SimpleDataSource#
  GSM8KSource.split: GSM8KSource#split.
  GSM8KSFTSource.split: GSM8KSFTSource#split.
  SimpleQASource.split: SimpleQASource#split.
  MMLUSource.split: MMLUSource#split.
  AIME24Source.year: AIME24Source#year.
  QualityTrainSource.split: QualityTrainSource#split.
  TokenizeTransform.data_key: TokenizeTransform#data_key.
  ChatFormatTransform: ChatFormatTransform#
  ChatFormatTransform.data_key: ChatFormatTransform#data_key.
  RekeyTransform: RekeyTransform#
  create_iter_dataset._shard: create_iter_dataset()._shard().
  TFDSSource.name: TFDSSource#name.
  TFDSSource.split: TFDSSource#split.
  HFSource.name: HFSource#name.
  HFSource.split: HFSource#split.
  HFSource.subset: HFSource#subset.
  GSM8KSource.start_index: GSM8KSource#start_index.
  GSM8KSource.end_index: GSM8KSource#end_index.
  GSM8KSFTSource: GSM8KSFTSource#
  GSM8KSFTSource.start_index: GSM8KSFTSource#start_index.
  GSM8KSFTSource.end_index: GSM8KSFTSource#end_index.
  SimpleQASource: SimpleQASource#
  MMLUSource.start_index: MMLUSource#start_index.
  MMLUSource.end_index: MMLUSource#end_index.
  DeepScaleRSource.start_index: DeepScaleRSource#start_index.
  DeepScaleRSource.end_index: DeepScaleRSource#end_index.
  AIME24Source.start_index: AIME24Source#start_index.
  AIME24Source.end_index: AIME24Source#end_index.
  MATH500Source.start_index: MATH500Source#start_index.
  MATH500Source.end_index: MATH500Source#end_index.
  GPQADiamondSource.start_index: GPQADiamondSource#start_index.
  GPQADiamondSource.end_index: GPQADiamondSource#end_index.
  QualityTrainSource.start_index: QualityTrainSource#start_index.
  QualityTrainSource.end_index: QualityTrainSource#end_index.
  TFExampleDeserializeTransform: TFExampleDeserializeTransform#
  DataSourceRegistry.namespace: DataSourceRegistry#namespace.
  DatasetConfigRegistry.namespace: DatasetConfigRegistry#namespace.
  SimpleDataSource.__len__: SimpleDataSource#__len__().
  SimpleDataSource.__getitem__: SimpleDataSource#__getitem__().
  DatasetConfig.lm_format_name: DatasetConfig#lm_format_name.
  DatasetConfig.packing: DatasetConfig#packing.
  DatasetConfig.data_key: DatasetConfig#data_key.
  DatasetConfig.tokenizer_name: DatasetConfig#tokenizer_name.
  DatasetConfig.add_eos: DatasetConfig#add_eos.
  DatasetConfig.add_bos: DatasetConfig#add_bos.
  MixtureConfig.pack_before_mix: MixtureConfig#pack_before_mix.
  MMLUSource: MMLUSource#
  MATH500Source: MATH500Source#
  QualityValSource.split: QualityValSource#split.
  TFExampleDeserializeTransform.map: TFExampleDeserializeTransform#map().
  NextTokenPredTransform.map: NextTokenPredTransform#map().
---
# Module: [`simply/data_lib.py`](../../../../../raw/code/simply/simply/data_lib.py)

## Classes
### `AIME24Source`
- def: [`simply/data_lib.py:655`](../../../../../raw/code/simply/simply/data_lib.py#L655)
- doc: AIME 2024 dataset source with lazy loading.
- signature: `class AIME24Source:`
- members:
  - `_examples(self)` — [`L664`](../../../../../raw/code/simply/simply/data_lib.py#L664) — Lazily loads and caches examples.
  - `end_index` — [`L661`](../../../../../raw/code/simply/simply/data_lib.py#L661)
  - `path` — [`L658`](../../../../../raw/code/simply/simply/data_lib.py#L658)
  - `start_index` — [`L660`](../../../../../raw/code/simply/simply/data_lib.py#L660)
  - `year` — [`L659`](../../../../../raw/code/simply/simply/data_lib.py#L659)
- protocol/private: `__getitem__`[`L683`](../../../../../raw/code/simply/simply/data_lib.py#L683), `__len__`[`L680`](../../../../../raw/code/simply/simply/data_lib.py#L680)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry), [`DATASETS_DIR`](data_lib.md#DATASETS_DIR)
- used by: [`AIME25Source`](data_lib.md#AIME25Source), [`AIME26Source`](data_lib.md#AIME26Source)

### `AIME25Source`
- def: [`simply/data_lib.py:689`](../../../../../raw/code/simply/simply/data_lib.py#L689)
- doc: AIME 2025 dataset source.
- signature: `class AIME25Source(AIME24Source):`
- members:
  - `year` — [`L692`](../../../../../raw/code/simply/simply/data_lib.py#L692)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry), [`AIME24Source`](data_lib.md#AIME24Source)

### `AIME26Source`
- def: [`simply/data_lib.py:697`](../../../../../raw/code/simply/simply/data_lib.py#L697)
- doc: AIME 2026 dataset source.
- signature: `class AIME26Source(AIME24Source):`
- members:
  - `year` — [`L700`](../../../../../raw/code/simply/simply/data_lib.py#L700)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry), [`AIME24Source`](data_lib.md#AIME24Source)

### `ArrayRecordSource`
- def: [`simply/data_lib.py:243`](../../../../../raw/code/simply/simply/data_lib.py#L243)
- doc: ArrayRecord data source with glob expansion and lazy loading.
- signature: `class ArrayRecordSource:`
- members:
  - `_source(self)` — [`L279`](../../../../../raw/code/simply/simply/data_lib.py#L279) — Lazily expands paths and creates a Grain ArrayRecordDataSource.
  - `paths` — [`L276`](../../../../../raw/code/simply/simply/data_lib.py#L276)
- protocol/private: `__getitem__`[`L294`](../../../../../raw/code/simply/simply/data_lib.py#L294), `__len__`[`L291`](../../../../../raw/code/simply/simply/data_lib.py#L291)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry)

### `BagzSource`
- def: [`simply/data_lib.py:300`](../../../../../raw/code/simply/simply/data_lib.py#L300)
- doc: Bagz data source with glob expansion and lazy loading.
- signature: `class BagzSource:`
- members:
  - `_source(self)` — [`L335`](../../../../../raw/code/simply/simply/data_lib.py#L335) — Lazily expands paths and creates a Grain BagDataSource.
  - `paths` — [`L332`](../../../../../raw/code/simply/simply/data_lib.py#L332)
- protocol/private: `__getitem__`[`L351`](../../../../../raw/code/simply/simply/data_lib.py#L351), `__len__`[`L348`](../../../../../raw/code/simply/simply/data_lib.py#L348)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry)

### `ChatFormatTransform`  ·  implements/extends Map
- def: [`simply/data_lib.py:1010`](../../../../../raw/code/simply/simply/data_lib.py#L1010)
- doc: Formats chat conversations with role tokens and tokenizes.
- signature: `class ChatFormatTransform(grain.MapTransform):`
- members:
  - `map(self, features: Mapping[str, Any])` — [`L1046`](../../../../../raw/code/simply/simply/data_lib.py#L1046)
  - `add_bos` — [`L1043`](../../../../../raw/code/simply/simply/data_lib.py#L1043)
  - `data_key` — [`L1042`](../../../../../raw/code/simply/simply/data_lib.py#L1042)
  - `lm_format_name` — [`L1041`](../../../../../raw/code/simply/simply/data_lib.py#L1041)
  - `tokenizer_name` — [`L1040`](../../../../../raw/code/simply/simply/data_lib.py#L1040)
  - `trainable_roles` — [`L1044`](../../../../../raw/code/simply/simply/data_lib.py#L1044)
- uses (calls/refs, reference-scoped): [`_get_tokenizer`](data_lib.md#_get_tokenizer), [`_get_lm_format`](data_lib.md#_get_lm_format)
- used by: [`_create_map_dataset`](data_lib.md#_create_map_dataset), [`test_trainable_roles_creates_loss_mask`](data_lib_test.md#ChatFormatTransformTest.test_trainable_roles_creates_loss_mask)

### `DataSourceRegistry`  ·  implements/extends RootRegistry
- def: [`simply/data_lib.py:142`](../../../../../raw/code/simply/simply/data_lib.py#L142) — documented in [simply-data_lib](../../concepts/simply-data_lib.md)
- doc: Registry for data sources (lazy-evaluated).
- signature: `class DataSourceRegistry(registry.RootRegistry):`
- members:
  - `namespace` — [`L145`](../../../../../raw/code/simply/simply/data_lib.py#L145)
- uses (calls/refs, reference-scoped): [`RootRegistry`](utils/registry.md#RootRegistry)
- used by: [`main`](eval/page_decode_eval.md#main), [`main`](eval/decode_eval.md#main), [`RootRegistry`](utils/registry.md#RootRegistry), [`main`](eval/remote_decode_eval.md#main), [`get_data_source`](data_lib.md#get_data_source), [`setUpClass`](data_lib_test.md#MixtureDatasetTest.setUpClass), [`SimpleQANumSource`](data_lib.md#SimpleQANumSource), [`TFDSSource`](data_lib.md#TFDSSource), [`setUpClass`](data_lib_test.md#CreateIterDatasetTest.setUpClass), [`_register_gsm8k_variants`](data_lib.md#_register_gsm8k_variants), [`AIME25Source`](data_lib.md#AIME25Source), [`GSM8KSFTTestSource`](data_lib.md#GSM8KSFTTestSource), [`GSM8KTestSource`](data_lib.md#GSM8KTestSource), [`QualityValSource`](data_lib.md#QualityValSource), [`AIME24Source`](data_lib.md#AIME24Source), [`GSM8KSource`](data_lib.md#GSM8KSource), [`QualityTrainSource`](data_lib.md#QualityTrainSource), [`ArrayRecordSource`](data_lib.md#ArrayRecordSource), [`BagzSource`](data_lib.md#BagzSource), [`DeepScaleRSource`](data_lib.md#DeepScaleRSource), [`GPQADiamondSource`](data_lib.md#GPQADiamondSource), [`HFSource`](data_lib.md#HFSource), [`id`](rl_lib_test.md#MockDeepScaleRJSONExample.id), [`path`](data_lib.md#SimpleQANumSource.path), [`split`](data_lib.md#GSM8KSFTTestSource.split), [`split`](data_lib.md#GSM8KTestSource.split), [`year`](data_lib.md#AIME25Source.year), [`year`](data_lib.md#AIME26Source.year)

### `DatasetConfig`
- def: [`simply/data_lib.py:357`](../../../../../raw/code/simply/simply/data_lib.py#L357) — documented in [simply-data_lib](../../concepts/simply-data_lib.md)
- doc: Configuration for a single dataset.
- signature: `class DatasetConfig:`
- members:
  - `add_bos` — [`L420`](../../../../../raw/code/simply/simply/data_lib.py#L420)
  - `add_eos` — [`L419`](../../../../../raw/code/simply/simply/data_lib.py#L419)
  - `data_key` — [`L417`](../../../../../raw/code/simply/simply/data_lib.py#L417)
  - `lm_format_name` — [`L415`](../../../../../raw/code/simply/simply/data_lib.py#L415)
  - `packing` — [`L416`](../../../../../raw/code/simply/simply/data_lib.py#L416)
  - `source` — [`L414`](../../../../../raw/code/simply/simply/data_lib.py#L414)
  - `tokenizer_name` — [`L418`](../../../../../raw/code/simply/simply/data_lib.py#L418)
  - `trainable_roles` — [`L421`](../../../../../raw/code/simply/simply/data_lib.py#L421)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DatasetConfigRegistry`](data_lib.md#DatasetConfigRegistry), [`SimpleDataSource`](data_lib.md#SimpleDataSource)
- used by: [`_create_map_dataset`](data_lib.md#_create_map_dataset), [`deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl`](config_lib.md#deepseek_qwen2_1p5b_it_dsr40k_r1_distill_cot_0shot_rl), [`gemma3_4b_it_simple_qa_number_only_tool_use_rl`](config_lib.md#gemma3_4b_it_simple_qa_number_only_tool_use_rl), [`lm_rl_test`](config_lib.md#lm_rl_test), [`gemma2_2b_gsm8k_0shot_rl`](config_lib.md#gemma2_2b_gsm8k_0shot_rl), [`flops6e20_tfm2b_c4_l2048`](config_lib.md#flops6e20_tfm2b_c4_l2048), [`lm_test`](config_lib.md#lm_test), [`test_pack_before_mix`](data_lib_test.md#MixtureDatasetTest.test_pack_before_mix), [`test_run_experiment_saves_checkpoint`](rl_lib_test.md#RunExperimentTest.test_run_experiment_saves_checkpoint), [`gemma2_2b_dsr40k_0shot_rl`](config_lib.md#gemma2_2b_dsr40k_0shot_rl), [`gemma2_2b_it_dsr40k_0shot_rl`](config_lib.md#gemma2_2b_it_dsr40k_0shot_rl), [`gemma2_2b_c4_vocab100864_l2048_bs1024`](config_lib.md#gemma2_2b_c4_vocab100864_l2048_bs1024), [`gemma2_2b_gsm8k_32examples_rl`](config_lib.md#gemma2_2b_gsm8k_32examples_rl), [`gemma2_2b_it_dsr40k_math500_0shot_no_ref_rl`](config_lib.md#gemma2_2b_it_dsr40k_math500_0shot_no_ref_rl), [`qwen3_4b_gsm8k_sft`](config_lib.md#qwen3_4b_gsm8k_sft), [`test_batch_contents_correct`](data_lib_test.md#CreateIterDatasetTest.test_batch_contents_correct), [`test_mixture_produces_batches`](data_lib_test.md#MixtureDatasetTest.test_mixture_produces_batches), [`test_validation_mode_produces_finite_batches`](data_lib_test.md#CreateIterDatasetTest.test_validation_mode_produces_finite_batches), [`lm_sft_test`](config_lib.md#lm_sft_test), [`test_produces_correct_batch_shape`](data_lib_test.md#CreateIterDatasetTest.test_produces_correct_batch_shape), [`pt_dataset_v1`](data_lib.md#pt_dataset_v1), [`test_requires_positive_weights`](data_lib_test.md#MixtureConfigValidationTest.test_requires_positive_weights), [`test_unknown_source_raises_error`](data_lib_test.md#GetDataSourceTest.test_unknown_source_raises_error), [`datasets`](data_lib.md#MixtureConfig.datasets)

### `DatasetConfigRegistry`  ·  implements/extends RootRegistry
- def: [`simply/data_lib.py:148`](../../../../../raw/code/simply/simply/data_lib.py#L148)
- doc: Registry for data configuration dataclasses.
- signature: `class DatasetConfigRegistry(registry.RootRegistry):`
- members:
  - `namespace` — [`L151`](../../../../../raw/code/simply/simply/data_lib.py#L151)
- uses (calls/refs, reference-scoped): [`RootRegistry`](utils/registry.md#RootRegistry)
- used by: [`RootRegistry`](utils/registry.md#RootRegistry), [`create_iter_dataset`](data_lib.md#create_iter_dataset), [`DatasetConfig`](data_lib.md#DatasetConfig), [`trainable_roles`](data_lib.md#DatasetConfig.trainable_roles)

### `DeepScaleRSource`
- def: [`simply/data_lib.py:623`](../../../../../raw/code/simply/simply/data_lib.py#L623)
- doc: DeepScaleR dataset source with lazy loading.
- signature: `class DeepScaleRSource:`
- members:
  - `_examples(self)` — [`L631`](../../../../../raw/code/simply/simply/data_lib.py#L631) — Lazily loads and caches examples.
  - `end_index` — [`L628`](../../../../../raw/code/simply/simply/data_lib.py#L628)
  - `path` — [`L626`](../../../../../raw/code/simply/simply/data_lib.py#L626)
  - `start_index` — [`L627`](../../../../../raw/code/simply/simply/data_lib.py#L627)
- protocol/private: `__getitem__`[`L649`](../../../../../raw/code/simply/simply/data_lib.py#L649), `__len__`[`L646`](../../../../../raw/code/simply/simply/data_lib.py#L646)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry), [`DATASETS_DIR`](data_lib.md#DATASETS_DIR)

### `GPQADiamondSource`
- def: [`simply/data_lib.py:740`](../../../../../raw/code/simply/simply/data_lib.py#L740)
- doc: GPQA-Diamond dataset source with lazy loading.
- signature: `class GPQADiamondSource:`
- members:
  - `_examples(self)` — [`L748`](../../../../../raw/code/simply/simply/data_lib.py#L748) — Lazily loads and caches examples from the GPQA-Diamond dataset.
  - `end_index` — [`L745`](../../../../../raw/code/simply/simply/data_lib.py#L745)
  - `path` — [`L743`](../../../../../raw/code/simply/simply/data_lib.py#L743)
  - `start_index` — [`L744`](../../../../../raw/code/simply/simply/data_lib.py#L744)
- protocol/private: `__getitem__`[`L769`](../../../../../raw/code/simply/simply/data_lib.py#L769), `__len__`[`L766`](../../../../../raw/code/simply/simply/data_lib.py#L766)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry), [`DATASETS_DIR`](data_lib.md#DATASETS_DIR)

### `GSM8KSFTSource`
- def: [`simply/data_lib.py:516`](../../../../../raw/code/simply/simply/data_lib.py#L516)
- doc: GSM8K dataset formatted as conversations for SFT.
- signature: `class GSM8KSFTSource:`
- members:
  - `_examples(self)` — [`L525`](../../../../../raw/code/simply/simply/data_lib.py#L525) — Lazily load and cache examples as conversations.
  - `end_index` — [`L522`](../../../../../raw/code/simply/simply/data_lib.py#L522)
  - `path` — [`L519`](../../../../../raw/code/simply/simply/data_lib.py#L519)
  - `split` — [`L520`](../../../../../raw/code/simply/simply/data_lib.py#L520)
  - `start_index` — [`L521`](../../../../../raw/code/simply/simply/data_lib.py#L521)
- protocol/private: `__getitem__`[`L547`](../../../../../raw/code/simply/simply/data_lib.py#L547), `__len__`[`L544`](../../../../../raw/code/simply/simply/data_lib.py#L544)
- uses (calls/refs, reference-scoped): [`DATASETS_DIR`](data_lib.md#DATASETS_DIR)
- used by: [`GSM8KSFTTestSource`](data_lib.md#GSM8KSFTTestSource)

### `GSM8KSFTTestSource`
- def: [`simply/data_lib.py:553`](../../../../../raw/code/simply/simply/data_lib.py#L553)
- doc: GSM8K test split formatted as conversations for SFT.
- signature: `class GSM8KSFTTestSource(GSM8KSFTSource):`
- members:
  - `split` — [`L556`](../../../../../raw/code/simply/simply/data_lib.py#L556)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry), [`GSM8KSFTSource`](data_lib.md#GSM8KSFTSource)

### `GSM8KSource`
- def: [`simply/data_lib.py:480`](../../../../../raw/code/simply/simply/data_lib.py#L480)
- doc: GSM8K dataset source with lazy loading.
- signature: `class GSM8KSource:`
- members:
  - `_examples(self)` — [`L489`](../../../../../raw/code/simply/simply/data_lib.py#L489) — Lazily load and cache examples.
  - `end_index` — [`L486`](../../../../../raw/code/simply/simply/data_lib.py#L486)
  - `path` — [`L483`](../../../../../raw/code/simply/simply/data_lib.py#L483)
  - `split` — [`L484`](../../../../../raw/code/simply/simply/data_lib.py#L484)
  - `start_index` — [`L485`](../../../../../raw/code/simply/simply/data_lib.py#L485)
- protocol/private: `__getitem__`[`L502`](../../../../../raw/code/simply/simply/data_lib.py#L502), `__len__`[`L499`](../../../../../raw/code/simply/simply/data_lib.py#L499)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry), [`DATASETS_DIR`](data_lib.md#DATASETS_DIR)
- used by: [`_register_gsm8k_variants`](data_lib.md#_register_gsm8k_variants), [`GSM8KTestSource`](data_lib.md#GSM8KTestSource)

### `GSM8KTestSource`
- def: [`simply/data_lib.py:508`](../../../../../raw/code/simply/simply/data_lib.py#L508)
- doc: GSM8K test split.
- signature: `class GSM8KTestSource(GSM8KSource):`
- members:
  - `split` — [`L511`](../../../../../raw/code/simply/simply/data_lib.py#L511)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry), [`GSM8KSource`](data_lib.md#GSM8KSource)

### `HFSource`
- def: [`simply/data_lib.py:202`](../../../../../raw/code/simply/simply/data_lib.py#L202)
- doc: HuggingFace data source with lazy loading.
- signature: `class HFSource:`
- members:
  - `name` — [`L224`](../../../../../raw/code/simply/simply/data_lib.py#L224)
  - `split` — [`L225`](../../../../../raw/code/simply/simply/data_lib.py#L225)
  - `subset` — [`L226`](../../../../../raw/code/simply/simply/data_lib.py#L226)
- protocol/private: `__getitem__`[`L237`](../../../../../raw/code/simply/simply/data_lib.py#L237), `__len__`[`L234`](../../../../../raw/code/simply/simply/data_lib.py#L234), `_source`[`L229`](../../../../../raw/code/simply/simply/data_lib.py#L229)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry)

### `MATH500Source`
- def: [`simply/data_lib.py:705`](../../../../../raw/code/simply/simply/data_lib.py#L705)
- doc: MATH500 test set source with lazy loading.
- signature: `class MATH500Source:`
- members:
  - `_examples(self)` — [`L713`](../../../../../raw/code/simply/simply/data_lib.py#L713) — Lazily loads and caches examples from the MATH500 test set.
  - `end_index` — [`L710`](../../../../../raw/code/simply/simply/data_lib.py#L710)
  - `path` — [`L708`](../../../../../raw/code/simply/simply/data_lib.py#L708)
  - `start_index` — [`L709`](../../../../../raw/code/simply/simply/data_lib.py#L709)
- protocol/private: `__getitem__`[`L734`](../../../../../raw/code/simply/simply/data_lib.py#L734), `__len__`[`L731`](../../../../../raw/code/simply/simply/data_lib.py#L731)
- uses (calls/refs, reference-scoped): [`DATASETS_DIR`](data_lib.md#DATASETS_DIR)

### `MMLUSource`
- def: [`simply/data_lib.py:596`](../../../../../raw/code/simply/simply/data_lib.py#L596)
- doc: MMLU dataset source with lazy loading.
- signature: `class MMLUSource:`
- members:
  - `end_index` — [`L602`](../../../../../raw/code/simply/simply/data_lib.py#L602)
  - `path` — [`L599`](../../../../../raw/code/simply/simply/data_lib.py#L599)
  - `split` — [`L600`](../../../../../raw/code/simply/simply/data_lib.py#L600)
  - `start_index` — [`L601`](../../../../../raw/code/simply/simply/data_lib.py#L601)
- protocol/private: `__getitem__`[`L617`](../../../../../raw/code/simply/simply/data_lib.py#L617), `__len__`[`L614`](../../../../../raw/code/simply/simply/data_lib.py#L614), `_examples`[`L605`](../../../../../raw/code/simply/simply/data_lib.py#L605)
- uses (calls/refs, reference-scoped): [`DATASETS_DIR`](data_lib.md#DATASETS_DIR)

### `MixtureConfig`
- def: [`simply/data_lib.py:426`](../../../../../raw/code/simply/simply/data_lib.py#L426)
- doc: Configuration for a mixture of datasets.
- signature: `class MixtureConfig:`
- members:
  - `datasets` — [`L462`](../../../../../raw/code/simply/simply/data_lib.py#L462)
  - `pack_before_mix` — [`L463`](../../../../../raw/code/simply/simply/data_lib.py#L463)
- protocol/private: `__post_init__`[`L465`](../../../../../raw/code/simply/simply/data_lib.py#L465)
- uses (calls/refs, reference-scoped): [`DatasetConfig`](data_lib.md#DatasetConfig)
- used by: [`create_iter_dataset`](data_lib.md#create_iter_dataset), [`test_pack_before_mix`](data_lib_test.md#MixtureDatasetTest.test_pack_before_mix), [`test_mixture_produces_batches`](data_lib_test.md#MixtureDatasetTest.test_mixture_produces_batches), [`test_requires_positive_weights`](data_lib_test.md#MixtureConfigValidationTest.test_requires_positive_weights), [`test_requires_datasets`](data_lib_test.md#MixtureConfigValidationTest.test_requires_datasets)

### `NextTokenPredTransform`  ·  implements/extends Map
- def: [`simply/data_lib.py:971`](../../../../../raw/code/simply/simply/data_lib.py#L971)
- doc: Converts tokens to next-token prediction format for decoder-only LMs.
- signature: `class NextTokenPredTransform(grain.MapTransform):`
- members:
  - `map(self, features: Mapping[str, Any])` — [`L986`](../../../../../raw/code/simply/simply/data_lib.py#L986)
- used by: [`_create_map_dataset`](data_lib.md#_create_map_dataset), [`test_loss_mask_shifted_correctly`](data_lib_test.md#NextTokenPredTransformTest.test_loss_mask_shifted_correctly), [`test_shifts_tokens_correctly`](data_lib_test.md#NextTokenPredTransformTest.test_shifts_tokens_correctly)

### `NumpyTransform`  ·  implements/extends Map
- def: [`simply/data_lib.py:1147`](../../../../../raw/code/simply/simply/data_lib.py#L1147)
- doc: Transform list-types to Numpy arrays.
- signature: `class NumpyTransform(grain.MapTransform):`
- members:
  - `map(self, features: Mapping[str, Any])` — [`L1152`](../../../../../raw/code/simply/simply/data_lib.py#L1152)
  - `type_cast_map` — [`L1150`](../../../../../raw/code/simply/simply/data_lib.py#L1150)
- used by: [`test_no_type_cast_map`](data_lib_test.md#NumpyTransformTest.test_no_type_cast_map), [`test_type_cast_map`](data_lib_test.md#NumpyTransformTest.test_type_cast_map), [`test_converts_to_numpy`](data_lib_test.md#NumpyTransformTest.test_converts_to_numpy)

### `PadTransform`  ·  implements/extends Map
- def: [`simply/data_lib.py:1099`](../../../../../raw/code/simply/simply/data_lib.py#L1099)
- doc: Pads sequences to a fixed length.
- signature: `class PadTransform(grain.MapTransform):`
- members:
  - `map(self, features: Mapping[str, Any])` — [`L1113`](../../../../../raw/code/simply/simply/data_lib.py#L1113)
  - `pad_id` — [`L1111`](../../../../../raw/code/simply/simply/data_lib.py#L1111)
  - `seq_len` — [`L1110`](../../../../../raw/code/simply/simply/data_lib.py#L1110)
- uses (calls/refs, reference-scoped): [`pad_to_len`](utils/common.md#pad_to_len)
- used by: [`_to_fixed_length`](data_lib.md#_to_fixed_length), [`test_composed_truncate_then_pad`](data_lib_test.md#TruncateAndPadTransformTest.test_composed_truncate_then_pad), [`test_pad_does_not_truncate`](data_lib_test.md#TruncateAndPadTransformTest.test_pad_does_not_truncate), [`test_pad_short_sequences`](data_lib_test.md#TruncateAndPadTransformTest.test_pad_short_sequences)

### `QualityTrainSource`
- def: [`simply/data_lib.py:775`](../../../../../raw/code/simply/simply/data_lib.py#L775)
- doc: QuALITY long-context multiple-choice QA dataset (train split).
- signature: `class QualityTrainSource:`
- members:
  - `_examples(self)` — [`L795`](../../../../../raw/code/simply/simply/data_lib.py#L795) — Lazily loads and caches examples with normalized field names.
  - `end_index` — [`L788`](../../../../../raw/code/simply/simply/data_lib.py#L788)
  - `split` — [`L786`](../../../../../raw/code/simply/simply/data_lib.py#L786)
  - `start_index` — [`L787`](../../../../../raw/code/simply/simply/data_lib.py#L787)
- protocol/private: `__getitem__`[`L837`](../../../../../raw/code/simply/simply/data_lib.py#L837), `__len__`[`L834`](../../../../../raw/code/simply/simply/data_lib.py#L834), `_source`[`L791`](../../../../../raw/code/simply/simply/data_lib.py#L791)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry), [`TFDSSource`](data_lib.md#TFDSSource)
- used by: [`QualityValSource`](data_lib.md#QualityValSource)

### `QualityValSource`
- def: [`simply/data_lib.py:843`](../../../../../raw/code/simply/simply/data_lib.py#L843)
- doc: QuALITY long-context multiple-choice QA dataset (validation split).
- signature: `class QualityValSource(QualityTrainSource):`
- members:
  - `split` — [`L846`](../../../../../raw/code/simply/simply/data_lib.py#L846)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry), [`QualityTrainSource`](data_lib.md#QualityTrainSource)

### `RekeyTransform`  ·  implements/extends Map
- def: [`simply/data_lib.py:1128`](../../../../../raw/code/simply/simply/data_lib.py#L1128)
- doc: Rekey features.
- signature: `class RekeyTransform(grain.MapTransform):`
- members:
  - `map(self, features: Mapping[str, Any])` — [`L1136`](../../../../../raw/code/simply/simply/data_lib.py#L1136)
  - `key_map` — [`L1134`](../../../../../raw/code/simply/simply/data_lib.py#L1134)
- used by: [`test_none_key_map`](data_lib_test.md#RekeyTransformTest.test_none_key_map), [`test_rekeys_features`](data_lib_test.md#RekeyTransformTest.test_rekeys_features)

### `SimpleDataSource`  ·  implements/extends Protocol
- def: [`simply/data_lib.py:154`](../../../../../raw/code/simply/simply/data_lib.py#L154)
- signature: `class SimpleDataSource(Protocol):`
- protocol/private: `__getitem__`[`L159`](../../../../../raw/code/simply/simply/data_lib.py#L159), `__len__`[`L156`](../../../../../raw/code/simply/simply/data_lib.py#L156)
- used by: [`get_data_source`](data_lib.md#get_data_source), [`source`](data_lib.md#DatasetConfig.source)

### `SimpleQANumSource`
- def: [`simply/data_lib.py:586`](../../../../../raw/code/simply/simply/data_lib.py#L586) — documented in [simply-data_lib](../../concepts/simply-data_lib.md)
- doc: Simple QA dataset with only number-only answers.
- signature: `class SimpleQANumSource(SimpleQASource):`
- members:
  - `path` — [`L589`](../../../../../raw/code/simply/simply/data_lib.py#L589)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry), [`DATASETS_DIR`](data_lib.md#DATASETS_DIR), [`SimpleQASource`](data_lib.md#SimpleQASource)

### `SimpleQASource`
- def: [`simply/data_lib.py:561`](../../../../../raw/code/simply/simply/data_lib.py#L561)
- doc: Simple QA dataset source with lazy loading.
- signature: `class SimpleQASource:`
- members:
  - `path` — [`L564`](../../../../../raw/code/simply/simply/data_lib.py#L564)
  - `split` — [`L565`](../../../../../raw/code/simply/simply/data_lib.py#L565)
- protocol/private: `__getitem__`[`L580`](../../../../../raw/code/simply/simply/data_lib.py#L580), `__len__`[`L577`](../../../../../raw/code/simply/simply/data_lib.py#L577), `_examples`[`L568`](../../../../../raw/code/simply/simply/data_lib.py#L568)
- uses (calls/refs, reference-scoped): [`DATASETS_DIR`](data_lib.md#DATASETS_DIR)
- used by: [`SimpleQANumSource`](data_lib.md#SimpleQANumSource)

### `TFDSSource`
- def: [`simply/data_lib.py:170`](../../../../../raw/code/simply/simply/data_lib.py#L170)
- doc: TFDS data source with lazy loading.
- signature: `class TFDSSource:`
- members:
  - `name` — [`L185`](../../../../../raw/code/simply/simply/data_lib.py#L185)
  - `split` — [`L186`](../../../../../raw/code/simply/simply/data_lib.py#L186)
- protocol/private: `__getitem__`[`L196`](../../../../../raw/code/simply/simply/data_lib.py#L196), `__len__`[`L193`](../../../../../raw/code/simply/simply/data_lib.py#L193), `_source`[`L189`](../../../../../raw/code/simply/simply/data_lib.py#L189)
- uses (calls/refs, reference-scoped): [`register`](utils/registry.md#RootRegistry.register), [`DataSourceRegistry`](data_lib.md#DataSourceRegistry)
- used by: [`flops6e20_tfm2b_c4_l2048`](config_lib.md#flops6e20_tfm2b_c4_l2048), [`lm_test`](config_lib.md#lm_test), [`gemma2_2b_c4_vocab100864_l2048_bs1024`](config_lib.md#gemma2_2b_c4_vocab100864_l2048_bs1024), [`_source`](data_lib.md#QualityTrainSource._source)

### `TFExampleDeserializeTransform`  ·  implements/extends Map
- def: [`simply/data_lib.py:894`](../../../../../raw/code/simply/simply/data_lib.py#L894)
- doc: Conditionally deserializes TFExample proto bytes to a dict.
- signature: `class TFExampleDeserializeTransform(grain.MapTransform):`
- members:
  - `map(self, features: bytes | Mapping[str, Any])` — [`L913`](../../../../../raw/code/simply/simply/data_lib.py#L913)
- used by: [`_create_map_dataset`](data_lib.md#_create_map_dataset)

### `TokenizeTransform`  ·  implements/extends Map
- def: [`simply/data_lib.py:942`](../../../../../raw/code/simply/simply/data_lib.py#L942)
- doc: Tokenizes text using a tokenizer from TokenizerRegistry.
- signature: `class TokenizeTransform(grain.MapTransform):`
- members:
  - `map(self, features: Mapping[str, Any])` — [`L954`](../../../../../raw/code/simply/simply/data_lib.py#L954)
  - `add_bos` — [`L952`](../../../../../raw/code/simply/simply/data_lib.py#L952)
  - `add_eos` — [`L951`](../../../../../raw/code/simply/simply/data_lib.py#L951)
  - `data_key` — [`L950`](../../../../../raw/code/simply/simply/data_lib.py#L950)
  - `tokenizer_name` — [`L949`](../../../../../raw/code/simply/simply/data_lib.py#L949)
- uses (calls/refs, reference-scoped): [`_get_tokenizer`](data_lib.md#_get_tokenizer)
- used by: [`_create_map_dataset`](data_lib.md#_create_map_dataset), [`test_basic_tokenization_with_eos`](data_lib_test.md#TokenizeTransformTest.test_basic_tokenization_with_eos), [`test_tokenization_with_bos`](data_lib_test.md#TokenizeTransformTest.test_tokenization_with_bos), [`test_bytes_input_decoded`](data_lib_test.md#TokenizeTransformTest.test_bytes_input_decoded)

### `TruncateTransform`  ·  implements/extends Map
- def: [`simply/data_lib.py:1071`](../../../../../raw/code/simply/simply/data_lib.py#L1071)
- doc: Truncates sequences to a maximum length from the left.
- signature: `class TruncateTransform(grain.MapTransform):`
- members:
  - `map(self, features: Mapping[str, Any])` — [`L1087`](../../../../../raw/code/simply/simply/data_lib.py#L1087)
  - `seq_len` — [`L1085`](../../../../../raw/code/simply/simply/data_lib.py#L1085)
- used by: [`_to_fixed_length`](data_lib.md#_to_fixed_length), [`test_composed_truncate_then_pad`](data_lib_test.md#TruncateAndPadTransformTest.test_composed_truncate_then_pad), [`test_truncate_long_sequences`](data_lib_test.md#TruncateAndPadTransformTest.test_truncate_long_sequences)

## Functions
- `_create_map_dataset(ds_config: DatasetConfig, tokenizer_name: str, seed: int, shuffle: bool, num_epochs: int | None, seed_offset: int = 0)` — [`L1308`](../../../../../raw/code/simply/simply/data_lib.py#L1308) — Process a single DatasetConfig to MapDataset (before packing). — documented in [simply-data_lib](../../concepts/simply-data_lib.md)
- `_finalize(iter_ds: grain.IterDataset)` — [`L1462`](../../../../../raw/code/simply/simply/data_lib.py#L1462)
- `_get_lm_format(lm_format_name: str)` — [`L888`](../../../../../raw/code/simply/simply/data_lib.py#L888) — Get LMFormat instance from LMFormatRegistry (cached).
- `_get_tokenizer(tokenizer_name: str)` — [`L882`](../../../../../raw/code/simply/simply/data_lib.py#L882) — Get tokenizer instance from TokenizerRegistry (cached).
- `_is_leaf(x)` — [`L1548`](../../../../../raw/code/simply/simply/data_lib.py#L1548)
- `_pack(map_ds: grain.MapDataset, packing: str)` — [`L1448`](../../../../../raw/code/simply/simply/data_lib.py#L1448) — documented in [simply-data_lib](../../concepts/simply-data_lib.md)
- `_register_gsm8k_variants()` — [`L849`](../../../../../raw/code/simply/simply/data_lib.py#L849) — Register GSM8K variants with limited examples.
- `_shard(map_ds: grain.MapDataset, shard_data_method: Literal['NO_SHARDING', 'BY_JAX_PROCESS'])` — [`L1437`](../../../../../raw/code/simply/simply/data_lib.py#L1437)
- `_to_fixed_length(dataset: grain.IterDataset, seq_len: int, packing_method: str, pad_id: int = 0, seed: int = 0, num_packing_bins: int = 64, shuffle_bins: bool = True)` — [`L1200`](../../../../../raw/code/simply/simply/data_lib.py#L1200) — Applies packing to create fixed-length sequences from an IterDataset. — documented in [simply-data_lib](../../concepts/simply-data_lib.md)
- `create_iter_dataset(config, training: bool = True)` — [`L1391`](../../../../../raw/code/simply/simply/data_lib.py#L1391) — Main entry point for creating datasets. — documented in [simply-data_lib](../../concepts/simply-data_lib.md)
- `get_batch_fn(batch_mode: str)` — [`L1182`](../../../../../raw/code/simply/simply/data_lib.py#L1182) — Returns batch_fn for grain.batch() based on mode.
- `get_data_source(source: str | SimpleDataSource)` — [`L1273`](../../../../../raw/code/simply/simply/data_lib.py#L1273) — Creates a Grain MapDataset from config. — documented in [simply-data_lib](../../concepts/simply-data_lib.md)
- `grain_iter_global_state(local_state: Mapping[str, Any], shard_data_method: Literal['NO_SHARDING', 'BY_JAX_PROCESS'])` — [`L1539`](../../../../../raw/code/simply/simply/data_lib.py#L1539) — Returns the global state for a given grain local state.
- `grain_iter_local_state(global_state: Mapping[str, Any])` — [`L1564`](../../../../../raw/code/simply/simply/data_lib.py#L1564) — Returns the local state for a given grain global state.
- `pt_dataset_v1(source)` — [`L866`](../../../../../raw/code/simply/simply/data_lib.py#L866)
- `register_hf_vocabs()` — [`L126`](../../../../../raw/code/simply/simply/data_lib.py#L126)
- `register_spm_vocabs()` — [`L103`](../../../../../raw/code/simply/simply/data_lib.py#L103) — Registers SentencePiece vocabularies in the TokenizerRegistry.

## Module values
- `BATCH_STACKED` — [`L1178`](../../../../../raw/code/simply/simply/data_lib.py#L1178)
- `BATCH_UNSTACKED` — [`L1179`](../../../../../raw/code/simply/simply/data_lib.py#L1179)
- `DATASETS_DIR` — [`L62`](../../../../../raw/code/simply/simply/data_lib.py#L62)
- `DEEPSEEK_R1_DISTILL_QWEN_VOCAB` — [`L79`](../../../../../raw/code/simply/simply/data_lib.py#L79)
- `FWEDU_100864_V1_VOCAB` — [`L70`](../../../../../raw/code/simply/simply/data_lib.py#L70)
- `GEMMA2_VOCAB` — [`L75`](../../../../../raw/code/simply/simply/data_lib.py#L75)
- `GEMMA2_VOCABS` — [`L92`](../../../../../raw/code/simply/simply/data_lib.py#L92)
- `GEMMA3_VOCAB` — [`L76`](../../../../../raw/code/simply/simply/data_lib.py#L76)
- `GEMMA3_VOCABS` — [`L93`](../../../../../raw/code/simply/simply/data_lib.py#L93)
- `HF_VOCABS` — [`L95`](../../../../../raw/code/simply/simply/data_lib.py#L95)
- `OPENMIX_V1_100864_VOCAB` — [`L69`](../../../../../raw/code/simply/simply/data_lib.py#L69)
- `OPENMIX_V1_32768_VOCAB` — [`L68`](../../../../../raw/code/simply/simply/data_lib.py#L68)
- `OPENMIX_V1_VOCABS` — [`L85`](../../../../../raw/code/simply/simply/data_lib.py#L85)
- `OPENMIX_V2_EDU_100864_V1P1_VOCAB` — [`L72`](../../../../../raw/code/simply/simply/data_lib.py#L72)
- `OPENMIX_V2_EDU_100864_VOCAB` — [`L71`](../../../../../raw/code/simply/simply/data_lib.py#L71)
- `OPENMIX_V2_VOCABS` — [`L88`](../../../../../raw/code/simply/simply/data_lib.py#L88)
- `OPENMIX_V3_100864_V1_VOCAB` — [`L73`](../../../../../raw/code/simply/simply/data_lib.py#L73)
- `OPENMIX_V3_100864_V2_VOCAB` — [`L74`](../../../../../raw/code/simply/simply/data_lib.py#L74)
- `OPENMIX_V3_VOCABS` — [`L90`](../../../../../raw/code/simply/simply/data_lib.py#L90)
- `PACKING_CONCAT_SPLIT` — [`L1170`](../../../../../raw/code/simply/simply/data_lib.py#L1170)
- `PACKING_FIRST_FIT` — [`L1171`](../../../../../raw/code/simply/simply/data_lib.py#L1171)
- `PACKING_NONE` — [`L1173`](../../../../../raw/code/simply/simply/data_lib.py#L1173) — documented in [simply-data_lib](../../concepts/simply-data_lib.md)
- `PACKING_PAD_OR_TRUNCATE` — [`L1172`](../../../../../raw/code/simply/simply/data_lib.py#L1172)
- `QWEN2P5_VOCAB` — [`L77`](../../../../../raw/code/simply/simply/data_lib.py#L77)
- `QWEN3_VOCAB` — [`L82`](../../../../../raw/code/simply/simply/data_lib.py#L82)
- `QWQ_VOCAB` — [`L78`](../../../../../raw/code/simply/simply/data_lib.py#L78)
- `VOCABS_DIR` — [`L63`](../../../../../raw/code/simply/simply/data_lib.py#L63)

