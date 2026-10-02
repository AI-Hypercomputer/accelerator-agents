---
title: 'Module: easydel/trainers/ray_scaler/distributed_trainer.py'
type: catalog
provenance: extracted
module: easydel/trainers/ray_scaler/distributed_trainer.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.ray_scaler.distributed_trainer`/
symbols:
  RayDistributedTrainer.train: RayDistributedTrainer#train().
  RayDistributedTrainer.save_config: RayDistributedTrainer#save_config().
  RayDistributedTrainer.from_config: RayDistributedTrainer#from_config().
  RayDistributedTrainer.config_variables: RayDistributedTrainer#config_variables.
  RayDistributedTrainer.create_config: RayDistributedTrainer#create_config().
  RayDistributedTrainer.bucket_path: RayDistributedTrainer#bucket_path.
  RayDistributedConfig.config_variables: RayDistributedConfig#config_variables.
  RayDistributedTrainer.pretrained_model_name_or_path: RayDistributedTrainer#pretrained_model_name_or_path.
  RayDistributedTrainer.create_model: RayDistributedTrainer#create_model().
  RayDistributedTrainer.create_trainer: RayDistributedTrainer#create_trainer().
  RayDistributedTrainer.__init__: RayDistributedTrainer#__init__().
  RayDistributedTrainer.convert_model_to_state: RayDistributedTrainer#convert_model_to_state().
  RayDistributedTrainer.create_model_from_config: RayDistributedTrainer#create_model_from_config().
  RayDistributedTrainer.load_processor: RayDistributedTrainer#load_processor().
  logger: logger.
  RayDistributedConfig.config_scaling_variables: RayDistributedConfig#config_scaling_variables.
  RayDistributedTrainer.model_class: RayDistributedTrainer#model_class.
  RayDistributedTrainer.trainer_module: RayDistributedTrainer#trainer_module.
  RayDistributedConfig._saving_preprocess: RayDistributedConfig#_saving_preprocess().
  RayDistributedConfig._loading_postprocess: RayDistributedConfig#_loading_postprocess().
  RayDistributedTrainer.CONFIG_VARIABLES: RayDistributedTrainer#CONFIG_VARIABLES.
  RayDistributedTrainer: RayDistributedTrainer#
  RayDistributedTrainer.model_task: RayDistributedTrainer#model_task.
  RayDistributedTrainer.state_class: RayDistributedTrainer#state_class.
  RayDistributedTrainer.config_scaling_variables: RayDistributedTrainer#config_scaling_variables.
  RayDistributedConfig.model_task: RayDistributedConfig#model_task.
  RayDistributedTrainer.processor: RayDistributedTrainer#processor().
  RayDistributedTrainer.offload_backend: RayDistributedTrainer#offload_backend.
  RayDistributedTrainer.model_type: RayDistributedTrainer#model_type.
  RayDistributedTrainer.process_sample_data: RayDistributedTrainer#process_sample_data().
  RayDistributedTrainer.process_messages_data: RayDistributedTrainer#process_messages_data().
  RayDistributedTrainer._get_offload_device: RayDistributedTrainer#_get_offload_device().
  RayDistributedConfig: RayDistributedConfig#
  RayDistributedConfig.pretrained_model_name_or_path: RayDistributedConfig#pretrained_model_name_or_path.
  RayDistributedConfig.model_type: RayDistributedConfig#model_type.
  RayDistributedConfig.offload_backend: RayDistributedConfig#offload_backend.
  RayDistributedTrainer.CONFIG_SCALING_VARIABLES: RayDistributedTrainer#CONFIG_SCALING_VARIABLES.
  RayDistributedTrainer._processor_loader_class: RayDistributedTrainer#_processor_loader_class.
  RayDistributedTrainer.__repr__: RayDistributedTrainer#__repr__().
  RayDistributedTrainer.extract_column_names: RayDistributedTrainer#extract_column_names().
  RayDistributedTrainer.__str__: RayDistributedTrainer#__str__.
---
# Module: [`easydel/trainers/ray_scaler/distributed_trainer.py`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py)

## Classes
### `RayDistributedConfig`  ·  implements/extends BaseModel
- def: [`easydel/trainers/ray_scaler/distributed_trainer.py:76`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L76)
- doc: Configuration for RayDistributedTrainer that can be persisted to JSON.
- signature: `class RayDistributedConfig(BaseModel):`
- members:
  - `_loading_postprocess(self)` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L120) — Convert string representations back to dtypes and PartitionAxis after loading.
  - `_saving_preprocess(self)` — [`L104`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L104) — Convert dtypes and PartitionAxis to JSON-serializable formats before saving.
  - `config_scaling_variables` — [`L101`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L101)
  - `config_variables` — [`L102`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L102)
  - `model_task` — [`L98`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L98)
  - `model_type` — [`L99`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L99)
  - `offload_backend` — [`L100`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L100)
  - `pretrained_model_name_or_path` — [`L97`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L97)
- uses (calls/refs, reference-scoped): [`TaskType`](../../infra/factory.md#TaskType)
- used by: [`save_config`](distributed_trainer.md#RayDistributedTrainer.save_config), [`from_config`](distributed_trainer.md#RayDistributedTrainer.from_config)

### `RayDistributedTrainer`
- def: [`easydel/trainers/ray_scaler/distributed_trainer.py:137`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L137)
- doc: Distributed trainer for Ray-based training with EasyDeL models.
- signature: `class RayDistributedTrainer:`
- members:
  - `__init__(self, pretrained_model_name_or_path: str, bucket_path: str | None = None, model_task: TaskType | None = None, model_type: str | None = None, model_class: type[EasyDeLBaseModule] | None = None, state_class: type[EasyDeLState] | None = None, offload_backend: str | None = None, trainer_module: type[BaseTrainer | Trainer] | None = None, config_scaling_variables: dict[str, int] | None = None, config_variables: dict[str, tp.Any] | None = None)` — [`L205`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L205) — Initialize the RayDistributedTrainer.
  - `_get_offload_device(self)` — [`L449`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L449) — Get the device for memory offloading.
  - `convert_model_to_state(self, model: EasyDeLBaseModule)` — [`L506`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L506) — Convert a model module to a state object.
  - `create_config(self, scaling_index: int)` — [`L426`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L426) — Create a model configuration with scaled dimensions.
  - `create_model(self, config: EasyDeLBaseConfig, dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.bfloat16, precision: lax.PrecisionLike | None = None, seed: int = 684, lazy: bool = False)` — [`L468`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L468) — Create a model instance from configuration.
  - `create_model_from_config(self, scaling_index: int)` — [`L522`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L522) — Create a model with configuration scaled by the given index.
  - `create_trainer(self, arguments: TrainingArguments, dataset_train: Dataset, dataset_eval: Dataset | None = None, data_collator: tp.Callable | None = None, state: EasyDeLState | None = None)` — [`L540`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L540) — Create a trainer instance for model training.
  - `extract_column_names(dataset: Dataset)` — [`L354`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L354) — Extract column names from a dataset.
  - `from_config(cls, path: str | os.PathLike, model_class: type[EasyDeLBaseModule] | None = None, state_class: type[EasyDeLState] | None = None, trainer_module: type[BaseTrainer | Trainer] | None = None)` — [`L277`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L277) — Create a RayDistributedTrainer from a saved configuration file.
  - `load_processor(self)` — [`L328`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L328) — Load the tokenizer/processor for the model.
  - `process_messages_data(self, messages: tp.Any, max_length: int, padding_side: str = "left")` — [`L398`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L398) — Process chat messages using the tokenizer's chat template.
  - `process_sample_data(self, sample: tp.Any, max_length: int, padding_side: str = "left")` — [`L370`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L370) — Process a text sample into model inputs.
  - `processor(self)` — [`L349`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L349) — Cached property for the tokenizer/processor.
  - `save_config(self, path: str | os.PathLike)` — [`L310`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L310) — Save the current configuration to a JSON file.
  - `train(self, scaling_index: int, arguments: TrainingArguments, dataset_train: Dataset, dataset_eval: Dataset | None = None, data_collator: tp.Callable | None = None, model: EasyDeLBaseModule | None = None, state: EasyDeLState | None = None)` — [`L569`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L569) — Execute distributed training with the configured model.
  - `CONFIG_SCALING_VARIABLES` — [`L178`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L178)
  - `CONFIG_VARIABLES` — [`L186`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L186)
  - `_processor_loader_class` — [`L203`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L203) — ---
  - `bucket_path` — [`L269`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L269)
  - `config_scaling_variables` — [`L262`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L262)
  - `config_variables` — [`L263`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L263)
  - `model_class` — [`L171`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L171) — ---
  - `model_task` — [`L169`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L169)
  - `model_type` — [`L170`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L170)
  - `offload_backend` — [`L174`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L174)
  - `pretrained_model_name_or_path` — [`L236`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L236)
  - `state_class` — [`L172`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L172) — ---
  - `trainer_module` — [`L176`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L176)
- protocol/private: `__repr__`[`L652`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L652), `__str__`[`L666`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L666)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../../infra/base_module.md#EasyDeLBaseModule), [`TaskType`](../../infra/factory.md#TaskType), [`EasyDeLBaseConfig`](../../infra/base_config.md#EasyDeLBaseConfig), [`EasyDeLState`](../../infra/base_state.md#EasyDeLState), [`EasyDeLGradientCheckPointers`](../../infra/etils.md#EasyDeLGradientCheckPointers), [`TrainingArguments`](../training_configurations.md#TrainingArguments), [`Trainer`](../trainer/trainer.md#Trainer), [`NONE`](../../infra/etils.md#EasyDeLGradientCheckPointers.NONE), [`to_state`](../../infra/base_module.md#EasyDeLBaseModule.to_state), [`load_state`](../../infra/base_state.md#EasyDeLState.load_state), [`BaseTrainer`](../base_trainer.md#BaseTrainer), [`get_modules_by_type`](../../modules/auto/auto_configuration.md#get_modules_by_type), [`EasyDeLBaseConfigDict`](../../infra/base_config.md#EasyDeLBaseConfigDict), [`sequential_init`](../../infra/base_module.md#EasyDeLBaseModule.sequential_init), [`config_variables`](distributed_trainer.md#RayDistributedConfig.config_variables), [`lazy_init`](../../infra/base_module.md#EasyDeLBaseModule.lazy_init), [`config_scaling_variables`](distributed_trainer.md#RayDistributedConfig.config_scaling_variables), [`logger`](distributed_trainer.md#logger), [`_loading_postprocess`](distributed_trainer.md#RayDistributedConfig._loading_postprocess), [`_saving_preprocess`](distributed_trainer.md#RayDistributedConfig._saving_preprocess), [`gradient_checkpointing`](../../infra/base_config.md#EasyDeLBaseConfigDict.gradient_checkpointing), [`_model_task`](../../infra/base_module.md#EasyDeLBaseModule._model_task), [`attn_mechanism`](../../infra/base_config.md#EasyDeLBaseConfigDict.attn_mechanism), [`model_task`](distributed_trainer.md#RayDistributedConfig.model_task), [`config_class`](../../infra/base_module.md#EasyDeLBaseModule.config_class), [`train`](../trainer_protocol.md#BaseTrainerProtocol.train), [`_model_type`](../../infra/base_module.md#EasyDeLBaseModule._model_type), [`RayDistributedConfig`](distributed_trainer.md#RayDistributedConfig), [`attn_dtype`](../../infra/base_config.md#EasyDeLBaseConfigDict.attn_dtype), [`freq_max_position_embeddings`](../../infra/base_config.md#EasyDeLBaseConfigDict.freq_max_position_embeddings), [`mask_max_position_embeddings`](../../infra/base_config.md#EasyDeLBaseConfigDict.mask_max_position_embeddings), [`model_type`](distributed_trainer.md#RayDistributedConfig.model_type), [`offload_backend`](distributed_trainer.md#RayDistributedConfig.offload_backend), [`pretrained_model_name_or_path`](distributed_trainer.md#RayDistributedConfig.pretrained_model_name_or_path), [`attn_softmax_dtype`](../../infra/base_config.md#EasyDeLBaseConfigDict.attn_softmax_dtype)
- used by: [`_import_structure`](../../__init__.md#_import_structure)

## Module values
- `logger` — [`L72`](../../../../../../../raw/code/EasyDeL/easydel/trainers/ray_scaler/distributed_trainer.py#L72)

