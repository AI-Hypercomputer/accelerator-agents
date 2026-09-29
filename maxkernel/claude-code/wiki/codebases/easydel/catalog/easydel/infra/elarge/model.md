---
title: 'Module: easydel/infra/elarge/model.py'
type: catalog
provenance: extracted
module: easydel/infra/elarge/model.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.elarge.model`/
symbols:
  eLargeModel._config: eLargeModel#_config.
  eLargeModel.build_trainer: eLargeModel#build_trainer().
  eLargeModel.run_benchmarks: eLargeModel#run_benchmarks().
  eLargeModel.train: eLargeModel#train().
  eLargeModel.eval: eLargeModel#eval().
  eLargeModel._tokenizer: eLargeModel#_tokenizer.
  eLargeModel: eLargeModel#
  eLargeModel.__str__: eLargeModel#__str__().
  eLargeModel.build_state: eLargeModel#build_state().
  eLargeModel.from_pretrained: eLargeModel#from_pretrained().
  eLargeModel.build_model: eLargeModel#build_model().
  eLargeModel.build_esurge: eLargeModel#build_esurge().
  eLargeModel._line: eLargeModel#_line().
  eLargeModel.set_esurge: eLargeModel#set_esurge().
  eLargeModel.build_teacher_model: eLargeModel#build_teacher_model().
  eLargeModel.task: eLargeModel#task().
  eLargeModel.build_reference_model: eLargeModel#build_reference_model().
  eLargeModel.build_reward_model: eLargeModel#build_reward_model().
  logger: logger.
  eLargeModel._model: eLargeModel#_model.
  eLargeModel.get_train_source: eLargeModel#get_train_source().
  eLargeModel.build_tokenizer: eLargeModel#build_tokenizer().
  eLargeModel.update_config: eLargeModel#update_config().
  eLargeModel.set_mixture: eLargeModel#set_mixture().
  eLargeModel.to_yaml: eLargeModel#to_yaml().
  eLargeModel.model_name: eLargeModel#model_name().
  eLargeModel.build_sharded_source: eLargeModel#build_sharded_source().
  eLargeModel.build_training_arguments: eLargeModel#build_training_arguments().
  _ESURGE_UNSET: _ESURGE_UNSET.
  eLargeModel.set_quantization: eLargeModel#set_quantization().
  eLargeModel.set_eval: eLargeModel#set_eval().
  eLargeModel.clear_cache: eLargeModel#clear_cache().
  eLargeModel.set_watchers: eLargeModel#set_watchers().
  eLargeModel.__repr__: eLargeModel#__repr__().
  eLargeModel._state: eLargeModel#_state.
  eLargeModel.get_trainer_config: eLargeModel#get_trainer_config().
  eLargeModel.validate: eLargeModel#validate().
  eLargeModel.get_load_state_kwargs: eLargeModel#get_load_state_kwargs().
  eLargeModel.build_dataset: eLargeModel#build_dataset().
  eLargeModel.from_json: eLargeModel#from_json().
  eLargeModel.from_yaml: eLargeModel#from_yaml().
  eLargeModel.config: eLargeModel#config().
  eLargeModel.set_model: eLargeModel#set_model().
  eLargeModel.set_teacher_model: eLargeModel#set_teacher_model().
  eLargeModel.set_reference_model: eLargeModel#set_reference_model().
  eLargeModel.set_dtype: eLargeModel#set_dtype().
  eLargeModel.set_sharding: eLargeModel#set_sharding().
  eLargeModel.set_operation_configs: eLargeModel#set_operation_configs().
  eLargeModel.add_dataset: eLargeModel#add_dataset().
  eLargeModel.to_json: eLargeModel#to_json().
  eLargeModel.get_from_pretrained_kwargs: eLargeModel#get_from_pretrained_kwargs().
  eLargeModel.get_esurge_kwargs: eLargeModel#get_esurge_kwargs().
  eLargeModel.get_base_config: eLargeModel#get_base_config().
  eLargeModel.get_data_mixture_kwargs: eLargeModel#get_data_mixture_kwargs().
  eLargeModel.set_trainer: eLargeModel#set_trainer().
  _QUANT_UNSET: _QUANT_UNSET.
  eLargeModel._fmt: eLargeModel#_fmt().
  eLargeModel._sep: eLargeModel#_sep().
  eLargeModel._load_yaml_config: eLargeModel#_load_yaml_config().
  BuildTrainerKws.reference_model: BuildTrainerKws#reference_model.
  BuildTrainerKws.reward_model: BuildTrainerKws#reward_model.
  BuildTrainerKws.teacher_model: BuildTrainerKws#teacher_model.
  eLargeModel.__init__: eLargeModel#__init__().
  eLargeModel.teacher_model_name: eLargeModel#teacher_model_name().
  eLargeModel.reference_model_name: eLargeModel#reference_model_name().
  eLargeModel.to_dict: eLargeModel#to_dict().
  eLargeModel._uses_removed_benchmark_task_alias: eLargeModel#_uses_removed_benchmark_task_alias().
  BuildTrainerKws: BuildTrainerKws#
  BuildTrainerKws.data_collator: BuildTrainerKws#data_collator.
  BuildTrainerKws.formatting_func: BuildTrainerKws#formatting_func.
  BuildTrainerKws.reward_processing_classes: BuildTrainerKws#reward_processing_classes.
  BuildTrainerKws.data_tokenize_fn: BuildTrainerKws#data_tokenize_fn.
  BuildTrainerKws.reward_funcs: BuildTrainerKws#reward_funcs.
  BuildTrainerKws.external_reward_funcs: BuildTrainerKws#external_reward_funcs.
  BuildTrainerKws.external_reward_processing_classes: BuildTrainerKws#external_reward_processing_classes.
  BuildTrainerKws.external_reward_weights: BuildTrainerKws#external_reward_weights.
  BuildTrainerKws.feedback_func: BuildTrainerKws#feedback_func.
  BuildTrainerKws.env_factory: BuildTrainerKws#env_factory.
  BuildTrainerKws.tools: BuildTrainerKws#tools.
---
# Module: [`easydel/infra/elarge/model.py`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py)

## Classes
### `BuildTrainerKws`  ·  implements/extends _TypedDict
- def: [`easydel/infra/elarge/model.py:100`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L100)
- doc: Type hints for optional keyword arguments when building trainers.
- signature: `class BuildTrainerKws(typing.TypedDict, total=False):`
- members:
  - `data_collator` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L120)
  - `data_tokenize_fn` — [`L123`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L123)
  - `env_factory` — [`L132`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L132)
  - `external_reward_funcs` — [`L128`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L128)
  - `external_reward_processing_classes` — [`L129`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L129)
  - `external_reward_weights` — [`L130`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L130)
  - `feedback_func` — [`L131`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L131)
  - `formatting_func` — [`L121`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L121)
  - `reference_model` — [`L124`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L124)
  - `reward_funcs` — [`L127`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L127)
  - `reward_model` — [`L125`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L125)
  - `reward_processing_classes` — [`L122`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L122)
  - `teacher_model` — [`L126`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L126)
  - `tools` — [`L133`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L133)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../base_module.md#EasyDeLBaseModule)
- used by: [`train`](model.md#eLargeModel.train)

### `eLargeModel`
- def: [`easydel/infra/elarge/model.py:136`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L136)
- doc: Master class for Easy Large Models (ELM) in EasyDeL.
- signature: `class eLargeModel:`
- members:
  - `__init__(self, config: eLMConfig | Mapping[str, Any] | str | os.PathLike | ePathLike | None = None)` — [`L181`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L181) — Initialize eLargeModel with configuration.
  - `__repr__(self)` — [`L2231`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L2231) — Developer-friendly string representation of eLargeModel.
  - `__str__(self)` — [`L2250`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L2250) — Human-readable string representation with formatted configuration.
  - `_fmt(val: Any)` — [`L2260`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L2260) — Format a value for display, handling enums and special types.
  - `_line(content: str)` — [`L2277`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L2277) — Create a padded line within the box.
  - `_load_yaml_config(yaml_path: str | os.PathLike | ePathLike)` — [`L214`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L214) — Load raw ELM config mapping from a YAML file.
  - `_sep()` — [`L2288`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L2288) — Create a separator line within the box.
  - `_uses_removed_benchmark_task_alias(task: Any)` — [`L2083`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L2083) — Check whether a task dict uses the removed ``task`` alias with extra keys.
  - `add_dataset(self, data_files: str | list[str], dataset_type: str | None = None, content_field: str = "content", split: str = "train", **kwargs)` — [`L789`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L789) — Add a dataset to the mixture configuration.
  - `build_dataset(self)` — [`L1312`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1312) — Build dataset from mixture configuration.
  - `build_esurge(self)` — [`L1223`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1223) — Build the eSurge inference engine.
  - `build_model(self, force_rebuild: bool = False)` — [`L1091`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1091) — Build the EasyDeL model from configuration.
  - `build_reference_model(self)` — [`L1263`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1263) — Build the reference model for preference optimization (DPO, etc.).
  - `build_reward_model(self)` — [`L1286`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1286) — Build the reward model for RLHF trainers (GRPO/GFPO/GSPO/XPO/Nash-MD).
  - `build_sharded_source(self)` — [`L1332`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1332) — Build dataset as ShardedDataSource for use with new data pipeline.
  - `build_state(self, force_rebuild: bool = False, state_class: type[EasyDeLState] | None = None, **load_state_overrides: Any)` — [`L1120`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1120) — Build an :class:`EasyDeLState` from a saved EasyDeL checkpoint.
  - `build_teacher_model(self)` — [`L1241`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1241) — Build the teacher model for distillation training.
  - `build_tokenizer(self, force_rebuild: bool = False)` — [`L1195`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1195) — Build or get the tokenizer for the model.
  - `build_trainer(self, train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None, eval_dataset: Dataset | IterableDataset | ShardedDataSource | None = None, reference_model: EasyDeLBaseModule | None = None, reward_model: EasyDeLBaseModule | None = None, teacher_model: EasyDeLBaseModule | None = None, reward_funcs: Any | None = None, base_state_class: type[EasyDeLState] | None = None, args_class: type[TrainingArguments] | None = None, trainer_class: type[Trainer] | None = None, **kwargs)` — [`L1624`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1624) — Build a trainer instance with the configured settings.
  - `build_training_arguments(self, args_class: TrainingArguments | None = None, **overrides)` — [`L1593`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1593) — Build TrainingArguments for the configured trainer.
  - `clear_cache(self)` — [`L1389`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1389) — Clear cached model, tokenizer, and inference engine instances.
  - `config(self)` — [`L341`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L341) — Get the normalized configuration dictionary.
  - `eval(self, tasks: str | list[str | dict[str, Any] | Any] | BenchmarkConfig | list[BenchmarkConfig], num_fewshot: int | None = None, output_path: str | None = None, **eval_overrides: Unpack[EvalKwargs])` — [`L2001`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L2001) — Run evaluation on specified tasks using lm-evaluation-harness.
  - `from_json(cls, json_path: str | os.PathLike | ePathLike)` — [`L249`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L249) — Create eLargeModel from JSON configuration file.
  - `from_pretrained(cls, model_name_or_path: str, task: TaskType | str | None = None, **kwargs)` — [`L300`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L300) — Create eLargeModel from pretrained model name or path.
  - `from_yaml(cls, yaml_path: str | os.PathLike | ePathLike)` — [`L261`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L261) — Create eLargeModel from a YAML configuration file.
  - `get_base_config(self, prefer: str = "base")` — [`L1070`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1070) — Get materialized base configuration.
  - `get_data_mixture_kwargs(self)` — [`L1377`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1377) — Get kwargs for DatasetMixture initialization.
  - `get_esurge_kwargs(self)` — [`L1052`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1052) — Get kwargs for eSurge initialization.
  - `get_from_pretrained_kwargs(self)` — [`L1004`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1004) — Get kwargs for model.from_pretrained() calls.
  - `get_load_state_kwargs(self)` — [`L1025`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1025) — Get kwargs for :meth:`EasyDeLState.load_state`.
  - `get_train_source(self)` — [`L1353`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1353) — Get training data as ShardedDataSource or Dataset.
  - `get_trainer_config(self)` — [`L1493`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1493) — Get normalized trainer configuration.
  - `model_name(self)` — [`L351`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L351) — Get the model name or path.
  - `reference_model_name(self)` — [`L379`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L379) — Get the reference model name or path for DPO/ORPO.
  - `run_benchmarks(self, benchmarks: BenchmarkConfig | list[BenchmarkConfig], output_path: str | None = None, engine_instance: eSurge | None = None, **default_eval_overrides: Unpack[EvalKwargs])` — [`L2154`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L2154) — Run one or more named benchmark suites sequentially, sharing a single eSurge engine.
  - `set_dtype(self, dtype: str)` — [`L480`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L480) — Set the data type for model loading.
  - `set_esurge(self, max_model_len: int | None = None, max_num_seqs: int = 16, hbm_utilization: float = 0.85, bind_graphstate_for_aot: bool | object = _ESURGE_UNSET, enable_window_aware_runtime_cap: bool | object = _ESURGE_UNSET, tool_parser: ToolParserName | None | object = _ESURGE_UNSET, reasoning_parser: ReasoningParserName | None | object = _ESURGE_UNSET, **kwargs)` — [`L647`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L647) — Configure eSurge inference settings.
  - `set_eval(self, max_new_tokens: int = 8192, hard_max_new_tokens: bool = False, enable_thinking: bool = False, ignore_benchmark_eos_flags: bool = False, temperature: float = 0, top_p: float = 0.95, batch_size: int | str | None = None, use_tqdm: bool = True, **kwargs: Unpack[EvalKwargs])` — [`L856`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L856) — Configure evaluation settings for lm-evaluation-harness.
  - `set_mixture(self, informs: list[TextDatasetInformCfg | VisualDatasetInformCfg] | None = None, batch_size: int = 32, streaming: bool = True, use_fast_loader: bool = True, **kwargs)` — [`L735`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L735) — Configure data mixture settings for training/evaluation.
  - `set_model(self, model_name_or_path: str)` — [`L412`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L412) — Set the model name or path.
  - `set_operation_configs(self, configs: Mapping[str, Any] | None = None, **kwargs)` — [`L595`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L595) — Configure ejkernel operation overrides.
  - `set_quantization(self, method: str | None = None, group_size: int = 128, use_qmm_best_config: bool = False, qmm_platform_override: str | None | object = _QUANT_UNSET, qmm_tpu_path_override: str | None | object = _QUANT_UNSET, **kwargs)` — [`L547`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L547) — Configure quantization settings.
  - `set_reference_model(self, model_name_or_path: str)` — [`L456`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L456) — Set the reference model name or path for preference optimization.
  - `set_sharding(self, axis_dims: tuple[int, ...] | None = None, axis_names: tuple[str, ...] | None = None, **kwargs)` — [`L504`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L504) — Configure model sharding for distributed training/inference.
  - `set_teacher_model(self, model_name_or_path: str)` — [`L432`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L432) — Set the teacher model name or path for distillation training.
  - `set_trainer(self, trainer_type: str, **kwargs)` — [`L1399`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1399) — Configure trainer settings.
  - `set_watchers(self, *watchers: LogWatcher)` — [`L1458`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1458) — Register LogWatcher instances for per-parameter metric logging.
  - `task(self)` — [`L360`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L360) — Get the resolved task type.
  - `teacher_model_name(self)` — [`L370`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L370) — Get the teacher model name or path for distillation.
  - `to_dict(self)` — [`L987`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L987) — Get configuration as dictionary.
  - `to_json(self, json_path: str | os.PathLike | ePathLike)` — [`L942`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L942) — Save configuration to JSON file.
  - `to_yaml(self, yaml_path: str | os.PathLike | ePathLike)` — [`L959`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L959) — Save configuration to a YAML file.
  - `train(self, train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None, eval_dataset: Dataset | IterableDataset | ShardedDataSource | None = None, base_state_class: type[EasyDeLState] | None = None, args_class: type[TrainingArguments] | None = None, trainer_class: type[Trainer] | None = None, **build_kwargs: Unpack[BuildTrainerKws])` — [`L1506`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L1506) — Train the model with the configured settings.
  - `update_config(self, updates: Mapping[str, Any])` — [`L387`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L387) — Update configuration with new values.
  - `validate(self)` — [`L930`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L930) — Validate the current configuration.
- protocol/private: `_config`[`L191`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L191), `_model`[`L209`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L209), `_state`[`L210`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L210), `_tokenizer`[`L211`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L211)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../base_module.md#EasyDeLBaseModule), [`TaskType`](../factory.md#TaskType), [`EasyDeLState`](../base_state.md#EasyDeLState), [`ShardedDataSource`](../../data/core/protocols.md#ShardedDataSource), [`model`](../base_state.md#EasyDeLState.model), [`TrainingArguments`](../../trainers/training_configurations.md#TrainingArguments), [`Trainer`](../../trainers/trainer/trainer.md#Trainer), [`build_model`](builders.md#build_model), [`eSurge`](../../inference/esurge/esurge_engine.md#eSurge), [`to_state`](../base_module.md#EasyDeLBaseModule.to_state), [`load_state`](../base_state.md#EasyDeLState.load_state), [`model`](../../trainers/base_trainer.md#BaseTrainer.model), [`infer_task_from_hf_config`](../../modules/auto/auto_configuration.md#infer_task_from_hf_config), [`run_lm_eval_with_esurge`](benchmarking.md#run_lm_eval_with_esurge), [`normalize`](processing.md#normalize), [`build_esurge`](builders.md#build_esurge), [`build_sharded_source`](builders.md#build_sharded_source), [`eLMConfig`](types/root.md#eLMConfig), [`normalize_benchmark_configs`](benchmarking.md#normalize_benchmark_configs), [`train`](../../trainers/trainer/trainer.md#Trainer.train), [`BenchmarkConfig`](types/eval.md#BenchmarkConfig), [`resolve_task`](processing.md#resolve_task), [`materialize_base_config`](processing.md#materialize_base_config), [`terminate`](../../inference/esurge/mixins/lifecycle.md#EngineLifecycleMixin.terminate), [`to_load_state_kwargs`](builders.md#to_load_state_kwargs), [`to_from_pretrained_kwargs`](builders.md#to_from_pretrained_kwargs), [`make_serializable`](processing.md#make_serializable), [`build_dataset`](builders.md#build_dataset), [`normalize_trainer_config`](types/training.md#normalize_trainer_config), [`ReasoningParserName`](../../inference/reasoning/abstract_reasoning.md#ReasoningParserName.ReasoningParserName), [`ToolParserName`](../../inference/tools/abstract_tool.md#ToolParserName.ToolParserName), [`to_data_mixture_kwargs`](builders.md#to_data_mixture_kwargs), [`save_elm_config`](processing.md#save_elm_config), [`logger`](model.md#logger), [`EvalKwargs`](types/eval.md#EvalKwargs), [`get_trainer_class`](types/training.md#get_trainer_class), [`get_training_arguments_class`](types/training.md#get_training_arguments_class), [`to_esurge_kwargs`](builders.md#to_esurge_kwargs), [`AUTO_BIND`](../factory.md#TaskType.AUTO_BIND), [`load_elm_config`](processing.md#load_elm_config)  (+14 more)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`__getattr__`](../__init__.md#__getattr__)

## Module values
- `_ESURGE_UNSET` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L96)
- `_QUANT_UNSET` — [`L97`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L97)
- `logger` — [`L95`](../../../../../../../raw/code/EasyDeL/easydel/infra/elarge/model.py#L95)

