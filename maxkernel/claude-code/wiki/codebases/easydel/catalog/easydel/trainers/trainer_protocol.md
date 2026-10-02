---
title: 'Module: easydel/trainers/trainer_protocol.py'
type: catalog
provenance: extracted
module: easydel/trainers/trainer_protocol.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.trainer_protocol`/
symbols:
  BaseTrainerProtocol.state_shardings: BaseTrainerProtocol#state_shardings.
  TrainerConfigureFunctionOutput: TrainerConfigureFunctionOutput#
  BaseTrainerProtocol.scheduler: BaseTrainerProtocol#scheduler.
  BaseTrainerProtocol.max_training_steps: BaseTrainerProtocol#max_training_steps.
  BaseTrainerProtocol.timer: BaseTrainerProtocol#timer.
  BaseTrainerProtocol._train_epoch: BaseTrainerProtocol#_train_epoch().
  BaseTrainerProtocol._eval_epoch: BaseTrainerProtocol#_eval_epoch().
  BaseTrainerProtocol._run_training_loop: BaseTrainerProtocol#_run_training_loop().
  BaseTrainerProtocol._run_evaluation: BaseTrainerProtocol#_run_evaluation().
  BaseTrainerProtocol.config: BaseTrainerProtocol#config.
  BaseTrainerProtocol.dataset_eval: BaseTrainerProtocol#dataset_eval.
  BaseTrainerProtocol.sharded_training_step_function: BaseTrainerProtocol#sharded_training_step_function.
  BaseTrainerProtocol.sharded_evaluation_step_function: BaseTrainerProtocol#sharded_evaluation_step_function.
  BaseTrainerProtocol._extra_forward_flops_per_token: BaseTrainerProtocol#_extra_forward_flops_per_token.
  BaseTrainerProtocol.__init__: BaseTrainerProtocol#__init__().
  BaseTrainerProtocol._execute_train_step: BaseTrainerProtocol#_execute_train_step().
  BaseTrainerProtocol.on_step_end: BaseTrainerProtocol#on_step_end().
  TrainerConfigureDataloaderOutput: TrainerConfigureDataloaderOutput#
  BaseTrainerProtocol: BaseTrainerProtocol#
  BaseTrainerProtocol.dataset_train: BaseTrainerProtocol#dataset_train.
  BaseTrainerProtocol.dataloader_train: BaseTrainerProtocol#dataloader_train.
  BaseTrainerProtocol.pruning_module: BaseTrainerProtocol#pruning_module.
  BaseTrainerProtocol._extra_backward_flops_per_token: BaseTrainerProtocol#_extra_backward_flops_per_token.
  BaseTrainerProtocol._model: BaseTrainerProtocol#_model.
  BaseTrainerProtocol.data_collator: BaseTrainerProtocol#data_collator.
  BaseTrainerProtocol.dataloader_eval: BaseTrainerProtocol#dataloader_eval.
  BaseTrainerProtocol.tx: BaseTrainerProtocol#tx.
  BaseTrainerProtocol._resumed_from_checkpoint: BaseTrainerProtocol#_resumed_from_checkpoint.
  BaseTrainerProtocol.configure_functions: BaseTrainerProtocol#configure_functions().
  BaseTrainerProtocol.configure_dataloaders: BaseTrainerProtocol#configure_dataloaders().
  BaseTrainerProtocol.configure_model: BaseTrainerProtocol#configure_model().
  BaseTrainerProtocol._save_state: BaseTrainerProtocol#_save_state().
  BaseTrainerProtocol.save_pretrained: BaseTrainerProtocol#save_pretrained().
  BaseTrainerProtocol._save_to_torch: BaseTrainerProtocol#_save_to_torch().
  BaseTrainerProtocol._create_hf_model_config: BaseTrainerProtocol#_create_hf_model_config().
  BaseTrainerProtocol.apply_training_hooks: BaseTrainerProtocol#apply_training_hooks().
  BaseTrainerProtocol._prepare_training_output: BaseTrainerProtocol#_prepare_training_output().
  BaseTrainerProtocol._handle_training_interruption: BaseTrainerProtocol#_handle_training_interruption().
  BaseTrainerProtocol.create_progress_bar: BaseTrainerProtocol#create_progress_bar().
  BaseTrainerProtocol.log_weight_distribution: BaseTrainerProtocol#log_weight_distribution().
  BaseTrainerProtocol.log_watchers: BaseTrainerProtocol#log_watchers().
  BaseTrainerProtocol.log_metrics: BaseTrainerProtocol#log_metrics().
  BaseTrainerProtocol._configure_grain_dataloader: BaseTrainerProtocol#_configure_grain_dataloader().
  BaseTrainerProtocol._configure_tfds_dataloader: BaseTrainerProtocol#_configure_tfds_dataloader().
  BaseTrainerProtocol._execute_eval_step: BaseTrainerProtocol#_execute_eval_step().
  BaseTrainerProtocol.eval: BaseTrainerProtocol#eval().
  BaseTrainerProtocol.on_step_start: BaseTrainerProtocol#on_step_start().
  BaseTrainerProtocol._preprocess_batch_input: BaseTrainerProtocol#_preprocess_batch_input().
  BaseTrainerProtocol.train_tracker: BaseTrainerProtocol#train_tracker.
  BaseTrainerProtocol.evalu_tracker: BaseTrainerProtocol#evalu_tracker.
  TrainerConfigureModelOutput: TrainerConfigureModelOutput#
  TrainerOutput: TrainerOutput#
  BaseTrainerProtocol.max_evaluation_steps: BaseTrainerProtocol#max_evaluation_steps.
  TrainerConfigureModelOutput.model: TrainerConfigureModelOutput#model.
  TrainerConfigureModelOutput.config: TrainerConfigureModelOutput#config.
  BaseTrainerProtocol.train: BaseTrainerProtocol#train().
  BaseTrainerProtocol._train_shared_fn_static_args_: BaseTrainerProtocol#_train_shared_fn_static_args_.
  BaseTrainerProtocol._train_shared_fn_extra_args_: BaseTrainerProtocol#_train_shared_fn_extra_args_.
  BaseTrainerProtocol._eval_shared_fn_static_args_: BaseTrainerProtocol#_eval_shared_fn_static_args_.
  BaseTrainerProtocol._eval_shared_fn_extra_args_: BaseTrainerProtocol#_eval_shared_fn_extra_args_.
  BaseTrainerProtocol._training_time_start: BaseTrainerProtocol#_training_time_start.
  TrainerOutput.state: TrainerOutput#state.
  BaseTrainerProtocol.arguments: BaseTrainerProtocol#arguments.
  BaseTrainerProtocol.model_state: BaseTrainerProtocol#model_state.
  BaseTrainerProtocol.model: BaseTrainerProtocol#model().
  BaseTrainerProtocol.training_batch_size: BaseTrainerProtocol#training_batch_size().
  BaseTrainerProtocol.evaluation_batch_size: BaseTrainerProtocol#evaluation_batch_size().
  BaseTrainerProtocol.is_process_zero: BaseTrainerProtocol#is_process_zero().
  BaseTrainerProtocol.is_enable: BaseTrainerProtocol#is_enable().
  BaseTrainerProtocol._initialize_attributes: BaseTrainerProtocol#_initialize_attributes().
  BaseTrainerProtocol._initialize_memory_tracking: BaseTrainerProtocol#_initialize_memory_tracking().
  BaseTrainerProtocol.initialize_trainer_utils: BaseTrainerProtocol#initialize_trainer_utils().
  BaseTrainerProtocol._initialize_wandb: BaseTrainerProtocol#_initialize_wandb().
  BaseTrainerProtocol._initialize_timer: BaseTrainerProtocol#_initialize_timer().
  BaseTrainerProtocol._configure_dataloaders: BaseTrainerProtocol#_configure_dataloaders().
  BaseTrainerProtocol._configure_model: BaseTrainerProtocol#_configure_model().
  BaseTrainerProtocol._configure_functions: BaseTrainerProtocol#_configure_functions().
  BaseTrainerProtocol._configure_state: BaseTrainerProtocol#_configure_state().
  BaseTrainerProtocol.create_grain_collect_function: BaseTrainerProtocol#create_grain_collect_function().
  BaseTrainerProtocol.create_tfds_collect_function: BaseTrainerProtocol#create_tfds_collect_function().
  BaseTrainerProtocol.create_collect_function: BaseTrainerProtocol#create_collect_function().
  BaseTrainerProtocol._get_current_step: BaseTrainerProtocol#_get_current_step().
  BaseTrainerProtocol._save_readme: BaseTrainerProtocol#_save_readme().
  BaseTrainerProtocol._format_partition_rules: BaseTrainerProtocol#_format_partition_rules().
  BaseTrainerProtocol._get_device_info: BaseTrainerProtocol#_get_device_info().
  BaseTrainerProtocol._get_information: BaseTrainerProtocol#_get_information().
  BaseTrainerProtocol.save_information: BaseTrainerProtocol#save_information().
  BaseTrainerProtocol.specs_to_name_sharding: BaseTrainerProtocol#specs_to_name_sharding().
  BaseTrainerProtocol.count_model_parameters: BaseTrainerProtocol#count_model_parameters().
  BaseTrainerProtocol._should_run_evaluation: BaseTrainerProtocol#_should_run_evaluation().
  BaseTrainerProtocol._setup_initial_metrics: BaseTrainerProtocol#_setup_initial_metrics().
  BaseTrainerProtocol._get_next_batch: BaseTrainerProtocol#_get_next_batch().
  BaseTrainerProtocol._train_shared_fn_extra_args: BaseTrainerProtocol#_train_shared_fn_extra_args().
  BaseTrainerProtocol._eval_shared_fn_extra_args: BaseTrainerProtocol#_eval_shared_fn_extra_args().
  BaseTrainerProtocol._train_shared_fn_static_args: BaseTrainerProtocol#_train_shared_fn_static_args().
  BaseTrainerProtocol._eval_shared_fn_static_args: BaseTrainerProtocol#_eval_shared_fn_static_args().
  BaseTrainerProtocol._finalize_training: BaseTrainerProtocol#_finalize_training().
  BaseTrainerProtocol.start_training_hook: BaseTrainerProtocol#start_training_hook().
  BaseTrainerProtocol.start_evaluation_hook: BaseTrainerProtocol#start_evaluation_hook().
  BaseTrainerProtocol._setup_static_metrics: BaseTrainerProtocol#_setup_static_metrics().
  BaseTrainerProtocol.compile_aot: BaseTrainerProtocol#compile_aot().
  BaseTrainerProtocol.finish: BaseTrainerProtocol#finish().
  BaseTrainerProtocol._ensure_functions_compiled: BaseTrainerProtocol#_ensure_functions_compiled().
  BaseTrainerProtocol.calculate_number_total_flops: BaseTrainerProtocol#calculate_number_total_flops().
  BaseTrainerProtocol.wandb_runtime: BaseTrainerProtocol#wandb_runtime.
  BaseTrainerProtocol.checkpoint_manager: BaseTrainerProtocol#checkpoint_manager.
  BaseTrainerProtocol.memory_monitor: BaseTrainerProtocol#memory_monitor.
  BaseTrainerProtocol._forward_flops_per_token: BaseTrainerProtocol#_forward_flops_per_token.
  BaseTrainerProtocol._backward_flops_per_token: BaseTrainerProtocol#_backward_flops_per_token.
  BaseTrainerProtocol._evaluation_time_start: BaseTrainerProtocol#_evaluation_time_start.
  TrainerConfigureDataloaderOutput.dataloader_train: TrainerConfigureDataloaderOutput#dataloader_train.
  TrainerConfigureDataloaderOutput.max_training_steps: TrainerConfigureDataloaderOutput#max_training_steps.
  TrainerConfigureDataloaderOutput.dataloader_eval: TrainerConfigureDataloaderOutput#dataloader_eval.
  TrainerConfigureDataloaderOutput.max_evaluation_steps: TrainerConfigureDataloaderOutput#max_evaluation_steps.
  TrainerConfigureModelOutput.tx: TrainerConfigureModelOutput#tx.
  TrainerConfigureModelOutput.scheduler: TrainerConfigureModelOutput#scheduler.
  TrainerConfigureFunctionOutput.sharded_training_step_function: TrainerConfigureFunctionOutput#sharded_training_step_function.
  TrainerConfigureFunctionOutput.mesh: TrainerConfigureFunctionOutput#mesh.
  TrainerConfigureFunctionOutput.checkpoint_manager: TrainerConfigureFunctionOutput#checkpoint_manager.
  TrainerConfigureFunctionOutput.sharded_evaluation_step_function: TrainerConfigureFunctionOutput#sharded_evaluation_step_function.
  BaseTrainerProtocol.finetune: BaseTrainerProtocol#finetune.
  BaseTrainerProtocol.mesh: BaseTrainerProtocol#mesh.
  __all__: __all__.
  logger: logger.
  TrainerOutput.mesh: TrainerOutput#mesh.
  TrainerOutput.last_save_file_name: TrainerOutput#last_save_file_name.
  TrainerOutput.checkpoint_path: TrainerOutput#checkpoint_path.
  BaseTrainerProtocol.checkpoint_path: BaseTrainerProtocol#checkpoint_path.
  BaseTrainerProtocol.dtype: BaseTrainerProtocol#dtype.
  BaseTrainerProtocol.param_dtype: BaseTrainerProtocol#param_dtype.
  BaseTrainerProtocol.state_shape: BaseTrainerProtocol#state_shape.
  BaseTrainerProtocol.state_partition_spec: BaseTrainerProtocol#state_partition_spec.
  BaseTrainerProtocol.state_named_sharding: BaseTrainerProtocol#state_named_sharding.
  BaseTrainerProtocol.state: BaseTrainerProtocol#state.
  BaseTrainerProtocol.__repr__: BaseTrainerProtocol#__repr__().
  BaseTrainerProtocol.__str__: BaseTrainerProtocol#__str__().
---
# Module: [`easydel/trainers/trainer_protocol.py`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py)

## Classes
### `BaseTrainerProtocol`
- def: [`easydel/trainers/trainer_protocol.py:153`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L153)
- doc: Abstract base protocol defining the interface for all trainer implementations.
- signature: `class BaseTrainerProtocol(metaclass=ABCMeta):`
- members:
  - `__init__(self, arguments: TrainingArguments | None = None, model_state: EasyDeLState | None = None, model: tp.type[EasyDeLBaseModule] | None = None, dataset_train: Dataset | None = None, dataset_eval: Dataset | None = None, data_collator: tp.Callable | None = None, finetune: bool = True, **deprecated_kwargs)` — [`L225`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L225) — Initialize the trainer with model and training configuration.
  - `__repr__(self)` — [`L1174`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1174) — Return a string representation of the trainer.
  - `__str__(self)` — [`L1179`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1179) — Return a string representation of the trainer.
  - `_configure_dataloaders(self)` — [`L358`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L358) — Configure dataloaders for training and evaluation.
  - `_configure_functions(self)` — [`L392`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L392) — Configure and JIT-compile training and evaluation step functions.
  - `_configure_grain_dataloader(self)` — [`L992`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L992) — Configure Grain dataloaders for training and evaluation.
  - `_configure_model(self)` — [`L375`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L375) — Configure model, optimizer, scheduler, and configuration.
  - `_configure_state(self)` — [`L409`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L409) — Configure and shard model state across devices.
  - `_configure_tfds_dataloader(self)` — [`L1010`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1010) — Configure TensorFlow Dataset dataloaders.
  - `_create_hf_model_config(self, state: EasyDeLState, model_config, model_type)` — [`L634`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L634) — Creates a Hugging Face model config from the current state
  - `_ensure_functions_compiled(self)` — [`L1169`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1169) — Ensure functions are compiled.
  - `_eval_epoch(self, state: EasyDeLState, eval_dataset, eval_iter, metrics_tracker: MetricsTracker, step_metrics: StepMetrics, pbar: BaseProgressBar)` — [`L949`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L949) — Handle evaluation for a single epoch.
  - `_execute_eval_step(self, state, batch)` — [`L1031`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1031) — Execute a single evaluation step.
  - `_execute_train_step(self, state, batch)` — [`L1048`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1048) — Execute a single training step.
  - `_finalize_training(self, output, run_exception)` — [`L1068`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1068) — Finalize training and prepare output.
  - `_format_partition_rules(self)` — [`L583`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L583) — Format partition rules with proper indentation and formatting.
  - `_get_current_step(self, state)` — [`L569`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L569) — Get the current step number.
  - `_get_device_info(self)` — [`L588`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L588) — Get information about available devices.
  - `_get_information(self)` — [`L593`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L593) — Generate formatted information about the model and training setup.
  - `_get_next_batch(self, data_iter, dataloader)` — [`L770`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L770) — Get next batch from iterator, reinitializing if needed.
  - `_handle_training_interruption(self, state: EasyDeLState, exception: Exception)` — [`L731`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L731) — Handle training interruption gracefully.
  - `_initialize_attributes(self)` — [`L276`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L276) — Initialize all trainer attributes with default values.
  - `_initialize_memory_tracking(self)` — [`L297`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L297) — Initialize memory tracking for GPU/TPU memory monitoring.
  - `_initialize_timer(self)` — [`L344`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L344) — Initialize the timer for performance monitoring.
  - `_initialize_wandb(self)` — [`L330`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L330) — Initialize Weights & Biases logging integration.
  - `_prepare_training_output(self, state: EasyDeLState, run_exception: Exception | None = None)` — [`L706`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L706) — Prepare final training output after training completion.
  - `_preprocess_batch_input(self, state: EasyDeLState, batch: dict[str, jax.Array], is_train: bool)` — [`L1160`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1160) — hook call before passing data to function (called in `_execute` functions)
  - `_run_evaluation(self, state: EasyDeLState, metrics_tracker: MetricsTracker, step_metrics: StepMetrics, start_time: float)` — [`L892`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L892) — Execute the core evaluation loop.
  - `_run_training_loop(self, state: EasyDeLState, metrics_tracker: MetricsTracker, step_metrics: StepMetrics)` — [`L868`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L868) — Execute the core training loop.
  - `_save_readme(self, checkpoint_dir)` — [`L576`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L576) — Saves a README file with model and training information.
  - `_save_state(self, state: EasyDeLState, *args, **kwargs)` — [`L554`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L554) — Saves the model state to a checkpoint file.
  - `_save_to_torch(self, state: EasyDeLState, save_directory: str | os.PathLike, easystate_to_huggingface_model_kwargs: dict | None = None, torch_save_pretrained_kwargs: dict | None = None)` — [`L621`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L621) — Saves the model state to a Torch compatible directory.
  - `_setup_initial_metrics(self, state)` — [`L753`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L753) — Setup initial metrics logging at training start.
  - `_setup_static_metrics(self)` — [`L1126`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1126) — Setup static metrics for logging.
  - `_should_run_evaluation(self, current_step)` — [`L689`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L689) — Determine if evaluation should be run at current step.
  - `_train_epoch(self, state: EasyDeLState, train_dataset, train_iter, metrics_tracker: MetricsTracker, step_metrics: StepMetrics, pbar: BaseProgressBar, epoch: int)` — [`L917`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L917) — Handle training for a single epoch.
  - `apply_training_hooks(self, metrics: LossMetrics)` — [`L668`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L668) — Apply training hooks to check for issues and enforce limits.
  - `calculate_number_total_flops(self, params, is_training=True)` — [`L1184`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1184) — Calculate total FLOPs for the model.
  - `compile_aot(self)` — [`L1131`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1131) — Compile the state ahead of time for faster execution.
  - `configure_dataloaders(self)` — [`L511`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L511) — Configure dataloaders for training and evaluation.
  - `configure_functions(self)` — [`L497`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L497) — Configure and JIT-compile training and evaluation step functions.
  - `configure_model(self)` — [`L533`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L533) — Configure model, optimizer, scheduler, and configuration.
  - `count_model_parameters(prm)` — [`L652`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L652) — Count total number of model parameters.
  - `create_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"])` — [`L473`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L473) — Create a generic data collection function for batch processing.
  - `create_grain_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"])` — [`L425`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L425) — Create a Grain data collection function for batch processing.
  - `create_progress_bar(self, total: int, desc: str = "", disabled: bool = False)` — [`L789`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L789) — Create a progress bar of the specified type.
  - `create_tfds_collect_function(self, max_sequence_length: int, truncation_mode: tp.Literal["keep_end", "keep_start"])` — [`L449`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L449) — Create a TensorFlow Dataset collection function for batch processing.
  - `eval(self, model_state: EasyDeLState)` — [`L1100`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1100) — Evaluate the model on the evaluation dataset.
  - `evaluation_batch_size(self)` — [`L265`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L265)
  - `finish(self)` — [`L1136`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1136) — Finalize the training process.
  - `initialize_trainer_utils(self)` — [`L311`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L311) — Initialize all trainer utilities in the correct order.
  - `is_enable(self)` — [`L273`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L273)
  - `is_process_zero(self)` — [`L269`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L269)
  - `log_metrics(self, metrics: dict[str, float], pbar: BaseProgressBar, step: int, mode: str = "train", *, update_progress: bool = True, log_to_backends: bool = True, force_report: bool = False)` — [`L838`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L838) — Log metrics and update progress bar.
  - `log_watchers(self, state: EasyDeLState, step: int)` — [`L829`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L829) — Run registered LogWatcher instances and log their metrics.
  - `log_weight_distribution(self, state: EasyDeLState, step: int)` — [`L815`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L815) — Log weight distribution statistics.
  - `model(self)` — [`L253`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L253)
  - `on_step_end(self, state: EasyDeLState, metrics: MetricsType, step: int)` — [`L1150`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1150) — hook process to call in start of the step.
  - `on_step_start(self, state: EasyDeLState, step: int)` — [`L1141`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1141) — hook process to call in start of the step.
  - `save_information(self, output_path: str | ePathLike)` — [`L600`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L600) — Save the generated information to a markdown file.
  - `save_pretrained(self, state: EasyDeLState, save_directory: str | None = None, to_torch: bool = False, easystate_to_huggingface_model_kwargs: dict | None = None, torch_save_pretrained_kwargs: dict | None = None)` — [`L607`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L607) — Saves the model state as a checkpoint file or to a Torch compatible directory.
  - `specs_to_name_sharding(self, tree, mesh=None)` — [`L646`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L646) — Convert specs to named sharding.
  - `start_evaluation_hook(self)` — [`L1121`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1121) — Hook to run before evaluation starts.
  - `start_training_hook(self)` — [`L1117`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1117) — Hook to run before training starts.
  - `train(self)` — [`L1086`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L1086) — Execute the complete training process.
  - `training_batch_size(self)` — [`L261`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L261)
  - `arguments` — [`L172`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L172)
  - `checkpoint_manager` — [`L199`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L199)
  - `checkpoint_path` — [`L177`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L177)
  - `config` — [`L188`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L188)
  - `data_collator` — [`L175`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L175)
  - `dataloader_eval` — [`L184`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L184)
  - `dataloader_train` — [`L183`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L183)
  - `dataset_eval` — [`L174`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L174)
  - `dataset_train` — [`L173`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L173)
  - `dtype` — [`L178`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L178)
  - `evalu_tracker` — [`L196`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L196)
  - `finetune` — [`L176`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L176)
  - `max_evaluation_steps` — [`L186`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L186)
  - `max_training_steps` — [`L185`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L185)
  - `memory_monitor` — [`L205`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L205)
  - `mesh` — [`L198`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L198)
  - `model_state` — [`L191`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L191)
  - `param_dtype` — [`L179`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L179)
  - `pruning_module` — [`L204`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L204)
  - `scheduler` — [`L189`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L189)
  - `sharded_evaluation_step_function` — [`L195`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L195)
  - `sharded_training_step_function` — [`L193`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L193)
  - `state` — [`L203`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L203)
  - `state_named_sharding` — [`L202`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L202)
  - `state_partition_spec` — [`L201`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L201)
  - `state_shape` — [`L200`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L200)
  - `state_shardings` — [`L220`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L220)
  - `timer` — [`L181`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L181)
  - `train_tracker` — [`L194`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L194)
  - `tx` — [`L190`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L190)
  - `wandb_runtime` — [`L182`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L182)
- protocol/private: `_backward_flops_per_token`[`L208`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L208), `_eval_shared_fn_extra_args`[`L983`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L983), `_eval_shared_fn_extra_args_`[`L217`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L217), `_eval_shared_fn_static_args`[`L989`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L989), `_eval_shared_fn_static_args_`[`L216`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L216), `_evaluation_time_start`[`L222`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L222), `_extra_backward_flops_per_token`[`L211`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L211), `_extra_forward_flops_per_token`[`L210`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L210), `_forward_flops_per_token`[`L207`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L207), `_model`[`L187`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L187), `_resumed_from_checkpoint`[`L219`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L219), `_train_shared_fn_extra_args`[`L980`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L980), `_train_shared_fn_extra_args_`[`L214`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L214), `_train_shared_fn_static_args`[`L986`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L986), `_train_shared_fn_static_args_`[`L213`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L213), `_training_time_start`[`L221`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L221)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLBaseConfig`](../infra/base_config.md#EasyDeLBaseConfig), [`EasyDeLState`](../infra/base_state.md#EasyDeLState), [`LossMetrics`](../infra/loss_utils.md#LossMetrics), [`_train_epoch`](trainer/trainer.md#Trainer._train_epoch), [`TrainingArguments`](training_configurations.md#TrainingArguments), [`_get_information`](base_trainer.md#BaseTrainer._get_information), [`_eval_epoch`](trainer/trainer.md#Trainer._eval_epoch), [`TrainerConfigureFunctionOutput`](trainer_protocol.md#TrainerConfigureFunctionOutput), [`_configure_state`](base_trainer.md#BaseTrainer._configure_state), [`_configure_tfds_dataloader`](base_trainer.md#BaseTrainer._configure_tfds_dataloader), [`_run_training_loop`](trainer/trainer.md#Trainer._run_training_loop), [`_configure_grain_dataloader`](base_trainer.md#BaseTrainer._configure_grain_dataloader), [`model`](base_trainer.md#BaseTrainer.model), [`BaseTrainer`](base_trainer.md#BaseTrainer), [`_prepare_training_output`](base_trainer.md#BaseTrainer._prepare_training_output), [`_configure_dataloaders`](base_trainer.md#BaseTrainer._configure_dataloaders), [`_save_state`](base_trainer.md#BaseTrainer._save_state), [`_setup_initial_metrics`](base_trainer.md#BaseTrainer._setup_initial_metrics), [`_execute_train_step`](trainer/trainer.md#Trainer._execute_train_step), [`_preprocess_batch_input`](base_trainer.md#BaseTrainer._preprocess_batch_input), [`_configure_model`](base_trainer.md#BaseTrainer._configure_model), [`_run_evaluation`](trainer/trainer.md#Trainer._run_evaluation), [`_configure_functions`](base_trainer.md#BaseTrainer._configure_functions), [`apply_training_hooks`](base_trainer.md#BaseTrainer.apply_training_hooks), [`log_metrics`](base_trainer.md#BaseTrainer.log_metrics), [`BaseProgressBar`](metrics.md#BaseProgressBar), [`_train_shared_fn_static_args`](base_trainer.md#BaseTrainer._train_shared_fn_static_args), [`MetricsType`](training_configurations.md#MetricsType), [`_eval_shared_fn_static_args`](base_trainer.md#BaseTrainer._eval_shared_fn_static_args), [`train`](trainer/trainer.md#Trainer.train), [`_train_shared_fn_extra_args`](base_trainer.md#BaseTrainer._train_shared_fn_extra_args), [`_eval_shared_fn_extra_args`](base_trainer.md#BaseTrainer._eval_shared_fn_extra_args), [`create_progress_bar`](base_trainer.md#BaseTrainer.create_progress_bar), [`configure_model`](base_trainer.md#BaseTrainer.configure_model), [`configure_dataloaders`](base_trainer.md#BaseTrainer.configure_dataloaders), [`eval`](trainer/trainer.md#Trainer.eval), [`_initialize_attributes`](base_trainer.md#BaseTrainer._initialize_attributes), [`compile_aot`](base_trainer.md#BaseTrainer.compile_aot), [`initialize_trainer_utils`](base_trainer.md#BaseTrainer.initialize_trainer_utils)  (+42 more)
- used by: [`_train_epoch`](trainer/trainer.md#Trainer._train_epoch), [`configure_functions`](distillation_trainer/distillation_trainer.md#DistillationTrainer.configure_functions), [`configure_functions`](direct_preference_optimization_trainer/dpo_trainer.md#DPOTrainer.configure_functions), [`configure_functions`](trainer/trainer.md#Trainer.configure_functions), [`configure_functions`](group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.configure_functions), [`configure_functions`](generalized_knowledge_distillation_trainer/gkd_trainer.md#GKDTrainer.configure_functions), [`_get_information`](base_trainer.md#BaseTrainer._get_information), [`_eval_dataset_steps_auto_clamped`](base_trainer.md#BaseTrainer._eval_dataset_steps_auto_clamped), [`_tpu_preemption_sync_available`](base_trainer.md#BaseTrainer._tpu_preemption_sync_available), [`configure_functions`](on_policy_distillation_trainer/on_policy_distillation_trainer.md#OnPolicyDistillationTrainer.configure_functions), [`_eval_epoch`](trainer/trainer.md#Trainer._eval_epoch), [`configure_functions`](odds_ratio_preference_optimization_trainer/orpo_trainer.md#ORPOTrainer.configure_functions), [`configure_functions`](sparse_distillation_trainer/sparse_distillation_trainer.md#SparseDistillationTrainer.configure_functions), [`concatenated_forward`](kto_trainer/kto_trainer.md#KTOTrainer.concatenated_forward), [`configure_functions`](self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer.configure_functions), [`configure_functions`](xpo_trainer/xpo_trainer.md#XPOTrainer.configure_functions), [`configure_functions`](proximal_policy_optimization_trainer/ppo_trainer.md#PPOTrainer.configure_functions), [`concatenated_forward`](contrastive_preference_optimization_trainer/cpo_trainer.md#CPOTrainer.concatenated_forward), [`configure_functions`](embedding_trainer/embedding_trainer.md#EmbeddingTrainer.configure_functions), [`__init__`](base_trainer.md#BaseTrainer.__init__), [`configure_functions`](reward_trainer/reward_trainer.md#RewardTrainer.configure_functions), [`concatenated_forward`](binary_classifier_optimization_trainer/bco_trainer.md#BCOTrainer.concatenated_forward), [`_configure_state`](base_trainer.md#BaseTrainer._configure_state), [`_configure_tfds_dataloader`](base_trainer.md#BaseTrainer._configure_tfds_dataloader), [`_run_training_loop`](trainer/trainer.md#Trainer._run_training_loop), [`configure_functions`](nash_md_trainer/nash_md_trainer.md#NashMDTrainer.configure_functions), [`train`](ray_scaler/distributed_trainer.md#RayDistributedTrainer.train), [`configure_functions`](group_filtered_policy_optimization/gfpo_trainer.md#GFPOTrainer.configure_functions), [`_configure_grain_dataloader`](base_trainer.md#BaseTrainer._configure_grain_dataloader), [`model`](base_trainer.md#BaseTrainer.model), [`BaseTrainer`](base_trainer.md#BaseTrainer), [`_configure_dataloaders`](base_trainer.md#BaseTrainer._configure_dataloaders), [`configure_dataloaders`](direct_preference_optimization_trainer/dpo_trainer.md#DPOTrainer.configure_dataloaders), [`create_generate_function`](base_trainer.md#BaseTrainer.create_generate_function), [`_execute_train_step`](trainer/trainer.md#Trainer._execute_train_step), [`_configure_model`](base_trainer.md#BaseTrainer._configure_model), [`_run_evaluation`](trainer/trainer.md#Trainer._run_evaluation), [`processing_class`](base_trainer.md#BaseTrainer.processing_class), [`_configure_functions`](base_trainer.md#BaseTrainer._configure_functions), [`apply_training_hooks`](base_trainer.md#BaseTrainer.apply_training_hooks)  (+22 more)

### `TrainerConfigureDataloaderOutput`
- def: [`easydel/trainers/trainer_protocol.py:73`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L73)
- doc: Output configuration for dataloader setup.
- signature: `class TrainerConfigureDataloaderOutput:`
- members:
  - `dataloader_eval` — [`L89`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L89)
  - `dataloader_train` — [`L87`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L87)
  - `max_evaluation_steps` — [`L90`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L90)
  - `max_training_steps` — [`L88`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L88)
- used by: [`_configure_tfds_dataloader`](base_trainer.md#BaseTrainer._configure_tfds_dataloader), [`_configure_grain_dataloader`](base_trainer.md#BaseTrainer._configure_grain_dataloader), [`_configure_dataloaders`](base_trainer.md#BaseTrainer._configure_dataloaders), [`configure_dataloaders`](base_trainer.md#BaseTrainer.configure_dataloaders), [`_configure_grain_dataloader`](trainer_protocol.md#BaseTrainerProtocol._configure_grain_dataloader), [`_configure_tfds_dataloader`](trainer_protocol.md#BaseTrainerProtocol._configure_tfds_dataloader), [`configure_dataloaders`](trainer_protocol.md#BaseTrainerProtocol.configure_dataloaders)

### `TrainerConfigureFunctionOutput`
- def: [`easydel/trainers/trainer_protocol.py:114`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L114)
- doc: Output configuration for training and evaluation functions.
- signature: `class TrainerConfigureFunctionOutput:`
- members:
  - `checkpoint_manager` — [`L129`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L129)
  - `mesh` — [`L128`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L128)
  - `sharded_evaluation_step_function` — [`L130`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L130)
  - `sharded_training_step_function` — [`L127`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L127)
- used by: [`configure_functions`](distillation_trainer/distillation_trainer.md#DistillationTrainer.configure_functions), [`configure_functions`](direct_preference_optimization_trainer/dpo_trainer.md#DPOTrainer.configure_functions), [`configure_functions`](trainer/trainer.md#Trainer.configure_functions), [`configure_functions`](group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.configure_functions), [`configure_functions`](generalized_knowledge_distillation_trainer/gkd_trainer.md#GKDTrainer.configure_functions), [`configure_functions`](on_policy_distillation_trainer/on_policy_distillation_trainer.md#OnPolicyDistillationTrainer.configure_functions), [`configure_functions`](odds_ratio_preference_optimization_trainer/orpo_trainer.md#ORPOTrainer.configure_functions), [`configure_functions`](sparse_distillation_trainer/sparse_distillation_trainer.md#SparseDistillationTrainer.configure_functions), [`concatenated_forward`](kto_trainer/kto_trainer.md#KTOTrainer.concatenated_forward), [`configure_functions`](self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer.configure_functions), [`configure_functions`](xpo_trainer/xpo_trainer.md#XPOTrainer.configure_functions), [`configure_functions`](proximal_policy_optimization_trainer/ppo_trainer.md#PPOTrainer.configure_functions), [`concatenated_forward`](contrastive_preference_optimization_trainer/cpo_trainer.md#CPOTrainer.concatenated_forward), [`configure_functions`](embedding_trainer/embedding_trainer.md#EmbeddingTrainer.configure_functions), [`configure_functions`](reward_trainer/reward_trainer.md#RewardTrainer.configure_functions), [`concatenated_forward`](binary_classifier_optimization_trainer/bco_trainer.md#BCOTrainer.concatenated_forward), [`configure_functions`](contrastive_preference_optimization_trainer/cpo_trainer.md#CPOTrainer.configure_functions), [`configure_functions`](nash_md_trainer/nash_md_trainer.md#NashMDTrainer.configure_functions), [`compute_refmodel_logps`](nash_md_trainer/nash_md_trainer.md#NashMDTrainer.compute_refmodel_logps), [`_configure_functions`](base_trainer.md#BaseTrainer._configure_functions), [`compute_refmodel_logps`](self_distillation_policy_optimization/sdpo_trainer.md#SDPOTrainer.compute_refmodel_logps), [`compute_rollout_logps_values`](proximal_policy_optimization_trainer/ppo_trainer.md#PPOTrainer.compute_rollout_logps_values), [`configure_functions`](binary_classifier_optimization_trainer/bco_trainer.md#BCOTrainer.configure_functions), [`configure_functions`](kto_trainer/kto_trainer.md#KTOTrainer.configure_functions), [`compute_refmodel_logps`](group_relative_policy_optimization/grpo_trainer.md#GRPOTrainer.compute_refmodel_logps), [`concatenated_forward`](direct_preference_optimization_trainer/dpo_trainer.md#DPOTrainer.concatenated_forward), [`concatenated_forward`](odds_ratio_preference_optimization_trainer/orpo_trainer.md#ORPOTrainer.concatenated_forward), [`configure_functions`](base_trainer.md#BaseTrainer.configure_functions), [`configure_functions`](trainer_protocol.md#BaseTrainerProtocol.configure_functions)

### `TrainerConfigureModelOutput`
- def: [`easydel/trainers/trainer_protocol.py:94`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L94)
- doc: Output configuration for model setup.
- signature: `class TrainerConfigureModelOutput:`
- members:
  - `config` — [`L110`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L110)
  - `model` — [`L107`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L107)
  - `scheduler` — [`L109`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L109)
  - `tx` — [`L108`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L108)
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLBaseConfig`](../infra/base_config.md#EasyDeLBaseConfig)
- used by: [`_configure_model`](base_trainer.md#BaseTrainer._configure_model), [`configure_model`](base_trainer.md#BaseTrainer.configure_model), [`configure_model`](trainer_protocol.md#BaseTrainerProtocol.configure_model)

### `TrainerOutput`
- def: [`easydel/trainers/trainer_protocol.py:134`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L134)
- doc: Final output from the training process.
- signature: `class TrainerOutput:`
- members:
  - `checkpoint_path` — [`L150`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L150)
  - `last_save_file_name` — [`L149`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L149)
  - `mesh` — [`L148`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L148)
  - `state` — [`L147`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L147)
- uses (calls/refs, reference-scoped): [`EasyDeLState`](../infra/base_state.md#EasyDeLState)
- used by: [`_prepare_training_output`](base_trainer.md#BaseTrainer._prepare_training_output), [`train`](trainer/trainer.md#Trainer.train)

## Module values
- `__all__` — [`L48`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L48)
- `logger` — [`L69`](../../../../../../raw/code/EasyDeL/easydel/trainers/trainer_protocol.py#L69)

