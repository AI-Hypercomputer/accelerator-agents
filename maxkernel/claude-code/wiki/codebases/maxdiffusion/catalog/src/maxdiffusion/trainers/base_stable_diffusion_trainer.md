---
title: 'Module: src/maxdiffusion/trainers/base_stable_diffusion_trainer.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/trainers/base_stable_diffusion_trainer.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.trainers.base_stable_diffusion_trainer`/
symbols:
  BaseStableDiffusionTrainer.start_training: BaseStableDiffusionTrainer#start_training().
  BaseStableDiffusionTrainer._time_and_log_call: BaseStableDiffusionTrainer#_time_and_log_call().
  _log_to_file: _log_to_file().
  BaseStableDiffusionTrainer.per_device_tflops: BaseStableDiffusionTrainer#per_device_tflops.
  BaseStableDiffusionTrainer.compile_train_step: BaseStableDiffusionTrainer#compile_train_step().
  BaseStableDiffusionTrainer.pre_training_steps: BaseStableDiffusionTrainer#pre_training_steps().
  BaseStableDiffusionTrainer.post_training_steps: BaseStableDiffusionTrainer#post_training_steps().
  BaseStableDiffusionTrainer.load_dataset: BaseStableDiffusionTrainer#load_dataset().
  BaseStableDiffusionTrainer.training_loop: BaseStableDiffusionTrainer#training_loop().
  BaseStableDiffusionTrainer.get_data_shardings: BaseStableDiffusionTrainer#get_data_shardings().
  BaseStableDiffusionTrainer.create_scheduler: BaseStableDiffusionTrainer#create_scheduler().
  BaseStableDiffusionTrainer.calculate_tflops: BaseStableDiffusionTrainer#calculate_tflops().
  BaseStableDiffusionTrainer: BaseStableDiffusionTrainer#
  BaseStableDiffusionTrainer.__init__: BaseStableDiffusionTrainer#__init__().
  BaseStableDiffusionTrainer.data_sharding: BaseStableDiffusionTrainer#data_sharding.
  BaseStableDiffusionTrainer.writer: BaseStableDiffusionTrainer#writer.
  BaseStableDiffusionTrainer.p_train_step: BaseStableDiffusionTrainer#p_train_step.
  BaseStableDiffusionTrainer.get_shaped_batch: BaseStableDiffusionTrainer#get_shaped_batch().
---
# Module: [`src/maxdiffusion/trainers/base_stable_diffusion_trainer.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py)

## Classes
### `BaseStableDiffusionTrainer`  ·  implements/extends BaseStableDiffusionCheckpointer
- def: [`src/maxdiffusion/trainers/base_stable_diffusion_trainer.py:39`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L39)
- signature: `class BaseStableDiffusionTrainer(BaseStableDiffusionCheckpointer):`
- members:
  - `_time_and_log_call(self, func_obj: Callable[..., Any], *func_args: Any, description: str = "", **func_kwargs: Any)` — [`L85`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L85) — Times a function call, logs its duration, and returns its result.
  - `calculate_tflops(self, pipeline, params)` — [`L108`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L108)
  - `compile_train_step(self, pipeline, params, train_states, state_shardings, data_shardings)` — [`L58`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L58)
  - `create_scheduler(self, pipeline, params)` — [`L82`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L82)
  - `get_data_shardings(self)` — [`L78`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L78)
  - `get_shaped_batch(self, config, pipeline)` — [`L54`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L54)
  - `load_dataset(self, pipeline, params, train_states)` — [`L70`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L70)
  - `post_training_steps(self, pipeline, params, train_states)` — [`L66`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L66)
  - `pre_training_steps(self)` — [`L62`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L62)
  - `start_training(self)` — [`L115`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L115)
  - `training_loop(self, p_train_step, pipeline, params, train_states, data_iterator, unet_learning_rate_scheduler)` — [`L74`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L74)
  - `data_sharding` — [`L45`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L45)
  - `p_train_step` — [`L51`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L51)
  - `per_device_tflops` — [`L47`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L47)
  - `writer` — [`L49`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L49)
- protocol/private: `__init__`[`L41`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L41)
- uses (calls/refs, reference-scoped): [`_log_to_file`](base_stable_diffusion_trainer.md#_log_to_file)

## Functions
- `_log_to_file(message: str, log_file: str = "")` — [`L29`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_stable_diffusion_trainer.py#L29) — Appends a message to the global log file with a timestamp.

