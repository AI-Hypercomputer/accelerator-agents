---
title: 'Module: src/maxdiffusion/trainers/base_wan_trainer.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/trainers/base_wan_trainer.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.trainers.base_wan_trainer`/
symbols:
  BaseWanTrainer.config: BaseWanTrainer#config.
  BaseWanTrainer.training_loop: BaseWanTrainer#training_loop().
  BaseWanTrainer.start_training: BaseWanTrainer#start_training().
  BaseWanTrainer.checkpointer: BaseWanTrainer#checkpointer.
  BaseWanTrainer.eval: BaseWanTrainer#eval().
  BaseWanTrainer._profiler: BaseWanTrainer#_profiler.
  generate_sample: generate_sample().
  BaseWanTrainer.load_dataset: BaseWanTrainer#load_dataset().
  TrainState: TrainState#
  _to_array: _to_array().
  print_ssim: print_ssim().
  BaseWanTrainer: BaseWanTrainer#
  BaseWanTrainer._get_checkpointer: BaseWanTrainer#_get_checkpointer().
  BaseWanTrainer.create_scheduler: BaseWanTrainer#create_scheduler().
  BaseWanTrainer.calculate_tflops: BaseWanTrainer#calculate_tflops().
  BaseWanTrainer.get_data_shardings: BaseWanTrainer#get_data_shardings().
  BaseWanTrainer.get_eval_data_shardings: BaseWanTrainer#get_eval_data_shardings().
  BaseWanTrainer.get_train_step: BaseWanTrainer#get_train_step().
  BaseWanTrainer.get_eval_step: BaseWanTrainer#get_eval_step().
  TrainState.graphdef: TrainState#graphdef.
  TrainState.rest_of_state: TrainState#rest_of_state.
  BaseWanTrainer.__init__: BaseWanTrainer#__init__().
  BaseWanTrainer.post_training_steps: BaseWanTrainer#post_training_steps().
---
# Module: [`src/maxdiffusion/trainers/base_wan_trainer.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py)

## Classes
### `BaseWanTrainer`  ·  implements/extends ABC
- def: [`src/maxdiffusion/trainers/base_wan_trainer.py:84`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L84)
- signature: `class BaseWanTrainer(abc.ABC):`
- members:
  - `_get_checkpointer(self)` — [`L94`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L94) — Returns the checkpointer for the trainer.
  - `calculate_tflops(pipeline)` — [`L108`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L108)
  - `create_scheduler(self)` — [`L100`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L100) — Creates and initializes the Flow Match scheduler for training.
  - `eval(self, mesh, eval_rng_key, step, p_eval_step, state, scheduler_state, writer)` — [`L210`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L210)
  - `get_data_shardings(self, mesh)` — [`L158`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L158) — Returns data shardings for training.
  - `get_eval_data_shardings(self, mesh)` — [`L162`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L162) — Returns data shardings for evaluation.
  - `get_eval_step(self, pipeline, mesh, state_shardings, eval_data_shardings)` — [`L174`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L174) — Returns the evaluation step function.
  - `get_train_step(self, pipeline, mesh, state_shardings, data_shardings)` — [`L170`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L170) — Returns the training step function.
  - `load_dataset(self, mesh, pipeline=None, is_training=True)` — [`L166`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L166) — Loads the dataset.
  - `post_training_steps(self, pipeline, params, train_states, msg="")` — [`L97`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L97)
  - `start_training(self)` — [`L177`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L177)
  - `training_loop(self, pipeline, optimizer, learning_rate_scheduler, train_data_iterator, restore_args: dict = {})` — [`L256`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L256)
  - `checkpointer` — [`L91`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L91)
  - `config` — [`L90`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L90)
- protocol/private: `__init__`[`L87`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L87), `_profiler`[`L85`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L85)
- uses (calls/refs, reference-scoped): [`generate_sample`](base_wan_trainer.md#generate_sample), [`TrainState`](base_wan_trainer.md#TrainState), [`_to_array`](base_wan_trainer.md#_to_array), [`print_ssim`](base_wan_trainer.md#print_ssim)

### `TrainState`  ·  implements/extends TrainState
- def: [`src/maxdiffusion/trainers/base_wan_trainer.py:41`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L41)
- signature: `class TrainState(train_state.TrainState):`
- members:
  - `graphdef` — [`L42`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L42)
  - `rest_of_state` — [`L43`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L43)
- used by: [`training_loop`](base_wan_trainer.md#BaseWanTrainer.training_loop)

## Functions
- `_to_array(x)` — [`L46`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L46)
- `generate_sample(config, pipeline, filename_prefix)` — [`L52`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L52) — Generates a video to validate training did not corrupt the model
- `print_ssim(pretrained_video_path, posttrained_video_path)` — [`L65`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/base_wan_trainer.py#L65)

