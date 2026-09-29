---
title: 'Module: src/maxdiffusion/trainers/dreambooth_trainer.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/trainers/dreambooth_trainer.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.trainers.dreambooth_trainer`/
symbols:
  DreamboothTrainer.rng: DreamboothTrainer#rng.
  DreamboothTrainer.compile_train_step: DreamboothTrainer#compile_train_step().
  DreamboothTrainer.training_loop: DreamboothTrainer#training_loop().
  DreamboothTrainer._profiler: DreamboothTrainer#_profiler.
  DreamboothTrainer.prepare_w_prior_preservation: DreamboothTrainer#prepare_w_prior_preservation().
  _train_step: _train_step().
  PromptDataset.__len__: PromptDataset#__len__().
  PromptDataset.__getitem__: PromptDataset#__getitem__().
  DreamboothTrainer.pre_training_steps: DreamboothTrainer#pre_training_steps().
  PromptDataset: PromptDataset#
  PromptDataset.prompt: PromptDataset#prompt.
  PromptDataset.num_samples: PromptDataset#num_samples.
  DreamboothTrainer.get_shaped_batch: DreamboothTrainer#get_shaped_batch().
  _train_step.compute_loss: _train_step().compute_loss().
  PromptDataset.__init__: PromptDataset#__init__().
  DreamboothTrainer: DreamboothTrainer#
  DreamboothTrainer.__init__: DreamboothTrainer#__init__().
  DreamboothTrainer.post_training_steps: DreamboothTrainer#post_training_steps().
  DreamboothTrainer.create_scheduler: DreamboothTrainer#create_scheduler().
  DreamboothTrainer.get_data_shardings: DreamboothTrainer#get_data_shardings().
  DreamboothTrainer.load_dataset: DreamboothTrainer#load_dataset().
---
# Module: [`src/maxdiffusion/trainers/dreambooth_trainer.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py)

## Classes
### `DreamboothTrainer`  ·  implements/extends BaseStableDiffusionTrainer
- def: [`src/maxdiffusion/trainers/dreambooth_trainer.py:69`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L69)
- signature: `class DreamboothTrainer(BaseStableDiffusionTrainer):`
- members:
  - `compile_train_step(self, pipeline, params, train_states, state_shardings, data_shardings)` — [`L166`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L166)
  - `create_scheduler(self, pipeline, params)` — [`L103`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L103)
  - `get_data_shardings(self)` — [`L110`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L110)
  - `get_shaped_batch(self, config, pipeline)` — [`L75`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L75)
  - `load_dataset(self, pipeline, params, train_states)` — [`L119`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L119)
  - `post_training_steps(self, pipeline, params, train_states)` — [`L100`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L100)
  - `pre_training_steps(self)` — [`L97`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L97)
  - `prepare_w_prior_preservation(self)` — [`L129`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L129)
  - `training_loop(self, p_train_step, pipeline, params, train_states, data_iterator, learning_rate_scheduler)` — [`L185`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L185)
  - `rng` — [`L153`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L153)
- protocol/private: `__init__`[`L72`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L72), `_profiler`[`L70`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L70)
- uses (calls/refs, reference-scoped): [`_train_step`](dreambooth_trainer.md#_train_step), [`PromptDataset`](dreambooth_trainer.md#PromptDataset)

### `PromptDataset`  ·  implements/extends Dataset
- def: [`src/maxdiffusion/trainers/dreambooth_trainer.py:52`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L52)
- doc: A simple dataset to prepare the prompts to generate class images on multiple GPUs.
- signature: `class PromptDataset(Dataset):`
- members:
  - `num_samples` — [`L57`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L57)
  - `prompt` — [`L56`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L56)
- protocol/private: `__getitem__`[`L62`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L62), `__init__`[`L55`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L55), `__len__`[`L59`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L59)
- used by: [`prepare_w_prior_preservation`](dreambooth_trainer.md#DreamboothTrainer.prepare_w_prior_preservation)

## Functions
- `_train_step(unet_state, text_encoder_state, batch, train_rng, config, pipeline, params)` — [`L252`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L252)
- `compute_loss(state_params)` — [`L268`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/trainers/dreambooth_trainer.py#L268)

