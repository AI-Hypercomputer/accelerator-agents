---
title: 'Module: easydel/infra/errors.py'
type: catalog
provenance: extracted
module: easydel/infra/errors.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.errors`/EasyDeL
symbols:
  EasyDeLTimerError: TimerError#
  EasyDeLBreakRequest: BreakRequest#
  EasyDeLPreemptionSignal: PreemptionSignal#
  EasyDeLBlockWiseFFNError: BlockWiseFFNError#
  EasyDeLRuntimeError: RuntimeError#
  EasyDeLSyntaxRuntimeError: SyntaxRuntimeError#
  EasyDeLProcessError: ProcessError#
  EasyDeLComputeError: ComputeError#
  EasyDeLNotImplementedFeatureError: NotImplementedFeatureError#
---
# Module: [`easydel/infra/errors.py`](../../../../../../raw/code/EasyDeL/easydel/infra/errors.py)

## Classes
### `EasyDeLBlockWiseFFNError`  ·  implements/extends Exception
- def: [`easydel/infra/errors.py:108`](../../../../../../raw/code/EasyDeL/easydel/infra/errors.py#L108)
- doc: Error in block-wise feed-forward network operations.
- signature: `class EasyDeLBlockWiseFFNError(Exception):`
- used by: [`block_wise_ffn`](utils.md#block_wise_ffn)

### `EasyDeLBreakRequest`  ·  implements/extends Exception
- def: [`easydel/infra/errors.py:82`](../../../../../../raw/code/EasyDeL/easydel/infra/errors.py#L82)
- doc: Signal to break out of a training or generation loop.
- signature: `class EasyDeLBreakRequest(Exception):`
- used by: [`_train_epoch`](../trainers/trainer/trainer.md#Trainer._train_epoch), [`_eval_epoch`](../trainers/trainer/trainer.md#Trainer._eval_epoch), [`_execute_train_step`](../trainers/trainer/trainer.md#Trainer._execute_train_step), [`apply_training_hooks`](../trainers/base_trainer.md#BaseTrainer.apply_training_hooks)

### `EasyDeLComputeError`  ·  implements/extends Exception
- def: [`easydel/infra/errors.py:142`](../../../../../../raw/code/EasyDeL/easydel/infra/errors.py#L142)
- doc: Error during numerical computation or model forward pass.
- signature: `class EasyDeLComputeError(Exception):`

### `EasyDeLNotImplementedFeatureError`  ·  implements/extends Exception
- def: [`easydel/infra/errors.py:156`](../../../../../../raw/code/EasyDeL/easydel/infra/errors.py#L156)
- doc: Requested feature is not yet implemented.
- signature: `class EasyDeLNotImplementedFeatureError(Exception):`

### `EasyDeLPreemptionSignal`  ·  implements/extends Exception
- def: [`easydel/infra/errors.py:99`](../../../../../../raw/code/EasyDeL/easydel/infra/errors.py#L99)
- doc: Signal that training stopped early after saving a TPU preemption checkpoint.
- signature: `class EasyDeLPreemptionSignal(Exception):`
- used by: [`_train_epoch`](../trainers/trainer/trainer.md#Trainer._train_epoch), [`_prepare_training_output`](../trainers/base_trainer.md#BaseTrainer._prepare_training_output)

### `EasyDeLProcessError`  ·  implements/extends Exception
- def: [`easydel/infra/errors.py:126`](../../../../../../raw/code/EasyDeL/easydel/infra/errors.py#L126)
- doc: Error in multi-process or distributed operations.
- signature: `class EasyDeLProcessError(Exception):`

### `EasyDeLRuntimeError`  ·  implements/extends Exception
- def: [`easydel/infra/errors.py:36`](../../../../../../raw/code/EasyDeL/easydel/infra/errors.py#L36)
- doc: General runtime error in EasyDeL operations.
- signature: `class EasyDeLRuntimeError(Exception):`
- used by: [`_import_structure`](../__init__.md#_import_structure)

### `EasyDeLSyntaxRuntimeError`  ·  implements/extends Exception
- def: [`easydel/infra/errors.py:51`](../../../../../../raw/code/EasyDeL/easydel/infra/errors.py#L51)
- doc: Syntax or configuration error in EasyDeL code.
- signature: `class EasyDeLSyntaxRuntimeError(Exception):`
- used by: [`_import_structure`](../__init__.md#_import_structure)

### `EasyDeLTimerError`  ·  implements/extends Exception
- def: [`easydel/infra/errors.py:67`](../../../../../../raw/code/EasyDeL/easydel/infra/errors.py#L67)
- doc: Error related to timing or profiling operations.
- signature: `class EasyDeLTimerError(Exception):`
- used by: [`_import_structure`](../__init__.md#_import_structure), [`_train_epoch`](../trainers/trainer/trainer.md#Trainer._train_epoch), [`_eval_epoch`](../trainers/trainer/trainer.md#Trainer._eval_epoch), [`_prepare_training_output`](../trainers/base_trainer.md#BaseTrainer._prepare_training_output), [`_execute_train_step`](../trainers/trainer/trainer.md#Trainer._execute_train_step), [`apply_training_hooks`](../trainers/base_trainer.md#BaseTrainer.apply_training_hooks), [`_handle_training_interruption`](../trainers/base_trainer.md#BaseTrainer._handle_training_interruption), [`ensure_training_time_limit`](../trainers/training_configurations.md#TrainingArguments.ensure_training_time_limit)

