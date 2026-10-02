---
title: 'Module: easydel/trainers/metrics.py'
type: catalog
provenance: extracted
module: easydel/trainers/metrics.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.trainers.metrics`/
symbols:
  StepMetrics.calculate: StepMetrics#calculate().
  BaseProgressBar: BaseProgressBar#
  run_watchers: run_watchers().
  StepMetrics._calculate_detailed_metrics: StepMetrics#_calculate_detailed_metrics().
  StepMetrics: StepMetrics#
  MetricsTracker.reset: MetricsTracker#reset().
  MetricsTracker: MetricsTracker#
  MetricsTracker.update: MetricsTracker#update().
  BaseProgressBar.update: BaseProgressBar#update().
  BaseProgressBar.close: BaseProgressBar#close().
  BaseProgressBar.set_postfix: BaseProgressBar#set_postfix().
  BaseProgressBar.reset: BaseProgressBar#reset().
  compute_weight_stats: compute_weight_stats().
  MetricsHistogram.variance: MetricsHistogram#variance().
  RichProgressBar.set_postfix: RichProgressBar#set_postfix().
  RichProgressBar.reset: RichProgressBar#reset().
  MetricsHistogram.mean: MetricsHistogram#mean().
  LogWatcher: LogWatcher#
  StepMetrics.accumulate_summary_metric: StepMetrics#accumulate_summary_metric().
  TqdmProgressBar.pbar: TqdmProgressBar#pbar.
  MetricsHistogram.from_array: MetricsHistogram#from_array().
  MetricsTracker.loss_sum: MetricsTracker#loss_sum.
  MetricsTracker.accuracy_sum: MetricsTracker#accuracy_sum.
  RichProgressBar.update: RichProgressBar#update().
  RichProgressBar.close: RichProgressBar#close().
  MetricsHistogram: MetricsHistogram#
  MetricsHistogram.numpy_histogram: MetricsHistogram#numpy_histogram().
  StepMetrics.start_step: StepMetrics#start_step().
  NullProgressBar: NullProgressBar#
  TqdmProgressBar: TqdmProgressBar#
  JSONProgressBar: JSONProgressBar#
  RichProgressBar: RichProgressBar#
  RichProgressBar.progress: RichProgressBar#progress.
  RichProgressBar.task_id: RichProgressBar#task_id.
  StepMetrics.summarize_metrics: StepMetrics#summarize_metrics().
  MetricsHistogram.std: MetricsHistogram#std().
  MetricsTracker.loss_count: MetricsTracker#loss_count.
  MetricsTracker.accuracy_count: MetricsTracker#accuracy_count.
  RichProgressBar._postfix: RichProgressBar#_postfix.
  MetricsColumn.render: MetricsColumn#render().
  TqdmProgressBar.update: TqdmProgressBar#update().
  TqdmProgressBar.set_postfix: TqdmProgressBar#set_postfix().
  TqdmProgressBar.reset: TqdmProgressBar#reset().
  TqdmProgressBar.close: TqdmProgressBar#close().
  JSONProgressBar.set_postfix: JSONProgressBar#set_postfix().
  StepMetrics.arguments: StepMetrics#arguments.
  StepMetrics.step_start_time: StepMetrics#step_start_time.
  StepMetrics._get_summary_metric_reduction: StepMetrics#_get_summary_metric_reduction().
  MetricsColumn.metrics_to_show: MetricsColumn#metrics_to_show.
  MetricsHistogram.bin_counts: MetricsHistogram#bin_counts.
  MetricsHistogram.bin_edges: MetricsHistogram#bin_edges.
  MetricsHistogram.size: MetricsHistogram#size.
  LogWatcher.interval: LogWatcher#interval.
  logger: logger.
  StepMetrics.start_time: StepMetrics#start_time.
  StepMetrics._coerce_summary_scalar: StepMetrics#_coerce_summary_scalar().
  MetricsTracker.step_offset: MetricsTracker#step_offset.
  MetricsHistogram.min: MetricsHistogram#min.
  MetricsHistogram.max: MetricsHistogram#max.
  MetricsHistogram.sum: MetricsHistogram#sum.
  MetricsHistogram.sum_squares: MetricsHistogram#sum_squares.
  MetricsHistogram._create_histogram_bin_edges: MetricsHistogram#_create_histogram_bin_edges().
  LogWatcher.name: LogWatcher#name.
  LogWatcher.fn: LogWatcher#fn.
  LogWatcher.pattern: LogWatcher#pattern.
  StepMetrics.__init__: StepMetrics#__init__().
  MetricsTracker.__init__: MetricsTracker#__init__().
  MetricsTracker.metrics_history: MetricsTracker#metrics_history.
  MetricsColumn: MetricsColumn#
  MetricsColumn.__init__: MetricsColumn#__init__().
  NullProgressBar.update: NullProgressBar#update().
  NullProgressBar.set_postfix: NullProgressBar#set_postfix().
  NullProgressBar.reset: NullProgressBar#reset().
  NullProgressBar.close: NullProgressBar#close().
  TqdmProgressBar.__init__: TqdmProgressBar#__init__().
  JSONProgressBar.__init__: JSONProgressBar#__init__().
  JSONProgressBar.desc: JSONProgressBar#desc.
  JSONProgressBar.update: JSONProgressBar#update().
  JSONProgressBar.reset: JSONProgressBar#reset().
  JSONProgressBar.close: JSONProgressBar#close().
  RichProgressBar.__init__: RichProgressBar#__init__().
---
# Module: [`easydel/trainers/metrics.py`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py)

## Classes
### `BaseProgressBar`  ·  implements/extends ABC
- def: [`easydel/trainers/metrics.py:469`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L469)
- doc: Abstract base class for progress bar implementations.
- signature: `class BaseProgressBar(abc.ABC):`
- members:
  - `close(self)` — [`L500`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L500) — Close and cleanup the progress bar.
  - `reset(self)` — [`L495`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L495) — Reset the progress bar to initial state.
  - `set_postfix(self, **kwargs)` — [`L486`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L486) — Set postfix metrics to display.
  - `update(self, n: int = 1)` — [`L477`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L477) — Update the progress bar.
- uses (calls/refs, reference-scoped): [`reset`](metrics.md#RichProgressBar.reset), [`set_postfix`](metrics.md#RichProgressBar.set_postfix), [`close`](metrics.md#RichProgressBar.close), [`update`](metrics.md#RichProgressBar.update), [`JSONProgressBar`](metrics.md#JSONProgressBar), [`NullProgressBar`](metrics.md#NullProgressBar), [`RichProgressBar`](metrics.md#RichProgressBar), [`TqdmProgressBar`](metrics.md#TqdmProgressBar), [`close`](metrics.md#TqdmProgressBar.close), [`reset`](metrics.md#TqdmProgressBar.reset), [`set_postfix`](metrics.md#JSONProgressBar.set_postfix), [`set_postfix`](metrics.md#TqdmProgressBar.set_postfix), [`update`](metrics.md#TqdmProgressBar.update), [`close`](metrics.md#JSONProgressBar.close), [`close`](metrics.md#NullProgressBar.close), [`reset`](metrics.md#JSONProgressBar.reset), [`reset`](metrics.md#NullProgressBar.reset), [`set_postfix`](metrics.md#NullProgressBar.set_postfix), [`update`](metrics.md#JSONProgressBar.update), [`update`](metrics.md#NullProgressBar.update)
- used by: [`_train_epoch`](trainer/trainer.md#Trainer._train_epoch), [`_eval_epoch`](trainer/trainer.md#Trainer._eval_epoch), [`_run_training_loop`](trainer/trainer.md#Trainer._run_training_loop), [`_run_evaluation`](trainer/trainer.md#Trainer._run_evaluation), [`log_metrics`](base_trainer.md#BaseTrainer.log_metrics), [`create_progress_bar`](base_trainer.md#BaseTrainer.create_progress_bar), [`_eval_epoch`](trainer_protocol.md#BaseTrainerProtocol._eval_epoch), [`_train_epoch`](trainer_protocol.md#BaseTrainerProtocol._train_epoch), [`create_progress_bar`](trainer_protocol.md#BaseTrainerProtocol.create_progress_bar), [`log_metrics`](trainer_protocol.md#BaseTrainerProtocol.log_metrics), [`JSONProgressBar`](metrics.md#JSONProgressBar), [`NullProgressBar`](metrics.md#NullProgressBar), [`RichProgressBar`](metrics.md#RichProgressBar), [`TqdmProgressBar`](metrics.md#TqdmProgressBar)

### `JSONProgressBar`  ·  implements/extends BaseProgressBar
- def: [`easydel/trainers/metrics.py:560`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L560)
- doc: JSON-based progress reporting.
- signature: `class JSONProgressBar(BaseProgressBar):`
- members:
  - `__init__(self, desc="")` — [`L570`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L570) — Initialize JSON progress reporter.
  - `close(self)` — [`L591`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L591)
  - `reset(self)` — [`L589`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L589)
  - `set_postfix(self, **kwargs)` — [`L580`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L580)
  - `update(self, n: int = 1)` — [`L578`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L578)
  - `desc` — [`L576`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L576)
- uses (calls/refs, reference-scoped): [`BaseProgressBar`](metrics.md#BaseProgressBar), [`logger`](metrics.md#logger)
- used by: [`BaseProgressBar`](metrics.md#BaseProgressBar), [`close`](metrics.md#BaseProgressBar.close), [`update`](metrics.md#BaseProgressBar.update), [`reset`](metrics.md#BaseProgressBar.reset), [`set_postfix`](metrics.md#BaseProgressBar.set_postfix), [`_hidden_rich_pbar`](base_trainer.md#BaseTrainer._hidden_rich_pbar)

### `LogWatcher`
- def: [`easydel/trainers/metrics.py:776`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L776)
- doc: A user-defined per-parameter metric that is evaluated at a fixed interval.
- signature: `class LogWatcher:`
- members:
  - `fn` — [`L806`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L806)
  - `interval` — [`L807`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L807)
  - `name` — [`L805`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L805)
  - `pattern` — [`L808`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L808)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`run_watchers`](metrics.md#run_watchers), [`set_watchers`](../infra/elarge/model.md#eLargeModel.set_watchers), [`watchers`](training_configurations.md#TrainingArguments.watchers)

### `MetricsColumn`  ·  implements/extends ProgressColumn
- def: [`easydel/trainers/metrics.py:404`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L404)
- doc: A custom Rich progress column for displaying metrics.
- signature: `class MetricsColumn(ProgressColumn):`
- members:
  - `__init__(self, metrics_to_show=None)` — [`L415`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L415) — Initialize the metrics column.
  - `render(self, task: Task)` — [`L424`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L424) — Render the metrics in an organized way.
  - `metrics_to_show` — [`L422`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L422)

### `MetricsHistogram`
- def: [`easydel/trainers/metrics.py:636`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L636)
- doc: Compute and store histogram data for model weights or activations.
- signature: `class MetricsHistogram:`
- members:
  - `_create_histogram_bin_edges(arr: jax.Array)` — [`L656`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L656) — Create histogram bins and counts.
  - `from_array(cls, arr: jax.Array)` — [`L679`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L679) — Create a histogram from an array.
  - `mean(self)` — [`L709`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L709) — Calculate mean of the original array.
  - `numpy_histogram(self)` — [`L700`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L700) — Return histogram data in numpy-compatible format.
  - `std(self)` — [`L730`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L730) — Calculate standard deviation of the original array.
  - `variance(self)` — [`L718`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L718) — Calculate variance of the original array.
  - `bin_counts` — [`L643`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L643)
  - `bin_edges` — [`L644`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L644)
  - `max` — [`L649`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L649)
  - `min` — [`L648`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L648)
  - `size` — [`L646`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L646)
  - `sum` — [`L650`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L650)
  - `sum_squares` — [`L652`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L652)
- used by: [`log_weight_distribution`](training_configurations.md#TrainingArguments.log_weight_distribution), [`compute_weight_stats`](metrics.md#compute_weight_stats)

### `MetricsTracker`
- def: [`easydel/trainers/metrics.py:334`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L334)
- doc: Tracks and aggregates training metrics over time.
- signature: `class MetricsTracker:`
- members:
  - `__init__(self)` — [`L347`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L347) — Initialize the metrics tracker with empty state.
  - `reset(self, step)` — [`L388`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L388) — Reset tracked metrics.
  - `update(self, loss, accuracy, step)` — [`L356`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L356) — Update tracked metrics with new values.
  - `accuracy_count` — [`L352`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L352)
  - `accuracy_sum` — [`L351`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L351)
  - `loss_count` — [`L350`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L350)
  - `loss_sum` — [`L349`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L349)
  - `metrics_history` — [`L353`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L353)
  - `step_offset` — [`L354`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L354)
- used by: [`_train_epoch`](trainer/trainer.md#Trainer._train_epoch), [`_eval_epoch`](trainer/trainer.md#Trainer._eval_epoch), [`_run_training_loop`](trainer/trainer.md#Trainer._run_training_loop), [`_run_evaluation`](trainer/trainer.md#Trainer._run_evaluation), [`train`](trainer/trainer.md#Trainer.train), [`eval`](trainer/trainer.md#Trainer.eval), [`_eval_epoch`](trainer_protocol.md#BaseTrainerProtocol._eval_epoch), [`_train_epoch`](trainer_protocol.md#BaseTrainerProtocol._train_epoch), [`_run_evaluation`](trainer_protocol.md#BaseTrainerProtocol._run_evaluation), [`_run_training_loop`](trainer_protocol.md#BaseTrainerProtocol._run_training_loop)

### `NullProgressBar`  ·  implements/extends BaseProgressBar
- def: [`easydel/trainers/metrics.py:505`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L505)
- doc: Dummy progress bar that does nothing.
- signature: `class NullProgressBar(BaseProgressBar):`
- members:
  - `close(self)` — [`L521`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L521)
  - `reset(self)` — [`L518`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L518)
  - `set_postfix(self, **kwargs)` — [`L515`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L515)
  - `update(self, n: int = 1)` — [`L512`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L512)
- uses (calls/refs, reference-scoped): [`BaseProgressBar`](metrics.md#BaseProgressBar)
- used by: [`BaseProgressBar`](metrics.md#BaseProgressBar), [`create_progress_bar`](base_trainer.md#BaseTrainer.create_progress_bar), [`close`](metrics.md#BaseProgressBar.close), [`update`](metrics.md#BaseProgressBar.update), [`reset`](metrics.md#BaseProgressBar.reset), [`set_postfix`](metrics.md#BaseProgressBar.set_postfix)

### `RichProgressBar`  ·  implements/extends BaseProgressBar
- def: [`easydel/trainers/metrics.py:594`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L594)
- doc: Wrapper for Rich library progress bar.
- signature: `class RichProgressBar(BaseProgressBar):`
- members:
  - `__init__(self, progress: Progress, task_id: TaskID)` — [`L606`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L606) — Initialize RichProgressBar with an existing Progress instance.
  - `close(self)` — [`L628`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L628)
  - `reset(self)` — [`L624`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L624)
  - `set_postfix(self, **kwargs)` — [`L620`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L620)
  - `update(self, n: int = 1)` — [`L617`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L617)
  - `progress` — [`L613`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L613)
  - `task_id` — [`L614`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L614)
- protocol/private: `_postfix`[`L615`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L615)
- uses (calls/refs, reference-scoped): [`BaseProgressBar`](metrics.md#BaseProgressBar)
- used by: [`BaseProgressBar`](metrics.md#BaseProgressBar), [`close`](metrics.md#BaseProgressBar.close), [`update`](metrics.md#BaseProgressBar.update), [`reset`](metrics.md#BaseProgressBar.reset), [`set_postfix`](metrics.md#BaseProgressBar.set_postfix), [`_hidden_rich_pbar`](base_trainer.md#BaseTrainer._hidden_rich_pbar)

### `StepMetrics`
- def: [`easydel/trainers/metrics.py:58`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L58)
- doc: Handles calculation and tracking of training metrics.
- signature: `class StepMetrics:`
- members:
  - `__init__(self, arguments)` — [`L75`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L75) — Initialize the metrics calculator.
  - `_calculate_detailed_metrics(self, metrics: LossMetrics)` — [`L297`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L297) — Calculate additional detailed metrics.
  - `accumulate_summary_metric(self, summary_metric_sums: dict[str, float], summary_metric_counts: dict[str, int], metric_name: str, metric_value: tp.Any, mode: tp.Literal["eval", "train"] | None = None)` — [`L247`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L247)
  - `calculate(self, metrics: LossMetrics, current_step: int, epoch: int, epoch_progress: float | None, flops_per_token: float, extra_flops_per_token: float, batch_size: int, seq_length: int, learning_rate: float, mode: tp.Literal["eval", "train"] | None = None, **extras)` — [`L93`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L93) — Calculate comprehensive metrics for the training step.
  - `start_step(self)` — [`L85`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L85) — Mark the start of a training step.
  - `summarize_metrics(self, last_metrics: dict[str, tp.Any], summary_metric_sums: dict[str, float], summary_metric_counts: dict[str, int], mode: tp.Literal["eval", "train"] | None = None)` — [`L263`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L263)
  - `arguments` — [`L81`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L81)
  - `start_time` — [`L82`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L82)
  - `step_start_time` — [`L83`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L83)
- protocol/private: `_coerce_summary_scalar`[`L205`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L205), `_get_summary_metric_reduction`[`L223`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L223)
- uses (calls/refs, reference-scoped): [`LossMetrics`](../infra/loss_utils.md#LossMetrics), [`flatten_dict`](../utils/traversals.md#flatten_dict), [`loss`](../infra/loss_utils.md#LossMetrics.loss), [`accuracy`](../infra/loss_utils.md#LossMetrics.accuracy), [`other_metrics`](../infra/loss_utils.md#LossMetrics.other_metrics), [`z_loss`](../infra/loss_utils.md#LossMetrics.z_loss), [`execution_time`](../infra/loss_utils.md#LossMetrics.execution_time), [`grad_norms`](../infra/loss_utils.md#LossMetrics.grad_norms), [`max_grad_norm`](../infra/loss_utils.md#LossMetrics.max_grad_norm), [`mean_grad_norm`](../infra/loss_utils.md#LossMetrics.mean_grad_norm), [`chosen_rewards`](../infra/loss_utils.md#LossMetrics.chosen_rewards), [`rejected_rewards`](../infra/loss_utils.md#LossMetrics.rejected_rewards)
- used by: [`_train_epoch`](trainer/trainer.md#Trainer._train_epoch), [`_eval_epoch`](trainer/trainer.md#Trainer._eval_epoch), [`_run_training_loop`](trainer/trainer.md#Trainer._run_training_loop), [`_run_evaluation`](trainer/trainer.md#Trainer._run_evaluation), [`train`](trainer/trainer.md#Trainer.train), [`eval`](trainer/trainer.md#Trainer.eval), [`_eval_epoch`](trainer_protocol.md#BaseTrainerProtocol._eval_epoch), [`_train_epoch`](trainer_protocol.md#BaseTrainerProtocol._train_epoch), [`_run_evaluation`](trainer_protocol.md#BaseTrainerProtocol._run_evaluation), [`_run_training_loop`](trainer_protocol.md#BaseTrainerProtocol._run_training_loop)

### `TqdmProgressBar`  ·  implements/extends BaseProgressBar
- def: [`easydel/trainers/metrics.py:525`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L525)
- doc: Wrapper for tqdm progress bar.
- signature: `class TqdmProgressBar(BaseProgressBar):`
- members:
  - `__init__(self, pbar: tqdm)` — [`L534`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L534) — Initialize with an existing tqdm progress bar.
  - `close(self)` — [`L556`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L556)
  - `reset(self)` — [`L552`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L552)
  - `set_postfix(self, **kwargs)` — [`L545`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L545)
  - `update(self, n: int = 1)` — [`L542`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L542)
  - `pbar` — [`L540`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L540)
- uses (calls/refs, reference-scoped): [`BaseProgressBar`](metrics.md#BaseProgressBar)
- used by: [`BaseProgressBar`](metrics.md#BaseProgressBar), [`create_progress_bar`](base_trainer.md#BaseTrainer.create_progress_bar), [`close`](metrics.md#BaseProgressBar.close), [`update`](metrics.md#BaseProgressBar.update), [`reset`](metrics.md#BaseProgressBar.reset), [`set_postfix`](metrics.md#BaseProgressBar.set_postfix)

## Functions
- `compute_weight_stats(params: dict[str, tp.Any], repattern: str)` — [`L740`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L740) — Compute statistics for model weights in a JIT-compatible way.
- `run_watchers(watchers: list[LogWatcher], params: dict[str, tp.Any], step: int)` — [`L811`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L811) — Execute all active watchers against model parameters and collect metrics.

## Module values
- `logger` — [`L55`](../../../../../../raw/code/EasyDeL/easydel/trainers/metrics.py#L55)

