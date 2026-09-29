---
title: 'Module: easydel/utils/helpers.py'
type: catalog
provenance: extracted
module: easydel/utils/helpers.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.utils.helpers`/
symbols:
  capture_time: capture_time().
  check_bool_flag: check_bool_flag().
  Timer.stop: Timer#stop().
  Timer.elapsed_time: Timer#elapsed_time().
  is_remote_path: is_remote_path().
  Timers.log: Timers#log().
  Timer.start: Timer#start().
  Timer.reset: Timer#reset().
  Timers.write: Timers#write().
  Timer.started: Timer#started.
  Timers: Timers#
  Timers.timers: Timers#timers.
  Timers.__call__: Timers#__call__().
  get_cache_dir: get_cache_dir().
  Timers._print_log: Timers#_print_log().
  Timer.elapsed: Timer#elapsed.
  Timer.start_time: Timer#start_time.
  Timer.__enter__: Timer#__enter__().
  Timer.__exit__: Timer#__exit__().
  Timers.timed: Timers#timed().
  quiet: quiet().
  Timer: Timer#
  Timer.name: Timer#name.
  Timers.use_wandb: Timers#use_wandb.
  Timers.tensorboard_writer: Timers#tensorboard_writer.
  DummyStream: DummyStream#
  logger: logger.
  capture_time.get_elapsed: capture_time().get_elapsed().
  Timer.__init__: Timer#__init__().
  Timers.__init__: Timers#__init__().
  DummyStream.write: DummyStream#write().
  DummyStream.flush: DummyStream#flush().
---
# Module: [`easydel/utils/helpers.py`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py)

## Classes
### `DummyStream`
- def: [`easydel/utils/helpers.py:380`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L380)
- doc: A null device-like stream that discards all writes.
- signature: `class DummyStream:`
- members:
  - `flush(self, *args, **kwargs)` — [`L391`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L391) — Discard all flush operations.
  - `write(self, *args, **kwargs)` — [`L387`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L387) — Discard all write operations.
- used by: [`quiet`](helpers.md#quiet)

### `Timer`
- def: [`easydel/utils/helpers.py:126`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L126)
- doc: Simple timer for measuring execution time.
- signature: `class Timer:`
- members:
  - `__init__(self, name)` — [`L150`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L150) — Initialize Timer.
  - `elapsed_time(self, reset=True)` — [`L189`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L189) — Get total elapsed time.
  - `reset(self)` — [`L183`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L183) — Reset the timer, clearing all accumulated elapsed time.
  - `start(self)` — [`L161`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L161) — Start the timer.
  - `stop(self)` — [`L172`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L172) — Stop the timer and accumulate elapsed time.
  - `elapsed` — [`L157`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L157)
  - `name` — [`L156`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L156)
  - `start_time` — [`L159`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L159)
  - `started` — [`L158`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L158)
- protocol/private: `__enter__`[`L205`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L205), `__exit__`[`L209`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L209)
- used by: [`__call__`](helpers.md#Timers.__call__)

### `Timers`
- def: [`easydel/utils/helpers.py:213`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L213)
- doc: Manager for multiple named timers with logging support.
- signature: `class Timers:`
- members:
  - `__call__(self, name)` — [`L242`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L242) — Get or create a timer by name.
  - `__init__(self, use_wandb, tensorboard_writer: SummaryWriter)` — [`L231`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L231) — Initialize Timers.
  - `log(self, names, normalizer=1, reset=True)` — [`L289`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L289) — Log timer values to the console with color-coded formatting.
  - `timed(self, name, log=True, reset=True)` — [`L329`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L329) — Context manager that times a block and optionally logs the result.
  - `write(self, names, iteration, normalizer=1, reset=False)` — [`L255`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L255) — Write timer values to configured logging backends.
  - `tensorboard_writer` — [`L240`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L240)
  - `timers` — [`L238`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L238)
  - `use_wandb` — [`L239`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L239)
- protocol/private: `_print_log`[`L312`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L312)
- uses (calls/refs, reference-scoped): [`Timer`](helpers.md#Timer), [`logger`](helpers.md#logger)
- used by: [`_configure_state`](../trainers/base_trainer.md#BaseTrainer._configure_state), [`_configure_dataloaders`](../trainers/base_trainer.md#BaseTrainer._configure_dataloaders), [`_configure_model`](../trainers/base_trainer.md#BaseTrainer._configure_model), [`timer`](../trainers/trainer_protocol.md#BaseTrainerProtocol.timer), [`checkpointer`](../trainers/base_trainer.md#BaseTrainer.checkpointer), [`_initialize_timer`](../trainers/base_trainer.md#BaseTrainer._initialize_timer)

## Functions
- `capture_time()` — [`L94`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L94) — Context manager that measures elapsed time.
- `check_bool_flag(name: str, default: bool = True)` — [`L437`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L437) — Parse boolean environment variable.
- `get_cache_dir()` — [`L351`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L351) — Get the EasyDeL cache directory.
- `get_elapsed()` — [`L114`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L114) — Get elapsed time in seconds.
- `is_remote_path(path: os.PathLike[str] | str | object)` — [`L76`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L76) — Return whether a path targets a remote/object-store backend.
- `quiet(suppress_stdout=True, suppress_stderr=True)` — [`L397`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L397) — Context manager to temporarily suppress stdout and/or stderr output.

## Module values
- `logger` — [`L73`](../../../../../../raw/code/EasyDeL/easydel/utils/helpers.py#L73)

