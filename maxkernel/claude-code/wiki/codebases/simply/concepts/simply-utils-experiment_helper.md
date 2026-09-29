---
title: simply.utils.experiment_helper — checkpoint/metric side effects behind one class
type: concept
provenance: mixed
concept: simply-utils-experiment_helper
updated: 2026-07-03
status: fresh
---
# simply.utils.experiment_helper — checkpoint/metric side effects behind one class

## Overview

[`ExperimentHelper`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.ckpt_mngr) is the
one object a training loop holds to do every experiment-lifecycle side effect: deciding *when* to
checkpoint ([`ckpt_save_policy`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.ckpt_save_policy)),
*how many* to keep ([`ckpt_preservation_policy`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.ckpt_preservation_policy)),
building the Orbax [`ckpt_mngr`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.ckpt_mngr)
itself, and writing scalar/text metrics through a
[`BaseMetricWriter`](../catalog/simply/utils/metric_writer.md#BaseMetricWriter). Every one of these
is gated by a single primary-process check
([`should_save_data`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.should_save_data)),
so multi-host training runs never have every host racing to write the same checkpoint or metric
file. A companion [`MetricsAggregator`](../catalog/simply/utils/experiment_helper.md#MetricsAggregator)
handles the separate concern of smoothing noisy per-step scalars over a trailing window before they
hit the metric writer.

## Diagram

```mermaid
flowchart TD
  EH["ExperimentHelper"] --> SSD["should_save_data\n(is_primary_process() and bool(experiment_dir))"]
  SSD -->|gates| CKPT["ckpt_mngr (cached_property)\norbax.CheckpointManager"]
  SSD -->|gates| MW["metric_writer (cached_property)"]
  CKPT --> SAVEPOL["ckpt_save_policy\n(FixedIntervalPolicy + SpecificStepsPolicy)"]
  CKPT --> PRESPOL["ckpt_preservation_policy\n(LatestN + EveryNSteps)"]
  EH --> AGG["metrics_aggregator: MetricsAggregator\n(cached_property)"]
  AGG -->|add(name, value)| DEQUE["per-metric collections.deque\n(bounded by average_last_n_steps)"]
  DEQUE --> GETAGG["get_aggregated_metrics() -> np.mean per metric"]
  EH --> SAVECKPT["save_ckpt(state, step) -> checkpoint_lib.save_checkpoint"]
```

## Design rationale (why it's built this way)

**Checkpoint save/keep policy is composed from Orbax's own policy primitives, not custom logic.**
[`ckpt_save_policy`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.ckpt_save_policy)
combines `ocp.checkpoint_managers.FixedIntervalPolicy` (save every `ckpt_interval` steps, only if
`> 0`) with `ocp.checkpoint_managers.SpecificStepsPolicy(steps=[num_train_steps])` (always save the
final step) via `AnySavePolicy` — the final-step save is unconditional regardless of interval
alignment, guaranteeing a checkpoint exists at training's end even if the interval never lands on
the last step exactly.

**Every filesystem/heavyweight resource is a `cached_property`, deferring construction until first
actual use — mirroring the same pattern in `tokenization.py`'s `HuggingFaceVocab`.**
[`ckpt_mngr`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.ckpt_mngr),
[`metric_writer`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.metric_writer), and
[`metrics_aggregator`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.metrics_aggregator)
are all `functools.cached_property` — an `ExperimentHelper` can be constructed cheaply (e.g. in a
config-validation pass or a non-primary-process worker) without ever touching the filesystem or
network, and only pays that cost when a checkpoint or metric write is actually attempted.

**`ckpt_keep_period` is validated against `ckpt_interval` at checkpoint-manager construction time,
not at config-parse time, because the invariant only makes sense together.**
[`ckpt_mngr`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.ckpt_mngr) raises
`ValueError` if `ckpt_keep_period % ckpt_interval != 0`, with the reasoning spelled out in the error
message: "Otherwise, it does not preserve anything" — a keep-period that doesn't align with the save
interval would specify a policy that never actually retains a checkpoint, since checkpoints only
exist at `ckpt_interval` multiples.

**Scalar metrics are filtered for genuine scalars before ever reaching the metric writer, with the
filtering itself being opt-out.** [`write_scalars`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.write_scalars)'s
`filter_nonscalars=True` default routes every value through
[`convert_to_scalar`](../catalog/simply/utils/experiment_helper.md#is_primary_process) (which uses
`np.asanyarray(x).reshape(())` to force a size-1 array/value into a 0-d array, returning `None` on
failure or on `np.size(x) != 1`), logging a warning and dropping any value that doesn't qualify —
this is what lets callers pass loosely-typed metric dicts (mixing Python floats, 0-d arrays, and
occasionally accidental non-scalar arrays) without every call site pre-validating shapes itself.

> [!inferred] `is_primary_task` is
> `functools.cache`d and reads `os.environ.get('TASK_ID')` — distinct from
> [`is_primary_process`](../catalog/simply/utils/experiment_helper.md#is_primary_process) (which uses
> `jax.process_index()`), suggesting these two "am I the one host that should do side effects"
> checks answer different questions: one about JAX's own multi-process model, one about an external
> job-scheduler's task numbering — likely relevant when multiple JAX processes run under one
> scheduler task, or vice versa.

## Entry points

- [`ExperimentHelper.save_ckpt`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.save_ckpt) —
  called once per training step (internally a no-op unless
  [`ckpt_mngr`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.ckpt_mngr) exists);
  delegates the actual save mechanics to `checkpoint_lib.save_checkpoint`.
- [`ExperimentHelper.add_metric`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.add_metric)/
  [`write_scalars`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.write_scalars) —
  the two metric-reporting paths; `add_metric` feeds the trailing-window aggregator,
  `write_scalars` writes straight through (after scalar filtering) to the metric writer.
- [`ExperimentHelper.save_config_info`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.save_config_info) —
  called once at experiment start to dump the resolved config and (optionally) model parameter
  shapes as both TensorBoard text and JSON files on disk.
- [`ExperimentHelper.close`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.close) —
  called once at experiment end; ensures the checkpoint manager and metric writer both flush/close.

## Mechanism (step-by-step)

1. **Construction is a frozen dataclass, cheap by design.** `__post_init__` only creates
   `experiment_dir` on disk, and only if
   [`should_save_data`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.should_save_data)
   is true — no checkpoint manager or metric writer is touched yet.
2. **On first checkpoint attempt,
   [`ckpt_mngr`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.ckpt_mngr) is built
   lazily.** It composes
   [`ckpt_save_policy`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.ckpt_save_policy)
   and [`ckpt_preservation_policy`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.ckpt_preservation_policy)
   into `ocp.CheckpointManagerOptions` (with a 100-hour async timeout) and constructs the
   `ocp.CheckpointManager` rooted at
   [`ckpt_dir`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.ckpt_dir).
3. **Metric writes filter, then delegate.**
   [`write_scalars`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.write_scalars)
   converts `AnnotatedArray`s to raw arrays via `common.get_raw_arrays`, filters non-scalars, then
   forwards to `self.metric_writer.write_scalars(step, scalars)` if a writer exists (`None` if
   `should_save_data` is false).
4. **The aggregator maintains a bounded deque per metric name, evicting the oldest on overflow.**
   [`MetricsAggregator.add`](../catalog/simply/utils/experiment_helper.md#MetricsAggregator.add)
   rejects non-scalar values outright (`raise ValueError`, stricter than `write_scalars`'s
   warn-and-drop), appends to `self.metrics[name]` (a `collections.deque`), and calls `.popleft()`
   once the deque exceeds
   [`average_last_n_steps`](../catalog/simply/utils/experiment_helper.md#MetricsAggregator.average_last_n_steps).
5. **[`save_config_info`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.save_config_info)
   and `save_state_info` both write the same info twice: once as TensorBoard
   text (via `write_texts`), once as a JSON file on disk** — the former for quick in-UI inspection,
   the latter as a durable, greppable artifact.

## Key data structures

- **[`ExperimentHelper`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.ckpt_mngr)**
  (frozen dataclass) — `experiment_dir`, `ckpt_interval`, `ckpt_max_to_keep`, `ckpt_keep_period`,
  `metric_log_interval`, `num_train_steps`, `should_save_ckpt`.
- **[`MetricsAggregator`](../catalog/simply/utils/experiment_helper.md#MetricsAggregator)** (frozen
  dataclass) — `average_last_n_steps` plus a `metrics: Mapping[str, deque]` cached property.

## Dynamics (design intent)

Because `ckpt_mngr`/`metric_writer`/`metrics_aggregator` are `cached_property`s on a *frozen*
dataclass, their cached values are set via the property machinery's own attribute-caching (not
`object.__setattr__` bypassing frozen-ness elsewhere) — the frozen-ness only prevents external
mutation of the declared dataclass fields, not the lazy-initialized derived properties.

## Edge cases

- [`ExperimentHelper.should_log_metrics`](../catalog/simply/utils/experiment_helper.md#ExperimentHelper.ckpt_mngr)
  always returns true on the *last* training step (`step == num_train_steps - 1`) in addition to the
  regular interval check — final-step metrics are never skipped regardless of interval alignment,
  mirroring the checkpoint save policy's same unconditional-final-step guarantee.
- [`MetricsAggregator.reset`](../catalog/simply/utils/experiment_helper.md#MetricsAggregator) replaces
  `self.metrics` with a plain `collections.defaultdict(collections.deque)` — note this doesn't match
  the `cached_property`'s own declared type (`defaultdict(deque[np.typing.ArrayLike])`), a minor
  looseness worth knowing about if type-checking this path.

## Open questions

- Whether `TASK_ID`-based `is_primary_task`
  is actually consulted anywhere in this packet's subgraph (it's defined but not visibly called by
  any cited symbol here) isn't resolved by this packet alone.

## See also
- [simply-utils-checkpoint_lib](simply-utils-checkpoint_lib.md) — `save_checkpoint`, the function
  `save_ckpt` delegates to.
- [simply-utils-common](simply-utils-common.md) — `PyTree`/`get_raw_arrays`, used to sanitize
  metrics before writing.
