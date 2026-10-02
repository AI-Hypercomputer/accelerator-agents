---
title: axlearn.common.trainer — SpmdTrainer, the compiled train-step loop
type: concept
provenance: mixed
concept: axlearn-common-trainer
updated: 2026-07-03
status: fresh
---
# axlearn.common.trainer — SpmdTrainer, the compiled train-step loop

## Overview

[`SpmdTrainer.run`](../catalog/axlearn/common/trainer.md#SpmdTrainer.run) ("Runs training") is
AXLearn's top-level training loop. Each step ultimately calls
[`_train_step`](../catalog/axlearn/common/trainer.md#SpmdTrainer._train_step), whose compiled form is
built once by
[`_get_compiled_train_step_fn`](../catalog/axlearn/common/trainer.md#SpmdTrainer._get_compiled_train_step_fn)
("Build a fully compiled train step function"). Because every AXLearn layer is a stateless `Module`
config-tree (see [axlearn-common-attention](axlearn-common-attention.md)), invoking the model's
forward/loss computation from inside the compiled step goes through
[`functional`](../catalog/axlearn/common/module.md#functional) ("Invokes `<module>.<method>` in a pure
functional fashion") — this is the seam between AXLearn's object-oriented layer API and JAX's
purely-functional transform requirements.

## Diagram

```mermaid
flowchart TD
  RUN["SpmdTrainer.run(prng_key)"] -->|per step| INPUT["_input_iter -> input_batch"]
  INPUT --> STEPFN["_get_compiled_train_step_fn\n(built once, reused every step)"]
  STEPFN -->|compile_train_step| COMPILED["compiled jax function"]
  RUN --> TRAINSTEP["_train_step(state, input_batch)"]
  TRAINSTEP -->|functional(module, prng_key, state, inputs, method='forward')| FUNC["pure-functional Module invocation\n(loss + gradients via optimizer)"]
  FUNC --> NEWSTATE["updated TrainerState"]
  INIT["_init_with_prebuilt_state(prebuilt_state)"] -.->|optional, e.g. warm-start| RUN
```

## Design rationale (why it's built this way)

**The compiled train-step function is built once, ahead of the training loop, not recompiled every
step — `_get_compiled_train_step_fn`'s own doc, "Build a fully compiled train step function," and its
signature taking a representative `trainer_state`/`input_batch` (for shape inference) rather than the
loop's actual per-step values, confirms this is a one-time compilation keyed on shape/dtype, reused
across every step with matching shapes.**

**`functional` is the one bridge between AXLearn's stateful-looking `Module` API and JAX's
purely-functional transform model — every module invocation from inside a compiled step goes through
it, threading `state`/`prng_key` explicitly rather than relying on captured Python state.** Its doc —
"Invokes `<module>.<method>` in a pure functional fashion" — and its explicit `state: NestedTensor`
and `prng_key: Optional[Tensor]` parameters make this the same "no hidden module state" pattern also
seen in the AttentionMask/config-tree design elsewhere in AXLearn — the actual parameters/state are
threaded as explicit function arguments, never captured closures.

**`_init_with_prebuilt_state` supports initializing from an externally-provided
`TrainerStateBuilder.State`, not only from a config-driven fresh init or an internal checkpoint
restore.** Its own doc — "Initializes `self._step` and `self._trainer_state`, optionally from
`prebuilt_state`" — is the seam that lets a caller inject a warm-started/converted state (e.g. from a
different checkpoint format or a partial fine-tune) without going through the trainer's own default
initialization path.

## Entry points

- [`SpmdTrainer.run`](../catalog/axlearn/common/trainer.md#SpmdTrainer.run) — the top-level training
  entry point.
- [`SpmdTrainer._get_compiled_train_step_fn`](../catalog/axlearn/common/trainer.md#SpmdTrainer._get_compiled_train_step_fn) —
  called once to build the reused compiled step function.
- [`functional`](../catalog/axlearn/common/module.md#functional) — called anywhere a `Module` needs to
  be invoked inside JAX-transformed (compiled/traced) code, not just from the trainer.

## Mechanism (step-by-step)

1. **[`run`](../catalog/axlearn/common/trainer.md#SpmdTrainer.run) prepares training**
   (`_prepare_training`), then iterates `_input_iter` to produce
   `input_batch`es, converting host arrays to global (sharded) arrays via `host_to_global_array`.
2. **The first call triggers
   [`_get_compiled_train_step_fn`](../catalog/axlearn/common/trainer.md#SpmdTrainer._get_compiled_train_step_fn)**,
   which builds a compiled function wrapping
   [`_train_step`](../catalog/axlearn/common/trainer.md#SpmdTrainer._train_step) via
   `compile_train_step`, keyed on the `trainer_state`/`input_batch` shapes.
3. **Each step calls the compiled
   [`_train_step`](../catalog/axlearn/common/trainer.md#SpmdTrainer._train_step) function with the
   current `TrainerState` and the new
   `input_batch`.** Internally, it computes `_opt_params`, dispatches the global batch, and
   calls [`functional`](../catalog/axlearn/common/module.md#functional) to invoke the model's forward
   method purely functionally against the current parameter state.
4. **The returned updated `TrainerState` and summaries are logged (`_step_log`, inside
   [`run`](../catalog/axlearn/common/trainer.md#SpmdTrainer.run)) and the loop
   continues** until the configured step count is reached.

## Key data structures

- **`SpmdTrainer`** — the trainer itself; holds
  [`config`](../catalog/axlearn/common/config.md#Configurable.config) (a `Config`) and internal
  `_trainer_state`/`_step`.
- **`TrainerState`** — the model parameters + optimizer state pytree threaded through every step.
- **`Nested`/`NestedTensor`** (from `axlearn.common.utils`) — the recursive dict-of-tensors container
  every batch/state/output uses.

## Dynamics (design intent)

`_get_compiled_train_step_fn`'s `with_xsc` parameter (defaulting `False`) suggests an optional XLA
step-check/SPMD-check debugging mode can be toggled without changing the trained step logic itself —
consistent with keeping the hot path (default `with_xsc=False`) free of debug-only overhead.

## Edge cases
None directly visible in this packet's subgraph beyond the prebuilt-state initialization path.

## Open questions
- The precise shape-inference contract for `_get_compiled_train_step_fn` (whether a shape/dtype
  mismatch on a later step re-triggers compilation or raises) isn't resolved by the symbols in this
  packet's subgraph.

## See also
- [axlearn-common-attention](axlearn-common-attention.md) — `MultiheadAttention`, one of the `Module`
  layers `functional` invokes during a train step.
