---
title: easydel/infra/base_state — EasyDeLState, the immutable train/infer state carried through jit
type: concept
provenance: mixed
concept: easydel-infra-base_state
updated: 2026-07-03
status: fresh
---
# easydel/infra/base_state — EasyDeLState, the immutable train/infer state carried through jit

## Overview
[`EasyDeLState`](../catalog/easydel/infra/base_state.md#EasyDeLState) is the single object a training or inference loop threads through JAX transformations. It is a Flax `struct.PyTreeNode` bundling everything stateful — the split model (`graphdef` + `graphstate` + `graphother`), the optimizer (`tx` + `opt_state`), and the `step` counter — so `jit`/`grad`/`pjit` see one clean pytree whose *leaves* are the arrays that need to be traced and whose *static* parts (graphdef, optimizer transform) ride along as non-pytree metadata. The key idea, inherited from Flax's `TrainState` lineage but adapted to NNX's split/merge: keep the model's structural definition (`graphdef`) separate from its parameter values (`graphstate`) so the whole model can be reconstructed on demand via [`model`](../catalog/easydel/infra/base_state.md#EasyDeLState.model) while the params stay a pure pytree the optimizer can update immutably.

## Diagram
```mermaid
flowchart TD
    CREATE["EasyDeLState.create(model, tx)"] --> STATE["EasyDeLState (PyTreeNode)"]
    STATE --> GD["graphdef (static: structure)"]
    STATE --> GS["graphstate (pytree: params)"]
    STATE --> GO["graphother (pytree: non-param state)"]
    STATE --> TX["tx (static: optax transform)"]
    STATE --> OPT["opt_state (pytree: momentum etc.)"]
    STATE --> STEP["step"]
    STATE -->|apply_gradients(grads)| STATE2["new EasyDeLState (step+1)"]
    STATE -->|.model property| MODEL["merge(graphdef, graphstate, graphother) → EasyDeLBaseModule"]
    LOAD["load_state(path)"] --> STATE
```

## Design rationale (why it's built this way)
- **Immutable, JAX-native.** The docstring is explicit: "implemented as a Flax struct PyTreeNode ... compatible with `jit`, `grad`, `vmap`, `pmap`. The state is immutable; all methods that modify state return new instances." A training step is therefore a pure function `state → state`, which is exactly what `jax.jit(train_step)` wants.
- **Three-way model split, not two.** Where a plain train-state keeps params and structure, `EasyDeLState` splits the model into `graphdef` (structure, non-pytree), `graphstate` (trainable `nn.Param`s, pytree), *and* `graphother` (non-param `nn.Variable`s like BN stats / RNG state, pytree). Separating `graphother` from `graphstate` means the optimizer's gradient update touches only true parameters — non-param state threads through untouched but still travels inside the pytree.
- **Optimizer is part of the state but static where it must be.** `tx` (the `optax.GradientTransformation`) is a non-pytree node while `opt_state` (momentum buffers) is a pytree node — so the optimizer *algorithm* specializes the compiled graph while its *buffers* are traced arrays that update each step.
- **eSurge cache scoping via a per-state UUID.** `esurge_cache_scope_key` is auto-generated per state so compiled serving caches are keyed to this exact state instance — preventing a stale compiled cache from a different state leaking across.

## Entry points
- [`EasyDeLState`](../catalog/easydel/infra/base_state.md#EasyDeLState) (`.create(model, tx, init_opt_state=...)`) — builds a state from a model + optimizer; the standard way a trainer obtains its carry.
- [`apply_gradients`](../catalog/easydel/infra/base_state.md#EasyDeLState) — the single optimization step: `state.apply_gradients(grads=...)` returns a new state with updated params, updated `opt_state`, and `step+1`; the hot function inside every `jit`'d train step.
- [`model`](../catalog/easydel/infra/base_state.md#EasyDeLState.model) — reconstructs the full [`EasyDeLBaseModule`](../catalog/easydel/infra/base_module.md#EasyDeLBaseModule) by merging `graphdef`+[`graphstate`](../catalog/easydel/infra/base_state.md#EasyDeLState.graphstate)+`graphother`; used to run inference or reach model methods/config from a state.
- [`load_state`](../catalog/easydel/infra/base_state.md#EasyDeLState.load_state) — reconstructs a state from a checkpoint path (with dtype/sharding options); the counterpart to `save_state`.

## Mechanism (step-by-step)
1. **Create splits the model.** `EasyDeLState.create(model, tx)` runs the NNX split to populate `graphdef`/[`graphstate`](../catalog/easydel/infra/base_state.md#EasyDeLState.graphstate)/`graphother`, stores the optimizer `tx`, optionally initializes `opt_state`, and sets `step=0` — producing an immutable pytree.
2. **Each step applies gradients functionally.** [`apply_gradients`](../catalog/easydel/infra/base_state.md#EasyDeLState) takes a `grads` pytree matching [`graphstate`](../catalog/easydel/infra/base_state.md#EasyDeLState.graphstate)'s structure, runs the optimizer (via a custom `apply_updates_hook` if the transform provides one, else `optax.apply_updates`), and returns a *new* `EasyDeLState` with updated params/opt-state and incremented step. It asserts `tx` and `opt_state` are non-`None` — an inference-only state (no optimizer) can't take a step.
3. **Model reconstructed on demand.** [`model`](../catalog/easydel/infra/base_state.md#EasyDeLState.model) merges the three model parts back into a live [`EasyDeLBaseModule`](../catalog/easydel/infra/base_module.md#EasyDeLBaseModule) (equivalent to `merge(self.graphstate)`), so a state can be turned back into a callable model for eval without keeping a separate model object around.
4. **Checkpoint round-trips through load/save.** [`load_state`](../catalog/easydel/infra/base_state.md#EasyDeLState.load_state) rebuilds the full state from disk (honoring dtype/sharding), and `shard_state` applies device sharding — so a saved state carries params, optimizer buffers, and step together.

## Key data structures
- [`EasyDeLState`](../catalog/easydel/infra/base_state.md#EasyDeLState) fields: `step`, `graphdef` (static), [`graphstate`](../catalog/easydel/infra/base_state.md#EasyDeLState.graphstate) (params pytree), `graphother` (non-param pytree), `tx` (static optimizer), `opt_state` (pytree), `apply_fn` (optional custom forward), `esurge_cache_scope_key`.

## Dynamics (design intent)
> [!inferred] The static-vs-pytree tagging of each field is what determines recompilation behavior: changing the optimizer algorithm (`tx`) or model structure (`graphdef`) re-specializes the compiled `train_step`, while param/opt-buffer/step changes are ordinary traced updates. This is why a trainer can `jit(train_step)` once and reuse it across the whole run.

## Edge cases
- **`apply_gradients` on an inference-only state** raises `AssertionError` (`tx`/`opt_state` is `None`) — states created without an optimizer are read-only.
- **`grads` must match `graphstate` structure exactly** — a mismatched tree (e.g. including non-param leaves) breaks the update.
- **`step` may be a Python int or a JAX array** — device-placed step counters keep the whole state on-device, but code comparing `step` must tolerate both.

## Open questions
> [!inferred] `create`, `merge`, `shard_state`, and the full checkpoint format are large adjacent methods only partially in this packet's subgraph; this page documents the state's shape and the cited create/apply/model/load surface.

## See also
- [easydel/infra/base_module](easydel-infra-base_module.md) — the model this state splits and reconstructs.
- [easydel/trainers/base_trainer](easydel-trainers-base_trainer.md) — the loop that threads `EasyDeLState` through `jit`.
- [easydel/infra/base_config](easydel-infra-base_config.md) — reachable via `state.model.config`.

## Sources
- raw/code/EasyDeL/easydel/infra/base_state.py
