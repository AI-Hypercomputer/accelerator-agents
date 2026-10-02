---
title: 'Module: src/maxtext/common/profiler.py'
type: catalog
provenance: extracted
module: src/maxtext/common/profiler.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.common.profiler`/
symbols:
  Profiler.activate: Profiler#activate().
  Profiler.deactivate: Profiler#deactivate().
  Profiler.mode: Profiler#mode.
  Profiler.maybe_activate_profiler: Profiler#maybe_activate_profiler().
  Profiler.maybe_deactivate_profiler: Profiler#maybe_deactivate_profiler().
  Profiler.profiling_options: Profiler#profiling_options.
  Profiler.finished_initial_profile_step: Profiler#finished_initial_profile_step.
  Profiler.start_initial_profile_step: Profiler#start_initial_profile_step.
  Profiler.should_activate_periodic_profile: Profiler#should_activate_periodic_profile().
  Profiler.should_deactivate_periodic_profile: Profiler#should_deactivate_periodic_profile().
  Profiler.profile_period: Profiler#profile_period.
  Profiler.prof: Profiler#prof.
  Profiler.libcudart: Profiler#libcudart.
  Profiler._set_last_profiler_step: Profiler#_set_last_profiler_step().
  Profiler.output_path: Profiler#output_path.
  Profiler.upload_all_profiler_results: Profiler#upload_all_profiler_results.
  Profiler.managed_mldiagnostics: Profiler#managed_mldiagnostics.
  Profiler.profile_cleanly: Profiler#profile_cleanly.
  Profiler.base_output_dir: Profiler#base_output_dir.
  Profiler._set_first_profiler_step: Profiler#_set_first_profiler_step().
  mldiag: mldiag.
  Profiler: Profiler#
  Profiler.__init__: Profiler#__init__().
  Profiler.post_process: Profiler#post_process().
---
# Module: [`src/maxtext/common/profiler.py`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py)

## Classes
### `Profiler`
- def: [`src/maxtext/common/profiler.py:32`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L32)
- doc: Activate/deactivate a profiler based on the 'profiler' config.
- signature: `class Profiler:`
- members:
  - `activate(self, blocking_object=None, optional_postfix="")` — [`L88`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L88) — Start the profiler.
  - `deactivate(self, blocking_object=None)` — [`L126`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L126) — End the profiler.
  - `maybe_activate_profiler(self, step, state)` — [`L78`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L78) — Conditionally activates the profiler based on the current step.
  - `maybe_deactivate_profiler(self, step, state)` — [`L117`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L117) — Conditionally deactivates the profiler based on the current step.
  - `post_process(self)` — [`L166`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L166)
  - `should_activate_periodic_profile(self, step)` — [`L160`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L160)
  - `should_deactivate_periodic_profile(self, step)` — [`L163`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L163)
  - `base_output_dir` — [`L39`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L39)
  - `finished_initial_profile_step` — [`L45`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L45)
  - `libcudart` — [`L36`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L36)
  - `managed_mldiagnostics` — [`L49`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L49)
  - `mode` — [`L37`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L37)
  - `output_path` — [`L40`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L40)
  - `prof` — [`L48`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L48)
  - `profile_cleanly` — [`L42`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L42)
  - `profile_period` — [`L43`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L43)
  - `profiling_options` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L53)
  - `start_initial_profile_step` — [`L44`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L44)
  - `upload_all_profiler_results` — [`L41`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L41)
- protocol/private: `__init__`[`L35`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L35), `_set_first_profiler_step`[`L154`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L154), `_set_last_profiler_step`[`L157`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L157)
- uses (calls/refs, reference-scoped): [`mldiag`](profiler.md#mldiag)

## Module values
- `mldiag` — [`L26`](../../../../../../../raw/code/maxtext/src/maxtext/common/profiler.py#L26)

