---
id: STRAT-SC-04
title: An ignored keyword argument is a silent no-op, and emit_pipeline(scratches=) is one
applies_to: any
lever: pipelining
generality: general
phase: [implement]
verify: "manual: pass a scratch via emit_pipeline(scratches=) and check it is wired"
---

# STRAT-SC-04 — `emit_pipeline(..., scratches=)` may be silently dropped

**Rule.** Toolchain-dependent; verify before relying. Pass scratch buffers into an `emit_pipeline`
body by **closure capture**, not via `scratches=`, unless you have confirmed the kwarg is honoured on
your toolchain. More generally: where an API has two lowering branches selected by a config flag, a
kwarg honoured on one and absorbed by a `**kwargs` on the other **fails without an error**. Zero
upstream call sites is the cheapest signal that a documented kwarg is not the supported path.

**Mechanism.** The kwarg is real on the legacy branch and forwarded through `**kwargs`;
`config.use_emit_pipeline_primitive` can route to a branch that drops unknown kwargs with no
diagnostic. Nothing fails — the scratch is simply never wired, so the kernel reads whatever was
already in that memory. Source to re-read: `jax/_src/pallas/mosaic/pipeline.py:emit_pipeline` and
`jax/_src/pallas/config.py:use_emit_pipeline_primitive`; use `tools/search_api.py` to find upstream
call sites and the closure-capture form in the SparseCore Pallas tests.

**Scope.** All Pallas TPU pipelines. Related pitfall worth checking at the same time: the 2-D
`Tiling.SPARSE_CORE` path may be unproven at the target lane count (the upstream test has been
gated on a specific `num_lanes`), so prefer the 1-D `BlockSpec` form unless probe
`sc_module_surface` and a run show otherwise.

**Does not apply when** you have pinned the legacy branch explicitly. Confirm which branch the flag
selects rather than assuming the default.

**How to check on your kernel.** Write a tiny `emit_pipeline` body that fills a scratch passed via
`scratches=` with a sentinel and copies it to the output. If the output does not hold the sentinel,
the kwarg is dropped on this toolchain: switch to closure capture.
