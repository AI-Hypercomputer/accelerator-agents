---
id: STRAT-SC-03
title: A chained-invocation clock must barrier the loop carry, or LICM hoists the kernel out of its own loop
applies_to: any
lever: method
generality: general
phase: [implement, profile, test]
verify: "manual: assert the HLO tpu_custom_call count equals the expected multiple of M, and fit T(M) = a + s*M across an M-sweep"
---

# STRAT-SC-03 — Barrier the loop carry, never the captured invariants

**Rule.** A microbenchmark that amortizes launch overhead by chaining `M` invocations must route its
data dependency **through the loop carry**. Barrier the invariants instead and the compiler is free
to prove the inputs unchanging, hoist the call out of the loop, and run `M` nominal invocations
**once**. Then assert the kernel count in the HLO equals the expected multiple of `M`, and derive
per-call cost from an **M-sweep slope** rather than dividing by `M` at a single `M`.

**Mechanism.** Loop-invariant code motion is a correctness-preserving transform, so **no correctness
guard can catch this**: a hoisted loop is idempotent and still returns the right answer. The
symptom is a physically impossible number — a per-call time below a hardware floor — and the tell in
the HLO is the async-start taking top-level operands instead of a `while`-carry
`%get-tuple-element`.

**Scope.** Any chained-invocation clock on any accelerator whose compiler does LICM. Gate the
"scales with M" check on the **kernels** (`tpu_custom_call`) only: loop-invariant non-kernel XLA ops
may hoist legally. But a kernel-only gate is blind to scalar prep outside the kernel hoisting out of
the chain, which inflates the intercept while the slope looks clean — cross-check the slope against
single-call and unchained clocks and report the range. The healthy ratio `T(M_hi)/T(1)` depends on
the host's launch overhead; derive it locally (probe `pallas_call_intercept_chained_clock` measures
the fixed per-call cost) — a ratio near one means the call was hoisted.

**Does not apply when** the harness times single unchained invocations — but then launch overhead is
in every number, and subtracting a fitted constant reintroduces a different way to be wrong.

**How to check on your kernel.** Dump the HLO of the chained benchmark, count `tpu_custom_call`
occurrences, and require the expected multiple of `M`. Run at least three values of `M`, fit
`T(M) = a + s·M`, and report `s` as per-call cost. Reject any per-call figure below the kernel's
roofline floor.
