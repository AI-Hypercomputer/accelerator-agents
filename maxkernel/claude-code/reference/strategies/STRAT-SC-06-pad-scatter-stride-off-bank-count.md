---
id: STRAT-SC-06
title: A per-lane stride sharing a factor with the bank count collapses every lane onto one bank
applies_to: sparsecore
lever: layout
generality: general
phase: [plan, implement]
verify: "manual: time the lane-private table at a power-of-two stride and at that stride + 1, at fixed work"
---

# STRAT-SC-06 — Pad a per-lane stride coprime to the bank count

**Rule.** When each lane owns a private region addressed `lane * STRIDE + offset`, the lanes land on
distinct banks only if **`STRIDE` is coprime to `banks`**. Share a factor and every lane touching
the same offset lands on one bank, serializing the access by up to the number of lanes. **Pad
`STRIDE` by one word.** The cost is one word per lane; the return is the full lane parallelism.

**Mechanism.** Bank index is the address modulo `banks`. If `STRIDE ≡ 0 (mod banks)` then
`lane * STRIDE + offset ≡ offset` for every lane — one bank, `lanes`-way serialization, independent
of the offset. If `gcd(STRIDE, banks) = 1` the map `lane ↦ lane * STRIDE (mod banks)` is a
bijection, so the lanes are spread regardless of which offset they chose. Conflict cost grows
roughly linearly in the number of ways.

**Scope.** Any lane-private table — histograms, counters, accumulators — on any banked memory.
`banks` is the parameter. `SparseCoreInfo` declares subcores and lanes but **not** banks, so the
bank count is not available at run time and any value you hold is inferred, not specified (see
[tpu_hardware_facts.md](../tpu_hardware_facts.md)). Do not confuse it with the subcore count `W` of
[STRAT-SC-05](STRAT-SC-05-wave-longest-row.md). If `banks` is a power of two, any odd stride is
coprime to it, so "pad to odd" is safe without knowing its exact value.

**Does not apply when** lanes read their regions at *different* offsets by construction, which
already spreads them. It is the shared-offset case that collapses.

**How to check on your kernel.** Find every lane-private region and its stride. If the stride is a
multiple of a power of two (typical: a bucket count like 256), time the kernel at that stride and at
stride + 1 with the work held fixed. A conflict shows as a gain at stride + 1 and a collapse of the
cross-distribution spread (see
[STRAT-SC-07](STRAT-SC-07-distribution-sensitivity-is-a-conflict.md)). Re-run the correctness oracle
after the change.
